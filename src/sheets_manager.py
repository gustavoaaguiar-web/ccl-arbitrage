"""
Google Sheets Manager
=====================
Hojas:
- CCL_Historial       → snapshots CCL para HMM
- HMM_Historial       → precios USD por snapshot (rolling 500) para HMM Simons
- Operaciones         → registro de cada trade cerrado
- Estado_Cartera      → snapshots del capital total
- Posiciones_Abiertas → posiciones abiertas (persiste entre reinicios)
- Simulador_Estado    → efectivo y contador (persiste entre reinicios)
"""

import logging
from typing import List
import gspread
from google.oauth2.service_account import Credentials

logger = logging.getLogger(__name__)

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

SHEET_NAME = "CCL-Arbitrage-Historial"

# Máximo de snapshots USD a retener en HMM_Historial (rolling window).
# A 60s/ciclo → 500 snapshots ≈ 1.5 días de trading.
HMM_MAX_SNAPSHOTS = 500

HEADERS = {
    "CCL_Historial": ["timestamp", "ccl_avg", "GGAL", "YPFD", "PAMP", "CEPU",
                      "AMZN", "MSFT", "NVDA", "TSLA", "AAPL", "META", "GOOGL",
                      "MELI", "BMA", "GLD", "IBIT", "SPY", TGSU2"],
    "HMM_Historial": ["ts", "sym_usd", "precio"],
    "Operaciones":   ["id", "symbol", "tipo", "cantidad", "precio_entry",
                      "precio_exit", "monto_entry", "monto_exit", "pnl",
                      "pnl_pct", "ts_entry", "ts_exit", "motivo_cierre"],
    "Estado_Cartera": ["timestamp", "capital_total", "efectivo", "en_posiciones",
                       "pnl_total", "pnl_pct", "operaciones_total", "win_rate",
                       "posiciones_abiertas"],
    "Posiciones_Abiertas": ["id", "symbol", "cantidad", "precio_entry",
                             "monto_entry", "ts_entry", "ccl_entry", "dev_entry"],
    "Simulador_Estado": ["efectivo", "op_counter"],
}


def _f(s):
    """Convierte string a float tolerando coma decimal."""
    try:
        return float(str(s).replace(",", ".").strip())
    except:
        return 0.0


class SheetsManager:

    def __init__(self, service_account_info: dict):
        creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
        self.gc = gspread.authorize(creds)
        self.sh = None
        self._hojas = {}

    def conectar(self) -> bool:
        try:
            self.sh = self.gc.open(SHEET_NAME)
            self._inicializar_hojas()
            logger.info(f"✅ Google Sheets conectado: {SHEET_NAME}")
            return True
        except Exception as e:
            logger.error(f"❌ Error conectando Sheets: {e}")
            return False

    def _inicializar_hojas(self):
        hojas_existentes = [ws.title for ws in self.sh.worksheets()]
        for nombre, headers in HEADERS.items():
            if nombre not in hojas_existentes:
                ws = self.sh.add_worksheet(title=nombre, rows=10000, cols=len(headers))
                ws.append_row(headers)
                logger.info(f"📋 Hoja creada: {nombre}")
            else:
                ws = self.sh.worksheet(nombre)
            self._hojas[nombre] = ws
        try:
            sheet1 = self.sh.worksheet("Sheet1")
            if len(sheet1.get_all_values()) <= 1:
                self.sh.del_worksheet(sheet1)
        except:
            pass

    # ─────────────── CCL HISTORIAL ───────────────────────

    def guardar_snapshot_ccl(self, ccl_map: dict, ccl_avg: float,
                              p_usd: dict = None, ts: str = None):
        """
        Guarda snapshot CCL en CCL_Historial.
        Si se pasa p_usd, también persiste los precios USD en HMM_Historial
        para que el HMM Simons sobreviva reinicios de la app.
        Mantiene un rolling window de HMM_MAX_SNAPSHOTS en HMM_Historial.
        """
        from datetime import datetime
        ts = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # — CCL_Historial (sin cambios) —
        ws_ccl = self._hojas.get("CCL_Historial")
        if ws_ccl:
            simbolos = HEADERS["CCL_Historial"][2:]
            fila = [ts, round(ccl_avg, 2)]
            fila += [round(ccl_map.get(sym, 0), 2) for sym in simbolos]
            ws_ccl.append_row(fila)

        # — HMM_Historial (nuevo) —
        if p_usd:
            ws_hmm = self._hojas.get("HMM_Historial")
            if not ws_hmm:
                return

            # Escribir una fila por símbolo USD
            filas_nuevas = [[ts, sym, round(precio, 6)]
                            for sym, precio in p_usd.items() if precio]
            if filas_nuevas:
                ws_hmm.append_rows(filas_nuevas)

            # Rolling window: eliminar filas antiguas si se supera el límite.
            # Contamos snapshots únicos por timestamp para calcular el corte.
            todas = ws_hmm.get_all_values()  # incluye header
            n_syms = len(p_usd)
            max_filas_datos = HMM_MAX_SNAPSHOTS * n_syms
            filas_datos = len(todas) - 1  # sin header

            if filas_datos > max_filas_datos:
                exceso = filas_datos - max_filas_datos
                # delete_rows(start, end) — filas 2..exceso+1 (1-indexed, saltando header)
                ws_hmm.delete_rows(2, exceso + 1)
                logger.info(f"🗑️ HMM_Historial: eliminadas {exceso} filas antiguas")

    def cargar_historial_ccl(self) -> list:
        """
        Carga el historial para el HMM Simons.
        Combina CCL_Historial (ccl/avg) con HMM_Historial (p_usd).
        Retorna lista de dicts con claves: ts, ccl, avg, usd.
        """
        # — Cargar precios USD desde HMM_Historial —
        usd_por_ts: dict = {}
        ws_hmm = self._hojas.get("HMM_Historial")
        if ws_hmm:
            filas_hmm = ws_hmm.get_all_values()
            for fila in filas_hmm[1:]:  # skip header
                try:
                    ts_h, sym, precio = fila[0], fila[1], _f(fila[2])
                    if precio > 0:
                        usd_por_ts.setdefault(ts_h, {})[sym] = precio
                except:
                    continue

        # — Cargar CCL desde CCL_Historial —
        ws_ccl = self._hojas.get("CCL_Historial")
        if not ws_ccl:
            # Si no hay CCL pero sí hay USD, devolver solo con usd
            return [{"ts": ts, "ccl": {}, "avg": 0, "usd": precios}
                    for ts, precios in sorted(usd_por_ts.items())]

        filas_ccl = ws_ccl.get_all_values()
        if len(filas_ccl) < 2:
            return [{"ts": ts, "ccl": {}, "avg": 0, "usd": precios}
                    for ts, precios in sorted(usd_por_ts.items())]

        headers  = filas_ccl[0]
        simbolos = headers[2:]
        historial = []

        for fila in filas_ccl[1:]:
            try:
                ts  = fila[0]
                avg = _f(fila[1])
                ccl_dic = {}
                for i, sym in enumerate(simbolos):
                    idx = i + 2
                    if idx < len(fila) and fila[idx]:
                        val = _f(fila[idx])
                        if val > 0:
                            ccl_dic[sym] = val
                if ccl_dic:
                    historial.append({
                        "ts":  ts,
                        "ccl": ccl_dic,
                        "avg": avg,
                        "usd": usd_por_ts.get(ts, {}),  # adjuntar USD si existe
                    })
            except:
                continue

        return historial

    # ─────────────── OPERACIONES ─────────────────────────

    def guardar_operacion(self, fila: list):
        ws = self._hojas.get("Operaciones")
        if ws:
            ws.append_row(fila)

    def cargar_operaciones(self) -> list:
        ws = self._hojas.get("Operaciones")
        if not ws:
            return []
        filas = ws.get_all_values()
        return filas[1:] if len(filas) > 1 else []

    # ─────────────── ESTADO CARTERA ──────────────────────

    def guardar_estado_cartera(self, fila: list):
        ws = self._hojas.get("Estado_Cartera")
        if ws:
            ws.append_row(fila)

    # ─────────────── POSICIONES ABIERTAS ─────────────────

    def guardar_posiciones(self, simulador):
        """Sobreescribe la hoja con las posiciones abiertas actuales."""
        ws = self._hojas.get("Posiciones_Abiertas")
        if not ws:
            return
        ws.clear()
        ws.append_row(HEADERS["Posiciones_Abiertas"])
        for sym, poss in simulador.posiciones.items():
            for pos in poss:
                ws.append_row([
                    pos.id, sym,
                    round(pos.cantidad, 6),
                    round(pos.precio_entry, 4),
                    round(pos.monto_entry, 2),
                    pos.ts_entry,
                    round(pos.ccl_entry, 4),
                    round(pos.dev_entry, 4),
                ])

    def cargar_posiciones(self, simulador):
        """Carga posiciones abiertas desde Sheets al simulador."""
        from simulator import Posicion
        ws = self._hojas.get("Posiciones_Abiertas")
        if not ws:
            return
        filas = ws.get_all_values()
        if len(filas) < 2:
            return
        simulador.posiciones = {}
        for fila in filas[1:]:
            try:
                pos = Posicion(
                    id=fila[0],
                    symbol=fila[1],
                    cantidad=_f(fila[2]),
                    precio_entry=_f(fila[3]),
                    monto_entry=_f(fila[4]),
                    ts_entry=fila[5],
                    ccl_entry=_f(fila[6]),
                    dev_entry=_f(fila[7]),
                    precio_actual=_f(fila[3]),
                )
                sym = fila[1]
                if sym not in simulador.posiciones:
                    simulador.posiciones[sym] = []
                simulador.posiciones[sym].append(pos)
            except Exception as e:
                logger.warning(f"Posición saltada: {e}")
        logger.info(f"✅ Posiciones cargadas: {sum(len(v) for v in simulador.posiciones.values())}")

    # ─────────────── SIMULADOR ESTADO ────────────────────

    def guardar_estado_simulador(self, simulador):
        """Guarda efectivo y contador de operaciones."""
        ws = self._hojas.get("Simulador_Estado")
        if not ws:
            return
        ws.clear()
        ws.append_row(HEADERS["Simulador_Estado"])
        ws.append_row([round(simulador.efectivo, 2), simulador._op_counter])

    def cargar_estado_simulador(self, simulador):
        """Restaura efectivo y contador."""
        ws = self._hojas.get("Simulador_Estado")
        if not ws:
            return
        filas = ws.get_all_values()
        if len(filas) < 2:
            return
        try:
            simulador.efectivo    = _f(filas[1][0])
            simulador._op_counter = int(_f(filas[1][1]))
            logger.info(f"✅ Estado simulador cargado: efectivo=${simulador.efectivo:,.0f} ops={simulador._op_counter}")
        except Exception as e:
            logger.warning(f"Error cargando estado simulador: {e}")
