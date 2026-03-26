import logging
from typing import List
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

logger = logging.getLogger(__name__)

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

SHEET_NAME = "CCL-Arbitrage-Historial"
HMM_MAX_SNAPSHOTS = 500

HEADERS = {
    "CCL_Historial": ["timestamp", "ccl_avg", "GGAL", "YPFD", "PAMP", "CEPU",
                      "AMZN", "MSFT", "NVDA", "TSLA", "AAPL", "META", "GOOGL",
                      "MELI", "BMA", "VIST"],
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
        ts = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws_ccl = self._hojas.get("CCL_Historial")
        if ws_ccl:
            simbolos = HEADERS["CCL_Historial"][2:]
            fila = [ts, round(ccl_avg, 2)]
            fila += [round(ccl_map.get(sym, 0), 2) for sym in simbolos]
            ws_ccl.append_row(fila)

        if p_usd:
            ws_hmm = self._hojas.get("HMM_Historial")
            if not ws_hmm: return
            filas_nuevas = [[ts, sym, round(precio, 6)] for sym, precio in p_usd.items() if precio]
            if filas_nuevas:
                ws_hmm.append_rows(filas_nuevas)
            
            todas = ws_hmm.get_all_values()
            n_syms = len(p_usd)
            max_filas_datos = HMM_MAX_SNAPSHOTS * n_syms
            filas_datos = len(todas) - 1
            if filas_datos > max_filas_datos:
                exceso = filas_datos - max_filas_datos
                ws_hmm.delete_rows(2, exceso + 1)
                logger.info(f"🗑️ HMM_Historial: eliminadas {exceso} filas antiguas")

    def cargar_historial_ccl(self) -> list:
        usd_por_ts: dict = {}
        ws_hmm = self._hojas.get("HMM_Historial")
        if ws_hmm:
            filas_hmm = ws_hmm.get_all_values()
            for fila in filas_hmm[1:]:
                try:
                    ts_h, sym, precio = fila[0], fila[1], _f(fila[2])
                    if precio > 0:
                        usd_por_ts.setdefault(ts_h, {})[sym] = precio
                except: continue

        ws_ccl = self._hojas.get("CCL_Historial")
        if not ws_ccl:
            return [{"ts": ts, "ccl": {}, "avg": 0, "usd": precios} for ts, precios in sorted(usd_por_ts.items())]

        filas_ccl = ws_ccl.get_all_values()
        if len(filas_ccl) < 2:
            return [{"ts": ts, "ccl": {}, "avg": 0, "usd": precios} for ts, precios in sorted(usd_por_ts.items())]

        headers, simbolos = filas_ccl[0], filas_ccl[0][2:]
        historial = []
        for fila in filas_ccl[1:]:
            try:
                ts, avg = fila[0], _f(fila[1])
                ccl_dic = {sym: _f(fila[i+2]) for i, sym in enumerate(simbolos) if i+2 < len(fila) and fila[i+2]}
                if ccl_dic:
                    historial.append({"ts": ts, "ccl": ccl_dic, "avg": avg, "usd": usd_por_ts.get(ts, {})})
            except: continue
        return historial

    # ─────────────── OPERACIONES (CON FILTRO ANTI-DUPLICADOS) ──────────

    def guardar_operacion(self, fila: list):
        """Verifica si el ID de la operación ya existe antes de escribir."""
        ws = self._hojas.get("Operaciones")
        if ws:
            try:
                nuevo_id = str(fila[0])
                # Obtenemos solo la columna de IDs para comparar
                ids_existentes = ws.col_values(1)
                
                if nuevo_id not in ids_existentes:
                    ws.append_row(fila)
                    logger.info(f"✅ Operación {nuevo_id} registrada con éxito.")
                else:
                    logger.warning(f"⚠️ El ID {nuevo_id} ya existe en Sheets. Omitiendo duplicado.")
            except Exception as e:
                logger.error(f"❌ Error al guardar operación: {e}")

    def cargar_operaciones(self) -> list:
        ws = self._hojas.get("Operaciones")
        if not ws: return []
        filas = ws.get_all_values()
        return filas[1:] if len(filas) > 1 else []

    # ─────────────── ESTADO CARTERA ──────────────────────

    def guardar_estado_cartera(self, fila: list):
        ws = self._hojas.get("Estado_Cartera")
        if ws: ws.append_row(fila)

    # ─────────────── POSICIONES ABIERTAS ─────────────────

    def guardar_posiciones(self, simulador):
        ws = self._hojas.get("Posiciones_Abiertas")
        if not ws: return
        ws.clear()
        ws.append_row(HEADERS["Posiciones_Abiertas"])
        for sym, poss in simulador.posiciones.items():
            for pos in poss:
                ws.append_row([pos.id, sym, round(pos.cantidad, 6), round(pos.precio_entry, 4),
                               round(pos.monto_entry, 2), pos.ts_entry, round(pos.ccl_entry, 4), 
                               round(pos.dev_entry, 4)])

    def cargar_posiciones(self, simulador):
        from simulator import Posicion
        ws = self._hojas.get("Posiciones_Abiertas")
        if not ws: return
        filas = ws.get_all_values()
        if len(filas) < 2: return
        simulador.posiciones = {}
        for fila in filas[1:]:
            try:
                pos = Posicion(id=fila[0], symbol=fila[1], cantidad=_f(fila[2]), precio_entry=_f(fila[3]),
                               monto_entry=_f(fila[4]), ts_entry=fila[5], ccl_entry=_f(fila[6]),
                               dev_entry=_f(fila[7]), precio_actual=_f(fila[3]))
                simulador.posiciones.setdefault(fila[1], []).append(pos)
            except Exception as e: logger.warning(f"Posición saltada: {e}")
        logger.info(f"✅ Posiciones cargadas: {sum(len(v) for v in simulador.posiciones.values())}")

    # ─────────────── SIMULADOR ESTADO ────────────────────

    def guardar_estado_simulador(self, simulador):
        ws = self._hojas.get("Simulador_Estado")
        if not ws: return
        ws.clear()
        ws.append_row(HEADERS["Simulador_Estado"])
        ws.append_row([round(simulador.efectivo, 2), simulador._op_counter])

    def cargar_estado_simulador(self, simulador):
        ws = self._hojas.get("Simulador_Estado")
        if not ws: return
        filas = ws.get_all_values()
        if len(filas) < 2: return
        try:
            simulador.efectivo = _f(filas[1][0])
            simulador._op_counter = int(_f(filas[1][1]))
            logger.info(f"✅ Estado simulador cargado: efectivo=${simulador.efectivo:,.0f} ops={simulador._op_counter}")
        except Exception as e: logger.warning(f"Error cargando estado simulador: {e}")
