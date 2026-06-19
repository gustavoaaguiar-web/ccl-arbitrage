"""
Google Sheets Manager — Sistema GG Swing
==========================================
Hojas:
- Historico_Merval_Raw → snapshots crudos de Merval (acumulación continua,
                          base para resamplear a velas 30min/4H)
- Operaciones          → registro de cada trade cerrado
- Estado_Cartera       → snapshots del capital total
- Posiciones_Abiertas  → posiciones abiertas (persiste entre reinicios)
- Simulador_Estado     → efectivo y contador (persiste entre reinicios)

NOTA DE TRANSICIÓN (jun-2026):
Las hojas CCL_Historial y HMM_Historial del sistema de arbitraje anterior
fueron eliminadas de este archivo. El sistema pivotó a swing trading
técnico (Sistema GG Swing) tras la compresión estructural de spreads CCL.
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

SHEET_NAME = "CCL-Arbitrage-Historial"  # nombre del spreadsheet en Drive — sin cambios

HEADERS = {
    "Historico_Merval_Raw": ["ts", "symbol", "precio", "apertura",
                              "maximo", "minimo", "volumen_nominal",
                              "cantidad_operaciones"],
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

    # ─────────────── HISTÓRICO MERVAL (Sistema GG Swing) ─────

    def guardar_tick_merval(self, snapshots: list):
        """
        Guarda snapshots crudos de Merval para construir velas 30min/4H
        más adelante. Cada snapshot es un dict con:
        {symbol, precio, apertura, maximo, minimo, volumen_nominal,
         cantidad_operaciones}

        Se escribe 1 fila por símbolo por ciclo (cada ~60s dentro de
        cada ejecución GHA). No hace rolling window — esta hoja se
        acumula indefinidamente para servir de base histórica del
        Sistema GG Swing.
        """
        from datetime import datetime
        ws = self._hojas.get("Historico_Merval_Raw")
        if not ws or not snapshots:
            return

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filas = []
        for s in snapshots:
            filas.append([
                ts,
                s.get("symbol", ""),
                round(s.get("precio", 0), 2),
                round(s.get("apertura", 0), 2),
                round(s.get("maximo", 0), 2),
                round(s.get("minimo", 0), 2),
                round(s.get("volumen_nominal", 0), 2),
                s.get("cantidad_operaciones", 0),
            ])
        if filas:
            ws.append_rows(filas)

    def cargar_historico_merval_raw(self, symbol: str = None) -> List[dict]:
        """
        Carga el histórico crudo de Merval. Si se pasa symbol, filtra
        solo ese símbolo. Retorna lista ordenada por ts ascendente,
        lista para resamplear a velas 30min/4H.
        """
        ws = self._hojas.get("Historico_Merval_Raw")
        if not ws:
            return []

        filas = ws.get_all_values()
        if len(filas) < 2:
            return []

        resultado = []
        for fila in filas[1:]:
            try:
                if len(fila) < 8:
                    continue
                if symbol and fila[1] != symbol:
                    continue
                resultado.append({
                    "ts":                    fila[0],
                    "symbol":                fila[1],
                    "precio":                _f(fila[2]),
                    "apertura":              _f(fila[3]),
                    "maximo":                _f(fila[4]),
                    "minimo":                _f(fila[5]),
                    "volumen_nominal":       _f(fila[6]),
                    "cantidad_operaciones":  int(_f(fila[7])) if fila[7] else 0,
                })
            except (ValueError, TypeError, IndexError):
                continue

        resultado.sort(key=lambda x: x["ts"])
        return resultado

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
        ids_vistos = set()  # guard anti-duplicados por solapamiento de runs GHA
        for fila in filas[1:]:
            try:
                pos_id = fila[0]
                if pos_id in ids_vistos:
                    logger.warning(f"Posición duplicada ignorada al cargar: {pos_id}")
                    continue
                ids_vistos.add(pos_id)
                pos = Posicion(
                    id=pos_id,
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
