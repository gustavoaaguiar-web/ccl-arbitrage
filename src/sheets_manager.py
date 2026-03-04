"""
Google Sheets Manager
=====================
Maneja la persistencia del simulador y del historial CCL en Google Sheets.

Hojas:
- CCL_Historial   → snapshots CCL para HMM
- Operaciones     → registro de cada trade cerrado
- Estado_Cartera  → snapshots del capital total
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

# Encabezados por hoja
HEADERS = {
    "CCL_Historial": ["timestamp", "ccl_avg", "GGAL", "YPFD", "PAMP", "CEPU",
                      "AMZN", "MSFT", "NVDA", "TSLA", "AAPL", "META", "GOOGL",
                      "MELI", "BMA", "VIST"],
    "Operaciones":   ["id", "symbol", "tipo", "cantidad", "precio_entry",
                      "precio_exit", "monto_entry", "monto_exit", "pnl",
                      "pnl_pct", "ts_entry", "ts_exit", "motivo_cierre"],
    "Estado_Cartera": ["timestamp", "capital_total", "efectivo", "en_posiciones",
                       "pnl_total", "pnl_pct", "operaciones_total", "win_rate",
                       "posiciones_abiertas"],
}


class SheetsManager:
    """Maneja todas las operaciones con Google Sheets."""

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
        """Crea las hojas si no existen y agrega encabezados."""
        hojas_existentes = [ws.title for ws in self.sh.worksheets()]

        for nombre, headers in HEADERS.items():
            if nombre not in hojas_existentes:
                ws = self.sh.add_worksheet(title=nombre, rows=10000, cols=len(headers))
                ws.append_row(headers)
                logger.info(f"📋 Hoja creada: {nombre}")
            else:
                ws = self.sh.worksheet(nombre)
            self._hojas[nombre] = ws

        # Renombrar Sheet1 si existe vacía
        try:
            sheet1 = self.sh.worksheet("Sheet1")
            if len(sheet1.get_all_values()) <= 1:
                self.sh.del_worksheet(sheet1)
        except:
            pass

    # ─────────────── CCL HISTORIAL ───────────────────────

    def guardar_snapshot_ccl(self, ccl_map: dict, ccl_avg: float, ts: str = None):
        """Guarda un snapshot de CCL para el HMM."""
        from datetime import datetime
        ws = self._hojas.get("CCL_Historial")
        if not ws:
            return
        headers = HEADERS["CCL_Historial"]
        simbolos = headers[2:]
        fila = [ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S"), round(ccl_avg, 2)]
        fila += [round(ccl_map.get(sym, 0), 2) for sym in simbolos]
        ws.append_row(fila)

    def cargar_historial_ccl(self) -> list:
        """Carga el historial CCL para alimentar el HMM."""
        ws = self._hojas.get("CCL_Historial")
        if not ws:
            return []
        filas = ws.get_all_values()
        if len(filas) < 2:
            return []

        headers  = filas[0]
        simbolos = headers[2:]

        historial = []
        for fila in filas[1:]:
            try:
                ts  = fila[0]
                avg = float(str(fila[1]).replace(",", "."))
                ccl_dic = {}
                for i, sym in enumerate(simbolos):
                    idx = i + 2
                    if idx < len(fila) and fila[idx]:
                        val = float(str(fila[idx]).replace(",", "."))
                        if val > 0:
                            ccl_dic[sym] = val
                if ccl_dic:
                    historial.append({"ts": ts, "ccl": ccl_dic, "avg": avg})
            except:
                continue
        return historial

    # ─────────────── OPERACIONES ─────────────────────────

    def guardar_operacion(self, fila: list):
        """Guarda una operación cerrada."""
        ws = self._hojas.get("Operaciones")
        if ws:
            ws.append_row(fila)

    def guardar_operaciones_batch(self, filas: List[list]):
        """Guarda múltiples operaciones de una vez."""
        ws = self._hojas.get("Operaciones")
        if ws and filas:
            for fila in filas:
                ws.append_row(fila)

    def cargar_operaciones(self) -> list:
        """Carga el historial de operaciones."""
        ws = self._hojas.get("Operaciones")
        if not ws:
            return []
        filas = ws.get_all_values()
        return filas[1:] if len(filas) > 1 else []

    # ─────────────── ESTADO CARTERA ──────────────────────

    def guardar_estado_cartera(self, fila: list):
        """Guarda snapshot del estado de la cartera."""
        ws = self._hojas.get("Estado_Cartera")
        if ws:
            ws.append_row(fila)

    def cargar_ultimo_estado(self) -> dict:
        """Carga el último estado guardado de la cartera."""
        ws = self._hojas.get("Estado_Cartera")
        if not ws:
            return {}
        filas = ws.get_all_values()
        if len(filas) < 2:
            return {}
        ultima = filas[-1]
        try:
            return {
                "ts":              ultima[0],
                "capital_total":   float(ultima[1]),
                "efectivo":        float(ultima[2]),
                "en_posiciones":   float(ultima[3]),
                "pnl_total":       float(ultima[4]),
                "pnl_pct":         float(ultima[5]),
                "operaciones_total": int(ultima[6]),
                "win_rate":        float(ultima[7]),
            }
        except:
            return {}
          
