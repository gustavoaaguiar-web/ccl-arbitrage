"""
Google Sheets Manager — Sistema GG Swing
==========================================
Hojas:
- Historico_Merval_Raw → snapshots crudos de Merval (acumulación continua,
                          base para resamplear a velas 30min/4H)
- Operaciones          → registro de cada cierre (parcial o final) de trade
- Estado_Cartera       → snapshots del capital total
- Posiciones_Abiertas  → posiciones abiertas (persiste entre reinicios)
- Simulador_Estado     → efectivo y contador (persiste entre reinicios)
- Backtest_Resultados  → trades individuales del backtest (Ruta A)
- Backtest_Metricas    → métricas agregadas por símbolo del backtest

NOTA DE TRANSICIÓN (jun-2026):
Las hojas CCL_Historial y HMM_Historial del sistema de arbitraje anterior
fueron eliminadas de este archivo. El sistema pivotó a swing trading
técnico (Sistema GG Swing) tras la compresión estructural de spreads CCL.

NOTA DE TRANSICIÓN #2 (jun-2026):
Posiciones_Abiertas se actualizó: los campos ccl_entry/dev_entry del
arbitraje CCL fueron reemplazados por score/precio_stop/precio_t1/
precio_t2/precio_t3/riesgo_pct + el estado de salida escalonada
(t1_alcanzado, t2_alcanzado, stop_en_breakeven, cantidad_restante,
cantidad_inicial), acorde a la nueva clase Posicion de simulator.py.
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
    "Posiciones_Abiertas": ["id", "symbol", "score", "cantidad_inicial",
                             "cantidad_restante", "precio_entry", "monto_entry",
                             "precio_stop", "precio_t1", "precio_t2", "precio_t3",
                             "riesgo_pct", "ts_entry", "t1_alcanzado", "t2_alcanzado",
                             "stop_en_breakeven", "cantidad_cerrada_t1",
                             "cantidad_cerrada_t2"],
    "Simulador_Estado": ["efectivo", "op_counter"],
    "Backtest_Resultados": ["Symbol", "Regimen", "Fecha Entry", "Fecha Salida",
                             "Entry", "Stop", "T1", "T2", "T3",
                             "Precio Salida Final", "Días", "Motivo Salida",
                             "R Realizado", "Score", "ATR%",
                             "T1 cerrada", "T2 cerrada", "T3 cerrada",
                             "Precio T1", "Precio T2", "Precio T3"],
    "Backtest_Metricas":   ["Symbol", "Trades", "Win Rate %", "R Promedio",
                             "R Total", "Profit Factor", "Max DD (R)",
                             "Mejor Trade R", "Peor Trade R",
                             "Días Prom en Trade"],
}


def _f(s):
    """Convierte string a float tolerando coma decimal."""
    try:
        return float(str(s).replace(",", ".").strip())
    except:
        return 0.0


def _b(s):
    """Convierte string de Sheets ('TRUE'/'FALSE'/'1'/'0') a bool."""
    return str(s).strip().upper() in ("TRUE", "1", "VERDADERO")


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
                fila1_actual = ws.row_values(1)
                if fila1_actual != headers:
                    ws.update('A1', [headers])
                    logger.warning(
                        f"⚠️ Headers de '{nombre}' actualizados a la versión Swing. "
                        f"Si había filas de datos del sistema CCL anterior, revisar "
                        f"manualmente — pueden no corresponder a las columnas nuevas."
                    )
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
        solo ese símbolo. Retorna lista ordenada por ts ascendente.
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
        filas = []
        for sym, pos in simulador.posiciones.items():
            filas.append([
                pos.id, sym,
                round(pos.score, 2),
                round(pos.cantidad_inicial, 6),
                round(pos.cantidad_restante, 6),
                round(pos.precio_entry, 4),
                round(pos.monto_entry, 2),
                round(pos.precio_stop, 4),
                round(pos.precio_t1, 4),
                round(pos.precio_t2, 4),
                round(pos.precio_t3, 4),
                round(pos.riesgo_pct, 6),
                pos.ts_entry,
                pos.t1_alcanzado,
                pos.t2_alcanzado,
                pos.stop_en_breakeven,
                round(pos.cantidad_cerrada_t1, 6),
                round(pos.cantidad_cerrada_t2, 6),
            ])
        if filas:
            ws.append_rows(filas)

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
        ids_vistos = set()
        for fila in filas[1:]:
            try:
                if len(fila) < 18:
                    logger.warning(f"Fila de Posiciones_Abiertas con columnas insuficientes, saltada: {fila}")
                    continue
                pos_id = fila[0]
                if pos_id in ids_vistos:
                    logger.warning(f"Posición duplicada ignorada al cargar: {pos_id}")
                    continue
                ids_vistos.add(pos_id)
                symbol = fila[1]
                pos = Posicion(
                    id=pos_id,
                    symbol=symbol,
                    score=_f(fila[2]),
                    cantidad_inicial=_f(fila[3]),
                    cantidad_restante=_f(fila[4]),
                    precio_entry=_f(fila[5]),
                    monto_entry=_f(fila[6]),
                    precio_stop=_f(fila[7]),
                    precio_t1=_f(fila[8]),
                    precio_t2=_f(fila[9]),
                    precio_t3=_f(fila[10]),
                    riesgo_pct=_f(fila[11]),
                    ts_entry=fila[12],
                    precio_actual=_f(fila[5]),
                    t1_alcanzado=_b(fila[13]),
                    t2_alcanzado=_b(fila[14]),
                    stop_en_breakeven=_b(fila[15]),
                    cantidad_cerrada_t1=_f(fila[16]),
                    cantidad_cerrada_t2=_f(fila[17]),
                )
                simulador.posiciones[symbol] = pos
            except Exception as e:
                logger.warning(f"Posición saltada: {e}")
        logger.info(f"✅ Posiciones cargadas: {len(simulador.posiciones)}")

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

    # ─────────────── BACKTEST ────────────────────────────

    def limpiar_y_escribir(self, nombre_hoja: str, filas: List[list]):
        """
        Reemplaza el contenido completo de una hoja con las filas dadas.
        La primera fila de `filas` debe ser el encabezado.
        Usado por run_backtest.py para subir resultados frescos en cada
        ejecución sin acumular runs anteriores.

        Si la hoja no existe en _hojas (por ejemplo Backtest_Resultados
        creada en _inicializar_hojas), lanza KeyError con mensaje claro.
        """
        ws = self._hojas.get(nombre_hoja)
        if not ws:
            raise KeyError(
                f"Hoja '{nombre_hoja}' no encontrada. "
                f"Hojas disponibles: {list(self._hojas.keys())}"
            )
        if not filas:
            logger.warning(f"limpiar_y_escribir('{nombre_hoja}'): sin filas, no se escribe nada")
            return

        ws.clear()
        # batch_update en un solo request para no agotar cuota de Sheets API
        ws.update(
            range_name="A1",
            values=filas,
        )
        logger.info(f"✅ '{nombre_hoja}' actualizada: {len(filas) - 1} filas de datos")
