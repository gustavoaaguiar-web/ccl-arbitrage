"""
Backtest Engine — Sistema GG Swing (Ruta A — Timeframe Diario)
===============================================================
Valida la lógica HMA-D + SMI + scoring sobre datos históricos diarios
antes de avanzar a paper trading en 30min/4H.

DISEÑO:
    - Walk-forward estricto: en cada barra i, solo usa datos hasta i
      (sin lookahead). La señal se detecta al cierre de la barra i,
      el entry se ejecuta al cierre de i (precio de cierre = entry).
    - En timeframe diario, 1D = 4H = 30min (todo colapsa a diario).
      Se pasan las mismas velas a los 3 timeframes de la cascada, con
      params del régimen MEDIA como base. Esto es una simplificación
      explícita para Ruta A — se refina en Ruta B con datos reales.
    - Gestión de trade: stop/T1/T2/T3 calculados al entry. Se evalúa
      cada barra siguiente con high/low para ver si se toca algún nivel.
    - Salida máxima: 21 días hábiles desde entry si no se tocó nada.
    - Salida escalonada:
        T1 → cierra 40% de la posición, stop sube a breakeven
        T2 → cierra 40% adicional, queda 20% con trailing HMA16
        T3 / trailing → cierra el 20% restante
        Stop / timeout → cierra 100%

OUTPUTS:
    - Lista de TradeResult con todos los campos necesarios para análisis
    - Métricas agregadas: win rate, R promedio, profit factor, max
      drawdown de equity (en R), trades totales / ganadores / perdedores
    - Exportable a Google Sheets vía to_sheets_rows()

USO TÍPICO:
    from backtest_engine import BacktestEngine, cargar_velas_desde_historico
    from signal_engine import VelasOHLCV

    velas = cargar_velas_desde_historico(rows_desde_sheets)
    engine = BacktestEngine(velas, symbol="GGAL")
    resultado = engine.correr()
    print(resultado.metricas)
    # Subir a Sheets:
    sheets.guardar_backtest(resultado.to_sheets_rows())
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# signal_engine debe estar en el path (src/ o mismo dir)
from signal_engine import (
    VelasOHLCV,
    PARAMS_REGIMEN,
    HMA16_ENTRY_PERIODO,
    HMA50_REGIMEN_PERIODO,
    hma,
    atr,
    smi,
    clasificar_regimen,
    calcular_niveles_salida,
    SCORE_MINIMO,
    PESO_ALINEACION_TF,
    PESO_ZONA_ENTRADA,
    PESO_CATALIZADOR,
    PESO_REGIMEN_TENDENCIA,
    PESO_GATE_FUNDAMENTAL,
)

logger = logging.getLogger(__name__)

# ── Constantes de gestión ──────────────────────────────────
MAX_HOLD_DIAS = 21        # barras diarias máximas de hold
FRACCION_T1   = 0.40      # % del trade cerrado en T1
FRACCION_T2   = 0.40      # % del trade cerrado en T2 (acumulado: 80%)
FRACCION_T3   = 0.20      # % restante cerrado en T3/trailing/timeout

# En Ruta A (diario) no hay datos 4H/30min separados — usamos régimen
# MEDIA como base por ser el más representativo del universo objetivo.
REGIMEN_BACKTEST_DEFAULT = "MEDIA"


# ════════════════════════════════════════════════════════════════
# ESTRUCTURAS
# ════════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    """Resultado de un trade individual simulado."""
    symbol: str
    regimen: str
    fecha_entry: str
    fecha_salida: str
    precio_entry: float
    precio_stop: float
    precio_t1: float
    precio_t2: float
    precio_t3: float
    precio_salida_final: float
    dias_en_trade: int
    motivo_salida: str        # "T1+T2+T3", "T1+T2+trailing", "stop", "timeout", etc.
    r_realizado: float        # R neto del trade completo (promedio ponderado de fracciones)
    score: float
    atr_pct: float
    # Desglose de fracciones
    fraccion_t1_cerrada: bool = False
    fraccion_t2_cerrada: bool = False
    fraccion_t3_cerrada: bool = False
    precio_salida_t1: Optional[float] = None
    precio_salida_t2: Optional[float] = None
    precio_salida_t3: Optional[float] = None


@dataclass
class ResultadoBacktest:
    """Resultado agregado de un backtest completo para un símbolo."""
    symbol: str
    trades: List[TradeResult] = field(default_factory=list)
    metricas: Dict = field(default_factory=dict)

    def calcular_metricas(self):
        t = self.trades
        if not t:
            self.metricas = {"trades_totales": 0}
            return

        total = len(t)
        ganadores = [x for x in t if x.r_realizado > 0]
        perdedores = [x for x in t if x.r_realizado <= 0]

        r_vals = [x.r_realizado for x in t]
        win_rate = len(ganadores) / total if total else 0

        r_ganadores = sum(x.r_realizado for x in ganadores)
        r_perdedores = abs(sum(x.r_realizado for x in perdedores))
        profit_factor = r_ganadores / r_perdedores if r_perdedores > 0 else float("inf")

        # Max drawdown de equity en R (curva acumulada)
        equity = np.cumsum(r_vals)
        max_dd = 0.0
        pico = equity[0]
        for e in equity:
            if e > pico:
                pico = e
            dd = pico - e
            if dd > max_dd:
                max_dd = dd

        self.metricas = {
            "trades_totales": total,
            "ganadores": len(ganadores),
            "perdedores": len(perdedores),
            "win_rate": round(win_rate * 100, 1),
            "r_promedio": round(np.mean(r_vals), 2),
            "r_mediano": round(float(np.median(r_vals)), 2),
            "r_total": round(float(np.sum(r_vals)), 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_r": round(max_dd, 2),
            "r_mejor_trade": round(max(r_vals), 2),
            "r_peor_trade": round(min(r_vals), 2),
            "dias_promedio_en_trade": round(np.mean([x.dias_en_trade for x in t]), 1),
        }

    def to_sheets_rows(self) -> List[List]:
        """
        Convierte los trades a filas para Google Sheets.
        Primera fila = encabezados.
        """
        headers = [
            "Symbol", "Regimen", "Fecha Entry", "Fecha Salida",
            "Entry", "Stop", "T1", "T2", "T3",
            "Precio Salida Final", "Días", "Motivo Salida",
            "R Realizado", "Score", "ATR%",
            "T1 cerrada", "T2 cerrada", "T3 cerrada",
            "Precio T1", "Precio T2", "Precio T3",
        ]
        rows = [headers]
        for t in self.trades:
            rows.append([
                t.symbol, t.regimen, t.fecha_entry, t.fecha_salida,
                t.precio_entry, t.precio_stop, t.precio_t1, t.precio_t2, t.precio_t3,
                t.precio_salida_final, t.dias_en_trade, t.motivo_salida,
                t.r_realizado, t.score, t.atr_pct,
                t.fraccion_t1_cerrada, t.fraccion_t2_cerrada, t.fraccion_t3_cerrada,
                t.precio_salida_t1 or "", t.precio_salida_t2 or "", t.precio_salida_t3 or "",
            ])
        return rows


# ════════════════════════════════════════════════════════════════
# HELPERS DE SCORING (versión diaria — sin gate fundamental)
# ════════════════════════════════════════════════════════════════

def _score_diario(
    p1_ok: bool, p2_ok: bool, p3_ok: bool, p4_ok: bool,
    smi_actual: Optional[float],
) -> float:
    """
    Scoring simplificado para backtest diario (Ruta A).
    Gate fundamental = 20pts siempre (backtest no llama a yfinance).
    Mismo cálculo que signal_engine.calcular_score() con gate_puntos=20.
    """
    alineacion = PESO_ALINEACION_TF if (p1_ok and p2_ok) else (
        PESO_ALINEACION_TF * 0.5 if (p1_ok or p2_ok) else 0.0
    )
    zona_entrada = PESO_ZONA_ENTRADA if (p2_ok and p4_ok) else (
        PESO_ZONA_ENTRADA * 0.5 if p2_ok else 0.0
    )
    if p3_ok:
        catalizador = PESO_CATALIZADOR
    elif smi_actual is not None and smi_actual > -100:
        catalizador = PESO_CATALIZADOR * 0.3
    else:
        catalizador = 0.0
    regimen_tendencia = PESO_REGIMEN_TENDENCIA if p1_ok else 0.0
    gate = float(PESO_GATE_FUNDAMENTAL)  # siempre 20pts en backtest

    return round(alineacion + zona_entrada + catalizador + regimen_tendencia + gate, 1)


# ════════════════════════════════════════════════════════════════
# DETECCIÓN DE SEÑAL EN BARRA i (walk-forward)
# ════════════════════════════════════════════════════════════════

def _detectar_señal_en_barra(
    i: int,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    regimen: str,
) -> Tuple[bool, float, float, Optional[float]]:
    """
    Evalúa si en la barra i hay señal de entrada, usando solo datos [0..i].
    En Ruta A, los 3 timeframes son el mismo (diario) — se pasan las
    mismas series a todos los pasos de la cascada.

    Retorna (señal_ok, score, atr_pct, smi_actual).
    """
    params = PARAMS_REGIMEN[regimen]
    hma_r_p = params["hma_rapida"]
    hma_l_p = params["hma_lenta"]
    smi_p   = params["smi_periodo"]
    umbral_os = params["umbral_os"]
    vol_min_pct = params["vol_min_pct"]

    slice_c = closes[:i + 1]
    slice_h = highs[:i + 1]
    slice_l = lows[:i + 1]
    slice_v = volumes[:i + 1]

    min_requerido = max(HMA50_REGIMEN_PERIODO + 5, hma_l_p + 5, smi_p + 10, HMA16_ENTRY_PERIODO + 5)
    if len(slice_c) < min_requerido:
        return False, 0.0, 0.0, None

    # ── Paso 1: HMA50 pendiente positiva ──
    hma50 = hma(slice_c, HMA50_REGIMEN_PERIODO)
    if np.isnan(hma50[-1]) or np.isnan(hma50[-2]):
        return False, 0.0, 0.0, None
    p1_ok = hma50[-1] > hma50[-2]

    # ATR%
    atr14 = atr(slice_h, slice_l, slice_c, 14)
    atr_valor = atr14[-1] if not np.isnan(atr14[-1]) else None
    atr_pct = (atr_valor / slice_c[-1] * 100) if atr_valor and slice_c[-1] else 0.0

    if not p1_ok:
        return False, 0.0, atr_pct, None

    # ── Paso 2: HMA-D (rápida > lenta, precio sobre lenta) ──
    hma_r = hma(slice_c, hma_r_p)
    hma_l = hma(slice_c, hma_l_p)
    if np.isnan(hma_r[-1]) or np.isnan(hma_l[-1]):
        return False, 0.0, atr_pct, None
    p2_ok = (slice_c[-1] > hma_l[-1]) and (hma_r[-1] > hma_l[-1])

    if not p2_ok:
        return False, 0.0, atr_pct, None

    # ── Paso 3: SMI saliendo de OS + volumen ──
    smi_vals = smi(slice_h, slice_l, slice_c, periodo_k=smi_p)
    smi_actual = smi_vals[-1] if not np.isnan(smi_vals[-1]) else None
    if smi_actual is None or np.isnan(smi_vals[-2]):
        return False, 0.0, atr_pct, None

    saliendo_os = smi_vals[-2] <= umbral_os and smi_vals[-1] > smi_vals[-2]

    vol_ok = False
    if len(slice_v) >= 20:
        vol_prom = np.mean(slice_v[-20:-1])
        vol_pct = (slice_v[-1] / vol_prom * 100) if vol_prom > 0 else 0
        vol_ok = vol_pct >= vol_min_pct

    p3_ok = saliendo_os and vol_ok

    if not p3_ok:
        return False, 0.0, atr_pct, smi_actual

    # ── Paso 4: cruce de precio sobre HMA16 ──
    hma16 = hma(slice_c, HMA16_ENTRY_PERIODO)
    if np.isnan(hma16[-1]) or np.isnan(hma16[-2]):
        return False, 0.0, atr_pct, smi_actual
    p4_ok = (slice_c[-2] <= hma16[-2]) and (slice_c[-1] > hma16[-1])

    if not p4_ok:
        return False, 0.0, atr_pct, smi_actual

    score = _score_diario(p1_ok, p2_ok, p3_ok, p4_ok, smi_actual)
    return True, score, atr_pct, smi_actual


# ════════════════════════════════════════════════════════════════
# SIMULACIÓN DE TRADE (walk-forward desde barra de entry)
# ════════════════════════════════════════════════════════════════

def _simular_trade(
    entry_idx: int,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    timestamps: List[str],
    stop: float,
    t1: float,
    t2: float,
    t3: float,
    precio_entry: float,
    regimen: str,
) -> Tuple[float, str, int, float, bool, bool, bool, Optional[float], Optional[float], Optional[float]]:
    """
    Simula el trade desde la barra entry_idx+1 hasta cierre o timeout.
    Retorna:
        (r_neto, motivo_salida, dias_en_trade, precio_salida_final,
         t1_ok, t2_ok, t3_ok,
         precio_salida_t1, precio_salida_t2, precio_salida_t3)

    Gestión de posición escalonada:
        T1 (40%): stop pasa a breakeven
        T2 (40%): trailing HMA16 sobre el 20% restante
        T3/trailing/timeout: cierre del 20% final
    """
    riesgo = precio_entry - stop
    if riesgo <= 0:
        return 0.0, "riesgo_invalido", 0, precio_entry, False, False, False, None, None, None

    n = len(closes)
    t1_ok = t2_ok = t3_ok = False
    precio_st1 = precio_st2 = precio_st3 = None
    stop_actual = stop  # se mueve a breakeven después de T1

    for j in range(entry_idx + 1, min(entry_idx + 1 + MAX_HOLD_DIAS, n)):
        high_j = highs[j]
        low_j  = lows[j]
        close_j = closes[j]
        dias = j - entry_idx

        if not t1_ok:
            # ── Pre-T1: stop original, buscamos T1 ──
            if low_j <= stop_actual:
                # Stop tocado antes de T1 — pérdida total
                r = -1.0 * FRACCION_T1 - 1.0 * FRACCION_T2 - 1.0 * FRACCION_T3
                # Asumimos exit en stop (conservador)
                return round(r, 3), "stop_pre_t1", dias, stop_actual, False, False, False, None, None, None

            if high_j >= t1:
                t1_ok = True
                precio_st1 = t1
                stop_actual = precio_entry  # breakeven

        elif not t2_ok:
            # ── Entre T1 y T2: stop en breakeven ──
            if low_j <= stop_actual:
                # Salida en breakeven: fracción T1 ganó, resto a breakeven (0R extra)
                r_t1 = (precio_st1 - precio_entry) / riesgo * FRACCION_T1
                r_resto = 0.0  # breakeven
                r = round(r_t1 + r_resto, 3)
                return r, "breakeven_post_t1", dias, stop_actual, True, False, False, precio_st1, None, None

            if high_j >= t2:
                t2_ok = True
                precio_st2 = t2

        else:
            # ── Post-T2: trailing HMA16 sobre 20% restante ──
            # Trailing = HMA16 calculado sobre closes hasta barra j
            trailing = _hma16_en_barra(closes, j)
            if trailing is None:
                trailing = stop_actual  # fallback

            if low_j <= trailing:
                t3_ok = True
                precio_st3 = trailing
                r_t1 = (precio_st1 - precio_entry) / riesgo * FRACCION_T1
                r_t2 = (precio_st2 - precio_entry) / riesgo * FRACCION_T2
                r_t3 = (trailing - precio_entry) / riesgo * FRACCION_T3
                r = round(r_t1 + r_t2 + r_t3, 3)
                return r, "trailing_post_t2", dias, trailing, True, True, True, precio_st1, precio_st2, precio_st3

            if high_j >= t3:
                t3_ok = True
                precio_st3 = t3
                r_t1 = (precio_st1 - precio_entry) / riesgo * FRACCION_T1
                r_t2 = (precio_st2 - precio_entry) / riesgo * FRACCION_T2
                r_t3 = (t3 - precio_entry) / riesgo * FRACCION_T3
                r = round(r_t1 + r_t2 + r_t3, 3)
                return r, "T1+T2+T3", dias, t3, True, True, True, precio_st1, precio_st2, precio_st3

    # ── Timeout (MAX_HOLD_DIAS) ──
    dias = min(MAX_HOLD_DIAS, n - entry_idx - 1)
    precio_cierre = closes[min(entry_idx + dias, n - 1)]

    if not t1_ok:
        # Nunca llegó a T1 — salida al precio actual
        r = (precio_cierre - precio_entry) / riesgo
        return round(r, 3), "timeout_sin_t1", dias, precio_cierre, False, False, False, None, None, None

    r_t1 = (precio_st1 - precio_entry) / riesgo * FRACCION_T1
    if not t2_ok:
        r_resto = (precio_cierre - precio_entry) / riesgo * (FRACCION_T2 + FRACCION_T3)
        return round(r_t1 + r_resto, 3), "timeout_post_t1", dias, precio_cierre, True, False, False, precio_st1, None, None

    r_t2 = (precio_st2 - precio_entry) / riesgo * FRACCION_T2
    r_t3 = (precio_cierre - precio_entry) / riesgo * FRACCION_T3
    return round(r_t1 + r_t2 + r_t3, 3), "timeout_post_t2", dias, precio_cierre, True, True, False, precio_st1, precio_st2, None


def _hma16_en_barra(closes: np.ndarray, idx: int) -> Optional[float]:
    """HMA16 calculado sobre closes[0..idx] (sin lookahead)."""
    if idx < HMA16_ENTRY_PERIODO + 5:
        return None
    val = hma(closes[:idx + 1], HMA16_ENTRY_PERIODO)[-1]
    return float(val) if not np.isnan(val) else None


# ════════════════════════════════════════════════════════════════
# ENGINE PRINCIPAL
# ════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Orquestador del backtest diario (Ruta A).

    Args:
        velas: VelasOHLCV con histórico diario completo del símbolo.
        symbol: ticker (para logging y resultado).
        regimen: régimen de volatilidad a usar. Si es None, se calcula
                 dinámicamente barra a barra desde ATR%(14).
    """

    def __init__(
        self,
        velas: VelasOHLCV,
        symbol: str,
        regimen: Optional[str] = None,
    ):
        self.velas   = velas
        self.symbol  = symbol
        self.regimen_fijo = regimen  # None = dinámico

        self.closes    = np.array(velas.closes, dtype=float)
        self.highs     = np.array(velas.highs,  dtype=float)
        self.lows      = np.array(velas.lows,   dtype=float)
        self.volumes   = np.array(velas.volumes, dtype=float)
        self.timestamps = velas.timestamps

    def _regimen_en_barra(self, i: int) -> str:
        """Determina el régimen de volatilidad en la barra i (walk-forward)."""
        if self.regimen_fijo:
            return self.regimen_fijo
        if i < 20:
            return REGIMEN_BACKTEST_DEFAULT
        atr14 = atr(self.highs[:i+1], self.lows[:i+1], self.closes[:i+1], 14)
        atr_val = atr14[-1]
        if np.isnan(atr_val) or self.closes[i] == 0:
            return REGIMEN_BACKTEST_DEFAULT
        atr_pct = atr_val / self.closes[i] * 100
        return clasificar_regimen(atr_pct) or REGIMEN_BACKTEST_DEFAULT

    def correr(self) -> ResultadoBacktest:
        """
        Ejecuta el backtest walk-forward barra a barra.
        Retorna ResultadoBacktest con todos los trades y métricas.
        """
        resultado = ResultadoBacktest(symbol=self.symbol)
        n = len(self.closes)
        en_trade = False
        siguiente_barra_libre = 0  # no abrir otro trade mientras hay uno abierto

        logger.info(f"Backtest {self.symbol}: {n} barras diarias")

        for i in range(n):
            if en_trade or i < siguiente_barra_libre:
                continue

            regimen = self._regimen_en_barra(i)
            señal_ok, score, atr_pct, smi_actual = _detectar_señal_en_barra(
                i, self.closes, self.highs, self.lows, self.volumes, regimen
            )

            if not señal_ok or score < SCORE_MINIMO:
                continue

            # Señal válida — calcular niveles y simular trade
            precio_entry = self.closes[i]
            atr14 = atr(self.highs[:i+1], self.lows[:i+1], self.closes[:i+1], 14)
            atr_valor = float(atr14[-1]) if not np.isnan(atr14[-1]) else precio_entry * 0.02

            stop, t1, t2, t3 = calcular_niveles_salida(precio_entry, atr_valor, regimen)

            fecha_entry = self.timestamps[i] if i < len(self.timestamps) else str(i)

            (
                r_neto, motivo_salida, dias_en_trade, precio_salida_final,
                t1_ok, t2_ok, t3_ok,
                precio_st1, precio_st2, precio_st3,
            ) = _simular_trade(
                i, self.closes, self.highs, self.lows, self.timestamps,
                stop, t1, t2, t3, precio_entry, regimen,
            )

            fecha_salida_idx = min(i + dias_en_trade, n - 1)
            fecha_salida = self.timestamps[fecha_salida_idx] if fecha_salida_idx < len(self.timestamps) else str(fecha_salida_idx)

            trade = TradeResult(
                symbol=self.symbol,
                regimen=regimen,
                fecha_entry=fecha_entry,
                fecha_salida=fecha_salida,
                precio_entry=round(precio_entry, 4),
                precio_stop=stop,
                precio_t1=t1,
                precio_t2=t2,
                precio_t3=t3,
                precio_salida_final=round(precio_salida_final, 4),
                dias_en_trade=dias_en_trade,
                motivo_salida=motivo_salida,
                r_realizado=r_neto,
                score=score,
                atr_pct=round(atr_pct, 2),
                fraccion_t1_cerrada=t1_ok,
                fraccion_t2_cerrada=t2_ok,
                fraccion_t3_cerrada=t3_ok,
                precio_salida_t1=precio_st1,
                precio_salida_t2=precio_st2,
                precio_salida_t3=precio_st3,
            )
            resultado.trades.append(trade)
            logger.info(
                f"  Trade {len(resultado.trades):03d} | {fecha_entry} | "
                f"entry={precio_entry:.2f} stop={stop:.2f} | "
                f"R={r_neto:+.2f} | {motivo_salida}"
            )

            # No abrir otro trade hasta que este cierre
            siguiente_barra_libre = i + dias_en_trade + 1
            en_trade = False  # reset (la lógica de bloqueo es siguiente_barra_libre)

        resultado.calcular_metricas()
        logger.info(f"Backtest {self.symbol} finalizado: {resultado.metricas}")
        return resultado


# ════════════════════════════════════════════════════════════════
# HELPER: CARGA DE VELAS DESDE HISTORICO SHEETS
# ════════════════════════════════════════════════════════════════

def cargar_velas_desde_historico(
    rows: List[dict],
    symbol: str,
) -> VelasOHLCV:
    """
    Convierte filas del formato de Historico_Merval_Raw (o similar)
    en un VelasOHLCV listo para el engine.

    Formato esperado de cada row (dict):
        {
            "symbol":          "GGAL",
            "timestamp":       "2026-06-20 10:30:00",
            "apertura":        1234.5,
            "maximo":          1260.0,
            "minimo":          1220.0,
            "precio":          1245.0,   # close
            "volumen_nominal": 9800000,
        }

    Rows deben llegar ya ordenados ascendentemente por timestamp y
    filtrados por el símbolo deseado (o se filtra acá si se pasa symbol).
    """
    velas = VelasOHLCV()
    for r in rows:
        if r.get("symbol") != symbol:
            continue
        try:
            velas.opens.append(float(r.get("apertura") or r.get("open") or 0))
            velas.highs.append(float(r.get("maximo") or r.get("high") or 0))
            velas.lows.append(float(r.get("minimo") or r.get("low") or 0))
            velas.closes.append(float(r.get("precio") or r.get("close") or 0))
            velas.volumes.append(float(r.get("volumen_nominal") or r.get("volume") or 0))
            velas.timestamps.append(str(r.get("timestamp", "")))
        except (ValueError, TypeError) as e:
            logger.warning(f"cargar_velas_desde_historico: fila ignorada ({e}): {r}")
    return velas


def cargar_velas_desde_alpaca(bars: List[dict], symbol: str) -> VelasOHLCV:
    """
    Convierte barras de la API de Alpaca (formato dict estándar) en VelasOHLCV.

    Formato esperado de cada bar:
        {
            "t": "2026-06-20T14:00:00Z",   # timestamp ISO
            "o": 150.2,
            "h": 152.5,
            "l": 149.8,
            "c": 151.3,
            "v": 1234567,
        }
    """
    velas = VelasOHLCV()
    for b in bars:
        try:
            velas.opens.append(float(b["o"]))
            velas.highs.append(float(b["h"]))
            velas.lows.append(float(b["l"]))
            velas.closes.append(float(b["c"]))
            velas.volumes.append(float(b["v"]))
            velas.timestamps.append(str(b.get("t", "")))
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"cargar_velas_desde_alpaca [{symbol}]: barra ignorada ({e}): {b}")
    return velas


# ════════════════════════════════════════════════════════════════
# BACKTEST MULTI-SÍMBOLO
# ════════════════════════════════════════════════════════════════

def correr_backtest_universo(
    velas_por_symbol: Dict[str, VelasOHLCV],
    regimen: Optional[str] = None,
) -> Dict[str, ResultadoBacktest]:
    """
    Corre el backtest para todos los símbolos del diccionario.
    Retorna un dict {symbol: ResultadoBacktest}.

    Args:
        velas_por_symbol: {symbol: VelasOHLCV}
        regimen: si se pasa, se usa fijo para todos. Si None, dinámico.
    """
    resultados = {}
    for symbol, velas in velas_por_symbol.items():
        if len(velas) < 60:
            logger.warning(f"Backtest {symbol}: menos de 60 barras, saltando")
            continue
        engine = BacktestEngine(velas, symbol, regimen=regimen)
        resultados[symbol] = engine.correr()
    return resultados


def metricas_agregadas(resultados: Dict[str, ResultadoBacktest]) -> Dict:
    """
    Agrega métricas de todos los símbolos en un resumen global.
    """
    todos_trades = []
    for r in resultados.values():
        todos_trades.extend(r.trades)

    if not todos_trades:
        return {"trades_totales": 0}

    r_vals = [t.r_realizado for t in todos_trades]
    ganadores = [x for x in todos_trades if x.r_realizado > 0]
    perdedores = [x for x in todos_trades if x.r_realizado <= 0]

    r_gan = sum(x.r_realizado for x in ganadores)
    r_per = abs(sum(x.r_realizado for x in perdedores))

    return {
        "simbolos_testeados": len(resultados),
        "trades_totales": len(todos_trades),
        "ganadores": len(ganadores),
        "perdedores": len(perdedores),
        "win_rate": round(len(ganadores) / len(todos_trades) * 100, 1),
        "r_promedio": round(float(np.mean(r_vals)), 2),
        "r_total": round(float(np.sum(r_vals)), 2),
        "profit_factor": round(r_gan / r_per, 2) if r_per > 0 else float("inf"),
        "mejor_simbolo": max(resultados, key=lambda s: resultados[s].metricas.get("r_total", 0)),
        "peor_simbolo":  min(resultados, key=lambda s: resultados[s].metricas.get("r_total", 0)),
    }
