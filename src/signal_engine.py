"""
signal_engine.py — Sistema GG Swing
Motor de señales técnicas LONG-only.

Implementa la cascada de 4 condiciones definida en el diseño (Camino B):
  1. Régimen      -> HMA(50) del propio timeframe en pendiente positiva
  2. Tendencia    -> precio > HMA lenta, HMA rápida > HMA lenta
  3. Momentum     -> SMI saliendo de zona de sobreventa + volumen confirmado
  4. Entry trigger-> cruce del precio al alza sobre la HMA rápida

NOTA DE DISEÑO — agnosticismo de timeframe:
El diseño original pedía 1D (régimen) + 4H (tendencia) + 30min (momentum/
entry) en simultáneo. Como Ruta A (backtest) solo cuenta con velas diarias
de IOL, y Ruta B (30min real) recién está acumulando historia, este motor
usa UNA sola serie de velas (recibida como parámetro) y deriva las 4
condiciones dentro de ese mismo timeframe usando HMAs de distinto período
como proxy de "régimen" vs "tendencia". Esto permite correrlo hoy sobre
velas diarias y mañana sobre 30min sin tocar el código — cuando haya
suficiente historia intradía se puede reintroducir el multi-timeframe real
pasando series separadas.

Uso típico:
    señal = generar_senal("GGAL", highs, lows, closes, volumes, es_cedear=False)
    if señal.senal_valida:
        ...
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Universo de activos
# ---------------------------------------------------------------------------
MERVAL = ["GGAL", "YPFD", "PAMP", "BMA", "CEPU", "TGSU2", "SUPV", "BBAR"]
CEDEARS = ["MELI", "NVDA", "TSLA", "MSFT", "PLTR", "VIST", "MU", "AMZN", "IBIT", "META", "AAPL", "VALO"]
# VALO cotiza en EEUU (como MELI), no en el panel Merval de IOL — por eso
# get_panel("MerVal") siempre devolvía 8/9 símbolos. Corregido el 12/jul/2026.

# ---------------------------------------------------------------------------
# Parámetros por régimen de volatilidad (según ATR% diario, 14 períodos)
# ---------------------------------------------------------------------------
REGIMENES = {
    "BAJA":     {"atr_max": 1.5,        "hma_rapida": 21, "hma_lenta": 55, "stop_mult": 1.8, "smi_periodo": 14, "smi_os": -150, "vol_min": 1.10, "t1_rr": 1.5, "t2_rr": 2.8},
    "MEDIA":    {"atr_max": 2.5,        "hma_rapida": 18, "hma_lenta": 50, "stop_mult": 2.0, "smi_periodo": 12, "smi_os": -170, "vol_min": 1.15, "t1_rr": 1.6, "t2_rr": 3.0},
    "ALTA":     {"atr_max": 4.0,        "hma_rapida": 16, "hma_lenta": 45, "stop_mult": 2.2, "smi_periodo": 10, "smi_os": -190, "vol_min": 1.20, "t1_rr": 1.7, "t2_rr": 3.2},
    "MUY_ALTA": {"atr_max": float("inf"), "hma_rapida": 13, "hma_lenta": 40, "stop_mult": 2.5, "smi_periodo": 8,  "smi_os": -200, "vol_min": 1.30, "t1_rr": 1.8, "t2_rr": 3.5},
}

PESOS_SCORE = {
    "alineacion_timeframes": 25,
    "calidad_entrada": 20,
    "catalizador_momentum": 20,
    "regimen_tendencia": 15,
    "gate_fundamental": 20,
}

UMBRAL_SENAL = 65
MIN_VELAS_REQUERIDAS = 60  # HMA50 + margen de warm-up


# ---------------------------------------------------------------------------
# Indicadores base
# ---------------------------------------------------------------------------
def wma(values: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average. NaN en el warm-up inicial."""
    n = len(values)
    out = np.full(n, np.nan)
    if period < 1 or n < period:
        return out
    weights = np.arange(1, period + 1)
    denom = weights.sum()
    for i in range(period - 1, n):
        window = values[i - period + 1: i + 1]
        out[i] = np.dot(window, weights) / denom
    return out


def hma(values: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half = max(1, period // 2)
    sqrt_p = max(1, int(round(np.sqrt(period))))
    wma_half = wma(values, half)
    wma_full = wma(values, period)
    raw = 2 * wma_half - wma_full
    raw_clean = np.where(np.isnan(raw), values, raw)  # evita romper el 2do wma
    return wma(raw_clean, sqrt_p)


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range (suavizado tipo Wilder)."""
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    out = np.full(n, np.nan)
    if n >= period:
        out[period - 1] = tr[:period].mean()
        for i in range(period, n):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def atr_pct(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    a = atr(highs, lows, closes, period)
    return (a / closes) * 100


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    alpha = 2 / (span + 1)
    first = next((j for j in range(len(x)) if not np.isnan(x[j])), None)
    if first is None:
        return out
    out[first] = x[first]
    for j in range(first + 1, len(x)):
        prev = out[j - 1]
        out[j] = alpha * x[j] + (1 - alpha) * prev if not np.isnan(x[j]) else prev
    return out


def smi(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        period: int = 14, smooth1: int = 3, smooth2: int = 3) -> np.ndarray:
    """Stochastic Momentum Index (doble suavizado EMA)."""
    n = len(closes)
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh[i] = np.max(highs[i - period + 1: i + 1])
        ll[i] = np.min(lows[i - period + 1: i + 1])
    mid = (hh + ll) / 2
    diff = hh - ll
    rel = closes - mid

    rel_s = _ema(_ema(rel, smooth1), smooth2)
    diff_s = _ema(_ema(diff, smooth1), smooth2)
    with np.errstate(divide="ignore", invalid="ignore"):
        smi_val = np.where(diff_s != 0, 200 * rel_s / diff_s, 0)
    return smi_val


def _clasificar_regimen(atr_pct_valor: float):
    if np.isnan(atr_pct_valor):
        return "MEDIA", REGIMENES["MEDIA"]
    for nombre in ["BAJA", "MEDIA", "ALTA", "MUY_ALTA"]:
        if atr_pct_valor <= REGIMENES[nombre]["atr_max"]:
            return nombre, REGIMENES[nombre]
    return "MUY_ALTA", REGIMENES["MUY_ALTA"]


# ---------------------------------------------------------------------------
# Gate fundamental
# ---------------------------------------------------------------------------
def gate_fundamental(symbol: str, es_cedear: bool, modo_backtest: bool = False) -> dict:
    """
    CEDEARs: bloqueo real si earnings <48hs (vía yfinance).
    Merval: informativo, no bloqueante, flat 20pts (sin fuente confiable de
            fechas de balance para CNV/BYMA).
    En modo_backtest se evita llamar a yfinance (mantiene la simulación
    self-contained) y se devuelve flat 20pts también para CEDEARs.
    """
    if not es_cedear:
        return {"puntos": 20, "bloqueado": False, "razon": "Merval - gate informativo, no bloqueante"}

    if modo_backtest:
        return {"puntos": 20, "bloqueado": False, "razon": "Backtest - yfinance no invocado, flat 20pts"}

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        earnings_date = None
        if cal is not None:
            try:
                earnings_date = cal.get("Earnings Date", [None])[0]
            except Exception:
                earnings_date = None
        if earnings_date:
            dias = (earnings_date - datetime.now().date()).days
            if 0 <= dias < 2:
                return {"puntos": 0, "bloqueado": True, "razon": f"Earnings en {dias}d - bloqueado"}
        return {"puntos": 20, "bloqueado": False, "razon": "Sin catalizador negativo detectado"}
    except Exception as e:
        return {"puntos": 20, "bloqueado": False, "razon": f"yfinance no disponible ({e}), gate por defecto"}


# ---------------------------------------------------------------------------
# Señal principal
# ---------------------------------------------------------------------------
@dataclass
class Senal:
    symbol: str
    score: int
    regimen: str
    senal_valida: bool
    entry: Optional[float]
    stop: Optional[float]
    t1: Optional[float]
    t2: Optional[float]
    detalle: dict = field(default_factory=dict)


def generar_senal(symbol: str, highs, lows, closes, volumes,
                   es_cedear: bool, modo_backtest: bool = False) -> Senal:
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    n = len(closes)
    if n < MIN_VELAS_REQUERIDAS:
        return Senal(symbol, 0, "N/A", False, None, None, None, None,
                     {"error": f"insuficientes velas ({n}/{MIN_VELAS_REQUERIDAS})"})

    atrp = atr_pct(highs, lows, closes, 14)
    regimen_nombre, params = _clasificar_regimen(atrp[-1])

    hma_rap = hma(closes, params["hma_rapida"])
    hma_lenta = hma(closes, params["hma_lenta"])
    hma50 = hma(closes, 50)
    smi_val = smi(highs, lows, closes, params["smi_periodo"])

    vol_avg20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_avg20[i] = volumes[i - 19: i + 1].mean()
    vol_ratio = volumes[-1] / vol_avg20[-1] if vol_avg20[-1] and not np.isnan(vol_avg20[-1]) else 0

    # Condición 1: régimen (HMA50 pendiente positiva)
    cond1 = bool(not np.isnan(hma50[-1]) and not np.isnan(hma50[-2]) and hma50[-1] > hma50[-2])

    # Condición 2: tendencia (precio sobre banda, HMA rápida sobre lenta)
    cond2 = bool(not np.isnan(hma_lenta[-1]) and closes[-1] > hma_lenta[-1] and hma_rap[-1] > hma_lenta[-1])

    # Condición 3: momentum (SMI saliendo de sobreventa + volumen)
    cond3 = bool(
        not np.isnan(smi_val[-1]) and not np.isnan(smi_val[-2])
        and smi_val[-2] < params["smi_os"] and smi_val[-1] > smi_val[-2]
        and vol_ratio >= params["vol_min"]
    )

    # Condición 4: entry trigger (cruce de precio sobre HMA rápida)
    cond4 = bool(
        not np.isnan(hma_rap[-2]) and not np.isnan(hma_rap[-1])
        and closes[-2] <= hma_rap[-2] and closes[-1] > hma_rap[-1]
    )

    gate = gate_fundamental(symbol, es_cedear, modo_backtest=modo_backtest)

    score = 0
    score += PESOS_SCORE["regimen_tendencia"] if cond1 else 0
    if cond1 and cond2:
        score += PESOS_SCORE["alineacion_timeframes"]
    elif cond2:
        score += PESOS_SCORE["alineacion_timeframes"] // 2
    score += PESOS_SCORE["catalizador_momentum"] if cond3 else 0
    score += PESOS_SCORE["calidad_entrada"] if cond4 else 0
    score += gate["puntos"]

    senal_valida = (score >= UMBRAL_SENAL) and cond4 and not gate["bloqueado"]

    entry = stop = t1 = t2 = None
    if senal_valida:
        entry = float(closes[-1])
        atr_val = atr(highs, lows, closes, 14)[-1]
        stop = entry - params["stop_mult"] * atr_val
        riesgo = entry - stop
        t1 = entry + params["t1_rr"] * riesgo
        t2 = entry + params["t2_rr"] * riesgo

    return Senal(
        symbol=symbol,
        score=int(score),
        regimen=regimen_nombre,
        senal_valida=senal_valida,
        entry=entry,
        stop=stop,
        t1=t1,
        t2=t2,
        detalle={
            "cond1_regimen": cond1,
            "cond2_tendencia": cond2,
            "cond3_momentum": cond3,
            "cond4_entry_trigger": cond4,
            "atr_pct": None if np.isnan(atrp[-1]) else round(float(atrp[-1]), 2),
            "vol_ratio": round(float(vol_ratio), 2),
            "smi_actual": None if np.isnan(smi_val[-1]) else round(float(smi_val[-1]), 1),
            "gate_fundamental": gate,
        },
    )
