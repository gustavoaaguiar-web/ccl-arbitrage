"""
Signal Engine — Sistema GG Swing
==================================
Núcleo de generación de señales: indicadores técnicos, cascada de 4
condiciones, scoring 0-100 y gate fundamental. Agnóstico a timeframe y
a fuente de datos — recibe velas OHLCV ya armadas (dicts o listas
paralelas), sin importar si vienen de Alpaca (CEDEARs, vía ADR) o de
IOL/Historico_Merval_Raw resampleado (Merval).

CASCADA DE 4 CONDICIONES (orden estricto, cualquiera que falle descarta
la señal sin seguir evaluando):
    1. Régimen diario   → HMA50 diaria en pendiente positiva
    2. Tendencia 4H     → precio > banda HMA-D, HMA_rápida > HMA_lenta
    3. Momentum 30min   → SMI saliendo de zona OS + volumen > umbral régimen
    4. Entry trigger     → precio cruza por encima de su HMA16 (30min)

El entry trigger usa el cruce de PRECIO sobre HMA16 (no cambio de
pendiente, no doble cruce de medias) porque los 3 filtros previos ya
mitigan el riesgo de falsa señal — es solo el gatillo de timing fino.

REGÍMENES DE VOLATILIDAD (ATR%(14) diario, re-evaluado en cada ciclo):
    Baja      ATR% < 2.0
    Media     2.0 <= ATR% < 3.5
    Alta      3.5 <= ATR% < 5.5
    Muy Alta  ATR% >= 5.5

NIVELES DE SALIDA (alimentan simulator.py):
    stop = precio_entry - (ATR(14) * stop_multiplier[régimen])
    T1   = precio_entry + (riesgo_por_unidad * target1_rr[régimen])
    T2   = precio_entry + (riesgo_por_unidad * target2_rr[régimen])
    T3   = precio_entry + (riesgo_por_unidad * (target2_rr[régimen] + 1.0))
    trailing post-T2 = HMA16 actual (recalculado cada ciclo, ver
                        calcular_trailing_stop())

SCORING (0-100, umbral mínimo 65pts para emitir señal, solo LONG):
    Alineación timeframes (1D+4H)     25pts
    Calidad zona de entrada (HMA-D)   20pts
    Catalizador SMI + volumen         20pts
    Régimen de tendencia (HMA50 1D)   15pts
    Gate fundamental                  20pts

GATE FUNDAMENTAL:
    CEDEARs (automatizado vía yfinance):
        - Bloqueo duro si earnings reportados en <48hs (no se emite señal)
        - Si no bloquea: 10pts si revenue growth interanual > 0%,
          10pts si no hay noticias negativas relevantes en 7 días.
          Si el dato no se puede obtener, se asignan 10pts neutros
          para ese componente (no se penaliza por falla de fuente).
    Merval (informativo, no bloqueante):
        - Siempre 20pts completos — no hay fuente automatizable
          confiable de fechas de balance para CNV/BYMA. La ventana
          estacional/noticias se adjunta como info en la alerta,
          sin restar score.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── UNIVERSO ─────────────────────────────────────────────
MERVAL_SET = {"GGAL", "YPFD", "PAMP", "BMA", "CEPU", "TGSU2", "SUPV", "BBAR", "VALO"}
CEDEARS_SET = {"MELI", "NVDA", "TSLA", "MSFT", "PLTR", "VIST", "MU", "AMZN", "IBIT", "META", "AAPL"}

# ─── SCORE ────────────────────────────────────────────────
SCORE_MINIMO = 65
PESO_ALINEACION_TF = 25
PESO_ZONA_ENTRADA = 20
PESO_CATALIZADOR = 20
PESO_REGIMEN_TENDENCIA = 15
PESO_GATE_FUNDAMENTAL = 20

# ─── REGÍMENES DE VOLATILIDAD ─────────────────────────────
# Cortes de ATR%(14) diario
REGIMEN_CORTES = [
    ("BAJA", 0.0, 2.0),
    ("MEDIA", 2.0, 3.5),
    ("ALTA", 3.5, 5.5),
    ("MUY_ALTA", 5.5, float("inf")),
]

# Parámetros adaptativos por régimen (de la tabla cerrada en el resumen)
PARAMS_REGIMEN = {
    "BAJA": {
        "hma_rapida": 21, "hma_lenta": 55, "stop_mult": 1.8,
        "smi_periodo": 14, "umbral_os": -150, "vol_min_pct": 110,
        "t1_rr": 1.5, "t2_rr": 2.8,
    },
    "MEDIA": {
        "hma_rapida": 18, "hma_lenta": 50, "stop_mult": 2.0,
        "smi_periodo": 12, "umbral_os": -170, "vol_min_pct": 115,
        "t1_rr": 1.6, "t2_rr": 3.0,
    },
    "ALTA": {
        "hma_rapida": 16, "hma_lenta": 45, "stop_mult": 2.2,
        "smi_periodo": 10, "umbral_os": -190, "vol_min_pct": 120,
        "t1_rr": 1.7, "t2_rr": 3.2,
    },
    "MUY_ALTA": {
        "hma_rapida": 13, "hma_lenta": 40, "stop_mult": 2.5,
        "smi_periodo": 8, "umbral_os": -200, "vol_min_pct": 130,
        "t1_rr": 1.8, "t2_rr": 3.5,
    },
}

HMA16_ENTRY_PERIODO = 16
HMA50_REGIMEN_PERIODO = 50


# ════════════════════════════════════════════════════════════════
# INDICADORES TÉCNICOS PUROS
# ════════════════════════════════════════════════════════════════

def wma(valores, periodo: int) -> np.ndarray:
    """Weighted Moving Average — pesos lineales, más peso al dato reciente."""
    valores = np.asarray(valores, dtype=float)
    resultado = np.full(len(valores), np.nan)
    if len(valores) < periodo:
        return resultado
    pesos = np.arange(1, periodo + 1)
    for i in range(periodo - 1, len(valores)):
        ventana = valores[i - periodo + 1: i + 1]
        if np.any(np.isnan(ventana)):
            continue
        resultado[i] = np.dot(ventana, pesos) / pesos.sum()
    return resultado


def hma(valores, periodo: int) -> np.ndarray:
    """
    Hull Moving Average estándar:
        HMA(n) = WMA( 2*WMA(n/2) - WMA(n), sqrt(n) )
    """
    n2 = max(1, int(round(periodo / 2)))
    nsq = max(1, int(round(np.sqrt(periodo))))
    wma_n2 = wma(valores, n2)
    wma_n = wma(valores, periodo)
    diff = 2 * wma_n2 - wma_n
    return wma(diff, nsq)


def ema(valores, periodo: int) -> np.ndarray:
    """Exponential Moving Average estándar."""
    valores = np.asarray(valores, dtype=float)
    resultado = np.full(len(valores), np.nan)
    if len(valores) == 0:
        return resultado
    alpha = 2 / (periodo + 1)
    resultado[0] = valores[0]
    for i in range(1, len(valores)):
        prev = resultado[i - 1]
        resultado[i] = valores[i] if np.isnan(prev) else alpha * valores[i] + (1 - alpha) * prev
    return resultado


def smi(highs, lows, closes, periodo_k: int = 10, periodo_d: int = 3, periodo_ema: int = 3) -> np.ndarray:
    """
    Stochastic Momentum Index clásico, escala -100/+100.
    SMI = 100 * doble_EMA(close - midpoint) / (0.5 * doble_EMA(rango))
    midpoint = (highest_high + lowest_low) / 2 sobre periodo_k.
    """
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)

    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(periodo_k - 1, n):
        highest_high[i] = np.max(highs[i - periodo_k + 1: i + 1])
        lowest_low[i] = np.min(lows[i - periodo_k + 1: i + 1])

    midpoint = (highest_high + lowest_low) / 2
    diff = closes - midpoint
    rango = highest_high - lowest_low

    diff_ema2 = ema(ema(diff, periodo_d), periodo_ema)
    rango_ema2 = ema(ema(rango, periodo_d), periodo_ema)

    with np.errstate(divide="ignore", invalid="ignore"):
        resultado = 100 * diff_ema2 / (0.5 * rango_ema2)
    return resultado


def atr(highs, lows, closes, periodo: int = 14) -> np.ndarray:
    """Average True Range (media móvil simple del True Range)."""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if n == 0:
        return np.array([])

    tr = np.full(n, np.nan)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    resultado = np.full(n, np.nan)
    for i in range(periodo - 1, n):
        resultado[i] = np.mean(tr[i - periodo + 1: i + 1])
    return resultado


def clasificar_regimen(atr_pct: float) -> Optional[str]:
    """Clasifica el régimen de volatilidad según ATR%(14) diario."""
    if atr_pct is None or np.isnan(atr_pct):
        return None
    for nombre, lo, hi in REGIMEN_CORTES:
        if lo <= atr_pct < hi:
            return nombre
    return "MUY_ALTA"  # fallback si excede todos los cortes


# ════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ════════════════════════════════════════════════════════════════

@dataclass
class VelasOHLCV:
    """Contenedor agnóstico de velas: listas paralelas alineadas por índice."""
    opens: List[float] = field(default_factory=list)
    highs: List[float] = field(default_factory=list)
    lows: List[float] = field(default_factory=list)
    closes: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)

    def __len__(self):
        return len(self.closes)

    @property
    def ultimo_close(self) -> Optional[float]:
        return self.closes[-1] if self.closes else None


@dataclass
class ResultadoCascada:
    """Resultado de evaluar la cascada de 4 condiciones para un símbolo."""
    symbol: str
    paso_alcanzado: int            # 0-4, cuántos pasos pasó (4 = señal completa)
    regimen: Optional[str]
    atr_pct: Optional[float]
    atr_valor: Optional[float]
    razones: List[str] = field(default_factory=list)
    score: float = 0.0
    score_detalle: Dict[str, float] = field(default_factory=dict)
    precio_entry: Optional[float] = None
    precio_stop: Optional[float] = None
    precio_t1: Optional[float] = None
    precio_t2: Optional[float] = None
    precio_t3: Optional[float] = None
    gate_bloqueado: bool = False
    gate_motivo: Optional[str] = None

    @property
    def señal_valida(self) -> bool:
        return (
            self.paso_alcanzado >= 4
            and not self.gate_bloqueado
            and self.score >= SCORE_MINIMO
        )


# ════════════════════════════════════════════════════════════════
# CASCADA DE 4 CONDICIONES
# ════════════════════════════════════════════════════════════════

def paso1_regimen_diario(velas_1d: VelasOHLCV) -> Tuple[bool, Optional[str], Optional[float], Optional[float], List[str]]:
    """
    Paso 1: HMA50 diaria en pendiente positiva.
    Retorna (paso_ok, regimen, atr_pct, atr_valor, razones).
    """
    razones = []
    if len(velas_1d) < HMA50_REGIMEN_PERIODO + 5:
        razones.append(f"Datos insuficientes para HMA50 diaria ({len(velas_1d)} velas)")
        return False, None, None, None, razones

    hma50 = hma(velas_1d.closes, HMA50_REGIMEN_PERIODO)
    if np.isnan(hma50[-1]) or np.isnan(hma50[-2]):
        razones.append("HMA50 diaria no calculable aún")
        return False, None, None, None, razones

    pendiente_positiva = hma50[-1] > hma50[-2]

    atr14 = atr(velas_1d.highs, velas_1d.lows, velas_1d.closes, 14)
    atr_valor = atr14[-1] if not np.isnan(atr14[-1]) else None
    atr_pct = (atr_valor / velas_1d.ultimo_close * 100) if atr_valor and velas_1d.ultimo_close else None
    regimen = clasificar_regimen(atr_pct) if atr_pct is not None else None

    if not pendiente_positiva:
        razones.append("HMA50 diaria sin pendiente positiva — régimen no alcista")
        return False, regimen, atr_pct, atr_valor, razones

    if regimen is None:
        razones.append("No se pudo calcular ATR% para clasificar régimen")
        return False, regimen, atr_pct, atr_valor, razones

    razones.append(f"Régimen diario alcista confirmado (HMA50 ↑, régimen={regimen}, ATR%={atr_pct:.2f})")
    return True, regimen, atr_pct, atr_valor, razones


def paso2_tendencia_4h(velas_4h: VelasOHLCV, regimen: str) -> Tuple[bool, List[str], Optional[float]]:
    """
    Paso 2: precio > banda HMA-D, HMA_rápida > HMA_lenta (4H).
    Retorna (paso_ok, razones, hma_lenta_actual).
    """
    razones = []
    params = PARAMS_REGIMEN[regimen]
    hma_r_periodo = params["hma_rapida"]
    hma_l_periodo = params["hma_lenta"]

    if len(velas_4h) < hma_l_periodo + 5:
        razones.append(f"Datos insuficientes para HMA-D 4H ({len(velas_4h)} velas, necesita {hma_l_periodo + 5})")
        return False, razones, None

    hma_rapida = hma(velas_4h.closes, hma_r_periodo)
    hma_lenta = hma(velas_4h.closes, hma_l_periodo)

    if np.isnan(hma_rapida[-1]) or np.isnan(hma_lenta[-1]):
        razones.append("HMA-D 4H no calculable aún")
        return False, razones, None

    precio_actual = velas_4h.ultimo_close
    sobre_banda = precio_actual > hma_lenta[-1]
    rapida_sobre_lenta = hma_rapida[-1] > hma_lenta[-1]

    if not (sobre_banda and rapida_sobre_lenta):
        razones.append(
            f"Tendencia 4H no confirmada (precio>{hma_lenta[-1]:.2f}: {sobre_banda}, "
            f"HMA_rap>HMA_lenta: {rapida_sobre_lenta})"
        )
        return False, razones, hma_lenta[-1]

    razones.append(f"Tendencia 4H confirmada (HMA{hma_r_periodo} > HMA{hma_l_periodo}, precio sobre banda)")
    return True, razones, hma_lenta[-1]


def paso3_momentum_30min(velas_30min: VelasOHLCV, regimen: str) -> Tuple[bool, List[str], Optional[float]]:
    """
    Paso 3: SMI saliendo de zona OS + volumen > umbral mínimo del régimen.
    Retorna (paso_ok, razones, smi_actual).
    """
    razones = []
    params = PARAMS_REGIMEN[regimen]
    smi_periodo = params["smi_periodo"]
    umbral_os = params["umbral_os"]
    vol_min_pct = params["vol_min_pct"]

    min_velas = smi_periodo + 10
    if len(velas_30min) < min_velas:
        razones.append(f"Datos insuficientes para SMI 30min ({len(velas_30min)} velas, necesita {min_velas})")
        return False, razones, None

    smi_vals = smi(velas_30min.highs, velas_30min.lows, velas_30min.closes, periodo_k=smi_periodo)
    if np.isnan(smi_vals[-1]) or np.isnan(smi_vals[-2]):
        razones.append("SMI 30min no calculable aún")
        return False, razones, None

    # "Saliendo de zona OS" = SMI estuvo bajo el umbral OS recientemente y ahora sube
    saliendo_de_os = smi_vals[-2] <= umbral_os and smi_vals[-1] > smi_vals[-2]

    if len(velas_30min.volumes) < 20:
        razones.append("Datos insuficientes de volumen")
        return False, razones, smi_vals[-1]

    vol_promedio = np.mean(velas_30min.volumes[-20:-1])  # promedio de las 19 anteriores (excluye la actual)
    vol_actual = velas_30min.volumes[-1]
    vol_pct = (vol_actual / vol_promedio * 100) if vol_promedio > 0 else 0
    volumen_confirma = vol_pct >= vol_min_pct

    if not (saliendo_de_os and volumen_confirma):
        razones.append(
            f"Momentum 30min no confirmado (SMI saliendo de OS: {saliendo_de_os} "
            f"[SMI={smi_vals[-1]:.1f}], volumen {vol_pct:.0f}% vs mínimo {vol_min_pct}%: {volumen_confirma})"
        )
        return False, razones, smi_vals[-1]

    razones.append(f"Momentum 30min confirmado (SMI={smi_vals[-1]:.1f} saliendo de OS, volumen={vol_pct:.0f}%)")
    return True, razones, smi_vals[-1]


def paso4_entry_trigger(velas_30min: VelasOHLCV) -> Tuple[bool, List[str], Optional[float]]:
    """
    Paso 4: precio cruza por encima de su HMA16 (30min).
    Retorna (paso_ok, razones, hma16_actual).
    """
    razones = []
    if len(velas_30min) < HMA16_ENTRY_PERIODO + 5:
        razones.append(f"Datos insuficientes para HMA16 entry trigger ({len(velas_30min)} velas)")
        return False, razones, None

    hma16 = hma(velas_30min.closes, HMA16_ENTRY_PERIODO)
    if np.isnan(hma16[-1]) or np.isnan(hma16[-2]):
        razones.append("HMA16 no calculable aún")
        return False, razones, None

    close_anterior = velas_30min.closes[-2]
    close_actual = velas_30min.closes[-1]
    cruce_al_alza = close_anterior <= hma16[-2] and close_actual > hma16[-1]

    if not cruce_al_alza:
        razones.append(f"Sin cruce de precio sobre HMA16 (precio={close_actual:.2f}, HMA16={hma16[-1]:.2f})")
        return False, razones, hma16[-1]

    razones.append(f"Entry trigger: precio cruzó sobre HMA16 (HMA16={hma16[-1]:.2f})")
    return True, razones, hma16[-1]


# ════════════════════════════════════════════════════════════════
# GATE FUNDAMENTAL
# ════════════════════════════════════════════════════════════════

def evaluar_gate_fundamental(symbol: str) -> Tuple[bool, Optional[str], float, Dict[str, str]]:
    """
    Evalúa el gate fundamental para un símbolo.
    Retorna (bloqueado, motivo_bloqueo, puntos_0_a_20, detalle_info).

    Merval: siempre 20pts, informativo, nunca bloquea (sin fuente
    automatizable confiable de fechas de balance CNV/BYMA).

    CEDEARs: bloqueo duro si earnings <48hs. Si no bloquea, 10pts por
    revenue growth >0% + 10pts por ausencia de noticias negativas en
    7 días. Si un dato no se puede obtener, ese componente da 10pts
    neutros (no penaliza fallas de fuente externa).
    """
    detalle = {}

    if symbol in MERVAL_SET:
        detalle["tipo"] = "Merval — gate informativo, no bloqueante"
        return False, None, float(PESO_GATE_FUNDAMENTAL), detalle

    if symbol not in CEDEARS_SET:
        detalle["tipo"] = "Símbolo fuera de universo conocido"
        return False, None, float(PESO_GATE_FUNDAMENTAL), detalle

    # ── CEDEARs vía yfinance ──
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance no instalado — gate fundamental CEDEARs en modo neutro")
        detalle["error"] = "yfinance no disponible"
        return False, None, float(PESO_GATE_FUNDAMENTAL), detalle

    puntos = 0.0

    try:
        ticker = yf.Ticker(symbol)

        # 1. Bloqueo duro por earnings <48hs
        try:
            calendar = ticker.calendar
            fecha_earnings = None
            if calendar is not None and not calendar.empty if hasattr(calendar, "empty") else calendar:
                if isinstance(calendar, dict):
                    fecha_earnings = calendar.get("Earnings Date", [None])
                    fecha_earnings = fecha_earnings[0] if isinstance(fecha_earnings, list) else fecha_earnings
                else:
                    try:
                        fecha_earnings = calendar.loc["Earnings Date"].iloc[0]
                    except Exception:
                        fecha_earnings = None

            if fecha_earnings is not None:
                if hasattr(fecha_earnings, "to_pydatetime"):
                    fecha_earnings = fecha_earnings.to_pydatetime()
                if isinstance(fecha_earnings, datetime):
                    horas_hasta_earnings = (fecha_earnings - datetime.now()).total_seconds() / 3600
                    detalle["proximos_earnings"] = fecha_earnings.strftime("%Y-%m-%d")
                    if 0 <= horas_hasta_earnings < 48:
                        return True, f"Earnings en <48hs ({fecha_earnings.strftime('%Y-%m-%d')})", 0.0, detalle
        except Exception as e:
            logger.warning(f"Gate fundamental {symbol}: error chequeando earnings — {e}")
            detalle["earnings_error"] = str(e)

        # 2. Revenue growth interanual
        try:
            info = ticker.info
            revenue_growth = info.get("revenueGrowth")
            if revenue_growth is not None:
                if revenue_growth > 0:
                    puntos += 10.0
                    detalle["revenue_growth"] = f"+{revenue_growth*100:.1f}% (positivo, +10pts)"
                else:
                    detalle["revenue_growth"] = f"{revenue_growth*100:.1f}% (negativo, +0pts)"
            else:
                puntos += 10.0  # dato no disponible → neutro
                detalle["revenue_growth"] = "no disponible (neutro, +10pts)"
        except Exception as e:
            puntos += 10.0
            detalle["revenue_growth"] = f"error obteniendo dato (neutro, +10pts): {e}"

        # 3. Noticias negativas recientes (7 días) — heurística simple por keywords
        try:
            noticias = ticker.news or []
            keywords_negativas = [
                "lawsuit", "investigation", "fraud", "recall", "downgrade",
                "miss", "decline", "plunge", "probe", "scandal",
            ]
            hace_7_dias = datetime.now() - timedelta(days=7)
            hay_negativa = False
            for n in noticias:
                ts = n.get("providerPublishTime")
                titulo = (n.get("title") or "").lower()
                if ts:
                    fecha_noticia = datetime.fromtimestamp(ts)
                    if fecha_noticia < hace_7_dias:
                        continue
                if any(kw in titulo for kw in keywords_negativas):
                    hay_negativa = True
                    break

            if hay_negativa:
                detalle["noticias"] = "noticia negativa detectada en 7 días (+0pts)"
            else:
                puntos += 10.0
                detalle["noticias"] = "sin noticias negativas relevantes (+10pts)"
        except Exception as e:
            puntos += 10.0
            detalle["noticias"] = f"error obteniendo noticias (neutro, +10pts): {e}"

    except Exception as e:
        logger.error(f"Gate fundamental {symbol}: fallo general yfinance — {e}")
        return False, None, float(PESO_GATE_FUNDAMENTAL), {"error": str(e), "modo": "neutro total"}

    return False, None, puntos, detalle


# ════════════════════════════════════════════════════════════════
# SCORING
# ════════════════════════════════════════════════════════════════

def calcular_score(
    paso1_ok: bool,
    paso2_ok: bool,
    paso3_ok: bool,
    paso4_ok: bool,
    regimen: Optional[str],
    smi_actual: Optional[float],
    vol_pct: Optional[float],
    gate_puntos: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Calcula el score 0-100 a partir de los componentes ya evaluados.
    Solo tiene sentido calcularlo si los 4 pasos de la cascada pasaron
    (si algún paso falla, no hay señal — pero el score se puede calcular
    igual para diagnóstico/logging).
    """
    detalle = {}

    # Alineación de timeframes (1D+4H) — 25pts: ambos pasos OK
    alineacion = PESO_ALINEACION_TF if (paso1_ok and paso2_ok) else (
        PESO_ALINEACION_TF * 0.5 if (paso1_ok or paso2_ok) else 0.0
    )
    detalle["alineacion_timeframes"] = alineacion

    # Calidad zona de entrada (HMA-D) — 20pts: paso2 OK + paso4 OK (trigger limpio)
    zona_entrada = PESO_ZONA_ENTRADA if (paso2_ok and paso4_ok) else (
        PESO_ZONA_ENTRADA * 0.5 if paso2_ok else 0.0
    )
    detalle["zona_entrada"] = zona_entrada

    # Catalizador SMI + volumen — 20pts: paso3 OK, escalado por qué tan fuerte es el catalizador
    if paso3_ok:
        catalizador = PESO_CATALIZADOR
    elif smi_actual is not None and smi_actual > -100:
        catalizador = PESO_CATALIZADOR * 0.3  # algo de momentum aunque no confirme del todo
    else:
        catalizador = 0.0
    detalle["catalizador_smi_volumen"] = catalizador

    # Régimen de tendencia (HMA50 1D) — 15pts: paso1 OK
    regimen_tendencia = PESO_REGIMEN_TENDENCIA if paso1_ok else 0.0
    detalle["regimen_tendencia"] = regimen_tendencia

    # Gate fundamental — ya viene calculado (0-20)
    detalle["gate_fundamental"] = gate_puntos

    score_total = alineacion + zona_entrada + catalizador + regimen_tendencia + gate_puntos
    return round(score_total, 1), detalle


# ════════════════════════════════════════════════════════════════
# NIVELES DE SALIDA
# ════════════════════════════════════════════════════════════════

def calcular_niveles_salida(
    precio_entry: float,
    atr_valor: float,
    regimen: str,
) -> Tuple[float, float, float, float]:
    """
    Calcula stop, T1, T2, T3 a partir del precio de entrada, ATR(14) y
    régimen de volatilidad. Retorna (stop, t1, t2, t3).
    """
    params = PARAMS_REGIMEN[regimen]
    stop = precio_entry - (atr_valor * params["stop_mult"])
    riesgo_por_unidad = precio_entry - stop

    t1 = precio_entry + (riesgo_por_unidad * params["t1_rr"])
    t2 = precio_entry + (riesgo_por_unidad * params["t2_rr"])
    t3 = precio_entry + (riesgo_por_unidad * (params["t2_rr"] + 1.0))

    return round(stop, 4), round(t1, 4), round(t2, 4), round(t3, 4)


def calcular_trailing_stop(velas_30min: VelasOHLCV) -> Optional[float]:
    """
    Trailing stop post-T2: valor actual de HMA16 sobre el timeframe de
    entrada (30min), recalculado en cada ciclo. Usado para alimentar
    trailing_stops en simulator.procesar_ciclo().
    """
    if len(velas_30min) < HMA16_ENTRY_PERIODO + 5:
        return None
    hma16 = hma(velas_30min.closes, HMA16_ENTRY_PERIODO)
    valor = hma16[-1]
    return round(float(valor), 4) if not np.isnan(valor) else None


# ════════════════════════════════════════════════════════════════
# ORQUESTADOR PRINCIPAL
# ════════════════════════════════════════════════════════════════

def evaluar_simbolo(
    symbol: str,
    velas_1d: VelasOHLCV,
    velas_4h: VelasOHLCV,
    velas_30min: VelasOHLCV,
) -> ResultadoCascada:
    """
    Evalúa la cascada completa de 4 condiciones + scoring + gate
    fundamental para un símbolo, y devuelve un ResultadoCascada con
    todo lo necesario para decidir si emitir una señal y con qué niveles.

    Detiene la cascada en el primer paso que falle (no sigue evaluando
    pasos posteriores), salvo el cálculo de régimen/ATR que siempre se
    intenta en el paso 1 para diagnóstico.
    """
    razones_totales: List[str] = []

    # Paso 1
    p1_ok, regimen, atr_pct, atr_valor, r1 = paso1_regimen_diario(velas_1d)
    razones_totales += r1
    if not p1_ok:
        return ResultadoCascada(
            symbol=symbol, paso_alcanzado=0, regimen=regimen,
            atr_pct=atr_pct, atr_valor=atr_valor, razones=razones_totales,
        )

    # Paso 2
    p2_ok, r2, hma_lenta_4h = paso2_tendencia_4h(velas_4h, regimen)
    razones_totales += r2
    if not p2_ok:
        return ResultadoCascada(
            symbol=symbol, paso_alcanzado=1, regimen=regimen,
            atr_pct=atr_pct, atr_valor=atr_valor, razones=razones_totales,
        )

    # Paso 3
    p3_ok, r3, smi_actual = paso3_momentum_30min(velas_30min, regimen)
    razones_totales += r3
    if not p3_ok:
        return ResultadoCascada(
            symbol=symbol, paso_alcanzado=2, regimen=regimen,
            atr_pct=atr_pct, atr_valor=atr_valor, razones=razones_totales,
        )

    # Paso 4
    p4_ok, r4, hma16_actual = paso4_entry_trigger(velas_30min)
    razones_totales += r4
    if not p4_ok:
        return ResultadoCascada(
            symbol=symbol, paso_alcanzado=3, regimen=regimen,
            atr_pct=atr_pct, atr_valor=atr_valor, razones=razones_totales,
        )

    # Los 4 pasos pasaron — calcular gate fundamental, score y niveles
    gate_bloqueado, gate_motivo, gate_puntos, gate_detalle = evaluar_gate_fundamental(symbol)

    if gate_bloqueado:
        razones_totales.append(f"Gate fundamental BLOQUEA: {gate_motivo}")
        return ResultadoCascada(
            symbol=symbol, paso_alcanzado=4, regimen=regimen,
            atr_pct=atr_pct, atr_valor=atr_valor, razones=razones_totales,
            gate_bloqueado=True, gate_motivo=gate_motivo,
        )

    score, score_detalle = calcular_score(
        p1_ok, p2_ok, p3_ok, p4_ok, regimen, smi_actual, None, gate_puntos
    )
    razones_totales.append(f"Score final: {score}/100 (mínimo {SCORE_MINIMO})")

    precio_entry = velas_30min.ultimo_close
    stop, t1, t2, t3 = calcular_niveles_salida(precio_entry, atr_valor, regimen)

    return ResultadoCascada(
        symbol=symbol, paso_alcanzado=4, regimen=regimen,
        atr_pct=atr_pct, atr_valor=atr_valor, razones=razones_totales,
        score=score, score_detalle=score_detalle,
        precio_entry=precio_entry, precio_stop=stop,
        precio_t1=t1, precio_t2=t2, precio_t3=t3,
        gate_bloqueado=False, gate_motivo=None,
    )
