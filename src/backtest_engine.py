"""
backtest_engine.py — Motor de backtesting walk-forward (Ruta A, diario)
Sistema GG Swing
==========================================================================
Self-contained: NO depende de simulator.py (que gestiona posiciones reales
de producción, todavía sin conectar a signal_engine.py). El backtest
trackea múltiplos de R (riesgo = entry - stop), no montos en pesos —
coherente con el mismo criterio de mantener la simulación autocontenida
que ya se usa en el gate fundamental (modo_backtest de signal_engine.py).

Salida de cada trade en 3 tramos, igual al diseño de producción:
  - T1 (40%) -> stop sube a breakeven
  - T2 (40%) -> queda 20% con trailing (HMA rápida del régimen del trade)
  - Máximo hold: 21 días corridos desde la entrada

LIMITACIÓN CONOCIDA: la entrada se ejecuta al close del día en que
signal_engine.py emitió la señal (igual métrica que en vivo), no al open
del día siguiente. Para un backtest más estricto contra look-ahead bias,
se puede desplazar `entry_idx` a bars[i+1]["o"] — queda como mejora futura,
no bloqueante para una primera validación de la lógica HMA-D+SMI.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

import signal_engine as se

MAX_HOLD_DIAS = 21


@dataclass
class TradeResultado:
    symbol: str
    regimen: str
    fecha_entry: str
    fecha_salida: str
    entry: float
    stop: float
    t1: float
    t2: float
    t3: Optional[float]
    precio_salida_final: float
    dias: int
    motivo_salida: str
    r_realizado: float
    score: int
    atr_pct: float
    t1_cerrada: bool
    t2_cerrada: bool
    t3_cerrada: bool
    precio_t1: Optional[float]
    precio_t2: Optional[float]
    precio_t3: Optional[float]

    def to_row(self) -> list:
        """Fila lista para Backtest_Resultados (mismo orden que HEADERS)."""
        return [
            self.symbol, self.regimen, self.fecha_entry, self.fecha_salida,
            round(self.entry, 4), round(self.stop, 4), round(self.t1, 4), round(self.t2, 4),
            "" if self.t3 is None else round(self.t3, 4),
            round(self.precio_salida_final, 4), self.dias, self.motivo_salida,
            round(self.r_realizado, 3), self.score, round(self.atr_pct, 2),
            self.t1_cerrada, self.t2_cerrada, self.t3_cerrada,
            "" if self.precio_t1 is None else round(self.precio_t1, 4),
            "" if self.precio_t2 is None else round(self.precio_t2, 4),
            "" if self.precio_t3 is None else round(self.precio_t3, 4),
        ]


def _cerrar_trade(symbol, senal, fecha_entry, fecha_salida, dias, motivo,
                   precio_salida_final, r_parciales, t1_c, t2_c, t3_c,
                   p_t1, p_t2, p_t3) -> TradeResultado:
    r_realizado = sum(frac * r for frac, r in r_parciales)
    return TradeResultado(
        symbol=symbol, regimen=senal.regimen, fecha_entry=fecha_entry, fecha_salida=fecha_salida,
        entry=senal.entry, stop=senal.stop, t1=senal.t1, t2=senal.t2, t3=None,
        precio_salida_final=precio_salida_final, dias=dias, motivo_salida=motivo,
        r_realizado=r_realizado, score=senal.score,
        atr_pct=senal.detalle.get("atr_pct") or 0.0,
        t1_cerrada=t1_c, t2_cerrada=t2_c, t3_cerrada=t3_c,
        precio_t1=p_t1, precio_t2=p_t2, precio_t3=p_t3,
    )


def _simular_trade(bars: list, entry_idx: int, senal, regimen_params: dict, symbol: str) -> TradeResultado:
    entry = senal.entry
    stop_inicial = senal.stop
    t1, t2 = senal.t1, senal.t2
    riesgo = entry - stop_inicial

    stop_actual = stop_inicial
    t1_cerrada = t2_cerrada = t3_cerrada = False
    precio_t1 = precio_t2 = precio_t3 = None
    r_parciales = []  # [(fraccion, r_multiple), ...]

    fecha_entry = bars[entry_idx]["t"][:10]
    n = len(bars)
    fin_idx = min(entry_idx + MAX_HOLD_DIAS, n - 1)
    closes_hist = [b["c"] for b in bars[: entry_idx + 1]]

    for i in range(entry_idx + 1, fin_idx + 1):
        bar = bars[i]
        closes_hist.append(bar["c"])

        # 1) Stop primero (conservador: si el día toca stop, se prioriza sobre targets)
        if bar["l"] <= stop_actual:
            precio_cierre = stop_actual
            if not t1_cerrada:
                r_parciales.append((1.0, (precio_cierre - entry) / riesgo))
                motivo = "stop_loss"
            else:
                fraccion_restante = 0.2 if t2_cerrada else 0.6
                r_parciales.append((fraccion_restante, (precio_cierre - entry) / riesgo))
                motivo = "stop_breakeven"
            return _cerrar_trade(symbol, senal, fecha_entry, bar["t"][:10], i - entry_idx,
                                  motivo, precio_cierre, r_parciales,
                                  t1_cerrada, t2_cerrada, t3_cerrada, precio_t1, precio_t2, precio_t3)

        # 2) Target 1
        if not t1_cerrada and bar["h"] >= t1:
            t1_cerrada = True
            precio_t1 = t1
            r_parciales.append((0.4, (t1 - entry) / riesgo))
            stop_actual = entry  # breakeven

        # 3) Target 2
        if t1_cerrada and not t2_cerrada and bar["h"] >= t2:
            t2_cerrada = True
            precio_t2 = t2
            r_parciales.append((0.4, (t2 - entry) / riesgo))

        # 4) Trailing del 20% remanente con HMA rápida del régimen del trade
        if t2_cerrada and not t3_cerrada:
            hma_rap_serie = se.hma(np.array(closes_hist), regimen_params["hma_rapida"])
            hma_actual = hma_rap_serie[-1]
            if not np.isnan(hma_actual) and bar["c"] < hma_actual:
                t3_cerrada = True
                precio_t3 = bar["c"]
                r_parciales.append((0.2, (bar["c"] - entry) / riesgo))
                return _cerrar_trade(symbol, senal, fecha_entry, bar["t"][:10], i - entry_idx,
                                      "trailing_hma", bar["c"], r_parciales,
                                      t1_cerrada, t2_cerrada, t3_cerrada, precio_t1, precio_t2, precio_t3)

    # Llegó al máximo hold (21 días) sin cerrar todo
    bar_final = bars[fin_idx]
    fraccion_pendiente = 1.0
    if t1_cerrada:
        fraccion_pendiente -= 0.4
    if t2_cerrada:
        fraccion_pendiente -= 0.4
    if fraccion_pendiente > 0:
        r_parciales.append((fraccion_pendiente, (bar_final["c"] - entry) / riesgo))
    return _cerrar_trade(symbol, senal, fecha_entry, bar_final["t"][:10], fin_idx - entry_idx,
                          "max_hold_21d", bar_final["c"], r_parciales,
                          t1_cerrada, t2_cerrada, t3_cerrada, precio_t1, precio_t2, precio_t3)


def backtest_symbol(symbol: str, bars: List[dict], es_cedear: bool) -> List[TradeResultado]:
    """
    Recorre `bars` (OHLCV diario, ascendente, shape {t,o,h,l,c,v}) día por
    día. Genera señales con signal_engine (modo_backtest=True) y simula el
    trade completo cuando aparece una señal válida, sin solapar con un
    trade ya abierto (avanza el cursor hasta el cierre antes de buscar la
    próxima señal).
    """
    resultados = []
    n = len(bars)
    i = se.MIN_VELAS_REQUERIDAS

    while i < n - 1:
        highs = [b["h"] for b in bars[: i + 1]]
        lows = [b["l"] for b in bars[: i + 1]]
        closes = [b["c"] for b in bars[: i + 1]]
        volumes = [b["v"] for b in bars[: i + 1]]

        senal = se.generar_senal(symbol, highs, lows, closes, volumes, es_cedear, modo_backtest=True)

        if senal.senal_valida:
            _, params = se._clasificar_regimen(senal.detalle.get("atr_pct") or 0.0)
            trade = _simular_trade(bars, i, senal, params, symbol)
            resultados.append(trade)
            dias_avanzados = trade.dias if trade.dias > 0 else 1
            i += dias_avanzados + 1
        else:
            i += 1

    return resultados


def calcular_metricas(symbol: str, trades: List[TradeResultado]) -> list:
    """Fila lista para Backtest_Metricas (mismo orden que HEADERS)."""
    if not trades:
        return [symbol, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    r_values = [t.r_realizado for t in trades]
    ganadores = [r for r in r_values if r > 0]
    perdedores = [r for r in r_values if r <= 0]

    win_rate = 100 * len(ganadores) / len(trades)
    r_prom = sum(r_values) / len(r_values)
    r_total = sum(r_values)
    suma_ganancias = sum(ganadores) if ganadores else 0.0
    suma_perdidas = abs(sum(perdedores)) if perdedores else 0.0
    profit_factor = (suma_ganancias / suma_perdidas) if suma_perdidas > 0 else float("inf")

    acumulado = np.cumsum(r_values)
    pico = np.maximum.accumulate(acumulado)
    drawdowns = pico - acumulado
    max_dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0

    mejor = max(r_values)
    peor = min(r_values)
    dias_prom = sum(t.dias for t in trades) / len(trades)

    return [
        symbol, len(trades), round(float(win_rate), 1), round(float(r_prom), 3), round(float(r_total), 2),
        round(float(profit_factor), 2) if profit_factor != float("inf") else "inf",
        round(float(max_dd), 2), round(float(mejor), 2), round(float(peor), 2), round(float(dias_prom), 1),
    ]
    
