"""
Simulador — Sistema GG Swing
============================
- Capital inicial: ARS 15.000.000 (ajustable, ver resumen: rango 15-20M)
- Sizing por RIESGO, no por % fijo de capital:
      cantidad = (capital_total × riesgo_pct) / (precio_entry − precio_stop)
  riesgo_pct depende del score de la señal (más score → más riesgo permitido):
      score 65–74  → 0.5%
      score 75–89  → 1.0%
      score 90–100 → 1.5%
- Máximo 1 posición abierta por símbolo a la vez.
- Sin tope explícito de exposición total (pendiente de revisar más adelante).
- Apertura: 10:30 hs Argentina. Cierre nuevas compras: 16:30. Cierre forzado: 16:50.

SALIDA ESCALONADA EN 3 TARGETS (reemplaza las 4 condiciones CCL del sistema viejo):
  T1 alcanzado → cierra 40% de la posición, stop sube a breakeven
                 (desde acá el peor resultado posible es PnL=0%, nunca pérdida)
  T2 alcanzado → cierra 40% adicional, queda 20% con trailing
  Cierre final → 3 variantes:
                   - trailing stop tocado (sobre el 20% remanente)
                   - stop loss directo (si nunca llegó a T1)
                   - máximo hold a 21 días

Esta clase NO calcula el trailing stop (HMA16) ni decide si hay señal de
entrada — eso es responsabilidad de signal_engine.py. Simulator solo
ejecuta la mecánica de capital/posiciones dado lo que signal_engine.py
le indica (entrar con tales niveles, o que el trailing actual es tal valor).

FIX (12/jul/2026): puede_comprar(), debe_cerrar_forzado() y
Posicion.dias_en_cartera() usaban datetime.now() naive como default
cuando no se les pasaba `ahora` explícito. En GitHub Actions (runner en
UTC) eso corre los horarios de mercado ~3hs — ej. a las 14:00 ART
(mercado abierto) el runner ve las 17:00 UTC y puede_comprar() devolvía
False de más, o debe_cerrar_forzado() disparaba un cierre forzado 3hs
antes de tiempo. Se corrigió usando datetime.now(TZ_ARG) como default.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
import logging

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python < 3.9

logger = logging.getLogger(__name__)

# ─── HORARIOS ────────────────────────────────────────────
HORA_APERTURA       = time(10, 30)
HORA_CIERRE_COMPRA  = time(16, 30)
HORA_CIERRE_FORZADO = time(16, 50)

# ─── PARÁMETROS DE CAPITAL ───────────────────────────────
CAPITAL_INICIAL    = 15_000_000.0
MAX_POSICIONES_POR_SIMBOLO = 1
MAX_HOLD_DIAS = 21

# ─── SIZING POR RIESGO, escalonado por score ─────────────
# (score_min, score_max) → riesgo_pct (decimal, ej. 0.005 = 0.5%)
TRAMOS_RIESGO_POR_SCORE = [
    (65, 75, 0.005),
    (75, 90, 0.010),
    (90, 101, 0.015),  # 101 para incluir score=100 en el tramo superior
]

# ─── SALIDA ESCALONADA ───────────────────────────────────
PCT_CIERRE_T1 = 0.40   # cierra 40% al llegar a T1
PCT_CIERRE_T2 = 0.40   # cierra 40% adicional al llegar a T2
# el 20% restante queda con trailing hasta cierre final

# ─── TIMEZONE ────────────────────────────────────────────
TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")


def ahora_argentina() -> datetime:
    """Hora actual con timezone de Argentina — usar SIEMPRE en vez de
    datetime.now() naive para cualquier decisión de horario de mercado."""
    return datetime.now(TZ_ARG)


def riesgo_pct_por_score(score: float) -> float:
    """Devuelve el % de riesgo de capital habilitado para un score dado."""
    for lo, hi, riesgo in TRAMOS_RIESGO_POR_SCORE:
        if lo <= score < hi:
            return riesgo
    # score fuera de rango esperado (no debería pasar el gate de 65pts) → no operar
    return 0.0


@dataclass
class Posicion:
    """Representa una posición abierta del Sistema GG Swing."""
    id:                 str
    symbol:             str
    score:              float
    cantidad_inicial:   float
    cantidad_restante:  float
    precio_entry:       float
    monto_entry:        float
    precio_stop:        float   # stop vigente — se mueve a breakeven tras T1
    precio_t1:           float
    precio_t2:           float
    precio_t3:           float   # referencia informativa (trailing maneja el cierre real)
    riesgo_pct:          float
    ts_entry:            str
    precio_actual:        float = 0.0
    pnl:                   float = 0.0
    pnl_pct:                float = 0.0
    pnl_max_pct:              float = 0.0
    t1_alcanzado:              bool = False
    t2_alcanzado:               bool = False
    stop_en_breakeven:           bool = False
    cantidad_cerrada_t1:        float = 0.0
    cantidad_cerrada_t2:        float = 0.0

    def dias_en_cartera(self, ahora: Optional[datetime] = None) -> int:
        ahora = ahora or ahora_argentina()
        try:
            entry_dt = datetime.strptime(self.ts_entry, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return 0
        return (ahora.date() - entry_dt.date()).days


@dataclass
class Operacion:
    """Registro de un cierre (parcial o total) del Sistema GG Swing."""
    id:             str
    symbol:         str
    tipo:           str          # PARCIAL_T1 / PARCIAL_T2 / CIERRE_FINAL
    cantidad:       float
    precio_entry:   float
    precio_exit:    float
    monto_entry:    float
    monto_exit:     float
    pnl:            float
    pnl_pct:        float
    ts_entry:       str
    ts_exit:        str
    motivo_cierre:  str   # TARGET_1 / TARGET_2 / TRAILING_STOP / STOP_LOSS /
                          # MAX_HOLD_21D / CIERRE_FORZADO / VENTA_MANUAL


class Simulador:
    """
    Motor de simulación del Sistema GG Swing.

    Flujo:
        ENTRADA → signal_engine.py decide; simulator.abrir_posicion() ejecuta
                  el sizing por riesgo y registra niveles (stop/T1/T2/T3).
        GESTIÓN → procesar_ciclo() actualiza precios y evalúa T1/T2/stop/
                  máximo hold. El trailing del remanente post-T2 se recibe
                  como parámetro (trailing_stops) calculado externamente.
        SALIDA   → cierre total o parcial según el caso, registrado como
                  Operacion.
    """

    def __init__(self, capital_inicial: float = CAPITAL_INICIAL):
        self.capital_inicial = capital_inicial
        self.efectivo         = capital_inicial
        self.posiciones:  Dict[str, Posicion] = {}   # 1 posición por símbolo
        self.operaciones: List[Operacion] = []
        self._op_counter = 0
        logger.info(f"✅ Simulador GG Swing inicializado. Capital: ${capital_inicial:,.0f}")

    # ──────────────────── ESTADO ─────────────────────────

    def capital_en_posiciones(self, precios: Dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.posiciones.items():
            precio = precios.get(sym, pos.precio_actual)
            total += pos.cantidad_restante * precio
        return total

    def capital_total(self, precios: Dict[str, float]) -> float:
        return self.efectivo + self.capital_en_posiciones(precios)

    def tiene_posicion(self, symbol: str) -> bool:
        return symbol in self.posiciones

    # ──────────────────── HORARIOS ───────────────────────

    def puede_comprar(self, ahora=None) -> bool:
        if ahora is None:
            ahora = ahora_argentina().time()
        t = ahora if isinstance(ahora, time) else ahora.time()
        return HORA_APERTURA <= t <= HORA_CIERRE_COMPRA

    def debe_cerrar_forzado(self, ahora=None) -> bool:
        if ahora is None:
            ahora = ahora_argentina().time()
        t = ahora if isinstance(ahora, time) else ahora.time()
        return t >= HORA_CIERRE_FORZADO

    # ──────────────────── SIZING ─────────────────────────

    def calcular_cantidad(
        self,
        score: float,
        precio_entry: float,
        precio_stop: float,
        precios: Dict[str, float],
    ) -> tuple:
        """
        Devuelve (cantidad, riesgo_pct, monto) según riesgo por score.
        cantidad = (capital_total × riesgo_pct) / (precio_entry − precio_stop)
        """
        riesgo_pct = riesgo_pct_por_score(score)
        if riesgo_pct <= 0:
            return 0.0, riesgo_pct, 0.0

        riesgo_por_unidad = precio_entry - precio_stop
        if riesgo_por_unidad <= 0:
            logger.warning(
                f"Stop inválido (>= entry): entry={precio_entry} stop={precio_stop}"
            )
            return 0.0, riesgo_pct, 0.0

        capital = self.capital_total(precios)
        monto_riesgo = capital * riesgo_pct
        cantidad = monto_riesgo / riesgo_por_unidad
        monto = cantidad * precio_entry

        # No exceder el efectivo disponible
        if monto > self.efectivo:
            cantidad = self.efectivo / precio_entry if precio_entry > 0 else 0.0
            monto = cantidad * precio_entry

        return cantidad, riesgo_pct, monto

    # ──────────────────── OPERACIONES ────────────────────

    def abrir_posicion(
        self,
        symbol: str,
        score: float,
        precio_entry: float,
        precio_stop: float,
        precio_t1: float,
        precio_t2: float,
        precio_t3: float,
        precios: Dict[str, float],
        ahora=None,
    ) -> Optional[Posicion]:
        if not self.puede_comprar(ahora):
            return None
        if self.tiene_posicion(symbol):
            return None
        if precio_entry <= 0 or precio_stop <= 0 or precio_stop >= precio_entry:
            return None

        cantidad, riesgo_pct, monto = self.calcular_cantidad(
            score, precio_entry, precio_stop, precios
        )
        if cantidad <= 0 or monto <= 0:
            return None

        self._op_counter += 1
        pos = Posicion(
            id=f"P{self._op_counter:04d}",
            symbol=symbol,
            score=score,
            cantidad_inicial=cantidad,
            cantidad_restante=cantidad,
            precio_entry=precio_entry,
            monto_entry=monto,
            precio_stop=precio_stop,
            precio_t1=precio_t1,
            precio_t2=precio_t2,
            precio_t3=precio_t3,
            riesgo_pct=riesgo_pct,
            ts_entry=ahora_argentina().strftime("%Y-%m-%d %H:%M:%S"),
            precio_actual=precio_entry,
        )

        self.efectivo -= monto
        self.posiciones[symbol] = pos

        logger.info(
            f"🟢 ENTRADA {symbol}: {cantidad:.2f} u. @ ${precio_entry:.2f} | "
            f"score={score:.0f} | riesgo={riesgo_pct*100:.1f}% | "
            f"stop=${precio_stop:.2f} T1=${precio_t1:.2f} T2=${precio_t2:.2f} T3=${precio_t3:.2f}"
        )
        return pos

    def _registrar_cierre(
        self,
        pos: Posicion,
        cantidad: float,
        precio_exit: float,
        motivo: str,
        tipo: str,
    ) -> Operacion:
        monto_entry_parcial = cantidad * pos.precio_entry
        monto_exit = cantidad * precio_exit
        pnl = monto_exit - monto_entry_parcial
        pnl_pct = ((precio_exit / pos.precio_entry) - 1) * 100 if pos.precio_entry else 0

        self.efectivo += monto_exit

        self._op_counter += 1
        op = Operacion(
            id=f"{pos.id}-{tipo}",
            symbol=pos.symbol,
            tipo=tipo,
            cantidad=cantidad,
            precio_entry=pos.precio_entry,
            precio_exit=precio_exit,
            monto_entry=monto_entry_parcial,
            monto_exit=monto_exit,
            pnl=pnl,
            pnl_pct=pnl_pct,
            ts_entry=pos.ts_entry,
            ts_exit=ahora_argentina().strftime("%Y-%m-%d %H:%M:%S"),
            motivo_cierre=motivo,
        )
        self.operaciones.append(op)

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} {tipo} {pos.symbol} [{motivo}]: {cantidad:.2f} u. "
            f"PnL ${pnl:+,.0f} ({pnl_pct:+.2f}%)"
        )
        return op

    def cerrar_todas(self, precios: Dict[str, float], motivo: str = "CIERRE_FORZADO"):
        cerradas = []
        for symbol in list(self.posiciones.keys()):
            pos = self.posiciones[symbol]
            precio = precios.get(symbol, pos.precio_actual) or pos.precio_entry
            op = self._registrar_cierre(pos, pos.cantidad_restante, precio, motivo, "CIERRE_FINAL")
            cerradas.append(op)
            del self.posiciones[symbol]
        return cerradas

    # ──────────────────── CICLO PRINCIPAL ────────────────

    def procesar_ciclo(
        self,
        precios: Dict[str, float],
        trailing_stops: Optional[Dict[str, float]] = None,
        ahora=None,
    ) -> dict:
        """
        trailing_stops: dict opcional {symbol: precio_trailing_actual},
        calculado externamente (signal_engine.py, vía HMA16) y solo
        relevante para posiciones que ya alcanzaron T2 (remanente 20%).
        """
        trailing_stops = trailing_stops or {}
        abiertas_evento = []
        cerradas = []
        parciales = []
        forzadas = []

        # 1. Cierre forzado 16:50
        if self.debe_cerrar_forzado(ahora):
            forzadas = self.cerrar_todas(precios, "CIERRE_FORZADO")
            return {"cerradas": [], "parciales": [], "forzadas": forzadas}

        # 2. Actualizar precios / PnL / pico de cada posición
        for sym, pos in self.posiciones.items():
            precio = precios.get(sym, 0)
            if precio <= 0:
                continue
            pos.precio_actual = precio
            pos.pnl = (precio - pos.precio_entry) * pos.cantidad_restante
            pos.pnl_pct = ((precio / pos.precio_entry) - 1) * 100 if pos.precio_entry else 0
            if pos.pnl_pct > pos.pnl_max_pct:
                pos.pnl_max_pct = pos.pnl_pct

        # 3. Evaluar salidas (orden de prioridad: stop > targets > max hold)
        for symbol in list(self.posiciones.keys()):
            pos = self.posiciones[symbol]
            precio = precios.get(symbol, 0)
            if precio <= 0:
                continue

            # 3a. Stop loss / breakeven vigente
            if precio <= pos.precio_stop:
                motivo = "STOP_LOSS" if not pos.stop_en_breakeven else "STOP_BREAKEVEN"
                op = self._registrar_cierre(pos, pos.cantidad_restante, precio, motivo, "CIERRE_FINAL")
                cerradas.append(op)
                del self.posiciones[symbol]
                continue

            # 3b. Trailing stop (solo aplica tras T2, sobre el remanente 20%)
            if pos.t2_alcanzado:
                trailing = trailing_stops.get(symbol)
                if trailing is not None and precio <= trailing:
                    op = self._registrar_cierre(
                        pos, pos.cantidad_restante, precio, "TRAILING_STOP", "CIERRE_FINAL"
                    )
                    cerradas.append(op)
                    del self.posiciones[symbol]
                    continue

            # 3c. Target 2 — cierra 40% adicional
            if pos.t1_alcanzado and not pos.t2_alcanzado and precio >= pos.precio_t2:
                cantidad_cerrar = pos.cantidad_inicial * PCT_CIERRE_T2
                cantidad_cerrar = min(cantidad_cerrar, pos.cantidad_restante)
                op = self._registrar_cierre(pos, cantidad_cerrar, precio, "TARGET_2", "PARCIAL_T2")
                parciales.append(op)
                pos.cantidad_restante -= cantidad_cerrar
                pos.cantidad_cerrada_t2 = cantidad_cerrar
                pos.t2_alcanzado = True
                if pos.cantidad_restante <= 0:
                    del self.posiciones[symbol]
                continue

            # 3d. Target 1 — cierra 40%, stop sube a breakeven
            if not pos.t1_alcanzado and precio >= pos.precio_t1:
                cantidad_cerrar = pos.cantidad_inicial * PCT_CIERRE_T1
                cantidad_cerrar = min(cantidad_cerrar, pos.cantidad_restante)
                op = self._registrar_cierre(pos, cantidad_cerrar, precio, "TARGET_1", "PARCIAL_T1")
                parciales.append(op)
                pos.cantidad_restante -= cantidad_cerrar
                pos.cantidad_cerrada_t1 = cantidad_cerrar
                pos.t1_alcanzado = True
                pos.precio_stop = pos.precio_entry  # breakeven: peor caso ya es PnL=0
                pos.stop_en_breakeven = True
                continue

            # 3e. Máximo hold 21 días
            if pos.dias_en_cartera(ahora if isinstance(ahora, datetime) else None) >= MAX_HOLD_DIAS:
                op = self._registrar_cierre(
                    pos, pos.cantidad_restante, precio, "MAX_HOLD_21D", "CIERRE_FINAL"
                )
                cerradas.append(op)
                del self.posiciones[symbol]
                continue

        return {"cerradas": cerradas, "parciales": parciales, "forzadas": forzadas}

    # ──────────────────── RESUMEN ────────────────────────

    def resumen(self, precios: Dict[str, float]) -> dict:
        cap_total = self.capital_total(precios)
        pnl_total = cap_total - self.capital_inicial
        pnl_pct = (pnl_total / self.capital_inicial) * 100 if self.capital_inicial else 0

        ops_finales = [o for o in self.operaciones if o.tipo == "CIERRE_FINAL"]
        ganadoras = [o for o in self.operaciones if o.pnl > 0]
        win_rate = (len(ganadoras) / len(self.operaciones) * 100) if self.operaciones else 0

        return {
            "capital_inicial":     self.capital_inicial,
            "efectivo":            self.efectivo,
            "en_posiciones":       self.capital_en_posiciones(precios),
            "capital_total":       cap_total,
            "pnl_total":           pnl_total,
            "pnl_pct":             pnl_pct,
            "operaciones_total":   len(self.operaciones),
            "trades_cerrados":     len(ops_finales),
            "operaciones_ganadoras": len(ganadoras),
            "win_rate":            win_rate,
            "posiciones_abiertas": len(self.posiciones),
        }

    def fila_sheets_operacion(self, op: Operacion) -> list:
        return [
            op.id, op.symbol, op.tipo,
            round(op.cantidad, 4),
            round(op.precio_entry, 2),
            round(op.precio_exit, 2),
            round(op.monto_entry, 2),
            round(op.monto_exit, 2),
            round(op.pnl, 2),
            round(op.pnl_pct, 4),
            op.ts_entry, op.ts_exit,
            op.motivo_cierre,
        ]

    def fila_sheets_estado(self, precios: Dict[str, float]) -> list:
        r = self.resumen(precios)
        return [
            ahora_argentina().strftime("%Y-%m-%d %H:%M:%S"),
            round(r["capital_total"], 2),
            round(r["efectivo"], 2),
            round(r["en_posiciones"], 2),
            round(r["pnl_total"], 2),
            round(r["pnl_pct"], 4),
            r["operaciones_total"],
            round(r["win_rate"], 2),
            r["posiciones_abiertas"],
          ]
          
