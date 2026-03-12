"""
Simulador de Arbitraje CCL
==========================
- Capital inicial: ARS 10.000.000
- Por operación: 15% del capital total (efectivo + posiciones valoradas)
- Máximo 2 posiciones por especie
- Apertura: 10:30 hs Argentina (ambos mercados abiertos)
- Cierre nuevas compras: 16:30 hs
- Cierre forzado: 16:50 hs

CONDICIONES DE SALIDA (cualquiera activa el cierre, en orden de prioridad):
  [C] PnL precio ≤ -0.80%                          → stop loss duro
  [A] desvío CCL ≥ +0.10%                          → reversión completa del spread
  [B] dev alguna vez ≥ 0%  AND  pnl_max ≥ +0.30%
      AND  caída desde pico ≥ 0.25%                → trailing con ganancia confirmada
  [D] PnL precio ≥ +2.40%                          → take profit puro

MODELO DE CLIMA (Simons):
  El dict `climas` que recibe procesar_ciclo() debe venir del HMM entrenado
  sobre log-returns del precio USD del subyacente (NO niveles CCL).
  Esto garantiza que clima y señal sean variables ortogonales:

    Señal  →  CCL del CEDEAR bajo vs mediana    (oportunidad de arbitraje)
    Clima  →  régimen bull/bear en USD           (momentum del subyacente)

  Compra solo cuando AMBOS coinciden. Venta solo por desvío CCL.
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# ─── HORARIOS ────────────────────────────────────────────
HORA_APERTURA       = time(10, 30)
HORA_CIERRE_COMPRA  = time(16, 30)
HORA_CIERRE_FORZADO = time(16, 50)

# ─── PARÁMETROS DE ENTRADA ───────────────────────────────
CAPITAL_INICIAL            = 10_000_000.0
PCT_POR_OPERACION          = 0.15
MAX_POSICIONES_POR_ESPECIE = 2

UMBRAL_COMPRA = -0.6   # desvío CCL mínimo para comprar (%)

# ─── PARÁMETROS DE SALIDA ────────────────────────────────
UMBRAL_VENTA_A          = 0.15   # desvío CCL — [A] reversión completa del spread (%)
UMBRAL_VENTA_B_DEV      = 0.00   # desvío CCL histórico — [B] spread alguna vez neutro (%)
UMBRAL_VENTA_B_PNL_MIN  = 0.30   # PnL % mínimo alcanzado para habilitar trailing B (%)
UMBRAL_VENTA_B_CAIDA    = 0.25   # caída desde pico PnL% para disparar trailing B (%)
TAKE_PROFIT_D           = 2.40   # PnL precio — [D] take profit puro (%)
STOP_LOSS_C             = -0.80  # PnL precio — [C] stop loss duro (%)


@dataclass
class Posicion:
    """Representa una posición abierta."""
    id:            str
    symbol:        str
    cantidad:      float
    precio_entry:  float
    monto_entry:   float
    ts_entry:      str
    ccl_entry:     float
    dev_entry:     float
    precio_actual: float = 0.0
    pnl:           float = 0.0
    pnl_pct:       float = 0.0   # PnL % actual
    pnl_max_pct:        float = 0.0    # pico máximo de PnL % (para trailing B)
    dev_max_alcanzado:  float = -99.0  # pico de desvío CCL alcanzado;
                                       # una vez que llega a ≥0% queda registrado
                                       # y el trailing B se activa aunque el dev
                                       # vuelva negativo en el ciclo siguiente


@dataclass
class Operacion:
    """Registro de una operación cerrada."""
    id:            str
    symbol:        str
    tipo:          str
    cantidad:      float
    precio_entry:  float
    precio_exit:   float
    monto_entry:   float
    monto_exit:    float
    pnl:           float
    pnl_pct:       float
    ts_entry:      str
    ts_exit:       str
    motivo_cierre: str   # SALIDA_A / SALIDA_B / TAKE_PROFIT_D / STOP_LOSS_C /
                         # CIERRE_FORZADO / VENTA_MANUAL


class Simulador:
    """
    Motor de simulación de arbitraje CCL intradiario.

    Lógica de decisión:
        COMPRA  →  desvío CCL < -0.5%  AND  clima == "🟢 BULL"
        VENTA   →  cualquiera de las condiciones de salida se cumple
    """

    def __init__(self, capital_inicial: float = CAPITAL_INICIAL):
        self.capital_inicial   = capital_inicial
        self.efectivo          = capital_inicial
        self.posiciones:  Dict[str, List[Posicion]] = {}
        self.operaciones: List[Operacion] = []
        self._op_counter = 0

    # ──────────────────── ESTADO ─────────────────────────

    def capital_en_posiciones(self, precios_ars: Dict[str, float]) -> float:
        total = 0.0
        for sym, poss in self.posiciones.items():
            precio = precios_ars.get(sym, 0)
            for pos in poss:
                total += pos.cantidad * precio
        return total

    def capital_total(self, precios_ars: Dict[str, float]) -> float:
        return self.efectivo + self.capital_en_posiciones(precios_ars)

    def monto_por_operacion(self, precios_ars: Dict[str, float]) -> float:
        return self.capital_total(precios_ars) * PCT_POR_OPERACION

    def posiciones_abiertas_count(self, symbol: str) -> int:
        return len(self.posiciones.get(symbol, []))

    # ──────────────────── HORARIOS ───────────────────────

    def puede_comprar(self, ahora=None) -> bool:
        if ahora is None:
            ahora = datetime.now().time()
        return HORA_APERTURA <= ahora <= HORA_CIERRE_COMPRA

    def debe_cerrar_forzado(self, ahora=None) -> bool:
        if ahora is None:
            ahora = datetime.now().time()
        return ahora >= HORA_CIERRE_FORZADO

    def mercado_abierto(self, ahora=None) -> bool:
        if ahora is None:
            ahora = datetime.now().time()
        return ahora >= HORA_APERTURA

    # ──────────────────── CONDICIONES DE SALIDA ──────────

    def _evaluar_salida(self, pos: Posicion, dev: float) -> Optional[str]:
        """
        Evalúa si una posición debe cerrarse. Retorna el motivo o None.

        Orden de prioridad:
          1. [C] Stop loss duro         → pnl_pct ≤ -0.80%
          2. [A] Reversión del spread   → dev ≥ +0.10%
          3. [B] Trailing confirmado    → dev_max ≥ 0%  AND  pnl_max ≥ +0.30%
                                          AND  caída desde pico ≥ 0.25%
          4. [D] Take profit puro       → pnl_pct ≥ +2.40%

        Todos los flags históricos (dev_max_alcanzado, pnl_max_pct) se
        actualizan en procesar_ciclo() antes de llamar a este método,
        por lo que las condiciones B no requieren que los eventos coincidan
        en el mismo ciclo de 60s.
        """
        pnl_pct = pos.pnl_pct

        # [C] Stop loss duro — prioridad máxima
        if pnl_pct <= STOP_LOSS_C:
            return "STOP_LOSS_C"

        # [A] Reversión completa del spread — solo si la operación está en ganancia
        # Evita cerrar con pérdida por una reversión del spread sin ganancia de precio
        if dev >= UMBRAL_VENTA_A and pnl_pct > 0:
            return "SALIDA_A"

        # [B] Trailing con ganancia confirmada:
        #   - El spread alguna vez se neutralizó (dev_max_alcanzado ≥ 0%)
        #   - El precio llegó a ganar al menos +0.30% en algún momento
        #   - Desde ese pico el precio cayó ≥ 0.25%
        if (pos.dev_max_alcanzado >= UMBRAL_VENTA_B_DEV
                and pos.pnl_max_pct >= UMBRAL_VENTA_B_PNL_MIN):
            caida_desde_pico = pos.pnl_max_pct - pnl_pct
            if caida_desde_pico >= UMBRAL_VENTA_B_CAIDA:
                return "SALIDA_B"

        # [D] Take profit puro — independiente del desvío CCL
        if pnl_pct >= TAKE_PROFIT_D:
            return "TAKE_PROFIT_D"

        return None

    # ──────────────────── OPERACIONES ────────────────────

    def abrir_posicion(
        self,
        symbol: str,
        precio_ars: float,
        ccl: float,
        dev: float,
        precios_ars: Dict[str, float],
        ahora=None,
    ) -> Optional[Posicion]:
        if not self.puede_comprar(ahora):
            return None
        if self.posiciones_abiertas_count(symbol) >= MAX_POSICIONES_POR_ESPECIE:
            return None
        if precio_ars <= 0:
            return None

        monto = self.monto_por_operacion(precios_ars)
        if monto > self.efectivo:
            monto = self.efectivo
        if monto <= 0:
            return None

        cantidad = monto / precio_ars

        self._op_counter += 1
        pos = Posicion(
            id=f"P{self._op_counter:04d}",
            symbol=symbol,
            cantidad=cantidad,
            precio_entry=precio_ars,
            monto_entry=monto,
            ts_entry=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ccl_entry=ccl,
            dev_entry=dev,
            precio_actual=precio_ars,
            pnl_pct=0.0,
            pnl_max_pct=0.0,
            dev_max_alcanzado=-99.0,
        )

        self.efectivo -= monto
        if symbol not in self.posiciones:
            self.posiciones[symbol] = []
        self.posiciones[symbol].append(pos)

        logger.info(f"🟢 COMPRA {symbol}: {cantidad:.2f} u. @ ${precio_ars:.2f} | Monto: ${monto:,.0f}")
        return pos

    def cerrar_posicion(
        self,
        symbol: str,
        pos: Posicion,
        precio_ars: float,
        motivo: str,
    ) -> Operacion:
        monto_exit = pos.cantidad * precio_ars
        pnl        = monto_exit - pos.monto_entry
        pnl_pct    = (pnl / pos.monto_entry) * 100

        self.efectivo += monto_exit

        op = Operacion(
            id=pos.id,
            symbol=symbol,
            tipo="COMPRA/VENTA",
            cantidad=pos.cantidad,
            precio_entry=pos.precio_entry,
            precio_exit=precio_ars,
            monto_entry=pos.monto_entry,
            monto_exit=monto_exit,
            pnl=pnl,
            pnl_pct=pnl_pct,
            ts_entry=pos.ts_entry,
            ts_exit=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            motivo_cierre=motivo,
        )
        self.operaciones.append(op)

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"{emoji} CIERRE {symbol} [{motivo}]: PnL ${pnl:+,.0f} ({pnl_pct:+.2f}%)")
        return op

    def cerrar_todas(self, precios_ars: Dict[str, float], motivo: str = "CIERRE_FORZADO"):
        cerradas = []
        for symbol in list(self.posiciones.keys()):
            precio = precios_ars.get(symbol, 0)
            if precio <= 0:
                continue
            for pos in list(self.posiciones[symbol]):
                op = self.cerrar_posicion(symbol, pos, precio, motivo)
                cerradas.append(op)
            self.posiciones[symbol] = []
        return cerradas

    # ──────────────────── CICLO PRINCIPAL ────────────────

    def procesar_ciclo(
        self,
        ccl_map: Dict[str, float],
        ccl_avg: float,
        precios_ars: Dict[str, float],
        climas: Dict[str, str],
        ahora=None,
    ) -> dict:
        abiertas = []
        cerradas = []
        forzadas = []

        # 1. Cierre forzado 16:50
        if self.debe_cerrar_forzado(ahora):
            forzadas = self.cerrar_todas(precios_ars, "CIERRE_FORZADO")
            return {"abiertas": [], "cerradas": [], "forzadas": forzadas}

        # 2. Actualizar precios, PnL y pico máximo de cada posición abierta
        for sym, poss in self.posiciones.items():
            precio = precios_ars.get(sym, 0)
            for pos in poss:
                pos.precio_actual = precio
                pos.pnl     = (precio - pos.precio_entry) * pos.cantidad
                pos.pnl_pct = ((precio / pos.precio_entry) - 1) * 100 if pos.precio_entry else 0
                # Actualizar pico de PnL — nunca retroceder
                if pos.pnl_pct > pos.pnl_max_pct:
                    pos.pnl_max_pct = pos.pnl_pct

        # 3. Evaluar condiciones de salida para posiciones abiertas
        for symbol in list(self.posiciones.keys()):
            if not self.posiciones[symbol]:
                continue
            ccl = ccl_map.get(symbol, 0)
            if ccl_avg == 0:
                continue
            dev    = (ccl / ccl_avg - 1) * 100
            precio = precios_ars.get(symbol, 0)
            if precio <= 0:
                continue

            for pos in list(self.posiciones[symbol]):
                # Actualizar pico de desvío — nunca retroceder
                if dev > pos.dev_max_alcanzado:
                    pos.dev_max_alcanzado = dev
                motivo = self._evaluar_salida(pos, dev)
                if motivo:
                    op = self.cerrar_posicion(symbol, pos, precio, motivo)
                    cerradas.append(op)

            # Remover las posiciones cerradas
            ids_cerradas = {op.id for op in cerradas}
            self.posiciones[symbol] = [
                p for p in self.posiciones[symbol] if p.id not in ids_cerradas
            ]

        # 4. Abrir posiciones: desvío bajo AND clima BULL
        if self.puede_comprar(ahora):
            for symbol, ccl in ccl_map.items():
                if ccl_avg == 0:
                    continue
                dev    = (ccl / ccl_avg - 1) * 100
                clima  = climas.get(symbol, "🔴 BEAR")
                precio = precios_ars.get(symbol, 0)

                if dev < UMBRAL_COMPRA and clima == "🟢 BULL" and precio > 0:
                    pos = self.abrir_posicion(symbol, precio, ccl, dev, precios_ars, ahora)
                    if pos:
                        abiertas.append(pos)

        return {"abiertas": abiertas, "cerradas": cerradas, "forzadas": forzadas}

    # ──────────────────── RESUMEN ────────────────────────

    def resumen(self, precios_ars: Dict[str, float]) -> dict:
        cap_total = self.capital_total(precios_ars)
        pnl_total = cap_total - self.capital_inicial
        pnl_pct   = (pnl_total / self.capital_inicial) * 100

        ops_ganadoras = [o for o in self.operaciones if o.pnl > 0]
        win_rate      = (len(ops_ganadoras) / len(self.operaciones) * 100) if self.operaciones else 0

        return {
            "capital_inicial":       self.capital_inicial,
            "efectivo":              self.efectivo,
            "en_posiciones":         self.capital_en_posiciones(precios_ars),
            "capital_total":         cap_total,
            "pnl_total":             pnl_total,
            "pnl_pct":               pnl_pct,
            "operaciones_total":     len(self.operaciones),
            "operaciones_ganadoras": len(ops_ganadoras),
            "win_rate":              win_rate,
            "posiciones_abiertas":   sum(len(v) for v in self.posiciones.values()),
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

    def fila_sheets_estado(self, precios_ars: Dict[str, float]) -> list:
        r = self.resumen(precios_ars)
        return [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(r["capital_total"], 2),
            round(r["efectivo"], 2),
            round(r["en_posiciones"], 2),
            round(r["pnl_total"], 2),
            round(r["pnl_pct"], 4),
            r["operaciones_total"],
            round(r["win_rate"], 2),
            r["posiciones_abiertas"],
        ]
