"""
Simulador de Arbitraje CCL
==========================
- Capital inicial: ARS 10.000.000
- Por operación: 15% del capital total (efectivo + posiciones valoradas)
- Máximo 2 posiciones por especie
- Apertura: 11:30 hs Argentina (ambos mercados abiertos)
- Cierre nuevas compras: 16:30 hs
- Cierre forzado: 16:50 hs
- Cierre anticipado: señal contraria (desvío > +0.5%)
- Persistencia: Google Sheets
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# ─── HORARIOS ────────────────────────────────────────────
HORA_APERTURA      = time(11, 30)   # Ambos mercados abiertos
HORA_CIERRE_COMPRA = time(16, 30)   # No más compras
HORA_CIERRE_FORZADO = time(16, 50)  # Cierre todas las posiciones

# ─── PARÁMETROS ──────────────────────────────────────────
CAPITAL_INICIAL    = 10_000_000.0   # ARS
PCT_POR_OPERACION  = 0.15           # 15% del capital total
MAX_POSICIONES_POR_ESPECIE = 2


@dataclass
class Posicion:
    """Representa una posición abierta."""
    id:           str
    symbol:       str
    cantidad:     float        # unidades compradas
    precio_entry: float        # precio ARS de entrada
    monto_entry:  float        # ARS invertidos
    ts_entry:     str
    ccl_entry:    float        # CCL al momento de entrada
    dev_entry:    float        # desviación al momento de entrada
    precio_actual: float = 0.0
    pnl:          float = 0.0  # ganancia/pérdida en ARS


@dataclass
class Operacion:
    """Registro de una operación cerrada."""
    id:           str
    symbol:       str
    tipo:         str          # COMPRA / VENTA
    cantidad:     float
    precio_entry: float
    precio_exit:  float
    monto_entry:  float
    monto_exit:   float
    pnl:          float
    pnl_pct:      float
    ts_entry:     str
    ts_exit:      str
    motivo_cierre: str         # SEÑAL_CONTRARIA / CIERRE_FORZADO


class Simulador:
    """
    Motor de simulación de arbitraje CCL intradiario.
    """

    def __init__(self, capital_inicial: float = CAPITAL_INICIAL):
        self.capital_inicial   = capital_inicial
        self.efectivo          = capital_inicial
        self.posiciones:  Dict[str, List[Posicion]] = {}   # {symbol: [pos1, pos2]}
        self.operaciones: List[Operacion] = []
        self._op_counter = 0

    # ──────────────────── ESTADO ─────────────────────────

    def capital_en_posiciones(self, precios_ars: Dict[str, float]) -> float:
        """Valor actual de todas las posiciones abiertas."""
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
        """
        Abre una posición de compra si se cumplen las condiciones.
        Retorna la posición creada o None si no se pudo abrir.
        """
        # Validaciones
        if not self.puede_comprar(ahora):
            return None
        if self.posiciones_abiertas_count(symbol) >= MAX_POSICIONES_POR_ESPECIE:
            return None
        if precio_ars <= 0:
            return None

        monto = self.monto_por_operacion(precios_ars)
        if monto > self.efectivo:
            monto = self.efectivo  # usar lo disponible si no alcanza

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
        """Cierra una posición y registra la operación."""
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
        """Cierra todas las posiciones abiertas."""
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
        """
        Procesa un ciclo de señales:
        - Cierra posiciones con señal contraria
        - Abre posiciones con señal de compra y clima BULL
        - Cierre forzado si es >= 16:50
        Retorna resumen del ciclo.
        """
        abiertas  = []
        cerradas  = []
        forzadas  = []

        # 1. Cierre forzado
        if self.debe_cerrar_forzado(ahora):
            forzadas = self.cerrar_todas(precios_ars, "CIERRE_FORZADO")
            return {"abiertas": [], "cerradas": [], "forzadas": forzadas}

        # 2. Actualizar precios y PnL de posiciones abiertas
        for sym, poss in self.posiciones.items():
            precio = precios_ars.get(sym, 0)
            for pos in poss:
                pos.precio_actual = precio
                pos.pnl = (precio - pos.precio_entry) * pos.cantidad

        # 3. Cerrar posiciones con señal contraria (desvío > +0.5%)
        for symbol in list(self.posiciones.keys()):
            if not self.posiciones[symbol]:
                continue
            ccl = ccl_map.get(symbol, 0)
            if ccl_avg == 0:
                continue
            dev = (ccl / ccl_avg - 1) * 100
            precio = precios_ars.get(symbol, 0)

            if dev > 0.5 and precio > 0:  # señal contraria
                for pos in list(self.posiciones[symbol]):
                    op = self.cerrar_posicion(symbol, pos, precio, "SEÑAL_CONTRARIA")
                    cerradas.append(op)
                self.posiciones[symbol] = []

        # 4. Abrir posiciones con señal de compra
        if self.puede_comprar(ahora):
            for symbol, ccl in ccl_map.items():
                if ccl_avg == 0:
                    continue
                dev    = (ccl / ccl_avg - 1) * 100
                clima  = climas.get(symbol, "🔴 BEAR")
                precio = precios_ars.get(symbol, 0)

                # Señal: desvío < -0.5% y clima BULL
                if dev < -0.5 and clima == "🟢 BULL" and precio > 0:
                    pos = self.abrir_posicion(symbol, precio, ccl, dev, precios_ars, ahora)
                    if pos:
                        abiertas.append(pos)

        return {"abiertas": abiertas, "cerradas": cerradas, "forzadas": forzadas}

    # ──────────────────── RESUMEN ────────────────────────

    def resumen(self, precios_ars: Dict[str, float]) -> dict:
        cap_total = self.capital_total(precios_ars)
        pnl_total = cap_total - self.capital_inicial
        pnl_pct   = (pnl_total / self.capital_inicial) * 100

        ops_cerradas = [o for o in self.operaciones]
        ops_ganadoras = [o for o in ops_cerradas if o.pnl > 0]

        win_rate = (len(ops_ganadoras) / len(ops_cerradas) * 100) if ops_cerradas else 0

        return {
            "capital_inicial":   self.capital_inicial,
            "efectivo":          self.efectivo,
            "en_posiciones":     self.capital_en_posiciones(precios_ars),
            "capital_total":     cap_total,
            "pnl_total":         pnl_total,
            "pnl_pct":           pnl_pct,
            "operaciones_total": len(ops_cerradas),
            "operaciones_ganadoras": len(ops_ganadoras),
            "win_rate":          win_rate,
            "posiciones_abiertas": sum(len(v) for v in self.posiciones.values()),
        }

    def fila_sheets_operacion(self, op: Operacion) -> list:
        """Convierte una operación a fila para Google Sheets."""
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
        """Guarda snapshot del estado de la cartera."""
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
            
