"""
alertas.py — Sistema GG Swing
==============================
Módulo central que conecta trader_job.py con signal_engine.py y
simulator.py. Responsabilidades:

  1. Construir la serie de velas diarias que signal_engine.py necesita
     para cada símbolo, combinando:
       - Historico_Diario_Cache (días ya cerrados, refrescado 1 vez/día)
       - la vela de "hoy" (mercado en curso), armada SIN requests extra:
           Merval  -> resample de Historico_Merval_Raw (ya se acumula
                      cada ciclo vía get_panel(), 1 request para 8 símbolos)
           CEDEARs -> alpaca.get_daily_bar_hoy() (reutiliza el snapshot
                      que ya se pide para el precio actual)
  2. Refrescar Historico_Diario_Cache una vez por día (chequea la fecha
     más reciente cacheada antes de volver a pedirle historia a
     IOL/Alpaca — evita el problema de cuota que tendría pedir histórico
     completo por símbolo en cada ciclo de 5 min).
  3. Correr signal_engine.generar_senal() por símbolo, abrir posiciones
     nuevas vía simulator.abrir_posicion(), evaluar salidas vía
     simulator.procesar_ciclo(), y devolver los eventos (entradas,
     parciales T1/T2, cierres) para que trader_job.py dispare los
     emails correspondientes — SOLO en esos eventos, nunca de más.

DÍAS DE HISTORIA PARA EL CACHE: se pide DIAS_HISTORIA_CACHE (default 200)
días corridos hacia atrás, suficiente margen sobre MIN_VELAS_REQUERIDAS=60
(hábiles) para cubrir feriados/fines de semana sin quedarse corto.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from zoneinfo import ZoneInfo

import signal_engine as se
import simulator as sim

logger = logging.getLogger(__name__)

TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")
DIAS_HISTORIA_CACHE = 200


def _hoy_str() -> str:
    return datetime.now(TZ_ARG).strftime("%Y-%m-%d")


def _refrescar_cache_si_hace_falta(symbol: str, es_cedear: bool, iol, alpaca, sheets) -> None:
    """
    Refresca Historico_Diario_Cache para un símbolo SOLO si la fecha más
    reciente cacheada no es de ayer o antes de hoy (es decir, si ya se
    refrescó hoy, no vuelve a pedir nada). No incluye la vela de hoy —
    esa se arma aparte, por fuera de este cache.
    """
    fecha_cache = sheets.fecha_mas_reciente_cache(symbol)
    hoy = _hoy_str()
    if fecha_cache == hoy:
        return  # ya refrescado hoy, no hay que hacer nada

    desde = (datetime.now(TZ_ARG) - timedelta(days=DIAS_HISTORIA_CACHE)).strftime("%Y-%m-%d")
    ayer = (datetime.now(TZ_ARG) - timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        if es_cedear:
            bars = alpaca.get_bars_diarias(symbol, desde=desde, hasta=ayer)
        else:
            bars = iol.get_historico_diario(symbol, desde, ayer)
    except Exception as e:
        logger.warning(f"Cache {symbol}: error refrescando histórico ({e}) — se sigue con lo que haya en cache")
        return

    if bars:
        sheets.guardar_historico_diario_cache(symbol, bars)
    else:
        logger.warning(f"Cache {symbol}: refresco devolvió 0 velas — se sigue con lo que haya en cache")


def _vela_hoy_merval(symbol: str, sheets) -> Optional[dict]:
    """
    Arma la vela de hoy para un símbolo Merval a partir de los ticks ya
    acumulados en Historico_Merval_Raw en este mismo proceso/día — sin
    request adicional a IOL.
    """
    ticks = sheets.cargar_historico_merval_raw(symbol)
    hoy = _hoy_str()
    ticks_hoy = [t for t in ticks if t["ts"][:10] == hoy]
    if not ticks_hoy:
        return None

    ticks_hoy.sort(key=lambda t: t["ts"])
    return {
        "t": hoy,
        "o": ticks_hoy[0]["apertura"] or ticks_hoy[0]["precio"],
        "h": max(t["maximo"] or t["precio"] for t in ticks_hoy),
        "l": min(t["minimo"] or t["precio"] for t in ticks_hoy if (t["minimo"] or t["precio"]) > 0),
        "c": ticks_hoy[-1]["precio"],
        "v": ticks_hoy[-1]["volumen_nominal"],  # último snapshot = volumen acumulado del día
    }


def construir_bars(symbol: str, es_cedear: bool, iol, alpaca, sheets) -> List[dict]:
    """
    Devuelve la lista de velas diarias lista para signal_engine.generar_senal,
    combinando cache (días cerrados) + vela de hoy (si ya hay datos del
    día en curso). Refresca el cache primero si hace falta.
    """
    _refrescar_cache_si_hace_falta(symbol, es_cedear, iol, alpaca, sheets)
    bars = sheets.cargar_historico_diario_cache(symbol)

    hoy = _hoy_str()
    if bars and bars[-1]["t"][:10] == hoy:
        # el cache ya incluye hoy (no debería pasar con la lógica actual,
        # pero por las dudas no duplicamos)
        return bars

    if es_cedear:
        vela_hoy = alpaca.get_daily_bar_hoy([symbol]).get(symbol)
    else:
        vela_hoy = _vela_hoy_merval(symbol, sheets)

    if vela_hoy:
        bars = bars + [vela_hoy]

    return bars


def procesar_alertas(
    iol,
    alpaca,
    sheets,
    simulador: sim.Simulador,
    ahora=None,
) -> dict:
    """
    Corre 1 ciclo completo de señales + gestión de posiciones sobre todo
    el universo (8 Merval + 11 CEDEARs). Devuelve un dict con los eventos
    ocurridos EN ESTE CICLO, para que trader_job.py dispare únicamente
    los emails correspondientes a compra/venta real:

        {
            "entradas": [Posicion, ...],
            "parciales": [Operacion, ...],   # TARGET_1 / TARGET_2
            "cierres":   [Operacion, ...],   # STOP_LOSS / TRAILING_STOP / MAX_HOLD_21D / CIERRE_FORZADO
        }
    """
    universo = se.MERVAL + se.CEDEARS
    precios: Dict[str, float] = {}
    trailing_stops: Dict[str, float] = {}
    entradas: List[sim.Posicion] = []
    señales_candidatas: List[tuple] = []  # (senal, es_cedear) — se abren ordenadas por score

    for symbol in universo:
        es_cedear = symbol in se.CEDEARS

        try:
            bars = construir_bars(symbol, es_cedear, iol, alpaca, sheets)
        except Exception as e:
            logger.warning(f"{symbol}: error construyendo velas ({e}) — saltado este ciclo")
            continue

        if len(bars) < se.MIN_VELAS_REQUERIDAS:
            logger.warning(f"{symbol}: solo {len(bars)} velas (mín {se.MIN_VELAS_REQUERIDAS}) — saltado")
            continue

        precio_actual = bars[-1]["c"]
        if precio_actual <= 0:
            continue
        precios[symbol] = precio_actual

        highs = [b["h"] for b in bars]
        lows = [b["l"] for b in bars]
        closes = [b["c"] for b in bars]
        volumes = [b["v"] for b in bars]

        if simulador.tiene_posicion(symbol):
            # Posición abierta: calcular trailing (HMA rápida del régimen
            # con el que se abrió) para el remanente post-T2. Se calcula
            # siempre que haya posición, aunque solo se usa si t2_alcanzado
            # — es barato (numpy sobre <=260 velas) y así procesar_ciclo()
            # ya lo tiene disponible sin lógica condicional acá.
            pos = simulador.posiciones[symbol]
            params = se.REGIMENES.get(pos.regimen)
            if params:
                hma_rap = se.hma(se.np.array(closes), params["hma_rapida"])
                if len(hma_rap) and not se.np.isnan(hma_rap[-1]):
                    trailing_stops[symbol] = float(hma_rap[-1])
            continue  # la apertura de nuevas posiciones no aplica a símbolos ya en cartera

        # Sin posición: evaluar señal de entrada. No se abre acá todavía —
        # se junta con las demás señales del ciclo para poder priorizar
        # por score cuando el capital disponible no alcanza para todas
        # (ver discusión 19-ago-2026: prioridad por score, no orden fijo).
        senal = se.generar_senal(symbol, highs, lows, closes, volumes, es_cedear, modo_backtest=False)
        if senal.senal_valida:
            señales_candidatas.append(senal)

    # Abrir posiciones en orden de score descendente — si el capital no
    # alcanza para todas, las de mejor score se sirven primero.
    señales_candidatas.sort(key=lambda s: s.score, reverse=True)
    for senal in señales_candidatas:
        pos = simulador.abrir_posicion(
            symbol=senal.symbol,
            score=senal.score,
            regimen=senal.regimen,
            precio_entry=senal.entry,
            precio_stop=senal.stop,
            precio_t1=senal.t1,
            precio_t2=senal.t2,
            precio_t3=senal.t2,  # T3 es el trailing en sí, no un precio fijo — se usa T2 como referencia informativa
            precios=precios,
            ahora=ahora,
        )
        if pos:
            entradas.append(pos)

    resultado_ciclo = simulador.procesar_ciclo(precios, trailing_stops, ahora=ahora)

    return {
        "entradas": entradas,
        "parciales": resultado_ciclo["parciales"],
        "cierres": resultado_ciclo["cerradas"] + resultado_ciclo["forzadas"],
  }
  
