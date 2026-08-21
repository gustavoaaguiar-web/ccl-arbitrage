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
     IOL/Alpaca).
  3. Correr signal_engine.generar_senal() por símbolo, abrir posiciones
     nuevas vía simulator.abrir_posicion() (priorizando por score si
     hay varias señales el mismo ciclo y el capital no alcanza para
     todas), evaluar salidas vía simulator.procesar_ciclo(), y devolver
     los eventos (entradas, parciales T1/T2, cierres) para que
     trader_job.py dispare los emails correspondientes.

FIX (21-ago-2026) — 429 Quota exceeded de Sheets API:
La versión anterior leía Historico_Diario_Cache y Historico_Merval_Raw
UNA VEZ POR SÍMBOLO (hasta 2-3 lecturas completas × 19 símbolos = 40-70+
requests de lectura en menos de 1 minuto), superando ampliamente el
límite gratuito de Sheets API. Se rediseñó para leer cada hoja UNA SOLA
VEZ POR CICLO (2 lecturas totales: cache completo + raw ticks completos),
trabajar en memoria, y escribir el cache actualizado en 1 sola escritura
al final del ciclo (si hubo símbolos refrescados). También se batchea el
pedido de "vela de hoy" de CEDEARs a Alpaca: antes era 1 request por
símbolo (11 requests), ahora es 1 solo request para los 11 juntos.

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


def _refrescar_cache_en_memoria(
    cache_completo: Dict[str, List[dict]], iol, alpaca
) -> bool:
    """
    Recorre el universo y refresca EN MEMORIA (dict cache_completo, ya
    cargado con 1 sola lectura previa) los símbolos cuya fecha más
    reciente no sea de hoy. No toca Sheets acá — eso se hace 1 sola vez
    al final, en guardar_historico_diario_cache_batch(), solo si hizo
    falta refrescar algo.

    Retorna True si se modificó algo (para saber si hay que escribir).
    """
    hoy = _hoy_str()
    desde = (datetime.now(TZ_ARG) - timedelta(days=DIAS_HISTORIA_CACHE)).strftime("%Y-%m-%d")
    ayer = (datetime.now(TZ_ARG) - timedelta(days=1)).strftime("%Y-%m-%d")

    hubo_cambios = False

    for symbol in se.MERVAL + se.CEDEARS:
        es_cedear = symbol in se.CEDEARS
        bars_actuales = cache_completo.get(symbol, [])
        fecha_reciente = bars_actuales[-1]["t"][:10] if bars_actuales else None

        if fecha_reciente == hoy:
            continue  # ya refrescado hoy

        try:
            if es_cedear:
                bars = alpaca.get_bars_diarias(symbol, desde=desde, hasta=ayer)
            else:
                bars = iol.get_historico_diario(symbol, desde, ayer)
        except Exception as e:
            logger.warning(f"Cache {symbol}: error refrescando histórico ({e}) — se sigue con lo que haya en cache")
            continue

        if bars:
            cache_completo[symbol] = bars
            hubo_cambios = True
        else:
            logger.warning(f"Cache {symbol}: refresco devolvió 0 velas — se sigue con lo que haya en cache")

    return hubo_cambios


def _velas_hoy_merval(raw_ticks_todos: List[dict]) -> Dict[str, dict]:
    """
    Arma la vela de hoy para TODOS los símbolos Merval a partir de los
    ticks ya cargados en memoria (1 sola lectura previa de
    Historico_Merval_Raw completo) — sin requests adicionales.
    """
    hoy = _hoy_str()
    ticks_hoy_por_symbol: Dict[str, List[dict]] = {}
    for t in raw_ticks_todos:
        if t["ts"][:10] != hoy:
            continue
        ticks_hoy_por_symbol.setdefault(t["symbol"], []).append(t)

    resultado = {}
    for symbol, ticks in ticks_hoy_por_symbol.items():
        ticks.sort(key=lambda t: t["ts"])
        minimos_validos = [t["minimo"] or t["precio"] for t in ticks if (t["minimo"] or t["precio"]) > 0]
        resultado[symbol] = {
            "t": hoy,
            "o": ticks[0]["apertura"] or ticks[0]["precio"],
            "h": max(t["maximo"] or t["precio"] for t in ticks),
            "l": min(minimos_validos) if minimos_validos else ticks[-1]["precio"],
            "c": ticks[-1]["precio"],
            "v": ticks[-1]["volumen_nominal"],
        }
    return resultado


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
    # ── Lecturas por lote — 1 vez por ciclo, no por símbolo ──────────
    cache_completo = sheets.cargar_historico_diario_cache_completo()
    raw_ticks_todos = sheets.cargar_historico_merval_raw()

    hubo_cambios = _refrescar_cache_en_memoria(cache_completo, iol, alpaca)
    if hubo_cambios:
        sheets.guardar_historico_diario_cache_batch(cache_completo)

    velas_hoy_merval = _velas_hoy_merval(raw_ticks_todos)
    velas_hoy_cedear = alpaca.get_daily_bar_hoy(se.CEDEARS)  # 1 solo request para los 11

    # ── Procesamiento por símbolo (todo en memoria, sin más I/O) ─────
    precios: Dict[str, float] = {}
    trailing_stops: Dict[str, float] = {}
    entradas: List[sim.Posicion] = []
    señales_candidatas: List = []

    for symbol in se.MERVAL + se.CEDEARS:
        es_cedear = symbol in se.CEDEARS

        bars = list(cache_completo.get(symbol, []))
        vela_hoy = velas_hoy_cedear.get(symbol) if es_cedear else velas_hoy_merval.get(symbol)
        if vela_hoy and (not bars or bars[-1]["t"][:10] != vela_hoy["t"][:10]):
            bars.append(vela_hoy)

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
            pos = simulador.posiciones[symbol]
            params = se.REGIMENES.get(pos.regimen)
            if params:
                hma_rap = se.hma(se.np.array(closes), params["hma_rapida"])
                if len(hma_rap) and not se.np.isnan(hma_rap[-1]):
                    trailing_stops[symbol] = float(hma_rap[-1])
            continue

        senal = se.generar_senal(symbol, highs, lows, closes, volumes, es_cedear, modo_backtest=False)
        if senal.senal_valida:
            señales_candidatas.append(senal)

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
            precio_t3=senal.t2,
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
  
