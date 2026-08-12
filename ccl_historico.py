"""
ccl_historico.py — Serie histórica de CCL para dolarizar el backtest de Merval
================================================================================
Objetivo: separar "el sistema tiene timing real" de "el peso se devaluó".
Los CEDEARs (vía Alpaca) ya cotizan en USD y no necesitan este ajuste — el
sesgo de devaluación nominal solo afecta a los símbolos Merval (vía IOL,
en pesos).

Fuente: ArgentinaDatos (API pública, gratuita, sin auth) — serie histórica
de Contado con Liquidación. https://argentinadatos.com/

⚠️ El schema exacto del JSON de respuesta no pudo confirmarse contra un
request real desde este entorno de desarrollo (sin acceso de red al
dominio). El parseo es defensivo: prueba varios nombres de campo posibles
para fecha y valor de venta. La primera vez que se corra de verdad, si
`obtener_ccl_historico()` devuelve un dict vacío pero sin excepción, revisar
manualmente la respuesta cruda del endpoint (agregar un log del JSON crudo)
y ajustar _CAMPOS_FECHA / _CAMPOS_VENTA a los nombres reales.

Uso:
    ccl = obtener_ccl_historico("2023-01-01", "2024-08-01")
    bars_usd = dolarizar_bars(bars_ars, ccl)
"""

import logging
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

CCL_API_URL = "https://api.argentinadatos.com/v1/cotizaciones/dolares/contadoconliqui"

_CAMPOS_FECHA = ["fecha", "fechaHora", "date"]
_CAMPOS_VENTA = ["venta", "valorVenta", "valor_venta", "sell", "value_sell"]


def _extraer(d: dict, campos: list):
    for c in campos:
        if c in d and d[c] is not None:
            return d[c]
    return None


def obtener_ccl_historico(desde: str, hasta: str) -> Dict[str, float]:
    """
    Retorna dict {"YYYY-MM-DD": ccl_venta} para el rango [desde, hasta].

    Si un día no tiene dato (feriado/fin de semana, o mismatch entre el
    calendario bursátil de BYMA y el del mercado de bonos que arma el
    CCL), NO se rellena acá — el forward-fill se hace en dolarizar_bars()
    al momento de alinear con los bars de IOL, para no perder trazabilidad
    de qué días tienen dato real de CCL.
    """
    try:
        resp = requests.get(CCL_API_URL, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error(f"Error trayendo CCL histórico: {e}")
        return {}
    except ValueError as e:
        logger.error(f"Respuesta no-JSON de CCL histórico: {e}")
        return {}

    if isinstance(data, dict):
        # algunas APIs envuelven la lista en {"contadoconliqui": [...]} o {"data": [...]}
        data = data.get("contadoconliqui") or data.get("data") or []
    if not isinstance(data, list):
        logger.error(f"Shape inesperado en respuesta CCL: {type(data)}")
        return {}

    resultado = {}
    for d in data:
        if not isinstance(d, dict):
            continue
        fecha = _extraer(d, _CAMPOS_FECHA)
        venta = _extraer(d, _CAMPOS_VENTA)
        if fecha is None or venta is None:
            continue
        fecha_str = str(fecha)[:10]  # normaliza a YYYY-MM-DD
        try:
            resultado[fecha_str] = float(venta)
        except (TypeError, ValueError):
            continue

    filtrado = {f: v for f, v in resultado.items() if desde <= f <= hasta}
    logger.info(
        f"CCL histórico: {len(filtrado)} días en rango {desde}..{hasta} "
        f"(de {len(resultado)} días totales recibidos)"
    )
    if not filtrado and resultado:
        logger.warning(
            "CCL histórico: se parsearon días pero ninguno cae en el rango pedido "
            "— revisar formato de fecha devuelto por la API."
        )
    elif not resultado:
        logger.warning(
            "CCL histórico: 0 días parseados desde la respuesta — revisar "
            "_CAMPOS_FECHA/_CAMPOS_VENTA contra el JSON real del endpoint."
        )
    return filtrado


def dolarizar_bars(bars: list, ccl_dict: Dict[str, float]) -> list:
    """
    Deflacta OHLC (no volumen) de una lista de bars ARS por el CCL del
    mismo día. Usa forward-fill del último CCL conocido cuando un día
    puntual no tiene dato propio, para no descartar el bar entero por un
    hueco de 1 día en la fuente de CCL. Bars anteriores al primer dato de
    CCL disponible se descartan (sin referencia válida).

    No muta el input. Retorna nueva lista, mismo shape {t,o,h,l,c,v},
    ahora en unidades ARS/CCL ≈ USD equivalente.
    """
    if not ccl_dict:
        logger.warning("dolarizar_bars: ccl_dict vacío — bars devueltos SIN dolarizar")
        return bars

    fechas_ccl = sorted(ccl_dict.keys())
    ultimo_ccl: Optional[float] = None
    idx = 0
    resultado = []

    for b in bars:
        fecha_bar = b["t"][:10]
        while idx < len(fechas_ccl) and fechas_ccl[idx] <= fecha_bar:
            ultimo_ccl = ccl_dict[fechas_ccl[idx]]
            idx += 1

        if not ultimo_ccl:
            continue  # antes del primer dato de CCL — sin referencia, se descarta

        resultado.append({
            "t": b["t"],
            "o": b["o"] / ultimo_ccl,
            "h": b["h"] / ultimo_ccl,
            "l": b["l"] / ultimo_ccl,
            "c": b["c"] / ultimo_ccl,
            "v": b["v"],
        })

    return resultado
