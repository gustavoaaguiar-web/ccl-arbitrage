"""
run_backtest.py — Backtest walk-forward Ruta A (diario), Sistema GG Swing
==========================================================================
Corre signal_engine.py sobre historia diaria completa del universo (8
Merval vía IOL, 11 CEDEARs vía Alpaca), simula cada trade con
backtest_engine.py, y sube los resultados a Google Sheets
(Backtest_Resultados / Backtest_Metricas) vía sheets.limpiar_y_escribir().

Flag --dolarizar:
    Deflacta los bars de Merval por el CCL del día (ver ccl_historico.py)
    antes de generar señales y simular trades, para separar "el sistema
    tiene timing real" de "el peso se devaluó" — HMA50 con pendiente
    positiva es casi trivial en pesos nominales durante un período
    inflacionario. Los CEDEARs (vía Alpaca) ya cotizan en USD y no se
    tocan con este flag.
    Cuando está activo, los resultados van a pestañas separadas
    (Backtest_Resultados_USD / Backtest_Metricas_USD) para no pisar los
    resultados nominales en pesos y poder comparar ambos lado a lado.

Uso:
    python run_backtest.py --desde 2023-01-01 --hasta 2024-08-01
    python run_backtest.py --solo GGAL,MELI
    python run_backtest.py --solo GGAL,YPFD,PAMP,BMA,CEPU,TGSU2,SUPV,BBAR --dolarizar

Credenciales via variables de entorno:
    IOL_USER, IOL_PASS, GCP_SERVICE_ACCOUNT
    ALPACA_KEY_ID, ALPACA_SECRET_KEY
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from iol_client      import IOLClient
from alpaca_client   import AlpacaClient
from sheets_manager  import SheetsManager, HEADERS
import signal_engine  as se
import backtest_engine as be
import ccl_historico as cclh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_secrets():
    try:
        return {
            "iol_user":   os.environ["IOL_USER"],
            "iol_pass":   os.environ["IOL_PASS"],
            "alpaca_key": os.environ["ALPACA_KEY_ID"],
            "alpaca_sec": os.environ["ALPACA_SECRET_KEY"],
            "gcp":        json.loads(os.environ["GCP_SERVICE_ACCOUNT"]),
        }
    except KeyError as e:
        logger.error(f"Variable de entorno faltante: {e}")
        sys.exit(1)


def fetch_bars_merval(iol: IOLClient, symbol: str, desde: str, hasta: str) -> list:
    """Histórico diario Merval vía IOL (seriehistorica) — ver iol_client.get_historico_diario."""
    return iol.get_historico_diario(symbol, desde, hasta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desde", default=(datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"),
                        help="Fecha inicio YYYY-MM-DD (default: 2 años atrás)")
    parser.add_argument("--hasta", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Fecha fin YYYY-MM-DD (default: hoy)")
    parser.add_argument("--solo", default=None,
                        help="Lista de símbolos separada por coma, para pruebas rápidas")
    parser.add_argument("--dolarizar", action="store_true",
                        help="Deflacta Merval por CCL antes de backtestear (ver ccl_historico.py). "
                             "No afecta CEDEARs, que ya cotizan en USD vía Alpaca.")
    args = parser.parse_args()

    logger.info(f"🚀 Backtest Ruta A — {args.desde} a {args.hasta}" + (" [DOLARIZADO vía CCL]" if args.dolarizar else ""))

    s = get_secrets()

    iol = IOLClient(s["iol_user"], s["iol_pass"])
    iol.login()

    alpaca = AlpacaClient(s["alpaca_key"], s["alpaca_sec"])

    sh = SheetsManager(s["gcp"])
    sh.conectar()

    universo = [x.strip() for x in args.solo.split(",")] if args.solo else (se.MERVAL + se.CEDEARS)

    ccl_dict = {}
    if args.dolarizar:
        ccl_dict = cclh.obtener_ccl_historico(args.desde, args.hasta)
        if not ccl_dict:
            logger.error(
                "⚠️ --dolarizar activo pero no se pudo traer CCL histórico. "
                "Los símbolos Merval se van a saltar para no correr un backtest "
                "silenciosamente en pesos nominales creyendo que está dolarizado."
            )

    sheet_resultados = "Backtest_Resultados_USD" if args.dolarizar else "Backtest_Resultados"
    sheet_metricas = "Backtest_Metricas_USD" if args.dolarizar else "Backtest_Metricas"

    filas_resultados = [HEADERS[sheet_resultados]]
    filas_metricas = [HEADERS[sheet_metricas]]

    for symbol in universo:
        es_cedear = symbol in se.CEDEARS
        etiqueta = "CEDEAR/Alpaca" if es_cedear else "Merval/IOL"
        if args.dolarizar and not es_cedear:
            etiqueta += ", dolarizado CCL"
        logger.info(f"── {symbol} ({etiqueta}) ──")

        try:
            if es_cedear:
                bars = alpaca.get_bars_diarias(symbol, desde=args.desde, hasta=args.hasta)
            else:
                bars = fetch_bars_merval(iol, symbol, args.desde, args.hasta)
                if args.dolarizar:
                    if not ccl_dict:
                        logger.warning(f"  {symbol}: saltado — sin CCL histórico disponible")
                        continue
                    bars_antes = len(bars)
                    bars = cclh.dolarizar_bars(bars, ccl_dict)
                    logger.info(f"  {symbol}: dolarizado {bars_antes} → {len(bars)} velas (descarta días previos al primer CCL)")
        except Exception as e:
            logger.error(f"  Error trayendo datos de {symbol}: {e}")
            continue

        if len(bars) < se.MIN_VELAS_REQUERIDAS:
            logger.warning(f"  {symbol}: solo {len(bars)} velas (mín {se.MIN_VELAS_REQUERIDAS}) — saltado")
            continue

        trades = be.backtest_symbol(symbol, bars, es_cedear)
        logger.info(f"  {symbol}: {len(trades)} trades generados sobre {len(bars)} velas")

        for t in trades:
            filas_resultados.append(t.to_row())
        filas_metricas.append(be.calcular_metricas(symbol, trades))

    if len(filas_resultados) > 1:
        sh.limpiar_y_escribir(sheet_resultados, filas_resultados)
        logger.info(f"✅ {sheet_resultados}: {len(filas_resultados) - 1} trades subidos")
    else:
        logger.warning(f"⚠️ Sin trades en ningún símbolo — no se sube {sheet_resultados}")

    if len(filas_metricas) > 1:
        sh.limpiar_y_escribir(sheet_metricas, filas_metricas)
        logger.info(f"✅ {sheet_metricas}: {len(filas_metricas) - 1} símbolos subidos")

    logger.info("✅ Backtest finalizado")


if __name__ == "__main__":
    main()
