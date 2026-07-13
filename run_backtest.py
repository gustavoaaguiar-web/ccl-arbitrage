"""
run_backtest.py — Backtest walk-forward Ruta A (diario), Sistema GG Swing
==========================================================================
Corre signal_engine.py sobre historia diaria completa de los 20 activos del
universo (8 Merval vía IOL, 12 CEDEARs vía Alpaca), simula cada trade con
backtest_engine.py, y sube los resultados a Google Sheets
(Backtest_Resultados / Backtest_Metricas) vía sheets.limpiar_y_escribir().

⚠️ BLOQUEANTE PARA MERVAL: iol_client.py todavía no tiene un método de
   histórico diario (get_historico_diario o similar) — ver el docstring
   de fetch_bars_merval() más abajo con el shape esperado y un endpoint
   sugerido a partir de lo que ya testeaste en Pydroid (seriehistorica).
   Los CEDEARs vía Alpaca SÍ están listos para correr ya
   (alpaca_client.get_bars_diarias ya existe).

Uso:
    python run_backtest.py --desde 2024-01-01 --hasta 2026-07-01
    python run_backtest.py --solo GGAL,MELI      # subset rápido para pruebas
    python run_backtest.py --solo MELI,NVDA,TSLA,MSFT,PLTR,VIST,MU,AMZN,IBIT,META,AAPL,VALO
                                                   # correr solo CEDEARs (ya funcional)

Credenciales via variables de entorno:
    IOL_USER, IOL_PASS, GCP_SERVICE_ACCOUNT   (igual que trader_job.py)
    ALPACA_KEY_ID, ALPACA_SECRET_KEY          (nuevas — no las usa trader_job.py)
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
    """
    ⚠️ get_historico_diario() no existe todavía en iol_client.py — este es
    el único punto del pipeline que depende de agregarlo.

    Shape esperado de retorno (igual al de alpaca_client.get_bars_diarias):
        [{"t": "2024-01-02T00:00:00Z", "o": .., "h": .., "l": .., "c": .., "v": ..}, ...]
        ordenado ascendente por fecha.

    Método sugerido para agregar a IOLClient, a partir del endpoint
    'seriehistorica' que ya testeaste en Pydroid contra la API real:

        def get_historico_diario(self, symbol, desde, hasta, mercado="bCBA"):
            self._ensure_token()
            resp = self.session.get(
                f"{IOL_BASE_URL}/api/v2/{mercado}/Titulos/{symbol}"
                f"/Cotizacion/seriehistorica/{desde}/{hasta}/sinAjustar",
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "t": d.get("fechaHora", ""),
                    "o": float(d.get("apertura", 0)),
                    "h": float(d.get("maximo", 0)),
                    "l": float(d.get("minimo", 0)),
                    "c": float(d.get("ultimoPrecio", 0)),
                    "v": float(d.get("volumen", 0)),
                }
                for d in data
            ]

    ⚠️ El nombre de los campos del JSON de respuesta (fechaHora, apertura,
    etc.) es una suposición basada en el resto de iol_client.py — hay que
    confirmarlo contra un request real antes de confiar en el resultado.
    """
    if not hasattr(iol, "get_historico_diario"):
        raise NotImplementedError(
            "IOLClient no tiene get_historico_diario(). Agregalo a iol_client.py "
            "antes de correr el backtest sobre símbolos Merval — ver el docstring "
            "de fetch_bars_merval() en run_backtest.py para el shape esperado y "
            "un endpoint sugerido."
        )
    return iol.get_historico_diario(symbol, desde, hasta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desde", default=(datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"),
                        help="Fecha inicio YYYY-MM-DD (default: 2 años atrás)")
    parser.add_argument("--hasta", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Fecha fin YYYY-MM-DD (default: hoy)")
    parser.add_argument("--solo", default=None,
                        help="Lista de símbolos separada por coma, para pruebas rápidas")
    args = parser.parse_args()

    logger.info(f"🚀 Backtest Ruta A — {args.desde} a {args.hasta}")

    s = get_secrets()

    iol = IOLClient(s["iol_user"], s["iol_pass"])
    iol.login()

    alpaca = AlpacaClient(s["alpaca_key"], s["alpaca_sec"])

    sh = SheetsManager(s["gcp"])
    sh.conectar()

    universo = [x.strip() for x in args.solo.split(",")] if args.solo else (se.MERVAL + se.CEDEARS)

    filas_resultados = [HEADERS["Backtest_Resultados"]]
    filas_metricas = [HEADERS["Backtest_Metricas"]]

    for symbol in universo:
        es_cedear = symbol in se.CEDEARS
        logger.info(f"── {symbol} ({'CEDEAR/Alpaca' if es_cedear else 'Merval/IOL'}) ──")

        try:
            if es_cedear:
                bars = alpaca.get_bars_diarias(symbol, desde=args.desde, hasta=args.hasta)
            else:
                bars = fetch_bars_merval(iol, symbol, args.desde, args.hasta)
        except NotImplementedError as e:
            logger.error(f"  {e}")
            continue
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
        sh.limpiar_y_escribir("Backtest_Resultados", filas_resultados)
        logger.info(f"✅ Backtest_Resultados: {len(filas_resultados) - 1} trades subidos")
    else:
        logger.warning("⚠️ Sin trades en ningún símbolo — no se sube Backtest_Resultados")

    if len(filas_metricas) > 1:
        sh.limpiar_y_escribir("Backtest_Metricas", filas_metricas)
        logger.info(f"✅ Backtest_Metricas: {len(filas_metricas) - 1} símbolos subidos")

    logger.info("✅ Backtest finalizado")


if __name__ == "__main__":
    main()
      
