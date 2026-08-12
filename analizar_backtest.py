"""
analizar_backtest.py — Análisis post-backtest, Sistema GG Swing
==========================================================================
Lee Backtest_Resultados (ya subida por run_backtest.py) y responde 2
preguntas que las métricas agregadas no muestran:

  1. Distribución de motivo_salida: ¿el sistema está saliendo por diseño
     (target1/target2/trailing_hma) o mayormente por max_hold_21d? Si es
     max_hold, en la práctica se está comportando como "comprar y aguantar
     21 días" más que como el sistema de salida escalonada diseñado.

  2. Buy & hold del mismo símbolo/período: ¿el sistema le gana a simplemente
     comprar al inicio del rango y no tocar nada? Relevante en particular
     para Merval, donde el drift nominal por inflación/devaluación puede
     inflar métricas sin que haya alpha real de timing.

No genera señales ni pisa nada — solo lee Sheets y (para buy&hold) vuelve
a pedir bars a IOL/Alpaca con el mismo rango de fechas ya usado.

Uso:
    python analizar_backtest.py --desde 2023-01-01 --hasta 2024-08-01
"""

import os
import sys
import json
import logging
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from iol_client      import IOLClient
from alpaca_client   import AlpacaClient
from sheets_manager   import SheetsManager
import signal_engine  as se

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
            "alpaca_key": os.environ.get("ALPACA_KEY_ID", ""),
            "alpaca_sec": os.environ.get("ALPACA_SECRET_KEY", ""),
            "gcp":        json.loads(os.environ["GCP_SERVICE_ACCOUNT"]),
        }
    except KeyError as e:
        logger.error(f"Variable de entorno faltante: {e}")
        sys.exit(1)


def leer_resultados(sh: SheetsManager) -> list:
    """Lee Backtest_Resultados cruda (headers + filas) vía gspread."""
    ws = sh._hojas.get("Backtest_Resultados")
    if not ws:
        return []
    valores = ws.get_all_values()
    if len(valores) < 2:
        return []
    headers = valores[0]
    filas = valores[1:]
    return [dict(zip(headers, fila)) for fila in filas]


def analizar_motivo_salida(filas: list):
    print("\n" + "=" * 60)
    print("DISTRIBUCIÓN DE MOTIVO DE SALIDA")
    print("=" * 60)

    grupos = {"Merval": defaultdict(int), "CEDEAR": defaultdict(int), "Total": defaultdict(int)}
    for f in filas:
        symbol = f.get("Symbol")
        motivo = f.get("Motivo Salida")
        grupo = "CEDEAR" if symbol in se.CEDEARS else "Merval"
        grupos[grupo][motivo] += 1
        grupos["Total"][motivo] += 1

    for grupo, conteos in grupos.items():
        total = sum(conteos.values())
        if total == 0:
            continue
        print(f"\n{grupo} ({total} trades):")
        for motivo, n in sorted(conteos.items(), key=lambda x: -x[1]):
            pct = 100 * n / total
            print(f"  {motivo:20s} {n:3d}  ({pct:5.1f}%)")


def buy_and_hold_pct(bars: list) -> float:
    """Retorno % simple entre el primer y último close del rango."""
    if len(bars) < 2:
        return float("nan")
    return 100 * (bars[-1]["c"] / bars[0]["c"] - 1)


def analizar_buy_and_hold(iol, alpaca, desde: str, hasta: str, symbols: list):
    print("\n" + "=" * 60)
    print(f"BUY & HOLD vs BACKTEST — {desde} a {hasta}")
    print("=" * 60)
    print(f"{'Symbol':8s} {'B&H %':>10s}")

    for symbol in symbols:
        es_cedear = symbol in se.CEDEARS
        try:
            if es_cedear:
                bars = alpaca.get_bars_diarias(symbol, desde=desde, hasta=hasta)
            else:
                bars = iol.get_historico_diario(symbol, desde, hasta)
        except Exception as e:
            logger.warning(f"  {symbol}: error trayendo bars ({e}) — saltado")
            continue

        bh = buy_and_hold_pct(bars)
        print(f"{symbol:8s} {bh:9.1f}%")

    print("\nNota: B&H está en % de precio simple; el backtest está en")
    print("múltiplos de R (riesgo unitario), no son directamente comparables")
    print("en magnitud — lo relevante es la DIRECCIÓN: si B&H ya captura la")
    print("mayoría del R total del backtest, el sistema no está agregando")
    print("mucho timing por encima del drift del activo.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desde", required=True, help="Fecha inicio YYYY-MM-DD (mismo rango que run_backtest.py)")
    parser.add_argument("--hasta", required=True, help="Fecha fin YYYY-MM-DD (mismo rango que run_backtest.py)")
    args = parser.parse_args()

    s = get_secrets()

    iol = IOLClient(s["iol_user"], s["iol_pass"])
    iol.login()

    alpaca = AlpacaClient(s["alpaca_key"], s["alpaca_sec"]) if s["alpaca_key"] else None

    sh = SheetsManager(s["gcp"])
    sh.conectar()

    filas = leer_resultados(sh)
    if not filas:
        logger.error("Backtest_Resultados está vacía — correr run_backtest.py primero.")
        sys.exit(1)

    analizar_motivo_salida(filas)

    symbols_presentes = sorted(set(f.get("Symbol") for f in filas))
    if alpaca:
        analizar_buy_and_hold(iol, alpaca, args.desde, args.hasta, symbols_presentes)
    else:
        logger.warning("ALPACA_KEY_ID no seteado — se salta buy&hold de CEDEARs (Merval igual corre).")
        analizar_buy_and_hold(iol, alpaca, args.desde, args.hasta,
                              [s for s in symbols_presentes if s in se.MERVAL])


if __name__ == "__main__":
    main()
