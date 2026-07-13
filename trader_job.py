"""
GG Swing Trader — Trader Job Standalone
=========================================
Script autónomo para correr desde GitHub Actions (sin Streamlit).
Ejecuta N ciclos de 60s y persiste estado en Google Sheets.

Único operador del sistema — app.py es solo dashboard de lectura.

NOTA DE TRANSICIÓN (jun-2026):
Este archivo reemplaza la versión anterior basada en arbitraje CCL,
descontinuada por compresión estructural de spreads (compras de USD
del BCRA). El sistema ahora construye hacia el Sistema GG Swing
(HMA-D + SMI multi-timeframe), todavía en fase de diseño del
signal_engine.py.

Por ahora este job cumple una única función operativa:
  → Acumular histórico crudo de Merval (Historico_Merval_Raw) para
    poder resamplear a velas 30min/4H una vez haya suficiente data.

Cuando signal_engine.py esté listo, este archivo se extiende para
generar señales y alertas reales — hoy solo recolecta datos.

Uso:
    python trader_job.py            # 5 ciclos (default)
    python trader_job.py --ciclos 3 # N ciclos custom

Credenciales via variables de entorno (GitHub Secrets):
    IOL_USER, IOL_PASS
    GCP_SERVICE_ACCOUNT  (JSON string)
    GMAIL_USER, GMAIL_APP_PASS   (no usado activamente todavía,
                                   se mantiene para cuando haya alertas)
"""

import os
import sys
import time
import json
import logging
import smtplib
import argparse

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from iol_client     import IOLClient
from sheets_manager import SheetsManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")

# ── CONSTANTES ────────────────────────────────────────────
REFRESH_SECONDS = 60
WARMUP_CICLOS   = 1   # ciclos sin operar al arranque, por consistencia con runs futuras

HORA_APERTURA = dtime(10, 30)
HORA_CIERRE   = dtime(17, 0)

# Universo Merval del Sistema GG Swing (8 activos que cotizan en BYMA/CNV
# vía IOL). VALO se sacó de este set el 12/jul/2026: cotiza en EEUU (como
# MELI), no en el panel Merval — por eso get_panel("MerVal") nunca lo
# devolvía y el log mostraba sistemáticamente "8/9 símbolos guardados".
MERVAL_SWING_SET = {
    "GGAL", "YPFD", "PAMP", "CEPU", "BMA",
    "TGSU2", "SUPV", "BBAR",
}

# Universo CEDEARs del Sistema GG Swing (12 activos — datos vía Alpaca,
# no se gestionan en este job todavía; se deja documentado para referencia
# cuando se integre alpaca_client.py renovado).
# VALO sumado acá el 12/jul/2026 (cotiza en EEUU, no en Merval).
CEDEAR_SWING_SET = {
    "MELI", "NVDA", "TSLA", "MSFT", "PLTR",
    "VIST", "MU", "AMZN", "IBIT", "META", "AAPL", "VALO",
}


def hora_argentina():
    return datetime.now(TZ_ARG)


# ── CREDENCIALES ──────────────────────────────────────────
def get_secrets():
    try:
        return {
            "iol_user":   os.environ["IOL_USER"],
            "iol_pass":   os.environ["IOL_PASS"],
            "gmail_user": os.environ.get("GMAIL_USER", ""),
            "gmail_pass": os.environ.get("GMAIL_APP_PASS", ""),
            "gcp":        json.loads(os.environ["GCP_SERVICE_ACCOUNT"]),
        }
    except KeyError as e:
        logger.error(f"Variable de entorno faltante: {e}")
        sys.exit(1)


# ── GMAIL ─────────────────────────────────────────────────
def enviar_mail(gmail_user, gmail_pass, subject, cuerpo):
    """Genérico — reutilizado tal cual para futuras alertas del Sistema GG Swing."""
    if not gmail_user:
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = msg["To"] = gmail_user
        msg["Subject"] = subject
        msg.attach(MIMEText(cuerpo, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_user, gmail_pass)
            smtp.send_message(msg)
        logger.info(f"📧 Mail enviado: {subject}")
    except Exception as e:
        logger.warning(f"Gmail error: {e}")


# ── PRECIOS / HISTÓRICO MERVAL ────────────────────────────
def fetch_detalle_merval(iol):
    """
    Trae detalle OHLC+volumen de los símbolos Merval del universo GG Swing,
    usando get_panel('MerVal') — 1 solo request, sin costo adicional de cuota
    (confirmado: el batch ya expone apertura/maximo/minimo/volumen).

    Retorna lista de dicts listos para sheets.guardar_tick_merval().
    """
    try:
        titulos = iol.get_panel("MerVal")
    except Exception as e:
        logger.warning(f"fetch_detalle_merval: error en get_panel: {e}")
        return []

    snapshots = []
    for t in titulos:
        sym = t.get("simbolo")
        if sym not in MERVAL_SWING_SET:
            continue
        if not t.get("ultimoPrecio"):
            continue
        snapshots.append({
            "symbol":               sym,
            "precio":               t.get("ultimoPrecio", 0),
            "apertura":             t.get("apertura", 0),
            "maximo":               t.get("maximo", 0),
            "minimo":               t.get("minimo", 0),
            "volumen_nominal":      t.get("volumen", 0),
            "cantidad_operaciones": int(t.get("cantidadOperaciones", 0) or 0),
        })
    return snapshots


# ── CICLO PRINCIPAL ───────────────────────────────────────
def ejecutar_ciclo(iol, sheets, n_ciclo):
    hora  = hora_argentina()
    ahora = hora.time()

    logger.info(f"─── Ciclo {n_ciclo} | {hora.strftime('%H:%M:%S')} ART ───")

    if not (HORA_APERTURA <= ahora <= HORA_CIERRE):
        logger.info("Fuera de horario de mercado — sin captura de datos")
        return

    # Acumulación histórico Merval para Sistema GG Swing
    # (única función operativa de este job mientras signal_engine.py
    # está en desarrollo)
    detalle_merval = fetch_detalle_merval(iol)
    if detalle_merval:
        sheets.guardar_tick_merval(detalle_merval)
        logger.info(f"  📊 Histórico Merval: {len(detalle_merval)}/{len(MERVAL_SWING_SET)} símbolos guardados")
    else:
        logger.warning("  Sin datos Merval este ciclo")


# ── MAIN ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ciclos", type=int, default=5,
                        help="Cantidad de ciclos de 60s a ejecutar (default: 5)")
    args = parser.parse_args()

    logger.info(f"🚀 GG Swing Trader Job iniciando — {args.ciclos} ciclos")
    logger.info("   (fase: acumulación de histórico Merval — sin señales todavía)")

    s = get_secrets()

    iol = IOLClient(s["iol_user"], s["iol_pass"])
    iol.login()

    sh = SheetsManager(s["gcp"])
    sh.conectar()

    for n in range(1, args.ciclos + 1):
        t_inicio = time.time()

        ejecutar_ciclo(iol, sh, n)

        if n < args.ciclos:
            elapsed = time.time() - t_inicio
            sleep_t = max(0, REFRESH_SECONDS - elapsed)
            logger.info(f"Esperando {sleep_t:.1f}s para próximo ciclo...")
            time.sleep(sleep_t)

    logger.info("✅ GG Swing Trader Job finalizado")


if __name__ == "__main__":
    main()
