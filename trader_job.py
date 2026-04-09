"""
GG HMM-CCL — Trader Job Standalone
=====================================
Script autónomo para correr desde GitHub Actions (sin Streamlit).
Ejecuta N ciclos de 60s y persiste estado en Google Sheets.

Uso:
    python trader_job.py            # 5 ciclos (default)
    python trader_job.py --ciclos 3 # N ciclos custom

Credenciales via variables de entorno (GitHub Secrets):
    IOL_USER, IOL_PASS
    ALPACA_KEY, ALPACA_SECRET
    GCP_SERVICE_ACCOUNT  (JSON string)
    GMAIL_USER, GMAIL_APP_PASS
"""

import os
import sys
import time
import json
import logging
import smtplib
import statistics
import argparse
import numpy as np

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from iol_client     import IOLClient
from alpaca_client  import AlpacaClient
from simulator      import Simulador
from sheets_manager import SheetsManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")

# ── CONSTANTES ────────────────────────────────────────────
REFRESH_SECONDS     = 60
HMM_MAX_SNAPSHOTS   = 500
HMM_BARRAS_LOOKBACK = 252   # barras 1D (~1 año)

HORA_APERTURA    = dtime(10, 30)
HORA_STOP_COMPRA = dtime(16, 30)
HORA_CIERRE      = dtime(16, 50)

# 20 pares CCL — LOMA eliminado
PARES = {
    "GGAL":  ("GGAL",   10), "YPFD":  ("YPF",    1),
    "PAMP":  ("PAM",    25), "CEPU":  ("CEPU",  10),
    "AMZN":  ("AMZN",  144), "MSFT":  ("MSFT",  30),
    "NVDA":  ("NVDA",   24), "TSLA":  ("TSLA",  15),
    "AAPL":  ("AAPL",   20), "META":  ("META",  24),
    "GOOGL": ("GOOGL",  58), "MELI":  ("MELI", 120),
    "BMA":   ("BMA",    10), "SPY":   ("SPY",   20),
    "TGSU2": ("TGS",     5), "IBIT":  ("IBIT",  10),
    "GLD":   ("GLD",    50), 
}


def hora_argentina():
    return datetime.now(TZ_ARG)


# ── CREDENCIALES ──────────────────────────────────────────
def get_secrets():
    try:
        return {
            "iol_user":   os.environ["IOL_USER"],
            "iol_pass":   os.environ["IOL_PASS"],
            "alp_key":    os.environ["ALPACA_KEY"],
            "alp_secret": os.environ["ALPACA_SECRET"],
            "gmail_user": os.environ.get("GMAIL_USER", ""),
            "gmail_pass": os.environ.get("GMAIL_APP_PASS", ""),
            "gcp":        json.loads(os.environ["GCP_SERVICE_ACCOUNT"]),
        }
    except KeyError as e:
        logger.error(f"Variable de entorno faltante: {e}")
        sys.exit(1)


# ── GMAIL ─────────────────────────────────────────────────
def enviar_mail(gmail_user, gmail_pass, subject, cuerpo):
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


# ── HMM ───────────────────────────────────────────────────
def fetch_barras_hmm(alpaca):
    """Descarga barras 1D para todos los símbolos USD — individual por símbolo."""
    syms_usd = list({v[0] for v in PARES.values()})
    return alpaca.get_bars(syms_usd, timeframe="1Day", limit=HMM_BARRAS_LOOKBACK)


def clima_hmm(sym, barras_cache):
    sym_usd = PARES[sym][0]
    precios = barras_cache.get(sym_usd, [])
    if len(precios) < 63:
        logger.debug(f"HMM {sym}: solo {len(precios)} barras — devolviendo 🔴")
        return "🔴"
    try:
        from hmmlearn.hmm import GaussianHMM
        ret    = np.diff(np.log(precios)).reshape(-1, 1)
        m      = GaussianHMM(n_components=2, random_state=42, n_iter=100).fit(ret)
        estado = m.predict(ret)[-1]
        bull   = int(np.argmax(m.means_.flatten()))
        # Guard: si la media del estado bull es negativa, no es bull real
        if m.means_.flatten()[bull] < -0.0005:
            return "🔴"
        return "🟢" if estado == bull else "🔴"
    except Exception as e:
        logger.warning(f"HMM error {sym}: {e}")
        return "🔴"


# ── PRECIOS ───────────────────────────────────────────────
def fetch_precios(iol, alpaca):
    p_ars = {}

    CEDEARS_SET = {
        "AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "META",
        "GOOGL", "MELI", "GLD", "IBIT", "SPY",
    }
    MERVAL_SET = {
        "GGAL", "YPFD", "PAMP", "CEPU", "BMA", "TGSU2",
    }

    # Batch 1 — CEDEARs (1 request)
    try:
        data = iol.get_panel("CEDEARs")
        for t in data:
            if t["simbolo"] in CEDEARS_SET:
                p_ars[t["simbolo"]] = t["ultimoPrecio"]
    except Exception as e:
        logger.warning(f"IOL CEDEARs batch: {e}")

    # Batch 2 — MerVal (1 request)
    try:
        data = iol.get_panel("MerVal")
        for t in data:
            if t["simbolo"] in MERVAL_SET:
                p_ars[t["simbolo"]] = t["ultimoPrecio"]
    except Exception as e:
        logger.warning(f"IOL MerVal batch: {e}")

    p_usd = {}
    syms_usd = list({v[0] for v in PARES.values()})
    try:
        snaps = alpaca.get_snapshots(syms_usd)
        for sym_usd, snap in snaps.items():
            p_usd[sym_usd] = snap.get("last") or 0
    except Exception as e:
        logger.warning(f"Alpaca snapshots: {e}")

    return p_ars, p_usd


def calcular_ccl(p_ars, p_usd):
    ccl_map = {}
    for sym, (sym_usd, ratio) in PARES.items():
        a = p_ars.get(sym)
        u = p_usd.get(sym_usd)
        if a and u and u > 0:
            ccl_map[sym] = (a / u) * ratio
    avg = statistics.median(ccl_map.values()) if ccl_map else 0
    return ccl_map, avg


# ── ALERTAS ───────────────────────────────────────────────
def alerta_operacion(gmail_user, gmail_pass, ops_abiertas, ops_cerradas, ccl_avg):
    if not ops_abiertas and not ops_cerradas:
        return
    ahora  = hora_argentina()
    cuerpo = f"💹 GG HMM-CCL — Operación Ejecutada\n"
    cuerpo += f"CCL Promedio: ${ccl_avg:.2f} | {ahora.strftime('%d/%m/%Y %H:%M:%S')}\n"
    cuerpo += f"{'─'*40}\n"
    if ops_abiertas:
        cuerpo += f"\n🟢 COMPRAS ({len(ops_abiertas)}):\n"
        for pos in ops_abiertas:
            cuerpo += f"  {pos.symbol:<8} ${pos.precio_entry:,.1f} | Monto: ${pos.monto_entry:,.0f} | dev: {pos.dev_entry:+.2f}%\n"
    if ops_cerradas:
        cuerpo += f"\n🔴 VENTAS ({len(ops_cerradas)}):\n"
        for op in ops_cerradas:
            emoji = "✅" if op.pnl > 0 else "❌"
            cuerpo += f"  {emoji} {op.symbol:<8} PnL: ${op.pnl:+,.0f} ({op.pnl_pct:+.2f}%) [{op.motivo_cierre}]\n"
    n = len(ops_abiertas) + len(ops_cerradas)
    enviar_mail(gmail_user, gmail_pass,
                f"💹 GG: {n} op(s) ejecutada(s) | CCL ${ccl_avg:.0f}", cuerpo)


# ── CICLO PRINCIPAL ───────────────────────────────────────
def ejecutar_ciclo(iol, alpaca, sim, sheets, barras_cache,
                   historial, gmail_user, gmail_pass, n_ciclo):
    hora  = hora_argentina()
    ahora = hora.time()

    logger.info(f"─── Ciclo {n_ciclo} | {hora.strftime('%H:%M:%S')} ART ───")

    # Warmup: primer ciclo de cada ejecución no opera (estabiliza precios)
    en_warmup = (n_ciclo == 1)

    p_ars, p_usd = fetch_precios(iol, alpaca)
    if not p_ars:
        logger.warning("Sin precios ARS — saltando ciclo")
        return

    ccl_map, ccl_avg = calcular_ccl(p_ars, p_usd)
    if not ccl_map:
        logger.warning("Sin CCL calculado — saltando ciclo")
        return

    logger.info(f"CCL promedio: ${ccl_avg:.2f} | {len(ccl_map)} activos")

    # Guardar snapshot en Sheets
    historial.append({"ts": hora.isoformat(), "ccl": ccl_map, "avg": ccl_avg, "usd": p_usd})
    if len(historial) > HMM_MAX_SNAPSHOTS:
        historial[:] = historial[-HMM_MAX_SNAPSHOTS:]
    sheets.guardar_snapshot_ccl(ccl_map, ccl_avg, p_usd=p_usd)

    # Climas HMM
    climas = {}
    for sym in ccl_map:
        clima = clima_hmm(sym, barras_cache)
        climas[sym] = "🟢 BULL" if clima == "🟢" else "🔴 BEAR"

    # Log señales con diagnóstico de bloqueos
    for sym, ccl in sorted(ccl_map.items(), key=lambda x: (x[1] / ccl_avg - 1)):
        dev   = (ccl / ccl_avg - 1) * 100
        clima = climas[sym]
        if dev < sim.umbral_compra:
            if clima == "🟢 BULL":
                logger.info(f"  {sym:<6} dev={dev:+.2f}%  {clima}  ✅ → señal COMPRA")
            else:
                logger.info(f"  {sym:<6} dev={dev:+.2f}%  {clima}  ❌ bloqueado por HMM")
        else:
            logger.info(f"  {sym:<6} dev={dev:+.2f}%  {clima}")

    # Verificar horario de mercado
    if not (HORA_APERTURA <= ahora):
        logger.info("Fuera de horario de mercado — sin operaciones")
        return

    # En warmup se pasa medianoche para que el simulador no opere
    ahora_sim = dtime(0, 0) if en_warmup else ahora
    resultado = sim.procesar_ciclo(ccl_map, ccl_avg, p_ars, climas, ahora_sim)

    ops_cerradas = resultado.get("cerradas", []) + resultado.get("forzadas", [])
    ops_abiertas = resultado.get("abiertas", [])
    hay_cambios  = bool(ops_abiertas or ops_cerradas)

    for op in ops_cerradas:
        sheets.guardar_operacion(sim.fila_sheets_operacion(op))
        logger.info(f"  VENTA {op.symbol} [{op.motivo_cierre}] PnL={op.pnl_pct:+.2f}%")

    for pos in ops_abiertas:
        logger.info(f"  COMPRA {pos.symbol} @ ${pos.precio_entry:,.1f} dev={pos.dev_entry:+.2f}%")

    if hay_cambios or n_ciclo % 5 == 0:
        sheets.guardar_estado_cartera(sim.fila_sheets_estado(p_ars))

    if hay_cambios:
        sheets.guardar_posiciones(sim)
        sheets.guardar_estado_simulador(sim)
        alerta_operacion(gmail_user, gmail_pass, ops_abiertas, ops_cerradas, ccl_avg)

    n_pos = sum(len(v) for v in sim.posiciones.values())
    logger.info(f"Posiciones abiertas: {n_pos} | Efectivo: ${sim.efectivo:,.0f}")


# ── MAIN ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ciclos", type=int, default=5,
                        help="Cantidad de ciclos de 60s a ejecutar (default: 5)")
    args = parser.parse_args()

    logger.info(f"🚀 GG Trader Job iniciando — {args.ciclos} ciclos")

    s = get_secrets()

    iol    = IOLClient(s["iol_user"], s["iol_pass"])
    iol.login()
    alpaca = AlpacaClient(s["alp_key"], s["alp_secret"])
    sh     = SheetsManager(s["gcp"])
    sh.conectar()

    # Cargar estado del simulador desde Sheets
    sim = Simulador(umbral_compra=-0.70)
    sh.cargar_estado_simulador(sim)
    sh.cargar_posiciones(sim)
    logger.info(f"Simulador cargado — efectivo: ${sim.efectivo:,.0f} | "
                f"posiciones: {sum(len(v) for v in sim.posiciones.values())} | "
                f"umbral_compra: {sim.umbral_compra:+.2f}%")

    historial = sh.cargar_historial_ccl()
    logger.info(f"Historial CCL: {len(historial)} snapshots")

    logger.info("Descargando barras 1D para HMM...")
    barras_cache = fetch_barras_hmm(alpaca)
    logger.info(f"Barras HMM cargadas: {len(barras_cache)} símbolos — {list(barras_cache.keys())}")

    for n in range(1, args.ciclos + 1):
        t_inicio = time.time()

        ejecutar_ciclo(
            iol, alpaca, sim, sh, barras_cache,
            historial, s["gmail_user"], s["gmail_pass"], n
        )

        if n < args.ciclos:
            elapsed = time.time() - t_inicio
            sleep_t = max(0, REFRESH_SECONDS - elapsed)
            logger.info(f"Esperando {sleep_t:.1f}s para próximo ciclo...")
            time.sleep(sleep_t)

    logger.info("✅ GG Trader Job finalizado")


if __name__ == "__main__":
    main()
