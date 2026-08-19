"""
GG Swing Trader — Trader Job Standalone
=========================================
Script autónomo para correr desde GitHub Actions (sin Streamlit).
Ejecuta N ciclos de 60s y persiste estado en Google Sheets.

Único operador del sistema — app.py es solo dashboard de lectura.

ACTUALIZACIÓN (19-ago-2026) — Sistema GG Swing EN VIVO:
Este job ya no es solo acumulación de histórico. Ahora, en cada ciclo
dentro de horario de mercado:
  1. Sigue acumulando Historico_Merval_Raw (igual que antes — insumo
     gratis para la vela "de hoy" de Merval en alertas.py).
  2. Llama a alertas.procesar_alertas(), que corre signal_engine.py
     sobre las 19 posiciones del universo (8 Merval + 11 CEDEARs),
     abre posiciones nuevas priorizando por score cuando hay varias
     señales el mismo ciclo, y gestiona T1/T2/trailing/stop/max-hold
     de las posiciones ya abiertas.
  3. Persiste Posiciones_Abiertas, Operaciones, Estado_Cartera y
     Simulador_Estado en cada ciclo con cambios.
  4. Envía UN email por cada evento real de compra/venta — nunca de
     forma informativa. Máximo 4 emails por trade (entrada, T1, T2,
     cierre final), mínimo 2 (entrada, cierre directo por stop).

El Simulador se reconstruye desde Sheets al arrancar cada proceso (cada
invocación de GHA es un runner nuevo, sin memoria entre corridas) vía
sheets.cargar_posiciones() / cargar_estado_simulador().

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
import argparse

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from iol_client     import IOLClient
from alpaca_client  import AlpacaClient
from sheets_manager import SheetsManager
import signal_engine as se
import simulator as sim
import alertas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")

# ── CONSTANTES ────────────────────────────────────────────
REFRESH_SECONDS = 60

HORA_APERTURA = dtime(10, 30)
HORA_CIERRE   = dtime(17, 0)

# Universo Merval del Sistema GG Swing (8 activos que cotizan en BYMA/CNV
# vía IOL) — coincide con signal_engine.MERVAL.
MERVAL_SWING_SET = set(se.MERVAL)


def hora_argentina():
    return datetime.now(TZ_ARG)


# ── CREDENCIALES ──────────────────────────────────────────
def get_secrets():
    try:
        return {
            "iol_user":     os.environ["IOL_USER"],
            "iol_pass":     os.environ["IOL_PASS"],
            "alpaca_key":   os.environ["ALPACA_KEY"],
            "alpaca_sec":   os.environ["ALPACA_SECRET"],
            "gmail_user":   os.environ.get("GMAIL_USER", ""),
            "gmail_pass":   os.environ.get("GMAIL_APP_PASS", ""),
            "gcp":          json.loads(os.environ["GCP_SERVICE_ACCOUNT"]),
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


# ── FORMATO DE ALERTAS (4 emails máximo por trade) ────────

def mail_entrada(pos: sim.Posicion) -> tuple:
    subject = f"🟢 ENTRADA {pos.symbol} — score {pos.score:.0f}"
    cuerpo = (
        f"Sistema GG Swing — nueva posición\n"
        f"{'='*40}\n"
        f"Símbolo:   {pos.symbol}\n"
        f"Régimen:   {pos.regimen}\n"
        f"Score:     {pos.score:.0f}/100\n"
        f"\n"
        f"Entrada:   ${pos.precio_entry:,.2f}\n"
        f"Stop:      ${pos.precio_stop:,.2f}\n"
        f"Target 1:  ${pos.precio_t1:,.2f}\n"
        f"Target 2:  ${pos.precio_t2:,.2f}\n"
        f"\n"
        f"Cantidad:  {pos.cantidad_inicial:,.2f} u.\n"
        f"Monto:     ${pos.monto_entry:,.0f}\n"
        f"Riesgo:    {pos.riesgo_pct*100:.1f}% del capital\n"
        f"\n"
        f"Hora entrada: {pos.ts_entry} ART\n"
    )
    return subject, cuerpo


def mail_parcial(op: sim.Operacion, restante_pct: int) -> tuple:
    etiqueta = "Target 1" if op.motivo_cierre == "TARGET_1" else "Target 2"
    subject = f"🎯 {etiqueta.upper()} {op.symbol} — PnL {op.pnl_pct:+.2f}%"
    cuerpo = (
        f"Sistema GG Swing — cierre parcial\n"
        f"{'='*40}\n"
        f"Símbolo:      {op.symbol}\n"
        f"Evento:       {etiqueta} alcanzado\n"
        f"\n"
        f"Cerrado:      {op.cantidad:,.2f} u. @ ${op.precio_exit:,.2f}\n"
        f"PnL parcial:  ${op.pnl:+,.0f} ({op.pnl_pct:+.2f}%)\n"
        f"\n"
        f"Queda {restante_pct}% de la posición abierta.\n"
        + ("Stop movido a breakeven — el peor resultado posible ya es $0.\n" if op.motivo_cierre == "TARGET_1" else "Remanente gestionado con trailing stop (HMA).\n")
        + f"\n"
        f"Hora: {op.ts_exit} ART\n"
    )
    return subject, cuerpo


def mail_cierre_final(op: sim.Operacion) -> tuple:
    etiquetas = {
        "STOP_LOSS": "Stop Loss",
        "STOP_BREAKEVEN": "Stop en Breakeven",
        "TRAILING_STOP": "Trailing Stop",
        "MAX_HOLD_21D": "Máximo Hold (21 días)",
        "CIERRE_FORZADO": "Cierre Forzado (16:50 ART)",
    }
    etiqueta = etiquetas.get(op.motivo_cierre, op.motivo_cierre)
    emoji = "✅" if op.pnl > 0 else ("➖" if op.pnl == 0 else "❌")
    subject = f"{emoji} CIERRE {op.symbol} [{etiqueta}] — PnL {op.pnl_pct:+.2f}%"
    cuerpo = (
        f"Sistema GG Swing — cierre final de posición\n"
        f"{'='*40}\n"
        f"Símbolo:      {op.symbol}\n"
        f"Motivo:       {etiqueta}\n"
        f"\n"
        f"Cerrado:      {op.cantidad:,.2f} u. @ ${op.precio_exit:,.2f}\n"
        f"Entrada:      ${op.precio_entry:,.2f}\n"
        f"PnL final:    ${op.pnl:+,.0f} ({op.pnl_pct:+.2f}%)\n"
        f"\n"
        f"Entrada:      {op.ts_entry} ART\n"
        f"Salida:       {op.ts_exit} ART\n"
    )
    return subject, cuerpo


def enviar_alertas(secrets, resultado: dict, simulador: sim.Simulador):
    """Dispara emails SOLO para los eventos reales de este ciclo — nunca informativo."""
    for pos in resultado["entradas"]:
        subject, cuerpo = mail_entrada(pos)
        enviar_mail(secrets["gmail_user"], secrets["gmail_pass"], subject, cuerpo)

    for op in resultado["parciales"]:
        restante_pct = 60 if op.motivo_cierre == "TARGET_1" else 20
        subject, cuerpo = mail_parcial(op, restante_pct)
        enviar_mail(secrets["gmail_user"], secrets["gmail_pass"], subject, cuerpo)

    for op in resultado["cierres"]:
        subject, cuerpo = mail_cierre_final(op)
        enviar_mail(secrets["gmail_user"], secrets["gmail_pass"], subject, cuerpo)


# ── PRECIOS / HISTÓRICO MERVAL (acumulación, igual que antes) ────
def fetch_detalle_merval(iol):
    """
    Trae detalle OHLC+volumen de los símbolos Merval del universo GG Swing,
    usando get_panel('MerVal') — 1 solo request, sin costo adicional de cuota.
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
def ejecutar_ciclo(iol, alpaca, sheets, simulador, secrets, n_ciclo):
    hora  = hora_argentina()
    ahora = hora.time()

    logger.info(f"─── Ciclo {n_ciclo} | {hora.strftime('%H:%M:%S')} ART ───")

    if not (HORA_APERTURA <= ahora <= HORA_CIERRE):
        logger.info("Fuera de horario de mercado — sin captura ni señales")
        return

    # 1. Acumulación histórico Merval (insumo gratis para la vela de hoy)
    detalle_merval = fetch_detalle_merval(iol)
    if detalle_merval:
        sheets.guardar_tick_merval(detalle_merval)
        logger.info(f"  📊 Histórico Merval: {len(detalle_merval)}/{len(MERVAL_SWING_SET)} símbolos guardados")
    else:
        logger.warning("  Sin datos Merval este ciclo")

    # 2. Señales + gestión de posiciones
    try:
        resultado = alertas.procesar_alertas(iol, alpaca, sheets, simulador, ahora=ahora)
    except Exception as e:
        logger.error(f"  ❌ Error en procesar_alertas: {e}")
        return

    n_eventos = len(resultado["entradas"]) + len(resultado["parciales"]) + len(resultado["cierres"])
    if n_eventos == 0:
        logger.info("  Sin eventos de compra/venta este ciclo")
        return

    logger.info(
        f"  🔔 Eventos: {len(resultado['entradas'])} entrada(s), "
        f"{len(resultado['parciales'])} parcial(es), {len(resultado['cierres'])} cierre(s)"
    )

    # 3. Alertas — solo en eventos reales
    enviar_alertas(secrets, resultado, simulador)

    # 4. Persistir estado (solo cuando hubo eventos, para no gastar cuota de más)
    sheets.guardar_posiciones(simulador)
    sheets.cargar_estado_simulador  # noqa — no se usa acá, solo referencia de simetría
    for op in resultado["parciales"] + resultado["cierres"]:
        sheets.guardar_operacion(simulador.fila_sheets_operacion(op))

    precios_actuales = {
        sym: pos.precio_actual for sym, pos in simulador.posiciones.items()
    }
    sheets.guardar_estado_cartera(simulador.fila_sheets_estado(precios_actuales))

    ws = sheets._hojas.get("Simulador_Estado")
    if ws:
        ws.clear()
        ws.append_row(["efectivo", "op_counter"])
        ws.append_row([round(simulador.efectivo, 2), simulador._op_counter])


# ── MAIN ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ciclos", type=int, default=5,
                        help="Cantidad de ciclos de 60s a ejecutar (default: 5)")
    args = parser.parse_args()

    logger.info(f"🚀 GG Swing Trader Job iniciando — {args.ciclos} ciclos")

    s = get_secrets()

    iol = IOLClient(s["iol_user"], s["iol_pass"])
    iol.login()

    alpaca = AlpacaClient(s["alpaca_key"], s["alpaca_sec"])

    sh = SheetsManager(s["gcp"])
    sh.conectar()

    # Reconstruir estado del simulador desde Sheets — cada invocación de
    # GHA es un proceso nuevo, sin memoria de ciclos anteriores.
    simulador = sim.Simulador()
    sh.cargar_estado_simulador(simulador)
    sh.cargar_posiciones(simulador)
    logger.info(
        f"Estado cargado: efectivo=${simulador.efectivo:,.0f} | "
        f"{len(simulador.posiciones)} posición(es) abierta(s)"
    )

    for n in range(1, args.ciclos + 1):
        t_inicio = time.time()

        ejecutar_ciclo(iol, alpaca, sh, simulador, s, n)

        if n < args.ciclos:
            elapsed = time.time() - t_inicio
            sleep_t = max(0, REFRESH_SECONDS - elapsed)
            logger.info(f"Esperando {sleep_t:.1f}s para próximo ciclo...")
            time.sleep(sleep_t)

    logger.info("✅ GG Swing Trader Job finalizado")


if __name__ == "__main__":
    main()
