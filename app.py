"""
CCL Arbitrage - App Streamlit Completa
=======================================
Dashboard en tiempo real con:
- CCL implícito por acción
- Señales 🟢🟡🔴 por desvío
- Clima HMM 🟢/🔴 por activo
- Simulador con interés compuesto
- Alertas Gmail
- Persistencia Google Sheets
"""

import time, json, logging, smtplib, statistics
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")

def hora_argentina():
    """Retorna datetime actual en horario de Argentina."""
    return datetime.now(TZ_ARG)

from iol_client import IOLClient
from alpaca_client import AlpacaClient
from simulator import Simulador
from sheets_manager import SheetsManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="CCL Arbitrage", page_icon="📊", layout="wide")

REFRESH_SECONDS  = 60
HORA_APERTURA    = dtime(11, 30)
HORA_STOP_COMPRA = dtime(16, 30)
HORA_CIERRE      = dtime(16, 50)

PARES = {
    "GGAL":  ("GGAL",   10), "YPFD":  ("YPF",    1),
    "PAMP":  ("PAM",    25), "CEPU":  ("CEPU",  10),
    "AMZN":  ("AMZN",  144), "MSFT":  ("MSFT",  30),
    "NVDA":  ("NVDA",   24), "TSLA":  ("TSLA",  15),
    "AAPL":  ("AAPL",   20), "META":  ("META",  24),
    "GOOGL": ("GOOGL",  58), "MELI":  ("MELI", 120),
    "BMA":   ("BMA",    10), "VIST":  ("VIST",   3),
}

# ── SECRETS ───────────────────────────────────────────────
def get_secrets():
    try:
        return {
            "iol_user":   st.secrets["IOL_USER"],
            "iol_pass":   st.secrets["IOL_PASS"],
            "alp_key":    st.secrets["ALPACA_KEY"],
            "alp_secret": st.secrets["ALPACA_SECRET"],
            "gmail_user": st.secrets["GMAIL_USER"],
            "gmail_pass": st.secrets["GMAIL_APP_PASS"],
            "gcp":        json.loads(st.secrets["GCP_SERVICE_ACCOUNT"]),
        }
    except:
        return None

# ── SESSION STATE ──────────────────────────────────────────
def init_state():
    s = get_secrets()
    if not s:
        return False
    if "ready" not in st.session_state:
        st.session_state.iol      = IOLClient(s["iol_user"], s["iol_pass"])
        st.session_state.iol.login()
        st.session_state.alpaca   = AlpacaClient(s["alp_key"], s["alp_secret"])
        sh = SheetsManager(s["gcp"])
        sh.conectar()
        st.session_state.sheets   = sh
        st.session_state.historial = sh.cargar_historial_ccl()
        # Restaurar simulador desde Sheets
        sim = Simulador()
        sh.cargar_estado_simulador(sim)
        sh.cargar_posiciones(sim)
        st.session_state.sim      = sim
        st.session_state.gmail    = {"user": s["gmail_user"], "pass": s["gmail_pass"]}
        st.session_state.alertadas = {}
        st.session_state.ready    = True
    return True

# ── HMM ───────────────────────────────────────────────────
def clima_hmm(sym, historial):
    vals = [h["ccl"].get(sym) for h in historial if h["ccl"].get(sym)]
    if len(vals) < 5:
        return "🔴"
    try:
        from hmmlearn.hmm import GaussianHMM
        X = np.array(vals).reshape(-1, 1)
        m = GaussianHMM(n_components=2, random_state=42, n_iter=100).fit(X)
        estado = m.predict(X)[-1]
        bull   = np.argmax(m.means_.flatten())
        return "🟢" if estado == bull else "🔴"
    except:
        return "🔴"

# ── CCL ───────────────────────────────────────────────────
def calcular_ccl(p_ars, p_usd):
    ccl_map = {}
    for sym, (sym_usd, ratio) in PARES.items():
        a = p_ars.get(sym)
        u = p_usd.get(sym_usd)
        if a and u and u > 0:
            ccl_map[sym] = (a / u) * ratio
    avg = statistics.median(ccl_map.values()) if ccl_map else 0
    return ccl_map, avg

# ── GMAIL ─────────────────────────────────────────────────
def enviar_mail(subject: str, cuerpo: str):
    """Envía un mail genérico."""
    g = st.session_state.gmail
    if not g["user"]:
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = msg["To"] = g["user"]
        msg["Subject"] = subject
        msg.attach(MIMEText(cuerpo, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(g["user"], g["pass"])
            smtp.send_message(msg)
        logger.info(f"📧 Mail enviado: {subject}")
    except Exception as e:
        logger.error(f"Gmail error: {e}")

def alerta_señales(señales, ccl_avg):
    """Avisa cuando hay señales activas (máx 1 vez cada 30 min por activo)."""
    if not señales:
        return
    ahora = hora_argentina()
    nuevas = [s for s in señales
              if (ahora - st.session_state.alertadas.get(s["sym"], datetime(2000,1,1,tzinfo=TZ_ARG))).seconds > 1800]
    if not nuevas:
        return
    for s in nuevas:
        st.session_state.alertadas[s["sym"]] = ahora
    cuerpo = f"📊 GG HMM-CCL — Señales Activas\n"
    cuerpo += f"CCL Promedio: ${ccl_avg:.2f} | {ahora.strftime('%d/%m/%Y %H:%M:%S')}\n"
    cuerpo += f"{'─'*40}\n"
    for s in nuevas:
        cuerpo += f"{s['sym']:<8} {s['dev']:>+7.2f}%  {s['clima']}  {s['señal']}\n"
    enviar_mail(f"🚨 GG: {len(nuevas)} señal(es) | CCL ${ccl_avg:.0f}", cuerpo)

def alerta_operacion(ops_abiertas, ops_cerradas, ccl_avg):
    """Avisa cuando el simulador ejecuta una compra o venta real."""
    if not ops_abiertas and not ops_cerradas:
        return
    ahora = hora_argentina()
    cuerpo = f"💹 GG HMM-CCL — Operación Ejecutada\n"
    cuerpo += f"CCL Promedio: ${ccl_avg:.2f} | {ahora.strftime('%d/%m/%Y %H:%M:%S')}\n"
    cuerpo += f"{'─'*40}\n"
    if ops_abiertas:
        cuerpo += f"\n🟢 COMPRAS ({len(ops_abiertas)}):\n"
        for pos in ops_abiertas:
            cuerpo += f"  {pos.symbol:<8} ${pos.precio_entry:,.1f} | Monto: ${pos.monto_entry:,.0f}\n"
    if ops_cerradas:
        cuerpo += f"\n🔴 VENTAS ({len(ops_cerradas)}):\n"
        for op in ops_cerradas:
            emoji = "✅" if op.pnl > 0 else "❌"
            cuerpo += f"  {emoji} {op.symbol:<8} PnL: ${op.pnl:+,.0f} ({op.pnl_pct:+.2f}%) [{op.motivo_cierre}]\n"
    n = len(ops_abiertas) + len(ops_cerradas)
    enviar_mail(f"💹 GG: {n} op(s) ejecutada(s) | CCL ${ccl_avg:.0f}", cuerpo)

# ── FETCH ─────────────────────────────────────────────────
def fetch_precios(ts_key):
    iol    = st.session_state.iol
    alpaca = st.session_state.alpaca
    p_ars  = {}
    for sym in PARES:
        q = iol.get_quote(sym)
        if q and q.get("last"):
            p_ars[sym] = q["last"]
        time.sleep(0.08)
    syms_usd = list({v[0] for v in PARES.values()})
    snaps    = alpaca.get_snapshots(syms_usd)
    p_usd    = {s: snaps[s]["last"] for s in snaps if snaps[s].get("last")}
    return p_ars, p_usd

# ── MAIN ──────────────────────────────────────────────────
def main():
    st.title("📊 GG Investment 🦅🤑")
    st.caption("IOL (ARS) + Alpaca (USD) | HMM Climate | Simulador Intradiario")

    if not init_state():
        st.error("⚠️ Configurar credenciales en Streamlit Secrets.")
        return

    sheets    = st.session_state.sheets
    sim       = st.session_state.sim
    historial = st.session_state.historial
    hora      = hora_argentina()
    ahora     = hora.time()

    # Fetch
    ts_key = str(int(time.time() // REFRESH_SECONDS))
    p_ars, p_usd = fetch_precios(ts_key)

    # CCL
    ccl_map, ccl_avg = calcular_ccl(p_ars, p_usd)

    # Guardar snapshot
    if ccl_map:
        historial.append({"ts": hora.isoformat(), "ccl": ccl_map, "avg": ccl_avg})
        sheets.guardar_snapshot_ccl(ccl_map, ccl_avg)

    # Señales y climas
    rows, señales_alerta, climas = [], [], {}
    for sym, ccl in ccl_map.items():
        dev   = (ccl / ccl_avg - 1) * 100 if ccl_avg else 0
        clima = clima_hmm(sym, historial)
        climas[sym] = "🟢 BULL" if clima == "🟢" else "🔴 BEAR"
        señal = "🟢 COMPRAR" if dev < -0.5 else "🔴 VENDER" if dev > 0.5 else "🟡 NEUTRAL"
        rows.append({
            "sym": sym, "ccl": ccl, "dev": dev, "clima": clima,
            "señal": señal, "p_ars": p_ars.get(sym, 0),
            "p_usd": p_usd.get(PARES[sym][0], 0),
        })
        if señal != "🟡 NEUTRAL":
            señales_alerta.append({"sym": sym, "dev": dev, "clima": clima, "señal": señal})

    # Simulador
    if HORA_APERTURA <= ahora:
        resultado = sim.procesar_ciclo(ccl_map, ccl_avg, p_ars, climas, ahora)
        ops_cerradas = resultado.get("cerradas", []) + resultado.get("forzadas", [])
        ops_abiertas = resultado.get("abiertas", [])
        hay_cambios = bool(ops_abiertas or ops_cerradas)
        for op in ops_cerradas:
            sheets.guardar_operacion(sim.fila_sheets_operacion(op))
        # Estado cartera: solo guardar cada 5 ciclos (~5 min) o si hay operaciones
        ciclo_actual = int(time.time() // REFRESH_SECONDS)
        if hay_cambios or ciclo_actual % 5 == 0:
            sheets.guardar_estado_cartera(sim.fila_sheets_estado(p_ars))
        # Posiciones y estado: solo si hubo cambios
        if hay_cambios:
            sheets.guardar_posiciones(sim)
            sheets.guardar_estado_simulador(sim)
        alerta_operacion(ops_abiertas, ops_cerradas, ccl_avg)

    # Alertas señales (independiente del simulador)
    alerta_señales(señales_alerta, ccl_avg)

    # ── KPIs ──────────────────────────────────────────────
    resumen = sim.resumen(p_ars)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CCL Promedio",   f"${ccl_avg:.2f}")
    c2.metric("Capital Total",  f"${resumen['capital_total']:,.0f}", f"{resumen['pnl_pct']:+.2f}%")
    c3.metric("Efectivo",       f"${resumen['efectivo']:,.0f}")
    c4.metric("En Posiciones",  f"${resumen['en_posiciones']:,.0f}")
    c5.metric("Win Rate",       f"{resumen['win_rate']:.0f}%", f"{resumen['operaciones_total']} ops")

    # ── Estado mercado ─────────────────────────────────────
    if ahora < HORA_APERTURA:
        st.warning(f"⏳ Mercado abre a las {HORA_APERTURA.strftime('%H:%M')} hs")
    elif ahora >= HORA_CIERRE:
        st.error("🔴 16:50 hs — Cierre forzado de posiciones activo")
    elif ahora >= HORA_STOP_COMPRA:
        st.warning("⚠️ 16:30 hs — Sin nuevas compras | Solo cierres")
    else:
        st.success(f"🟢 Mercado abierto | {resumen['posiciones_abiertas']} posiciones abiertas")

    # ── Gráfico ────────────────────────────────────────────
    rows_sorted = sorted(rows, key=lambda x: x["dev"])
    colors = ["#00C851" if r["señal"]=="🟢 COMPRAR" else "#FF4444" if r["señal"]=="🔴 VENDER" else "#888" for r in rows_sorted]
    fig = go.Figure(go.Bar(
        x=[r["sym"] for r in rows_sorted],
        y=[r["dev"] for r in rows_sorted],
        marker_color=colors,
        text=[f"{r['dev']:+.2f}%" for r in rows_sorted],
        textposition="outside",
    ))
    fig.add_hline(y=0.5,  line_dash="dash", line_color="#FF4444")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="#00C851")
    fig.update_layout(
        title="Desviación CCL vs Promedio",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla señales ──────────────────────────────────────
    st.subheader("📋 Señales en Tiempo Real")
    df = pd.DataFrame([{
        "Activo": r["sym"],
        "P. ARS": f"${r['p_ars']:,.1f}",
        "P. USD": f"${r['p_usd']:.3f}",
        "CCL":    f"${r['ccl']:,.2f}",
        "Desvío": f"{r['dev']:+.2f}%",
        "Clima":  r["clima"],
        "Señal":  r["señal"],
    } for r in rows_sorted])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Posiciones abiertas ────────────────────────────────
    if any(sim.posiciones.values()):
        st.subheader("💼 Posiciones Abiertas")
        pos_rows = []
        for sym, poss in sim.posiciones.items():
            for pos in poss:
                precio_actual = p_ars.get(sym, pos.precio_entry)
                pnl = (precio_actual - pos.precio_entry) * pos.cantidad
                pos_rows.append({
                    "ID": pos.id, "Activo": sym,
                    "Entrada": f"${pos.precio_entry:,.1f}",
                    "Actual":  f"${precio_actual:,.1f}",
                    "Cant.":   f"{pos.cantidad:.2f}",
                    "Invertido": f"${pos.monto_entry:,.0f}",
                    "PnL": f"${pnl:+,.0f}",
                    "Hora entrada": pos.ts_entry,
                })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

    # ── Historial ops ──────────────────────────────────────
    with st.expander("📜 Historial de Operaciones"):
        ops = sheets.cargar_operaciones()
        if ops:
            cols = ["ID","Activo","Tipo","Cant.","P.Entry","P.Exit",
                    "M.Entry","M.Exit","PnL","PnL%","Apertura","Cierre","Motivo"]
            st.dataframe(pd.DataFrame(ops, columns=cols), use_container_width=True, hide_index=True)
        else:
            st.info("Sin operaciones registradas aún.")

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Config")
        st.markdown(f"**Snapshots HMM:** {len(historial)}")
        st.markdown(f"**Actualizado:** {hora.strftime('%H:%M:%S')}")
        st.divider()
        st.markdown("**Simulador**")
        st.markdown(f"Capital inicial: $10.000.000")
        st.markdown(f"Por operación: 15%")
        st.markdown(f"Máx./especie: 2")
        st.markdown(f"Ventana: 11:30 → 16:50")
        st.divider()
        if st.button("🔄 Reset simulador"):
            sim_nuevo = Simulador()
            # Limpiar hojas de estado en Sheets
            sheets.guardar_posiciones(sim_nuevo)
            sheets.guardar_estado_simulador(sim_nuevo)
            st.session_state.sim = sim_nuevo
            st.success("✅ Simulador reseteado")
            st.rerun()

    st.caption(f"⏱ Próxima actualización en {REFRESH_SECONDS}s")
    time.sleep(REFRESH_SECONDS)
    st.rerun()


if __name__ == "__main__":
    main()
