"""
GG HMM-CCL — Dashboard Streamlit (solo lectura)
================================================
Visualización en tiempo real del sistema de arbitraje CCL.

RESPONSABILIDADES DE ESTE ARCHIVO:
  - Mostrar estado actual (precios, CCL, señales, HMM)
  - Mostrar posiciones abiertas y KPIs del simulador
  - Mostrar historial de operaciones
  - Permitir venta manual de posiciones (única escritura)

NO HACE TRADING — el único operador es trader_job.py vía GitHub Actions.
Streamlit puede dormir sin consecuencias: el GHA opera igual.

MODELO DE CLIMA (Simons):
  El HMM se entrena sobre log-returns del precio USD del subyacente,
  NO sobre niveles del CCL. Esto garantiza que la señal (CCL bajo) y
  el clima (momentum USD alcista) sean variables ortogonales e independientes.

  Señal de compra = spread CCL favorable  AND  régimen bull en USD
  Señal de venta  = spread CCL desfavorable  (HMM no interviene)

  IDENTIFICACIÓN DE ESTADO BULL:
  Se usa argmax de medias puras para identificar el estado bull.
  Guard adicional: si la media más alta es negativa, ambos estados son
  bajistas (mercado bear con rebotes violentos) y se retorna 🔴.

FETCH DE BARRAS:
  Se fetchea símbolo por símbolo (no en batch) para garantizar que Alpaca
  retorne exactamente `limit` barras por símbolo.
"""

import time, json, logging, statistics
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")

def hora_argentina():
    return datetime.now(TZ_ARG)

from iol_client     import IOLClient
from alpaca_client  import AlpacaClient
from simulator      import Simulador
from sheets_manager import SheetsManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="CCL Arbitrage", page_icon="📊", layout="wide")

REFRESH_SECONDS  = 60
HORA_APERTURA    = dtime(10, 30)
HORA_STOP_COMPRA = dtime(16, 30)
HORA_CIERRE      = dtime(16, 50)
HMM_MAX_SNAPSHOTS     = 500
HMM_BARRAS_LOOKBACK   = 252
HMM_BARRAS_REFRESH_MIN = 10

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

# Umbral de display para el gráfico (consistente con trader_job.py)
UMBRAL_COMPRA_DISPLAY = -0.70
UMBRAL_VENTA_DISPLAY  =  0.10

# ── SECRETS ───────────────────────────────────────────────
def get_secrets():
    try:
        return {
            "iol_user":   st.secrets["IOL_USER"],
            "iol_pass":   st.secrets["IOL_PASS"],
            "alp_key":    st.secrets["ALPACA_KEY"],
            "alp_secret": st.secrets["ALPACA_SECRET"],
            "gcp":        json.loads(st.secrets["GCP_SERVICE_ACCOUNT"]),
        }
    except:
        return None

# ── SESSION STATE ──────────────────────────────────────────
def init_state():
    s = get_secrets()
    if not s:
        return False
    if "sim" not in st.session_state:
        st.session_state.iol    = IOLClient(s["iol_user"], s["iol_pass"])
        st.session_state.iol.login()
        st.session_state.alpaca = AlpacaClient(s["alp_key"], s["alp_secret"])
        sh = SheetsManager(s["gcp"])
        sh.conectar()
        st.session_state.sheets   = sh

        # Cargar estado del simulador desde Sheets (solo lectura — no opera)
        sim = Simulador()
        sh.cargar_estado_simulador(sim)
        sh.cargar_posiciones(sim)
        st.session_state.sim = sim

        # Historial CCL en memoria para cálculo HMM intradiario (solo lectura)
        st.session_state.historial = sh.cargar_historial_ccl()

        st.session_state.vp_cache = {
            "timestamp": None, "intraday": None, "dos_semanas": None
        }
        st.session_state.ready = True
    return True


# ── HMM — MODELO SIMONS (barras 1D) ──────────────────────
def _fetch_barras_1d():
    """
    Descarga barras 1D via AlpacaClient — individual por símbolo.
    Solo para visualización del clima en el dashboard.
    """
    alpaca = st.session_state.get("alpaca")
    if not alpaca:
        return {}
    syms_usd  = list({v[0] for v in PARES.values()})
    resultado = {}
    for sym in syms_usd:
        barras = alpaca.get_bars([sym], timeframe="1Day", limit=HMM_BARRAS_LOOKBACK)
        if sym in barras:
            resultado[sym] = barras[sym]
            logger.info(f"HMM fetch: {sym} → {len(barras[sym])} barras")
        else:
            logger.warning(f"HMM fetch: {sym} → sin datos")
    return resultado


def _refrescar_barras_si_necesario():
    """Refresca el cache de barras 1D si pasaron más de HMM_BARRAS_REFRESH_MIN minutos."""
    ahora  = hora_argentina()
    ultimo = st.session_state.get("hmm_barras_ts")
    if ultimo is None or (ahora - ultimo).seconds >= HMM_BARRAS_REFRESH_MIN * 60:
        barras = _fetch_barras_1d()
        if barras:
            st.session_state["hmm_barras"]    = barras
            st.session_state["hmm_barras_ts"] = ahora
            logger.info(f"HMM 1D: barras actualizadas → {len(barras)} símbolos")


def clima_hmm(sym, historial=None):
    """
    Retorna '🟢' si el subyacente USD está en régimen bull, '🔴' si no.
    n_iter=200 para convergencia robusta. Mínimo 63 barras.
    Agrega retorno intradiario si hay historial disponible.
    """
    sym_usd = PARES[sym][0]
    barras  = st.session_state.get("hmm_barras", {})
    precios = barras.get(sym_usd, [])

    if len(precios) < 63:
        return "🔴"

    try:
        from hmmlearn.hmm import GaussianHMM

        ret_hist = np.diff(np.log(precios)).reshape(-1, 1)
        m = GaussianHMM(n_components=2, random_state=42, n_iter=200).fit(ret_hist)

        means = m.means_.flatten()
        bull  = int(np.argmax(means))

        if means[bull] < -0.0005:
            return "🔴"

        ret_pred = ret_hist.copy()
        if historial:
            precio_actual = historial[-1].get("usd", {}).get(sym_usd)
            ultimo_cierre = precios[-1]
            if precio_actual and ultimo_cierre > 0:
                ret_hoy  = np.log(precio_actual / ultimo_cierre)
                ret_pred = np.vstack([ret_hist, [[ret_hoy]]])

        estado = m.predict(ret_pred)[-1]
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


# ── FETCH PRECIOS (solo display) ──────────────────────────
def fetch_precios():
    iol    = st.session_state.iol
    alpaca = st.session_state.alpaca
    p_ars  = {}

    CEDEARS_SET = {"AAPL","AMZN","MSFT","NVDA","TSLA","META","GOOGL","MELI","GLD","IBIT","SPY"}
    MERVAL_SET  = {"GGAL","YPFD","PAMP","CEPU","BMA","TGSU2"}

    try:
        data = iol.get_panel("CEDEARs")
        for t in data:
            if t["simbolo"] in CEDEARS_SET:
                p_ars[t["simbolo"]] = t["ultimoPrecio"]
    except Exception as e:
        st.warning(f"IOL CEDEARs batch: {e}")

    try:
        data = iol.get_panel("MerVal")
        for t in data:
            if t["simbolo"] in MERVAL_SET:
                p_ars[t["simbolo"]] = t["ultimoPrecio"]
    except Exception as e:
        st.warning(f"IOL MerVal batch: {e}")

    syms_usd = list({v[0] for v in PARES.values()})
    snaps    = alpaca.get_snapshots(syms_usd)
    p_usd    = {s: snaps[s]["last"] for s in snaps if snaps[s].get("last")}
    return p_ars, p_usd


# ── MAIN ──────────────────────────────────────────────────
def main():
    st.title("📊 GG Investments 🦅🤑")
    st.caption("IOL (ARS) + Alpaca (USD) | HMM Climate | Dashboard — operado por GHA")

    if not init_state():
        st.error("⚠️ Configurar credenciales en Streamlit Secrets.")
        return

    sheets    = st.session_state.sheets
    sim       = st.session_state.sim
    historial = st.session_state.historial
    hora      = hora_argentina()
    ahora     = hora.time()

    # Refrescar estado del simulador desde Sheets en cada ciclo
    # (para reflejar lo que operó el GHA mientras la app estaba cerrada)
    try:
        sh_sim_nuevo = Simulador()
        sheets.cargar_estado_simulador(sh_sim_nuevo)
        sheets.cargar_posiciones(sh_sim_nuevo)
        st.session_state.sim = sh_sim_nuevo
        sim = sh_sim_nuevo
    except Exception as e:
        logger.warning(f"No se pudo refrescar sim desde Sheets: {e}")

    # Refrescar barras HMM cada 10 min (solo para display)
    _refrescar_barras_si_necesario()

    p_ars, p_usd = fetch_precios()
    ccl_map, ccl_avg = calcular_ccl(p_ars, p_usd)

    # Actualizar historial en memoria para HMM intradiario (solo lectura local,
    # no se persiste — el escritor es trader_job.py)
    if ccl_map:
        historial.append({"ts": hora.isoformat(), "ccl": ccl_map, "avg": ccl_avg, "usd": p_usd})
        if len(historial) > HMM_MAX_SNAPSHOTS:
            historial[:] = historial[-HMM_MAX_SNAPSHOTS:]

    # ── Señales y climas ───────────────────────────────────
    rows = []
    for sym, ccl in ccl_map.items():
        dev   = (ccl / ccl_avg - 1) * 100 if ccl_avg else 0
        clima = clima_hmm(sym, historial)

        if dev <= UMBRAL_COMPRA_DISPLAY:
            desvio_color = "🟢"
        elif dev >= UMBRAL_VENTA_DISPLAY:
            desvio_color = "🔴"
        else:
            desvio_color = "🟡"

        if dev < UMBRAL_COMPRA_DISPLAY and clima == "🟢":
            accion = "🚀 COMPRA"
        elif dev >= UMBRAL_VENTA_DISPLAY:
            accion = "🔴 VENTA"
        else:
            accion = "⏳ ESPERAR"

        rows.append({
            "sym": sym, "ccl": ccl, "dev": dev, "clima": clima,
            "desvio_color": desvio_color, "accion": accion,
            "p_ars": p_ars.get(sym, 0),
            "p_usd": p_usd.get(PARES[sym][0], 0),
        })

    # ── KPIs ──────────────────────────────────────────────
    resumen = sim.resumen(p_ars)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CCL Promedio",  f"${ccl_avg:.2f}")
    c2.metric("Capital Total", f"${resumen['capital_total']:,.0f}", f"{resumen['pnl_pct']:+.2f}%")
    c3.metric("Efectivo",      f"${resumen['efectivo']:,.0f}")
    c4.metric("En Posiciones", f"${resumen['en_posiciones']:,.0f}")
    c5.metric("Win Rate",      f"{resumen['win_rate']:.0f}%", f"{resumen['operaciones_total']} ops")

    # ── Estado mercado ─────────────────────────────────────
    if ahora < HORA_APERTURA:
        st.warning(f"⏳ Mercado abre a las {HORA_APERTURA.strftime('%H:%M')} hs")
    elif ahora >= HORA_CIERRE:
        st.error("🔴 16:50 hs — Cierre forzado de posiciones activo")
    elif ahora >= HORA_STOP_COMPRA:
        st.warning("⚠️ 16:30 hs — Sin nuevas compras | Solo cierres")
    else:
        st.success(f"🟢 Mercado abierto | {resumen['posiciones_abiertas']} posiciones abiertas")

    # ── Banner: operador activo ────────────────────────────
    st.info("🤖 **Sistema operado por GitHub Actions** — esta pantalla es solo visualización")

    # ── Gráfico ────────────────────────────────────────────
    rows_sorted = sorted(rows, key=lambda x: x["dev"])
    colors = [
        "#00C851" if r["accion"] == "🚀 COMPRA"
        else "#FF4444" if r["accion"] == "🔴 VENTA"
        else "#888"
        for r in rows_sorted
    ]
    fig = go.Figure(go.Bar(
        x=[r["sym"] for r in rows_sorted],
        y=[r["dev"] for r in rows_sorted],
        marker_color=colors,
        text=[f"{r['dev']:+.2f}%" for r in rows_sorted],
        textposition="outside",
    ))
    fig.add_hline(y=UMBRAL_VENTA_DISPLAY,  line_dash="dash", line_color="#FF4444",
                  annotation_text=f"+{UMBRAL_VENTA_DISPLAY:.2f}%")
    fig.add_hline(y=UMBRAL_COMPRA_DISPLAY, line_dash="dash", line_color="#00C851",
                  annotation_text=f"{UMBRAL_COMPRA_DISPLAY:.2f}%")
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
        "P. ARS": f"${r['p_ars']:,.2f}",
        "P. USD": f"${r['p_usd']:.2f}",
        "CCL":    f"${r['ccl']:,.2f}",
        "Desvío": f"{r['desvio_color']} {r['dev']:+.2f}%",
        "Clima":  r["clima"],
        "Acción": r["accion"],
    } for r in rows_sorted])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Posiciones abiertas ────────────────────────────────
    if any(sim.posiciones.values()):
        st.subheader("💼 Posiciones Abiertas")
        for sym, poss in sim.posiciones.items():
            for pos in poss:
                p_entry = getattr(pos, 'precio_entry', 0)
                m_entry = getattr(pos, 'monto_entry', 0)
                p_id    = getattr(pos, 'id', 'S/N')

                if p_entry <= 0 or m_entry <= 0:
                    st.warning(f"⚠️ Dato inválido en {sym} (ID: {p_id}). Revisar Google Sheets.")
                    continue

                precio_actual = p_ars.get(sym, p_entry)
                pnl     = (precio_actual - p_entry) * pos.cantidad
                pnl_pct = ((precio_actual / p_entry) - 1) * 100
                emoji   = "✅" if pnl >= 0 else "🔻"

                with st.expander(
                    f"{emoji} {p_id} — {sym} | PnL: ${pnl:+,.0f} ({pnl_pct:+.2f}%)",
                    expanded=True
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Entrada",   f"${p_entry:,.1f}")
                    c2.metric("Actual",    f"${precio_actual:,.1f}", f"{pnl_pct:+.2f}%")
                    c3.metric("Invertido", f"${m_entry:,.0f}")
                    c4.metric("Cantidad",  f"{pos.cantidad:.2f} u.")

                    # Única escritura legítima en app.py: venta manual
                    btn_key = f"v_manual_{p_id}_{sym}_{int(p_entry)}"
                    if st.button(f"🔴 Vender {sym} ({p_id})", key=btn_key, type="primary"):
                        op = sim.cerrar_posicion(sym, pos, precio_actual, "VENTA_MANUAL")
                        sim.posiciones[sym] = [p for p in sim.posiciones[sym] if p.id != pos.id]
                        sheets.guardar_operacion(sim.fila_sheets_operacion(op))
                        sheets.guardar_posiciones(sim)
                        sheets.guardar_estado_simulador(sim)
                        st.success(f"✅ Venta registrada para {sym}")
                        time.sleep(1)
                        st.rerun()

    # ── Historial ops ──────────────────────────────────────
    with st.expander("📜 Historial de Operaciones"):
        try:
            ops = sheets.cargar_operaciones()
            if ops:
                cols = ["ID","Activo","Tipo","Cant.","P.Entry","P.Exit",
                        "M.Entry","M.Exit","PnL","PnL%","Apertura","Cierre","Motivo"]
                st.dataframe(
                    pd.DataFrame(ops, columns=cols),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Sin operaciones registradas aún.")
        except Exception as e:
            st.warning(f"⚠️ Error cargando historial (Sheets rate limit): {e}")

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Config")
        st.markdown(f"**Snapshots HMM:** {len(historial)}")
        st.markdown(f"**Actualizado:** {hora.strftime('%H:%M:%S')}")

        barras_cache = st.session_state.get("hmm_barras", {})
        if barras_cache:
            ts_cache = st.session_state.get("hmm_barras_ts")
            ts_str   = ts_cache.strftime("%H:%M:%S") if ts_cache else "—"
            with st.expander(f"📊 Barras HMM ({ts_str})", expanded=False):
                for s, b in sorted(barras_cache.items()):
                    icono = "✅" if len(b) >= 63 else "⚠️"
                    st.caption(f"{icono} {s:<6} {len(b)} barras")

        st.divider()
        st.markdown("**Simulador**")
        st.markdown("Capital inicial: $10.000.000")
        st.markdown("Por operación: 15%")
        st.markdown("Máx./especie: 2")
        st.markdown("Ventana: 10:30 → 16:50")
        st.divider()

        if st.button("🔄 Reset simulador"):
            sim_nuevo = Simulador()
            sheets.guardar_posiciones(sim_nuevo)
            sheets.guardar_estado_simulador(sim_nuevo)
            st.session_state.sim = sim_nuevo
            st.success("✅ Simulador reseteado")
            st.rerun()

    st.caption(f"⏱ Próxima actualización en {REFRESH_SECONDS}s | "
               f"🤖 Trading autónomo vía GitHub Actions")
    time.sleep(REFRESH_SECONDS)
    st.rerun()


if __name__ == "__main__":
    main()
