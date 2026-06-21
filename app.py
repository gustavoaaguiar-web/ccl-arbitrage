"""
GG Swing — Dashboard Streamlit (solo lectura + venta manual)
==============================================================
Visualización en tiempo real del Sistema GG Swing.

RESPONSABILIDADES DE ESTE ARCHIVO:
  - Mostrar precios en vivo del universo de 20 activos (Merval + CEDEARs, vía IOL)
  - Mostrar posiciones abiertas y KPIs del simulador
  - Mostrar historial de operaciones
  - Permitir venta manual de posiciones (única escritura)

NO HACE TRADING — el único operador es trader_job.py vía GitHub Actions.
Streamlit puede dormir sin consecuencias: el GHA opera igual.

NOTA DE TRANSICIÓN (jun-2026):
Este archivo reemplaza la versión anterior (GG HMM-CCL Trader), que
calculaba desvíos CCL y clima de mercado vía HMM sobre datos de Alpaca.
El Sistema GG Swing pivotó a análisis técnico (HMA-D + SMI + scoring)
sobre precio ARS directo. Esta versión del dashboard todavía NO muestra
señales/score — eso depende de signal_engine.py, aún no construido.
Por ahora solo expone precios en vivo, posiciones, KPIs e historial.

NO USA ALPACA — el panel de precios funciona enteramente con IOL
(get_panel("MerVal") + get_panel("CEDEARs")), igual que trader_job.py.
"""

import time
import json
import logging
import streamlit as st
import pandas as pd
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")


def hora_argentina():
    return datetime.now(TZ_ARG)


from iol_client import IOLClient
from simulator import Simulador
from sheets_manager import SheetsManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="GG Swing", page_icon="📈", layout="wide")

# ─── REFRESH ──────────────────────────────────────────────
REFRESH_SECONDS = 300   # 5 minutos — cuida la cuota free tier de IOL

# ─── HORARIOS (deben coincidir con simulator.py) ─────────
HORA_APERTURA    = dtime(10, 30)
HORA_STOP_COMPRA = dtime(16, 30)
HORA_CIERRE      = dtime(16, 50)

# ─── UNIVERSO DE 20 ACTIVOS ───────────────────────────────
MERVAL_SET = {"GGAL", "YPFD", "PAMP", "BMA", "CEPU", "TGSU2", "SUPV", "BBAR", "VALO"}
CEDEARS_SET = {"MELI", "NVDA", "TSLA", "MSFT", "PLTR", "VIST", "MU", "AMZN", "IBIT", "META", "AAPL"}
UNIVERSO = MERVAL_SET | CEDEARS_SET


# ── SECRETS ───────────────────────────────────────────────
def get_secrets():
    try:
        return {
            "iol_user": st.secrets["IOL_USER"],
            "iol_pass": st.secrets["IOL_PASS"],
            "gcp":      json.loads(st.secrets["GCP_SERVICE_ACCOUNT"]),
        }
    except Exception:
        return None


# ── SESSION STATE ──────────────────────────────────────────
def init_state():
    s = get_secrets()
    if not s:
        return False
    if "ready" not in st.session_state:
        st.session_state.iol = IOLClient(s["iol_user"], s["iol_pass"])
        st.session_state.iol.login()

        sh = SheetsManager(s["gcp"])
        sh.conectar()
        st.session_state.sheets = sh

        sim = Simulador()
        sh.cargar_estado_simulador(sim)
        sh.cargar_posiciones(sim)
        st.session_state.sim = sim

        st.session_state.ultimo_refresh = None
        st.session_state.ready = True
    return True


# ── FETCH PRECIOS (solo display, vía IOL) ─────────────────
def fetch_precios():
    """
    Trae precios ARS en vivo del universo de 20 activos.
    Mismo patrón que trader_job.py: 2 requests (get_panel x2), sin
    pedir cotización individual por símbolo (evita romper rate limit).
    """
    iol = st.session_state.iol
    precios = {}

    try:
        data = iol.get_panel("MerVal")
        for t in data:
            if t["simbolo"] in MERVAL_SET:
                precios[t["simbolo"]] = t.get("ultimoPrecio", 0)
    except Exception as e:
        st.warning(f"IOL MerVal: {e}")

    try:
        data = iol.get_panel("CEDEARs")
        for t in data:
            if t["simbolo"] in CEDEARS_SET:
                precios[t["simbolo"]] = t.get("ultimoPrecio", 0)
    except Exception as e:
        st.warning(f"IOL CEDEARs: {e}")

    return precios


# ── MAIN ──────────────────────────────────────────────────
def main():
    st.title("📈 GG Investments — Sistema GG Swing")
    st.caption("IOL (ARS) | Dashboard — operado por GitHub Actions")

    if not init_state():
        st.error("⚠️ Configurar credenciales en Streamlit Secrets (IOL_USER, IOL_PASS, GCP_SERVICE_ACCOUNT).")
        return

    sheets = st.session_state.sheets
    hora = hora_argentina()
    ahora = hora.time()

    necesita_refresh = (
        st.session_state.ultimo_refresh is None
        or (hora - st.session_state.ultimo_refresh).total_seconds() >= REFRESH_SECONDS
    )

    col_refresh, col_info = st.columns([1, 4])
    with col_refresh:
        refrescar_manual = st.button("🔄 Actualizar ahora")

    if necesita_refresh or refrescar_manual:
        # Refrescar estado del simulador desde Sheets (refleja lo que
        # operó el GHA mientras el dashboard estaba cerrado/dormido)
        try:
            sim_nuevo = Simulador()
            sheets.cargar_estado_simulador(sim_nuevo)
            sheets.cargar_posiciones(sim_nuevo)
            st.session_state.sim = sim_nuevo
        except Exception as e:
            logger.warning(f"No se pudo refrescar sim desde Sheets: {e}")

        st.session_state.precios = fetch_precios()
        st.session_state.ultimo_refresh = hora

    sim = st.session_state.sim
    precios = st.session_state.get("precios", {})

    with col_info:
        ts_str = st.session_state.ultimo_refresh.strftime("%H:%M:%S") if st.session_state.ultimo_refresh else "—"
        st.caption(f"Última actualización: {ts_str} ART | próxima automática en ~{REFRESH_SECONDS // 60} min")

    # ── KPIs ──────────────────────────────────────────────
    resumen = sim.resumen(precios)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Capital Total", f"${resumen['capital_total']:,.0f}", f"{resumen['pnl_pct']:+.2f}%")
    c2.metric("Efectivo", f"${resumen['efectivo']:,.0f}")
    c3.metric("En Posiciones", f"${resumen['en_posiciones']:,.0f}")
    c4.metric("Win Rate", f"{resumen['win_rate']:.0f}%", f"{resumen['operaciones_total']} ops")
    c5.metric("Posiciones Abiertas", f"{resumen['posiciones_abiertas']}")

    # ── Estado mercado ─────────────────────────────────────
    if ahora < HORA_APERTURA:
        st.warning(f"⏳ Mercado abre a las {HORA_APERTURA.strftime('%H:%M')} hs")
    elif ahora >= HORA_CIERRE:
        st.error("🔴 16:50 hs — Cierre forzado de posiciones activo")
    elif ahora >= HORA_STOP_COMPRA:
        st.warning("⚠️ 16:30 hs — Sin nuevas entradas | Solo cierres")
    else:
        st.success(f"🟢 Mercado abierto | {resumen['posiciones_abiertas']} posiciones abiertas")

    st.info("🤖 **Sistema operado por GitHub Actions** — esta pantalla es solo visualización. "
            "Las señales de entrada (score, HMA, SMI) todavía no están integradas aquí.")

    # ── Tabla de precios en vivo ───────────────────────────
    st.subheader("📋 Precios en Vivo — Universo de 20 activos")
    if precios:
        filas = []
        for sym in sorted(UNIVERSO):
            mercado = "Merval" if sym in MERVAL_SET else "CEDEAR"
            tiene_pos = sim.tiene_posicion(sym)
            filas.append({
                "Activo": sym,
                "Mercado": mercado,
                "Precio ARS": f"${precios.get(sym, 0):,.2f}" if precios.get(sym) else "—",
                "Posición abierta": "🟢 Sí" if tiene_pos else "—",
            })
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
    else:
        st.info("Sin datos de precios todavía. Tocá 'Actualizar ahora'.")

    # ── Posiciones abiertas ────────────────────────────────
    if sim.posiciones:
        st.subheader("💼 Posiciones Abiertas")
        for sym, pos in sim.posiciones.items():
            precio_actual = precios.get(sym, pos.precio_actual or pos.precio_entry)
            pnl = (precio_actual - pos.precio_entry) * pos.cantidad_restante
            pnl_pct = ((precio_actual / pos.precio_entry) - 1) * 100 if pos.precio_entry else 0
            emoji = "✅" if pnl >= 0 else "🔻"

            estado_partes = []
            if pos.t1_alcanzado:
                estado_partes.append("T1 ✅")
            if pos.t2_alcanzado:
                estado_partes.append("T2 ✅")
            if pos.stop_en_breakeven:
                estado_partes.append("Stop en breakeven")
            estado_str = " | ".join(estado_partes) if estado_partes else "Sin targets alcanzados"

            with st.expander(
                f"{emoji} {pos.id} — {sym} | PnL: ${pnl:+,.0f} ({pnl_pct:+.2f}%) | {estado_str}",
                expanded=True
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Entrada", f"${pos.precio_entry:,.2f}")
                c2.metric("Actual", f"${precio_actual:,.2f}", f"{pnl_pct:+.2f}%")
                c3.metric("Restante", f"{pos.cantidad_restante:.2f} u.")
                c4.metric("Score", f"{pos.score:.0f}")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Stop", f"${pos.precio_stop:,.2f}")
                c6.metric("Target 1", f"${pos.precio_t1:,.2f}")
                c7.metric("Target 2", f"${pos.precio_t2:,.2f}")
                c8.metric("Target 3", f"${pos.precio_t3:,.2f}")

                btn_key = f"v_manual_{pos.id}_{sym}"
                if st.button(f"🔴 Vender {sym} ({pos.id}) — cierre total", key=btn_key, type="primary"):
                    op = sim._registrar_cierre(
                        pos, pos.cantidad_restante, precio_actual, "VENTA_MANUAL", "CIERRE_FINAL"
                    )
                    del sim.posiciones[sym]
                    sheets.guardar_operacion(sim.fila_sheets_operacion(op))
                    sheets.guardar_posiciones(sim)
                    sheets.guardar_estado_simulador(sim)
                    st.success(f"✅ Venta registrada para {sym}")
                    time.sleep(1)
                    st.rerun()
    else:
        st.caption("Sin posiciones abiertas actualmente.")

    # ── Historial ops ──────────────────────────────────────
    with st.expander("📜 Historial de Operaciones"):
        try:
            ops = sheets.cargar_operaciones()
            if ops:
                cols = ["ID", "Activo", "Tipo", "Cant.", "P.Entry", "P.Exit",
                        "M.Entry", "M.Exit", "PnL", "PnL%", "Apertura", "Cierre", "Motivo"]
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
        st.title("⚙️ Config — Sistema GG Swing")
        st.markdown(f"**Actualizado:** {hora.strftime('%H:%M:%S')} ART")
        st.markdown(f"**Refresh automático:** cada {REFRESH_SECONDS // 60} min")
        st.divider()
        st.markdown("**Simulador**")
        st.markdown(f"Capital inicial: ${sim.capital_inicial:,.0f}")
        st.markdown("Sizing: por riesgo (0.5–1.5% según score)")
        st.markdown("Máx. posiciones/símbolo: 1")
        st.markdown("Ventana: 10:30 → 16:50 ART")
        st.markdown("Salida: T1 (40%) → T2 (40%) → trailing (20%)")
        st.divider()

        if st.button("🔄 Reset simulador"):
            sim_nuevo = Simulador()
            sheets.guardar_posiciones(sim_nuevo)
            sheets.guardar_estado_simulador(sim_nuevo)
            st.session_state.sim = sim_nuevo
            st.success("✅ Simulador reseteado")
            st.rerun()

    st.caption(f"⏱ Próxima actualización automática en ~{REFRESH_SECONDS // 60} min | "
               f"🤖 Trading autónomo vía GitHub Actions")


if __name__ == "__main__":
    main()
