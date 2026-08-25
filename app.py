"""
GG Swing — Dashboard Streamlit (solo lectura + venta manual)
==============================================================
Visualización en tiempo real del Sistema GG Swing.

RESPONSABILIDADES DE ESTE ARCHIVO:
  - Mostrar precios en vivo del universo de 19 activos (8 Merval vía IOL
    en ARS, 11 CEDEARs vía Alpaca en USD — ver FIX 20-ago-2026)
  - Mostrar posiciones abiertas y KPIs del simulador
  - Mostrar historial de operaciones
  - Permitir venta manual de posiciones (única escritura)

NO HACE TRADING — el único operador es trader_job.py vía GitHub Actions.
Streamlit puede dormir sin consecuencias: el GHA opera igual.

NOTA DE TRANSICIÓN (jun-2026):
Este archivo reemplaza la versión anterior (GG HMM-CCL Trader), que
calculaba desvíos CCL y clima de mercado vía HMM sobre datos de Alpaca.
El Sistema GG Swing pivotó a análisis técnico (HMA-D + SMI + scoring).

FIX (13/jul/2026): VALO se sacó del universo por completo. Universo
queda en 19 activos, mismo criterio que signal_engine.py/trader_job.py.

FIX (20-ago-2026) — mismatch de moneda en CEDEARs:
La versión anterior traía precios de CEDEARs vía iol.get_panel("CEDEARs")
(precio del CEDEAR en ARS). Pero desde que signal_engine.py + alertas.py
están conectados en vivo, las posiciones de CEDEARs se abren con
entry/stop/T1/T2 calculados sobre datos de Alpaca (el ADR en NYSE, en
USD) — no sobre el CEDEAR en pesos. Mostrar el precio ARS del CEDEAR
junto a niveles en USD generaba un PnL sin sentido en pantalla (mezcla
de escalas/monedas). Se corrige trayendo el precio de CEDEARs desde
Alpaca (mismo origen que usó la señal), dejando IOL solo para los 8
Merval. Requiere ALPACA_KEY / ALPACA_SECRET en Streamlit Secrets
(mismos valores que ya están en los GitHub Secrets del repo).
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
from alpaca_client import AlpacaClient
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

# ─── UNIVERSO DE 19 ACTIVOS ───────────────────────────────
# VALO se sacó del universo — ver nota de FIX arriba.
MERVAL_SET = {"GGAL", "YPFD", "PAMP", "BMA", "CEPU", "TGSU2", "SUPV", "BBAR"}
CEDEARS_SET = {"MELI", "NVDA", "TSLA", "MSFT", "PLTR", "VIST", "MU", "AMZN", "IBIT", "META", "AAPL"}
UNIVERSO = MERVAL_SET | CEDEARS_SET


# ── SECRETS ───────────────────────────────────────────────
def get_secrets():
    try:
        return {
            "iol_user":     st.secrets["IOL_USER"],
            "iol_pass":     st.secrets["IOL_PASS"],
            "alpaca_key":   st.secrets["ALPACA_KEY"],
            "alpaca_sec":   st.secrets["ALPACA_SECRET"],
            "gcp":          json.loads(st.secrets["GCP_SERVICE_ACCOUNT"]),
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

        st.session_state.alpaca = AlpacaClient(s["alpaca_key"], s["alpaca_sec"])

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


# ── FETCH PRECIOS ──────────────────────────────────────────
def fetch_precios():
    """
    Trae precios en vivo del universo de 19 activos:
      - Merval (8): vía IOL, en ARS — 1 request (get_panel).
      - CEDEARs (11): vía Alpaca, en USD — 1 request (get_snapshots).
        Mismo origen de datos que usó signal_engine.py para calcular
        entry/stop/T1/T2 de esas posiciones (ver FIX 20-ago-2026).

    Retorna {symbol: {"precio": float, "moneda": "ARS"|"USD"}}.
    """
    iol = st.session_state.iol
    alpaca = st.session_state.alpaca
    precios = {}

    try:
        data = iol.get_panel("MerVal")
        for t in data:
            if t["simbolo"] in MERVAL_SET:
                precios[t["simbolo"]] = {"precio": t.get("ultimoPrecio", 0), "moneda": "ARS"}
    except Exception as e:
        st.warning(f"IOL MerVal: {e}")

    try:
        snaps = alpaca.get_snapshots(list(CEDEARS_SET))
        for sym, snap in snaps.items():
            if sym in CEDEARS_SET and snap.get("last"):
                precios[sym] = {"precio": snap["last"], "moneda": "USD"}
    except Exception as e:
        st.warning(f"Alpaca CEDEARs: {e}")

    return precios


def _precio_num(precios: dict, symbol: str) -> float:
    """Extrae el precio numérico de un símbolo (0 si no hay dato)."""
    return precios.get(symbol, {}).get("precio", 0) or 0


def _precio_fmt(precios: dict, symbol: str) -> str:
    """Formatea el precio con símbolo de moneda correcto (ARS $ / USD u$s)."""
    info = precios.get(symbol)
    if not info or not info.get("precio"):
        return "—"
    simbolo = "u$s" if info["moneda"] == "USD" else "$"
    return f"{simbolo}{info['precio']:,.2f}"


# ── MAIN ──────────────────────────────────────────────────
def main():
    st.title("GG Investments 📊🦅")
    st.caption("IOL (ARS) + Alpaca (USD) | Dashboard — operado por GitHub Actions")

    if not init_state():
        st.error("⚠️ Configurar credenciales en Streamlit Secrets "
                  "(IOL_USER, IOL_PASS, ALPACA_KEY, ALPACA_SECRET, GCP_SERVICE_ACCOUNT).")
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
    precios_num = {sym: _precio_num(precios, sym) for sym in UNIVERSO}

    with col_info:
        ts_str = st.session_state.ultimo_refresh.strftime("%H:%M:%S") if st.session_state.ultimo_refresh else "—"
        st.caption(f"Última actualización: {ts_str} ART | próxima automática en ~{REFRESH_SECONDS // 60} min")

    # ── KPIs ──────────────────────────────────────────────
    # NOTA: capital_total mezcla ARS (Merval + efectivo) y USD (CEDEARs)
    # sin convertir — mismo criterio que el simulador en producción, que
    # tampoco dolariza el capital. Los KPIs son una aproximación cuando
    # hay posiciones CEDEAR abiertas, no un total consolidado real.
    resumen = sim.resumen(precios_num)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Capital Total (aprox.)", f"${resumen['capital_total']:,.0f}", f"{resumen['pnl_pct']:+.2f}%")
    c2.metric("Efectivo", f"${resumen['efectivo']:,.0f}")
    c3.metric("En Posiciones", f"${resumen['en_posiciones']:,.0f}")
    c4.metric("Win Rate", f"{resumen['win_rate']:.0f}%", f"{resumen['operaciones_total']} ops")
    c5.metric("Posiciones Abiertas", f"{resumen['posiciones_abiertas']}")

    # ── Estado mercado ─────────────────────────────────────
    if ahora < HORA_APERTURA:
        st.warning(f"⏳ Mercado abre a las {HORA_APERTURA.strftime('%H:%M')} hs")
    elif ahora >= HORA_CIERRE:
        st.info("🌙 Mercado cerrado — posiciones abiertas se gestionan al reabrir (sin cierre forzado, es swing trading)")
    elif ahora >= HORA_STOP_COMPRA:
        st.warning("⚠️ 16:30 hs — Sin nuevas entradas | Solo gestión de posiciones abiertas")
    else:
        st.success(f"🟢 Mercado abierto | {resumen['posiciones_abiertas']} posiciones abiertas")

    st.info("🤖 **Sistema operado por GitHub Actions** — esta pantalla es solo visualización.")

    # ── Tabla de precios en vivo ───────────────────────────
    st.subheader("📋 Precios en Vivo — Universo de 19 activos")
    if precios:
        filas = []
        for sym in sorted(UNIVERSO):
            mercado = "Merval" if sym in MERVAL_SET else "CEDEAR"
            tiene_pos = sim.tiene_posicion(sym)
            filas.append({
                "Activo": sym,
                "Mercado": mercado,
                "Precio": _precio_fmt(precios, sym),
                "Posición abierta": "🟢 Sí" if tiene_pos else "—",
            })
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
    else:
        st.info("Sin datos de precios todavía. Tocá 'Actualizar ahora'.")

    # ── Posiciones abiertas ────────────────────────────────
    if sim.posiciones:
        st.subheader("💼 Posiciones Abiertas")
        for sym, pos in sim.posiciones.items():
            moneda = "u$s" if sym in CEDEARS_SET else "$"
            precio_actual = precios_num.get(sym) or pos.precio_actual or pos.precio_entry
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
                f"{emoji} {pos.id} — {sym} | PnL: {moneda}{pnl:+,.0f} ({pnl_pct:+.2f}%) | {estado_str}",
                expanded=True
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Entrada", f"{moneda}{pos.precio_entry:,.2f}")
                c2.metric("Actual", f"{moneda}{precio_actual:,.2f}", f"{pnl_pct:+.2f}%")
                c3.metric("Restante", f"{pos.cantidad_restante:.2f} u.")
                c4.metric("Score", f"{pos.score:.0f} ({pos.regimen})")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Stop", f"{moneda}{pos.precio_stop:,.2f}")
                c6.metric("Target 1", f"{moneda}{pos.precio_t1:,.2f}")
                c7.metric("Target 2", f"{moneda}{pos.precio_t2:,.2f}")
                c8.metric("Target 3", f"{moneda}{pos.precio_t3:,.2f}")

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
