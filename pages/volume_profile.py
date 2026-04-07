"""
pages/volume_profile.py — Dashboard de Volume Profile (POC Analysis)
Parte del sistema GG Investments — CCL Arbitrage

Muestra:
1. Intraday (hoy): histograma de volumen, POC actual, distancia CCL
2. 2 Semanas: patrón recurrente de niveles de precio
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import sys
sys.path.insert(0, '.')

from src.sheets_manager import SheetsManager
from src.analytics import (
    calcular_volume_profile,
    graficar_volume_profile,
    calcular_distancia_al_poc
)

TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")


def get_sheets() -> SheetsManager:
    """Reutiliza SheetsManager de session_state si ya fue inicializado por app.py."""
    if "sheets" in st.session_state and st.session_state.sheets is not None:
        return st.session_state.sheets
    return SheetsManager()


def cargar_precios_actual(sheets: SheetsManager):
    """
    Carga precios actual (CCL) desde Sheets.
    Retorna dict con precio actual y símbolo.
    """
    try:
        historial = sheets.cargar_ccl_historial(dias=1)
        if historial:
            return historial[0]
        return None
    except Exception as e:
        st.error(f"Error cargando precios: {e}")
        return None


def preparar_cache_vp():
    """Inicializa session_state para Volume Profile cache."""
    if "vp_cache" not in st.session_state:
        st.session_state.vp_cache = {
            "timestamp": None,
            "intraday": None,
            "dos_semanas": None
        }


def verificar_cache_valido(cache_timestamp) -> bool:
    """Verifica si el cache es válido (menos de 10 min)."""
    if not cache_timestamp:
        return False
    ahora = datetime.now(TZ_ARG)
    delta = (ahora - cache_timestamp).total_seconds()
    return delta < 600  # 10 minutos


def main():
    st.set_page_config(page_title="📊 Volume Profile", layout="wide")
    st.title("📊 Volume Profile — POC Analysis")

    preparar_cache_vp()

    try:
        sheets = get_sheets()
    except Exception as e:
        st.error(f"Error inicializando SheetsManager: {e}")
        st.stop()

    ahora = datetime.now(TZ_ARG)

    # ============================================================================
    # SECCIÓN: INTRADAY (HOY)
    # ============================================================================
    st.subheader("📈 Intraday (Hoy)")

    col_info1, col_info2, col_refresh1 = st.columns([1, 1, 0.5])

    with col_refresh1:
        if st.button("🔄 Refrescar Intraday", key="refresh_intraday"):
            st.session_state.vp_cache["timestamp"] = None

    vp_intraday = None
    cache_valido = verificar_cache_valido(st.session_state.vp_cache["timestamp"])

    if cache_valido and st.session_state.vp_cache["intraday"]:
        vp_intraday = st.session_state.vp_cache["intraday"]
        estado_cache = "✅ (desde cache, 10 min)"
    else:
        with st.spinner("Cargando histórico intraday..."):
            try:
                historial_hoy = sheets.cargar_ccl_historial(dias=1)
                vp_intraday = calcular_volume_profile(historial_hoy, bins=0.0001)
                st.session_state.vp_cache["timestamp"] = ahora
                st.session_state.vp_cache["intraday"] = vp_intraday
                estado_cache = "🆕 (recalculado)"
            except Exception as e:
                st.error(f"Error cargando intraday: {e}")
                vp_intraday = None
                estado_cache = "❌ Error"

    if vp_intraday and vp_intraday['es_valido']:
        poc = vp_intraday['poc']
        precio_actual = cargar_precios_actual(sheets)
        ccl_actual = precio_actual['ccl'] if precio_actual else None

        with col_info1:
            st.metric("Point of Control (POC)", f"{poc:.6f}")

        with col_info2:
            if ccl_actual:
                st.metric("CCL Actual", f"{ccl_actual:.6f}")
                dist = calcular_distancia_al_poc(ccl_actual, poc)
                if dist:
                    color = "🟢" if dist['en_poc'] else "🟡" if abs(dist['distancia_pct']) < 0.02 else "🔵"
                    st.metric("Distancia al POC", f"{dist['distancia_pct']:.3f}% ({color} {dist['estado']})")

        st.caption(estado_cache)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Snapshots", vp_intraday['total_snapshots'])
        with col2:
            st.metric("Rango Mín", f"{vp_intraday['min_nivel']:.6f}")
        with col3:
            st.metric("Rango Máx", f"{vp_intraday['max_nivel']:.6f}")
        with col4:
            rango = vp_intraday['max_nivel'] - vp_intraday['min_nivel']
            st.metric("Rango Amplitud", f"{rango:.6f} ({rango*100:.4f}%)")

        st.subheader("Histograma de Volumen — Intraday")
        fig_intraday = graficar_volume_profile(
            vp_intraday,
            ccl_actual if ccl_actual else poc,
            titulo="Volume Profile Intraday — POC vs CCL Actual"
        )
        if fig_intraday:
            st.plotly_chart(fig_intraday, use_container_width=True)
        else:
            st.warning("No se pudo generar gráfico")
    else:
        st.warning("⚠️ Datos insuficientes para intraday (mín. 10 snapshots)")

    st.divider()

    # ============================================================================
    # SECCIÓN: 2 SEMANAS (PATRÓN RECURRENTE)
    # ============================================================================
    st.subheader("📊 Patrón 2 Semanas")

    col_info3, col_info4, col_refresh2 = st.columns([1, 1, 0.5])

    with col_refresh2:
        if st.button("🔄 Refrescar 2 Semanas", key="refresh_2sem"):
            st.session_state.vp_cache["timestamp"] = None

    vp_2sem = None

    if cache_valido and st.session_state.vp_cache["dos_semanas"]:
        vp_2sem = st.session_state.vp_cache["dos_semanas"]
        estado_cache2 = "✅ (desde cache, 10 min)"
    else:
        with st.spinner("Cargando histórico 2 semanas..."):
            try:
                historial_2sem = sheets.cargar_ccl_historial(dias=14)
                vp_2sem = calcular_volume_profile(historial_2sem, bins=0.0001)
                if st.session_state.vp_cache["timestamp"] is None:
                    st.session_state.vp_cache["timestamp"] = ahora
                st.session_state.vp_cache["dos_semanas"] = vp_2sem
                estado_cache2 = "🆕 (recalculado)"
            except Exception as e:
                st.error(f"Error cargando 2 semanas: {e}")
                vp_2sem = None
                estado_cache2 = "❌ Error"

    if vp_2sem and vp_2sem['es_valido']:
        poc_2sem = vp_2sem['poc']

        with col_info3:
            st.metric("POC 2 Semanas", f"{poc_2sem:.6f}")

        with col_info4:
            rango_2sem = vp_2sem['max_nivel'] - vp_2sem['min_nivel']
            st.metric("Rango Histórico", f"{vp_2sem['min_nivel']:.6f} — {vp_2sem['max_nivel']:.6f}")

        st.caption(estado_cache2)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Snapshots (2sem)", vp_2sem['total_snapshots'])
        with col2:
            distancia_a_min = ((vp_2sem['min_nivel'] - poc_2sem) / poc_2sem) * 100
            st.metric("Distancia a Mín", f"{distancia_a_min:.3f}%")
        with col3:
            distancia_a_max = ((vp_2sem['max_nivel'] - poc_2sem) / poc_2sem) * 100
            st.metric("Distancia a Máx", f"{distancia_a_max:.3f}%")
        with col4:
            amplitud_pct = (rango_2sem / poc_2sem) * 100
            st.metric("Amplitud %", f"{amplitud_pct:.4f}%")

        st.subheader("Histograma de Volumen — 2 Semanas")
        fig_2sem = graficar_volume_profile(
            vp_2sem,
            vp_2sem['poc'],
            titulo="Volume Profile 2 Semanas — Patrón Recurrente"
        )
        if fig_2sem:
            st.plotly_chart(fig_2sem, use_container_width=True)
    else:
        st.warning("⚠️ Datos insuficientes para 2 semanas (mín. 10 snapshots)")

    st.divider()

    # ============================================================================
    # SECCIÓN: COMPARATIVA
    # ============================================================================
    if vp_intraday and vp_intraday['es_valido'] and vp_2sem and vp_2sem['es_valido']:
        st.subheader("🔍 Comparativa: Intraday vs 2 Semanas")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**POC Intraday**")
            st.code(f"{vp_intraday['poc']:.6f}")

        with col2:
            st.write("**POC 2 Semanas**")
            st.code(f"{vp_2sem['poc']:.6f}")

        with col3:
            diferencia = vp_intraday['poc'] - vp_2sem['poc']
            diferencia_pct = (diferencia / vp_2sem['poc']) * 100
            color_diff = "🟢" if diferencia_pct > 0 else "🔴"
            st.write("**Diferencia (hoy vs 2sem)**")
            st.code(f"{diferencia:.6f} ({diferencia_pct:+.3f}%) {color_diff}")

        st.write("#### Interpretación:")

        pocintras_vs_poc2sem = vp_intraday['poc'] - vp_2sem['poc']
        if pocintras_vs_poc2sem > 0:
            st.info("✅ El POC intraday está **ARRIBA** del patrón histórico (2 semanas). "
                    "El CCL se está moviendo hacia precios más altos.")
        elif pocintras_vs_poc2sem < 0:
            st.warning("⚠️ El POC intraday está **ABAJO** del patrón histórico (2 semanas). "
                       "El CCL se está moviendo hacia precios más bajos.")
        else:
            st.success("🟢 El POC intraday coincide con el patrón histórico.")

        if vp_intraday['max_nivel'] > vp_2sem['max_nivel']:
            st.warning("⚠️ Hoy se alcanzaron máximos no vistos en las últimas 2 semanas.")

        if vp_intraday['min_nivel'] < vp_2sem['min_nivel']:
            st.warning("⚠️ Hoy se alcanzaron mínimos no vistos en las últimas 2 semanas.")


if __name__ == "__main__":
    main()
