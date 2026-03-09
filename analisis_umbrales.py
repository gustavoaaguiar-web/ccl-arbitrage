"""
Análisis de Umbrales Óptimos CCL
=================================
Lee CCL_Historial directamente desde Google Sheets y calcula:
- Distribución de desvíos
- Umbral de compra óptimo (en múltiplos de σ)
- Velocidad de reversión
- Umbral de venta óptimo empírico
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from sheets_manager import SheetsManager

st.set_page_config(page_title="Análisis Umbrales", page_icon="📐", layout="wide")
st.title("📐 Análisis de Umbrales Óptimos CCL")

# ── Conexión Sheets ───────────────────────────────────────
@st.cache_resource
def conectar_sheets():
    try:
        gcp = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
        sh  = SheetsManager(gcp)
        sh.conectar()
        return sh
    except Exception as e:
        st.error(f"Error conectando Sheets: {e}")
        return None

@st.cache_data(ttl=300)
def cargar_desvios():
    sh = conectar_sheets()
    if not sh:
        return None, None
    ws = sh._hojas.get("CCL_Historial")
    if not ws:
        return None, None
    filas = ws.get_all_values()
    if len(filas) < 10:
        return None, None
    df = pd.DataFrame(filas[1:], columns=filas[0])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    ccl_cols = df.columns[2:]
    for col in ccl_cols:
        df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")
    df = df.dropna(subset=ccl_cols, how="all")
    mediana = df[ccl_cols].median(axis=1)
    desvios = (df[ccl_cols].div(mediana, axis=0) - 1) * 100
    desvios["timestamp"] = df["timestamp"].values
    return desvios, ccl_cols.tolist()

# ── Cargar datos ──────────────────────────────────────────
with st.spinner("Cargando historial desde Sheets..."):
    desvios, ccl_cols = cargar_desvios()

if desvios is None:
    st.warning("⏳ Insuficientes snapshots. Volvé después de algunas horas de operación.")
    st.stop()

n_snapshots = len(desvios)
st.success(f"✅ {n_snapshots} snapshots cargados")

if st.button("🔄 Actualizar datos"):
    st.cache_data.clear()
    st.rerun()

st.divider()

# ── Estadísticas generales ────────────────────────────────
st.subheader("📊 Distribución de Desvíos")

serie_total = desvios[ccl_cols].stack().dropna()
sigma  = serie_total.std()
media  = serie_total.mean()
p5     = serie_total.quantile(0.05)
p95    = serie_total.quantile(0.95)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Snapshots",   f"{n_snapshots}")
c2.metric("Media",       f"{media:+.3f}%")
c3.metric("σ global",    f"{sigma:.3f}%")
c4.metric("Percentil 5", f"{p5:.3f}%")
c5.metric("Percentil 95",f"{p95:.3f}%")

# Histograma
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=serie_total,
    nbinsx=80,
    marker_color="#4A90D9",
    opacity=0.8,
    name="Desvíos",
))
for u, color, label in [
    (-0.5, "#00C851", "Umbral actual -0.5%"),
    (-sigma, "#FFD700", f"-1σ ({-sigma:.2f}%)"),
    (-1.5*sigma, "#FF8C00", f"-1.5σ ({-1.5*sigma:.2f}%)"),
]:
    fig_hist.add_vline(x=u, line_dash="dash", line_color=color,
                       annotation_text=label, annotation_position="top right")
fig_hist.update_layout(
    title="Distribución de todos los desvíos CCL",
    xaxis_title="Desvío (%)", yaxis_title="Frecuencia",
    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    font_color="white", height=350,
)
st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ── Tabla de umbrales ─────────────────────────────────────
st.subheader("🎯 Frecuencia de Señales por Umbral de Compra")

total_obs = serie_total.count()
rows_umbral = []
for u in [0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50]:
    n     = (serie_total < -u).sum()
    pct   = n / total_obs * 100
    sigma_mult = u / sigma if sigma > 0 else 0
    rows_umbral.append({
        "Umbral compra": f"-{u:.2f}%",
        "Múltiplo σ":    f"{sigma_mult:.2f}σ",
        "Señales":        int(n),
        "% del tiempo":  f"{pct:.1f}%",
        "Señales/día (est.)": f"{n / max(n_snapshots/390, 1):.1f}",
    })

df_u = pd.DataFrame(rows_umbral)
st.dataframe(df_u, use_container_width=True, hide_index=True)

st.caption("390 ciclos ≈ 1 día de trading completo (10:30–17:00 a 60s/ciclo)")

st.divider()

# ── Velocidad de reversión ────────────────────────────────
st.subheader("⏱ Velocidad de Reversión al Equilibrio")

umbral_analisis = st.slider(
    "Umbral de entrada para análisis de reversión (%)",
    min_value=-2.0, max_value=-0.2,
    value=-0.5, step=0.05,
    format="%.2f%%"
)

resultados = []
for sym in ccl_cols:
    serie = desvios[sym].dropna().values
    for i in range(len(serie) - 30):
        if serie[i] < umbral_analisis:
            dev_max_pos = 0.0
            revertio    = False
            for j in range(1, 31):
                if serie[i+j] > dev_max_pos:
                    dev_max_pos = serie[i+j]
                if serie[i+j] >= 0:
                    resultados.append({
                        "sym":         sym,
                        "dev_entrada": round(serie[i], 3),
                        "ciclos":      j,
                        "dev_pico":    round(dev_max_pos, 3),
                        "revirtió":    True,
                    })
                    revertio = True
                    break
            if not revertio:
                resultados.append({
                    "sym":         sym,
                    "dev_entrada": round(serie[i], 3),
                    "ciclos":      30,
                    "dev_pico":    round(dev_max_pos, 3),
                    "revirtió":    False,
                })

if resultados:
    df_r = pd.DataFrame(resultados)
    total_casos  = len(df_r)
    revirtieron  = df_r["revirtió"].sum()
    pct_reversion = revirtieron / total_casos * 100 if total_casos else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Casos analizados",   total_casos)
    c2.metric("Reversiones a 0%",   f"{revirtieron} ({pct_reversion:.0f}%)")
    c3.metric("Ciclos promedio",     f"{df_r[df_r['revirtió']]['ciclos'].mean():.1f}" if revirtieron else "—")
    c4.metric("Dev pico promedio",   f"{df_r['dev_pico'].mean():+.3f}%")

    # Distribución del pico alcanzado → calibra umbral de venta
    st.markdown("**Distribución del desvío máximo alcanzado tras la entrada** — calibra el umbral de venta")
    picos = df_r["dev_pico"]
    fig_picos = go.Figure()
    fig_picos.add_trace(go.Histogram(
        x=picos, nbinsx=50,
        marker_color="#FF6B6B", opacity=0.8, name="Dev pico"
    ))
    for v, color, label in [
        (0.35, "#FFD700", "Salida A 0.35%"),
        (0.50, "#FF4444", "Salida original 0.50%"),
    ]:
        fig_picos.add_vline(x=v, line_dash="dash", line_color=color,
                            annotation_text=label)
    fig_picos.update_layout(
        title="¿Hasta dónde llega el desvío después de entrar?",
        xaxis_title="Desvío máximo alcanzado (%)",
        yaxis_title="Frecuencia",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", height=320,
    )
    st.plotly_chart(fig_picos, use_container_width=True)

    # Tabla por símbolo
    with st.expander("Ver detalle por símbolo"):
        resumen_sym = df_r.groupby("sym").agg(
            casos=("dev_entrada", "count"),
            pct_reversion=("revirtió", lambda x: f"{x.mean()*100:.0f}%"),
            ciclos_prom=("ciclos", "mean"),
            dev_pico_prom=("dev_pico", "mean"),
        ).round(2)
        resumen_sym.columns = ["Casos", "% Reversión", "Ciclos prom.", "Dev pico prom."]
        st.dataframe(resumen_sym, use_container_width=True)
else:
    st.info("No hay suficientes eventos con ese umbral. Probá un umbral menos restrictivo.")

st.divider()

# ── Recomendación final ───────────────────────────────────
st.subheader("💡 Recomendación basada en tus datos")

if resultados and len(df_r) >= 5:
    pico_p50 = df_r["dev_pico"].quantile(0.50)
    pico_p25 = df_r["dev_pico"].quantile(0.25)
    rec_venta_conservadora = round(max(pico_p25, 0.10), 2)
    rec_venta_agresiva     = round(max(pico_p50, 0.25), 2)
    rec_compra_1s          = round(-sigma, 2)
    rec_compra_15s         = round(-1.5 * sigma, 2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Umbral de compra sugerido**")
        st.markdown(f"- Moderado (-1σ):  `{rec_compra_1s:.2f}%`")
        st.markdown(f"- Selectivo (-1.5σ): `{rec_compra_15s:.2f}%`")
        st.markdown(f"- Actual: `-0.50%`")
    with col2:
        st.markdown("**Umbral de venta sugerido (Salida A)**")
        st.markdown(f"- Conservador (p25 del pico): `+{rec_venta_conservadora:.2f}%`")
        st.markdown(f"- Balanceado (p50 del pico):  `+{rec_venta_agresiva:.2f}%`")
        st.markdown(f"- Actual Salida A: `+0.35%`")
else:
    st.info("Necesitás más datos para una recomendación confiable. Volvé después de 1–2 días de trading.")

st.caption("Los umbrales óptimos cambian con el régimen de mercado. Revisá semanalmente.")
  
