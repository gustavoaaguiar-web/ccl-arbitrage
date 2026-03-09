"""
Análisis de Umbrales Óptimos CCL
=================================
Lee CCL_Historial directamente desde Google Sheets y calcula:
- Distribución de desvíos (total, líquidos, ilíquidos)
- Umbral de compra óptimo (en múltiplos de σ)
- Velocidad de reversión
- Umbral de venta óptimo empírico
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from sheets_manager import SheetsManager

st.set_page_config(page_title="Análisis Umbrales", page_icon="📐", layout="wide")
st.title("📐 Análisis de Umbrales Óptimos CCL")

# ── Clasificación de liquidez ─────────────────────────────
LIQUIDOS  = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "MELI"]
ILIQUIDOS = ["GGAL", "YPFD", "PAMP", "CEPU", "BMA", "VIST"]

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

# ── Selector de grupo ─────────────────────────────────────
grupo = st.radio(
    "Grupo de activos para el análisis",
    ["Solo líquidos", "Solo ilíquidos", "Todos"],
    horizontal=True,
    index=0,  # default: líquidos
)

cols_liq  = [c for c in LIQUIDOS  if c in ccl_cols]
cols_iliq = [c for c in ILIQUIDOS if c in ccl_cols]

if grupo == "Solo líquidos":
    cols_analisis = cols_liq
    label_grupo   = f"Líquidos: {', '.join(cols_liq)}"
elif grupo == "Solo ilíquidos":
    cols_analisis = cols_iliq
    label_grupo   = f"Ilíquidos: {', '.join(cols_iliq)}"
else:
    cols_analisis = ccl_cols
    label_grupo   = "Todos los activos"

st.caption(f"📌 {label_grupo}")
st.divider()

# ── Estadísticas generales ────────────────────────────────
st.subheader("📊 Distribución de Desvíos")

serie_total = desvios[cols_analisis].stack().dropna()
sigma = serie_total.std()
media = serie_total.mean()
p5    = serie_total.quantile(0.05)
p95   = serie_total.quantile(0.95)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Snapshots",    f"{n_snapshots}")
c2.metric("Media",        f"{media:+.3f}%")
c3.metric("σ",            f"{sigma:.3f}%")
c4.metric("Percentil 5",  f"{p5:.3f}%")
c5.metric("Percentil 95", f"{p95:.3f}%")

# Histograma
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=serie_total, nbinsx=80,
    marker_color="#4A90D9", opacity=0.8, name=grupo,
))

# Si es "Todos", superponer líquidos en verde para ver la diferencia
if grupo == "Todos" and cols_liq:
    fig_hist.add_trace(go.Histogram(
        x=desvios[cols_liq].stack().dropna(),
        nbinsx=80, marker_color="#00C851", opacity=0.5,
        name="Líquidos (overlay)",
    ))
    fig_hist.update_layout(barmode="overlay")

for u, color, label in [
    (-0.5,         "#888888", "Actual -0.5%"),
    (p5,           "#00C851", f"p5 ({p5:.2f}%)"),
    (-sigma,       "#FFD700", f"-1σ ({-sigma:.2f}%)"),
    (-1.5 * sigma, "#FF8C00", f"-1.5σ ({-1.5*sigma:.2f}%)"),
]:
    fig_hist.add_vline(x=u, line_dash="dash", line_color=color,
                       annotation_text=label, annotation_position="top right")

fig_hist.update_layout(
    title=f"Distribución de desvíos CCL — {grupo}",
    xaxis_title="Desvío (%)", yaxis_title="Frecuencia",
    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    font_color="white", height=360,
)
st.plotly_chart(fig_hist, use_container_width=True)

# σ por símbolo
with st.expander("Ver σ por símbolo"):
    sigma_sym = desvios[cols_analisis].std().sort_values()
    fig_s = go.Figure(go.Bar(
        x=sigma_sym.index,
        y=sigma_sym.values,
        marker_color=["#00C851" if s in LIQUIDOS else "#FF6B6B"
                      for s in sigma_sym.index],
        text=[f"{v:.3f}%" for v in sigma_sym.values],
        textposition="outside",
    ))
    fig_s.update_layout(
        title="σ por símbolo (verde = líquido)",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", height=300,
    )
    st.plotly_chart(fig_s, use_container_width=True)

st.divider()

# ── Tabla de umbrales ─────────────────────────────────────
st.subheader("🎯 Frecuencia de Señales por Umbral de Compra")

total_obs = serie_total.count()
rows_umbral = []
for u in [0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50, 2.00]:
    n           = (serie_total < -u).sum()
    pct         = n / total_obs * 100
    sigma_mult  = u / sigma if sigma > 0 else 0
    señales_dia = n / max(n_snapshots / 390, 1)
    rows_umbral.append({
        "Umbral compra":      f"-{u:.2f}%",
        "Múltiplo σ":         f"{sigma_mult:.2f}σ",
        "Señales totales":    int(n),
        "% del tiempo":       f"{pct:.1f}%",
        "Señales/día (est.)": f"{señales_dia:.1f}",
    })

st.dataframe(pd.DataFrame(rows_umbral), use_container_width=True, hide_index=True)
st.caption("390 ciclos ≈ 1 día de trading completo (10:30–17:00 a 60s/ciclo)")

st.divider()

# ── Velocidad de reversión ────────────────────────────────
st.subheader("⏱ Velocidad de Reversión al Equilibrio")

umbral_analisis = st.slider(
    "Umbral de entrada para análisis de reversión (%)",
    min_value=-3.0, max_value=-0.2,
    value=round(p5, 1), step=0.05,
    format="%.2f%%",
    help="Por defecto en el percentil 5 de tu distribución actual",
)

resultados = []
for sym in cols_analisis:
    serie = desvios[sym].dropna().values
    for i in range(len(serie) - 30):
        if serie[i] < umbral_analisis:
            dev_max_pos = 0.0
            revertio    = False
            for j in range(1, 31):
                if serie[i + j] > dev_max_pos:
                    dev_max_pos = serie[i + j]
                if serie[i + j] >= 0:
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
    df_r        = pd.DataFrame(resultados)
    total_casos = len(df_r)
    revirtieron = df_r["revirtió"].sum()
    pct_rev     = revirtieron / total_casos * 100 if total_casos else 0
    p25_pico    = df_r["dev_pico"].quantile(0.25)
    p50_pico    = df_r["dev_pico"].quantile(0.50)
    p75_pico    = df_r["dev_pico"].quantile(0.75)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Casos analizados",  total_casos)
    c2.metric("Revirtieron a 0%",  f"{revirtieron} ({pct_rev:.0f}%)")
    c3.metric("Ciclos promedio",
              f"{df_r[df_r['revirtió']]['ciclos'].mean():.1f}" if revirtieron else "—")
    c4.metric("Dev pico promedio", f"{df_r['dev_pico'].mean():+.3f}%")

    st.markdown("**Desvío máximo alcanzado tras la entrada** — calibra el umbral de venta óptimo")
    fig_picos = go.Figure()
    fig_picos.add_trace(go.Histogram(
        x=df_r["dev_pico"], nbinsx=50,
        marker_color="#FF6B6B", opacity=0.8, name="Dev pico",
    ))
    for v, color, label in [
        (p25_pico, "#FFD700", f"p25 ({p25_pico:.2f}%)"),
        (p50_pico, "#FF8C00", f"p50 ({p50_pico:.2f}%)"),
        (p75_pico, "#FF4444", f"p75 ({p75_pico:.2f}%)"),
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
    st.info("No hay suficientes eventos con ese umbral. Bajá el umbral con el slider.")

st.divider()

# ── Recomendación final ───────────────────────────────────
st.subheader("💡 Recomendación basada en tus datos")

if resultados and len(df_r) >= 5:
    rec_compra_1s  = round(-sigma, 2)
    rec_compra_15s = round(-1.5 * sigma, 2)
    rec_venta_cons = round(max(p25_pico, 0.10), 2)
    rec_venta_bal  = round(max(p50_pico, 0.20), 2)
    stop_loss      = round(umbral_analisis * 1.5, 2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Umbral de compra sugerido**")
        st.markdown(f"- Moderado (-1σ):    `{rec_compra_1s:.2f}%`")
        st.markdown(f"- Selectivo (-1.5σ): `{rec_compra_15s:.2f}%`")
        st.markdown(f"- Actual en código:  `-0.50%`")
    with col2:
        st.markdown("**Umbrales de venta sugeridos**")
        st.markdown(f"- Salida A (p25 pico): `+{rec_venta_cons:.2f}%`")
        st.markdown(f"- Salida original (p50 pico): `+{rec_venta_bal:.2f}%`")
        st.markdown(f"- Stop loss sugerido: `{stop_loss:.2f}%`")

    st.info(
        f"📌 Con {grupo.lower()}, σ = {sigma:.3f}%. "
        f"Umbral de compra óptimo entre {rec_compra_1s:.2f}% y {rec_compra_15s:.2f}%. "
        f"La mayoría de las reversiones alcanzan un pico de {p50_pico:.2f}% — "
        f"usá ese valor como Salida A."
    )
else:
    st.info("Necesitás más datos para una recomendación confiable. Volvé después de 1–2 días de trading.")

st.caption("Los umbrales óptimos cambian con el régimen de mercado. Revisá semanalmente.")
