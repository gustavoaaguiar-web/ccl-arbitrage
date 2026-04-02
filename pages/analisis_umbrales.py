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
LIQUIDOS  = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "META", "AMZN", "MELI", "GLD", "IBIT", "SPY"]
ILIQUIDOS = ["GGAL", "YPFD", "PAMP", "CEPU", "BMA", "TGSU2"]

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
def cargar_precios():
    """Devuelve los precios CCL crudos (sin calcular desvíos todavía)."""
    sh = conectar_sheets()
    if not sh:
        return None, None
    ws = sh._hojas.get("CCL_Historial")
    if not ws:
        return None, None
    filas = ws.get_all_values()
    if len(filas) < 10:
        return None, None

    headers = filas[0]
    df_raw  = pd.DataFrame(filas[1:], columns=headers)

    # Columnas CCL: todo lo que esté a partir de la posición 2
    ccl_cols = [c for c in headers[2:] if c and c != "timestamp"]

    # Usar iloc para evitar el AttributeError con nombres de columna duplicados.
    # df[col] con nombre duplicado devuelve un DataFrame en vez de Series → .str explota.
    # .iloc[:, idx] siempre devuelve una Series sin importar duplicados.
    df_out = pd.DataFrame()
    df_out["timestamp"] = pd.to_datetime(df_raw.iloc[:, 0], errors="coerce")

    for idx, col in enumerate(headers[2:]):
        if not col or col == "timestamp":
            continue
        serie = df_raw.iloc[:, idx + 2].astype(str).str.replace(",", ".", regex=False)
        df_out[col] = pd.to_numeric(serie, errors="coerce")

    df_out = df_out.dropna(subset=ccl_cols, how="all")
    return df_out, ccl_cols


def calcular_desvios(df_precios, cols):
    """
    Calcula desvíos usando la mediana de las columnas indicadas (grupo propio).
    Filtra outliers (|desvío| > 10%) que son precios malos de IOL (0, stale, etc).
    """
    mediana = df_precios[cols].median(axis=1)
    desvios = (df_precios[cols].div(mediana, axis=0) - 1) * 100
    # Reemplazar outliers con NaN — no eliminar filas completas
    desvios = desvios.where(desvios.abs() <= 10)
    desvios["timestamp"] = df_precios["timestamp"].values
    return desvios

# ── Cargar datos ──────────────────────────────────────────
with st.spinner("Cargando historial desde Sheets..."):
    df_precios, ccl_cols = cargar_precios()

if df_precios is None:
    st.warning("⏳ Insuficientes snapshots. Volvé después de algunas horas de operación.")
    st.stop()

n_snapshots = len(df_precios)

# Calcular cuántos ticks malos hay en total para informar al usuario
desvios_raw = (df_precios[ccl_cols].div(df_precios[ccl_cols].median(axis=1), axis=0) - 1) * 100
n_outliers = (desvios_raw.abs() > 10).sum().sum()
st.success(f"✅ {n_snapshots} snapshots cargados — {n_outliers} ticks malos filtrados (|desvío| > 10%)")

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

# CLAVE: mediana calculada solo con columnas del grupo → desvíos internamente consistentes
desvios = calcular_desvios(df_precios, cols_analisis)

st.caption(f"📌 {label_grupo} — mediana calculada dentro del grupo")
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

# Si es "Todos", superponer líquidos con su propia mediana para comparar
if grupo == "Todos" and cols_liq:
    desvios_liq = calcular_desvios(df_precios, cols_liq)
    fig_hist.add_trace(go.Histogram(
        x=desvios_liq[cols_liq].stack().dropna(),
        nbinsx=80, marker_color="#00C851", opacity=0.5,
        name="Líquidos (mediana propia)",
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
        st.markdown(f"- Actual en código:  `-0.60%`")
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
st.divider()

# ── Rendimiento simulado por símbolo ─────────────────────
st.subheader("💰 Rendimiento Simulado por Acción")
st.caption(
    "Simula operaciones usando las reglas actuales: "
    "compra si desvío < -0.5%, venta si desvío ≥ +0.15% Y precio ≥ +1.0%, "
    "o cierre forzado a los 30 ciclos. "
    "El PnL se aproxima por el movimiento del desvío CCL (no precio real)."
)

UMBRAL_COMPRA_SIM = -0.5
UMBRAL_VENTA_SIM  =  0.10
CICLOS_MAX        = 30   # cierre forzado

@st.cache_data(ttl=300)
def simular_por_simbolo(desvios_df, cols):
    resultados = []
    for sym in cols:
        serie = desvios_df[sym].dropna().values
        n     = len(serie)
        i     = 0
        while i < n - 1:
            # Buscar entrada
            if serie[i] < UMBRAL_COMPRA_SIM:
                dev_entrada = serie[i]
                dev_pico    = dev_entrada
                cerrado     = False
                for j in range(1, CICLOS_MAX + 1):
                    if i + j >= n:
                        break
                    dev_actual = serie[i + j]
                    # Actualizar pico
                    if dev_actual > dev_pico:
                        dev_pico = dev_actual
                    # Calcular PnL aproximado (reversión del spread)
                    pnl_aprox = dev_actual - dev_entrada  # positivo si el spread cerró
                    # Salida A: dev >= +0.10% Y pnl > 0
                    if dev_actual >= UMBRAL_VENTA_SIM and pnl_aprox > 0:
                        resultados.append({
                            "sym":         sym,
                            "dev_entrada": round(dev_entrada, 3),
                            "dev_salida":  round(dev_actual, 3),
                            "dev_pico":    round(dev_pico, 3),
                            "pnl_pct":     round(pnl_aprox, 3),
                            "ciclos":      j,
                            "tipo_salida": "SALIDA_A",
                        })
                        i += j
                        cerrado = True
                        break
                if not cerrado:
                    # Cierre forzado al ciclo máximo
                    j_final    = min(CICLOS_MAX, n - 1 - i)
                    dev_salida = serie[i + j_final]
                    pnl_aprox  = dev_salida - dev_entrada
                    resultados.append({
                        "sym":         sym,
                        "dev_entrada": round(dev_entrada, 3),
                        "dev_salida":  round(dev_salida, 3),
                        "dev_pico":    round(dev_pico, 3),
                        "pnl_pct":     round(pnl_aprox, 3),
                        "ciclos":      j_final,
                        "tipo_salida": "FORZADO",
                    })
                    i += j_final
            i += 1
    return pd.DataFrame(resultados)

df_sim = simular_por_simbolo(desvios, cols_analisis)

if df_sim.empty:
    st.info("No hay suficientes operaciones simuladas con el umbral actual.")
else:
    # Resumen por símbolo
    resumen_sim = df_sim.groupby("sym").agg(
        n_ops       = ("pnl_pct", "count"),
        pnl_prom    = ("pnl_pct", "mean"),
        pnl_total   = ("pnl_pct", "sum"),
        win_rate    = ("pnl_pct", lambda x: (x > 0).mean() * 100),
        ciclos_prom = ("ciclos",  "mean"),
        dev_pico    = ("dev_pico", "mean"),
    ).round(3).reset_index()

    resumen_sim = resumen_sim.sort_values("pnl_prom", ascending=False)

    # KPIs globales
    total_ops  = len(df_sim)
    pnl_global = df_sim["pnl_pct"].mean()
    wr_global  = (df_sim["pnl_pct"] > 0).mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Operaciones totales", total_ops)
    c2.metric("PnL promedio global", f"{pnl_global:+.3f}%")
    c3.metric("Win Rate global",     f"{wr_global:.1f}%")

    # Gráfico de barras — PnL promedio por símbolo
    colores_bar = ["#00C851" if v > 0 else "#FF4444"
                   for v in resumen_sim["pnl_prom"]]
    fig_pnl = go.Figure(go.Bar(
        x=resumen_sim["sym"],
        y=resumen_sim["pnl_prom"],
        marker_color=colores_bar,
        text=[f"{v:+.3f}%" for v in resumen_sim["pnl_prom"]],
        textposition="outside",
    ))
    fig_pnl.add_hline(y=0, line_color="white", line_width=1)
    fig_pnl.update_layout(
        title="PnL promedio por operación según símbolo (aproximado por desvío CCL)",
        xaxis_title="Símbolo",
        yaxis_title="PnL promedio (%)",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", height=360,
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Tabla detallada
    with st.expander("📋 Ver tabla completa por símbolo"):
        tabla = resumen_sim.copy()
        tabla.columns = ["Símbolo", "N ops", "PnL prom %", "PnL total %",
                         "Win Rate %", "Ciclos prom", "Dev pico prom %"]
        # Colorear PnL prom
        def color_pnl(val):
            try:
                return "color: #00C851" if float(val) > 0 else "color: #FF4444"
            except:
                return ""
        st.dataframe(
            tabla.style.map(color_pnl, subset=["PnL prom %", "PnL total %"]),
            use_container_width=True,
            hide_index=True,
        )

    # Distribución de salidas
    with st.expander("📊 Ver distribución por tipo de salida"):
        por_salida = df_sim.groupby(["sym", "tipo_salida"]).size().unstack(fill_value=0)
        st.dataframe(por_salida, use_container_width=True)

    st.caption(
        "⚠️ PnL aproximado por movimiento del desvío CCL. "
        "No incluye spreads de compra/venta ni slippage. "
        "Usalo para comparar activos entre sí, no como PnL real esperado."
    )

st.divider()

# ── Pico de ganancia intra-operación ─────────────────────
st.subheader("📈 Pico de Ganancia Intra-Operación")
st.caption(
    "Para cada entrada (dev < -0.5%), calcula el máximo PnL alcanzado "
    "dentro de los 30 ciclos siguientes. "
    "Sirve para calibrar la Salida A: si el p75 es +0.25%, "
    "una Salida A en +0.15% captura la mayoría de operaciones pero deja ganancia sobre la mesa."
)

def calcular_picos_ganancia(desvios_df, cols, umbral_entrada=-0.5, ciclos_max=30):
    """
    Para cada entrada con dev < umbral, rastrea el pico máximo de PnL
    (medido como dev_actual - dev_entrada) dentro de ciclos_max.
    """
    rows = []
    for sym in cols:
        serie = desvios_df[sym].dropna().values
        n     = len(serie)
        i     = 0
        while i < n - 1:
            if serie[i] < umbral_entrada:
                dev_entrada  = serie[i]
                pico_pnl     = 0.0
                ciclo_pico   = 0
                llego_pos    = False
                for j in range(1, min(ciclos_max + 1, n - i)):
                    pnl_j = serie[i + j] - dev_entrada
                    if pnl_j > pico_pnl:
                        pico_pnl   = pnl_j
                        ciclo_pico = j
                    if pnl_j > 0:
                        llego_pos = True
                rows.append({
                    "sym":         sym,
                    "dev_entrada": round(dev_entrada, 3),
                    "pico_pnl":    round(pico_pnl, 3),
                    "ciclo_pico":  ciclo_pico,
                    "llego_pos":   llego_pos,
                })
                i += ciclos_max  # saltar para no solapar entradas
            i += 1
    return pd.DataFrame(rows)

try:
    df_picos = calcular_picos_ganancia(desvios, cols_analisis)
except Exception as e:
    st.warning(f"Error calculando picos: {e}")
    df_picos = pd.DataFrame()

if df_picos.empty:
    st.info("Sin datos suficientes para el análisis de picos.")
else:
    total_entradas = len(df_picos)
    pct_pos        = df_picos["llego_pos"].mean() * 100
    pico_media     = df_picos["pico_pnl"].mean()
    pico_p50       = df_picos["pico_pnl"].quantile(0.50)
    pico_p75       = df_picos["pico_pnl"].quantile(0.75)
    pico_p90       = df_picos["pico_pnl"].quantile(0.90)
    pico_max       = df_picos["pico_pnl"].max()

    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Entradas analizadas", total_entradas)
    c2.metric("% llegaron a PnL > 0", f"{pct_pos:.1f}%")
    c3.metric("Media pico",  f"+{pico_media:.3f}%")
    c4.metric("p50 pico",    f"+{pico_p50:.3f}%")
    c5.metric("p75 pico",    f"+{pico_p75:.3f}%")
    c6.metric("p90 pico",    f"+{pico_p90:.3f}%")

    # Histograma del pico de ganancia
    fig_pk = go.Figure()
    fig_pk.add_trace(go.Histogram(
        x=df_picos["pico_pnl"],
        nbinsx=60,
        marker_color="#00C851",
        opacity=0.8,
        name="Pico PnL",
    ))
    # Líneas de referencia — umbrales candidatos de Salida A
    for v, color, label in [
        (0.10, "#FFD700",  "+0.10%"),
        (0.15, "#FF8C00",  "+0.15% (actual)"),
        (0.20, "#FF4444",  "+0.20%"),
        (pico_p75, "#00BFFF", f"p75 ({pico_p75:.3f}%)"),
    ]:
        fig_pk.add_vline(
            x=v, line_dash="dash", line_color=color,
            annotation_text=label, annotation_position="top right",
        )
    fig_pk.update_layout(
        title="Distribución del pico máximo de ganancia por operación",
        xaxis_title="Pico PnL máximo alcanzado (%)",
        yaxis_title="Frecuencia",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", height=360,
    )
    st.plotly_chart(fig_pk, use_container_width=True)

    # Tabla por símbolo
    with st.expander("📋 Ver picos por símbolo"):
        picos_sym = df_picos.groupby("sym").agg(
            entradas   = ("pico_pnl", "count"),
            pct_pos    = ("llego_pos", lambda x: f"{x.mean()*100:.0f}%"),
            media_pico = ("pico_pnl", "mean"),
            p50_pico   = ("pico_pnl", lambda x: x.quantile(0.50)),
            p75_pico   = ("pico_pnl", lambda x: x.quantile(0.75)),
            p90_pico   = ("pico_pnl", lambda x: x.quantile(0.90)),
            max_pico   = ("pico_pnl", "max"),
        ).round(3).reset_index()
        picos_sym.columns = ["Símbolo", "Entradas", "% PnL>0",
                              "Media %", "p50 %", "p75 %", "p90 %", "Máx %"]
        st.dataframe(picos_sym, use_container_width=True, hide_index=True)

    # Interpretación automática
    salida_a_actual = 1.00  # ahora es PnL precio, no desvío
    pct_captura = (df_picos["pico_pnl"] >= salida_a_actual).mean() * 100
    st.info(
        f"📌 Salida A requiere dev ≥ +0.15% Y precio ≥ +1.00%. "
        f"El **{pct_captura:.1f}%** de las entradas alcanza un pico ≥ +{salida_a_actual:.2f}% — "
        f"las restantes quedan para Salida B (trailing) o cierre forzado."
    )
