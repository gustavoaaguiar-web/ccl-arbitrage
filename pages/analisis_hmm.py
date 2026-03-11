"""
Análisis Estadístico del Modelo HMM
=====================================
Evalúa qué frecuencia de barras (1H, 2H, 4H, 1D) produce el mejor
filtro HMM para capturar subidas de precio en los activos del portafolio.

Metodología:
  Para cada frecuencia y cada símbolo:
    1. Descargar historial de barras de Alpaca (últimos 60 días)
    2. Entrenar HMM de 2 estados sobre log-returns
    3. Etiquetar cada barra como BULL o BEAR
    4. Medir el retorno forward (próximas N barras) en cada estado
    5. Calcular: media BULL vs BEAR, hit rate, ratio señal/ruido

Resultado: tabla comparativa de frecuencias + recomendación.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Análisis HMM", page_icon="🧠", layout="wide")
st.title("🧠 Análisis Estadístico del Modelo HMM")

# ── Configuración ─────────────────────────────────────────
PARES = {
    "GGAL":  "GGAL",  "YPFD":  "YPF",  "PAMP":  "PAM",  "CEPU": "CEPU",
    "AMZN":  "AMZN",  "MSFT":  "MSFT", "NVDA":  "NVDA", "TSLA": "TSLA",
    "AAPL":  "AAPL",  "META":  "META", "GOOGL": "GOOGL","MELI": "MELI",
    "BMA":   "BMA",   "VIST":  "VIST",
}
LIQUIDOS  = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "MELI"]
ILIQUIDOS = ["GGAL", "YPFD", "PAMP", "CEPU", "BMA", "VIST"]

TIMEFRAMES = {
    "1H":  {"tf": "1Hour",  "dias": 120, "label": "1 hora"},
    "2H":  {"tf": "2Hour",  "dias": 120, "label": "2 horas"},
    "4H":  {"tf": "4Hour",  "dias": 120, "label": "4 horas"},
    "1D":  {"tf": "1Day",   "dias": 365, "label": "Diaria"},
}

# ── Auth ──────────────────────────────────────────────────
@st.cache_resource
def get_alpaca_headers():
    try:
        return {
            "APCA-API-KEY-ID":     st.secrets["ALPACA_KEY"],
            "APCA-API-SECRET-KEY": st.secrets["ALPACA_SECRET"],
        }
    except:
        return None

# ── Fetch barras ──────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_barras(sym_usd: str, timeframe: str, dias: int) -> pd.DataFrame:
    """
    Descarga barras OHLCV de Alpaca para un símbolo.
    Retorna DataFrame con columnas: ts, open, high, low, close, volume.
    """
    headers = get_alpaca_headers()
    if not headers:
        return pd.DataFrame()
    start = (datetime.now(timezone.utc) - timedelta(days=dias)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url   = "https://data.alpaca.markets/v2/stocks/bars"
    barras, token = [], None
    while True:
        params = {
            "symbols":   sym_usd,
            "timeframe": timeframe,
            "start":     start,
            "limit":     1000,
            "feed":      "iex",
        }
        if token:
            params["page_token"] = token
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            r.raise_for_status()
            data  = r.json()
            items = data.get("bars", {}).get(sym_usd, [])
            barras.extend(items)
            token = data.get("next_page_token")
            if not token:
                break
        except Exception as e:
            st.warning(f"Error Alpaca {sym_usd}/{timeframe}: {e}")
            break

    if not barras:
        return pd.DataFrame()
    df = pd.DataFrame(barras)
    df.rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low",
                        "c": "close", "v": "volume"}, inplace=True)
    df["ts"] = pd.to_datetime(df["ts"])
    return df[["ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

# ── HMM ───────────────────────────────────────────────────
def entrenar_hmm(closes: np.ndarray):
    """
    Entrena GaussianHMM de 2 estados sobre log-returns.
    Retorna (estados, bull_state, modelo) o None si falla.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
        ret    = np.diff(np.log(closes)).reshape(-1, 1)
        m      = GaussianHMM(n_components=2, random_state=42, n_iter=200,
                              covariance_type="full").fit(ret)
        estados = m.predict(ret)
        bull    = int(np.argmax(m.means_.flatten()))
        return estados, bull, m
    except Exception as e:
        return None

def analizar_forward_returns(df: pd.DataFrame, estados: np.ndarray, bull: int,
                              n_forward: int = 3) -> dict:
    """
    Para cada barra, calcula el retorno forward a N barras.
    Separa resultados por estado BULL / BEAR.
    n_forward: cantidad de barras hacia adelante para medir retorno.
    """
    closes  = df["close"].values
    n       = len(estados)
    returns = []
    for i in range(n):
        if i + n_forward >= len(closes):
            break
        ret_fwd = (closes[i + n_forward] / closes[i + 1] - 1) * 100
        returns.append({
            "estado":  estados[i],
            "ret_fwd": ret_fwd,
            "bull":    estados[i] == bull,
        })
    df_ret = pd.DataFrame(returns)
    if df_ret.empty:
        return {}

    bull_rets = df_ret[df_ret["bull"]]["ret_fwd"]
    bear_rets = df_ret[~df_ret["bull"]]["ret_fwd"]

    # Hit rate: % de veces que BULL predice retorno positivo
    hit_rate  = (bull_rets > 0).mean() * 100 if len(bull_rets) > 0 else 0

    # Separación: diferencia de medias BULL - BEAR (señal/ruido)
    sep = bull_rets.mean() - bear_rets.mean() if len(bear_rets) > 0 else 0

    # Sharpe simplificado del estado BULL
    sharpe = (bull_rets.mean() / bull_rets.std()) if len(bull_rets) > 1 and bull_rets.std() > 0 else 0

    return {
        "n_bull":        len(bull_rets),
        "n_bear":        len(bear_rets),
        "pct_bull":      len(bull_rets) / len(df_ret) * 100,
        "media_bull":    bull_rets.mean(),
        "media_bear":    bear_rets.mean(),
        "std_bull":      bull_rets.std(),
        "hit_rate":      hit_rate,
        "separacion":    sep,
        "sharpe_bull":   sharpe,
        "bull_rets":     bull_rets.values,
        "bear_rets":     bear_rets.values,
    }

# ── UI ────────────────────────────────────────────────────
headers = get_alpaca_headers()
if not headers:
    st.error("⚠️ Configurar ALPACA_KEY y ALPACA_SECRET en Streamlit Secrets.")
    st.stop()

# Controles
col1, col2, col3 = st.columns(3)
with col1:
    grupo = st.radio("Grupo de activos", ["Líquidos", "Ilíquidos", "Todos"], horizontal=True)
with col2:
    n_forward = st.slider("Barras forward para medir retorno", 1, 10, 3,
                           help="Cuántas barras hacia adelante mide el retorno tras la señal BULL")
with col3:
    tfs_sel = st.multiselect("Frecuencias a analizar",
                              list(TIMEFRAMES.keys()),
                              default=["1H", "2H", "4H", "1D"])

if grupo == "Líquidos":
    syms = LIQUIDOS
elif grupo == "Ilíquidos":
    syms = ILIQUIDOS
else:
    syms = list(PARES.keys())

syms_usd = list({PARES[s] for s in syms})

if not tfs_sel:
    st.warning("Seleccioná al menos una frecuencia.")
    st.stop()

# ── Carga y análisis ──────────────────────────────────────
st.markdown("---")
st.subheader("📊 Comparativa de Frecuencias")
st.caption(f"Activos: {', '.join(syms_usd)} | Forward: {n_forward} barras | Feed: IEX")

resultados = []   # fila por (timeframe, sym_usd)
bull_dist  = {}   # para distribuciones

prog = st.progress(0, text="Descargando barras...")
total = len(tfs_sel) * len(syms_usd)
paso  = 0

for tf_key in tfs_sel:
    cfg = TIMEFRAMES[tf_key]
    bull_dist[tf_key] = {"bull": [], "bear": []}

    for sym_usd in syms_usd:
        paso += 1
        prog.progress(paso / total, text=f"Analizando {sym_usd} @ {tf_key}...")

        df = fetch_barras(sym_usd, cfg["tf"], cfg["dias"])
        if df.empty or len(df) < 15:
            continue

        res_hmm = entrenar_hmm(df["close"].values)
        if res_hmm is None:
            continue

        estados, bull, modelo = res_hmm
        stats = analizar_forward_returns(df, estados, bull, n_forward)
        if not stats:
            continue

        bull_dist[tf_key]["bull"].extend(stats["bull_rets"].tolist())
        bull_dist[tf_key]["bear"].extend(stats["bear_rets"].tolist())

        resultados.append({
            "Frecuencia":   tf_key,
            "Símbolo":      sym_usd,
            "N barras":     len(df),
            "% BULL":       round(stats["pct_bull"], 1),
            "Media BULL %": round(stats["media_bull"], 4),
            "Media BEAR %": round(stats["media_bear"], 4),
            "Separación %": round(stats["separacion"], 4),
            "Hit Rate %":   round(stats["hit_rate"], 1),
            "Sharpe BULL":  round(stats["sharpe_bull"], 3),
        })

prog.empty()

if not resultados:
    st.error("No se obtuvieron datos. Verificar conexión Alpaca o símbolos.")
    st.stop()

df_res = pd.DataFrame(resultados)

# ── Tabla detalle ──────────────────────────────────────────
with st.expander("📋 Ver tabla detallada por símbolo", expanded=False):
    st.dataframe(df_res, use_container_width=True, hide_index=True)

# ── Resumen por frecuencia ─────────────────────────────────
st.subheader("📈 Resumen por Frecuencia")
resumen = df_res.groupby("Frecuencia").agg(
    Media_BULL   = ("Media BULL %", "mean"),
    Media_BEAR   = ("Media BEAR %", "mean"),
    Separacion   = ("Separación %", "mean"),
    Hit_Rate     = ("Hit Rate %",   "mean"),
    Sharpe       = ("Sharpe BULL",  "mean"),
    Pct_BULL     = ("% BULL",       "mean"),
).round(4).reset_index()

# Ordenar por Sharpe BULL — métrica correcta para estrategias long-only
# (separación Bull-Bear es útil para long/short, no para long-only)
resumen = resumen.sort_values("Sharpe", ascending=False).reset_index(drop=True)
resumen.index = resumen.index + 1  # ranking 1-based

# Highlight mejor frecuencia
mejor_tf = resumen.iloc[0]["Frecuencia"]
st.success(f"🏆 Frecuencia recomendada: **{mejor_tf}** ({TIMEFRAMES[mejor_tf]['label']}) "
           f"— mayor Sharpe BULL = {resumen.iloc[0]['Sharpe']:.3f} | Hit Rate = {resumen.iloc[0]['Hit_Rate']:.1f}%")

# Métricas en columnas
cols = st.columns(len(resumen))
for i, (_, row) in enumerate(resumen.iterrows()):
    with cols[i]:
        tf = row["Frecuencia"]
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        st.metric(f"{emoji} {tf}", f"Sharpe: {row['Sharpe']:.3f}")
        st.caption(f"Bull: {row['Media_BULL']:+.4f}% | Hit: {row['Hit_Rate']:.1f}%\n"
                   f"Bear: {row['Media_BEAR']:+.4f}% | Sep: {row['Separacion']:+.4f}%\n"
                   f"% tiempo BULL: {row['Pct_BULL']:.1f}%")

# ── Gráfico: separación por frecuencia ────────────────────
st.markdown("---")
st.subheader("🔬 Sharpe BULL por Frecuencia")
st.caption("Sharpe BULL = Media/σ de retornos en estado BULL. Métrica correcta para long-only (no long/short).")

fig_sep = go.Figure()
colores = {"1H": "#4FC3F7", "2H": "#81C784", "4H": "#FFB74D", "1D": "#F06292"}
for _, row in resumen.iterrows():
    tf = row["Frecuencia"]
    fig_sep.add_trace(go.Bar(
        x=[tf],
        y=[row["Sharpe"]],
        name=tf,
        marker_color=colores.get(tf, "#888"),
        text=f"{row['Sharpe']:.3f}",
        textposition="outside",
    ))
fig_sep.update_layout(
    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    font_color="white", height=320,
    showlegend=False,
    yaxis_title="Sharpe BULL",
)
st.plotly_chart(fig_sep, use_container_width=True)

# ── Gráfico: hit rate ──────────────────────────────────────
st.subheader("🎯 Hit Rate en Estado BULL por Frecuencia")
st.caption(f"% de señales BULL donde el precio subió en las siguientes {n_forward} barras.")

fig_hr = go.Figure()
for _, row in resumen.iterrows():
    tf = row["Frecuencia"]
    color = "#00C851" if row["Hit_Rate"] >= 55 else "#FF4444" if row["Hit_Rate"] < 50 else "#FFB74D"
    fig_hr.add_trace(go.Bar(
        x=[tf], y=[row["Hit_Rate"]],
        name=tf, marker_color=color,
        text=f"{row['Hit_Rate']:.1f}%", textposition="outside",
    ))
fig_hr.add_hline(y=50, line_dash="dash", line_color="white",
                  annotation_text="50% (azar)", annotation_position="right")
fig_hr.update_layout(
    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
    font_color="white", height=320, showlegend=False,
    yaxis_title="Hit Rate (%)", yaxis_range=[0, 80],
)
st.plotly_chart(fig_hr, use_container_width=True)

# ── Distribuciones de retornos ─────────────────────────────
st.markdown("---")
st.subheader("📉 Distribución de Retornos BULL vs BEAR")
tf_viz = st.selectbox("Frecuencia a visualizar", tfs_sel,
                        index=tfs_sel.index(mejor_tf) if mejor_tf in tfs_sel else 0)

if tf_viz in bull_dist:
    b_rets = np.array(bull_dist[tf_viz]["bull"])
    br_rets = np.array(bull_dist[tf_viz]["bear"])

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=b_rets, name="BULL 🟢",
        marker_color="#00C851", opacity=0.7,
        nbinsx=60, histnorm="probability",
    ))
    fig_dist.add_trace(go.Histogram(
        x=br_rets, name="BEAR 🔴",
        marker_color="#FF4444", opacity=0.7,
        nbinsx=60, histnorm="probability",
    ))
    if len(b_rets) > 0:
        fig_dist.add_vline(x=float(np.mean(b_rets)),
                            line_dash="solid", line_color="#00C851",
                            annotation_text=f"μ BULL {np.mean(b_rets):+.4f}%")
    if len(br_rets) > 0:
        fig_dist.add_vline(x=float(np.mean(br_rets)),
                            line_dash="solid", line_color="#FF4444",
                            annotation_text=f"μ BEAR {np.mean(br_rets):+.4f}%")
    fig_dist.add_vline(x=0, line_dash="dash", line_color="white")
    fig_dist.update_layout(
        barmode="overlay",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", height=380,
        xaxis_title=f"Retorno a {n_forward} barras (%)",
        yaxis_title="Probabilidad",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Estadísticas textuales
    if len(b_rets) > 0 and len(br_rets) > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Media BULL",
                  f"{np.mean(b_rets):+.4f}%",
                  f"vs Bear {np.mean(br_rets):+.4f}%")
        c2.metric("Hit Rate BULL",
                  f"{(b_rets > 0).mean()*100:.1f}%",
                  f"{(b_rets > 0).mean()*100 - 50:+.1f}% vs azar")
        c3.metric("Separación",
                  f"{np.mean(b_rets) - np.mean(br_rets):+.4f}%",
                  "Bull − Bear")

# ── Análisis por símbolo ───────────────────────────────────
st.markdown("---")
with st.expander(f"🔍 Ver separación por símbolo en {mejor_tf}", expanded=False):
    df_mejor = df_res[df_res["Frecuencia"] == mejor_tf].sort_values("Separación %", ascending=False)
    if not df_mejor.empty:
        fig_sym = px.bar(
            df_mejor, x="Símbolo", y="Separación %",
            color="Hit Rate %",
            color_continuous_scale="RdYlGn",
            text="Separación %",
            title=f"Separación BULL/BEAR por símbolo — {mejor_tf}",
        )
        fig_sym.update_layout(
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
            font_color="white", height=380,
        )
        st.plotly_chart(fig_sym, use_container_width=True)

        # Tabla compacta
        st.dataframe(
            df_mejor[["Símbolo", "N barras", "% BULL", "Media BULL %",
                        "Media BEAR %", "Separación %", "Hit Rate %", "Sharpe BULL"]],
            use_container_width=True, hide_index=True,
        )

# ── Recomendación final ────────────────────────────────────
st.markdown("---")
st.subheader("💡 Recomendación")

row_mejor = resumen[resumen["Frecuencia"] == mejor_tf].iloc[0]
sep_val   = row_mejor["Separacion"]
hr_val    = row_mejor["Hit_Rate"]
sharpe_val = row_mejor["Sharpe"]

if hr_val >= 55 and sharpe_val > 0.05:
    calidad = "✅ Alta calidad"
    msg = "El filtro HMM agrega valor estadísticamente significativo."
elif hr_val >= 52 or sharpe_val > 0.01:
    calidad = "⚠️ Calidad moderada"
    msg = "El HMM agrega algo de valor, pero la señal es débil. Considerar más datos."
else:
    calidad = "❌ Señal insuficiente"
    msg = "El HMM no discrimina bien a ninguna frecuencia. Revisar los datos o el modelo."

st.info(
    f"**Frecuencia óptima:** {mejor_tf} ({TIMEFRAMES[mejor_tf]['label']})\n\n"
    f"**Calidad del filtro:** {calidad}\n\n"
    f"**Separación BULL/BEAR:** {sep_val:+.4f}% | "
    f"**Hit Rate:** {hr_val:.1f}% | "
    f"**Sharpe BULL:** {sharpe_val:.3f}\n\n"
    f"{msg}\n\n"
    f"*Los umbrales óptimos dependen del régimen de mercado. Revisá semanalmente.*"
)

st.caption("HMM_BARRAS_REFRESH_MIN en app.py controla cada cuánto se actualiza el clima en producción.")
      
