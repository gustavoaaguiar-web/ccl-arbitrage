"""
pages/analisis_temporal.py
Análisis estadístico temporal: cuándo conviene operar y cuándo no.
Fuente: sheet 'Operaciones' via SheetsManager.get_operaciones()
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sheets_manager import SheetsManager
from app import get_secrets

st.set_page_config(page_title="Análisis Temporal", page_icon="🕐", layout="wide")

# ── Clasificación de motivo_cierre ─────────────────────────────────────────────
WIN_MOTIVOS  = {"SALIDA_A", "SALIDA_B", "TAKE_PROFIT_D"}
STOP_MOTIVOS = {"STOP_LOSS_C"}
# CIERRE_FORZADO / VENTA_MANUAL / SEÑAL_CONTRARIA → neutros

DIAS_ES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

# ── Carga de datos ─────────────────────────────────────────────────────────────
COLS_OPS = [
    "id", "symbol", "tipo", "cantidad", "precio_entry",
    "precio_exit", "monto_entry", "monto_exit", "pnl",
    "pnl_pct", "ts_entry", "ts_exit", "motivo_cierre",
]

@st.cache_data(ttl=300)
def load_operaciones() -> pd.DataFrame:
    try:
        secrets = get_secrets()
        if not secrets:
            st.error("No se pudieron cargar los secrets.")
            return pd.DataFrame()
        sm = SheetsManager(service_account_info=secrets["gcp"])
        if not sm.conectar():
            st.error("No se pudo conectar a Google Sheets.")
            return pd.DataFrame()
        filas = sm.cargar_operaciones()   # lista de listas, sin header
    except Exception as e:
        st.error(f"Error cargando desde Sheets: {e}")
        return pd.DataFrame()

    if not filas:
        return pd.DataFrame()

    df = pd.DataFrame(filas, columns=COLS_OPS)

    if df.empty:
        return pd.DataFrame()

    from zoneinfo import ZoneInfo
    TZ_ART = ZoneInfo("America/Argentina/Buenos_Aires")
    TZ_NY  = ZoneInfo("America/New_York")

    df["ts_entry"] = pd.to_datetime(df["ts_entry"], errors="coerce")
    df = df.dropna(subset=["ts_entry"])
    df["ts_art"] = df["ts_entry"].dt.tz_localize(TZ_ART)
    df["ts_ny"]  = df["ts_art"].dt.tz_convert(TZ_NY)

    # pnl_pct: tolerar coma decimal (Sheets a veces exporta "1,45")
    df["pnl_pct"] = (
        df["pnl_pct"].astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    df["hora"]       = df["ts_art"].dt.hour
    df["hora_ny"]    = df["ts_ny"].dt.hour
    df["dow"]        = df["ts_art"].dt.dayofweek
    df["dia"]        = df["dow"].map(lambda d: DIAS_ES[d])
    df["is_win"]     = df["motivo_cierre"].isin(WIN_MOTIVOS).astype(int)
    df["is_stop"]    = df["motivo_cierre"].isin(STOP_MOTIVOS).astype(int)
    # Etiqueta de eje con ambos husos
    hora_ny_map = df.groupby("hora")["hora_ny"].first()
    df["hora_label"] = df["hora"].map(
        lambda h: f"{h:02d}:00 ({hora_ny_map.get(h, h-1):02d}:00 NY)"
    )

    return df


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🕐 Análisis Temporal de Trades")
st.caption("Estadísticas por hora y día de la semana · fuente: Operaciones")

df = load_operaciones()

if df.empty:
    st.warning("Sin datos de operaciones disponibles.")
    st.stop()

# ── Sidebar filtros ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filtros")
    symbols = ["Todos"] + sorted(df["symbol"].dropna().unique().tolist())
    sel_sym = st.selectbox("Symbol", symbols)
    motivos_disp = sorted(df["motivo_cierre"].dropna().unique().tolist())
    sel_motivos = st.multiselect("Motivo cierre", motivos_disp, default=motivos_disp)
    if st.button("🔄 Recargar datos"):
        st.cache_data.clear()
        st.rerun()

dff = df.copy()
if sel_sym != "Todos":
    dff = dff[dff["symbol"] == sel_sym]
if sel_motivos:
    dff = dff[dff["motivo_cierre"].isin(sel_motivos)]

if dff.empty:
    st.warning("Sin datos con los filtros seleccionados.")
    st.stop()

# ── KPIs rápidos ──────────────────────────────────────────────────────────────
n       = len(dff)
win_r   = dff["is_win"].mean() * 100
stop_r  = dff["is_stop"].mean() * 100
pnl_avg = dff["pnl_pct"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trades analizados", n)
c2.metric("Win rate", f"{win_r:.1f}%")
c3.metric("Stop-loss rate", f"{stop_r:.1f}%")
c4.metric("PnL promedio", f"{pnl_avg:+.2f}%")

st.divider()

# ── Helpers ────────────────────────────────────────────────────────────────────
ALL_HOURS = list(range(9, 22))
ALL_DAYS  = DIAS_ES[:5]   # Lun–Vie

def hora_labels(dff):
    """Ordered list of ART+NY labels for unique hours in dff."""
    mapping = (
        dff.drop_duplicates("hora")
        .sort_values("hora")
        .set_index("hora")["hora_label"]
    )
    return mapping

def make_pivot(data, value_col, agg_fn, index="hora", columns="dia"):
    piv = (
        data.groupby([index, columns])[value_col]
        .agg(agg_fn)
        .reset_index()
        .pivot(index=index, columns=columns, values=value_col)
    )
    if index == "hora":
        piv = piv.reindex(ALL_HOURS)
    piv = piv.reindex(columns=[d for d in ALL_DAYS if d in piv.columns])
    return piv

# ── 1. Heatmap Win Rate hora × día ────────────────────────────────────────────
st.subheader("📊 Win Rate (%) por hora y día")

hora_ny_map = dff.groupby("hora")["hora_ny"].first()
piv_win = make_pivot(dff, "is_win", "mean") * 100

fig_wr = go.Figure(go.Heatmap(
    z=piv_win.values,
    x=piv_win.columns.tolist(),
    y=[f"{h:02d}:00 ({hora_ny_map.get(h,h-1):02d}:00 NY)" for h in piv_win.index],
    colorscale="RdYlGn",
    zmin=0, zmax=100,
    text=[[f"{v:.0f}%" if not pd.isna(v) else "" for v in row] for row in piv_win.values],
    texttemplate="%{text}",
    hovertemplate="Día: %{x}<br>Hora: %{y}<br>Win rate: %{z:.1f}%<extra></extra>",
    colorbar=dict(title="Win %"),
))
fig_wr.update_layout(
    height=440,
    xaxis_title="Día",
    yaxis_title="Hora entrada",
    margin=dict(l=60, r=20, t=20, b=40),
)
st.plotly_chart(fig_wr, use_container_width=True)

# ── 2. Tasa de Stop-Loss por hora ─────────────────────────────────────────────
st.subheader("🛑 Tasa de Stop-Loss por hora de entrada")

stop_h = (
    dff.groupby("hora")
    .agg(n_trades=("is_stop", "count"), stop_rate=("is_stop", "mean"))
    .reset_index()
)
stop_h["stop_pct"] = stop_h["stop_rate"] * 100

fig_stop = go.Figure(go.Bar(
    x=[f"{h:02d}:00 ({hora_ny_map.get(h,h-1):02d}:00 NY)" for h in stop_h["hora"]],
    y=stop_h["stop_pct"],
    marker_color=stop_h["stop_pct"].apply(
        lambda v: "#ef4444" if v > 30 else "#f97316" if v > 15 else "#22c55e"
    ),
    text=stop_h["stop_pct"].apply(lambda v: f"{v:.0f}%"),
    textposition="outside",
    hovertemplate="Hora: %{x}<br>Stop rate: %{y:.1f}%<extra></extra>",
))
fig_stop.update_layout(
    height=320,
    xaxis_title="Hora de entrada",
    yaxis_title="% stops",
    yaxis=dict(range=[0, max(stop_h["stop_pct"].max() * 1.25, 10)]),
    margin=dict(l=50, r=20, t=10, b=40),
)
st.plotly_chart(fig_stop, use_container_width=True)

# ── 3. PnL promedio por hora ───────────────────────────────────────────────────
st.subheader("💰 PnL promedio (%) por hora de entrada")

pnl_h = (
    dff.groupby("hora")["pnl_pct"]
    .agg(["mean", "median", "count"])
    .reset_index()
    .rename(columns={"mean": "media", "median": "mediana", "count": "n"})
)

fig_pnl = go.Figure()
fig_pnl.add_trace(go.Bar(
    name="Media",
    x=[f"{h:02d}:00 ({hora_ny_map.get(h,h-1):02d}:00 NY)" for h in pnl_h["hora"]],
    y=pnl_h["media"],
    marker_color=pnl_h["media"].apply(lambda v: "#22c55e" if v >= 0 else "#ef4444"),
    text=pnl_h["media"].apply(lambda v: f"{v:+.2f}%"),
    textposition="outside",
))
fig_pnl.add_trace(go.Scatter(
    name="Mediana",
    x=[f"{h:02d}:00 ({hora_ny_map.get(h,h-1):02d}:00 NY)" for h in pnl_h["hora"]],
    y=pnl_h["mediana"],
    mode="lines+markers",
    line=dict(color="#a78bfa", width=2, dash="dot"),
))
fig_pnl.update_layout(
    height=320,
    xaxis_title="Hora de entrada",
    yaxis_title="PnL %",
    margin=dict(l=50, r=20, t=10, b=40),
    legend=dict(orientation="h", y=1.08),
)
st.plotly_chart(fig_pnl, use_container_width=True)

# ── 4. Cantidad de trades por hora y día ──────────────────────────────────────
st.subheader("📈 Volumen de trades")

col_a, col_b = st.columns(2)

with col_a:
    cnt_h = dff.groupby("hora").size().reset_index(name="n")
    fig_ch = px.bar(
        cnt_h,
        x=cnt_h["hora"].apply(lambda h: f"{h:02d}:00 ({hora_ny_map.get(h,h-1):02d}:00 NY)"),
        y="n",
        labels={"x": "Hora", "n": "Trades"},
        color_discrete_sequence=["#60a5fa"],
        text="n",
        title="Por hora",
    )
    fig_ch.update_traces(textposition="outside")
    fig_ch.update_layout(height=300, margin=dict(l=40, r=10, t=40, b=40), showlegend=False)
    st.plotly_chart(fig_ch, use_container_width=True)

with col_b:
    cnt_d = (
        dff.groupby(["dow", "dia"]).size()
        .reset_index(name="n").sort_values("dow")
    )
    fig_cd = px.bar(
        cnt_d, x="dia", y="n",
        labels={"dia": "Día", "n": "Trades"},
        color_discrete_sequence=["#34d399"],
        text="n",
        title="Por día",
    )
    fig_cd.update_traces(textposition="outside")
    fig_cd.update_layout(
        height=300, margin=dict(l=40, r=10, t=40, b=40),
        xaxis=dict(categoryorder="array", categoryarray=ALL_DAYS),
        showlegend=False,
    )
    st.plotly_chart(fig_cd, use_container_width=True)

# ── Tabla resumen por hora ─────────────────────────────────────────────────────
st.subheader("📋 Resumen por hora")

resumen = (
    dff.groupby("hora")
    .agg(
        trades    = ("pnl_pct",        "count"),
        win_rate  = ("is_win",         "mean"),
        stop_rate = ("is_stop",        "mean"),
        pnl_medio = ("pnl_pct",        "mean"),
        pnl_total = ("pnl_pct",        "sum"),
    )
    .reset_index()
)
resumen["Hora"] = resumen["hora"].apply(lambda h: f"{h:02d}:00 ({hora_ny_map.get(h,h-1):02d}:00 NY)")
resumen["Win %"]      = (resumen["win_rate"]  * 100).round(1)
resumen["Stop %"]     = (resumen["stop_rate"] * 100).round(1)
resumen["PnL medio"]  = resumen["pnl_medio"].round(2)
resumen["PnL suma"]   = resumen["pnl_total"].round(2)

def color_pnl(val):
    return "color: #22c55e" if val > 0 else "color: #ef4444" if val < 0 else ""

st.dataframe(
    resumen[["Hora", "trades", "Win %", "Stop %", "PnL medio", "PnL suma"]]
    .rename(columns={"trades": "Trades"})
    .style
    .applymap(color_pnl, subset=["PnL medio", "PnL suma"])
    .format({"Win %": "{:.1f}%", "Stop %": "{:.1f}%",
             "PnL medio": "{:+.2f}%", "PnL suma": "{:+.2f}%"}),
    use_container_width=True,
    height=420,
)
