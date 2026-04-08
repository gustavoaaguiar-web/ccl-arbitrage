"""
pages/temporal_strategy.py — Dashboard de Estrategia Temporal
Parte del sistema GG Investments — CCL Arbitrage

Muestra:
1. Umbrales dinámicos por franja horaria (09:50–16:50)
2. Performance histórica de operaciones por hora/rango
3. Indicador de "compra permitida" según hora actual
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, time

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import sys
sys.path.insert(0, '.')

from src.sheets_manager import SheetsManager
from src.analytics import (
    obtener_umbral_por_hora,
    analizar_temporalidad_compras
)

TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")


def get_sheets() -> SheetsManager:
    """Reutiliza SheetsManager de session_state si ya fue inicializado por app.py."""
    if "sheets" in st.session_state and st.session_state.sheets is not None:
        return st.session_state.sheets
    return SheetsManager()


def obtener_umbrales_config() -> dict:
    """
    Retorna la configuración de umbrales dinámicos.
    IMPORTANTE: sincronizada con src/simulator.py
    """
    return {
        "09:50-11:30": -0.0065,     # Apertura — más exigente
        "11:31-15:29": -0.0050,     # Mediodía — estándar
        "15:30-16:29": -0.0065,     # Cierre — más exigente
        "16:30+": None              # Bloqueado
    }


def construir_dataframe_horarios(umbrales_config: dict) -> pd.DataFrame:
    datos = [
        {"Rango": "09:50–11:30", "Nombre": "Apertura",    "Umbral": "-0.65%",   "Umbral_float": -0.0065, "Color": "🔴 Exigente"},
        {"Rango": "11:31–15:29", "Nombre": "Mediodía",    "Umbral": "-0.50%",   "Umbral_float": -0.0050, "Color": "🟡 Estándar"},
        {"Rango": "15:30–16:29", "Nombre": "Cierre",      "Umbral": "-0.65%",   "Umbral_float": -0.0065, "Color": "🔴 Exigente"},
        {"Rango": "16:30–16:50", "Nombre": "Post-cierre", "Umbral": "BLOQUEADO","Umbral_float": None,    "Color": "🔴 No compra"},
    ]
    return pd.DataFrame(datos)


def obtener_rango_horario_actual(ahora: datetime) -> tuple:
    hora_min = ahora.hour * 60 + ahora.minute

    if 9*60+50 <= hora_min <= 11*60+30:
        return ("Apertura", -0.0065, True)
    elif 11*60+31 <= hora_min <= 15*60+29:
        return ("Mediodía", -0.0050, True)
    elif 15*60+30 <= hora_min <= 16*60+29:
        return ("Cierre", -0.0065, True)
    elif 16*60+30 <= hora_min <= 16*60+50:
        return ("Post-cierre", None, False)
    else:
        return ("Fuera de horario", None, False)


def graficar_curva_umbrales() -> go.Figure:
    horas_puntos = [
        (9.833,  -0.0065),
        (11.5,   -0.0065),
        (11.517, -0.0050),
        (15.483, -0.0050),
        (15.5,   -0.0065),
        (16.483, -0.0065),
        (16.5,   None),
    ]

    horas_activas   = []
    umbrales_activos = []

    for h, u in horas_puntos:
        if u is not None:
            horas_activas.append(h)
            umbrales_activos.append(u * 100)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=horas_activas,
        y=umbrales_activos,
        mode='lines+markers',
        name='Umbral Dinámico',
        line=dict(color='rgb(31, 119, 180)', width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)',
        hovertemplate='<b>%{x:.2f}h</b><br>Umbral: %{y:.3f}%<extra></extra>'
    ))

    fig.add_vrect(
        x0=16.5, x1=16.833,
        fillcolor="rgba(255, 0, 0, 0.1)",
        line_width=0,
        annotation_text="🔴 BLOQUEADO",
        annotation_position="top left"
    )

    fig.add_vrect(x0=9.833,  x1=11.5,   fillcolor="rgba(255, 200, 0, 0.05)", layer="below", line_width=0)
    fig.add_vrect(x0=11.517, x1=15.483, fillcolor="rgba(0, 200, 0, 0.05)",   layer="below", line_width=0)
    fig.add_vrect(x0=15.5,   x1=16.483, fillcolor="rgba(255, 200, 0, 0.05)", layer="below", line_width=0)

    fig.update_layout(
        title="Umbral de Compra Dinámico — Curva Temporal",
        xaxis_title="Hora ART",
        yaxis_title="Umbral de Compra (%)",
        height=400,
        hovermode='x unified',
        xaxis=dict(
            tickvals=[10, 11, 12, 13, 14, 15, 16, 17],
            ticktext=["10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00"]
        ),
        yaxis=dict(tickformat=".2%")
    )

    return fig


def graficar_performance_temporal(operaciones_df: pd.DataFrame) -> dict:
    if operaciones_df.empty:
        return {}

    # Los timestamps vienen de Sheets como strings UTC naive
    # (datetime.now() en GitHub Actions / Streamlit Cloud corre en UTC)
    import pytz
    UTC = pytz.utc
    ART = pytz.timezone('America/Argentina/Buenos_Aires')

    ts = pd.to_datetime(operaciones_df['ts_entry'])   # naive
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(UTC)
    ts_art = ts.dt.tz_convert(ART)

    operaciones_df['ts_entry'] = ts_art
    operaciones_df['hora']     = ts_art.dt.hour
    operaciones_df['es_ganancia'] = operaciones_df['pnl_pct'] >= 0

    por_hora = operaciones_df.groupby('hora').agg({
        'id':          'count',
        'es_ganancia': 'mean',
        'pnl_pct':     ['mean', 'std', 'max', 'min']
    }).reset_index()

    por_hora.columns = ['hora', 'operaciones', 'win_rate', 'pnl_mean', 'pnl_std', 'pnl_max', 'pnl_min']

    fig_wr = go.Figure()
    fig_wr.add_trace(go.Bar(
        x=por_hora['hora'],
        y=por_hora['win_rate'] * 100,
        marker=dict(color=por_hora['win_rate'], colorscale='RdYlGn', showscale=True, colorbar=dict(title="Win %")),
        text=por_hora['win_rate'].apply(lambda x: f"{x*100:.0f}%"),
        textposition='outside',
        hovertemplate='<b>Hora: %{x}:00</b><br>Win Rate: %{y:.1f}%<extra></extra>'
    ))
    fig_wr.update_layout(
        title="Win Rate por Hora del Día",
        xaxis_title="Hora ART", yaxis_title="Win Rate (%)",
        height=350, xaxis=dict(tickmode='linear', tick0=9, dtick=1)
    )

    colores = ['green' if x >= 0 else 'red' for x in por_hora['pnl_mean']]
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Bar(
        x=por_hora['hora'],
        y=por_hora['pnl_mean'] * 100,
        marker=dict(color=colores),
        text=por_hora['pnl_mean'].apply(lambda x: f"{x*100:+.2f}%"),
        textposition='outside',
        hovertemplate='<b>Hora: %{x}:00</b><br>PnL: %{y:+.2f}%<extra></extra>'
    ))
    fig_pnl.update_layout(
        title="PnL Promedio por Hora del Día",
        xaxis_title="Hora ART", yaxis_title="PnL (%)",
        height=350, xaxis=dict(tickmode='linear', tick0=9, dtick=1), hovermode='x unified'
    )

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=por_hora['hora'],
        y=por_hora['operaciones'],
        marker=dict(color='rgb(100, 150, 200)'),
        text=por_hora['operaciones'],
        textposition='outside',
        hovertemplate='<b>Hora: %{x}:00</b><br>Operaciones: %{y}<extra></extra>'
    ))
    fig_vol.update_layout(
        title="Volumen de Operaciones por Hora",
        xaxis_title="Hora ART", yaxis_title="Cantidad",
        height=350, xaxis=dict(tickmode='linear', tick0=9, dtick=1)
    )

    return {
        'win_rate': fig_wr,
        'pnl':      fig_pnl,
        'volumen':  fig_vol,
        'datos':    por_hora
    }


def main():
    st.set_page_config(page_title="⏰ Temporal Strategy", layout="wide")
    st.title("⏰ Temporal Strategy — Umbrales Dinámicos")

    try:
        sheets = get_sheets()
    except Exception as e:
        st.error(f"Error inicializando SheetsManager: {e}")
        st.stop()

    ahora          = datetime.now(TZ_ARG)
    umbrales_config = obtener_umbrales_config()

    # ============================================================================
    # SECCIÓN 1: INDICADOR ACTUAL
    # ============================================================================
    st.subheader("🔴 Estado Actual")

    rango_actual, umbral_actual, permitido = obtener_rango_horario_actual(ahora)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Hora ART", ahora.strftime("%H:%M:%S"))
    with col2:
        st.metric("Rango Activo", rango_actual)
    with col3:
        st.metric("Umbral", f"{umbral_actual*100:.2f}%" if umbral_actual is not None else "---")
    with col4:
        if permitido:
            st.success("✅ Compras PERMITIDAS")
        else:
            st.error("🔴 Compras BLOQUEADAS")

    st.divider()

    # ============================================================================
    # SECCIÓN 2: TABLA DE RANGOS
    # ============================================================================
    st.subheader("📋 Configuración de Umbrales")
    df_rangos = construir_dataframe_horarios(umbrales_config)
    st.dataframe(df_rangos, use_container_width=True, hide_index=True)

    st.divider()

    # ============================================================================
    # SECCIÓN 3: CURVA TEMPORAL
    # ============================================================================
    st.subheader("📈 Curva de Umbrales en el Día")
    fig_curva = graficar_curva_umbrales()
    st.plotly_chart(fig_curva, use_container_width=True)

    st.divider()

    # ============================================================================
    # SECCIÓN 4: PERFORMANCE HISTÓRICA
    # ============================================================================
    st.subheader("📊 Performance Histórica (últimos 30 días)")

    with st.spinner("Cargando historial de operaciones..."):
        try:
            fecha_inicio = ahora - timedelta(days=30)
            operaciones  = sheets.cargar_operaciones_en_rango(fecha_inicio, ahora)

            if operaciones:
                df_ops = pd.DataFrame(operaciones)
                stats_temporales = analizar_temporalidad_compras(df_ops.copy(), agrupador='hora')
                graficos         = graficar_performance_temporal(df_ops.copy())

                if graficos:
                    st.subheader("Win Rate por Hora")
                    st.plotly_chart(graficos['win_rate'], use_container_width=True)

                    st.subheader("PnL Promedio por Hora")
                    st.plotly_chart(graficos['pnl'], use_container_width=True)

                    st.subheader("Volumen de Operaciones")
                    st.plotly_chart(graficos['volumen'], use_container_width=True)

                    st.subheader("Tabla Detallada — Por Hora")
                    st.dataframe(
                        graficos['datos'].astype({
                            'win_rate': 'float64',
                            'pnl_mean': 'float64',
                            'pnl_std':  'float64',
                            'pnl_max':  'float64',
                            'pnl_min':  'float64'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader("🔍 Insights")
                    mejor_hora_idx = graficos['datos']['win_rate'].idxmax()
                    mejor_hora     = graficos['datos'].loc[mejor_hora_idx, 'hora']
                    mejor_wr       = graficos['datos'].loc[mejor_hora_idx, 'win_rate']

                    peor_hora_idx = graficos['datos']['win_rate'].idxmin()
                    peor_hora     = graficos['datos'].loc[peor_hora_idx, 'hora']
                    peor_wr       = graficos['datos'].loc[peor_hora_idx, 'win_rate']

                    st.info(f"✅ Mejor hora: **{int(mejor_hora)}:00** (Win Rate {mejor_wr*100:.1f}%)")
                    st.warning(f"⚠️ Peor hora: **{int(peor_hora)}:00** (Win Rate {peor_wr*100:.1f}%)")
            else:
                st.warning("No hay operaciones en los últimos 30 días")

        except Exception as e:
            st.error(f"Error cargando operaciones: {e}")

    st.divider()

    # ============================================================================
    # SECCIÓN 5: EXPLICACIÓN
    # ============================================================================
    st.subheader("📝 Explicación de la Estrategia")

    with st.expander("¿Por qué umbrales dinámicos?"):
        st.markdown("""
        - **Apertura (09:50–11:30):** Spreads más anchos, volatilidad de "matching". Umbral -0.65% (más exigente)
        - **Mediodía (11:31–15:29):** Mercado estabilizado. Umbral -0.50% (estándar)
        - **Cierre (15:30–16:29):** De nuevo spreads y volatilidad. Umbral -0.65% (más exigente)
        - **Post-cierre (16:30–16:50):** Cierre forzado. NO se abren nuevas posiciones
        """)

    with st.expander("¿Cómo se integra con el HMM?"):
        st.markdown("""
        La condición de compra REQUIERE ambas:
        
        ```
        if dev_ccl < umbral_por_hora(ahora) AND clima_hmm == "🟢 BULL":
            → COMPRA
        ```
        
        El HMM sigue siendo el guardián de la dirección del mercado.
        El umbral dinámico es una capa adicional de control temporal.
        """)


if __name__ == "__main__":
    main()
