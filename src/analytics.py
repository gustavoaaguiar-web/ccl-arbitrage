"""
analytics.py — Funciones de Volume Profile y análisis temporal
Parte del sistema GG Investments — CCL Arbitrage
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def calcular_volume_profile(
    historial_ccl: List[Dict],
    bins: float = 0.0001,
    min_snapshots: int = 10
) -> Dict:
    """
    Calcula Volume Profile (POC) a partir de histórico de CCL.
    
    Args:
        historial_ccl: Lista de dicts con keys:
            - 'ts' o 'timestamp' (datetime)
            - 'ccl' o 'precio' (float)
            - 'volumen' (int, opcional)
        bins: Tamaño de cada bin en CCL (default 0.01% = 0.0001)
        min_snapshots: Mínimo de snapshots para considerar valido
    
    Returns:
        {
            'poc': float,               # Point of Control
            'histograma': {nivel: count},
            'min_nivel': float,
            'max_nivel': float,
            'total_snapshots': int,
            'timestamp_calculo': datetime,
            'es_valido': bool
        }
    """
    if not historial_ccl or len(historial_ccl) < min_snapshots:
        return {
            'poc': None,
            'histograma': {},
            'min_nivel': None,
            'max_nivel': None,
            'total_snapshots': 0,
            'timestamp_calculo': datetime.now(),
            'es_valido': False
        }
    
    # Extraer CCL values
    ccl_values = []
    for snapshot in historial_ccl:
        try:
            ccl = snapshot.get('ccl') or snapshot.get('precio')
            if ccl:
                ccl_values.append(float(ccl))
        except (ValueError, TypeError):
            continue
    
    if not ccl_values:
        return {
            'poc': None,
            'histograma': {},
            'min_nivel': None,
            'max_nivel': None,
            'total_snapshots': 0,
            'timestamp_calculo': datetime.now(),
            'es_valido': False
        }
    
    ccl_array = np.array(ccl_values)
    min_ccl = ccl_array.min()
    max_ccl = ccl_array.max()
    
    # Binning: redondear cada CCL al bin inferior
    histograma = defaultdict(int)
    for ccl in ccl_values:
        bin_level = np.floor(ccl / bins) * bins
        bin_level = round(bin_level, 6)  # evitar floating point errors
        histograma[bin_level] += 1
    
    # POC = bin con mayor count
    poc = max(histograma, key=histograma.get) if histograma else None
    
    return {
        'poc': poc,
        'histograma': dict(histograma),
        'min_nivel': round(min_ccl, 6),
        'max_nivel': round(max_ccl, 6),
        'total_snapshots': len(ccl_values),
        'timestamp_calculo': datetime.now(),
        'es_valido': True
    }


def graficar_volume_profile(
    vp: Dict,
    ccl_actual: float,
    titulo: str = "Volume Profile",
    use_plotly: bool = True
) -> 'plotly.graph_objects.Figure':
    """
    Genera gráfico de Volume Profile con POC y CCL actual.
    
    Args:
        vp: Dict retornado por calcular_volume_profile()
        ccl_actual: CCL actual (para marcar línea)
        titulo: Título del gráfico
        use_plotly: Si True, retorna Plotly Figure; si False, matplotlib
    
    Returns:
        plotly.graph_objects.Figure
    """
    if not vp['es_valido']:
        return None
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    histograma = vp['histograma']
    if not histograma:
        return None
    
    niveles = sorted(histograma.keys())
    counts = [histograma[n] for n in niveles]
    
    fig = go.Figure()
    
    # Barras de volumen
    fig.add_trace(go.Bar(
        x=counts,
        y=niveles,
        orientation='h',
        marker=dict(color='rgba(31, 119, 180, 0.7)', line=dict(color='rgb(31, 119, 180)')),
        name='Volumen',
        hovertemplate='<b>Nivel:</b> %{y:.6f}<br><b>Snapshots:</b> %{x}<extra></extra>'
    ))
    
    # Línea POC
    if vp['poc']:
        fig.add_hline(
            y=vp['poc'],
            line_dash='dash',
            line_color='red',
            annotation_text=f"POC: {vp['poc']:.6f}",
            annotation_position='right'
        )
    
    # Línea CCL actual
    fig.add_hline(
        y=ccl_actual,
        line_dash='solid',
        line_color='green',
        annotation_text=f"CCL Actual: {ccl_actual:.6f}",
        annotation_position='right'
    )
    
    fig.update_layout(
        title=titulo,
        xaxis_title='Snapshots (Volumen Relativo)',
        yaxis_title='Nivel de Precio (CCL)',
        height=500,
        hovermode='closest'
    )
    
    return fig


def obtener_umbral_por_hora(hora_ars: datetime, umbrales_config: Dict) -> Optional[float]:
    """
    Retorna el umbral de compra dinámico según la hora ART.
    
    Args:
        hora_ars: datetime con timezone America/Argentina/Buenos_Aires
        umbrales_config: Dict con estructura:
            {
                "09:50-11:30": -0.0065,
                "11:31-15:29": -0.0050,
                "15:30-16:29": -0.0065,
                "16:30+": None
            }
    
    Returns:
        float (umbral) o None (sin compras permitidas)
    """
    hora_min = hora_ars.hour * 60 + hora_ars.minute
    
    # Rangos en minutos desde medianoche
    rangos = [
        ((9 * 60 + 50, 11 * 60 + 30), umbrales_config.get("09:50-11:30", -0.0065)),
        ((11 * 60 + 31, 15 * 60 + 29), umbrales_config.get("11:31-15:29", -0.0050)),
        ((15 * 60 + 30, 16 * 60 + 29), umbrales_config.get("15:30-16:29", -0.0065)),
        ((16 * 60 + 30, 23 * 60 + 59), umbrales_config.get("16:30+", None)),
    ]
    
    for (inicio, fin), umbral in rangos:
        if inicio <= hora_min <= fin:
            return umbral
    
    # Fuera de horario (antes de 09:50)
    return None


def calcular_distancia_al_poc(ccl_actual: float, poc: float) -> Dict:
    """
    Calcula métricas de distancia CCL actual vs POC.
    
    Args:
        ccl_actual: CCL actual
        poc: Point of Control
    
    Returns:
        {
            'distancia_pct': float,      # (ccl_actual - poc) / poc * 100
            'distancia_bps': float,      # basis points
            'en_poc': bool,              # si está dentro de ±0.01% del POC
            'estado': str                # 'SOBRE_POC', 'EN_POC', 'BAJO_POC'
        }
    """
    if not poc:
        return None
    
    distancia = (ccl_actual - poc) / poc
    distancia_pct = distancia * 100
    distancia_bps = distancia * 10000
    
    tolerance = 0.0001  # 0.01%
    
    if abs(distancia) < tolerance:
        estado = 'EN_POC'
    elif ccl_actual > poc:
        estado = 'SOBRE_POC'
    else:
        estado = 'BAJO_POC'
    
    return {
        'distancia_pct': round(distancia_pct, 3),
        'distancia_bps': round(distancia_bps, 1),
        'en_poc': estado == 'EN_POC',
        'estado': estado
    }


def analizar_temporalidad_compras(
    operaciones_df: pd.DataFrame,
    agrupador: str = 'hora'
) -> Dict:
    """
    Analiza performance de compras por franja horaria.
    
    Args:
        operaciones_df: DataFrame de operaciones con columna 'ts_entry'
        agrupador: 'hora', 'hora_rango' (apertura/mediodía/cierre)
    
    Returns:
        {
            'por_hora': {9: {'count': 5, 'win_rate': 0.8, 'pnl_promedio': 0.0045}, ...},
            'por_rango': {'apertura': {...}, 'mediodía': {...}, 'cierre': {...}}
        }
    """
    if operaciones_df.empty:
        return {'por_hora': {}, 'por_rango': {}}
    
    # Asegurar que ts_entry es datetime
    if not pd.api.types.is_datetime64_any_dtype(operaciones_df['ts_entry']):
        operaciones_df['ts_entry'] = pd.to_datetime(operaciones_df['ts_entry'])
    
    # Convertir a ART si es necesario
    if operaciones_df['ts_entry'].dt.tz is None:
        operaciones_df['ts_entry'] = operaciones_df['ts_entry'].dt.tz_localize('UTC').dt.tz_convert('America/Argentina/Buenos_Aires')
    else:
        operaciones_df['ts_entry'] = operaciones_df['ts_entry'].dt.tz_convert('America/Argentina/Buenos_Aires')
    
    operaciones_df['hora'] = operaciones_df['ts_entry'].dt.hour
    operaciones_df['es_ganancia'] = operaciones_df['pnl_pct'] >= 0
    
    # Por hora
    por_hora = {}
    for hora, grupo in operaciones_df.groupby('hora'):
        por_hora[hora] = {
            'count': len(grupo),
            'win_rate': grupo['es_ganancia'].mean(),
            'pnl_promedio': grupo['pnl_pct'].mean(),
            'pnl_std': grupo['pnl_pct'].std(),
            'max_gain': grupo['pnl_pct'].max(),
            'max_loss': grupo['pnl_pct'].min()
        }
    
    # Por rango
    def clasificar_rango(hora):
        if 9 <= hora < 11:
            return 'apertura'
        elif 11 <= hora < 15:
            return 'mediodía'
        elif 15 <= hora < 17:
            return 'cierre'
        else:
            return 'fuera_horario'
    
    operaciones_df['rango'] = operaciones_df['hora'].apply(clasificar_rango)
    
    por_rango = {}
    for rango, grupo in operaciones_df.groupby('rango'):
        por_rango[rango] = {
            'count': len(grupo),
            'win_rate': grupo['es_ganancia'].mean(),
            'pnl_promedio': grupo['pnl_pct'].mean(),
            'pnl_std': grupo['pnl_pct'].std(),
            'max_gain': grupo['pnl_pct'].max(),
            'max_loss': grupo['pnl_pct'].min()
        }
    
    return {
        'por_hora': por_hora,
        'por_rango': por_rango
    }
