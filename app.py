"""
CCL Arbitrage - App Streamlit Completa
=======================================
Dashboard en tiempo real con:
- CCL implícito por acción
- Señales 🟢🟡🔴 por desvío + filtro HMM (consistente con simulador)
- Clima HMM 🟢/🔴 por activo — modelo Simons (log-returns USD)
- Simulador con interés compuesto
- Alertas Gmail
- Persistencia Google Sheets

MODELO DE CLIMA (Simons):
  El HMM se entrena sobre log-returns del precio USD del subyacente,
  NO sobre niveles del CCL. Esto garantiza que la señal (CCL bajo) y
  el clima (momentum USD alcista) sean variables ortogonales e independientes,
  lo que mejora significativamente la calidad de las señales de compra.

  Señal de compra = spread CCL favorable  AND  régimen bull en USD
  Señal de venta  = spread CCL desfavorable  (HMM no interviene)

  IDENTIFICACIÓN DE ESTADO BULL:
  Se usa argmax de medias puras para identificar el estado bull.
  Guard adicional: si la media más alta es negativa, ambos estados son
  bajistas (mercado bear con rebotes violentos) y se retorna 🔴.
  Esto evita el problema del Sharpe: para activos ilíquidos, el estado
  "quieto" (returns ~0, std muy baja) tiene Sharpe artificialmente alto
  y era clasificado como BULL aunque el mercado no lo fuera.

UMBRALES DIFERENCIADOS:
  Activos volátiles (YPF/YPFD, TGSU2) usan umbral -0.65% para evitar
  falsas señales por volatilidad sin directionalidad.
  CEDEARs y otros usan -0.50%.

HISTORIAL HMM:
  El historial de precios USD se persiste en Google Sheets (HMM_Historial)
  con un rolling window de 500 snapshots (~1.5 días de trading a 60s/ciclo).
  Esto garantiza que el HMM arranque con historia aunque la app se reinicie.

FETCH DE BARRAS:
  Se fetchea símbolo por símbolo (no en batch) para garantizar que Alpaca
  retorne exactamente `limit` barras por símbolo. El endpoint multi-símbolo
  reparte el limit entre todos, resultando en pocas barras por símbolo.

CAMBIOS REALIZADOS:
  - Eliminado VIST de PARES (baja liquidez, ratio 3:1 → ruido)
  - YPFD y TGSU2 piden umbral -0.65% automáticamente
  - Logs diagnósticos distinguen bloques por HMM vs umbral

PENDIENTE (dinero real):
- Lógica de órdenes IOL (compra/venta)
- Manejo de puntas y precio límite
- Ver bloque marcado con # TODO: REAL TRADING
"""

import time, json, logging, smtplib, statistics
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

TZ_ARG = ZoneInfo("America/Argentina/Buenos_Aires")

def hora_argentina():
    """Retorna datetime actual en horario de Argentina."""
    return datetime.now(TZ_ARG)

from iol_client import IOLClient
from alpaca_client import AlpacaClient
from simulator import Simulador, SIMBOLOS_VOLATILES, UMBRAL_COMPRA_VOLATIL
from sheets_manager import SheetsManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="CCL Arbitrage", page_icon="📊", layout="wide")

REFRESH_SECONDS  = 60
# Apertura ajustada por DST de EEUU (desde segundo domingo de marzo):
# NYSE abre 9:30 ET (UTC-4) = 10:30 ART (UTC-3)
# Cierre sin cambio: NYSE cierra 16:00 ET = 17:00 ART siempre
HORA_APERTURA    = dtime(10, 30)
HORA_STOP_COMPRA = dtime(16, 30)
HORA_CIERRE      = dtime(16, 50)

# Rolling window del historial en memoria (consistente con HMM_Historial en Sheets)
HMM_MAX_SNAPSHOTS = 500

# PARES: {IOL_symbol: (Alpaca_symbol, ratio)}
# VIST removido por baja liquidez y ratio extremo (3:1)
# LOMA removido (no disponible en IOL)
PARES = {
    "GGAL":  ("GGAL",   10), "YPFD":  ("YPF",    1),
    "PAMP":  ("PAM",    25), "CEPU":  ("CEPU",  10),
    "AMZN":  ("AMZN",  144), "MSFT":  ("MSFT",  30),
    "NVDA":  ("NVDA",   24), "TSLA":  ("TSLA",  15),
    "AAPL":  ("AAPL",   20), "META":  ("META",  24),
    "GOOGL": ("GOOGL",  58), "meli":  ("MELI", 120),
    "BMA":   ("BMA",    10), "SUPV":  ("SUPV",   5),
    "TGSU2": ("TGS",     5), "TXR":   ("TX",     4),
    "GLD":   ("GLD",    50), "IBIT":  ("IBIT",  10),
    "SPY":   ("SPY",    20),
}

# ── SECRETS ───────────────────────────────────────────────
def get_secrets():
    try:
        return {
            "iol_user":   st.secrets["IOL_USER"],
            "iol_pass":   st.secrets["IOL_PASS"],
            "alp_key":    st.secrets["ALPACA_KEY"],
            "alp_secret": st.secrets["ALPACA_SECRET"],
            "gmail_user": st.secrets["GMAIL_USER"],
            "gmail_pass": st.secrets["GMAIL_APP_PASS"],
            "gcp":        json.loads(st.secrets["GCP_SERVICE_ACCOUNT"]),
        }
    except:
        return None

# ── SESSION STATE ──────────────────────────────────────────
def init_state():
    s = get_secrets()
    if not s:
        return False
    if "sim" not in st.session_state:
        st.session_state.iol      = IOLClient(s["iol_user"], s["iol_pass"])
        st.session_state.iol.login()
        st.session_state.alpaca   = AlpacaClient(s["alp_key"], s["alp_secret"])
        sh = SheetsManager(s["gcp"])
        sh.conectar()
        st.session_state.sheets   = sh
        st.session_state.historial = sh.cargar_historial_ccl()
        sim = Simulador()
        sh.cargar_estado_simulador(sim)
        sh.cargar_posiciones(sim)
        st.session_state.sim = sim  # FIX: asignar sim a session_state

        # FIX: pre-poblar ops_guardadas desde Sheets al arrancar.
        # Usa clave compuesta (id, ts_entry) porque _op_counter se resetea
        # cada jornada y el mismo ID puede repetirse en días distintos.
        # Esto evita que un reinicio de la app vuelva a guardar operaciones
        # que ya estaban escritas en Sheets antes del crash.
        try:
            ops_existentes = sh.cargar_operaciones()
            st.session_state.ops_guardadas = {
                (fila[0], fila[10]) for fila in ops_existentes  # [0]=id, [10]=ts_entry
            }
        except Exception as e:
            logger.warning(f"No se pudo pre-cargar ops_guardadas: {e}")
            st.session_state.ops_guardadas = set()

        st.session_state.gmail        = {"user": s["gmail_user"], "pass": s["gmail_pass"]}
        st.session_state.alertadas    = {}
        st.session_state.ciclos_warmup = 0
        st.session_state.ready         = True
    return True

# ── HMM — MODELO SIMONS (barras 1D) ──────────────────────
HMM_BARRAS_REFRESH_MIN = 10   # refrescar barras cada N minutos
HMM_BARRAS_LOOKBACK    = 252  # cantidad de barras 1D a pedir (~1 año)

def _fetch_barras_1d():
    """
    Descarga barras 1D via AlpacaClient para todos los simbolos USD en PARES.

    IMPORTANTE: se fetchea simbolo por simbolo, NO en batch.
    El endpoint multi-simbolo de Alpaca interpreta `limit` como total entre
    todos los simbolos, resultando en ~12 barras por simbolo con 20 simbolos.
    Fetching individual garantiza exactamente `limit` barras por simbolo.

    Retorna dict {sym_usd: [close, close, ...]} o {} si falla.
    """
    alpaca = st.session_state.get("alpaca")
    if not alpaca:
        return {}

    syms_usd  = list({v[0] for v in PARES.values()})
    resultado = {}

    for sym in syms_usd:
        barras = alpaca.get_bars([sym], timeframe="1Day", limit=HMM_BARRAS_LOOKBACK)
        if sym in barras:
            resultado[sym] = barras[sym]
            logger.info(f"HMM fetch: {sym} → {len(barras[sym])} barras")
        else:
            logger.warning(f"HMM fetch: {sym} → sin datos")

    return resultado

def _refrescar_barras_si_necesario():
    """Refresca el cache de barras 1D si pasaron mas de HMM_BARRAS_REFRESH_MIN minutos."""
    ahora  = hora_argentina()
    ultimo = st.session_state.get("hmm_barras_ts")
    if ultimo is None or (ahora - ultimo).seconds >= HMM_BARRAS_REFRESH_MIN * 60:
        barras = _fetch_barras_1d()
        if barras:
            st.session_state["hmm_barras"]    = barras
            st.session_state["hmm_barras_ts"] = ahora

# ── HMM (Simons) ──────────────────────────────────────────
def entrenar_hmm_simbolo(sym_usd: str, closes: list) -> Optional[dict]:
    """
    Entrena un HMM de 2 estados sobre log-returns 1D.

    Retorna:
        {"mean_bull": float, "mean_bear": float, "state": str}

    Guard: si mean_bull < -0.0005, ambos estados son bajistas → retorna 🔴.
    """
    if len(closes) < 63:
        return None

    try:
        from hmmlearn.hmm import GaussianHMM
        closes_arr = np.array(closes, dtype=float)
        log_returns = np.diff(np.log(closes_arr)).reshape(-1, 1)

        model = GaussianHMM(n_components=2, random_state=42, n_iter=200)
        model.fit(log_returns)

        means = model.means_.flatten()
        bull_idx = np.argmax(means)
        bear_idx = 1 - bull_idx

        mean_bull = means[bull_idx]
        mean_bear = means[bear_idx]

        # Guard: si el estado "bull" tiene media negativa, ambos son bajistas
        if mean_bull < -0.0005:
            return {"mean_bull": mean_bull, "mean_bear": mean_bear, "state": "🔴"}

        return {"mean_bull": mean_bull, "mean_bear": mean_bear, "state": "🟢"}

    except Exception as e:
        logger.error(f"HMM error para {sym_usd}: {e}")
        return None

# ── FETCH PRECIOS ──────────────────────────────────────────
def fetch_precios():
    iol    = st.session_state.get("iol")
    alpaca = st.session_state.get("alpaca")
    if not iol or not alpaca:
        return None, None

    # Precios ARS: batch CEDEARs + MerVal
    iols_cedears = list({v[0] for v in PARES.values() if v[0] not in ["GGAL","YPFD","PAMP","CEPU","BMA","SUPV","TXR"]})
    iols_merval  = [k for k in PARES.keys() if k in ["GGAL","YPFD","PAMP","CEPU","BMA","SUPV","TXR"]]

    p_ars = {}
    if iols_cedears:
        p_ars.update(iol.get_panel(iols_cedears))
    if iols_merval:
        p_ars.update(iol.get_panel(iols_merval))

    # Precios USD: snapshots individual
    syms_usd = list({v[0] for v in PARES.values()})
    snapshots = alpaca.get_snapshots(syms_usd)
    p_usd = {k: v["price"] for k, v in snapshots.items() if "price" in v}

    return p_ars, p_usd

# ── LÓGICA DE SEÑALES ──────────────────────────────────────
def calcular_climas():
    """
    Calcula clima HMM para cada símbolo USD.

    Retorna dict {sym_iol: clima_str} donde clima_str es "🟢 BULL" o "🔴 BEAR".
    """
    _refrescar_barras_si_necesario()
    barras = st.session_state.get("hmm_barras", {})
    climas = {}

    for sym_iol, (sym_usd, _) in PARES.items():
        closes = barras.get(sym_usd, [])
        if not closes:
            climas[sym_iol] = "🔴 BEAR"  # fallback: sin datos → bear
            continue

        # Agregar retorno intradiario como última observación
        if len(closes) >= 2:
            última_cierre_1d = closes[-1]
            últimos_snapshots = st.session_state.historial[-1:] if st.session_state.historial else []
            if últimos_snapshots and sym_usd in últimos_snapshots:
                snap = últimos_snapshots[0]
                precio_actual = snap.get(sym_usd, última_cierre_1d)
                log_ret_intra = np.log(precio_actual / última_cierre_1d)
                closes_aug = list(closes) + [último_cierre_1d * np.exp(log_ret_intra)]
            else:
                closes_aug = closes
        else:
            closes_aug = closes

        result = entrenar_hmm_simbolo(sym_usd, closes_aug)
        climas[sym_iol] = result["state"] if result else "🔴 BEAR"

    return climas

def calcular_señales(ccl_map: dict, ccl_avg: float, climas: dict) -> dict:
    """Calcula señales de compra/venta para cada símbolo."""
    señales = {}
    for sym_iol, (sym_usd, ratio) in PARES.items():
        ccl = ccl_map.get(sym_iol, 0)
        if ccl <= 0 or ccl_avg == 0:
            señales[sym_iol] = ("🔴 SIN DATO", None, None)
            continue

        dev = (ccl / ccl_avg - 1) * 100
        clima = climas.get(sym_iol, "🔴 BEAR")

        # Obtener umbral específico del símbolo (respeta la lógica del simulador)
        if sym_iol in SIMBOLOS_VOLATILES:
            umbral = UMBRAL_COMPRA_VOLATIL
        else:
            umbral = -0.50  # default

        # Lógica de señal
        if dev >= 0.15:
            señal = "🔴 VENTA"
        elif dev < umbral and clima == "🟢 BULL":
            señal = "🚀 COMPRA"
        elif dev < -0.50 and clima == "🔴 BEAR":
            señal = "🟡 ESPERA (sin clima)"
        else:
            señal = "⚪ OBSERVAR"

        señales[sym_iol] = (señal, dev, clima)

    return señales

# ── ALERTAS ────────────────────────────────────────────────
def enviar_email(asunto: str, cuerpo: str):
    """Envía alerta por Gmail."""
    gmail = st.session_state.get("gmail")
    if not gmail:
        return

    try:
        msg = MIMEMultipart()
        msg["From"]    = gmail["user"]
        msg["To"]      = gmail["user"]
        msg["Subject"] = asunto
        msg.attach(MIMEText(cuerpo, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail["user"], gmail["pass"])
            server.sendmail(gmail["user"], gmail["user"], msg.as_string())
        logger.info(f"✉️ Email enviado: {asunto}")
    except Exception as e:
        logger.error(f"Email error: {e}")

def alerta_señales(señales_alerta: list, ccl_avg: float):
    """Alerta de señales detectadas."""
    if not señales_alerta:
        return
    for sym, dev, clima in señales_alerta:
        cuerpo = f"🚀 COMPRA detectada\nActivo: {sym}\nDesvío: {dev:+.2f}%\nClima: {clima}\nCCL promedio: ${ccl_avg:.2f}"
        enviar_email(f"🚨 GG: SEÑAL {sym}", cuerpo)
        time.sleep(0.5)

def alerta_operacion(abiertas: list, cerradas: list, ccl_avg: float):
    """Alerta de operaciones ejecutadas."""
    for pos in abiertas:
        cuerpo = f"Compra ejecutada\nActivo: {pos.symbol}\nCantidad: {pos.cantidad:.2f}\nPrecio: ${pos.precio_entry:.2f}\nMonto: ${pos.monto_entry:,.0f}"
        enviar_email(f"💹 GG: COMPRA {pos.symbol}", cuerpo)
        time.sleep(0.5)

    for op in cerradas:
        emoji = "✅" if op.pnl > 0 else "❌"
        cuerpo = f"{emoji} Operación cerrada [{op.motivo_cierre}]\nActivo: {op.symbol}\nPnL: ${op.pnl:+,.0f} ({op.pnl_pct:+.2f}%)"
        enviar_email(f"💹 GG: CIERRE {op.symbol}", cuerpo)
        time.sleep(0.5)

# ── MAIN ───────────────────────────────────────────────────
def main():
    if not init_state():
        st.error("❌ Error inicializando sistema (falta credenciales)")
        return

    iol    = st.session_state.iol
    alpaca = st.session_state.alpaca
    sheets = st.session_state.sheets
    sim    = st.session_state.sim
    historial = st.session_state.historial

    ahora = hora_argentina()
    st.title("📊 GG Investments — CCL Arbitrage")
    st.caption(f"⏰ {ahora.strftime('%Y-%m-%d %H:%M:%S')} ART")

    # Fetch de precios
    p_ars, p_usd = fetch_precios()
    if not p_ars or not p_usd:
        st.error("❌ Error fetching precios")
        return

    # Cálculo de CCL
    ccl_map = {}
    for sym_iol, (sym_usd, ratio) in PARES.items():
        if sym_iol in p_ars and sym_usd in p_usd:
            p_a = p_ars[sym_iol]
            p_u = p_usd[sym_usd]
            ccl_map[sym_iol] = (p_a / p_u) * ratio

    if not ccl_map:
        st.error("❌ Sin datos de CCL")
        return

    ccl_avg = statistics.median(ccl_map.values())

    # HMM y señales
    climas = calcular_climas()
    señales = calcular_señales(ccl_map, ccl_avg, climas)

    # Procesar ciclo del simulador
    rows = []
    señales_alerta = []

    for sym_iol, (sym_usd, ratio) in PARES.items():
        if sym_iol not in ccl_map or sym_iol not in p_ars:
            continue

        ccl = ccl_map[sym_iol]
        dev = (ccl / ccl_avg - 1) * 100
        clima = climas.get(sym_iol, "🔴 BEAR")
        señal_txt, _, _ = señales[sym_iol]

        # Desvío visual
        if dev >= 0.15:
            desvio_color = "🔴"
        elif dev < -0.50:
            desvio_color = "🟢"
        else:
            desvio_color = "🟡"

        rows.append({
            "sym": sym_iol,
            "p_ars": p_ars[sym_iol],
            "p_usd": p_usd[sym_usd],
            "ccl": ccl,
            "dev": dev,
            "desvio_color": desvio_color,
            "clima": clima,
            "accion": señal_txt,
        })

        if "COMPRA" in señal_txt:
            señales_alerta.append((sym_iol, dev, clima))

    # Persistir snapshots HMM
    snapshot_usd = {sym_usd: p_usd[sym_usd] for _, (sym_usd, _) in PARES.items() if sym_usd in p_usd}
    if snapshot_usd:
        historial.append(snapshot_usd)
        if len(historial) > HMM_MAX_SNAPSHOTS:
            historial = historial[-HMM_MAX_SNAPSHOTS:]
        st.session_state.historial = historial
        sheets.guardar_historial_ccl(historial)

    # Procesar ciclo simulator
    resultado = sim.procesar_ciclo(ccl_map, ccl_avg, p_ars, climas, ahora.time())
    ops_abiertas = resultado.get("abiertas", [])
    ops_cerradas = resultado.get("cerradas", [])

    hay_cambios  = bool(ops_abiertas or ops_cerradas)

    # FIX: guard con clave compuesta (id, ts_entry) para sobrevivir reinicios.
    # El _op_counter se resetea cada jornada, por lo que el mismo ID (ej. P0001)
    # puede aparecer en días distintos. La clave compuesta es globalmente única.
    for op in ops_cerradas:
        clave = (op.id, op.ts_entry)
        if clave not in st.session_state.ops_guardadas:
            sheets.guardar_operacion(sim.fila_sheets_operacion(op))
            st.session_state.ops_guardadas.add(clave)

    ciclo_actual = int(time.time() // REFRESH_SECONDS)
    if hay_cambios or ciclo_actual % 5 == 0:
        sheets.guardar_estado_cartera(sim.fila_sheets_estado(p_ars))

    if hay_cambios:
        sheets.guardar_posiciones(sim)
        sheets.guardar_estado_simulador(sim)

    alerta_operacion(ops_abiertas, ops_cerradas, ccl_avg)

    # ── TODO: REAL TRADING ─────────────────────────────
    # Cuando se opere con dinero real en IOL, agregar aquí:
    #
    # for pos in ops_abiertas:
    #     iol.place_order(
    #         symbol   = pos.symbol,
    #         cantidad = pos.cantidad,
    #         precio   = pos.precio_entry,
    #         tipo     = "compra",
    #         mercado  = "bCBA",
    #     )
    #
    # for op in ops_cerradas:
    #     if op.motivo_cierre != "CIERRE_FORZADO":
    #         iol.place_order(
    #             symbol   = op.symbol,
    #             cantidad = op.cantidad,
    #             precio   = op.precio_exit,
    #             tipo     = "venta",
    #             mercado  = "bCBA",
    #         )
    # ──────────────────────────────────────────────────

    alerta_señales(señales_alerta, ccl_avg)

    # ── KPIs ──────────────────────────────────────────────
    resumen = sim.resumen(p_ars)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CCL Promedio",   f"${ccl_avg:.2f}")
    c2.metric("Capital Total",  f"${resumen['capital_total']:,.0f}", f"{resumen['pnl_pct']:+.2f}%")
    c3.metric("Efectivo",       f"${resumen['efectivo']:,.0f}")
    c4.metric("En Posiciones",  f"${resumen['en_posiciones']:,.0f}")
    c5.metric("Win Rate",       f"{resumen['win_rate']:.0f}%", f"{resumen['operaciones_total']} ops")

    # ── Estado mercado ─────────────────────────────────────
    if ahora.time() < HORA_APERTURA:
        st.warning(f"⏳ Mercado abre a las {HORA_APERTURA.strftime('%H:%M')} hs")
    elif ahora.time() >= HORA_CIERRE:
        st.error("🔴 16:50 hs — Cierre forzado de posiciones activo")
    elif ahora.time() >= HORA_STOP_COMPRA:
        st.warning("⚠️ 16:30 hs — Sin nuevas compras | Solo cierres")
    else:
        st.success(f"🟢 Mercado abierto | {resumen['posiciones_abiertas']} posiciones abiertas")

    # ── Gráfico ────────────────────────────────────────────
    rows_sorted = sorted(rows, key=lambda x: x["dev"])
    colors = [
        "#00C851" if r["accion"] == "🚀 COMPRA"
        else "#FF4444" if r["accion"] == "🔴 VENTA"
        else "#888"
        for r in rows_sorted
    ]
    fig = go.Figure(go.Bar(
        x=[r["sym"] for r in rows_sorted],
        y=[r["dev"] for r in rows_sorted],
        marker_color=colors,
        text=[f"{r['dev']:+.2f}%" for r in rows_sorted],
        textposition="outside",
    ))
    fig.add_hline(y=0.15, line_dash="dash", line_color="#FF4444", annotation_text="+0.15%")
    fig.add_hline(y=-0.50, line_dash="dash", line_color="#00C851", annotation_text="-0.50%")
    fig.add_hline(y=-0.65, line_dash="dash", line_color="#FFB800", annotation_text="-0.65% (volátiles)")
    fig.update_layout(
        title="Desviación CCL vs Promedio",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font_color="white", height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla señales ──────────────────────────────────────
    st.subheader("📋 Señales en Tiempo Real")
    df = pd.DataFrame([{
        "Activo": r["sym"],
        "P. ARS": f"${r['p_ars']:,.2f}",
        "P. USD": f"${r['p_usd']:.2f}",
        "CCL":    f"${r['ccl']:,.2f}",
        "Desvío": f"{r['desvio_color']} {r['dev']:+.2f}%",
        "Clima":  r["clima"],
        "Acción": r["accion"],
    } for r in rows_sorted])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Posiciones abiertas ────────────────────────────────
    if any(sim.posiciones.values()):
        st.subheader("💼 Posiciones Abiertas")
        for sym, poss in sim.posiciones.items():
            for pos in poss:
                p_entry = getattr(pos, 'precio_entry', 0)
                m_entry = getattr(pos, 'monto_entry', 0)
                p_id    = getattr(pos, 'id', 'S/N')

                if p_entry <= 0 or m_entry <= 0:
                    st.warning(f"⚠️ Dato inválido en {sym} (ID: {p_id}). Revisar Google Sheets.")
                    continue

                precio_actual = p_ars.get(sym, p_entry)
                pnl     = (precio_actual - p_entry) * pos.cantidad
                pnl_pct = ((precio_actual / p_entry) - 1) * 100
                emoji   = "✅" if pnl >= 0 else "🔻"

                with st.expander(f"{emoji} {p_id} — {sym} | PnL: ${pnl:+,.0f} ({pnl_pct:+.2f}%)", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Entrada",   f"${p_entry:,.1f}")
                    c2.metric("Actual",    f"${precio_actual:,.1f}", f"{pnl_pct:+.2f}%")
                    c3.metric("Invertido", f"${m_entry:,.0f}")
                    c4.metric("Cantidad",  f"{pos.cantidad:.2f} u.")

                    btn_key = f"v_manual_{p_id}_{sym}_{int(p_entry)}"
                    if st.button(f"🔴 Vender {sym} ({p_id})", key=btn_key, type="primary"):
                        op = sim.cerrar_posicion(sym, pos, precio_actual, "VENTA_MANUAL")
                        sim.posiciones[sym] = [p for p in sim.posiciones[sym] if p.id != pos.id]
                        sheets.guardar_operacion(sim.fila_sheets_operacion(op))
                        sheets.guardar_posiciones(sim)
                        sheets.guardar_estado_simulador(sim)
                        st.success(f"✅ Venta registrada para {sym}")
                        time.sleep(1)
                        st.rerun()

    # ── Historial ops ──────────────────────────────────────
    with st.expander("📜 Historial de Operaciones"):
        try:
            ops = sheets.cargar_operaciones()
            if ops:
                cols = ["ID","Activo","Tipo","Cant.","P.Entry","P.Exit",
                        "M.Entry","M.Exit","PnL","PnL%","Apertura","Cierre","Motivo"]
                st.dataframe(pd.DataFrame(ops, columns=cols), use_container_width=True, hide_index=True)
            else:
                st.info("Sin operaciones registradas aún.")
        except Exception as e:
            st.warning(f"⚠️ Error cargando historial (Sheets rate limit): {e}")

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Config")
        st.markdown(f"**Snapshots HMM:** {len(historial)}")
        st.markdown(f"**Actualizado:** {ahora.strftime('%H:%M:%S')}")

        # Debug barras HMM: muestra cuántas barras tiene cada símbolo en cache.
        # ✅ = >= 63 barras (HMM activo)  |  ⚠️ = < 63 (HMM fallback 🔴)
        barras_cache = st.session_state.get("hmm_barras", {})
        if barras_cache:
            ts_cache = st.session_state.get("hmm_barras_ts")
            ts_str   = ts_cache.strftime("%H:%M:%S") if ts_cache else "—"
            with st.expander(f"📊 Barras HMM ({ts_str})", expanded=False):
                for s, b in sorted(barras_cache.items()):
                    icono = "✅" if len(b) >= 63 else "⚠️"
                    st.caption(f"{icono} {s:<6} {len(b)} barras")

        st.divider()
        st.markdown("**Simulador**")
        st.markdown(f"Capital inicial: $10.000.000")
        st.markdown(f"Por operación: 15%")
        st.markdown(f"Máx./especie: 2")
        st.markdown(f"Umbral base: -0.50% | Volátiles: -0.65%")
        st.markdown(f"Ventana: 10:30 → 16:50")
        st.divider()
        if st.button("🔄 Reset simulador"):
            sim_nuevo = Simulador()
            sheets.guardar_posiciones(sim_nuevo)
            sheets.guardar_estado_simulador(sim_nuevo)
            st.session_state.sim = sim_nuevo
            st.success("✅ Simulador reseteado")
            st.rerun()

    st.caption(f"⏱ Próxima actualización en {REFRESH_SECONDS}s")
    time.sleep(REFRESH_SECONDS)
    st.rerun()


if __name__ == "__main__":
    main()
