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

HISTORIAL HMM:
  El historial de precios USD se persiste en Google Sheets (HMM_Historial)
  con un rolling window de 500 snapshots (~1.5 días de trading a 60s/ciclo).
  Esto garantiza que el HMM arranque con historia aunque la app se reinicie.

FETCH DE BARRAS:
  Se fetchea símbolo por símbolo (no en batch) para garantizar que Alpaca
  retorne exactamente `limit` barras por símbolo. El endpoint multi-símbolo
  reparte el limit entre todos, resultando en pocas barras por símbolo.

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
from simulator import Simulador
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

PARES = {
    "GGAL":  ("GGAL",   10), "YPFD":  ("YPF",    1),
    "PAMP":  ("PAM",    25), "CEPU":  ("CEPU",  10),
    "AMZN":  ("AMZN",  144), "MSFT":  ("MSFT",  30),
    "NVDA":  ("NVDA",   24), "TSLA":  ("TSLA",  15),
    "AAPL":  ("AAPL",   20), "META":  ("META",  24),
    "GOOGL": ("GOOGL",  58), "MELI":  ("MELI", 120),
    "BMA":   ("BMA",    10), "SPY":   ("SPY",   20),
    "TGSU2": ("TGS",     5), "IBIT":  ("IBIT",  10), 
    "GLD":   ("GLD",    50),
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
        
        # Umbrales dinámicos por franja horaria
        umbrales_por_hora = {
            "09:50-11:30": -0.0065,     # Apertura — más exigente
            "11:31-15:29": -0.0050,     # Mediodía — estándar
            "15:30-16:29": -0.0065,     # Cierre — más exigente
            "16:30+": None              # Bloqueado
        }
        
        sim = Simulador(umbrales_por_hora=umbrales_por_hora)
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
        st.session_state.vp_cache    = {"timestamp": None, "intraday": None, "dos_semanas": None}  # Volume Profile cache
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
            conteo = {s: len(b) for s, b in barras.items()}
            logger.info(f"HMM 1D: barras actualizadas → {conteo}")

def clima_hmm(sym, historial=None):
    """
    Retorna verde si el subyacente USD esta en regimen bull, rojo si no.
    Usa barras 1D de Alpaca fetched individualmente (cache de 10 min).

    Identificacion de estado BULL por argmax de medias puras.
    Guard: si la media mas alta es negativa, ambos estados son bajistas
    (mercado bear con rebotes) y se retorna rojo directamente.

    Esto evita el problema del Sharpe en activos iliquidos: el estado
    "quieto" (returns ~0, std muy baja) tiene Sharpe artificialmente alto
    y era incorrectamente clasificado como BULL.

    RETORNO INTRADIARIO: el modelo se entrena solo con cierres historicos
    (sin look-ahead), pero al predecir se agrega el retorno de hoy
    (precio actual vs ultimo cierre) como ultima observacion. Esto permite
    que movimientos fuertes del dia corriente (como +2% en YPF) sean
    considerados al clasificar el regimen actual.

    Requiere minimo 63 barras (1 trimestre) para HMM estadisticamente estable.
    """
    sym_usd = PARES[sym][0]
    barras  = st.session_state.get("hmm_barras", {})
    precios = barras.get(sym_usd, [])

    if len(precios) < 63:
        return "🔴"

    try:
        from hmmlearn.hmm import GaussianHMM

        # Entrenar solo con cierres historicos (sin look-ahead)
        ret_hist = np.diff(np.log(precios)).reshape(-1, 1)
        m = GaussianHMM(n_components=2, random_state=42, n_iter=200).fit(ret_hist)

        means = m.means_.flatten()
        bull  = int(np.argmax(means))

        # Si la media del estado "bull" es negativa, ambos estados son
        # bajistas (mercado bear con rebotes violentos, no bull genuino)
        if means[bull] < -0.0005:
            return "🔴"

        # Predecir incluyendo retorno intradiario de hoy si esta disponible
        # precio actual (snapshot Alpaca) vs ultimo cierre historico
        ret_pred = ret_hist.copy()
        if historial:
            precio_actual = historial[-1].get("usd", {}).get(sym_usd)
            ultimo_cierre = precios[-1]
            if precio_actual and ultimo_cierre > 0:
                ret_hoy = np.log(precio_actual / ultimo_cierre)
                ret_pred = np.vstack([ret_hist, [[ret_hoy]]])
                logger.info(f"HMM {sym_usd}: ret_hoy={ret_hoy:+.4f} ({precio_actual:.2f} vs cierre {ultimo_cierre:.2f})")

        estado = m.predict(ret_pred)[-1]
        return "🟢" if estado == bull else "🔴"
    except:
        return "🔴"

# ── CCL ───────────────────────────────────────────────────
def calcular_ccl(p_ars, p_usd):
    ccl_map = {}
    for sym, (sym_usd, ratio) in PARES.items():
        a = p_ars.get(sym)
        u = p_usd.get(sym_usd)
        if a and u and u > 0:
            ccl_map[sym] = (a / u) * ratio
    avg = statistics.median(ccl_map.values()) if ccl_map else 0
    return ccl_map, avg

# ── GMAIL ─────────────────────────────────────────────────
def enviar_mail(subject: str, cuerpo: str):
    g = st.session_state.gmail
    if not g["user"]:
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = msg["To"] = g["user"]
        msg["Subject"] = subject
        msg.attach(MIMEText(cuerpo, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(g["user"], g["pass"])
            smtp.send_message(msg)
        logger.info(f"📧 Mail enviado: {subject}")
    except Exception as e:
        logger.error(f"Gmail error: {e}")

def alerta_señales(señales, ccl_avg):
    if not señales:
        return
    ahora = hora_argentina()
    nuevas = [s for s in señales
              if (ahora - st.session_state.alertadas.get(s["sym"], datetime(2000,1,1,tzinfo=TZ_ARG))).seconds > 1800]
    if not nuevas:
        return
    for s in nuevas:
        st.session_state.alertadas[s["sym"]] = ahora
    cuerpo = f"📊 GG HMM-CCL — Señales Activas\n"
    cuerpo += f"CCL Promedio: ${ccl_avg:.2f} | {ahora.strftime('%d/%m/%Y %H:%M:%S')}\n"
    cuerpo += f"{'─'*40}\n"
    for s in nuevas:
        cuerpo += f"{s['sym']:<8} {s['dev']:>+7.2f}%  {s['clima']}  {s['señal']}\n"
    enviar_mail(f"🚨 GG: {len(nuevas)} señal(es) | CCL ${ccl_avg:.0f}", cuerpo)

def alerta_operacion(ops_abiertas, ops_cerradas, ccl_avg):
    if not ops_abiertas and not ops_cerradas:
        return
    ahora = hora_argentina()
    cuerpo = f"💹 GG HMM-CCL — Operación Ejecutada\n"
    cuerpo += f"CCL Promedio: ${ccl_avg:.2f} | {ahora.strftime('%d/%m/%Y %H:%M:%S')}\n"
    cuerpo += f"{'─'*40}\n"
    if ops_abiertas:
        cuerpo += f"\n🟢 COMPRAS ({len(ops_abiertas)}):\n"
        for pos in ops_abiertas:
            cuerpo += f"  {pos.symbol:<8} ${pos.precio_entry:,.1f} | Monto: ${pos.monto_entry:,.0f}\n"
    if ops_cerradas:
        cuerpo += f"\n🔴 VENTAS ({len(ops_cerradas)}):\n"
        for op in ops_cerradas:
            emoji = "✅" if op.pnl > 0 else "❌"
            cuerpo += f"  {emoji} {op.symbol:<8} PnL: ${op.pnl:+,.0f} ({op.pnl_pct:+.2f}%) [{op.motivo_cierre}]\n"
    n = len(ops_abiertas) + len(ops_cerradas)
    enviar_mail(f"💹 GG: {n} op(s) ejecutada(s) | CCL ${ccl_avg:.0f}", cuerpo)

# ── FETCH ─────────────────────────────────────────────────
def fetch_precios(ts_key):
    iol    = st.session_state.iol
    alpaca = st.session_state.alpaca
    p_ars  = {}

    CEDEARS_SET = {"AAPL","AMZN","MSFT","NVDA","TSLA","META","GOOGL","MELI","GLD","IBIT","SPY"}
    MERVAL_SET  = {"GGAL","YPFD","PAMP","CEPU","BMA","TGSU2"}

    # Batch 1 — CEDEARs (1 request)
    try:
        data = iol.get_panel("CEDEARs")
        for t in data:
            if t["simbolo"] in CEDEARS_SET:
                p_ars[t["simbolo"]] = t["ultimoPrecio"]
    except Exception as e:
        st.warning(f"IOL CEDEARs batch: {e}")

    # Batch 2 — MerVal (1 request)
    try:
        data = iol.get_panel("MerVal")
        for t in data:
            if t["simbolo"] in MERVAL_SET:
                p_ars[t["simbolo"]] = t["ultimoPrecio"]
    except Exception as e:
        st.warning(f"IOL MerVal batch: {e}")

    syms_usd = list({v[0] for v in PARES.values()})
    snaps    = alpaca.get_snapshots(syms_usd)
    p_usd    = {s: snaps[s]["last"] for s in snaps if snaps[s].get("last")}
    return p_ars, p_usd

# ── MAIN ──────────────────────────────────────────────────
def main():
    st.title("📊 GG Investments 🦅🤑")
    st.caption("IOL (ARS) + Alpaca (USD) | HMM Climate | Simulador Intradiario")

    if not init_state():
        st.error("⚠️ Configurar credenciales en Streamlit Secrets.")
        return

    sheets    = st.session_state.sheets
    sim       = st.session_state.sim
    historial = st.session_state.historial
    hora      = hora_argentina()
    ahora     = hora.time()

    # Refrescar barras 1D para HMM (cada 10 min, no cada 60s)
    _refrescar_barras_si_necesario()

    # Warmup: los primeros 2 ciclos no se opera para estabilizar HMM y precios
    WARMUP_CICLOS = 2
    if 'ciclos_warmup' not in st.session_state:
        st.session_state.ciclos_warmup = 0
    en_warmup = st.session_state.ciclos_warmup < WARMUP_CICLOS
    st.session_state.ciclos_warmup += 1

    ts_key = str(int(time.time() // REFRESH_SECONDS))
    p_ars, p_usd = fetch_precios(ts_key)

    ccl_map, ccl_avg = calcular_ccl(p_ars, p_usd)

    if ccl_map:
        historial.append({"ts": hora.isoformat(), "ccl": ccl_map, "avg": ccl_avg, "usd": p_usd})
        if len(historial) > HMM_MAX_SNAPSHOTS:
            historial[:] = historial[-HMM_MAX_SNAPSHOTS:]
        sheets.guardar_snapshot_ccl(ccl_map, ccl_avg, p_usd=p_usd)

    # ── Señales y climas ───────────────────────────────────
    rows, señales_alerta, climas = [], [], {}
    for sym, ccl in ccl_map.items():
        dev   = (ccl / ccl_avg - 1) * 100 if ccl_avg else 0
        clima = clima_hmm(sym, historial)
        climas[sym] = "🟢 BULL" if clima == "🟢" else "🔴 BEAR"

        if dev <= -0.5:
            desvio_color = "🟢"
        elif dev >= 0.1:
            desvio_color = "🔴"
        else:
            desvio_color = "🟡"

        if dev <= -0.5 and clima == "🟢":
            accion = "🚀 COMPRA"
        elif dev >= 0.1:
            accion = "🔴 VENTA"
        else:
            accion = "⏳ ESPERAR"

        rows.append({
            "sym": sym, "ccl": ccl, "dev": dev, "clima": clima,
            "desvio_color": desvio_color, "accion": accion,
            "p_ars": p_ars.get(sym, 0),
            "p_usd": p_usd.get(PARES[sym][0], 0),
        })
        if accion == "🚀 COMPRA":
            señales_alerta.append({"sym": sym, "dev": dev, "clima": clima, "señal": accion})
        elif accion == "🔴 VENTA":
            if sim.posiciones_abiertas_count(sym) > 0:
                señales_alerta.append({"sym": sym, "dev": dev, "clima": clima, "señal": accion})

    # ── Simulador ──────────────────────────────────────────
    if HORA_APERTURA <= ahora:
        ahora_sim = dtime(0, 0) if en_warmup else ahora
        resultado = sim.procesar_ciclo(ccl_map, ccl_avg, p_ars, climas, ahora_sim)
        ops_cerradas = resultado.get("cerradas", []) + resultado.get("forzadas", [])
        ops_abiertas = resultado.get("abiertas", [])
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
    if ahora < HORA_APERTURA:
        st.warning(f"⏳ Mercado abre a las {HORA_APERTURA.strftime('%H:%M')} hs")
    elif ahora >= HORA_CIERRE:
        st.error("🔴 16:50 hs — Cierre forzado de posiciones activo")
    elif ahora >= HORA_STOP_COMPRA:
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
    fig.add_hline(y=-0.5, line_dash="dash", line_color="#00C851", annotation_text="-0.50%")
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
        st.markdown(f"**Actualizado:** {hora.strftime('%H:%M:%S')}")

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
