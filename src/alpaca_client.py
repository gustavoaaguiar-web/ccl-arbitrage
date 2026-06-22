"""
Cliente Alpaca Markets para precios en tiempo real de ADRs argentinos en NYSE.
Usa WebSocket v2 de Alpaca Data API (plan gratuito incluido).
Docs: https://docs.alpaca.markets/reference/stocklatesttrade
"""

import logging
import threading
from typing import Callable, Dict, List, Optional
import requests
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

ALPACA_BASE_URL = "https://data.alpaca.markets/v2"
ALPACA_WS_URL   = "wss://stream.data.alpaca.markets/v2/iex"  # IEX = gratuito

# Símbolos ADR necesarios (del mapping en iol_client.py)
ADR_SYMBOLS = ["GGAL", "YPF", "PAM", "BMA", "TGSU2", "CEPU"]


class AlpacaClient:
    """
    Cliente REST + WebSocket para Alpaca Markets.
    - REST: precios actuales (snapshot) y barras OHLCV históricas
    - WebSocket: stream en tiempo real
    """

    def __init__(self, api_key: str, api_secret: str):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.headers    = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        self._prices: Dict[str, dict] = {}
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False

    # ─────────────────────────── REST SNAPSHOT ───────────────────────

    def get_snapshots(self, symbols: list = None) -> Dict[str, dict]:
        """
        Obtiene último precio, bid y ask para una lista de símbolos.
        Endpoint: GET /v2/stocks/snapshots
        """
        symbols = symbols or ADR_SYMBOLS
        try:
            resp = requests.get(
                f"{ALPACA_BASE_URL}/stocks/snapshots",
                headers=self.headers,
                params={"symbols": ",".join(symbols), "feed": "iex"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            result = {}
            for sym, snap in data.items():
                latest_trade = snap.get("latestTrade", {})
                latest_quote = snap.get("latestQuote", {})
                result[sym] = {
                    "symbol": sym,
                    "last":   latest_trade.get("p"),
                    "bid":    latest_quote.get("bp"),
                    "ask":    latest_quote.get("ap"),
                    "ts":     latest_trade.get("t"),
                }
            self._prices.update(result)
            return result
        except requests.RequestException as e:
            logger.error(f"Error Alpaca snapshots: {e}")
            return {}

    def get_price(self, symbol: str) -> Optional[float]:
        """Retorna el último precio conocido de un símbolo."""
        if symbol not in self._prices:
            self.get_snapshots([symbol])
        return self._prices.get(symbol, {}).get("last")

    # ─────────────────────────── BARRAS OHLCV ────────────────────────

    def get_bars(
        self,
        symbols: list,
        timeframe: str = "1Day",
        limit: int = 252,
        desde: Optional[str] = None,
        hasta: Optional[str] = None,
    ) -> Dict[str, List[dict]]:
        """
        Descarga barras OHLCV completas para una lista de símbolos.

        Retorna:
            {symbol: [{"t": ..., "o": ..., "h": ..., "l": ..., "c": ..., "v": ...}, ...]}

        Args:
            symbols:   lista de tickers
            timeframe: "1Day", "4Hour", "30Min", etc.
            limit:     máximo de barras a pedir (default 252 = ~1 año diario)
            desde:     fecha inicio ISO "YYYY-MM-DD" (tiene precedencia sobre limit)
            hasta:     fecha fin ISO "YYYY-MM-DD" (default: hoy)

        Si se pasa `desde`, se ignora el cálculo dinámico de lookback por limit
        y se usa la fecha directamente.
        """
        if desde:
            start = f"{desde}T00:00:00Z"
        else:
            # Calcular lookback dinámico según timeframe y limit
            if "Day" in timeframe:
                dias = int(limit * 1.6)
            elif "Hour" in timeframe:
                try:
                    horas = int(timeframe.replace("Hour", "").replace("h", ""))
                except ValueError:
                    horas = 1
                dias = int((limit * horas / 6.5) * 1.6) + 10
            elif "Min" in timeframe:
                try:
                    mins = int(timeframe.replace("Min", "").replace("min", ""))
                except ValueError:
                    mins = 30
                dias = int((limit * mins / (6.5 * 60)) * 1.6) + 5
            else:
                dias = limit * 2
            start = (datetime.now(timezone.utc) - timedelta(days=dias)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        params = {
            "symbols":   ",".join(symbols),
            "timeframe": timeframe,
            "start":     start,
            "limit":     limit,
            "feed":      "iex",
        }
        if hasta:
            params["end"] = f"{hasta}T23:59:59Z"

        try:
            resp = requests.get(
                f"{ALPACA_BASE_URL}/stocks/bars",
                headers=self.headers,
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("bars", {})

            result = {}
            for sym, bars in data.items():
                if len(bars) < 5:
                    logger.warning(f"Alpaca: {sym} solo tiene {len(bars)} barras — ignorado")
                    continue
                # Normalizar a dicts con claves consistentes
                result[sym] = [
                    {
                        "t": b.get("t", ""),
                        "o": float(b.get("o", 0)),
                        "h": float(b.get("h", 0)),
                        "l": float(b.get("l", 0)),
                        "c": float(b.get("c", 0)),
                        "v": float(b.get("v", 0)),
                    }
                    for b in bars
                ]

            logger.info(
                f"Alpaca bars ({timeframe}): "
                f"{', '.join(f'{s}={len(v)}bars' for s, v in result.items())}"
            )
            return result

        except requests.RequestException as e:
            logger.error(f"Error Alpaca bars ({timeframe}): {e}")
            return {}

    def get_bars_diarias(
        self,
        symbol: str,
        desde: str,
        hasta: Optional[str] = None,
    ) -> List[dict]:
        """
        Alias conveniente para run_backtest.py — retorna barras diarias
        OHLCV de un único símbolo como lista de dicts.

        Args:
            symbol: ticker (ej. "GGAL", "MELI")
            desde:  fecha inicio "YYYY-MM-DD"
            hasta:  fecha fin "YYYY-MM-DD" (default: hoy)

        Retorna lista de dicts [{t, o, h, l, c, v}, ...] ordenada
        ascendentemente por fecha.
        """
        result = self.get_bars(
            symbols=[symbol],
            timeframe="1Day",
            limit=1000,       # techo generoso — el rango desde/hasta acota igual
            desde=desde,
            hasta=hasta,
        )
        bars = result.get(symbol, [])
        # Alpaca ya devuelve en orden ascendente, pero aseguramos
        bars.sort(key=lambda b: b["t"])
        return bars

    # ──────────────────────── WEBSOCKET RT ───────────────────────

    def start_stream(self, on_update: Callable[[str, float], None] = None):
        """
        Inicia el stream WebSocket en un thread separado.
        on_update(symbol, price) se llama con cada actualización.
        """
        try:
            import websocket
            import json
        except ImportError:
            logger.warning("websocket-client no instalado. Usando REST polling.")
            return

        def on_message(ws, message):
            import json as _json
            msgs = _json.loads(message)
            for msg in msgs:
                if msg.get("T") == "t":  # trade
                    sym   = msg["S"]
                    price = msg["p"]
                    self._prices[sym] = {**self._prices.get(sym, {}), "last": price, "symbol": sym}
                    if on_update:
                        on_update(sym, price)

        def on_open(ws):
            import json as _json
            ws.send(_json.dumps({
                "action": "auth",
                "key":    self.api_key,
                "secret": self.api_secret,
            }))
            ws.send(_json.dumps({
                "action":  "subscribe",
                "trades":  ADR_SYMBOLS,
            }))
            logger.info("Alpaca WebSocket conectado y suscrito.")

        def run():
            ws = websocket.WebSocketApp(
                ALPACA_WS_URL,
                on_message=on_message,
                on_open=on_open,
                on_error=lambda ws, e: logger.error(f"WS error: {e}"),
                on_close=lambda ws, c, m: logger.info("WS cerrado."),
            )
            ws.run_forever(ping_interval=30)

        self._running = True
        self._ws_thread = threading.Thread(target=run, daemon=True)
        self._ws_thread.start()

    def stop_stream(self):
        self._running = False

    def test_connection(self) -> dict:
        """Prueba conexión Alpaca y retorna sample de GGAL."""
        snaps = self.get_snapshots(["GGAL", "YPF"])
        if snaps:
            return {"ok": True, "msg": "Alpaca conectado OK", "sample": snaps}
        return {"ok": False, "msg": "Error conectando a Alpaca. Verificar API keys."}
