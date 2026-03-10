"""
Cliente Alpaca Markets para precios en tiempo real de ADRs argentinos en NYSE.
Usa WebSocket v2 de Alpaca Data API (plan gratuito incluido).
Docs: https://docs.alpaca.markets/reference/stocklatesttrade
"""

import logging
import threading
from typing import Callable, Dict, Optional
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

ALPACA_BASE_URL = "https://data.alpaca.markets/v2"
ALPACA_WS_URL   = "wss://stream.data.alpaca.markets/v2/iex"  # IEX = gratuito

# Símbolos ADR necesarios (del mapping en iol_client.py)
ADR_SYMBOLS = ["GGAL", "YPF", "PAM", "BMA", "TEO", "BBAR", "CEPU", "LOMA", "SUPV", "TX"]


class AlpacaClient:
    """
    Cliente REST + WebSocket para Alpaca Markets.
    - REST: precios actuales (snapshot)
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

    # ─────────────────────── REST SNAPSHOT ───────────────────────

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
                    "last":   latest_trade.get("p"),      # precio último trade
                    "bid":    latest_quote.get("bp"),      # bid price
                    "ask":    latest_quote.get("ap"),      # ask price
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
            msgs = json.loads(message)
            for msg in msgs:
                if msg.get("T") == "t":  # trade
                    sym   = msg["S"]
                    price = msg["p"]
                    self._prices[sym] = {**self._prices.get(sym, {}), "last": price, "symbol": sym}
                    if on_update:
                        on_update(sym, price)

        def on_open(ws):
            ws.send(json.dumps({
                "action": "auth",
                "key":    self.api_key,
                "secret": self.api_secret,
            }))
            ws.send(json.dumps({
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

    def get_bars(self, symbols: list, timeframe: str = "4Hour", limit: int = 60) -> Dict[str, list]:
        """
        Descarga barras OHLCV para una lista de simbolos.
        Retorna dict {sym: [close, close, ...]} con los ultimos `limit` cierres.
        Usado por el HMM para entrenarse sobre log-returns de barras 4H.
        """
        from datetime import timezone, timedelta
        start = (datetime.now(timezone.utc) - timedelta(days=35)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            resp = requests.get(
                f"{ALPACA_BASE_URL}/stocks/bars",
                headers=self.headers,
                params={
                    "symbols":   ",".join(symbols),
                    "timeframe": timeframe,
                    "start":     start,
                    "limit":     limit,
                    "feed":      "iex",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("bars", {})
            return {sym: [b["c"] for b in bars] for sym, bars in data.items() if len(bars) >= 5}
        except requests.RequestException as e:
            logger.error(f"Error Alpaca bars ({timeframe}): {e}")
            return {}

    def test_connection(self) -> dict:
        """Prueba conexión Alpaca y retorna sample de GGAL."""
        snaps = self.get_snapshots(["GGAL", "YPF"])
        if snaps:
            return {"ok": True, "msg": "Alpaca conectado OK", "sample": snaps}
        return {"ok": False, "msg": "Error conectando a Alpaca. Verificar API keys."}
                    
