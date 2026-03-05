"""
Cliente para la API REST de InvertirOnLine (IOL).
Maneja autenticación OAuth2, refresh de token y cotizaciones en tiempo real.
"""

import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

IOL_BASE_URL = "https://api.invertironline.com"

# Pares Cedear (símbolo IOL) → ADR (símbolo Alpaca) + ratio de conversión
# ratio = cuántos cedears equivalen a 1 ADR
CEDEAR_ADR_PAIRS = {
    "GGAL":  {"adr": "GGAL",  "ratio": 10},
    "YPF":   {"adr": "YPF",   "ratio": 1},
    "PAMP":  {"adr": "PAM",   "ratio": 25},
    "BBAR":  {"adr": "BMA",   "ratio": 10},
    "TECO2": {"adr": "TEO",   "ratio": 5},
    "BFR":   {"adr": "BBAR",  "ratio": 3},
    "CEPU":  {"adr": "CEPU",  "ratio": 20},
    "LOMA":  {"adr": "LOMA",  "ratio": 5},
    "SUPV":  {"adr": "SUPV",  "ratio": 5},
    "TXAR":  {"adr": "TX",    "ratio": 1},
}


class IOLClient:
    """Cliente para InvertirOnLine API v2."""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self.session = requests.Session()

    # ─────────────────────────── AUTH ────────────────────────────

    def login(self) -> bool:
        """Obtiene token inicial con usuario y contraseña."""
        try:
            resp = self.session.post(
                f"{IOL_BASE_URL}/token",
                data={
                    "username": self.username,
                    "password": self.password,
                    "grant_type": "password",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return self._parse_token(resp.json())
        except requests.RequestException as e:
            logger.error(f"Error de login IOL: {e}")
            return False

    def _refresh_access_token(self) -> bool:
        """Refresca el token usando el refresh_token."""
        if not self.refresh_token:
            return self.login()
        try:
            resp = self.session.post(
                f"{IOL_BASE_URL}/token",
                data={
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return self._parse_token(resp.json())
        except requests.RequestException:
            logger.warning("Refresh falló, reintentando login completo...")
            return self.login()

    def _parse_token(self, data: dict) -> bool:
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")
        expires_in = int(data.get("expires_in", 1800))
        self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
        if self.access_token:
            self.session.headers.update(
                {"Authorization": f"Bearer {self.access_token}"}
            )
            logger.info("Token IOL obtenido correctamente.")
            return True
        return False

    def _ensure_token(self):
        """Verifica que el token esté vigente; lo refresca si no."""
        if not self.access_token or datetime.now() >= self.token_expiry:
            self._refresh_access_token()

    # ──────────────────────── COTIZACIONES ───────────────────────

    def get_quote(self, symbol: str, market: str = "bCBA") -> Optional[dict]:
        """
        Retorna la cotización de un instrumento.
        market: 'bCBA' para BYMA, 'nYSE' para NYSE (algunos brokers permiten).
        """
        self._ensure_token()
        try:
            resp = self.session.get(
                f"{IOL_BASE_URL}/api/v2/{market}/Titulos/{symbol}/cotizacion",
                params={"plazo": "t0"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "symbol": symbol,
                "last":   data.get("ultimoPrecio"),
                "bid":    data.get("puntas", [{}])[0].get("precioCompra") if data.get("puntas") else None,
                "ask":    data.get("puntas", [{}])[0].get("precioVenta") if data.get("puntas") else None,
                "volume": data.get("volumen"),
                "ts":     datetime.now().isoformat(),
            }
        except requests.RequestException as e:
            logger.error(f"Error cotización {symbol}: {e}")
            return None

    def get_all_cedear_quotes(self) -> dict:
        """Trae cotizaciones de todos los cedears configurados."""
        quotes = {}
        for symbol in CEDEAR_ADR_PAIRS:
            q = self.get_quote(symbol)
            if q and q["last"]:
                quotes[symbol] = q
            time.sleep(0.1)  # Rate limiting suave
        return quotes

    def test_connection(self) -> dict:
        """Prueba la conexión y devuelve un resumen."""
        ok = self.login()
        if not ok:
            return {"ok": False, "msg": "Credenciales inválidas o sin conexión."}
        quote = self.get_quote("GGAL")
        return {
            "ok": True,
            "msg": "Conexión exitosa a IOL",
            "sample": quote,
      }
      
