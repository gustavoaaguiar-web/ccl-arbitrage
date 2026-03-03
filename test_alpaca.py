"""
Test de conexión Alpaca Markets
Ejecutar: python test_alpaca.py
"""

import requests

# ─── COMPLETAR CON TUS DATOS ───────────────────────────
ALPACA_KEY    = "PKGZCMLLJFIYE22CO2OUHMRRZK"   # tu Key ID
ALPACA_SECRET = "8aiTk8VpcxRwbzM9rmsNN8XJCvQh8VWPJJU2rPfo24er"                 # pegá tu Secret
# ───────────────────────────────────────────────────────

# ADRs argentinos en NYSE
SYMBOLS = ["GGAL", "YPF", "PAM", "BMA", "TEO", "CEPU", "LOMA", "SUPV"]

def test_alpaca():
    headers = {
        "APCA-API-KEY-ID":     ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

    print("🔌 Probando conexión a Alpaca...\n")

    # 1. Verificar cuenta
    resp = requests.get(
        "https://paper-api.alpaca.markets/v2/account",
        headers=headers,
        timeout=10,
    )
    if resp.status_code == 200:
        acc = resp.json()
        print(f"✅ Cuenta OK: {acc.get('account_number')}")
        print(f"   Status: {acc.get('status')}")
        print(f"   Buying Power: ${float(acc.get('buying_power', 0)):,.0f}\n")
    else:
        print(f"❌ Error de cuenta: {resp.status_code} - {resp.text}")
        return

    # 2. Traer precios de ADRs argentinos
    print("📡 Obteniendo precios de ADRs argentinos...\n")

    data_headers = {**headers}  # mismo header, distinto endpoint
    resp2 = requests.get(
        "https://data.alpaca.markets/v2/stocks/snapshots",
        headers=data_headers,
        params={"symbols": ",".join(SYMBOLS), "feed": "iex"},
        timeout=10,
    )

    if resp2.status_code == 200:
        snaps = resp2.json()
        print(f"{'Símbolo':<8} {'Último precio':>14} {'Bid':>10} {'Ask':>10}")
        print("-" * 45)
        for sym in SYMBOLS:
            data = snaps.get(sym, {})
            trade = data.get("latestTrade", {})
            quote = data.get("latestQuote", {})
            last  = trade.get("p", "N/D")
            bid   = quote.get("bp", "N/D")
            ask   = quote.get("ap", "N/D")
            last_fmt = f"${last:.2f}" if isinstance(last, float) else last
            bid_fmt  = f"${bid:.2f}"  if isinstance(bid,  float) else bid
            ask_fmt  = f"${ask:.2f}"  if isinstance(ask,  float) else ask
            print(f"{sym:<8} {last_fmt:>14} {bid_fmt:>10} {ask_fmt:>10}")
        print("\n✅ Alpaca funcionando correctamente!")
    else:
        print(f"❌ Error datos: {resp2.status_code} - {resp2.text}")

if __name__ == "__main__":
    test_alpaca()
