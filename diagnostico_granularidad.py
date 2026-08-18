"""
diagnostico_granularidad.py — Chequeo puntual de get_historico_diario()
==========================================================================
El backtest de GGAL 2020-2024 devolvió 7.507 velas para un rango que
debería dar ~1.100-1.150 (días hábiles). Este script trae una ventana
chica y conocida (10 días corridos de enero 2020) e imprime CADA registro
crudo, para ver si:
  a) hay más de 1 registro por fecha (duplicados por plazo t0/t1/t2, u
     otra causa), o
  b) el campo de fecha trae hora distinta de 00:00:00 (indicaría
     granularidad intradía en vez de diaria).

No modifica nada, no gasta cuota más allá de 1 request a IOL.

Uso:
    python diagnostico_granularidad.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from iol_client import IOLClient, IOL_BASE_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    iol = IOLClient(os.environ["IOL_USER"], os.environ["IOL_PASS"])
    iol.login()
    iol._ensure_token()

    desde, hasta = "2020-01-01", "2020-01-15"  # ventana chica, ~10 días hábiles esperados

    resp = iol.session.get(
        f"{IOL_BASE_URL}/api/v2/bCBA/Titulos/GGAL/Cotizacion/seriehistorica/{desde}/{hasta}/sinAjustar",
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    print(f"\nTotal registros crudos para GGAL {desde}..{hasta}: {len(data)}")
    print("(esperado si es 1 registro/día hábil: ~10)\n")

    fechas_vistas = {}
    for i, d in enumerate(data):
        fecha_completa = d.get("fechaHora", "SIN_FECHA")
        fecha_dia = fecha_completa[:10]
        fechas_vistas[fecha_dia] = fechas_vistas.get(fecha_dia, 0) + 1
        if i < 20:  # primeros 20 registros crudos, para ver el shape real
            print(f"  [{i}] {d}")

    print("\nConteo de registros por día:")
    for fecha, n in sorted(fechas_vistas.items()):
        marca = "  ⚠️ DUPLICADO" if n > 1 else ""
        print(f"  {fecha}: {n} registro(s){marca}")


if __name__ == "__main__":
    main()
