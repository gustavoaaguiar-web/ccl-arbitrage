"""
diagnostico_granularidad_v2.py — Localiza dónde aparecen los duplicados
==========================================================================
El diagnóstico anterior (2 semanas de enero 2020) salió limpio: 1 registro
por día hábil, sin duplicados. Pero el rango completo 2020-01-01 a
2024-08-01 devolvió 7.507 registros (~6,5x lo esperado). Este script trae
el rango completo una sola vez y resume, sin imprimir cada registro:
  - cuántas fechas distintas hay vs. cuántos registros totales
  - las 10 fechas con más registros duplicados
  - en qué punto del rango (por año) se concentran los duplicados

Uso:
    python diagnostico_granularidad_v2.py
"""

import os
import sys
import logging
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from iol_client import IOLClient, IOL_BASE_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    iol = IOLClient(os.environ["IOL_USER"], os.environ["IOL_PASS"])
    iol.login()
    iol._ensure_token()

    desde, hasta = "2020-01-01", "2024-08-01"

    logger.info(f"Trayendo GGAL {desde}..{hasta} (puede tardar, mismo rango que dio 7507 registros)...")
    resp = iol.session.get(
        f"{IOL_BASE_URL}/api/v2/bCBA/Titulos/GGAL/Cotizacion/seriehistorica/{desde}/{hasta}/sinAjustar",
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    print(f"\nTotal registros crudos: {len(data)}")

    conteo_por_fecha = defaultdict(int)
    conteo_por_anio = defaultdict(int)
    for d in data:
        fecha_completa = d.get("fechaHora", "SIN_FECHA")
        fecha_dia = fecha_completa[:10]
        anio = fecha_dia[:4]
        conteo_por_fecha[fecha_dia] += 1
        conteo_por_anio[anio] += 1

    print(f"Fechas distintas: {len(conteo_por_fecha)}")
    print(f"Promedio registros/fecha: {len(data) / max(1, len(conteo_por_fecha)):.2f}\n")

    print("Registros totales por año:")
    for anio, n in sorted(conteo_por_anio.items()):
        print(f"  {anio}: {n} registros")

    duplicados = {f: n for f, n in conteo_por_fecha.items() if n > 1}
    print(f"\nFechas con más de 1 registro: {len(duplicados)} de {len(conteo_por_fecha)}")

    top10 = sorted(duplicados.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 fechas con más duplicados:")
    for fecha, n in top10:
        print(f"  {fecha}: {n} registros")
        # muestra los registros crudos de esa fecha puntual para ver qué los distingue
        registros_fecha = [d for d in data if d.get("fechaHora", "")[:10] == fecha]
        for r in registros_fecha[:5]:
            print(f"      -> fechaHora={r.get('fechaHora')} ultimoPrecio={r.get('ultimoPrecio')} "
                  f"plazo={r.get('plazo')} moneda={r.get('moneda')}")


if __name__ == "__main__":
    main()
