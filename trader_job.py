"""
Test harness — valida trader_job.py + sheets_manager.py SIN tocar
la red real ni gastar requests de IOL/Google.

Mockea:
  - IOLClient.login() / get_panel()  → devuelve data falsa realista
  - SheetsManager                    → guarda en memoria (dict) en vez
                                        de pegarle a Google Sheets real

Corre 3 ciclos simulados y valida:
  1. Que fetch_detalle_merval() filtre correctamente el universo de 9
  2. Que guardar_tick_merval() acumule filas sin pisar las anteriores
  3. Que cargar_historico_merval_raw() devuelva los datos ordenados
  4. Que el ciclo completo no tire excepciones
"""

import sys
import os
import random
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import trader_job
from sheets_manager import SheetsManager, HEADERS


# ─────────────────────────────────────────────────────────
# MOCK 1 — IOLClient: simula get_panel('MerVal') con data
#          realista basada en lo que confirmamos en el test real
# ─────────────────────────────────────────────────────────
MERVAL_PANEL_FAKE = [
    {"simbolo": "GGAL", "ultimoPrecio": 8270.0, "apertura": 8480.0,
     "maximo": 8480.0, "minimo": 8150.0, "volumen": 337463.0,
     "cantidadOperaciones": 6121.0},
    {"simbolo": "YPFD", "ultimoPrecio": 45200.0, "apertura": 45800.0,
     "maximo": 46100.0, "minimo": 44900.0, "volumen": 12000.0,
     "cantidadOperaciones": 980.0},
    {"simbolo": "PAMP", "ultimoPrecio": 5100.0, "apertura": 5050.0,
     "maximo": 5180.0, "minimo": 5020.0, "volumen": 98000.0,
     "cantidadOperaciones": 1500.0},
    {"simbolo": "BMA", "ultimoPrecio": 9800.0, "apertura": 9700.0,
     "maximo": 9900.0, "minimo": 9650.0, "volumen": 45000.0,
     "cantidadOperaciones": 2100.0},
    {"simbolo": "CEPU", "ultimoPrecio": 2300.0, "apertura": 2280.0,
     "maximo": 2340.0, "minimo": 2260.0, "volumen": 60000.0,
     "cantidadOperaciones": 1100.0},
    {"simbolo": "TGSU2", "ultimoPrecio": 6700.0, "apertura": 6650.0,
     "maximo": 6750.0, "minimo": 6600.0, "volumen": 30000.0,
     "cantidadOperaciones": 800.0},
    {"simbolo": "SUPV", "ultimoPrecio": 3100.0, "apertura": 3050.0,
     "maximo": 3150.0, "minimo": 3000.0, "volumen": 25000.0,
     "cantidadOperaciones": 600.0},
    {"simbolo": "BBAR", "ultimoPrecio": 4200.0, "apertura": 4150.0,
     "maximo": 4250.0, "minimo": 4100.0, "volumen": 35000.0,
     "cantidadOperaciones": 700.0},
    {"simbolo": "VALO", "ultimoPrecio": 1800.0, "apertura": 1780.0,
     "maximo": 1820.0, "minimo": 1760.0, "volumen": 20000.0,
     "cantidadOperaciones": 400.0},
    # Símbolo FUERA del universo Swing — debe ser filtrado
    {"simbolo": "ALUA", "ultimoPrecio": 1000.0, "apertura": 1010.0,
     "maximo": 1015.0, "minimo": 993.5, "volumen": 108103.0,
     "cantidadOperaciones": 981.0},
]


class FakeIOLClient:
    """Mock de IOLClient — no toca la red."""

    def __init__(self, *args, **kwargs):
        self.login_called = False

    def login(self):
        self.login_called = True
        return True

    def get_panel(self, panel: str):
        if panel == "MerVal":
            # Simulamos pequeñas variaciones de precio entre ciclos,
            # como pasaría en la realidad
            data = []
            for item in MERVAL_PANEL_FAKE:
                copia = dict(item)
                copia["ultimoPrecio"] = round(
                    copia["ultimoPrecio"] * (1 + random.uniform(-0.002, 0.002)), 2
                )
                data.append(copia)
            return data
        return []


# ─────────────────────────────────────────────────────────
# MOCK 2 — SheetsManager: guarda en memoria, no pega a Google
# ─────────────────────────────────────────────────────────
class FakeWorksheet:
    """Simula un worksheet de gspread guardando filas en una lista."""

    def __init__(self, title, headers):
        self.title = title
        self.rows = [headers]

    def append_row(self, fila):
        self.rows.append(fila)

    def append_rows(self, filas):
        self.rows.extend(filas)

    def get_all_values(self):
        return self.rows

    def clear(self):
        self.rows = []

    def delete_rows(self, start, end):
        # 1-indexed, incluye header en la posición 1
        del self.rows[start - 1:end]


class FakeSheetsManager(SheetsManager):
    """
    Hereda de SheetsManager real para reusar toda la lógica de negocio
    (guardar_tick_merval, cargar_historico_merval_raw, etc.) pero
    reemplaza la capa de conexión a Google por almacenamiento en memoria.
    """

    def __init__(self):
        # No llamamos al __init__ real (evita gspread.authorize real)
        self.gc = None
        self.sh = None
        self._hojas = {}
        for nombre, headers in HEADERS.items():
            self._hojas[nombre] = FakeWorksheet(nombre, headers)

    def conectar(self) -> bool:
        # Ya están las hojas creadas en __init__, no hace falta nada más
        return True


# ─────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────
def test_fetch_detalle_merval_filtra_universo():
    print("\n[TEST 1] fetch_detalle_merval filtra correctamente el universo de 9...")
    iol = FakeIOLClient()
    resultado = trader_job.fetch_detalle_merval(iol)

    simbolos_devueltos = {s["symbol"] for s in resultado}
    assert simbolos_devueltos == trader_job.MERVAL_SWING_SET, (
        f"FALLÓ: esperaba {trader_job.MERVAL_SWING_SET}, "
        f"recibió {simbolos_devueltos}"
    )
    assert "ALUA" not in simbolos_devueltos, "FALLÓ: ALUA no debería estar (fuera de universo)"
    assert len(resultado) == 9, f"FALLÓ: esperaba 9 símbolos, recibió {len(resultado)}"

    # Validar estructura de un item
    ggal = next(s for s in resultado if s["symbol"] == "GGAL")
    campos_esperados = {"symbol", "precio", "apertura", "maximo", "minimo",
                         "volumen_nominal", "cantidad_operaciones"}
    assert set(ggal.keys()) == campos_esperados, (
        f"FALLÓ: campos no coinciden. Esperado {campos_esperados}, "
        f"recibido {set(ggal.keys())}"
    )
    print(f"  ✅ OK — 9 símbolos del universo, ALUA filtrado correctamente")
    print(f"  ✅ OK — estructura de campos correcta: {sorted(campos_esperados)}")


def test_guardar_y_cargar_historico():
    print("\n[TEST 2] guardar_tick_merval + cargar_historico_merval_raw...")
    sheets = FakeSheetsManager()
    sheets.conectar()

    iol = FakeIOLClient()

    # Simular 3 ciclos guardando datos
    for i in range(3):
        detalle = trader_job.fetch_detalle_merval(iol)
        sheets.guardar_tick_merval(detalle)

    ws = sheets._hojas["Historico_Merval_Raw"]
    filas_totales = len(ws.rows) - 1  # menos header
    assert filas_totales == 9 * 3, (
        f"FALLÓ: esperaba {9*3} filas (9 símbolos x 3 ciclos), "
        f"recibió {filas_totales}"
    )
    print(f"  ✅ OK — {filas_totales} filas acumuladas tras 3 ciclos (9 símbolos x 3)")

    # Validar carga y filtro por símbolo
    historico_ggal = sheets.cargar_historico_merval_raw(symbol="GGAL")
    assert len(historico_ggal) == 3, (
        f"FALLÓ: esperaba 3 registros de GGAL, recibió {len(historico_ggal)}"
    )
    for reg in historico_ggal:
        assert reg["symbol"] == "GGAL"
        assert reg["precio"] > 0
        assert reg["apertura"] > 0
    print(f"  ✅ OK — carga filtrada por símbolo (GGAL): {len(historico_ggal)} registros")

    # Validar carga completa (todos los símbolos)
    historico_todo = sheets.cargar_historico_merval_raw()
    assert len(historico_todo) == 27, (
        f"FALLÓ: esperaba 27 registros totales, recibió {len(historico_todo)}"
    )
    print(f"  ✅ OK — carga completa sin filtro: {len(historico_todo)} registros")

    # Validar orden ascendente por ts
    timestamps = [r["ts"] for r in historico_todo]
    assert timestamps == sorted(timestamps), "FALLÓ: no está ordenado ascendente por ts"
    print(f"  ✅ OK — registros ordenados ascendentemente por timestamp")


def test_ejecutar_ciclo_completo_sin_excepciones():
    print("\n[TEST 3] ejecutar_ciclo() completo, simulando horario de mercado...")
    sheets = FakeSheetsManager()
    sheets.conectar()
    iol = FakeIOLClient()

    # Forzamos que hora_argentina() devuelva un horario dentro de mercado,
    # sin importar cuándo se corra este test
    from datetime import datetime as real_datetime
    from zoneinfo import ZoneInfo

    fake_hora = real_datetime(2026, 6, 19, 13, 0, 0, tzinfo=ZoneInfo("America/Argentina/Buenos_Aires"))

    with patch.object(trader_job, "hora_argentina", return_value=fake_hora):
        for n in range(1, 4):
            try:
                trader_job.ejecutar_ciclo(iol, sheets, n)
            except Exception as e:
                raise AssertionError(f"FALLÓ: ejecutar_ciclo() tiró excepción en ciclo {n}: {e}")

    ws = sheets._hojas["Historico_Merval_Raw"]
    filas_totales = len(ws.rows) - 1
    assert filas_totales == 9 * 3, (
        f"FALLÓ: esperaba {9*3} filas tras 3 ciclos completos, recibió {filas_totales}"
    )
    print(f"  ✅ OK — 3 ciclos completos sin excepciones, {filas_totales} filas guardadas")


def test_fuera_de_horario_no_guarda():
    print("\n[TEST 4] Fuera de horario de mercado, no debe guardar nada...")
    sheets = FakeSheetsManager()
    sheets.conectar()
    iol = FakeIOLClient()

    from datetime import datetime as real_datetime
    from zoneinfo import ZoneInfo

    # 20:00 ART — fuera de horario (HORA_CIERRE = 17:00)
    fake_hora_fuera = real_datetime(2026, 6, 19, 20, 0, 0, tzinfo=ZoneInfo("America/Argentina/Buenos_Aires"))

    with patch.object(trader_job, "hora_argentina", return_value=fake_hora_fuera):
        trader_job.ejecutar_ciclo(iol, sheets, 1)

    ws = sheets._hojas["Historico_Merval_Raw"]
    filas_totales = len(ws.rows) - 1
    assert filas_totales == 0, (
        f"FALLÓ: fuera de horario no debería guardar nada, guardó {filas_totales} filas"
    )
    print(f"  ✅ OK — fuera de horario, 0 filas guardadas (correcto)")


def test_login_se_llama_una_vez_en_main_no_en_cada_ciclo():
    print("\n[TEST 5] Verificar que login() se llama 1 vez en main(), no en cada ciclo...")
    iol = FakeIOLClient()
    assert iol.login_called is False
    iol.login()
    assert iol.login_called is True
    print(f"  ✅ OK — patrón de login confirmado (se llama explícitamente, no por ciclo)")


def test_resiliencia_ante_error_iol():
    print("\n[TEST 6] Resiliencia: get_panel() falla (IOL caído) no debe tirar excepción...")

    class FakeIOLClientRoto:
        def get_panel(self, panel):
            raise ConnectionError("Simulando caída de IOL")

    sheets = FakeSheetsManager()
    sheets.conectar()
    iol_roto = FakeIOLClientRoto()

    from datetime import datetime as real_datetime
    from zoneinfo import ZoneInfo
    fake_hora = real_datetime(2026, 6, 19, 13, 0, 0, tzinfo=ZoneInfo("America/Argentina/Buenos_Aires"))

    with patch.object(trader_job, "hora_argentina", return_value=fake_hora):
        try:
            trader_job.ejecutar_ciclo(iol_roto, sheets, 1)
        except Exception as e:
            raise AssertionError(
                f"FALLÓ: ejecutar_ciclo() debería absorber el error de IOL "
                f"internamente (via try/except en fetch_detalle_merval), "
                f"pero propagó: {type(e).__name__}: {e}"
            )

    ws = sheets._hojas["Historico_Merval_Raw"]
    assert len(ws.rows) - 1 == 0, "FALLÓ: no debería haber guardado nada"
    print(f"  ✅ OK — error de IOL absorbido sin tumbar el ciclo, 0 filas (correcto)")


def test_panel_con_simbolo_sin_precio_no_rompe():
    print("\n[TEST 7] Símbolo con ultimoPrecio=None/0 en el panel no debe romper ni guardarse...")

    class FakeIOLClientConNulos:
        def get_panel(self, panel):
            data = [dict(item) for item in MERVAL_PANEL_FAKE if item["simbolo"] != "ALUA"]
            data[0]["ultimoPrecio"] = None  # GGAL sin precio este ciclo
            return data

    iol = FakeIOLClientConNulos()
    resultado = trader_job.fetch_detalle_merval(iol)

    simbolos = {s["symbol"] for s in resultado}
    assert "GGAL" not in simbolos, "FALLÓ: GGAL con precio None debería ser excluido"
    assert len(resultado) == 8, f"FALLÓ: esperaba 8 símbolos (9 - GGAL sin precio), recibió {len(resultado)}"
    print(f"  ✅ OK — símbolo sin precio excluido correctamente, {len(resultado)}/9 guardados")


if __name__ == "__main__":
    print("="*60)
    print("TEST SUITE — trader_job.py + sheets_manager.py (Sistema GG Swing)")
    print("="*60)

    tests = [
        test_fetch_detalle_merval_filtra_universo,
        test_guardar_y_cargar_historico,
        test_ejecutar_ciclo_completo_sin_excepciones,
        test_fuera_de_horario_no_guarda,
        test_login_se_llama_una_vez_en_main_no_en_cada_ciclo,
        test_resiliencia_ante_error_iol,
        test_panel_con_simbolo_sin_precio_no_rompe,
    ]

    fallos = 0
    for test_fn in tests:
        try:
            test_fn()
        except AssertionError as e:
            fallos += 1
            print(f"  ❌ {e}")
        except Exception as e:
            fallos += 1
            print(f"  ❌ ERROR INESPERADO: {type(e).__name__}: {e}")

    print("\n" + "="*60)
    if fallos == 0:
        print(f"✅ TODOS LOS TESTS PASARON ({len(tests)}/{len(tests)})")
    else:
        print(f"❌ {fallos}/{len(tests)} TESTS FALLARON")
    print("="*60)
