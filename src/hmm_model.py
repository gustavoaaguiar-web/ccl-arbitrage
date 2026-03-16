"""
Modelo HMM (Hidden Markov Model) para determinar el "clima" del mercado.

Estados ocultos:
    0 → BULL  (mercado alcista, alta volatilidad positiva)
    1 → BEAR  (mercado bajista, alta volatilidad negativa)
    2 → LATERAL (mercado sin tendencia, baja volatilidad)

Observaciones (features):
    - Retorno promedio de los cedears (%)
    - Volatilidad de las desviaciones CCL
    - Dispersión entre CCL_i y CCL_promedio

El modelo se entrena con historial y luego predice el estado actual.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

STATE_LABELS = {0: "🟢 BULL", 1: "🔴 BEAR", 2: "🟡 LATERAL"}
STATE_COLORS = {0: "#00C851", 1: "#FF4444", 2: "#FFD700"}


class HMMMarketModel:
    """
    HMM Gaussiano para clasificar el clima de mercado.
    Usa hmmlearn.GaussianHMM internamente.
    Funciona con mínimo ~20 observaciones de historial.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.model    = None
        self.trained  = False
        self._obs_buffer: List[np.ndarray] = []

    def _build_observation(self, ccl_history: list) -> Optional[np.ndarray]:
        """
        Construye vector de observación a partir del historial CCL.
        Retorna None si no hay suficientes datos.
        """
        if len(ccl_history) < 2:
            return None

        obs = []
        for i in range(1, len(ccl_history)):
            prev = ccl_history[i - 1]
            curr = ccl_history[i]

            devs_prev = list(prev["entries"].values())
            devs_curr = list(curr["entries"].values())

            if not devs_prev or not devs_curr:
                continue

            ret      = np.mean(devs_curr) - np.mean(devs_prev)   # retorno medio
            vol      = np.std(devs_curr) if len(devs_curr) > 1 else 0.0
            dispersion = np.max(devs_curr) - np.min(devs_curr)   # spread

            obs.append([ret, vol, dispersion])

        return np.array(obs) if obs else None

    def train(self, ccl_history: list) -> bool:
        """
        Entrena el HMM con el historial de desvíos CCL.
        Requiere al menos 20 snapshots.
        """
        if len(ccl_history) < 20:
            logger.info(f"HMM necesita ≥20 obs. Tiene {len(ccl_history)}.")
            return False

        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.error("hmmlearn no instalado. Pip install hmmlearn.")
            return False

        obs = self._build_observation(ccl_history)
        if obs is None or len(obs) < 10:
            return False

        try:
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=self.n_iter,
                random_state=42,
            )
            self.model.fit(obs)
            self.trained = True
            logger.info(f"HMM entrenado con {len(obs)} observaciones.")
            return True
        except Exception as e:
            logger.error(f"Error entrenando HMM: {e}")
            return False

    def predict_state(self, ccl_history: list) -> Tuple[int, str, float]:
        """
        Predice el estado actual del mercado.
        Retorna (state_id, label, confidence).
        """
        if not self.trained or self.model is None:
            # Fallback heurístico simple
            return self._heuristic_state(ccl_history)

        obs = self._build_observation(ccl_history[-21:])  # últimas 20 transiciones
        if obs is None or len(obs) == 0:
            return self._heuristic_state(ccl_history)

        try:
            states = self.model.predict(obs)
            probs  = self.model.predict_proba(obs)
            last_state      = int(states[-1])
            last_confidence = float(probs[-1][last_state])
            label = STATE_LABELS.get(last_state, "DESCONOCIDO")
            return last_state, label, last_confidence
        except Exception as e:
            logger.error(f"Error prediciendo estado: {e}")
            return self._heuristic_state(ccl_history)

    def _heuristic_state(self, ccl_history: list) -> Tuple[int, str, float]:
        """
        Clasificación heurística simple cuando no hay modelo entrenado.
        Basada en la dispersión y tendencia de los últimos desvíos.
        """
        if not ccl_history:
            return 2, STATE_LABELS[2], 0.5

        last = ccl_history[-1]
        devs = list(last.get("entries", {}).values())

        if not devs:
            return 2, STATE_LABELS[2], 0.5

        mean_dev = np.mean(devs)
        vol_dev  = np.std(devs) if len(devs) > 1 else 0

        if mean_dev > 0.3 and vol_dev > 0.2:
            return 0, STATE_LABELS[0], 0.6    # BULL
        elif mean_dev < -0.3 and vol_dev > 0.2:
            return 1, STATE_LABELS[1], 0.6    # BEAR
        else:
            return 2, STATE_LABELS[2], 0.6    # LATERAL

    def get_trading_recommendation(self, state_id: int) -> str:
        """Recomendación de trading según el clima."""
        recs = {
            0: "Mercado BULL: señales BUY_ARS tienen mayor probabilidad de éxito.",
            1: "Mercado BEAR: señales SELL_ARS tienen mayor probabilidad de éxito.",
            2: "Mercado LATERAL: reducir tamaño de posición. Esperar confirmación.",
        }
        return recs.get(state_id, "Sin recomendación disponible.")
      
