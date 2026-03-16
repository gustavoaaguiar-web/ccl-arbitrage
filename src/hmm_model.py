"""
Modelo HMM (Hidden Markov Model) - Arquitectura Simons

Determina el "clima" del mercado (BULL / BEAR) basándose EXCLUSIVAMENTE 
en los log-returns del activo subyacente en USD. 

Identificación de Estados:
A diferencia de usar solo la media (que puede fallar por alta volatilidad en crashes),
este modelo calcula el Ratio de Sharpe de cada estado oculto (Media / Desvío Estándar).
El estado con el mayor Sharpe se clasifica dinámicamente como el régimen BULL (🟢).
Esto previene el "State Flipping" cuando se re-entrena el modelo.
"""

import logging
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

class SimonsHMM:
    """
    HMM Gaussiano para clasificar el clima de mercado basado en subyacentes USD.
    """
    def __init__(self, n_states: int = 2, n_iter: int = 100, min_obs: int = 50):
        self.n_states = n_states
        self.n_iter = n_iter
        self.min_obs = min_obs

    def predict_climate(self, prices: List[float]) -> str:
        """
        Recibe una serie de precios históricos (ej. barras 1D) del activo en USD.
        Devuelve '🟢' si el régimen actual es BULL, o '🔴' si es BEAR/LATERAL.
        """
        if not prices or len(prices) < self.min_obs:
            logger.warning(f"HMM: Insuficientes datos ({len(prices) if prices else 0} < {self.min_obs}). Default a 🔴.")
            return "🔴"

        try:
            from hmmlearn.hmm import GaussianHMM
            
            # Calcular log-returns (ortogonal a los niveles del CCL)
            ret = np.diff(np.log(prices)).reshape(-1, 1)
            
            # Entrenar el modelo
            m = GaussianHMM(
                n_components=self.n_states, 
                random_state=42, # Semilla fija para consistencia inicial
                n_iter=self.n_iter,
                covariance_type="diag"
            )
            m.fit(ret)
            
            # Predecir el estado de la observación más reciente
            estado_actual = m.predict(ret)[-1]
            
            # Identificar cuál estado es realmente el BULL usando el Ratio de Sharpe
            means = m.means_.flatten()
            variances = m.covars_.flatten()
            
            # Raíz cuadrada de la varianza = Desvío estándar (volatilidad)
            # np.maximum evita divisiones por cero en casos atípicos
            std_devs = np.sqrt(np.maximum(variances, 1e-8))
            
            # Sharpe = Retorno medio / Volatilidad
            sharpes = means / std_devs
            
            # El estado con mejor relación riesgo/beneficio es nuestro régimen BULL
            bull_state = np.argmax(sharpes)
            
            return "🟢" if estado_actual == bull_state else "🔴"
            
        except ImportError:
            logger.error("hmmlearn no está instalado. Ejecutá: pip install hmmlearn")
            return "🔴"
        except Exception as e:
            logger.error(f"Error entrenando/prediciendo SimonsHMM: {e}")
            return "🔴"
