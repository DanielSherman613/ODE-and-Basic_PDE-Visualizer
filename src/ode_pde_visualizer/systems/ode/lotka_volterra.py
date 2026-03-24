from __future__ import annotations

import numpy as np

from ode_pde_visualizer.core.parameters import ParameterSet
from ode_pde_visualizer.systems.ode.base import ODESystem


class LotkaVolterraSystem(ODESystem):
    name = "Lotka Volterra"
    axisNames = ("prey", "predator", "")

    def initialState(self, params: ParameterSet) -> np.ndarray:
        prey0 = float(params.values.get("prey0", 2.0))
        predator0 = float(params.values.get("predator0", 1.0))
        return np.array([prey0, predator0], dtype=np.float64)

    def derivative(
        self,
        timeValue: float,
        state: np.ndarray,
        params: ParameterSet,
    ) -> np.ndarray:
        alpha = float(params.values.get("alpha", 1.5))
        beta = float(params.values.get("beta", 1.0))
        delta = float(params.values.get("delta", 1.0))
        gamma = float(params.values.get("gamma", 3.0))
        prey, predator = map(float, state[:2])
        return np.array(
            [
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator,
            ],
            dtype=np.float64,
        )
