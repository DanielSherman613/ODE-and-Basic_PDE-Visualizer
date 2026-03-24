from __future__ import annotations

import numpy as np

from ode_pde_visualizer.core.parameters import ParameterSet
from ode_pde_visualizer.systems.ode.base import ODESystem


class LorenzSystem(ODESystem):
    name = "Lorenz"
    axisNames = ("x", "y", "z")

    def initialState(self, params: ParameterSet) -> np.ndarray:
        x0 = float(params.values.get("x0", 1.0))
        y0 = float(params.values.get("y0", 1.0))
        z0 = float(params.values.get("z0", 1.0))
        return np.array([x0, y0, z0], dtype=np.float64)

    def derivative(
        self,
        timeValue: float,
        state: np.ndarray,
        params: ParameterSet,
    ) -> np.ndarray:
        sigma = float(params.values.get("sigma", 10.0))
        rho = float(params.values.get("rho", 28.0))
        beta = float(params.values.get("beta", 8.0 / 3.0))
        x, y, z = map(float, state[:3])
        return np.array(
            [
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z,
            ],
            dtype=np.float64,
        )
