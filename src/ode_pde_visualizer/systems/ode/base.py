from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ode_pde_visualizer.core.parameters import ParameterSet


class ODESystem(ABC):
    name: str
    axisNames: tuple[str, str, str]

    @abstractmethod
    def initialState(self, params: ParameterSet) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def derivative(
        self,
        timeValue: float,
        state: np.ndarray,
        params: ParameterSet,
    ) -> np.ndarray:
        raise NotImplementedError
