from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ode_pde_visualizer.core.grids import HyperGrid
from ode_pde_visualizer.core.parameters import ParameterSet


class PDESystem(ABC):
    """Abstract base class for time dependent PDE systems.

    A PDE system owns the spatial operators and the state update rule.
    Time integration is handled by the solver layer.
    """

    name: str
    fieldNames: list[str]

    @abstractmethod
    def initialCondition(
        self,
        grid: HyperGrid,
        params: ParameterSet,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        state: dict[str, Any],
        grid: HyperGrid,
        dt: float,
        params: ParameterSet,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def stableTimeStep(
        self,
        grid: HyperGrid,
        params: ParameterSet,
    ) -> float | None:
        """Return a recommended stable explicit time step if known.

        Systems that do not provide a closed form step size bound may return
        ``None`` and let callers pick their own time step.
        """
        return None
