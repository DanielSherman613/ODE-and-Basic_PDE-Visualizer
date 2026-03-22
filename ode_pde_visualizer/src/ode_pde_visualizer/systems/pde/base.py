from abc import ABC, abstractmethod
from ode_pde_visualizer import src
from ode_pde_visualizer.src.ode_pde_visualizer import core
from ode_pde_visualizer.src.ode_pde_visualizer.core import grids
from ode_pde_visualizer.src.ode_pde_visualizer.core.grids import HyperGrid
from ode_pde_visualizer.src.ode_pde_visualizer.core.parameters import \
    ParameterSet


class PDESystem(ABC):
    name: str
    fieldNames: list[str]

    @abstractmethod
    def initialCondition(self, grid: HyperGrid, params: ParameterSet) -> dict[
        str, object]:
        raise NotImplementedError

    @abstractmethod
    def step(
            self,
            state: dict[str, object],
            grid: HyperGrid,
            dt: float,
            params: ParameterSet,
    ) -> dict[str, object]:
        raise NotImplementedError
