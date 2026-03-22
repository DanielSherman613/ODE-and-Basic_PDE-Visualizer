import numpy as np

from ode_pde_visualizer.src.ode_pde_visualizer.core.grids import HyperGrid
from ode_pde_visualizer.src.ode_pde_visualizer.core.parameters import \
    ParameterSet
from ode_pde_visualizer.src.ode_pde_visualizer.systems.pde.base import PDESystem


class NDimHeatEquation(PDESystem):
    name = "N Dimensional Heat Equation"
    fieldNames = ["u"]

    def __init__(self, diffusivityParam: str = "alpha") -> None:
        self.diffusivityParam = diffusivityParam

    def initialCondition(self, grid: HyperGrid, params: ParameterSet) -> dict[str, np.ndarray]:
        mesh = np.meshgrid(*[axis.coords for axis in grid.axes], indexing="ij")
        center = [0.5 * (axis.coords[0] + axis.coords[-1]) for axis in grid.axes]

        squaredRadius = np.zeros(grid.shape, dtype=float)
        for i, arr in enumerate(mesh):
            squaredRadius += (arr - center[i]) ** 2

        u0 = np.exp(-10.0 * squaredRadius)
        return {"u": u0}

    def laplacian(self, u: np.ndarray, grid: HyperGrid) -> np.ndarray:
        result = np.zeros_like(u)
        for axisIndex, axis in enumerate(grid.axes):
            dx = axis.spacing
            result += (
                np.roll(u, -1, axis=axisIndex)
                - 2.0 * u
                + np.roll(u, 1, axis=axisIndex)
            ) / (dx ** 2)
        return result

    def step(
        self,
        state: dict[str, np.ndarray],
        grid: HyperGrid,
        dt: float,
        params: ParameterSet,
    ) -> dict[str, np.ndarray]:
        alpha = params.values[self.diffusivityParam]
        u = state["u"]
        uNext = u + dt * alpha * self.laplacian(u, grid)
        return {"u": uNext}