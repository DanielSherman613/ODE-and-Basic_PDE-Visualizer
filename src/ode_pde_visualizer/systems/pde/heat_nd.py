from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ode_pde_visualizer.core.grids import HyperGrid
from ode_pde_visualizer.core.parameters import ParameterSet
from ode_pde_visualizer.systems.pde.base import PDESystem

BoundaryMode = Literal["neumann", "dirichlet", "periodic"]


@dataclass(slots=True)
class HeatNDConfig:
    diffusivityParam: str = "alpha"
    amplitude: float = 1.0
    gaussianSharpness: float = 10.0
    boundaryMode: BoundaryMode = "neumann"


class NDimHeatEquation(PDESystem):
    name = "N Dimensional Heat Equation"
    fieldNames = ["u"]

    def __init__(
        self,
        diffusivityParam: str = "alpha",
        amplitude: float = 1.0,
        gaussianSharpness: float = 10.0,
        boundaryMode: BoundaryMode = "neumann",
    ) -> None:
        self.config = HeatNDConfig(
            diffusivityParam=diffusivityParam,
            amplitude=float(amplitude),
            gaussianSharpness=float(gaussianSharpness),
            boundaryMode=boundaryMode,
        )

    def initialCondition(
        self,
        grid: HyperGrid,
        params: ParameterSet,
    ) -> dict[str, np.ndarray]:
        mesh = np.meshgrid(*[axis.coords for axis in grid.axes], indexing="ij")
        center = [0.5 * (float(axis.coords[0]) + float(axis.coords[-1])) for axis in grid.axes]

        squaredRadius = np.zeros(grid.shape, dtype=np.float32)
        for axisIndex, arr in enumerate(mesh):
            squaredRadius += (arr.astype(np.float32) - center[axisIndex]) ** 2

        u0 = self.config.amplitude * np.exp(-self.config.gaussianSharpness * squaredRadius)
        return {"u": u0.astype(np.float32, copy=False)}

    def stableTimeStep(
        self,
        grid: HyperGrid,
        params: ParameterSet,
    ) -> float:
        alpha = float(params.values[self.config.diffusivityParam])
        if alpha <= 0.0:
            raise ValueError("Heat equation diffusivity must be positive.")

        inverseSpacingSquaredSum = 0.0
        for axis in grid.axes:
            dx = float(axis.spacing)
            if dx <= 0.0:
                raise ValueError("Grid spacing must be positive on every axis.")
            inverseSpacingSquaredSum += 1.0 / (dx * dx)

        if inverseSpacingSquaredSum <= 0.0:
            raise ValueError("Could not compute a stable time step for the heat equation.")

        return 1.0 / (2.0 * alpha * inverseSpacingSquaredSum)

    def laplacian(self, u: np.ndarray, grid: HyperGrid) -> np.ndarray:
        if self.config.boundaryMode == "periodic":
            return self._laplacianPeriodic(u, grid)
        if self.config.boundaryMode == "dirichlet":
            return self._laplacianPadded(u, grid, padMode="constant")
        if self.config.boundaryMode == "neumann":
            return self._laplacianPadded(u, grid, padMode="edge")
        raise ValueError(f"Unsupported boundary mode: {self.config.boundaryMode}")

    def step(
        self,
        state: dict[str, np.ndarray],
        grid: HyperGrid,
        dt: float,
        params: ParameterSet,
    ) -> dict[str, np.ndarray]:
        alpha = float(params.values[self.config.diffusivityParam])
        u = np.asarray(state["u"], dtype=np.float32)
        lap = self.laplacian(u, grid)
        uNext = u + np.float32(dt * alpha) * lap
        return {"u": uNext.astype(np.float32, copy=False)}

    @staticmethod
    def _laplacianPeriodic(u: np.ndarray, grid: HyperGrid) -> np.ndarray:
        result = np.zeros_like(u, dtype=np.float32)
        for axisIndex, axis in enumerate(grid.axes):
            dx = float(axis.spacing)
            result += (
                np.roll(u, -1, axis=axisIndex)
                - 2.0 * u
                + np.roll(u, 1, axis=axisIndex)
            ) / np.float32(dx * dx)
        return result

    @staticmethod
    def _laplacianPadded(
        u: np.ndarray,
        grid: HyperGrid,
        padMode: str,
    ) -> np.ndarray:
        result = np.zeros_like(u, dtype=np.float32)
        padded = np.pad(u, [(1, 1)] * u.ndim, mode=padMode)
        centerSlice = tuple(slice(1, -1) for _ in range(u.ndim))

        for axisIndex, axis in enumerate(grid.axes):
            dx = float(axis.spacing)
            minusSlice = [slice(1, -1)] * u.ndim
            plusSlice = [slice(1, -1)] * u.ndim
            minusSlice[axisIndex] = slice(0, -2)
            plusSlice[axisIndex] = slice(2, None)

            secondDerivative = (
                padded[tuple(plusSlice)]
                - 2.0 * padded[centerSlice]
                + padded[tuple(minusSlice)]
            ) / np.float32(dx * dx)
            result += secondDerivative.astype(np.float32, copy=False)
        return result
