from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

from ode_pde_visualizer.core.grids import HyperGrid
from ode_pde_visualizer.core.parameters import ParameterSet
from ode_pde_visualizer.core.time_series import PDETimeSeries
from ode_pde_visualizer.solvers.explicit_time_stepper import forwardEulerStep
from ode_pde_visualizer.systems.pde.base import PDESystem


@dataclass(slots=True)
class PDERunConfig:
    totalTime: float
    dt: float | None = None
    maxStoredFrames: int = 80
    storeEveryNSteps: int | None = None
    stabilitySafetyFactor: float = 0.9


@dataclass(slots=True)
class PDERunResult:
    finalState: dict[str, Any]
    timeSeries: PDETimeSeries
    dt: float
    totalSteps: int


class ExplicitPDERunner:
    """Generic explicit PDE runner for systems with a system.step rule."""

    def run(
        self,
        system: PDESystem,
        grid: HyperGrid,
        params: ParameterSet,
        config: PDERunConfig,
    ) -> PDERunResult:
        if config.totalTime < 0.0:
            raise ValueError("totalTime must be non negative.")

        state = system.initialCondition(grid, params)
        times = [0.0]
        fieldsByName = {fieldName: [np.array(state[fieldName], copy=True)] for fieldName in system.fieldNames}

        if config.totalTime == 0.0:
            return PDERunResult(
                finalState=state,
                timeSeries=PDETimeSeries(times=np.array(times, dtype=np.float32), fieldsByName=fieldsByName),
                dt=0.0,
                totalSteps=0,
            )

        dt = self._resolveDt(system, grid, params, config)
        totalSteps = max(1, int(ceil(config.totalTime / dt)))
        dt = float(config.totalTime / totalSteps)

        storeStride = self._resolveStoreStride(totalSteps, config.maxStoredFrames, config.storeEveryNSteps)
        currentTime = 0.0

        for stepIndex in range(1, totalSteps + 1):
            state = forwardEulerStep(system, state, grid, dt, params)
            currentTime = stepIndex * dt

            shouldStore = (stepIndex % storeStride == 0) or (stepIndex == totalSteps)
            if shouldStore:
                times.append(currentTime)
                for fieldName in system.fieldNames:
                    fieldsByName[fieldName].append(np.array(state[fieldName], copy=True))

        timeSeries = PDETimeSeries(times=np.array(times, dtype=np.float32), fieldsByName=fieldsByName)
        return PDERunResult(
            finalState=state,
            timeSeries=timeSeries,
            dt=dt,
            totalSteps=totalSteps,
        )

    @staticmethod
    def _resolveDt(
        system: PDESystem,
        grid: HyperGrid,
        params: ParameterSet,
        config: PDERunConfig,
    ) -> float:
        if config.dt is not None:
            if config.dt <= 0.0:
                raise ValueError("dt must be positive when provided.")
            return float(config.dt)

        stableDt = system.stableTimeStep(grid, params)
        if stableDt is None:
            raise ValueError("This system does not provide a default stable dt. Pass dt explicitly.")

        dt = float(stableDt) * float(config.stabilitySafetyFactor)
        if dt <= 0.0:
            raise ValueError("Resolved dt must be positive.")
        return dt

    @staticmethod
    def _resolveStoreStride(
        totalSteps: int,
        maxStoredFrames: int,
        storeEveryNSteps: int | None,
    ) -> int:
        if storeEveryNSteps is not None:
            return max(1, int(storeEveryNSteps))
        if maxStoredFrames <= 1:
            return max(1, totalSteps)
        return max(1, int(ceil(totalSteps / (maxStoredFrames - 1))))


def runPDE(
    system: PDESystem,
    grid: HyperGrid,
    params: ParameterSet,
    config: PDERunConfig,
) -> PDERunResult:
    return ExplicitPDERunner().run(system=system, grid=grid, params=params, config=config)
