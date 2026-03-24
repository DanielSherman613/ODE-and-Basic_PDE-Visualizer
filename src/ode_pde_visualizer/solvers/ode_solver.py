from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from ode_pde_visualizer.core.parameters import ParameterSet
from ode_pde_visualizer.core.trajectory import TrajectorySeries
from ode_pde_visualizer.systems.ode.base import ODESystem


@dataclass(slots=True)
class ODERunConfig:
    totalTime: float
    dt: float
    maxStoredFrames: int = 1200
    method: str = "rk4"


@dataclass(slots=True)
class ODERunResult:
    finalState: np.ndarray
    trajectory: TrajectorySeries
    dt: float
    totalSteps: int


class ODESolver:
    def run(
        self,
        system: ODESystem,
        params: ParameterSet,
        config: ODERunConfig,
    ) -> ODERunResult:
        if config.totalTime < 0.0:
            raise ValueError("totalTime must be non negative.")
        if config.dt <= 0.0:
            raise ValueError("dt must be positive.")

        state = np.asarray(system.initialState(params), dtype=np.float64)
        stateDimension = int(state.size)
        totalSteps = max(1, int(ceil(config.totalTime / config.dt)))
        dt = float(config.totalTime / totalSteps) if config.totalTime > 0.0 else float(config.dt)
        storeStride = self._resolveStoreStride(totalSteps, config.maxStoredFrames)

        times = [0.0]
        storedStates = [np.array(state, copy=True)]
        currentTime = 0.0

        for stepIndex in range(1, totalSteps + 1):
            state = self._step(system, currentTime, state, dt, params, method=config.method)
            currentTime = stepIndex * dt
            if (stepIndex % storeStride == 0) or (stepIndex == totalSteps):
                times.append(currentTime)
                storedStates.append(np.array(state, copy=True))

        stateMatrix = np.asarray(storedStates, dtype=np.float64)
        if stateDimension == 1:
            positions = np.column_stack([stateMatrix[:, 0], np.zeros((stateMatrix.shape[0], 2))])
            axisNames = (system.axisNames[0], "", "")
        elif stateDimension == 2:
            positions = np.column_stack([stateMatrix, np.zeros(stateMatrix.shape[0])])
            axisNames = system.axisNames
        else:
            positions = stateMatrix[:, :3]
            axisNames = system.axisNames

        return ODERunResult(
            finalState=np.array(state, copy=True),
            trajectory=TrajectorySeries(
                times=np.asarray(times, dtype=np.float32),
                positions=np.asarray(positions, dtype=np.float32),
                axisNames=axisNames,
            ),
            dt=dt,
            totalSteps=totalSteps,
        )

    @staticmethod
    def _step(
        system: ODESystem,
        timeValue: float,
        state: np.ndarray,
        dt: float,
        params: ParameterSet,
        method: str,
    ) -> np.ndarray:
        if method.lower() == "euler":
            return state + dt * system.derivative(timeValue, state, params)

        k1 = system.derivative(timeValue, state, params)
        k2 = system.derivative(timeValue + 0.5 * dt, state + 0.5 * dt * k1, params)
        k3 = system.derivative(timeValue + 0.5 * dt, state + 0.5 * dt * k2, params)
        k4 = system.derivative(timeValue + dt, state + dt * k3, params)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @staticmethod
    def _resolveStoreStride(totalSteps: int, maxStoredFrames: int) -> int:
        if maxStoredFrames <= 1:
            return max(1, totalSteps)
        return max(1, int(ceil(totalSteps / (maxStoredFrames - 1))))


def solveODE(
    system: ODESystem,
    params: ParameterSet,
    config: ODERunConfig,
) -> ODERunResult:
    return ODESolver().run(system=system, params=params, config=config)
