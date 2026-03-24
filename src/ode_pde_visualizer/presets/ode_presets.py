from __future__ import annotations

import numpy as np

from ode_pde_visualizer.app.controller import ViewerModel
from ode_pde_visualizer.core.grids import AxisSpec, HyperGrid
from ode_pde_visualizer.core.parameters import ParameterSet
from ode_pde_visualizer.core.time_series import PDETimeSeries
from ode_pde_visualizer.core.trajectory import TrajectorySeries
from ode_pde_visualizer.core.view_state import DimensionWindow, HiddenAxisPolicy, ReductionMode, RenderMode, ViewState
from ode_pde_visualizer.rendering.color_policy import ScalarColorPolicy
from ode_pde_visualizer.solvers.ode_solver import ODERunConfig, solveODE
from ode_pde_visualizer.systems.ode.lorenz import LorenzSystem
from ode_pde_visualizer.systems.ode.lotka_volterra import LotkaVolterraSystem


def _buildTrajectoryViewerModel(trajectory: TrajectorySeries) -> ViewerModel:
    mins = trajectory.positions.min(axis=0)
    maxs = trajectory.positions.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0e-3)
    margins = 0.15 * spans
    axes = []
    for index, axisName in enumerate(trajectory.axisNames):
        low = float(mins[index] - margins[index])
        high = float(maxs[index] + margins[index])
        if axisName:
            coords = np.linspace(low, high, 64, dtype=np.float32)
        else:
            coords = np.array([0.0], dtype=np.float32)
        axes.append(AxisSpec(axisName or f"_pad{index+1}", coords))
    grid = HyperGrid(axes)
    zeroField = np.zeros(grid.shape, dtype=np.float32)
    timeSeries = PDETimeSeries(
        times=np.asarray(trajectory.times, dtype=np.float32),
        fieldsByName={"u": [zeroField.copy() for _ in range(len(trajectory.times))]},
    )
    return ViewerModel(
        grid=grid,
        timeSeries=timeSeries,
        activeFieldName="u",
        viewState=ViewState(
            timeIndex=0,
            dimensionWindow=DimensionWindow(startAxis=0, windowSize=3, wrap=False),
            hiddenAxisPolicy=HiddenAxisPolicy(reductionMode=ReductionMode.SLICE),
            renderMode=RenderMode.VOLUME,
        ),
        colorPolicy=ScalarColorPolicy(cmapName="viridis", symmetricAboutZero=False),
        trajectorySeries=trajectory,
    )


def buildLorenzModel(
    totalTime: float = 40.0,
    dt: float = 0.01,
    maxStoredFrames: int = 3000,
) -> ViewerModel:
    params = ParameterSet(
        {
            "sigma": 10.0,
            "rho": 28.0,
            "beta": 8.0 / 3.0,
            "x0": 1.0,
            "y0": 1.0,
            "z0": 1.0,
        }
    )
    result = solveODE(
        system=LorenzSystem(),
        params=params,
        config=ODERunConfig(totalTime=float(totalTime), dt=float(dt), maxStoredFrames=int(maxStoredFrames)),
    )
    return _buildTrajectoryViewerModel(result.trajectory)


def buildLotkaVolterraModel(
    totalTime: float = 30.0,
    dt: float = 0.02,
    maxStoredFrames: int = 2200,
) -> ViewerModel:
    params = ParameterSet(
        {
            "alpha": 1.5,
            "beta": 1.0,
            "delta": 1.0,
            "gamma": 3.0,
            "prey0": 2.0,
            "predator0": 1.0,
        }
    )
    result = solveODE(
        system=LotkaVolterraSystem(),
        params=params,
        config=ODERunConfig(totalTime=float(totalTime), dt=float(dt), maxStoredFrames=int(maxStoredFrames)),
    )
    return _buildTrajectoryViewerModel(result.trajectory)
