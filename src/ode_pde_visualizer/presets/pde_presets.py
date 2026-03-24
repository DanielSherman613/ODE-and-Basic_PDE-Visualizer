from __future__ import annotations

import math

import numpy as np

from ode_pde_visualizer.app.controller import ViewerModel
from ode_pde_visualizer.core.grids import AxisSpec, HyperGrid
from ode_pde_visualizer.core.parameters import ParameterSet
from ode_pde_visualizer.core.projection import ReductionMode, RenderMode
from ode_pde_visualizer.core.view_state import DimensionWindow, HiddenAxisPolicy, ViewState
from ode_pde_visualizer.rendering.color_policy import ScalarColorPolicy
from ode_pde_visualizer.solvers.pde_runner import PDERunConfig, runPDE
from ode_pde_visualizer.systems.pde.heat_nd import NDimHeatEquation


def recommendedResolutionForDimensionCount(
    spatialDimensions: int,
    targetCellBudget: int = 300_000,
    minimumResolution: int = 10,
) -> int:
    if spatialDimensions <= 0:
        raise ValueError("spatialDimensions must be positive.")
    estimate = int(round(targetCellBudget ** (1.0 / spatialDimensions)))
    return max(minimumResolution, estimate)


def buildHeatNDModel(
    spatialDimensions: int = 5,
    axisHalfSpan: float = 1.0,
    resolutionPerAxis: int | None = None,
    alpha: float = 0.02,
    totalTime: float = 1.0,
    dt: float | None = None,
    storedFrames: int = 64,
    boundaryMode: str = "neumann",
) -> ViewerModel:
    if spatialDimensions < 3:
        raise ValueError("Heat ND viewer currently expects at least 3 spatial dimensions.")

    resolution = resolutionPerAxis or recommendedResolutionForDimensionCount(spatialDimensions)
    coords = np.linspace(-axisHalfSpan, axisHalfSpan, resolution, dtype=np.float32)
    axes = [AxisSpec(f"x{axisIndex + 1}", coords.copy()) for axisIndex in range(spatialDimensions)]
    grid = HyperGrid(axes)

    system = NDimHeatEquation(boundaryMode=boundaryMode)
    params = ParameterSet({"alpha": float(alpha)})
    runResult = runPDE(
        system=system,
        grid=grid,
        params=params,
        config=PDERunConfig(
            totalTime=float(totalTime),
            dt=dt,
            maxStoredFrames=int(storedFrames),
        ),
    )

    maxStartAxis = max(0, grid.ndim - 3)
    initialStartAxis = min(0, maxStartAxis)

    return ViewerModel(
        grid=grid,
        timeSeries=runResult.timeSeries,
        activeFieldName="u",
        viewState=ViewState(
            timeIndex=0,
            dimensionWindow=DimensionWindow(startAxis=initialStartAxis, windowSize=3, wrap=False),
            hiddenAxisPolicy=HiddenAxisPolicy(reductionMode=ReductionMode.SLICE),
            renderMode=RenderMode.VOLUME,
        ),
        colorPolicy=ScalarColorPolicy(cmapName="viridis", symmetricAboutZero=False),
    )
