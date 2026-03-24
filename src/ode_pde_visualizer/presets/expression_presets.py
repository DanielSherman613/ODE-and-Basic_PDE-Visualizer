from __future__ import annotations

import numpy as np

from ode_pde_visualizer.app.controller import ViewerModel
from ode_pde_visualizer.core.grids import AxisSpec, HyperGrid
from ode_pde_visualizer.core.time_series import PDETimeSeries
from ode_pde_visualizer.core.view_state import DimensionWindow, HiddenAxisPolicy, RenderMode, ReductionMode, ViewState
from ode_pde_visualizer.rendering.color_policy import ScalarColorPolicy


def buildBlankExpressionModel(
    axisHalfSpan: float = 1.0,
    axisResolution: int = 48,
) -> ViewerModel:
    coords = np.linspace(-axisHalfSpan, axisHalfSpan, axisResolution, dtype=np.float32)
    grid = HyperGrid([
        AxisSpec("x", coords.copy()),
        AxisSpec("y", coords.copy()),
        AxisSpec("z", coords.copy()),
    ])
    zeroField = np.zeros(grid.shape, dtype=np.float32)
    timeSeries = PDETimeSeries(
        times=np.array([0.0], dtype=np.float32),
        fieldsByName={"u": [zeroField]},
    )
    viewState = ViewState(
        timeIndex=0,
        dimensionWindow=DimensionWindow(startAxis=0, windowSize=3, wrap=False),
        hiddenAxisPolicy=HiddenAxisPolicy(reductionMode=ReductionMode.SLICE),
        renderMode=RenderMode.VOLUME,
    )
    return ViewerModel(
        grid=grid,
        timeSeries=timeSeries,
        activeFieldName="u",
        viewState=viewState,
        colorPolicy=ScalarColorPolicy(cmapName="viridis", symmetricAboutZero=False),
    )
