import numpy as np

from ode_pde_visualizer.app.controller import HyperPDEController, ViewerModel
from ode_pde_visualizer.core.grids import AxisSpec, HyperGrid
from ode_pde_visualizer.core.parameters import ParameterSet
from ode_pde_visualizer.core.time_series import PDETimeSeries
from ode_pde_visualizer.core.view_state import DimensionWindow, HiddenAxisPolicy, ViewState
from ode_pde_visualizer.core.projection import ReductionMode, RenderMode
from ode_pde_visualizer.interaction.wheel_binding import PyVistaInteractionBinder
from ode_pde_visualizer.rendering.color_policy import ScalarColorPolicy
from ode_pde_visualizer.systems.pde.heat_nd import NDimHeatEquation
from ode_pde_visualizer.core.graph_settings import InfiniteAxesSettings
from ode_pde_visualizer.rendering.pyvista_renderer import PyVistaVolumeRenderer

def buildExampleModel() -> ViewerModel:
    axes = [
        AxisSpec("x1", np.linspace(-1.0, 1.0, 32)),
        AxisSpec("x2", np.linspace(-1.0, 1.0, 32)),
        AxisSpec("x3", np.linspace(-1.0, 1.0, 32)),
        AxisSpec("x4", np.linspace(-1.0, 1.0, 32)),
        AxisSpec("x5", np.linspace(-1.0, 1.0, 32)),
    ]
    grid = HyperGrid(axes)

    system = NDimHeatEquation()
    params = ParameterSet({"alpha": 0.02})
    state = system.initialCondition(grid, params)

    times = [0.0]
    uFrames = [state["u"].copy()]

    dt = 0.005
    for n in range(15):
        state = system.step(state, grid, dt, params)
        times.append((n + 1) * dt)
        uFrames.append(state["u"].copy())

    series = PDETimeSeries(
        times=np.array(times),
        fieldsByName={"u": uFrames},
    )

    return ViewerModel(
        grid=grid,
        timeSeries=series,
        activeFieldName="u",
        viewState=ViewState(
            timeIndex=0,
            dimensionWindow=DimensionWindow(startAxis=0, windowSize=3,
                                            wrap=False),
            hiddenAxisPolicy=HiddenAxisPolicy(
                reductionMode=ReductionMode.SLICE),
            renderMode=RenderMode.VOLUME,
        ),
        colorPolicy=ScalarColorPolicy(
            cmapName="viridis",
            symmetricAboutZero=False,
        ),
    )


def main() -> None:
    model = buildExampleModel()

    axisSettings = InfiniteAxesSettings(
        tickTargetCount=24,
        tickLengthFraction=0.12,
        labelOffsetFraction=0.18,
        minAxisHalfSpan=8.0,
        axisScaleWithDistance=1.0,
        updatePositionTolerance=0.5,
        updateDistanceTolerance=0.5,
    )

    renderer = PyVistaVolumeRenderer(axisSettings=axisSettings)
    controller = HyperPDEController(model, renderer)

    binder = PyVistaInteractionBinder(renderer, controller)
    binder.bind()

    controller.refresh()
    renderer.show()


if __name__ == "__main__":
    main()
