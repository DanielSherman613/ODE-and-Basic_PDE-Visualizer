from __future__ import annotations

from ode_pde_visualizer.app.controller import HyperPDEController, ViewerModel
from ode_pde_visualizer.core.graph_settings import InfiniteAxesSettings
from ode_pde_visualizer.interaction.wheel_binding import PyVistaInteractionBinder
from ode_pde_visualizer.presets.pde_presets import buildHeatNDModel
from ode_pde_visualizer.rendering.pyvista_renderer import PyVistaVolumeRenderer


def buildExampleModel() -> ViewerModel:
    """Build the default desktop demo model.

    This now uses the systems/ + solvers/ pipeline rather than inlining the
    heat equation stepping inside the script.
    """
    return buildHeatNDModel(
        spatialDimensions=5,
        axisHalfSpan=1.0,
        resolutionPerAxis=12,
        alpha=0.02,
        totalTime=1.0,
        storedFrames=80,
        boundaryMode="neumann",
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
