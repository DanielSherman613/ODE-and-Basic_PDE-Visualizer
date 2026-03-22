from ode_pde_visualizer.src.ode_pde_visualizer.app.controller import \
    HyperPDEController
from ode_pde_visualizer.src.ode_pde_visualizer.rendering.pyvista_renderer import \
    PyVistaVolumeRenderer


class PyVistaInteractionBinder:
    def __init__(self, renderer: PyVistaVolumeRenderer, controller: HyperPDEController) -> None:
        self.renderer = renderer
        self.controller = controller

    def bind(self) -> None:
        iren = self.renderer.plotter.iren

        def onWheelForward(*_args) -> None:
            self.controller.scrollDimensionWindow(+1)

        def onWheelBackward(*_args) -> None:
            self.controller.scrollDimensionWindow(-1)

        def onRightBracket() -> None:
            self.controller.nextFrame()

        def onLeftBracket() -> None:
            self.controller.previousFrame()

        iren.add_observer("MouseWheelForwardEvent", onWheelForward)
        iren.add_observer("MouseWheelBackwardEvent", onWheelBackward)

        self.renderer.plotter.add_key_event("]", onRightBracket)
        self.renderer.plotter.add_key_event("[", onLeftBracket)