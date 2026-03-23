import numpy as np
import pyvista as pv

from ode_pde_visualizer.core.graph_settings import InfiniteAxesSettings
from ode_pde_visualizer.core.projection import ProjectionResult
from ode_pde_visualizer.rendering.color_policy import ScalarColorPolicy
from ode_pde_visualizer.rendering.infinite_axes_renderer import InfiniteAxesRenderer
class PyVistaVolumeRenderer:
    def __init__(
            self,
            axisSettings: InfiniteAxesSettings | None = None,
    ) -> None:
        self.plotter = pv.Plotter()
        self._initialized = False
        self._volumeActor = None
        self._infoActor = None

        # Integration point: reuse the same plotter that renders the PDE/ODE
        # volume so the custom infinite axes follow the same camera.
        self.axisSettings = axisSettings or InfiniteAxesSettings()
        self.axisOverlay = InfiniteAxesRenderer(
            settings=self.axisSettings,
            plotter=self.plotter,
        )

    def _buildImageData(self, projection: ProjectionResult) -> pv.ImageData:
        volume = projection.volume
        xCoords, yCoords, zCoords = projection.visibleCoords

        spacing = (
            float(xCoords[1] - xCoords[0]) if len(xCoords) > 1 else 1.0,
            float(yCoords[1] - yCoords[0]) if len(yCoords) > 1 else 1.0,
            float(zCoords[1] - zCoords[0]) if len(zCoords) > 1 else 1.0,
        )
        origin = (
            float(xCoords[0]),
            float(yCoords[0]),
            float(zCoords[0]),
        )

        image = pv.ImageData()
        image.dimensions = volume.shape
        image.spacing = spacing
        image.origin = origin
        image.point_data["u"] = volume.flatten(order="F")
        return image

    def render(self, projection: ProjectionResult,
               colorPolicy: ScalarColorPolicy) -> None:
        image = self._buildImageData(projection)

        if not self._initialized:
            # Install the axis overlay once. In integrated mode we do not let it
            # override the app's existing input bindings.
            self.axisOverlay.install(
                enableStandaloneBindings=False,
                addHelpText=False,
            )
            self._initialized = True

        if self._volumeActor is not None:
            try:
                self.plotter.remove_actor(self._volumeActor, render=False)
            except Exception:
                pass
            self._volumeActor = None

        if self._infoActor is not None:
            try:
                self.plotter.remove_actor(self._infoActor, render=False)
            except Exception:
                pass
            self._infoActor = None

        if colorPolicy.symmetricAboutZero:
            vmaxAbs = float(np.max(np.abs(projection.volume)))
            clim = (-vmaxAbs, vmaxAbs)
        elif colorPolicy.vmin is not None and colorPolicy.vmax is not None:
            clim = (colorPolicy.vmin, colorPolicy.vmax)
        else:
            clim = None

        self._volumeActor = self.plotter.add_volume(
            image,
            scalars="u",
            cmap=colorPolicy.cmapName,
            clim=clim,
            shade=True,
        )

        self._infoActor = self.plotter.add_text(
            "\n".join([
                f"Visible axes: {projection.visibleAxisNames}",
                f"Hidden axes: {projection.hiddenAxisSummary}",
            ]),
            position="upper_left",
            font_size=10,
            name="projectionInfo",
        )

        # Rebuild the custom axis last so it stays visible after the volume is
        # updated and follows the current shared camera.
        self.axisOverlay.refresh(force=True)
        self.plotter.render()

    def show(self) -> None:
        self.plotter.show(auto_close=False)
