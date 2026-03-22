import numpy as np
import pyvista as pv

from ode_pde_visualizer.core.projection import ProjectionResult
from ode_pde_visualizer.rendering.color_policy import ScalarColorPolicy

class PyVistaVolumeRenderer:
    def __init__(self) -> None:
        self.plotter = pv.Plotter()
        self._initialized = False

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
            self.plotter.add_axes()
            self.plotter.show_grid()
            self._initialized = True

        self.plotter.clear()

        if colorPolicy.symmetricAboutZero:
            vmaxAbs = float(np.max(np.abs(projection.volume)))
            clim = (-vmaxAbs, vmaxAbs)
        elif colorPolicy.vmin is not None and colorPolicy.vmax is not None:
            clim = (colorPolicy.vmin, colorPolicy.vmax)
        else:
            clim = None

        self.plotter.add_volume(
            image,
            scalars="u",
            cmap=colorPolicy.cmapName,
            clim=clim,
            shade=True,
        )

        self.plotter.add_text(
            "\n".join([
                f"Visible axes: {projection.visibleAxisNames}",
                f"Hidden axes: {projection.hiddenAxisSummary}",
            ]),
            position="upper_left",
            font_size=10,
        )

        self.plotter.render()

    def show(self) -> None:
        self.plotter.show(auto_close=False)
