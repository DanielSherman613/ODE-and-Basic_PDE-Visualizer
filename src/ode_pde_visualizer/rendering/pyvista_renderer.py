from __future__ import annotations

import numpy as np
import pyvista as pv

from ode_pde_visualizer.core.graph_settings import InfiniteAxesSettings
from ode_pde_visualizer.core.projection import ProjectionResult
from ode_pde_visualizer.core.trajectory import TrajectorySeries
from ode_pde_visualizer.core.view_state import RenderMode
from ode_pde_visualizer.rendering.color_policy import ScalarColorPolicy
from ode_pde_visualizer.rendering.infinite_axes_renderer import InfiniteAxesRenderer


class PyVistaVolumeRenderer:
    def __init__(
        self,
        plotter: pv.Plotter | None = None,
        axisSettings: InfiniteAxesSettings | None = None,
    ) -> None:
        self._ownsPlotter = plotter is None
        self.plotter = plotter if plotter is not None else pv.Plotter()

        self._initialized = False
        self._mainActor = None
        self._secondaryActor = None
        self._infoActor = None

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

    def render(
        self,
        projection: ProjectionResult,
        colorPolicy: ScalarColorPolicy,
    ) -> None:
        self._ensureInitialized()
        self._clearSceneActors()

        clim = self._resolveColorLimits(projection, colorPolicy)
        renderMode = self._inferRenderMode(projection)

        if renderMode == "curve":
            self._mainActor = self._renderCurve(projection, colorPolicy, clim)
        elif renderMode == "surface":
            self._mainActor = self._renderSurface(projection, colorPolicy, clim)
        elif renderMode == "implicit":
            self._mainActor = self._renderImplicit(projection, colorPolicy, clim)
        else:
            self._mainActor = self._renderVolume(projection, colorPolicy, clim)

        shownAxisNames = tuple(name for name in projection.visibleAxisNames if name)
        self._infoActor = self.plotter.add_text(
            "\n".join([
                f"Visible axes: {shownAxisNames if shownAxisNames else ('none',)}",
                f"Hidden axes: {projection.hiddenAxisSummary}",
                f"Render mode: {renderMode}",
            ]),
            position="upper_left",
            font_size=10,
            name="projectionInfo",
        )

        self.axisOverlay.refresh(force=True)
        self.plotter.render()

    def renderTrajectory(
        self,
        trajectory: TrajectorySeries,
        timeIndex: int,
        colorPolicy: ScalarColorPolicy,
    ) -> None:
        self._ensureInitialized()
        self._clearSceneActors()

        if trajectory.positions.size == 0:
            return

        currentIndex = max(0, min(int(timeIndex), trajectory.frameCount - 1))
        history = np.asarray(trajectory.positions[: currentIndex + 1], dtype=float)
        currentPoint = np.asarray(trajectory.positions[currentIndex], dtype=float)

        if len(history) == 1:
            history = np.vstack([history, history])

        line = pv.lines_from_points(history, close=False)
        line["u"] = np.linspace(0.0, 1.0, history.shape[0], dtype=float)
        self._mainActor = self.plotter.add_mesh(
            line,
            scalars="u",
            cmap=colorPolicy.cmapName,
            line_width=4,
            render_lines_as_tubes=True,
            show_scalar_bar=False,
        )

        currentMesh = pv.PolyData(currentPoint.reshape(1, 3))
        self._secondaryActor = self.plotter.add_mesh(
            currentMesh,
            point_size=14,
            render_points_as_spheres=True,
        )

        shownAxisNames = tuple(name for name in trajectory.axisNames if name)
        currentText = ", ".join(
            f"{name or f'x{index+1}'}={value:.3f}"
            for index, (name, value) in enumerate(zip(trajectory.axisNames, currentPoint))
            if name or index < 3
        )
        self._infoActor = self.plotter.add_text(
            "\n".join([
                f"ODE trajectory axes: {shownAxisNames if shownAxisNames else ('none',)}",
                f"Frame: {currentIndex + 1}/{trajectory.frameCount}",
                currentText,
            ]),
            position="upper_left",
            font_size=10,
            name="projectionInfo",
        )

        self.axisOverlay.refresh(force=True)
        self.plotter.render()

    def show(self) -> None:
        if self._ownsPlotter:
            self.plotter.show(auto_close=False)

    def _ensureInitialized(self) -> None:
        if not self._initialized:
            self.axisOverlay.install(
                enableStandaloneBindings=False,
                addHelpText=False,
            )
            self._initialized = True

    def _clearSceneActors(self) -> None:
        self._removeActor(self._mainActor)
        self._removeActor(self._secondaryActor)
        self._removeActor(self._infoActor)
        self._mainActor = None
        self._secondaryActor = None
        self._infoActor = None

    def _resolveColorLimits(
        self,
        projection: ProjectionResult,
        colorPolicy: ScalarColorPolicy,
    ) -> tuple[float, float] | None:
        if colorPolicy.symmetricAboutZero:
            vmaxAbs = float(np.max(np.abs(projection.volume)))
            return (-vmaxAbs, vmaxAbs)
        if colorPolicy.vmin is not None and colorPolicy.vmax is not None:
            return (colorPolicy.vmin, colorPolicy.vmax)
        return None

    @staticmethod
    def _inferRenderMode(projection: ProjectionResult) -> str:
        if projection.renderMode == RenderMode.ISOSURFACE:
            return "implicit"

        shape = projection.volume.shape
        nonSingletonCount = sum(size > 1 for size in shape)
        if nonSingletonCount <= 1:
            return "curve"
        if nonSingletonCount == 2:
            return "surface"
        return "volume"

    def _renderCurve(
        self,
        projection: ProjectionResult,
        colorPolicy: ScalarColorPolicy,
        clim: tuple[float, float] | None,
    ):
        xCoords, _, _ = projection.visibleCoords
        values = np.asarray(projection.volume[:, 0, 0], dtype=float)
        zeros = np.zeros_like(values)
        points = np.column_stack([xCoords, values, zeros])
        poly = pv.lines_from_points(points, close=False)
        poly["u"] = values
        actor = self.plotter.add_mesh(
            poly,
            scalars="u",
            cmap=colorPolicy.cmapName,
            clim=clim,
            line_width=4,
            render_lines_as_tubes=True,
            show_scalar_bar=True,
        )
        self._secondaryActor = self.plotter.add_points(points, render_points_as_spheres=True, point_size=5)
        return actor

    def _renderSurface(
        self,
        projection: ProjectionResult,
        colorPolicy: ScalarColorPolicy,
        clim: tuple[float, float] | None,
    ):
        xCoords, yCoords, _ = projection.visibleCoords
        values = np.asarray(projection.volume[:, :, 0], dtype=float)
        xx, yy = np.meshgrid(xCoords, yCoords, indexing="ij")
        zz = values
        grid = pv.StructuredGrid(xx, yy, zz)
        grid["u"] = values.ravel(order="F")
        return self.plotter.add_mesh(
            grid,
            scalars="u",
            cmap=colorPolicy.cmapName,
            clim=clim,
            show_scalar_bar=True,
            show_edges=False,
            smooth_shading=True,
        )

    def _renderImplicit(
        self,
        projection: ProjectionResult,
        colorPolicy: ScalarColorPolicy,
        clim: tuple[float, float] | None,
    ):
        nonSingletonCount = sum(size > 1 for size in projection.volume.shape)
        scalarClim = clim
        if scalarClim is None:
            vmaxAbs = float(max(np.max(np.abs(projection.volume)), 1.0e-6))
            scalarClim = (-vmaxAbs, vmaxAbs)

        if nonSingletonCount <= 2:
            xCoords, yCoords, zCoords = projection.visibleCoords
            values = np.asarray(projection.volume[:, :, 0], dtype=float)
            xx, yy = np.meshgrid(xCoords, yCoords, indexing="ij")
            zz = np.full_like(xx, float(zCoords[0]) if len(zCoords) > 0 else 0.0)
            grid = pv.StructuredGrid(xx, yy, zz)
            grid["u"] = values.ravel(order="F")
            contour = grid.contour(isosurfaces=[0.0], scalars="u")
            return self.plotter.add_mesh(
                contour,
                scalars="u",
                cmap=colorPolicy.cmapName,
                clim=scalarClim,
                line_width=4,
                render_lines_as_tubes=True,
                show_scalar_bar=True,
            )

        image = self._buildImageData(projection)
        contour = image.contour(isosurfaces=[0.0], scalars="u")
        return self.plotter.add_mesh(
            contour,
            scalars="u",
            cmap=colorPolicy.cmapName,
            clim=scalarClim,
            show_scalar_bar=True,
            smooth_shading=True,
        )

    def _renderVolume(
        self,
        projection: ProjectionResult,
        colorPolicy: ScalarColorPolicy,
        clim: tuple[float, float] | None,
    ):
        image = self._buildImageData(projection)
        return self.plotter.add_volume(
            image,
            scalars="u",
            cmap=colorPolicy.cmapName,
            clim=clim,
            shade=True,
        )

    def _removeActor(self, actor) -> None:
        if actor is None:
            return
        try:
            self.plotter.remove_actor(actor, render=False)
        except Exception:
            pass
