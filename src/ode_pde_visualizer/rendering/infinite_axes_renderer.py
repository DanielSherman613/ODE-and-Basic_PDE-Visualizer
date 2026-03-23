from __future__ import annotations

import math

import numpy as np
import pyvista as pv

from ode_pde_visualizer.core.graph_settings import InfiniteAxesSettings


class InfiniteAxesRenderer:
    def __init__(
            self,
            settings: InfiniteAxesSettings,
            plotter: pv.Plotter | None = None,
    ) -> None:
        pv.OFF_SCREEN = False
        self.settings = settings

        # If a plotter is supplied, this renderer behaves like an overlay that
        # attaches infinite axes onto an existing PyVista scene. Otherwise it
        # creates and owns its own standalone plotter window.
        self._ownsPlotter = plotter is None
        self.plotter = plotter or pv.Plotter(
            off_screen=False,
            window_size=[1200, 800],
        )

        self._axisActors: list[object] = []
        self._lastSignature: tuple[int, int, int, int] | None = None
        self._busy = False
        self._wheelObserversAdded = False
        self._renderCallbackAdded = False
        self._standaloneBindingsAdded = False
        self._overlayInitialized = False

        # ===== Tick / label tuning ===== These are the main visual
        # parameters to tweak when spacing looks wrong. They are defined in
        # InfiniteAxesSettings.
        self.tickTargetCount = getattr(settings, "tickTargetCount", 20)
        self.tickLengthFraction = getattr(settings, "tickLengthFraction", 0.18)
        self.labelOffsetFraction = getattr(settings, "labelOffsetFraction",
                                           0.35)
        self.tickLineWidth = getattr(settings, "tickLineWidth", 2)
        self.tickLabelFontSize = getattr(settings, "tickLabelFontSize", 14)
        self.showZeroLabel = getattr(settings, "showZeroLabel", True)
        self._manualScaleMultiplier = 1.0

        self._wheelForwardCount = 0
        self._wheelBackwardCount = 0
        self._wheelStepsPerScaleChange = 5
        self._scaleStepFactor = 1.5

        self._initialCameraPosition = [
            (18.0, 18.0, 12.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]



    def _cameraVectors(self) -> tuple[np.ndarray, np.ndarray, float]:
        cam = self.plotter.camera
        position = np.array(cam.position, dtype=float)
        focal = np.array(cam.focal_point, dtype=float)
        distance = float(np.linalg.norm(position - focal))
        return position, focal, max(distance, 1e-6)

    def _axisHalfSpan(self, distance: float) -> float:
        raw = (
            self.settings.axisScaleWithDistance
            * self._manualScaleMultiplier
            * distance
        )
        return float(
            np.clip(
                raw,
                self.settings.minAxisHalfSpan,
                self.settings.maxAxisHalfSpan,
            )
        )

    def _cameraSignature(self) -> tuple[int, int, int, int]:
        _, focal, distance = self._cameraVectors()

        posTol = self.settings.updatePositionTolerance
        distTol = self.settings.updateDistanceTolerance

        return (
            int(round(focal[0] / posTol)),
            int(round(focal[1] / posTol)),
            int(round(focal[2] / posTol)),
            int(round(distance / distTol)),
        )

    def _removeAxisActors(self) -> None:
        for actor in self._axisActors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._axisActors.clear()

    def _addArrow(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        color: str,
        shaftRadius: float,
        tipRadius: float,
        tipLength: float,
    ) -> None:
        arrow = pv.Arrow(
            start=start,
            direction=direction,
            tip_length=tipLength,
            tip_radius=tipRadius,
            shaft_radius=shaftRadius,
            scale="auto",
        )
        actor = self.plotter.add_mesh(
            arrow,
            color=color,
            lighting=False,
            render=False,
        )
        self._axisActors.append(actor)

    def _worldPointToDisplay(self, point: np.ndarray) -> tuple[float, float, float]:
        renderer = self.plotter.renderer
        renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        renderer.WorldToDisplay()
        x, y, z = renderer.GetDisplayPoint()
        return float(x), float(y), float(z)

    def _isWorldPointVisible(self, point: np.ndarray) -> bool:
        position, focal, _ = self._cameraVectors()

        forward = focal - position
        forwardNorm = np.linalg.norm(forward)
        if forwardNorm <= 1e-9:
            return False
        forward = forward / forwardNorm

        toPoint = point - position
        if float(np.dot(toPoint, forward)) <= 0.0:
            return False

        x, y, z = self._worldPointToDisplay(point)

        width, height = self.plotter.window_size
        inWindow = 0.0 <= x <= float(width) and 0.0 <= y <= float(height)
        inDepth = 0.0 <= z <= 1.0

        return inWindow and inDepth

    def _updateEdgeWarning(self, visibleEdges: list[str]) -> None:
        if visibleEdges:
            message = "you reached the edge of " + ", ".join(visibleEdges) + "!"
        else:
            message = ""

        self.plotter.add_text(
            message,
            position="upper_right",
            font_size=12,
            color="crimson",
            name="edgeWarning",
            shadow=True,
            render=False,
        )

    def _niceStep(self, rawStep: float) -> float:
        if rawStep <= 0.0:
            return 1.0

        exponent = math.floor(math.log10(rawStep))
        scale = 10.0 ** exponent
        fraction = rawStep / scale

        if fraction <= 1.0:
            niceFraction = 1.0
        elif fraction <= 2.0:
            niceFraction = 2.0
        elif fraction <= 5.0:
            niceFraction = 5.0
        else:
            niceFraction = 10.0

        return niceFraction * scale

    def _formatTickLabel(self, value: float) -> str:
        if abs(value) < 1e-10:
            value = 0.0

        roundedInt = round(value)
        if abs(value - roundedInt) < 1e-10:
            return str(int(roundedInt))

        return f"{value:.6g}"

    def _tickValues(self, minValue: float, maxValue: float, step: float) -> \
    list[float]:
        if step <= 0.0:
            return []

        start = math.ceil(minValue / step) * step
        values: list[float] = []

        current = start
        while current <= maxValue + 1e-10:
            if self.showZeroLabel or abs(current) > 1e-10:
                values.append(current)
            current += step

        return values

    def _addTickLines(self, tickSegments: list[list[float]],
                      color: str) -> None:
        if not tickSegments:
            return

        tickPoints = np.array(tickSegments, dtype=float)
        actor = self.plotter.add_lines(
            tickPoints,
            color=color,
            width=self.tickLineWidth,
            connected=False,
        )
        self._axisActors.append(actor)

    def _addTickLabels(
            self,
            labelPoints: list[list[float]],
            labels: list[str],
            color: str,
            name: str,
    ) -> None:
        if not labelPoints:
            return

        actor = self.plotter.add_point_labels(
            np.array(labelPoints, dtype=float),
            labels,
            text_color=color,
            font_size=self.tickLabelFontSize,
            bold=False,
            show_points=False,
            shape=None,
            always_visible=False,
            name=name,
            render=False,
        )
        self._axisActors.append(actor)

    def _buildAxisTicksAndLabels(
            self,
            axisName: str,
            centerValue: float,
            halfSpan: float,
            color: str,
    ) -> None:
        visibleMin = centerValue - halfSpan
        visibleMax = centerValue + halfSpan

        rawStep = (2.0 * halfSpan) / max(self.tickTargetCount, 1)
        step = self._niceStep(rawStep)

        # ===== Main tick geometry controls =====
        # IMPORTANT:
        # Tick size and label distance should scale with the tick spacing,
        # not the whole visible axis span.
        #
        # If ticks are too long, reduce tickLengthFraction in settings.
        # If labels are too far away, reduce labelOffsetFraction in settings.
        tickHalfLength = self.tickLengthFraction * step
        labelOffset = self.labelOffsetFraction * step

        tickValues = self._tickValues(visibleMin, visibleMax, step)

        tickSegments: list[list[float]] = []
        labelPoints: list[list[float]] = []
        labels: list[str] = []

        for tickValue in tickValues:
            if axisName == "x":
                tickSegments.append([tickValue, -tickHalfLength, 0.0])
                tickSegments.append([tickValue, tickHalfLength, 0.0])
                labelPoints.append([tickValue, labelOffset, 0.0])

            elif axisName == "y":
                tickSegments.append([-tickHalfLength, tickValue, 0.0])
                tickSegments.append([tickHalfLength, tickValue, 0.0])
                labelPoints.append([labelOffset, tickValue, 0.0])

            elif axisName == "z":
                tickSegments.append([-tickHalfLength, 0.0, tickValue])
                tickSegments.append([tickHalfLength, 0.0, tickValue])
                labelPoints.append([labelOffset, 0.0, tickValue])

            labels.append(self._formatTickLabel(tickValue))

        self._addTickLines(tickSegments, color=color)
        self._addTickLabels(
            labelPoints,
            labels,
            color=color,
            name=f"{axisName}TickLabels",
        )

    def _buildAxisSegments(self) -> None:
        _, focal, distance = self._cameraVectors()
        halfSpan = self._axisHalfSpan(distance)

        fx, fy, fz = float(focal[0]), float(focal[1]), float(focal[2])

        xPoints = np.array(
            [
                [fx - halfSpan, 0.0, 0.0],
                [fx + halfSpan, 0.0, 0.0],
            ],
            dtype=float,
        )

        yPoints = np.array(
            [
                [0.0, fy - halfSpan, 0.0],
                [0.0, fy + halfSpan, 0.0],
            ],
            dtype=float,
        )

        zPoints = np.array(
            [
                [0.0, 0.0, fz - halfSpan],
                [0.0, 0.0, fz + halfSpan],
            ],
            dtype=float,
        )

        self._removeAxisActors()

        self._axisActors.append(
            self.plotter.add_lines(
                xPoints,
                color=self.settings.xColor,
                width=self.settings.axisLineWidth,
            )
        )
        self._axisActors.append(
            self.plotter.add_lines(
                yPoints,
                color=self.settings.yColor,
                width=self.settings.axisLineWidth,
            )
        )
        self._axisActors.append(
            self.plotter.add_lines(
                zPoints,
                color=self.settings.zColor,
                width=self.settings.axisLineWidth,
            )
        )

        self._buildAxisTicksAndLabels(
            axisName="x",
            centerValue=fx,
            halfSpan=halfSpan,
            color=self.settings.xColor,
        )
        self._buildAxisTicksAndLabels(
            axisName="y",
            centerValue=fy,
            halfSpan=halfSpan,
            color=self.settings.yColor,
        )
        self._buildAxisTicksAndLabels(
            axisName="z",
            centerValue=fz,
            halfSpan=halfSpan,
            color=self.settings.zColor,
        )

        arrowLength = max(0.8, 0.08 * halfSpan)
        shaftRadius = max(0.03, 0.08 * 1/arrowLength)
        tipRadius = max(0.08, 0.18 * 1/arrowLength)
        tipLength = 0.30

        xPosTip = np.array([fx + halfSpan, 0.0, 0.0], dtype=float)
        xNegTip = np.array([fx - halfSpan, 0.0, 0.0], dtype=float)
        yPosTip = np.array([0.0, fy + halfSpan, 0.0], dtype=float)
        yNegTip = np.array([0.0, fy - halfSpan, 0.0], dtype=float)
        zPosTip = np.array([0.0, 0.0, fz + halfSpan], dtype=float)
        zNegTip = np.array([0.0, 0.0, fz - halfSpan], dtype=float)

        self._addArrow(
            start=xPosTip - np.array([arrowLength, 0.0, 0.0]),
            direction=np.array([arrowLength, 0.0, 0.0]),
            color=self.settings.xColor,
            shaftRadius=shaftRadius,
            tipRadius=tipRadius,
            tipLength=tipLength,
        )
        self._addArrow(
            start=xNegTip + np.array([arrowLength, 0.0, 0.0]),
            direction=np.array([-arrowLength, 0.0, 0.0]),
            color=self.settings.xColor,
            shaftRadius=shaftRadius,
            tipRadius=tipRadius,
            tipLength=tipLength,
        )

        self._addArrow(
            start=yPosTip - np.array([0.0, arrowLength, 0.0]),
            direction=np.array([0.0, arrowLength, 0.0]),
            color=self.settings.yColor,
            shaftRadius=shaftRadius,
            tipRadius=tipRadius,
            tipLength=tipLength,
        )
        self._addArrow(
            start=yNegTip + np.array([0.0, arrowLength, 0.0]),
            direction=np.array([0.0, -arrowLength, 0.0]),
            color=self.settings.yColor,
            shaftRadius=shaftRadius,
            tipRadius=tipRadius,
            tipLength=tipLength,
        )

        self._addArrow(
            start=zPosTip - np.array([0.0, 0.0, arrowLength]),
            direction=np.array([0.0, 0.0, arrowLength]),
            color=self.settings.zColor,
            shaftRadius=shaftRadius,
            tipRadius=tipRadius,
            tipLength=tipLength,
        )
        self._addArrow(
            start=zNegTip + np.array([0.0, 0.0, arrowLength]),
            direction=np.array([0.0, 0.0, -arrowLength]),
            color=self.settings.zColor,
            shaftRadius=shaftRadius,
            tipRadius=tipRadius,
            tipLength=tipLength,
        )

        visibleEdges: list[str] = []
        if self._isWorldPointVisible(xPosTip):
            visibleEdges.append("+x axis")
        if self._isWorldPointVisible(xNegTip):
            visibleEdges.append("-x axis")
        if self._isWorldPointVisible(yPosTip):
            visibleEdges.append("+y axis")
        if self._isWorldPointVisible(yNegTip):
            visibleEdges.append("-y axis")
        if self._isWorldPointVisible(zPosTip):
            visibleEdges.append("+z axis")
        if self._isWorldPointVisible(zNegTip):
            visibleEdges.append("-z axis")

        self._updateEdgeWarning(visibleEdges)

        nearClip = 0.01
        farClip = max(1000.0, 8.0 * halfSpan)
        self.plotter.camera.clipping_range = (nearClip, farClip)

    def _updateAxesIfNeeded(self, force: bool = False) -> None:
        if self._busy:
            return

        signature = self._cameraSignature()
        if not force and signature == self._lastSignature:
            return

        self._busy = True
        try:
            self._buildAxisSegments()
            self._lastSignature = signature
        finally:
            self._busy = False

    def _expandAxes(self) -> None:
        self._manualScaleMultiplier *= self._scaleStepFactor
        self._wheelForwardCount = 0
        self._wheelBackwardCount = 0
        self._lastSignature = None
        self._updateAxesIfNeeded(force=True)
        self.plotter.render()

    def _shrinkAxes(self) -> None:
        self._manualScaleMultiplier /= self._scaleStepFactor
        self._manualScaleMultiplier = max(1.0, self._manualScaleMultiplier)
        self._wheelForwardCount = 0
        self._wheelBackwardCount = 0
        self._lastSignature = None
        self._updateAxesIfNeeded(force=True)
        self.plotter.render()

    def _resetAxesState(self) -> None:
        self._busy = False
        self._manualScaleMultiplier = 1.0
        self._wheelForwardCount = 0
        self._wheelBackwardCount = 0
        self._lastSignature = None

        self._removeAxisActors()

        self.plotter.camera_position = self._initialCameraPosition
        self.plotter.camera.clipping_range = (0.01, 1000.0)

        self.plotter.add_text(
            "",
            position="upper_right",
            font_size=12,
            color="crimson",
            name="edgeWarning",
            shadow=True,
            render=False,
        )

        self._updateAxesIfNeeded(force=True)
        self.plotter.render()

    def _onWheelZoomOut(self, *_args) -> None:
        self._wheelBackwardCount += 1
        self._wheelForwardCount = 0

        if self._wheelBackwardCount >= self._wheelStepsPerScaleChange:
            self._wheelBackwardCount = 0
            self._expandAxes()

    def _onWheelZoomIn(self, *_args) -> None:
        self._wheelForwardCount += 1
        self._wheelBackwardCount = 0

        if self._wheelForwardCount >= self._wheelStepsPerScaleChange:
            self._wheelForwardCount = 0
            self._shrinkAxes()
    def install(
            self,
            enableStandaloneBindings: bool = True,
            addHelpText: bool = True,
    ) -> None:
        """Attach the infinite axis overlay to the current plotter.

        In standalone mode this also installs the custom camera bindings and
        axis scaling hotkeys. In integrated mode, pass
        enableStandaloneBindings=False so the main app keeps control of input.
        """
        pl = self.plotter
        pl.set_background(self.settings.backgroundColor)
        pl.disable_parallel_projection()

        if not self._overlayInitialized:
            pl.add_axes(
                xlabel=self.settings.xlabel,
                ylabel=self.settings.ylabel,
                zlabel=self.settings.zlabel,
                line_width=2,
            )
            self._overlayInitialized = True

        pl.add_text(
            "",
            position="upper_right",
            font_size=12,
            color="crimson",
            name="edgeWarning",
            shadow=True,
            render=False,
        )

        if enableStandaloneBindings and not self._standaloneBindingsAdded:
            pl.enable_custom_trackball_style(
                left="pan",
                shift_left="rotate",
                middle="pan",
                right="dolly",
            )

            if addHelpText:
                pl.add_text(
                    "Left drag: pan    Shift + left: rotate    Right drag: zoom    K: expand    J: shrink    V: reset",
                    position="upper_left",
                    font_size=10,
                    color="black",
                    name="helpText",
                )

            pl.camera_position = self._initialCameraPosition
            pl.add_key_event("k", self._expandAxes)
            pl.add_key_event("j", self._shrinkAxes)
            pl.add_key_event("v", self._resetAxesState)

            if not self._wheelObserversAdded:
                pl.iren.add_observer("MouseWheelBackwardEvent",
                                     self._onWheelZoomOut)
                pl.iren.add_observer("MouseWheelForwardEvent", self._onWheelZoomIn)
                self._wheelObserversAdded = True

            self._standaloneBindingsAdded = True

        if not self._renderCallbackAdded:
            def onRender(_plotter) -> None:
                self._updateAxesIfNeeded(force=False)

            pl.add_on_render_callback(onRender, render_event=True)
            self._renderCallbackAdded = True

        self._updateAxesIfNeeded(force=True)

    def refresh(self, force: bool = False) -> None:
        self._updateAxesIfNeeded(force=force)

    def render(self) -> None:
        self.install(enableStandaloneBindings=True, addHelpText=True)

    # def render(self) -> None:
    #     pl = self.plotter
    #     pl.set_background(self.settings.backgroundColor)
    #
    #     pl.disable_parallel_projection()
    #
    #     pl.enable_custom_trackball_style(
    #         left="pan",
    #         shift_left="rotate",
    #         middle="pan",
    #         right="dolly",
    #     )
    #
    #     pl.add_axes(
    #         xlabel=self.settings.xlabel,
    #         ylabel=self.settings.ylabel,
    #         zlabel=self.settings.zlabel,
    #         line_width=2,
    #     )
    #
    # pl.add_text( "Left drag: pan    Shift + left: rotate    Right drag:
    # zoom    K: expand    J: shrink    V: reset", position="upper_left",
    # font_size=10, color="black", name="helpText", )
    #
    #     pl.add_text(
    #         "",
    #         position="upper_right",
    #         font_size=12,
    #         color="crimson",
    #         name="edgeWarning",
    #         shadow=True,
    #     )
    #
    #     pl.camera_position = self._initialCameraPosition
    #
    #     pl.add_key_event("k", self._expandAxes)
    #     pl.add_key_event("j", self._shrinkAxes)
    #     pl.add_key_event("v", self._resetAxesState)
    #
    #     self._updateAxesIfNeeded(force=True)
    #
    # if not self._wheelObserversAdded: pl.iren.add_observer(
    # "MouseWheelBackwardEvent", self._onWheelZoomOut) pl.iren.add_observer(
    # "MouseWheelForwardEvent", self._onWheelZoomIn)
    # self._wheelObserversAdded = True
    #
    #     def onRender(_plotter) -> None:
    #         self._updateAxesIfNeeded(force=False)
    #
    #     pl.add_on_render_callback(onRender, render_event=True)

    def show(self) -> None:
        self.plotter.show(auto_close=False, interactive=True)