from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QSplitter, QWidget
from pyvistaqt import QtInteractor

from ode_pde_visualizer.app.controller import HyperPDEController, ViewerModel
from ode_pde_visualizer.interaction.wheel_binding import PyVistaInteractionBinder
from ode_pde_visualizer.math_tools.expression_parser import (
    ParsedMathExpression,
    formatExpressionSignature,
)
from ode_pde_visualizer.presets.expression_presets import buildBlankExpressionModel
from ode_pde_visualizer.presets.pde_presets import buildHeatNDModel
from ode_pde_visualizer.rendering.pyvista_renderer import PyVistaVolumeRenderer
from ode_pde_visualizer.ui.desktop.equation_panel import EquationPanel


class DesktopMainWindow(QMainWindow):
    def __init__(self, model: ViewerModel) -> None:
        super().__init__()
        self.setWindowTitle("ODE / PDE Visualizer")
        self.resize(1500, 900)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        splitter = QSplitter()
        layout.addWidget(splitter)

        self.equationPanel = EquationPanel()
        self.plotterWidget = QtInteractor(self)

        splitter.addWidget(self.equationPanel)
        splitter.addWidget(self.plotterWidget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1120])

        self.renderer = PyVistaVolumeRenderer(plotter=self.plotterWidget)
        self.controller = HyperPDEController(model, self.renderer)
        self.binder = PyVistaInteractionBinder(self.renderer, self.controller)
        self.binder.bind()

        self._playTimer = QTimer(self)
        self._playTimer.setInterval(80)
        self._playTimer.timeout.connect(self._advanceAnimation)

        self.equationPanel.expressionApplied.connect(self._onExpressionApplied)
        self.equationPanel.systemSelectionChanged.connect(self._onSystemSelectionChanged)
        self.equationPanel.playPauseClicked.connect(self._togglePlayback)

        self._currentExpression: ParsedMathExpression | None = None
        self._activeSystemName = "Heat ND"
        self.equationPanel.setSelectedSystem(self._activeSystemName)
        self.controller.refresh()
        self._refreshOverlayText()
        self._updateStatusBar()
        self.equationPanel.setPlayButtonState(False)

    def _onExpressionApplied(
        self,
        parsed: ParsedMathExpression,
        parameterValues: dict[str, float],
    ) -> None:
        self._stopPlayback()
        self._currentExpression = parsed
        if self._activeSystemName != "Expression":
            self._activeSystemName = "Expression"
            self.equationPanel.setSelectedSystem("Expression")
            self.controller.loadModel(buildBlankExpressionModel(), clearExpression=True)

        self.controller.setExpression(parsed, parameterValues=parameterValues)
        self._refreshOverlayText()
        self._updateStatusBar()
        self.plotterWidget.render()

    def _onSystemSelectionChanged(self, systemName: str) -> None:
        if systemName == self._activeSystemName:
            return

        self._stopPlayback()
        self._activeSystemName = systemName
        self._currentExpression = None

        if systemName == "Expression":
            self.controller.loadModel(buildBlankExpressionModel(), clearExpression=True)
        elif systemName == "Heat ND":
            self.controller.loadModel(buildHeatNDModel(), clearExpression=True)
        else:
            return

        self._refreshOverlayText()
        self._updateStatusBar()

    def _togglePlayback(self) -> None:
        if self.controller.frameCount() <= 1:
            self.equationPanel.setPlayButtonState(False)
            return

        if self._playTimer.isActive():
            self._stopPlayback()
            return

        self._playTimer.start()
        self.equationPanel.setPlayButtonState(True)

    def _stopPlayback(self) -> None:
        if self._playTimer.isActive():
            self._playTimer.stop()
        self.equationPanel.setPlayButtonState(False)

    def _advanceAnimation(self) -> None:
        frameCount = self.controller.frameCount()
        if frameCount <= 1:
            self._stopPlayback()
            return

        nextIndex = self.controller.currentTimeIndex() + 1
        if nextIndex >= frameCount:
            nextIndex = 0
        self.controller.setTimeIndex(nextIndex)
        self._updateStatusBar()

    def _refreshOverlayText(self) -> None:
        signature = self.controller.currentExpressionSignature()
        if signature is None:
            overlayText = f"System: {self._activeSystemName}"
        else:
            summaryText = formatExpressionSignature(signature)
            parameterValues = self.controller.currentExpressionParameterValues()
            parameterText = ", ".join(f"{name}={value:g}" for name, value in parameterValues.items())
            if not parameterText:
                parameterText = "none"
            overlayText = (
                f"Equation: {self._currentExpression.rawText if self._currentExpression is not None else ''}\n"
                f"{summaryText}\n"
                f"Parameter values: {parameterText}"
            )

        self.plotterWidget.add_text(
            overlayText,
            position="lower_left",
            font_size=10,
            name="equationTextOverlay",
        )
        self.plotterWidget.render()

    def _updateStatusBar(self) -> None:
        signature = self.controller.currentExpressionSignature()
        frameText = f"Frame {self.controller.currentTimeIndex() + 1}/{self.controller.frameCount()}"
        if signature is None:
            self.statusBar().showMessage(f"System: {self._activeSystemName} | {frameText}")
            return

        self.statusBar().showMessage(f"{formatExpressionSignature(signature)} | {frameText}")
