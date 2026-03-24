from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QSplitter, QWidget
from pyvistaqt import QtInteractor

from ode_pde_visualizer.app.controller import HyperPDEController, ViewerModel
from ode_pde_visualizer.interaction.wheel_binding import PyVistaInteractionBinder
from ode_pde_visualizer.math_tools.expression_parser import (
    ParsedMathExpression,
    formatExpressionSignature,
)
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
        self.controller.refresh()

        self.equationPanel.expressionApplied.connect(self._onExpressionApplied)
        self._currentExpression: ParsedMathExpression | None = None
        self._updateStatusBar()

    def _onExpressionApplied(self, parsed: ParsedMathExpression) -> None:
        self._currentExpression = parsed
        self.controller.setExpression(parsed)

        signature = self.controller.currentExpressionSignature()
        summaryText = formatExpressionSignature(signature) if signature is not None else ""
        self.plotterWidget.add_text(
            f"Equation: {parsed.rawText}\n{summaryText}",
            position="lower_left",
            font_size=10,
            name="equationTextOverlay",
        )
        self._updateStatusBar()
        self.plotterWidget.render()

    def _updateStatusBar(self) -> None:
        signature = self.controller.currentExpressionSignature()
        if signature is None:
            self.statusBar().showMessage("Base demo model loaded.")
            return

        self.statusBar().showMessage(formatExpressionSignature(signature))
