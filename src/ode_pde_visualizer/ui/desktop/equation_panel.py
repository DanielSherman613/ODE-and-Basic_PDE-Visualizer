from __future__ import annotations

import io

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ode_pde_visualizer.math_tools.expression_parser import (
    ParsedMathExpression,
    analyzeParsedExpression,
    formatExpressionSignature,
    parseMathExpression,
)


class LatexPreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("Type an expression to see a formatted preview.")
        self.setMinimumHeight(90)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setFrameShape(QFrame.Shape.StyledPanel)

    def setLatex(self, latexText: str) -> None:
        figure = Figure(figsize=(0.01, 0.01), dpi=180)
        figure.patch.set_alpha(0.0)
        canvas = FigureCanvasAgg(figure)

        textArtist = figure.text(0.0, 0.0, f"${latexText}$", fontsize=16)
        canvas.draw()
        bbox = textArtist.get_window_extent(renderer=canvas.get_renderer()).expanded(1.08, 1.35)

        width = max(1, int(bbox.width))
        height = max(1, int(bbox.height))
        figure.set_size_inches(width / figure.dpi, height / figure.dpi)
        figure.clear()
        figure.patch.set_alpha(0.0)
        figure.text(0.03, 0.15, f"${latexText}$", fontsize=16)

        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.05,
        )

        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue(), "PNG")
        self.setPixmap(pixmap)

    def clearPreview(self) -> None:
        self.clear()
        self.setText("Type an expression to see a formatted preview.")


class EquationPanel(QWidget):
    expressionApplied = pyqtSignal(object, object)
    systemSelectionChanged = pyqtSignal(str)
    playPauseClicked = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.systemCombo = QComboBox()
        self.systemCombo.addItems(["Expression", "Heat ND", "Lorenz", "Lotka Volterra"])

        self.examplesButton = QToolButton()
        self.examplesButton.setText("✦")
        self.examplesButton.setToolTip("Insert an example that shows a feature of the visualizer")
        self.examplesButton.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.examplesButton.setAutoRaise(True)
        self.examplesMenu = self._buildExamplesMenu()
        self.examplesButton.setMenu(self.examplesMenu)

        self.input = QLineEdit()
        self.input.setPlaceholderText(
            "Examples: x^2+y^2+z^2=r^2, diff(sin(x),x), lap(sin(x)*cos(y)), div(y,-x)"
        )

        self.preview = LatexPreviewLabel()
        self.status = QLabel(
            "Type a function, an implicit equation, or apply an operator from the panel below."
        )
        self.status.setWordWrap(True)

        self.parameterTitle = QLabel("Parameters")
        self.parameterForm = QFormLayout()
        self.parameterContainer = QWidget()
        self.parameterContainer.setLayout(self.parameterForm)
        self._parameterInputs: dict[str, QLineEdit] = {}
        self.parameterTitle.hide()
        self.parameterContainer.hide()

        self.operatorTitle = QLabel("Operators")
        self.operatorContainer = QWidget()
        operatorGrid = QGridLayout(self.operatorContainer)
        operatorGrid.setContentsMargins(0, 0, 0, 0)
        operatorGrid.setHorizontalSpacing(6)
        operatorGrid.setVerticalSpacing(6)
        self._buildOperatorButtons(operatorGrid)

        self.applyButton = QPushButton("Apply")
        self.playPauseButton = QPushButton("Play")

        systemRow = QHBoxLayout()
        systemRow.addWidget(QLabel("System"))
        systemRow.addWidget(self.systemCombo)

        topRow = QHBoxLayout()
        topRow.addWidget(QLabel("f(...) ="))
        topRow.addWidget(self.input)

        buttonRow = QHBoxLayout()
        buttonRow.addWidget(self.applyButton)
        buttonRow.addWidget(self.playPauseButton)

        headerRow = QHBoxLayout()
        headerRow.addWidget(QLabel("Equation editor"))
        headerRow.addStretch(1)
        headerRow.addWidget(self.examplesButton)

        layout = QVBoxLayout(self)
        layout.addLayout(headerRow)
        layout.addLayout(systemRow)
        layout.addLayout(topRow)
        layout.addWidget(QLabel("Formatted preview"))
        layout.addWidget(self.preview)
        layout.addWidget(self.parameterTitle)
        layout.addWidget(self.parameterContainer)
        layout.addWidget(self.operatorTitle)
        layout.addWidget(self.operatorContainer)
        layout.addLayout(buttonRow)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.input.textChanged.connect(self._onTextChanged)
        self.input.returnPressed.connect(self._applyCurrentExpression)
        self.applyButton.clicked.connect(self._applyCurrentExpression)
        self.playPauseButton.clicked.connect(self.playPauseClicked.emit)
        self.systemCombo.currentTextChanged.connect(self.systemSelectionChanged.emit)

    def setSelectedSystem(self, systemName: str) -> None:
        index = self.systemCombo.findText(systemName)
        if index >= 0 and self.systemCombo.currentIndex() != index:
            self.systemCombo.setCurrentIndex(index)

    def setPlayButtonState(self, isPlaying: bool) -> None:
        self.playPauseButton.setText("Pause" if isPlaying else "Play")

    def currentParameterValues(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, editor in self._parameterInputs.items():
            text = editor.text().strip()
            values[name] = float(text) if text else 1.0
        return values

    def _onTextChanged(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            self.preview.clearPreview()
            self._rebuildParameterEditors([])
            self.status.setText(
                "Type a function, an implicit equation, or apply an operator from the panel below."
            )
            return

        try:
            parsed = parseMathExpression(stripped)
            signature = analyzeParsedExpression(parsed)
        except Exception as exc:
            self.preview.clearPreview()
            self._rebuildParameterEditors([])
            self.status.setText(f"Parse error: {exc}")
            return

        self.preview.setLatex(parsed.latexText)
        self._rebuildParameterEditors(signature.parameterNames)
        self.status.setText(formatExpressionSignature(signature))

    def _applyCurrentExpression(self) -> None:
        try:
            parsed = parseMathExpression(self.input.text())
            parameterValues = self.currentParameterValues()
        except Exception as exc:
            self.status.setText(f"Parse error: {exc}")
            return

        self.expressionApplied.emit(parsed, parameterValues)
        signature = analyzeParsedExpression(parsed)
        self.status.setText(f"Applied. {formatExpressionSignature(signature)}")

    def _buildExamplesMenu(self) -> QMenu:
        menu = QMenu(self)
        groups = [
            (
                "Shapes",
                [
                    ("Paraboloid", "x^2 + y^2"),
                    ("Sphere", "x^2 + y^2 + z^2 = r^2"),
                    ("Ripple surface", "sin(3*x) * cos(3*y)"),
                ],
            ),
            (
                "Time",
                [
                    ("Decaying surface", "exp(-t) * (x^2 + y^2)"),
                    ("Breathing sphere", "x^2 + y^2 + z^2 = (1 + 0.25*sin(t))^2"),
                ],
            ),
            (
                "Operators",
                [
                    ("Laplacian", "lap(sin(x) * cos(y))"),
                    ("Gradient magnitude", "grad(x^2 + y^2 + z^2)"),
                    ("Divergence", "div(x*y, -x*y, z)"),
                ],
            ),
            (
                "Derivatives and integrals",
                [
                    ("d/dx of sin(x)", "diff(sin(x), x)"),
                    ("Integral of x^2", "integrate(x^2, x)"),
                ],
            ),
            (
                "Higher dimensions",
                [
                    ("4D polynomial slice", "(x+1)*(y+1)*(z+1)*(n+1)"),
                    ("5D mixed axes", "x1^2 + x2^2 + x3^2 - x4 + x5"),
                ],
            ),
        ]

        for groupLabel, examples in groups:
            section = menu.addMenu(groupLabel)
            for title, expressionText in examples:
                action = section.addAction(title)
                action.triggered.connect(
                    lambda checked=False, expr=expressionText: self._applyExampleExpression(expr)
                )

        return menu

    def _applyExampleExpression(self, expressionText: str) -> None:
        self.systemCombo.setCurrentText("Expression")
        self.input.setText(expressionText)
        self.input.setFocus()
        self.input.selectAll()

    def _rebuildParameterEditors(self, parameterNames: list[str]) -> None:
        existingValues = {
            name: editor.text()
            for name, editor in self._parameterInputs.items()
        }
        while self.parameterForm.rowCount() > 0:
            self.parameterForm.removeRow(0)
        self._parameterInputs = {}

        if not parameterNames:
            self.parameterTitle.hide()
            self.parameterContainer.hide()
            return

        self.parameterTitle.show()
        self.parameterContainer.show()
        for name in parameterNames:
            editor = QLineEdit(existingValues.get(name, "1.0"))
            editor.setPlaceholderText("1.0")
            self.parameterForm.addRow(f"{name} =", editor)
            self._parameterInputs[name] = editor

    def _buildOperatorButtons(self, grid: QGridLayout) -> None:
        specs = [
            ("∂/∂x", lambda: self._wrapUnary("diff({expr}, x)")),
            ("∂/∂y", lambda: self._wrapUnary("diff({expr}, y)")),
            ("∂/∂z", lambda: self._wrapUnary("diff({expr}, z)")),
            ("∂²/∂x²", lambda: self._wrapUnary("diff({expr}, x, 2)")),
            ("∂²/∂y²", lambda: self._wrapUnary("diff({expr}, y, 2)")),
            ("∂²/∂z²", lambda: self._wrapUnary("diff({expr}, z, 2)")),
            ("|∇f|", lambda: self._wrapUnary("grad({expr})")),
            ("∇²f", lambda: self._wrapUnary("lap({expr})")),
            ("∇·F", self._insertDivergenceTemplate),
        ]

        for index, (label, callback) in enumerate(specs):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row = index // 3
            column = index % 3
            grid.addWidget(button, row, column)

    def _wrapUnary(self, template: str) -> None:
        expr = self._selectedOrWholeExpression()
        if not expr:
            expr = "x"
        replacement = template.format(expr=expr)
        self._replaceSelectionOrWhole(replacement)

    def _insertDivergenceTemplate(self) -> None:
        expr = self._selectedOrWholeExpression()
        if expr and "," in expr:
            replacement = f"div({expr})"
        else:
            replacement = "div(Fx, Fy, Fz)"
        self._replaceSelectionOrWhole(replacement)

    def _selectedOrWholeExpression(self) -> str:
        selected = self.input.selectedText().strip()
        if selected:
            return selected
        return self.input.text().strip()

    def _replaceSelectionOrWhole(self, replacement: str) -> None:
        if self.input.hasSelectedText():
            self.input.insert(replacement)
        else:
            self.input.setText(replacement)
        self.input.setFocus()
