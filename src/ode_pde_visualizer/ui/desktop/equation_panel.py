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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
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
        self.systemCombo.addItems(["Expression", "Heat ND"])

        self.input = QLineEdit()
        self.input.setPlaceholderText(
            "Examples: x^2, x^2+y^2, x^2+y^2+z^2=r^2, x+y+z+x4, exp(-t)*(x^2+y^2)"
        )

        self.preview = LatexPreviewLabel()
        self.status = QLabel(
            "Type a function or an implicit equation. Example: x^2 + y^2 + z^2 = r^2"
        )
        self.status.setWordWrap(True)

        self.parameterTitle = QLabel("Parameters")
        self.parameterForm = QFormLayout()
        self.parameterContainer = QWidget()
        self.parameterContainer.setLayout(self.parameterForm)
        self._parameterInputs: dict[str, QLineEdit] = {}
        self.parameterTitle.hide()
        self.parameterContainer.hide()

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

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Equation editor"))
        layout.addLayout(systemRow)
        layout.addLayout(topRow)
        layout.addWidget(QLabel("Formatted preview"))
        layout.addWidget(self.preview)
        layout.addWidget(self.parameterTitle)
        layout.addWidget(self.parameterContainer)
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
                "Type a function or an implicit equation. Example: x^2 + y^2 + z^2 = r^2"
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
