from __future__ import annotations

import io

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
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
    expressionApplied = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()

        self.input = QLineEdit()
        self.input.setPlaceholderText("Examples: x^2, sin(x), x^2 + y^2, exp(-x^2)")

        self.preview = LatexPreviewLabel()
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)

        self.applyButton = QPushButton("Apply")

        topRow = QHBoxLayout()
        topRow.addWidget(QLabel("f(x, y, z, t) ="))
        topRow.addWidget(self.input)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Equation editor"))
        layout.addLayout(topRow)
        layout.addWidget(QLabel("Formatted preview"))
        layout.addWidget(self.preview)
        layout.addWidget(self.applyButton)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.input.textChanged.connect(self._onTextChanged)
        self.input.returnPressed.connect(self._applyCurrentExpression)
        self.applyButton.clicked.connect(self._applyCurrentExpression)

    def _onTextChanged(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            self.preview.clearPreview()
            self.status.setText("Ready")
            return

        try:
            parsed = parseMathExpression(stripped)
        except Exception as exc:
            self.preview.clearPreview()
            self.status.setText(f"Parse error: {exc}")
            return

        self.preview.setLatex(parsed.latexText)
        self.status.setText("Expression parsed successfully.")

    def _applyCurrentExpression(self) -> None:
        try:
            parsed = parseMathExpression(self.input.text())
        except Exception as exc:
            self.status.setText(f"Parse error: {exc}")
            return

        self.expressionApplied.emit(parsed)
        self.status.setText(f"Applied: {parsed.rawText}")
