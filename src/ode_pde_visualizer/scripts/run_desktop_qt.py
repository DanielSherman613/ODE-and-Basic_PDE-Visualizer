from __future__ import annotations

import sys
from pathlib import Path

# Make both the project root and src/ importable when this file is run directly.
projectRoot = Path(__file__).resolve().parents[1]
srcRoot = projectRoot / "src"

if str(projectRoot) not in sys.path:
    sys.path.insert(0, str(projectRoot))
if str(srcRoot) not in sys.path:
    sys.path.insert(0, str(srcRoot))

from PyQt6.QtWidgets import QApplication

from scripts.run_desktop import buildExampleModel
from ode_pde_visualizer.ui.desktop.window import DesktopMainWindow


def main() -> None:
    app = QApplication(sys.argv)
    model = buildExampleModel()
    window = DesktopMainWindow(model)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()