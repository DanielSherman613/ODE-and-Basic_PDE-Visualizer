from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from ode_pde_visualizer.core.grids import HyperGrid
from ode_pde_visualizer.core.view_state import ViewState


class ReductionMode(Enum):
    SLICE = auto()
    MEAN = auto()
    MAX = auto()
    MIN = auto()


class RenderMode(Enum):
    VOLUME = auto()
    ISOSURFACE = auto()
    ORTHO_SLICES = auto()


@dataclass
class ProjectionResult:
    volume: np.ndarray
    visibleAxes: tuple[int, int, int]
    visibleAxisNames: tuple[str, str, str]
    visibleCoords: tuple[np.ndarray, np.ndarray, np.ndarray]
    hiddenAxisSummary: str


class ProjectionEngine:
    def project(
            self,
            field: np.ndarray,
            grid: HyperGrid,
            viewState: ViewState,
    ) -> ProjectionResult:
        if field.shape != grid.shape:
            raise ValueError(
                f"Field shape {field.shape} does not match grid shape {grid.shape}")

        visible = viewState.dimensionWindow.visibleAxes(grid.ndim)
        hidden = [axis for axis in range(grid.ndim) if axis not in visible]
        policy = viewState.hiddenAxisPolicy

        working = field
        hiddenSummaryParts: list[str] = []

        if policy.reductionMode == ReductionMode.SLICE:
            indexer: list[object] = []
            for axis in range(grid.ndim):
                if axis in visible:
                    indexer.append(slice(None))
                else:
                    idx = policy.getSliceIndex(grid, axis)
                    hiddenSummaryParts.append(
                        f"{grid.axes[axis].name}=slice[{idx}]")
                    indexer.append(idx)

            reduced = working[tuple(indexer)]

        else:
            reduced = working
            for axis in sorted(hidden, reverse=True):
                axisName = grid.axes[axis].name

                if policy.reductionMode == ReductionMode.MEAN:
                    reduced = reduced.mean(axis=axis)
                    hiddenSummaryParts.append(f"{axisName}=mean")
                elif policy.reductionMode == ReductionMode.MAX:
                    reduced = reduced.max(axis=axis)
                    hiddenSummaryParts.append(f"{axisName}=max")
                elif policy.reductionMode == ReductionMode.MIN:
                    reduced = reduced.min(axis=axis)
                    hiddenSummaryParts.append(f"{axisName}=min")
                else:
                    raise ValueError("Unsupported reduction mode")

        if reduced.ndim != 3:
            raise ValueError(
                f"Projection must end with 3 dimensions, but got {reduced.ndim}."
            )

        visibleAxisNames = tuple(grid.axes[i].name for i in visible)
        visibleCoords = tuple(grid.axes[i].coords for i in visible)
        hiddenSummary = ", ".join(
            hiddenSummaryParts) if hiddenSummaryParts else "none"

        return ProjectionResult(
            volume=reduced,
            visibleAxes=visible,
            visibleAxisNames=visibleAxisNames,
            visibleCoords=visibleCoords,
            hiddenAxisSummary=hiddenSummary,
        )
