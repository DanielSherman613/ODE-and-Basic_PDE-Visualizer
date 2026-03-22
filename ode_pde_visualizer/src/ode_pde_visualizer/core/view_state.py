from dataclasses import dataclass, field
from enum import Enum, auto


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
class DimensionWindow:
    startAxis: int = 0
    windowSize: int = 3
    wrap: bool = False

    def visibleAxes(self, ndim: int) -> tuple[int, int, int]:
        if ndim < 3:
            raise ValueError("Need at least 3 spatial dimensions for a 3D window view.")

        maxStart = ndim - self.windowSize
        if self.wrap:
            start = self.startAxis % (maxStart + 1)
        else:
            start = max(0, min(self.startAxis, maxStart))

        return tuple(range(start, start + self.windowSize))

    def scroll(self, delta: int, ndim: int) -> None:
        maxStart = ndim - self.windowSize
        if maxStart < 0:
            return

        newStart = self.startAxis + delta
        if self.wrap:
            self.startAxis = newStart % (maxStart + 1)
        else:
            self.startAxis = max(0, min(newStart, maxStart))


@dataclass
class HiddenAxisPolicy:
    reductionMode: ReductionMode = ReductionMode.SLICE
    sliceIndices: dict[int, int] = field(default_factory=dict)

    def getSliceIndex(self, grid, axis: int) -> int:
        if axis not in self.sliceIndices:
            self.sliceIndices[axis] = grid.defaultSliceIndex(axis)
        return self.sliceIndices[axis]


@dataclass
class ViewState:
    timeIndex: int = 0
    dimensionWindow: DimensionWindow = field(default_factory=DimensionWindow)
    hiddenAxisPolicy: HiddenAxisPolicy = field(default_factory=HiddenAxisPolicy)
    renderMode: RenderMode = RenderMode.VOLUME