from dataclasses import dataclass

import numpy as np


@dataclass
class AxisSpec:
    name: str
    coords: np.ndarray

    @property
    def size(self) -> int:
        return len(self.coords)

    @property
    def spacing(self) -> float:
        if len(self.coords) < 2:
            return 1.0
        return float(self.coords[1] - self.coords[0])


@dataclass
class HyperGrid:
    axes: list[AxisSpec]

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)

    def axisNames(self) -> list[str]:
        return [axis.name for axis in self.axes]

    def defaultSliceIndex(self, axis: int) -> int:
        return self.axes[axis].size // 2
