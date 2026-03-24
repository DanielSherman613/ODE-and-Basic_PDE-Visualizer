from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TrajectorySeries:
    times: np.ndarray
    positions: np.ndarray
    axisNames: tuple[str, str, str]

    def pointAt(self, timeIndex: int) -> np.ndarray:
        return np.asarray(self.positions[timeIndex], dtype=float)

    @property
    def frameCount(self) -> int:
        return int(len(self.times))
