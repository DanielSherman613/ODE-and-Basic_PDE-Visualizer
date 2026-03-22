from dataclasses import dataclass
import numpy as np

@dataclass
class PDETimeSeries:
    times: np.ndarray
    fieldsByName: dict[str, list[np.ndarray]]

    def getFieldAt(self, fieldName: str, timeIndex: int) -> np.ndarray:
        return self.fieldsByName[fieldName][timeIndex]
