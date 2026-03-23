from dataclasses import dataclass
import math


@dataclass
class InfiniteAxesSettings:
    # Rough number of ticks to aim for across the visible full axis span.
    # Larger = denser tick marks and labels.
    tickTargetCount: int = 1000

    # VISUAL SPACING PARAMETERS
    # These now scale relative to the tick STEP, not the full axis span.
    # Increase tickLengthFraction to make tick marks longer.
    # Decrease it to make tick marks shorter.
    tickLengthFraction: float = 0.1

    # Increase labelOffsetFraction to push numbers farther from the axis.
    # Decrease it to bring numbers closer to the axis.
    labelOffsetFraction: float = 0.1

    tickLineWidth: int = 2
    tickLabelFontSize: int = 14
    showZeroLabel: bool = True

    minAxisHalfSpan: float = 12.0
    maxAxisHalfSpan: float = float("inf")
    axisScaleWithDistance: float = 1

    xColor: str = "#222222"
    yColor: str = "#222222"
    zColor: str = "#222222"
    axisLineWidth: int = 1

    backgroundColor: str = "white"

    updatePositionTolerance: float = 1
    updateDistanceTolerance: float = 1

    xlabel: str = "x"
    ylabel: str = "y"
    zlabel: str = "z"
