from dataclasses import dataclass

@dataclass
class ScalarColorPolicy:
    cmapName: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    symmetricAboutZero: bool = False
    lockAcrossFrames: bool = True