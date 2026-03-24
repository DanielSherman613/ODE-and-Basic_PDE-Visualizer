from dataclasses import dataclass


@dataclass
class ParameterSet:
    values: dict[str, float]
