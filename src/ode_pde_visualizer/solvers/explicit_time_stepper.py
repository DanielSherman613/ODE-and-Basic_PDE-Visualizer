from __future__ import annotations

from typing import Any

from ode_pde_visualizer.core.grids import HyperGrid
from ode_pde_visualizer.core.parameters import ParameterSet
from ode_pde_visualizer.systems.pde.base import PDESystem


def forwardEulerStep(
    system: PDESystem,
    state: dict[str, Any],
    grid: HyperGrid,
    dt: float,
    params: ParameterSet,
) -> dict[str, Any]:
    """Advance a PDE state by one explicit Forward Euler step."""
    return system.step(state=state, grid=grid, dt=dt, params=params)
