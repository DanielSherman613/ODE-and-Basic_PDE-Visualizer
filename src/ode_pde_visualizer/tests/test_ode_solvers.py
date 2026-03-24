import numpy as np

from ode_pde_visualizer.core.parameters import ParameterSet
from ode_pde_visualizer.solvers.ode_solver import ODERunConfig, solveODE
from ode_pde_visualizer.systems.ode.lorenz import LorenzSystem
from ode_pde_visualizer.systems.ode.lotka_volterra import LotkaVolterraSystem


def test_lorenz_solver_produces_trajectory() -> None:
    result = solveODE(
        system=LorenzSystem(),
        params=ParameterSet({"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0, "x0": 1.0, "y0": 1.0, "z0": 1.0}),
        config=ODERunConfig(totalTime=0.2, dt=0.01, maxStoredFrames=50),
    )
    assert result.trajectory.positions.shape[1] == 3
    assert result.trajectory.frameCount >= 2
    assert np.isfinite(result.trajectory.positions).all()


def test_lotka_volterra_solver_pads_to_3d_positions() -> None:
    result = solveODE(
        system=LotkaVolterraSystem(),
        params=ParameterSet({"alpha": 1.5, "beta": 1.0, "delta": 1.0, "gamma": 3.0, "prey0": 2.0, "predator0": 1.0}),
        config=ODERunConfig(totalTime=0.2, dt=0.01, maxStoredFrames=50),
    )
    assert result.trajectory.positions.shape[1] == 3
    assert np.allclose(result.trajectory.positions[:, 2], 0.0)
