from typing import Any, Tuple

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from dmech.system import System


def integrate_system(
    system: System,
    t_max: float = 8.0,
    fps: int = 60,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str = "RK45",
) -> Tuple[Any, np.ndarray]:
    """Integrate a initialized System over [0, t_max] using scipy."""
    n_coords = len(system.coords)
    y0 = np.concatenate([np.array(system.coords), np.zeros(n_coords)])
    t_eval = np.linspace(0, t_max, int(t_max * fps))

    def dynamics(t: float, y: np.ndarray) -> np.ndarray:
        q = jnp.array(y[:n_coords], dtype=jnp.float32)
        q_dot = jnp.array(y[n_coords:], dtype=jnp.float32)
        q_ddot, _ = system.solve(q, q_dot)
        return np.concatenate([np.array(q_dot), np.array(q_ddot)])

    solution = solve_ivp(
        dynamics,
        [0, t_max],
        y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return solution, t_eval
