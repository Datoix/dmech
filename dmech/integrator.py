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
    q_dot0: np.ndarray | None = None,
) -> Tuple[Any, np.ndarray]:
    """
    Integrate an initialized System over the time interval [0, t_max] using scipy's solve_ivp.

    Parameters
    ----------
    system : System
        A fully initialized mechanical system (see dmech.system).
    t_max : float
        The total simulation time.
    fps : int
        Frames per second for result sampling (affects t_eval points).
    rtol : float
        Relative tolerance for the ODE solver.
    atol : float
        Absolute tolerance for the ODE solver.
    method : str
        Integration method for solve_ivp (default: 'RK45').
    q_dot0 : np.ndarray or None
        Optional initial velocities for the system. If None, uses zeros.

    Returns
    -------
    solution : OdeResult (see scipy.integrate.solve_ivp)
        The ODE solver's output, including the time history of the state.
    t_eval : np.ndarray
        Array of time points at which the solution is sampled.
    """

    # Number of generalized coordinates in the system
    n_coords = len(system.coords)

    # Set initial velocity vector
    if q_dot0 is None:
        v0 = np.zeros(n_coords)
    else:
        v0 = np.asarray(q_dot0, dtype=np.float64)

    # Initial state vector: [coordinates, velocities]
    # y0 = [q0, v0]
    y0 = np.concatenate([np.array(system.coords), v0])

    # Times at which to evaluate the solution (for e.g. animation frames)
    t_eval = np.linspace(0, t_max, int(t_max * fps))

    def dynamics(t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute time derivative of the system state.

        Parameters
        ----------
        t : float
            Current time (unused here; but required by scipy's ODE signature).
        y : np.ndarray
            Current state [q, q_dot].

        Returns
        -------
        dydt : np.ndarray
            Derivative of state [dq/dt, d^2q/dt^2]
        """
        # Extract position (q) and velocity (q_dot) from state vector y
        q = jnp.array(y[:n_coords], dtype=jnp.float32)       # q: shape (n,)
        q_dot = jnp.array(y[n_coords:], dtype=jnp.float32)   # q_dot: shape (n,)

        # Compute acceleration: q_ddot and (optionally) constraint forces
        # system.solve returns (q_ddot, lagrange_multipliers)
        q_ddot, _ = system.solve(q, q_dot)

        # Return time derivative of the full state:
        # dy/dt = [dq/dt, d^2q/dt^2] = [q_dot, q_ddot]
        return np.concatenate([np.array(q_dot), np.array(q_ddot)])

    # Integrate the ODE system defined by dynamics()
    solution = solve_ivp(
        dynamics,
        [0, t_max],   # Time interval
        y0,           # Initial state
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    # Return the full solution and time vector
    return solution, t_eval
