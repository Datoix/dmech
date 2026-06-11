from dataclasses import dataclass
from typing import Any, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from dmech.system import System


def coil_path(
    ax_: float,
    ay_: float,
    bx_: float,
    by_: float,
    n_coils: int = 10,
    amplitude: float = 0.09,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zigzag polyline that looks like a coil spring."""
    tang = np.array([bx_ - ax_, by_ - ay_])
    leng = np.linalg.norm(tang) + 1e-10
    unit_t = tang / leng
    unit_n = np.array([-unit_t[1], unit_t[0]])
    n_pts = n_coils * 2 + 2
    pts = []
    for i, t in enumerate(np.linspace(0.0, 1.0, n_pts)):
        base = np.array([ax_, ay_]) + t * tang
        if 0 < i < n_pts - 1:
            base = base + (1 if i % 2 == 1 else -1) * amplitude * unit_n
        pts.append(base)
    pts = np.array(pts)
    return pts[:, 0], pts[:, 1]


@dataclass
class SpringPendulumView:
    spring_k: float
    rest_length: float
    gravity: float
    view_radius: float
    trail_length: int = 150
    fps: int = 60

    @classmethod
    def from_config(cls, config) -> "SpringPendulumView":
        return cls(
            spring_k=config.spring_k,
            rest_length=config.rest_length,
            gravity=config.gravity,
            view_radius=config.view_radius,
        )


def _energy_drift(
    system: System,
    solution: Any,
    spring_k: float,
    rest_length: float,
    gravity: float,
) -> Tuple[np.ndarray, float]:
    n = len(system.coords)
    pos = solution.y[:n]
    vel = solution.y[n:]
    masses = np.array([m for entity in system.entities for m in entity.mass])
    ke = 0.5 * np.sum(masses[:, None] * vel**2, axis=0)
    pe_grav = masses[1] * gravity * pos[1] + masses[3] * gravity * pos[3] + masses[5] * gravity * pos[5]
    l1 = np.sqrt((pos[2] - pos[0]) ** 2 + (pos[3] - pos[1]) ** 2)
    l2 = np.sqrt((pos[4] - pos[2]) ** 2 + (pos[5] - pos[3]) ** 2)
    pe_spring = 0.5 * spring_k * (l1 - rest_length) ** 2 + 0.5 * spring_k * (l2 - rest_length) ** 2
    energy = ke + pe_grav + pe_spring
    e0 = energy[0]
    return np.abs(energy - e0), e0


def animate_spring_pendulum(
    system: System,
    solution: Any,
    t_eval: np.ndarray,
    view: SpringPendulumView,
    show: bool = True,
) -> animation.FuncAnimation:
    """Render the spring double-pendulum simulation."""
    view_r = view.view_radius
    energy_drift, e0 = _energy_drift(
        system, solution, view.spring_k, view.rest_length, view.gravity
    )

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(-view_r, view_r)
    ax.set_ylim(-view_r - 0.2, 0.6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.set_title(
        f"Spring Double Pendulum via KKT  (k={view.spring_k} N/m, L₀={view.rest_length} m)"
    )
    ax.plot(0, 0, "ks", markersize=10, zorder=5)

    trail_x: List[float] = []
    trail_y: List[float] = []
    trail_line, = ax.plot([], [], "-", lw=0.9, color="steelblue", alpha=0.45)
    spring1_line, = ax.plot([], [], "-", lw=2.0, color="#3a86ff", zorder=3)
    spring2_line, = ax.plot([], [], "-", lw=2.0, color="#ff6b35", zorder=3)
    bob1_dot, = ax.plot([], [], "o", markersize=12, color="#3a86ff", zorder=4)
    bob2_dot, = ax.plot([], [], "o", markersize=12, color="#ff6b35", zorder=4)
    info_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        family="monospace",
        color="#444",
    )

    def update(frame: int):
        state = solution.y[:, frame]
        p1x, p1y = state[0], state[1]
        p2x, p2y = state[2], state[3]
        p3x, p3y = state[4], state[5]

        spring1_line.set_data(*coil_path(p1x, p1y, p2x, p2y))
        spring2_line.set_data(*coil_path(p2x, p2y, p3x, p3y))
        bob1_dot.set_data([p2x], [p2y])
        bob2_dot.set_data([p3x], [p3y])

        trail_x.append(p3x)
        trail_y.append(p3y)
        if len(trail_x) > view.trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        l1 = np.sqrt((p2x - p1x) ** 2 + (p2y - p1y) ** 2)
        l2 = np.sqrt((p3x - p2x) ** 2 + (p3y - p2y) ** 2)
        info_text.set_text(
            f"L₁={l1:.3f} m   L₂={l2:.3f} m   (rest {view.rest_length:.2f})\n"
            f"ΔE = {energy_drift[frame]:.2e} J  (E₀={e0:.2f})"
        )

        return spring1_line, spring2_line, bob1_dot, bob2_dot, trail_line, info_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t_eval),
        interval=1000 / view.fps,
        blit=True,
    )

    if show:
        plt.show()

    return ani


@dataclass
class RodPendulumView:
    title: str
    rod_lengths: tuple[float, ...]
    gravity: float
    view_radius: float
    trail_length: int = 150
    fps: int = 60

    @classmethod
    def from_pendulum(cls, config) -> "RodPendulumView":
        return cls(
            title=config.title,
            rod_lengths=(config.rod_length,),
            gravity=config.gravity,
            view_radius=config.view_radius,
        )

    @classmethod
    def from_double(cls, config) -> "RodPendulumView":
        return cls(
            title=config.title,
            rod_lengths=config.rod_lengths,
            gravity=config.gravity,
            view_radius=config.view_radius,
        )


def _rod_constraint_drift(solution: Any, rod_lengths: tuple[float, ...]) -> np.ndarray:
    pos = solution.y[: len(solution.y) // 2]
    n_bobs = pos.shape[0] // 2
    errors = []
    for i, length in enumerate(rod_lengths):
        x1, y1 = pos[i * 2], pos[i * 2 + 1]
        x2, y2 = pos[i * 2 + 2], pos[i * 2 + 3]
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        errors.append(np.abs(dist - length))
    return np.max(errors, axis=0) if errors else np.zeros(pos.shape[1])


def animate_rod_pendulum(
    solution: Any,
    t_eval: np.ndarray,
    view: RodPendulumView,
    show: bool = True,
) -> animation.FuncAnimation:
    """Render a rigid-rod pendulum (1 or 2 rods)."""
    n_coords = len(solution.y) // 2
    n_points = n_coords // 2
    constraint_drift = _rod_constraint_drift(solution, view.rod_lengths)
    view_r = view.view_radius

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-view_r, view_r)
    ax.set_ylim(-view_r, 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.set_title(view.title)
    ax.plot(0, 0, "ks", markersize=10, zorder=5)

    colors = ["#534ab7", "#0f6e56"]
    trail_x: List[float] = []
    trail_y: List[float] = []
    trail_line, = ax.plot([], [], "-", lw=0.9, color="steelblue", alpha=0.45)
    rod_line, = ax.plot([], [], "o-", lw=2.5, color="#888480", markersize=10, zorder=3)
    bob_dots = [
        ax.plot([], [], "o", markersize=12, color=colors[i % len(colors)], zorder=4)[0]
        for i in range(1, n_points)
    ]
    info_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        family="monospace",
        color="#444",
    )

    def update(frame: int):
        state = solution.y[:, frame]
        xs = [state[i] for i in range(0, n_coords, 2)]
        ys = [state[i] for i in range(1, n_coords, 2)]
        rod_line.set_data(xs, ys)

        for i, dot in enumerate(bob_dots):
            dot.set_data([xs[i + 1]], [ys[i + 1]])

        trail_x.append(xs[-1])
        trail_y.append(ys[-1])
        if len(trail_x) > view.trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        if len(view.rod_lengths) == 1:
            dist = np.sqrt((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2)
            info_text.set_text(
                f"L={dist:.3f} m  (target {view.rod_lengths[0]:.2f})\n"
                f"|C|_max = {constraint_drift[frame]:.2e} m"
            )
        else:
            l1 = np.sqrt((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2)
            l2 = np.sqrt((xs[2] - xs[1]) ** 2 + (ys[2] - ys[1]) ** 2)
            info_text.set_text(
                f"L₁={l1:.3f}  L₂={l2:.3f}  (targets {view.rod_lengths[0]:.2f}, {view.rod_lengths[1]:.2f})\n"
                f"|C|_max = {constraint_drift[frame]:.2e} m"
            )

        artists = [rod_line, trail_line, info_text, *bob_dots]
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t_eval),
        interval=1000 / view.fps,
        blit=True,
    )

    if show:
        plt.show()

    return ani
