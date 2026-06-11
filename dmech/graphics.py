from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

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


def positions_from_state(
    state: np.ndarray, n_coords: int | None = None
) -> Tuple[List[float], List[float]]:
    """Extract (xs, ys) from a scipy state vector [q, q_dot]."""
    if n_coords is None:
        n_coords = len(state) // 2
    xs = [float(state[i]) for i in range(0, n_coords, 2)]
    ys = [float(state[i]) for i in range(1, n_coords, 2)]
    return xs, ys


def segment_lengths(xs: Sequence[float], ys: Sequence[float]) -> List[float]:
    return [
        float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]))
        for i in range(len(xs) - 1)
    ]


@dataclass
class AnimationView:
    title: str
    view_radius: float
    trail_length: int = 150
    fps: int = 60
    y_top: float = 0.5
    figsize: Tuple[float, float] = (6, 6)


@dataclass
class SpringPendulumView(AnimationView):
    spring_k: float = 40.0
    rest_length: float = 1.0
    gravity: float = 9.81

    @classmethod
    def from_config(cls, config) -> "SpringPendulumView":
        return cls(
            title=f"Spring Double Pendulum via KKT  (k={config.spring_k} N/m, L₀={config.rest_length} m)",
            spring_k=config.spring_k,
            rest_length=config.rest_length,
            gravity=config.gravity,
            view_radius=config.view_radius,
            y_top=0.6,
            figsize=(6, 7),
        )


@dataclass
class RodPendulumView(AnimationView):
    rod_lengths: tuple[float, ...] = ()
    gravity: float = 9.81

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


class Trail:
    def __init__(self, ax, max_length: int):
        self._xs: List[float] = []
        self._ys: List[float] = []
        self._max = max_length
        self.line, = ax.plot([], [], "-", lw=0.9, color="steelblue", alpha=0.45)

    def track(self, x: float, y: float):
        self._xs.append(x)
        self._ys.append(y)
        if len(self._xs) > self._max:
            self._xs.pop(0)
            self._ys.pop(0)
        self.line.set_data(self._xs, self._ys)

    @property
    def artist(self):
        return self.line


class Animator(ABC):
    def __init__(self, solution: Any, t_eval: np.ndarray, view: AnimationView):
        self.solution = solution
        self.t_eval = t_eval
        self.view = view
        self.n_coords = len(solution.y) // 2
        self.n_points = self.n_coords // 2
        self.fig, self.ax = plt.subplots(figsize=view.figsize)
        self._setup_axes()
        self.trail = Trail(self.ax, view.trail_length)
        self.info_text = self.ax.text(
            0.02,
            0.97,
            "",
            transform=self.ax.transAxes,
            fontsize=8,
            va="top",
            family="monospace",
            color="#444",
        )
        self._artists: List[Any] = []
        self._setup_scene()

    def _setup_axes(self):
        r = self.view.view_radius
        self.ax.set_xlim(-r, r)
        self.ax.set_ylim(-r - 0.2, self.view.y_top)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.4)
        self.ax.set_title(self.view.title)
        self.ax.plot(0, 0, "ks", markersize=10, zorder=5)

    @abstractmethod
    def _setup_scene(self):
        pass

    @abstractmethod
    def _update_scene(self, frame: int, xs: List[float], ys: List[float]):
        pass

    @abstractmethod
    def _info_text(self, frame: int, xs: List[float], ys: List[float]) -> str:
        pass

    def _update(self, frame: int):
        state = self.solution.y[:, frame]
        xs, ys = positions_from_state(state, self.n_coords)
        self._update_scene(frame, xs, ys)
        self.trail.track(xs[-1], ys[-1])
        self.info_text.set_text(self._info_text(frame, xs, ys))
        return [*self._artists, self.trail.artist, self.info_text]

    def run(self, show: bool = True) -> animation.FuncAnimation:
        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=len(self.t_eval),
            interval=1000 / self.view.fps,
            blit=True,
        )
        if show:
            plt.show()
        return ani


class RodPendulumAnimator(Animator):
    def __init__(self, solution: Any, t_eval: np.ndarray, view: RodPendulumView):
        self.rod_view = view
        self._constraint_drift = self._precompute_constraint_drift(solution, view.rod_lengths)
        super().__init__(solution, t_eval, view)

    @staticmethod
    def _precompute_constraint_drift(
        solution: Any, rod_lengths: tuple[float, ...]
    ) -> np.ndarray:
        pos = solution.y[: len(solution.y) // 2]
        errors = []
        for i, length in enumerate(rod_lengths):
            x1, y1 = pos[i * 2], pos[i * 2 + 1]
            x2, y2 = pos[i * 2 + 2], pos[i * 2 + 3]
            errors.append(np.abs(np.hypot(x2 - x1, y2 - y1) - length))
        return np.max(errors, axis=0) if errors else np.zeros(pos.shape[1])

    def _setup_scene(self):
        colors = ["#534ab7", "#0f6e56"]
        self.rod_line, = self.ax.plot(
            [], [], "o-", lw=2.5, color="#888480", markersize=10, zorder=3
        )
        self.bob_dots = [
            self.ax.plot([], [], "o", markersize=12, color=colors[i % len(colors)], zorder=4)[0]
            for i in range(1, self.n_points)
        ]
        self._artists = [self.rod_line, *self.bob_dots]

    def _update_scene(self, frame: int, xs: List[float], ys: List[float]):
        self.rod_line.set_data(xs, ys)
        for i, dot in enumerate(self.bob_dots):
            dot.set_data([xs[i + 1]], [ys[i + 1]])

    def _info_text(self, frame: int, xs: List[float], ys: List[float]) -> str:
        lengths = segment_lengths(xs, ys)
        targets = self.rod_view.rod_lengths
        drift = self._constraint_drift[frame]

        if len(targets) == 1:
            return (
                f"L={lengths[0]:.3f} m  (target {targets[0]:.2f})\n"
                f"|C|_max = {drift:.2e} m"
            )
        labels = "  ".join(f"L{i + 1}={lengths[i]:.3f}" for i in range(len(lengths)))
        target_str = ", ".join(f"{t:.2f}" for t in targets)
        return f"{labels}  (targets {target_str})\n|C|_max = {drift:.2e} m"


class SpringPendulumAnimator(Animator):
    def __init__(
        self,
        system: System,
        solution: Any,
        t_eval: np.ndarray,
        view: SpringPendulumView,
    ):
        self.system = system
        self.spring_view = view
        self._energy_drift, self._e0 = self._precompute_energy_drift(system, solution, view)
        super().__init__(solution, t_eval, view)

    @staticmethod
    def _precompute_energy_drift(
        system: System, solution: Any, view: SpringPendulumView
    ) -> Tuple[np.ndarray, float]:
        n = len(system.coords)
        pos = solution.y[:n]
        vel = solution.y[n:]
        masses = np.array([m for entity in system.entities for m in entity.mass])
        ke = 0.5 * np.sum(masses[:, None] * vel**2, axis=0)
        pe_grav = (
            masses[1] * view.gravity * pos[1]
            + masses[3] * view.gravity * pos[3]
            + masses[5] * view.gravity * pos[5]
        )
        xs, ys = pos[0::2], pos[1::2]
        pe_spring = np.zeros(pos.shape[1])
        for i in range(len(xs) - 1):
            dist = np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            pe_spring += 0.5 * view.spring_k * (dist - view.rest_length) ** 2
        energy = ke + pe_grav + pe_spring
        e0 = float(energy[0])
        return np.abs(energy - e0), e0

    def _setup_scene(self):
        self.spring_lines = [
            self.ax.plot([], [], "-", lw=2.0, color=color, zorder=3)[0]
            for color in ("#3a86ff", "#ff6b35")
        ]
        self.bob_dots = [
            self.ax.plot([], [], "o", markersize=12, color=color, zorder=4)[0]
            for color in ("#3a86ff", "#ff6b35")
        ]
        self._artists = [*self.spring_lines, *self.bob_dots]

    def _update_scene(self, frame: int, xs: List[float], ys: List[float]):
        for i, spring in enumerate(self.spring_lines):
            spring.set_data(*coil_path(xs[i], ys[i], xs[i + 1], ys[i + 1]))
        for i, dot in enumerate(self.bob_dots):
            dot.set_data([xs[i + 1]], [ys[i + 1]])

    def _info_text(self, frame: int, xs: List[float], ys: List[float]) -> str:
        lengths = segment_lengths(xs, ys)
        length_str = "   ".join(f"L{i + 1}={lengths[i]:.3f} m" for i in range(len(lengths)))
        return (
            f"{length_str}   (rest {self.spring_view.rest_length:.2f})\n"
            f"ΔE = {self._energy_drift[frame]:.2e} J  (E₀={self._e0:.2f})"
        )


def animate_spring_pendulum(
    system: System,
    solution: Any,
    t_eval: np.ndarray,
    view: SpringPendulumView,
    show: bool = True,
) -> animation.FuncAnimation:
    return SpringPendulumAnimator(system, solution, t_eval, view).run(show)


def animate_rod_pendulum(
    solution: Any,
    t_eval: np.ndarray,
    view: RodPendulumView,
    show: bool = True,
) -> animation.FuncAnimation:
    return RodPendulumAnimator(solution, t_eval, view).run(show)
