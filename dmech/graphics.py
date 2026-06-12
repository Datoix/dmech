from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Tuple

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from dmech import models
from dmech.system import System


def positions(state: np.ndarray, n_coords: int) -> Tuple[List[float], List[float]]:
    xs = [float(state[i]) for i in range(0, n_coords, 2)]
    ys = [float(state[i]) for i in range(1, n_coords, 2)]
    return xs, ys


def segment_lengths(xs: Sequence[float], ys: Sequence[float]) -> List[float]:
    return [float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])) for i in range(len(xs) - 1)]


def constraint_drift_series(system: System, solution: Any, n_coords: int) -> np.ndarray:
    drifts = []
    for frame in range(solution.y.shape[1]):
        q = jnp.array(solution.y[:n_coords, frame], dtype=jnp.float32)
        c = system._global_constraints(q)
        drifts.append(float(np.max(np.abs(np.array(c)))))
    return np.array(drifts)


def gear_outline(cx: float, cy: float, radius: float, angle: float, n_teeth: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, n_teeth * 2 + 1) + angle
    radii = np.array([
        radius * (1.12 if i % 2 else 1.0) for i in range(len(angles))
    ])
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    return xs, ys


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


class AnimationView(ABC):
    def __init__(self, model: models.ModelBase, solution: Any, system: System | None = None):
        self.model = model
        self.solution = solution
        self.system = system
        self.n_coords = len(solution.y) // 2
        self._info_text = None
        self._artists: List[Any] = []

    @property
    def fps(self) -> int:
        return self.model.fps

    @property
    def figsize(self) -> Tuple[float, float]:
        return self.model.figsize

    def _setup_axes(self, ax) -> None:
        r = self.model.view_radius
        ax.set_xlim(-r, r)
        ax.set_ylim(-r - 0.2, self.model.y_top)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.4)
        ax.set_title(self.model.title)
        ax.plot(0, 0, "ks", markersize=10, zorder=5)

    def _setup_info_text(self, ax) -> None:
        self._info_text = ax.text(
            0.02, 0.97, "", transform=ax.transAxes,
            fontsize=8, va="top", family="monospace", color="#444",
        )

    @abstractmethod
    def setup(self, ax) -> None:
        ...

    @abstractmethod
    def update(self, frame: int) -> List[Any]:
        ...

    def _blit_artists(self) -> List[Any]:
        return [*self._artists, self._info_text]


class RodView(AnimationView):
    def __init__(self, model: models.RodModel, solution: Any, system: System | None = None):
        super().__init__(model, solution, system)
        self._constraint_drift = self._constraint_drift_series()

    def _constraint_drift_series(self) -> np.ndarray:
        pos = self.solution.y[: self.n_coords]
        errors = []
        for i, length in enumerate(self.model.rod_lengths):
            x1, y1 = pos[i * 2], pos[i * 2 + 1]
            x2, y2 = pos[i * 2 + 2], pos[i * 2 + 3]
            errors.append(np.abs(np.hypot(x2 - x1, y2 - y1) - length))
        return np.max(errors, axis=0)

    def setup(self, ax) -> None:
        model = self.model
        self._setup_axes(ax)
        self._setup_info_text(ax)
        self.links = [
            ax.plot([], [], "o-", lw=2.5, color=model.rod_color, markersize=10, zorder=3)[0]
        ]
        n_points = self.n_coords // 2
        self.bobs = [
            ax.plot([], [], "o", markersize=12, color=model.bob_colors[i % len(model.bob_colors)], zorder=4)[0]
            for i in range(n_points - 1)
        ]
        self.trail = Trail(ax, model.trail_length)
        self._artists = [*self.links, *self.bobs]

    def update(self, frame: int) -> List[Any]:
        state = self.solution.y[:, frame]
        xs, ys = positions(state, self.n_coords)
        self.links[0].set_data(xs, ys)
        for i, bob in enumerate(self.bobs):
            bob.set_data([xs[i + 1]], [ys[i + 1]])
        self.trail.track(xs[-1], ys[-1])

        lengths = segment_lengths(xs, ys)
        model = self.model
        drift = self._constraint_drift[frame]
        if len(model.rod_lengths) == 1:
            self._info_text.set_text(
                f"L={lengths[0]:.3f} m  (target {model.rod_lengths[0]:.2f})\n|C|_max = {drift:.2e} m",
            )
        else:
            labels = "  ".join(f"L{i + 1}={lengths[i]:.3f}" for i in range(len(lengths)))
            targets = ", ".join(f"{t:.2f}" for t in model.rod_lengths)
            self._info_text.set_text(
                f"{labels}  (targets {targets})\n|C|_max = {drift:.2e} m",
            )
        return [*self._blit_artists(), self.trail.line]


class SpringView(AnimationView):
    def __init__(self, model: models.SpringModel, solution: Any, system: System):
        super().__init__(model, solution, system)
        self._energy_drift, self._e0 = self._energy_drift_series()

    @staticmethod
    def _coil_path(ax_: float, ay_: float, bx_: float, by_: float) -> Tuple[np.ndarray, np.ndarray]:
        tang = np.array([bx_ - ax_, by_ - ay_])
        leng = np.linalg.norm(tang) + 1e-10
        unit_t = tang / leng
        unit_n = np.array([-unit_t[1], unit_t[0]])
        pts = []
        for i, t in enumerate(np.linspace(0.0, 1.0, 22)):
            base = np.array([ax_, ay_]) + t * tang
            if 0 < i < 21:
                base = base + (1 if i % 2 else -1) * 0.09 * unit_n
            pts.append(base)
        pts = np.array(pts)
        return pts[:, 0], pts[:, 1]

    def _energy_drift_series(self) -> Tuple[np.ndarray, float]:
        model = self.model
        n = self.n_coords
        pos = self.solution.y[:n]
        vel = self.solution.y[n:]
        masses = np.array([m for entity in self.system.entities for m in entity.mass])
        ke = 0.5 * np.sum(masses[:, None] * vel**2, axis=0)
        pe_grav = (
            masses[1] * model.gravity * pos[1]
            + masses[3] * model.gravity * pos[3]
            + masses[5] * model.gravity * pos[5]
        )
        xs, ys = pos[0::2], pos[1::2]
        pe_spring = np.zeros(pos.shape[1])
        for i in range(len(xs) - 1):
            dist = np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            pe_spring += 0.5 * model.spring_k * (dist - model.rest_length) ** 2
        energy = ke + pe_grav + pe_spring
        return np.abs(energy - energy[0]), float(energy[0])

    def setup(self, ax) -> None:
        model = self.model
        self._setup_axes(ax)
        self._setup_info_text(ax)
        self.links = [
            ax.plot([], [], "-", lw=2.0, color=c, zorder=3)[0]
            for c in model.bob_colors
        ]
        n_points = self.n_coords // 2
        self.bobs = [
            ax.plot([], [], "o", markersize=12, color=model.bob_colors[i % len(model.bob_colors)], zorder=4)[0]
            for i in range(n_points - 1)
        ]
        self.trail = Trail(ax, model.trail_length)
        self._artists = [*self.links, *self.bobs]

    def update(self, frame: int) -> List[Any]:
        state = self.solution.y[:, frame]
        xs, ys = positions(state, self.n_coords)
        for i, link in enumerate(self.links):
            link.set_data(*self._coil_path(xs[i], ys[i], xs[i + 1], ys[i + 1]))
        for i, bob in enumerate(self.bobs):
            bob.set_data([xs[i + 1]], [ys[i + 1]])
        self.trail.track(xs[-1], ys[-1])

        model = self.model
        lengths = segment_lengths(xs, ys)
        labels = "   ".join(f"L{i + 1}={lengths[i]:.3f} m" for i in range(len(lengths)))
        self._info_text.set_text(
            f"{labels}   (rest {model.rest_length:.2f})\n"
            f"ΔE = {self._energy_drift[frame]:.2e} J  (E₀={self._e0:.2f})",
        )
        return [*self._blit_artists(), self.trail.line]


class GearView(AnimationView):
    def __init__(self, model: models.GearModel, solution: Any, system: System):
        super().__init__(model, solution, system)
        self._constraint_drift = constraint_drift_series(system, solution, self.n_coords)

    def setup(self, ax) -> None:
        model = self.model
        r = model.view_radius
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.4)
        ax.set_title(model.title)
        self._setup_info_text(ax)

        self.gear_artists = [
            ax.plot([], [], "-", lw=1.5, color=model.gear_colors[i % len(model.gear_colors)], zorder=3)[0]
            for i in range(len(model.gear_centers))
        ]
        self.center_markers = [
            ax.plot([], [], "+", markersize=8, color="#333", zorder=4)[0]
            for _ in model.gear_centers
        ]
        self._artists = [*self.gear_artists, *self.center_markers]

    def update(self, frame: int) -> List[Any]:
        model = self.model
        state = self.solution.y[:, frame]
        for i, (cx, cy) in enumerate(model.gear_centers):
            angle = float(state[model.angle_indices[i]])
            self.gear_artists[i].set_data(
                *gear_outline(cx, cy, model.gear_radii[i], angle),
            )
            self.center_markers[i].set_data([cx], [cy])

        drift = self._constraint_drift[frame]
        lines = [f"|C|_max = {drift:.2e}"]
        n = self.n_coords
        vel = self.solution.y[n:, frame] if frame < self.solution.y.shape[1] else self.solution.y[n:, -1]
        if len(model.angle_indices) >= 2:
            w1 = float(vel[model.angle_indices[0]])
            w2 = float(vel[model.angle_indices[1]])
            if abs(w1) > 1e-6:
                lines.append(f"ω₂/ω₁ = {w2 / w1:.3f}")
        self._info_text.set_text("\n".join(lines))
        return self._blit_artists()


class RackPinionView(AnimationView):
    def __init__(self, model: models.RackPinionModel, solution: Any, system: System):
        super().__init__(model, solution, system)
        self._constraint_drift = constraint_drift_series(system, solution, self.n_coords)

    @staticmethod
    def _rack_outline(
        x: float, y: float, length: float, height: float, n_teeth: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        half = length / 2.0
        tooth_w = length / n_teeth
        pts: List[Tuple[float, float]] = []
        x0 = x - half
        for i in range(n_teeth):
            xi = x0 + i * tooth_w
            pts.append((xi, y - height / 2))
            pts.append((xi + tooth_w * 0.5, y - height / 2))
            pts.append((xi + tooth_w * 0.5, y + height / 2))
            pts.append((xi + tooth_w, y + height / 2))
        pts.append((x0 + length, y - height / 2))
        arr = np.array(pts)
        return arr[:, 0], arr[:, 1]

    def setup(self, ax) -> None:
        model = self.model
        r = model.view_radius
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.4)
        ax.set_title(model.title)
        self._setup_info_text(ax)

        self.pinion_artist = ax.plot([], [], "-", lw=1.5, color=model.pinion_color, zorder=3)[0]
        self.pinion_center_marker = ax.plot([], [], "+", markersize=8, color="#333", zorder=4)[0]
        self.rack_artist = ax.plot([], [], "-", lw=2.0, color=model.rack_color, zorder=2)[0]
        self._artists = [self.pinion_artist, self.pinion_center_marker, self.rack_artist]

    def update(self, frame: int) -> List[Any]:
        model = self.model
        assert model.rack is not None
        state = self.solution.y[:, frame]
        cx, cy = model.pinion_center
        angle = float(state[model.angle_index])
        self.pinion_artist.set_data(*gear_outline(cx, cy, model.pinion_radius, angle))
        self.pinion_center_marker.set_data([cx], [cy])
        rack_x = float(state[model.rack.x_index])
        self.rack_artist.set_data(
            *self._rack_outline(rack_x, model.rack.y, model.rack.length, model.rack.height),
        )

        n = self.n_coords
        vel = self.solution.y[n:, frame] if frame < self.solution.y.shape[1] else self.solution.y[n:, -1]
        x_dot = float(vel[model.rack.x_index])
        theta_dot = float(vel[model.angle_index])
        self._info_text.set_text(
            f"|C|_max = {self._constraint_drift[frame]:.2e}\n"
            f"ẋ = {x_dot:.3f}   θ̇ = {theta_dot:.3f}",
        )
        return self._blit_artists()


class Animator:
    def __init__(self, t_eval: np.ndarray, view: AnimationView):
        self.t_eval = t_eval
        self.view = view
        self.fig, self.ax = plt.subplots(figsize=view.figsize)
        view.setup(self.ax)

    def _tick(self, frame: int):
        return self.view.update(frame)

    def run(self, show: bool = True) -> animation.FuncAnimation:
        ani = animation.FuncAnimation(
            self.fig, self._tick, frames=len(self.t_eval),
            interval=1000 / self.view.fps, blit=True,
        )
        if show:
            plt.show()
        return ani
