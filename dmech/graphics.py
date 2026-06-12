from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple, Union

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from dmech.system import System

View = Union["ViewRod", "ViewSpring", "ViewGear"]


def coil_path(ax_: float, ay_: float, bx_: float, by_: float) -> Tuple[np.ndarray, np.ndarray]:
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


def gear_outline(
    cx: float, cy: float, radius: float, angle: float, n_teeth: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, n_teeth * 2 + 1) + angle
    radii = np.array([
        radius * (1.12 if i % 2 else 1.0) for i in range(len(angles))
    ])
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    return xs, ys


def rack_outline(
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


def positions(state: np.ndarray, n_coords: int) -> Tuple[List[float], List[float]]:
    xs = [float(state[i]) for i in range(0, n_coords, 2)]
    ys = [float(state[i]) for i in range(1, n_coords, 2)]
    return xs, ys


def segment_lengths(xs: Sequence[float], ys: Sequence[float]) -> List[float]:
    return [float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])) for i in range(len(xs) - 1)]


@dataclass
class ViewBase:
    title: str
    view_radius: float
    trail_length: int = 150
    fps: int = 60
    y_top: float = 0.5
    figsize: Tuple[float, float] = (6, 6)


@dataclass
class ViewRod(ViewBase):
    rod_lengths: tuple[float, ...] = ()
    gravity: float = 9.81
    bob_colors: tuple[str, ...] = ("#534ab7", "#0f6e56")
    rod_color: str = "#888480"

    @classmethod
    def from_config(cls, config) -> "ViewRod":
        lengths = (config.rod_length,) if hasattr(config, "rod_length") else config.rod_lengths
        return cls(
            title=config.title,
            view_radius=config.view_radius,
            rod_lengths=lengths,
            gravity=config.gravity,
        )


@dataclass
class ViewSpring(ViewBase):
    spring_k: float = 40.0
    rest_length: float = 1.0
    gravity: float = 9.81
    bob_colors: tuple[str, ...] = ("#3a86ff", "#ff6b35")
    y_top: float = 0.6
    figsize: Tuple[float, float] = (6, 7)

    @classmethod
    def from_config(cls, config) -> "ViewSpring":
        return cls(
            title=f"Spring Double Pendulum via KKT  (k={config.spring_k} N/m, L₀={config.rest_length} m)",
            view_radius=config.view_radius,
            spring_k=config.spring_k,
            rest_length=config.rest_length,
            gravity=config.gravity,
        )


@dataclass
class RackViewSpec:
    y: float
    length: float
    height: float
    x_index: int


@dataclass
class ViewGear(ViewBase):
    gear_centers: tuple[tuple[float, float], ...] = ()
    gear_radii: tuple[float, ...] = ()
    angle_indices: tuple[int, ...] = ()
    rack: RackViewSpec | None = None
    gear_colors: tuple[str, ...] = ("#534ab7", "#0f6e56", "#c45c26")
    rack_color: str = "#888480"
    y_top: float = 1.2
    figsize: Tuple[float, float] = (7, 5)

    @classmethod
    def from_config(cls, config) -> "ViewGear":
        return cls(
            title=config.title,
            view_radius=config.view_radius,
            gear_centers=config.gear_centers,
            gear_radii=config.gear_radii,
            angle_indices=config.angle_indices,
            rack=getattr(config, "rack", None),
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


class Animator:
    def __init__(self, solution: Any, t_eval: np.ndarray, view: View, system: System | None = None):
        self.solution = solution
        self.t_eval = t_eval
        self.view = view
        self.system = system
        self.n_coords = len(solution.y) // 2
        self.is_gear = isinstance(view, ViewGear)
        self.n_points = self.n_coords // 2 if not self.is_gear else 0
        self._precompute_overlay()

        self.fig, self.ax = plt.subplots(figsize=view.figsize)
        r = view.view_radius
        if self.is_gear:
            self.ax.set_xlim(-r, r)
            self.ax.set_ylim(-r, r)
        else:
            self.ax.set_xlim(-r, r)
            self.ax.set_ylim(-r - 0.2, view.y_top)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.4)
        self.ax.set_title(view.title)
        if not self.is_gear:
            self.ax.plot(0, 0, "ks", markersize=10, zorder=5)

        self.trail = None if self.is_gear else Trail(self.ax, view.trail_length)
        self.info_text = self.ax.text(
            0.02, 0.97, "", transform=self.ax.transAxes,
            fontsize=8, va="top", family="monospace", color="#444",
        )
        self._artists = self._make_artists()

    def _precompute_overlay(self):
        if isinstance(self.view, ViewSpring):
            self._energy_drift, self._e0 = self._energy_drift_series()
        elif isinstance(self.view, ViewGear):
            self._constraint_drift = self._gear_constraint_drift_series()
        else:
            self._constraint_drift = self._constraint_drift_series()

    def _gear_constraint_drift_series(self) -> np.ndarray:
        assert self.system is not None
        n = self.n_coords
        drifts = []
        for frame in range(self.solution.y.shape[1]):
            q = jnp.array(self.solution.y[:n, frame], dtype=jnp.float32)
            c = self.system._global_constraints(q)
            drifts.append(float(np.max(np.abs(np.array(c)))))
        return np.array(drifts)

    def _constraint_drift_series(self) -> np.ndarray:
        view = self.view
        assert isinstance(view, ViewRod)
        pos = self.solution.y[: self.n_coords]
        errors = []
        for i, length in enumerate(view.rod_lengths):
            x1, y1 = pos[i * 2], pos[i * 2 + 1]
            x2, y2 = pos[i * 2 + 2], pos[i * 2 + 3]
            errors.append(np.abs(np.hypot(x2 - x1, y2 - y1) - length))
        return np.max(errors, axis=0)

    def _energy_drift_series(self) -> Tuple[np.ndarray, float]:
        view = self.view
        assert isinstance(view, ViewSpring)
        n = self.n_coords
        pos = self.solution.y[:n]
        vel = self.solution.y[n:]
        masses = np.array([m for entity in self.system.entities for m in entity.mass])
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
        return np.abs(energy - energy[0]), float(energy[0])

    def _make_artists(self) -> List[Any]:
        view = self.view
        artists: List[Any] = []

        if isinstance(view, ViewGear):
            self.gear_artists = [
                self.ax.plot([], [], "-", lw=1.5, color=view.gear_colors[i % len(view.gear_colors)], zorder=3)[0]
                for i in range(len(view.gear_centers))
            ]
            self.center_markers = [
                self.ax.plot([], [], "+", markersize=8, color="#333", zorder=4)[0]
                for _ in view.gear_centers
            ]
            self.rack_artist = None
            if view.rack is not None:
                self.rack_artist = self.ax.plot([], [], "-", lw=2.0, color=view.rack_color, zorder=2)[0]
            artists.extend(self.gear_artists)
            artists.extend(self.center_markers)
            if self.rack_artist is not None:
                artists.append(self.rack_artist)
            return artists

        if isinstance(view, ViewSpring):
            self.links = [
                self.ax.plot([], [], "-", lw=2.0, color=c, zorder=3)[0]
                for c in view.bob_colors
            ]
        else:
            self.links = [
                self.ax.plot([], [], "o-", lw=2.5, color=view.rod_color, markersize=10, zorder=3)[0]
            ]

        self.bobs = [
            self.ax.plot([], [], "o", markersize=12, color=view.bob_colors[i % len(view.bob_colors)], zorder=4)[0]
            for i in range(self.n_points - 1)
        ]
        artists.extend(self.links)
        artists.extend(self.bobs)
        return artists

    def _update_gear_frame(self, state: np.ndarray):
        view = self.view
        assert isinstance(view, ViewGear)
        for i, (cx, cy) in enumerate(view.gear_centers):
            angle = float(state[view.angle_indices[i]])
            self.gear_artists[i].set_data(
                *gear_outline(cx, cy, view.gear_radii[i], angle),
            )
            self.center_markers[i].set_data([cx], [cy])
        if view.rack is not None and self.rack_artist is not None:
            rack_x = float(state[view.rack.x_index])
            self.rack_artist.set_data(
                *rack_outline(rack_x, view.rack.y, view.rack.length, view.rack.height),
            )

    def _update_links(self, xs: List[float], ys: List[float]):
        if isinstance(self.view, ViewSpring):
            for i, link in enumerate(self.links):
                link.set_data(*coil_path(xs[i], ys[i], xs[i + 1], ys[i + 1]))
        else:
            self.links[0].set_data(xs, ys)

    def _gear_overlay_text(self, frame: int, state: np.ndarray) -> str:
        view = self.view
        assert isinstance(view, ViewGear)
        drift = self._constraint_drift[frame]
        lines = [f"|C|_max = {drift:.2e}"]

        n = self.n_coords
        vel = self.solution.y[n:, frame] if frame < self.solution.y.shape[1] else self.solution.y[n:, -1]
        if len(view.angle_indices) >= 2:
            w1 = float(vel[view.angle_indices[0]])
            w2 = float(vel[view.angle_indices[1]])
            if abs(w1) > 1e-6:
                lines.append(f"ω₂/ω₁ = {w2 / w1:.3f}")
        elif len(view.angle_indices) == 1 and view.rack is not None:
            x_dot = float(vel[view.rack.x_index])
            theta_dot = float(vel[view.angle_indices[0]])
            lines.append(f"ẋ = {x_dot:.3f}   θ̇ = {theta_dot:.3f}")

        return "\n".join(lines)

    def _overlay_text(self, frame: int, lengths: List[float]) -> str:
        view = self.view
        if isinstance(view, ViewSpring):
            labels = "   ".join(f"L{i + 1}={lengths[i]:.3f} m" for i in range(len(lengths)))
            return (
                f"{labels}   (rest {view.rest_length:.2f})\n"
                f"ΔE = {self._energy_drift[frame]:.2e} J  (E₀={self._e0:.2f})"
            )

        drift = self._constraint_drift[frame]
        if len(view.rod_lengths) == 1:
            return f"L={lengths[0]:.3f} m  (target {view.rod_lengths[0]:.2f})\n|C|_max = {drift:.2e} m"
        labels = "  ".join(f"L{i + 1}={lengths[i]:.3f}" for i in range(len(lengths)))
        targets = ", ".join(f"{t:.2f}" for t in view.rod_lengths)
        return f"{labels}  (targets {targets})\n|C|_max = {drift:.2e} m"

    def _tick(self, frame: int):
        state = self.solution.y[:, frame]
        if self.is_gear:
            self._update_gear_frame(state)
            self.info_text.set_text(self._gear_overlay_text(frame, state))
            return [*self._artists, self.info_text]

        xs, ys = positions(state, self.n_coords)
        self._update_links(xs, ys)
        for i, bob in enumerate(self.bobs):
            bob.set_data([xs[i + 1]], [ys[i + 1]])
        self.trail.track(xs[-1], ys[-1])
        self.info_text.set_text(self._overlay_text(frame, segment_lengths(xs, ys)))
        return [*self._artists, self.trail.line, self.info_text]

    def run(self, show: bool = True) -> animation.FuncAnimation:
        ani = animation.FuncAnimation(
            self.fig, self._tick, frames=len(self.t_eval),
            interval=1000 / self.view.fps, blit=True,
        )
        if show:
            plt.show()
        return ani
