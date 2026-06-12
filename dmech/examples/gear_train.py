from dataclasses import dataclass, field

import numpy as np

from dmech import Entity, GearRatio, System
from dmech import models
from dmech.graphics import Animator, GearView
from dmech.integrator import integrate_system


@dataclass
class GearTrainConfig:
    system: System
    gear_centers: tuple[tuple[float, float], ...]
    gear_radii: tuple[float, ...]
    angle_indices: tuple[int, ...]
    view_radius: float
    q_dot0: np.ndarray = field(repr=False)
    title: str = "Three-Gear Train"


def build() -> GearTrainConfig:
    r1, r2, r3 = 0.3, 0.4, 0.5
    cx_a = -(r1 + r2) - (r2 + r3) / 2 + r1
    cx_b = cx_a + r1 + r2
    cx_c = cx_b + r2 + r3

    gear_a = Entity(mass=[0.4])
    gear_b = Entity(mass=[0.6])
    gear_c = Entity(mass=[1.0])

    system = System()
    system.add_entity(gear_a)
    system.add_entity(gear_b)
    system.add_entity(gear_c)

    mesh_ab = GearRatio(r1, r2)
    mesh_ab.mesh(gear_a, gear_b)
    system.add_constraint(mesh_ab)

    mesh_bc = GearRatio(r2, r3)
    mesh_bc.mesh(gear_b, gear_c)
    system.add_constraint(mesh_bc)

    system.initialize()

    omega0 = 3.0
    q_dot0 = np.array([omega0, -r1 / r2 * omega0, r1 / r3 * omega0])

    return GearTrainConfig(
        system=system,
        gear_centers=((cx_a, 0.0), (cx_b, 0.0), (cx_c, 0.0)),
        gear_radii=(r1, r2, r3),
        angle_indices=(0, 1, 2),
        view_radius=max(abs(cx_a), abs(cx_c)) + r3 + 0.3,
        q_dot0=q_dot0,
    )


def run():
    config = build()
    model = models.GearModel.from_config(config)
    solution, t_eval = integrate_system(
        config.system, fps=model.fps, t_max=6.0, q_dot0=config.q_dot0,
    )
    print("Calculation complete! Playing animation...")
    Animator(t_eval, GearView(model, solution, config.system)).run()
