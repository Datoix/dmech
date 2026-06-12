from dataclasses import dataclass

from dmech import ConstantForce, Entity, GearRatio, System
from dmech import models
from dmech.graphics import Animator, GearView
from dmech.integrator import integrate_system


@dataclass
class GearPairConfig:
    system: System
    gear_centers: tuple[tuple[float, float], ...]
    gear_radii: tuple[float, ...]
    angle_indices: tuple[int, ...]
    view_radius: float
    title: str = "Meshed Gear Pair"


def build() -> GearPairConfig:
    r1, r2 = 0.6, 0.4
    separation = r1 + r2
    cx_a, cx_b = -separation / 2, separation / 2

    gear_a = Entity(mass=[0.8])
    gear_b = Entity(mass=[0.5])
    gear_a[0] = 0.0
    gear_b[0] = 0.0

    system = System()
    system.add_entity(gear_a)
    system.add_entity(gear_b)

    mesh = GearRatio(r1, r2)
    mesh.mesh(gear_a, gear_b)
    system.add_constraint(mesh)

    system.add_force(ConstantForce(gear_a, 0, 2.0))
    system.initialize()

    return GearPairConfig(
        system=system,
        gear_centers=((cx_a, 0.0), (cx_b, 0.0)),
        gear_radii=(r1, r2),
        angle_indices=(0, 1),
        view_radius=separation * 1.4,
    )


def run():
    config = build()
    model = models.GearModel.from_config(config)
    solution, t_eval = integrate_system(config.system, fps=model.fps, t_max=6.0)
    print("Calculation complete! Playing animation...")
    Animator(t_eval, GearView(model, solution, config.system)).run()
