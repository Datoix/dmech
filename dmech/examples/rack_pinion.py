from dataclasses import dataclass

from dmech import ConstantForce, Entity, RackPinion, System
from dmech import models
from dmech.graphics import Animator, RackPinionView
from dmech.integrator import integrate_system


@dataclass
class RackPinionConfig:
    system: System
    pinion_center: tuple[float, float]
    pinion_radius: float
    angle_index: int
    rack: models.RackSpec
    view_radius: float
    title: str = "Rack and Pinion"


def build() -> RackPinionConfig:
    radius = 0.5
    rack_mass = 2.0
    pinion_inertia = 0.3

    rack = Entity(mass=[rack_mass])
    pinion = Entity(mass=[pinion_inertia])
    rack[0] = 0.0
    pinion[0] = 0.0

    system = System()
    system.add_entity(rack)
    system.add_entity(pinion)

    coupling = RackPinion(radius)
    coupling.connect(rack, pinion)
    system.add_constraint(coupling)

    system.add_force(ConstantForce(rack, 0, 4.0))
    system.initialize()

    return RackPinionConfig(
        system=system,
        pinion_center=(0.0, radius),
        pinion_radius=radius,
        angle_index=1,
        rack=models.RackSpec(y=0.0, length=3.0, height=0.15, x_index=0),
        view_radius=2.0,
    )


def run():
    config = build()
    model = models.RackPinionModel.from_config(config)
    solution, t_eval = integrate_system(config.system, fps=model.fps, t_max=6.0)
    print("Calculation complete! Playing animation...")
    Animator(t_eval, RackPinionView(model, solution, config.system)).run()
