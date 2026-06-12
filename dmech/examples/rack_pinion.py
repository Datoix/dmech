from dataclasses import dataclass

from dmech import ConstantForce, Entity, RackPinion, System
from dmech.graphics import Animator, RackViewSpec, ViewGear
from dmech.integrator import integrate_system


@dataclass
class RackPinionConfig:
    system: System
    gear_centers: tuple[tuple[float, float], ...]
    gear_radii: tuple[float, ...]
    angle_indices: tuple[int, ...]
    rack: RackViewSpec
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
        gear_centers=((0.0, radius),),
        gear_radii=(radius,),
        angle_indices=(1,),
        rack=RackViewSpec(y=0.0, length=3.0, height=0.15, x_index=0),
        view_radius=2.0,
    )


def run():
    config = build()
    view = ViewGear.from_config(config)
    solution, t_eval = integrate_system(config.system, fps=view.fps, t_max=6.0)
    print("Calculation complete! Playing animation...")
    Animator(solution, t_eval, view, system=config.system).run()
