import math
from dataclasses import dataclass

import jax.numpy as jnp

from dmech import ConstantForce, Distance, Entity, Fixed, System
from dmech import models
from dmech.graphics import Animator, RodView
from dmech.integrator import integrate_system


@dataclass
class PendulumConfig:
    system: System
    rod_length: float
    gravity: float
    view_radius: float
    title: str = "Simple Pendulum"


def build() -> PendulumConfig:
    rod_length = 1.0
    bob_mass = 1.5
    gravity = 9.81
    theta0 = math.pi / 2 + 0.5

    pivot = Entity(mass=[1.0, 1.0])
    bob = Entity(mass=[bob_mass, bob_mass])

    pivot[0], pivot[1] = 0.0, 0.0
    bob[0] = rod_length * math.cos(theta0)
    bob[1] = rod_length * math.sin(theta0)

    system = System()
    system.add_entity(pivot)
    system.add_entity(bob)

    pin = Fixed([0.0, 0.0])
    pin.add_entity(pivot)
    pin.map_coords(jnp.array([[0, 0], [0, 1]]))
    system.add_constraint(pin)

    rod = Distance(rod_length)
    rod.link(pivot, bob)
    system.add_constraint(rod)

    system.add_force(ConstantForce(bob, 1, -gravity * bob.mass[0]))
    system.initialize()

    return PendulumConfig(
        system=system,
        rod_length=rod_length,
        gravity=gravity,
        view_radius=rod_length * 1.6,
    )


def run():
    config = build()
    model = models.RodModel.from_config(config)
    solution, t_eval = integrate_system(config.system, fps=model.fps)
    print("Calculation complete! Playing animation...")
    Animator(t_eval, RodView(model, solution)).run()
