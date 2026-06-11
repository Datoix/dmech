from dataclasses import dataclass

import jax.numpy as jnp

from dmech import ConstantForce, Entity, Fixed, SpringForce, System
from dmech.graphics import Animator, ViewSpring
from dmech.integrator import integrate_system


@dataclass
class SpringPendulumConfig:
    system: System
    spring_k: float
    rest_length: float
    gravity: float
    view_radius: float


def build() -> SpringPendulumConfig:
    spring_k = 40.0
    rest_length = 1.0
    bob_mass = 2.0
    gravity = 9.81
    l_eq = rest_length + bob_mass * gravity / spring_k

    pivot = Entity(mass=[1.0, 1.0])
    bob1 = Entity(mass=[bob_mass, bob_mass])
    bob2 = Entity(mass=[bob_mass, bob_mass])

    pivot[0], pivot[1] = 0.0, 0.0
    bob1[0], bob1[1] = 0.6, -l_eq
    bob2[0], bob2[1] = -0.4, -l_eq * 2.0

    system = System()
    system.add_entity(pivot)
    system.add_entity(bob1)
    system.add_entity(bob2)

    pin = Fixed([0.0, 0.0])
    pin.add_entity(pivot)
    pin.map_coords(jnp.array([[0, 0], [0, 1]]))
    system.add_constraint(pin)

    for entity in (pivot, bob1, bob2):
        system.add_force(ConstantForce(entity, 1, -gravity * entity.mass[0]))

    system.add_force(SpringForce(pivot, bob1, rest_length=rest_length, stiffness=spring_k))
    system.add_force(SpringForce(bob1, bob2, rest_length=rest_length, stiffness=spring_k))
    system.initialize()

    return SpringPendulumConfig(
        system=system,
        spring_k=spring_k,
        rest_length=rest_length,
        gravity=gravity,
        view_radius=l_eq * 2 + 0.8,
    )


def run():
    config = build()
    view = ViewSpring.from_config(config)
    solution, t_eval = integrate_system(config.system, fps=view.fps)
    print("Calculation complete! Playing animation...")
    Animator(solution, t_eval, view, system=config.system).run()
