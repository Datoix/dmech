import math
from dataclasses import dataclass

import jax.numpy as jnp

from dmech import ConstantForce, Distance, Entity, Fixed, System


@dataclass
class DoublePendulumConfig:
    system: System
    rod_lengths: tuple[float, float]
    gravity: float
    view_radius: float
    title: str = "Double Pendulum"


def build_double_pendulum() -> DoublePendulumConfig:
    """Two bobs connected by rigid rods (matches _bkp/example.py setup)."""
    l1, l2 = 1.0, 0.9
    m1, m2 = 1.5, 1.0
    gravity = 9.81
    theta1 = math.pi / 2 + 0.4
    theta2 = math.pi / 2 + 1.2

    pivot = Entity(mass=[1.0, 1.0])
    bob1 = Entity(mass=[m1, m1])
    bob2 = Entity(mass=[m2, m2])

    pivot[0], pivot[1] = 0.0, 0.0
    bob1[0] = l1 * math.cos(theta1)
    bob1[1] = l1 * math.sin(theta1)
    bob2[0] = bob1[0] + l2 * math.cos(theta2)
    bob2[1] = bob1[1] + l2 * math.sin(theta2)

    system = System()
    system.add_entity(pivot)
    system.add_entity(bob1)
    system.add_entity(bob2)

    pin = Fixed([0.0, 0.0])
    pin.add_entity(pivot)
    pin.map_coords(jnp.array([[0, 0], [0, 1]]))
    system.add_constraint(pin)

    rod1 = Distance(l1)
    rod1.link(pivot, bob1)
    system.add_constraint(rod1)

    rod2 = Distance(l2)
    rod2.link(bob1, bob2)
    system.add_constraint(rod2)

    for bob in (bob1, bob2):
        system.add_force(ConstantForce(bob, 1, -gravity * bob.mass[0]))

    system.initialize()

    return DoublePendulumConfig(
        system=system,
        rod_lengths=(l1, l2),
        gravity=gravity,
        view_radius=max(l1 + l2, l1) * 1.2,
    )
