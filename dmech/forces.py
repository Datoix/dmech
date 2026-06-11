from abc import ABC, abstractmethod
from typing import List

import jax.numpy as jnp

from dmech.entities import Entity


class Force(ABC):
    """Base class for forces. evaluate() receives the full coordinate vector."""

    @abstractmethod
    def get_indices(self) -> List[int]:
        """Returns the global DOF indices this force writes into."""
        pass

    @abstractmethod
    def evaluate(self, q: jnp.ndarray) -> jnp.ndarray:
        """Returns force values aligned with get_indices()."""
        pass


class ConstantForce(Force):
    """A simple, coordinate-independent force (e.g. gravity)."""

    def __init__(self, entity: Entity, coord: int, value: float):
        self.entity = entity
        self.coord = coord
        self.value = value

    def get_indices(self) -> List[int]:
        return [self.entity.index + self.coord]

    def evaluate(self, q: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([self.value])


class SpringForce(Force):
    """
    Hooke's law spring between two 2-D entities.

    Computes the restoring force on both endpoints and writes it into
    the four DOFs [x1, y1, x2, y2].  Damping (velocity-level) is NOT
    included here because velocities aren't threaded into evaluate() —
    if you want damping, add a DampedSpringForce that takes q_dot too,
    or handle it via a separate dissipation pass.
    """

    def __init__(
        self,
        entity_a: Entity,
        entity_b: Entity,
        rest_length: float,
        stiffness: float,
    ):
        self.entity_a = entity_a
        self.entity_b = entity_b
        self.rest_length = rest_length
        self.stiffness = stiffness

    def get_indices(self) -> List[int]:
        ia = self.entity_a.index
        ib = self.entity_b.index
        return [ia, ia + 1, ib, ib + 1]

    def evaluate(self, q: jnp.ndarray) -> jnp.ndarray:
        ia = self.entity_a.index
        ib = self.entity_b.index

        xa, ya = q[ia], q[ia + 1]
        xb, yb = q[ib], q[ib + 1]

        dx = xb - xa
        dy = yb - ya
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-8)

        ux = dx / dist
        uy = dy / dist

        extension = dist - self.rest_length
        magnitude = self.stiffness * extension

        fa_x = magnitude * ux
        fa_y = magnitude * uy
        fb_x = -magnitude * ux
        fb_y = -magnitude * uy

        return jnp.array([fa_x, fa_y, fb_x, fb_y])
