from abc import ABC, abstractmethod
from typing import Iterator, List

import jax.numpy as jnp

from dmech.entities import Entity


class Constraint(ABC):
    def __init__(self):
        self.entities: List[Entity] = []
        self.coords_map: jnp.ndarray = None
        self._cached_indices: jnp.ndarray = None

    @abstractmethod
    def evaluate(self, coords: jnp.ndarray) -> jnp.ndarray:
        pass

    def add_entity(self, entity: Entity):
        self.entities.append(entity)

    def map_coords(self, coords_map: jnp.ndarray):
        self.coords_map = coords_map

    def get_indices(self) -> Iterator[int]:
        for mapped_coord in self.coords_map:
            entity = self.entities[mapped_coord[0]]
            coord = mapped_coord[1]
            yield entity.index + coord

    def cache_indices(self):
        self._cached_indices = jnp.array(list(self.get_indices()), dtype=jnp.int32)


class Fixed(Constraint):
    def __init__(self, target_coords: List[float]):
        super().__init__()
        self.target = jnp.array(target_coords, dtype=jnp.float32)

    def evaluate(self, coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.atleast_1d(coords - self.target)


class Distance(Constraint):
    """Keep two 2-D points separated by a fixed length (rigid rod)."""

    def __init__(self, length: float):
        super().__init__()
        self.length = length

    def link(self, entity_a: Entity, entity_b: Entity):
        self.add_entity(entity_a)
        self.add_entity(entity_b)
        self.map_coords(jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

    def evaluate(self, coords: jnp.ndarray) -> jnp.ndarray:
        x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
        dist = jnp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-8)
        return jnp.atleast_1d(dist - self.length)


class GearRatio(Constraint):
    """External gear mesh: r1*theta1 + r2*theta2 = phase."""

    def __init__(self, r1: float, r2: float, phase: float = 0.0):
        super().__init__()
        self.r1 = jnp.float32(r1)
        self.r2 = jnp.float32(r2)
        self.phase = jnp.float32(phase)

    def mesh(self, gear_a: Entity, gear_b: Entity):
        self.add_entity(gear_a)
        self.add_entity(gear_b)
        self.map_coords(jnp.array([[0, 0], [1, 0]]))

    def evaluate(self, coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.atleast_1d(self.r1 * coords[0] + self.r2 * coords[1] - self.phase)


class RackPinion(Constraint):
    """Rolling without slip: x - r*theta = phase."""

    def __init__(self, radius: float, phase: float = 0.0):
        super().__init__()
        self.radius = jnp.float32(radius)
        self.phase = jnp.float32(phase)

    def connect(self, rack: Entity, pinion: Entity):
        self.add_entity(rack)
        self.add_entity(pinion)
        self.map_coords(jnp.array([[0, 0], [1, 0]]))

    def evaluate(self, coords: jnp.ndarray) -> jnp.ndarray:
        return jnp.atleast_1d(coords[0] - self.radius * coords[1] - self.phase)
