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
