from typing import List

import jax.numpy as jnp


class Entity:
    def __init__(self, mass: List[float]):
        self.mass = mass
        self.initial_coords = jnp.zeros(len(mass), dtype=jnp.float32)
        self.index: int = None

    def __getitem__(self, key: int) -> float:
        return self.initial_coords[key]

    def __setitem__(self, key: int, value: float):
        self.initial_coords = self.initial_coords.at[key].set(value)
