from typing import List, Tuple

import jax
import jax.numpy as jnp

from dmech.constraints import Constraint
from dmech.entities import Entity
from dmech.forces import Force


class System:
    def __init__(self):
        self.entities: List[Entity] = []
        self.constraints: List[Constraint] = []
        self.forces: List[Force] = []
        self.coords: jnp.ndarray = None
        self.M: jnp.ndarray = None
        self._compiled_solve = None

    def add_entity(self, entity: Entity):
        self.entities.append(entity)

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)

    def add_force(self, force: Force):
        self.forces.append(force)

    def initialize(self):
        index = 0
        coords_builder = []
        mass_builder = []

        for entity in self.entities:
            entity.index = index
            index += len(entity.mass)
            coords_builder.extend([*entity.initial_coords])
            mass_builder.extend(entity.mass)

        self.coords = jnp.array(coords_builder, dtype=jnp.float32)
        self.M = jnp.diag(jnp.array(mass_builder, dtype=jnp.float32))

        for constraint in self.constraints:
            constraint.cache_indices()

        self._build_solver()

    def _global_constraints(self, q: jnp.ndarray) -> jnp.ndarray:
        vals = [c.evaluate(q[c._cached_indices]) for c in self.constraints]
        return jnp.concatenate(vals) if vals else jnp.array([])

    def _generalized_forces(self, q: jnp.ndarray) -> jnp.ndarray:
        forces_arr = jnp.zeros_like(q)
        for force in self.forces:
            indices = jnp.array(force.get_indices(), dtype=jnp.int32)
            values = force.evaluate(q)
            forces_arr = forces_arr.at[indices].add(values)
        return forces_arr

    def _build_solver(self):
        jacobian_fn = jax.jacobian(self._global_constraints)

        @jax.jit
        def solve_step(q: jnp.ndarray, q_dot: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            M = self.M
            Q = self._generalized_forces(q)

            ks = 150.0
            kd = 20.0

            C = self._global_constraints(q)
            J, J_dot = jax.jvp(jacobian_fn, (q,), (q_dot,))
            C_dot = J @ q_dot

            n, m = M.shape[0], J.shape[0]

            top_row = jnp.block([M, -J.T])
            bottom_row = jnp.block([J, jnp.zeros((m, m))])
            KKT_matrix = jnp.vstack([top_row, bottom_row])

            rhs_top = Q
            rhs_bottom = (-J_dot @ q_dot) - (ks * C) - (kd * C_dot)
            KKT_rhs = jnp.concatenate([rhs_top, rhs_bottom])

            x = jax.scipy.linalg.solve(KKT_matrix, KKT_rhs)
            return x[:n], x[n:]

        self._compiled_solve = solve_step

    def solve(self, q: jnp.ndarray, q_dot: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._compiled_solve(q, q_dot)
