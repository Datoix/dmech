# dmech

2D multibody dynamics with holonomic constraints. Gravity and spring forces are handled as generalized forces. Constraints are solved each step using a KKT system. Integration uses scipy RK45, animation uses matplotlib.

## Run

```bash
pip install jax scipy numpy matplotlib
python main.py
```

Default demo: spring double pendulum (`dmech/examples/spring_pendulum.py`).

## Equations

- State: `q` (coords), `q_dot` (velocities), diagonal mass matrix `M`
- Gravity (y-direction on mass i): `F_gi = -m_i * g`
- Spring (between a, b): `F_a = k (|b-a| - L0) * (b-a)/|b-a|`, `F_b = -F_a`
- Constraints: `C(q) = 0`, e.g., fixed pivots

At each step:
```
[ M  -J.T ] [ q_ddot   ]   [ Q ]
[ J   0   ] [ lambda   ] = [ -J_dot*q_dot - ks*C - kd*C_dot ]
```
where
- `J` = constraint Jacobian (`∂C/∂q`)
- `Q` = generalized forces
- `ks`, `kd` = stabilization gains

## Layout

```
dmech/
  entities.py      point masses / DOFs
  forces.py        gravity, springs, ...
  constraints.py   fixed joints, ...
  system.py        KKT engine
  integrator.py    scipy time stepping
  graphics.py      animation
  examples/        model setups
main.py            entry point
```

## Note

This repo was built with AI assistance (Cursor). Physics and architecture are mine; the agent handled modularization, boilerplate, and docs.
