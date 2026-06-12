# dmech

2D multibody dynamics with holonomic constraints. Gravity and spring forces are handled as generalized forces. Constraints are solved each step using a KKT system. Integration uses scipy RK45, animation uses matplotlib.

## Run

First, set up a virtual environment:

<details>
<summary><strong>Windows</strong></summary>

```bat
python -m venv .venv
.venv\Scripts\activate
pip install jax scipy numpy matplotlib
```
</details>

<details>
<summary><strong>Linux/macOS</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jax scipy numpy matplotlib
```
</details>

Then run:

```bash
python main.py              # spring double pendulum (default)
python main.py spring_double
python main.py pendulum     # single rigid rod
python main.py double       # rigid double pendulum
python main.py gear         # meshed gear pair
python main.py gear_train   # three-gear train
python main.py rack         # rack and pinion
```

Examples live in `dmech/examples/`.

## Equations

- State: `q` (coords), `q_dot` (velocities), diagonal mass matrix `M`
- Gravity (y-direction on mass i): `F_gi = -m_i * g`
- Spring (between a, b): `F_a = k (|b-a| - L0) * (b-a)/|b-a|`, `F_b = -F_a`
- Rigid rod (between a, b): `|b-a| - L = 0`
- External gear mesh: `r₁θ₁ + r₂θ₂ = φ₀` (opposite rotation at contact)
- Rack and pinion: `x − rθ = x₀` (rolling without slip)
- Applied torque (on angle DOF): `Q_θ = τ`
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