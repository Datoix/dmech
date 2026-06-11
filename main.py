from dmech.examples.spring_pendulum import build_spring_pendulum
from dmech.graphics import SpringPendulumView, animate_spring_pendulum
from dmech.integrator import integrate_system


def main():
    config = build_spring_pendulum()
    view = SpringPendulumView.from_config(config)

    print("Pre-calculating spring pendulum trajectory with Scipy (RK45)...")
    solution, t_eval = integrate_system(config.system, fps=view.fps)
    print("Calculation complete! Playing animation...")

    animate_spring_pendulum(config.system, solution, t_eval, view)


if __name__ == "__main__":
    main()
