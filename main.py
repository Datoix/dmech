import argparse

from dmech.examples.double_pendulum import build_double_pendulum
from dmech.examples.pendulum import build_pendulum
from dmech.examples.spring_pendulum import build_spring_pendulum
from dmech.graphics import (
    RodPendulumView,
    SpringPendulumView,
    animate_rod_pendulum,
    animate_spring_pendulum,
)
from dmech.integrator import integrate_system

DEMOS = {
    "spring": "spring",
    "spring_double": "spring",
    "pendulum": "pendulum",
    "double": "double",
}


def main():
    parser = argparse.ArgumentParser(description="dmech demo runner")
    parser.add_argument(
        "demo",
        nargs="?",
        default="spring",
        choices=DEMOS,
        help="which model to run (default: spring)",
    )
    args = parser.parse_args()

    demo = DEMOS[args.demo]
    print(f"Pre-calculating {demo} trajectory with Scipy (RK45)...")

    if demo == "spring":
        config = build_spring_pendulum()
        view = SpringPendulumView.from_config(config)
        solution, t_eval = integrate_system(config.system, fps=view.fps)
        print("Calculation complete! Playing animation...")
        animate_spring_pendulum(config.system, solution, t_eval, view)
    elif demo == "pendulum":
        config = build_pendulum()
        view = RodPendulumView.from_pendulum(config)
        solution, t_eval = integrate_system(config.system, fps=view.fps)
        print("Calculation complete! Playing animation...")
        animate_rod_pendulum(solution, t_eval, view)
    else:
        config = build_double_pendulum()
        view = RodPendulumView.from_double(config)
        solution, t_eval = integrate_system(config.system, fps=view.fps)
        print("Calculation complete! Playing animation...")
        animate_rod_pendulum(solution, t_eval, view)


if __name__ == "__main__":
    main()
