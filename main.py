import argparse

from dmech.examples import double_pendulum, pendulum, spring_pendulum
from dmech.examples.registry import Example, ExampleRegistry

REGISTRY = ExampleRegistry([
    Example("spring", spring_pendulum.run, aliases=("spring_double",), default=True),
    Example("pendulum", pendulum.run),
    Example("double", double_pendulum.run),
])


def main():
    parser = argparse.ArgumentParser(description="dmech example runner")
    parser.add_argument(
        "example",
        nargs="?",
        default=REGISTRY.default_name,
        choices=REGISTRY.names(),
        help=f"which example to run (default: {REGISTRY.default_name})",
    )
    args = parser.parse_args()

    example = REGISTRY.get(args.example)
    print(f"Pre-calculating {example.name} trajectory with Scipy (RK45)...")
    example.run()


if __name__ == "__main__":
    main()
