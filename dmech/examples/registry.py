from dataclasses import dataclass
from typing import Callable


@dataclass
class Example:
    name: str
    run: Callable[[], None]
    aliases: tuple[str, ...] = ()
    default: bool = False


class ExampleRegistry:
    def __init__(self, examples: list[Example]):
        self._examples = examples
        self._by_key: dict[str, Example] = {}
        self._default: str | None = None

        for example in examples:
            for key in (example.name, *example.aliases):
                if key in self._by_key:
                    raise ValueError(f"duplicate example key: {key}")
                self._by_key[key] = example
            if example.default:
                if self._default is not None:
                    raise ValueError("only one example can be default")
                self._default = example.name

        if self._default is None and examples:
            self._default = examples[0].name

    def names(self) -> list[str]:
        return sorted(self._by_key)

    def get(self, key: str) -> Example:
        return self._by_key[key]

    @property
    def default_name(self) -> str:
        return self._default
