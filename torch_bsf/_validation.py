"""Private input validation shared by public APIs and numerical kernels."""


def _choice(value: object, name: str, options: tuple[str, ...], *, message: str | None = None) -> None:
    """Require a string choice without evaluating array-like equality results."""
    if not isinstance(value, str) or value not in options:
        raise ValueError(message if message is not None else f"Unknown {name} {value!r}.")
