"""Safe display labels for model outputs with uncertain class semantics."""

from __future__ import annotations

from collections.abc import Sequence


def display_class_name(
    class_index: int,
    class_names: Sequence[str],
    verified: bool,
) -> str:
    """Return a semantic label only when its output-index mapping is verified."""
    if isinstance(class_index, bool) or not isinstance(class_index, int):
        raise TypeError("class_index must be an integer")
    if class_index < 0 or class_index >= len(class_names):
        raise IndexError("class_index is outside the available output range")
    if verified:
        return class_names[class_index]
    return f"Output {class_index}"
