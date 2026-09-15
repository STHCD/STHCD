from __future__ import annotations

from typing import Mapping


def resolve_step_value(
    step_index: int,
    schedule: Mapping[int, int],
    default_value: int,
) -> int:
    """Select the largest threshold at or below the step, regardless of key order.

    Before the first threshold, use its value. Empty schedules use the default.
    """
    if not schedule:
        return int(default_value)

    threshold = max(
        (threshold for threshold in schedule if threshold <= step_index),
        default=min(schedule),
    )
    return int(schedule[threshold])
