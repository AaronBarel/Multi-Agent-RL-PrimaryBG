"""Reward functions (spec §4.5)."""

from __future__ import annotations

from collections.abc import Sequence

from primary.data.board import TOTAL_VOTES
from primary.engine.actions import GameOver


def terminal_rewards(result: GameOver, n: int) -> list[float]:
    """+1 for an outright win; a k-way draw gives 1/k to each tied player."""
    share = 1.0 / len(result.winners)
    return [share if seat in result.winners else 0.0 for seat in range(n)]


def shaped_rewards(before: Sequence[int], after: Sequence[int]) -> list[float]:
    """Optional shaping: votes gained this step as a share of all votes in the game."""
    return [(a - b) / TOTAL_VOTES for a, b in zip(after, before, strict=True)]
