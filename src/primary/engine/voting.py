"""Vote awards for a region and end-of-game winner selection (spec §3 rules 17–20)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class Placement(StrEnum):
    FIRST = "won_1st"
    SECOND = "won_2nd"
    LOST = "lost"
    ABSENT = "not_participated"


@dataclass(frozen=True)
class RegionResult:
    votes: tuple[int, ...]
    placements: tuple[Placement, ...]


def _split(
    votes: list[int], placements: list[Placement], seats: list[int], total: int, place: Placement
) -> None:
    share = total // len(seats)
    for seat in seats:
        votes[seat] = share
        placements[seat] = place


def compute_awards(influence: Sequence[int], awards: tuple[int, ...]) -> RegionResult:
    """Votes per seat for one region.

    ``awards`` is ``(first,)`` for winner-take-all or ``(first, second)``.
    """
    votes = [0] * len(influence)
    placements = [Placement.LOST if x > 0 else Placement.ABSENT for x in influence]
    present = [i for i, x in enumerate(influence) if x > 0]
    if present:
        top = max(influence[i] for i in present)
        firsts = [i for i in present if influence[i] == top]
        if len(awards) == 1:
            _split(votes, placements, firsts, awards[0], Placement.FIRST)
        elif len(present) == 1 or len(firsts) > 1:
            _split(votes, placements, firsts, awards[0] + awards[1], Placement.FIRST)
        else:
            _split(votes, placements, firsts, awards[0], Placement.FIRST)
            rest = [i for i in present if i not in firsts]
            runner_up = max(influence[i] for i in rest)
            seconds = [i for i in rest if influence[i] == runner_up]
            _split(votes, placements, seconds, awards[1], Placement.SECOND)
    return RegionResult(tuple(votes), tuple(placements))


class TieBreak(StrEnum):
    SCORE = "score"
    HAND = "hand"
    BANK = "bank"
    DRAW = "draw"


def determine_winners(
    scores: Sequence[int], hands: Sequence[int], banks: Sequence[int]
) -> tuple[tuple[int, ...], TieBreak]:
    """Most votes, then most cubes in hand, then most cubes in bank; otherwise a draw."""
    contenders = list(range(len(scores)))
    for key, label in ((scores, TieBreak.SCORE), (hands, TieBreak.HAND), (banks, TieBreak.BANK)):
        best = max(key[i] for i in contenders)
        contenders = [i for i in contenders if key[i] == best]
        if len(contenders) == 1:
            return tuple(contenders), label
    return tuple(contenders), TieBreak.DRAW
