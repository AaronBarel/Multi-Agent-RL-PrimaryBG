"""Conservation checks run after every step in tests and in debug mode (spec §5.5)."""

from __future__ import annotations

from primary.data.candidates import TOTAL_CUBES
from primary.engine.state import GameState


class InvariantError(AssertionError):
    """The game state violates a rule that must always hold."""


def check_invariants(state: GameState) -> None:
    for seat, player in enumerate(state.players):
        column = [row[seat] for row in state.cubes]
        if player.hand < 0 or player.bank < 0 or min(column) < 0:
            raise InvariantError(f"seat {seat} has a negative cube count")
        total = player.hand + player.bank + sum(column)
        if total != TOTAL_CUBES:
            raise InvariantError(f"seat {seat} owns {total} cubes, expected {TOTAL_CUBES}")
