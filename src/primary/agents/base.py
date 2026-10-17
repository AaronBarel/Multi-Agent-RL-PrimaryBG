"""The interface every bot and (later) every trained policy implements."""

from __future__ import annotations

from typing import Protocol

from primary.engine.actions import DecisionRequest
from primary.engine.game import Game


class Agent(Protocol):
    name: str

    def act(self, game: Game, request: DecisionRequest) -> int:
        """Return one of ``request.legal_actions``.

        Agents may read ``game.state`` but must only use public information plus their own
        committed cards (``state.rnd.queue[request.seat]``); opponents' queues are hidden.
        """
        ...
