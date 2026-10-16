"""Uniformly random legal play: the weakest baseline."""

from __future__ import annotations

import numpy as np

from primary.engine.actions import DecisionRequest
from primary.engine.game import Game


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, game: Game, request: DecisionRequest) -> int:
        return int(self.rng.choice(request.legal_actions))
