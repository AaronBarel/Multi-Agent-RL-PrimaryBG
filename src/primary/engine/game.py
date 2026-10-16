"""Public engine API: reset/step a whole game (spec §5.2)."""

from __future__ import annotations

from collections.abc import Generator, Sequence

import numpy as np

from primary.data.board import NUM_ROUNDS
from primary.data.candidates import MAX_PLAYERS, MIN_PLAYERS, Candidate
from primary.engine.actions import (
    DecisionRequest,
    EngineError,
    GameOver,
    IllegalActionError,
    describe_action,
)
from primary.engine.events import EventSink
from primary.engine.invariants import check_invariants
from primary.engine.phases import commit_phase, execute_phase, news_phase, setup_phase
from primary.engine.round_end import finish_game, resolve_round_end
from primary.engine.state import GameState, new_game_state
from primary.engine.table import Table

Outcome = DecisionRequest | GameOver
GameFlow = Generator[DecisionRequest, int, GameOver]


class Game:
    """One game of The Primary.

    ``reset`` returns the first ``DecisionRequest``; answer each with ``step(action)`` until a
    ``GameOver`` is returned. Decisions with a single legal action are taken automatically.
    """

    def __init__(
        self, num_players: int = 5, sink: EventSink | None = None, check_invariants: bool = False
    ) -> None:
        if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
            raise ValueError(f"need {MIN_PLAYERS}-{MAX_PLAYERS} players, got {num_players}")
        self.num_players = num_players
        self.sink = sink
        self.check_invariants = check_invariants
        self.seed: int | None = None
        self.pending: DecisionRequest | None = None
        self.result: GameOver | None = None
        self.history: list[int] = []
        self._table: Table | None = None
        self._flow: GameFlow | None = None

    @property
    def table(self) -> Table:
        if self._table is None:
            raise EngineError("call reset() first")
        return self._table

    @property
    def state(self) -> GameState:
        return self.table.state

    def reset(self, seed: int, candidates: Sequence[Candidate]) -> Outcome:
        if len(candidates) != self.num_players:
            raise ValueError(f"expected {self.num_players} candidates, got {len(candidates)}")
        self.seed = seed
        self._table = Table(new_game_state(candidates, np.random.default_rng(seed)), self.sink)
        self._flow = self._play(self._table)
        self.pending = None
        self.result = None
        self.history = []
        return self._advance(None)

    def step(self, action: int) -> Outcome:
        request = self.pending
        if request is None:
            raise EngineError("the game is over (or not started); call reset()")
        if action not in request.legal_actions:
            legal = ", ".join(describe_action(a) for a in request.legal_actions)
            raise IllegalActionError(
                f"{describe_action(action)} is not legal for {request.decision.name}; "
                f"legal actions: {legal}"
            )
        self.history.append(action)
        return self._advance(action)

    def _advance(self, action: int | None) -> Outcome:
        assert self._flow is not None
        try:
            request = next(self._flow) if action is None else self._flow.send(action)
        except StopIteration as stop:
            result: GameOver = stop.value
            self.pending = None
            self.result = result
            if self.check_invariants:
                check_invariants(self.state)
            return result
        if self.check_invariants:
            check_invariants(self.state)
        self.pending = request
        return request

    @staticmethod
    def _play(table: Table) -> GameFlow:
        yield from setup_phase(table)
        for rnd in range(1, NUM_ROUNDS + 1):
            table.begin_round(rnd)
            yield from news_phase(table)
            yield from commit_phase(table)
            yield from execute_phase(table)
            resolve_round_end(table)
        return finish_game(table.state)
