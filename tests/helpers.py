"""Scenario builders shared by the engine tests."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from primary.data.board import NUM_ROUNDS, VOTING_SLOTS, Region
from primary.data.candidates import STARTING_HAND, TOTAL_CUBES, Candidate
from primary.data.news import News
from primary.engine.actions import DecisionRequest
from primary.engine.events import ListSink, Phase
from primary.engine.state import GameState, RoundState, Schedule, new_game_state
from primary.engine.table import Table

# Abilities that rarely interfere with card-level tests. Seat 0 is Kenneth, seat 4 Georgia.
DEFAULT_CANDIDATES = (
    Candidate.KENNETH_WALKER,
    Candidate.KRISTEN_RAMIREZ,
    Candidate.BRYCE_MCKAY,
    Candidate.CHESTER_FREEMAN,
    Candidate.GEORGIA_BARNES,
)


def fixed_schedule(assign: Mapping[int, Sequence[Region]]) -> Schedule:
    """A valid schedule with the given regions in the given rounds; the rest fill in by index."""
    used = {r for regions in assign.values() for r in regions}
    pool = [r for r in Region if r not in used]
    schedule: Schedule = {}
    for rnd in range(1, NUM_ROUNDS + 1):
        regions = list(assign.get(rnd, ()))
        assert len(regions) <= VOTING_SLOTS[rnd], f"round {rnd} has {VOTING_SLOTS[rnd]} slots"
        while len(regions) < VOTING_SLOTS[rnd]:
            regions.append(pool.pop(0))
        schedule[rnd] = tuple(regions)
    return schedule


def make_state(
    candidates: Sequence[Candidate] = DEFAULT_CANDIDATES,
    *,
    round_: int = 2,
    pawns: Sequence[Region] | None = None,
    hands: Sequence[int] | None = None,
    cubes: Mapping[tuple[Region, int], int] | None = None,
    scores: Sequence[int] | None = None,
    voted: Iterable[Region] = (),
    schedule: Mapping[int, Sequence[Region]] | None = None,
    seed: int = 0,
) -> GameState:
    """A mid-game state. Pawns default to Region(seat); banks are derived so each player owns 30."""
    state = new_game_state(candidates, np.random.default_rng(seed))
    state.set_schedule(fixed_schedule(schedule or {}))
    state.round = round_
    state.start_seat = 0
    for (region, seat), count in (cubes or {}).items():
        state.cubes[region][seat] = count
    for seat, player in enumerate(state.players):
        player.pawn = pawns[seat] if pawns is not None else Region(seat)
        player.hand = hands[seat] if hands is not None else STARTING_HAND[player.candidate]
        player.score = scores[seat] if scores is not None else 0
        player.bank = TOTAL_CUBES - player.hand - state.cubes_on_map(seat)
        assert player.bank >= 0, f"seat {seat} has more than {TOTAL_CUBES} cubes"
    for region in voted:
        state.voted[region] = True
    state.rnd = RoundState.empty(state.players)
    return state


def make_table(
    state: GameState | None = None,
    *,
    news: News | None = None,
    phase: Phase = Phase.EXECUTE,
    **kwargs: Any,
) -> tuple[Table, ListSink]:
    if state is None:
        state = make_state(**kwargs)
    sink = ListSink()
    table = Table(state, sink)
    table.phase = phase
    table.set_news(news)
    return table, sink


def drive(
    flow: Generator[DecisionRequest, int, Any], answers: Iterable[int] = ()
) -> tuple[list[DecisionRequest], Any]:
    """Run an engine flow, answering each request from ``answers`` in order."""
    pending = list(answers)
    requests: list[DecisionRequest] = []
    try:
        request = next(flow)
        while True:
            requests.append(request)
            assert pending, f"no answer scripted for {request}"
            answer = pending.pop(0)
            assert answer in request.legal_actions, f"{answer} not legal for {request}"
            request = flow.send(answer)
    except StopIteration as stop:
        assert not pending, f"unused answers: {pending}"
        return requests, stop.value
