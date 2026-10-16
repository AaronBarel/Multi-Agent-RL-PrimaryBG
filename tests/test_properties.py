"""Property-based checks over many random games (spec §5.5, §7)."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from primary.data.board import Region
from primary.data.candidates import Candidate
from primary.engine.actions import DecisionRequest, GameOver
from primary.engine.events import ListSink
from primary.engine.game import Game


def random_candidates(rng: np.random.Generator, n: int) -> list[Candidate]:
    return [Candidate(int(c)) for c in rng.choice(len(Candidate), n, replace=False)]


@settings(max_examples=2000, deadline=None)  # spec §7: thousands of seeded games
@given(seed=st.integers(0, 2**31 - 2), num_players=st.integers(3, 6))
def test_random_games_respect_invariants(seed, num_players):
    rng = np.random.default_rng(seed)
    sink = ListSink()
    game = Game(num_players, sink=sink, check_invariants=True)  # cube conservation every step
    outcome = game.reset(seed, random_candidates(rng, num_players))
    previous = [0] * num_players
    while isinstance(outcome, DecisionRequest):
        assert outcome.legal_actions
        outcome = game.step(int(rng.choice(outcome.legal_actions)))
        scores = [p.score for p in game.state.players]
        assert all(now >= before for now, before in zip(scores, previous, strict=True))
        previous = scores
    assert isinstance(outcome, GameOver)
    state = game.state
    assert state.round == 11
    assert sorted(r for regions in state.schedule.values() for r in regions) == list(Region)
    assert all(state.voted)
    assert all(sum(row) == 0 for row in state.cubes)  # every region voted, so all cubes went home
    assert len(state.news_history) == 11
    assert len(sink.rounds) == 11 * num_players
    assert sum(r.votes_gained for r in sink.rounds) == sum(outcome.scores)
    assert outcome.winners


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 2))
def test_games_are_deterministic(seed):
    def run():
        rng = np.random.default_rng(seed)
        sink = ListSink()
        game = Game(5, sink=sink)
        outcome = game.reset(seed, random_candidates(rng, 5))
        while isinstance(outcome, DecisionRequest):
            outcome = game.step(int(rng.choice(outcome.legal_actions)))
        return outcome, sink.events, sink.rounds

    assert run() == run()
