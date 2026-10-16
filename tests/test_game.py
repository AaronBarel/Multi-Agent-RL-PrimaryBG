import pytest

from primary.engine.actions import (
    NUM_ACTIONS,
    DecisionRequest,
    EngineError,
    GameOver,
    IllegalActionError,
)
from primary.engine.events import EventType, ListSink
from primary.engine.game import Game
from tests.helpers import DEFAULT_CANDIDATES, play_random


def test_full_random_game_reaches_game_over():  # rules 3, 33
    sink = ListSink()
    game = Game(5, sink=sink, check_invariants=True)
    result = play_random(game, seed=7)
    assert isinstance(result, GameOver)
    state = game.state
    assert state.round == 11
    assert len(state.news_history) == 11
    assert all(state.voted)
    assert len(sink.rounds) == 55
    assert result.winners
    assert game.result == result
    assert game.pending is None


def test_step_after_game_over_raises():
    game = Game(5)
    play_random(game, seed=1)
    with pytest.raises(EngineError):
        game.step(0)


def test_illegal_action_is_rejected_and_the_game_continues():
    game = Game(5)
    request = game.reset(3, DEFAULT_CANDIDATES)
    assert isinstance(request, DecisionRequest)
    illegal = next(a for a in range(NUM_ACTIONS) if a not in request.legal_actions)
    with pytest.raises(IllegalActionError, match="legal"):
        game.step(illegal)
    assert game.pending == request
    game.step(request.legal_actions[0])


def test_same_seed_and_actions_replay_identically():
    a, b = ListSink(), ListSink()
    result_a = play_random(Game(5, sink=a), seed=11)
    result_b = play_random(Game(5, sink=b), seed=11)
    assert result_a == result_b
    assert a.events == b.events
    assert a.rounds == b.rounds


def test_history_holds_only_requested_decisions():
    sink = ListSink()
    game = Game(5, sink=sink)
    play_random(game, seed=5)
    requested = [e.action for e in sink.of_type(EventType.DECISION) if not e.forced]
    assert game.history == requested


def test_invalid_player_counts_and_candidates_are_rejected():  # rule 1
    with pytest.raises(ValueError):
        Game(5).reset(0, DEFAULT_CANDIDATES[:4])
    with pytest.raises(ValueError):
        Game(7)


def test_state_before_reset_raises():
    with pytest.raises(EngineError):
        _ = Game(5).state
