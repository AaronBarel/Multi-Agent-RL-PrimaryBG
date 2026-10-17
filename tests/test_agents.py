import pytest

from primary.agents import AGENT_NAMES, make_agent
from primary.agents.heuristic import HeuristicAgent, plan_cards
from primary.agents.random_agent import RandomAgent
from primary.data.board import Region
from primary.data.candidates import Candidate
from primary.data.cards import Card
from primary.engine.actions import DecisionRequest, GameOver
from primary.engine.effects import build_modifiers
from primary.engine.game import Game
from tests.helpers import make_state

KENNETH = build_modifiers(Candidate.KENNETH_WALKER, None)
F, RALLY, BUS, PLANE = Card.FUNDRAISER, Card.RALLY, Card.BUS, Card.PLANE


def play(agents, candidates, seed):
    game = Game(len(candidates), check_invariants=True)
    outcome = game.reset(seed, candidates)
    while isinstance(outcome, DecisionRequest):
        action = agents[outcome.seat].act(game, outcome)
        assert action in outcome.legal_actions
        outcome = game.step(action)
    return outcome


def test_random_agent_completes_a_game():
    agents = [RandomAgent(seed=s) for s in range(5)]
    assert isinstance(play(agents, list(Candidate)[:5], seed=0), GameOver)


def test_heuristic_completes_games_with_every_candidate():
    for start in range(len(Candidate)):
        candidates = [Candidate((start + k) % len(Candidate)) for k in range(5)]
        agents = [HeuristicAgent() for _ in range(5)]
        assert isinstance(play(agents, candidates, seed=start), GameOver)


def test_mixed_table_completes_games():
    agents = [HeuristicAgent(), RandomAgent(1), HeuristicAgent(), RandomAgent(2), RandomAgent(3)]
    for seed in range(5):
        assert isinstance(play(agents, list(Candidate)[5:10], seed=seed), GameOver)


def test_make_agent():
    assert set(AGENT_NAMES) == {"random", "heuristic"}
    assert make_agent("heuristic").name == "heuristic"
    assert make_agent("random", seed=1).name == "random"
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("nope")


def _state_near_new_england(pawn, hand):
    return make_state(
        hands=[hand, 3, 3, 3, 1],
        pawns=[pawn, Region.WEST, Region.SOUTH_WEST, Region.GREATER_TEXAS, Region.NORTH],
        schedule={2: (Region.NEW_ENGLAND,)},
    )


def test_plan_fundraises_before_rallying_with_an_empty_hand():
    state = _state_near_new_england(Region.NEW_ENGLAND, hand=0)
    assert plan_cards(state, 0, KENNETH) == [F, F, RALLY, F]


def test_plan_takes_the_bus_to_an_adjacent_target():
    state = _state_near_new_england(Region.NORTH_EAST, hand=3)
    assert plan_cards(state, 0, KENNETH) == [BUS, RALLY, F, RALLY]


def test_plan_flies_when_the_bus_is_frozen():
    state = _state_near_new_england(Region.NORTH_EAST, hand=3)
    state.frozen[0] = {Card.BUS: 1}
    assert plan_cards(state, 0, KENNETH) == [PLANE, RALLY, F, F]
