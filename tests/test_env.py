import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from primary.data.candidates import TOTAL_CUBES, Candidate
from primary.data.cards import Card
from primary.engine.actions import (
    PLAYER_BASE,
    DecisionRequest,
    DecisionType,
    GameOver,
    IllegalActionError,
    player_action,
)
from primary.engine.voting import TieBreak
from primary.env.aec_env import env
from primary.env.encoding import encode_mask, encode_observation, obs_size, to_absolute_action
from primary.env.rewards import shaped_rewards, terminal_rewards
from tests.helpers import make_state


# Dict observations with an action mask are the standard PettingZoo pattern for masked games;
# api_test still warns about them, so silence just those two warnings.
@pytest.mark.filterwarnings("ignore:Observation space for each agent probably should be")
@pytest.mark.filterwarnings("ignore:Observation is not a NumPy array")
def test_pettingzoo_api_compliance():
    api_test(env(), num_cycles=1000, verbose_progress=False)


def test_seeded_resets_are_reproducible():
    seed_test(env, num_cycles=200)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_observation_size(n):
    state = make_state(candidates=tuple(Candidate)[:n])
    assert encode_observation(state, 0, None).shape == (obs_size(n),)


def test_opponents_face_down_cards_are_hidden():
    a, b = make_state(), make_state()
    a.rnd.queue[1].extend([Card.RALLY] * 3 + [Card.BUS])
    b.rnd.queue[1].extend([Card.SUPER_PAC] * 4)
    assert np.array_equal(encode_observation(a, 0, None), encode_observation(b, 0, None))
    a.rnd.revealed[1].append(Card.RALLY)
    b.rnd.revealed[1].append(Card.SUPER_PAC)
    assert not np.array_equal(encode_observation(a, 0, None), encode_observation(b, 0, None))


def test_own_queue_is_visible():
    a, b = make_state(), make_state()
    a.rnd.queue[0].append(Card.RALLY)
    b.rnd.queue[0].append(Card.BUS)
    assert not np.array_equal(encode_observation(a, 0, None), encode_observation(b, 0, None))


def test_observation_is_egocentric():
    state = make_state(hands=[2, 9, 3, 3, 1])
    hands_offset = 2 * 14 * 5  # after the cube and pawn blocks
    assert encode_observation(state, 1, None)[hands_offset] == pytest.approx(9 / TOTAL_CUBES)
    assert encode_observation(state, 0, None)[hands_offset] == pytest.approx(2 / TOTAL_CUBES)


def test_player_actions_are_rotated_to_the_observer():
    request = DecisionRequest(
        seat=1,
        decision=DecisionType.NEGATIVE_AD_OWNER,
        legal_actions=(player_action(0), player_action(2)),
    )
    mask = encode_mask(request, me=1, n=5)
    assert set(np.flatnonzero(mask)) == {PLAYER_BASE + 1, PLAYER_BASE + 4}
    assert to_absolute_action(PLAYER_BASE + 1, me=1, n=5) == player_action(2)
    assert to_absolute_action(PLAYER_BASE + 4, me=1, n=5) == player_action(0)


def test_terminal_and_shaped_rewards():
    win = GameOver((10, 5, 0), (0, 0, 0), (0, 0, 0), (0,), TieBreak.SCORE)
    assert terminal_rewards(win, 3) == [1.0, 0.0, 0.0]
    draw = GameOver((10, 5, 10), (0, 0, 0), (0, 0, 0), (0, 2), TieBreak.DRAW)
    assert terminal_rewards(draw, 3) == [0.5, 0.0, 0.5]
    assert shaped_rewards([0, 0], [24, 0]) == [pytest.approx(0.1), 0.0]


def run_episode(e, seed):
    rng = np.random.default_rng(seed)
    e.reset(seed=seed)
    totals = dict.fromkeys(e.possible_agents, 0.0)
    for agent in e.agent_iter():
        obs, reward, termination, truncation, _ = e.last()
        totals[agent] += reward
        assert obs["observation"].min() >= 0.0 and obs["observation"].max() <= 1.0
        done = termination or truncation
        e.step(None if done else int(rng.choice(np.flatnonzero(obs["action_mask"]))))
    return totals


def test_unshaped_episode_hands_out_exactly_one_point():
    totals = run_episode(env(), seed=3)
    assert sum(totals.values()) == pytest.approx(1.0)
    assert min(totals.values()) >= 0.0


def test_shaped_episode_adds_vote_share():
    e = env(shaped_reward=True)
    totals = run_episode(e, seed=3)
    scores = e.unwrapped.game.result.scores
    assert sum(totals.values()) == pytest.approx(1.0 + sum(scores) / 240)


def test_illegal_action_raises():
    e = env()
    e.reset(seed=0)
    mask = e.observe(e.agent_selection)["action_mask"]
    illegal = int(np.flatnonzero(mask == 0)[0])
    with pytest.raises(IllegalActionError):
        e.step(illegal)


def test_fixed_candidates_option():
    e = env()
    chosen = list(Candidate)[:5]
    e.reset(seed=1, options={"candidates": chosen})
    assert [p.candidate for p in e.unwrapped.game.state.players] == chosen
