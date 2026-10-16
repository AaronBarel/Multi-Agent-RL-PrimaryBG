import numpy as np
import pytest

from primary.data.board import VOTING_SLOTS, Region
from primary.data.candidates import TOTAL_CUBES, Candidate
from primary.engine.state import make_schedule, new_game_state

C = Candidate


def test_new_game_state_splits_cubes_between_hand_and_bank():  # rule 4
    state = new_game_state([C.GEORGIA_BARNES, C.ANTONIO_PARADES, C.RYAN_GUPTA], np.random.default_rng(0))
    assert [p.hand for p in state.players] == [1, 5, 1]
    assert all(p.hand + p.bank == TOTAL_CUBES for p in state.players)
    assert state.cubes == [[0, 0, 0] for _ in Region]


def test_schedule_votes_each_region_once():  # rule 3
    for seed in range(20):
        schedule = make_schedule(np.random.default_rng(seed))
        assert sorted(r for regions in schedule.values() for r in regions) == list(Region)
        for rnd, regions in schedule.items():
            assert len(regions) == VOTING_SLOTS[rnd]


def test_vote_round_matches_schedule():
    state = new_game_state(list(C)[:5], np.random.default_rng(3))
    for rnd, regions in state.schedule.items():
        for region in regions:
            assert state.vote_round[region] == rnd


def test_same_seed_same_setup():
    a = new_game_state(list(C)[:5], np.random.default_rng(42))
    b = new_game_state(list(C)[:5], np.random.default_rng(42))
    assert (a.schedule, a.news_deck, a.start_seat) == (b.schedule, b.news_deck, b.start_seat)


@pytest.mark.parametrize(
    "candidates",
    [[C.ANNA_PATEL, C.ANNA_PATEL, C.RYAN_GUPTA], [C.ANNA_PATEL, C.RYAN_GUPTA], list(C)[:7]],
)
def test_rejects_invalid_candidate_lists(candidates):  # rule 1
    with pytest.raises(ValueError):
        new_game_state(candidates, np.random.default_rng(0))


def test_turn_order_starts_at_token_holder():  # rule 2
    state = new_game_state(list(C)[:5], np.random.default_rng(0))
    state.start_seat = 3
    assert state.turn_order() == [3, 4, 0, 1, 2]
