import pytest

from primary.data.board import Region
from primary.data.news import News
from primary.engine.actions import EngineError
from primary.engine.events import EventType, Source
from primary.engine.invariants import InvariantError, check_invariants
from tests.helpers import make_state, make_table

R = Region


def test_gain_moves_bank_to_hand_and_is_capped():  # rule 16
    table, sink = make_table(hands=[28, 3, 3, 3, 1])
    player = table.state.players[0]
    assert player.bank == 2
    assert table.gain(0, 5, "test") == 2
    assert (player.hand, player.bank) == (30, 0)
    event = sink.of_type(EventType.GAIN)[-1]
    assert (event.amount, event.hand_before, event.hand_after, event.bank_after) == (2, 28, 30, 0)


def test_gain_from_empty_bank_moves_nothing():  # rule 16
    table, _ = make_table(hands=[30, 3, 3, 3, 1])
    assert table.gain(0, 4, "test") == 0
    assert table.state.players[0].hand == 30


def test_pay_moves_hand_to_bank():
    table, _ = make_table(hands=[2, 3, 3, 3, 1])
    assert table.pay(0, 1, "plane")
    assert table.state.players[0].hand == 1
    assert not table.pay(0, 2, "plane")
    assert table.state.players[0].hand == 1


def test_place_requires_enough_cubes_in_source():  # rule 16
    table, _ = make_table(hands=[1, 3, 3, 3, 1])
    assert not table.place(0, R.WEST, 2, Source.HAND, "rally")
    assert table.state.cubes[R.WEST][0] == 0
    assert table.place(0, R.WEST, 1, Source.HAND, "ad")
    assert table.state.cubes[R.WEST][0] == 1
    assert table.state.players[0].hand == 0


def test_place_from_bank():
    table, _ = make_table()
    bank = table.state.players[1].bank
    assert table.place(1, R.SOUTH, 2, Source.BANK, "home")
    assert table.state.players[1].bank == bank - 2
    assert table.state.cubes[R.SOUTH][1] == 2


def test_remove_returns_cube_to_owner_bank_even_own_cube():  # rule 13
    table, sink = make_table(cubes={(R.SOUTH, 0): 1, (R.SOUTH, 2): 2})
    bank2 = table.state.players[2].bank
    table.remove(R.SOUTH, 2, by_seat=0, reason="negative_ad")
    assert table.state.cubes[R.SOUTH][2] == 1
    assert table.state.players[2].bank == bank2 + 1
    table.remove(R.SOUTH, 0, by_seat=0, reason="negative_ad")
    assert table.state.cubes[R.SOUTH][0] == 0
    first = sink.of_type(EventType.CUBE_REMOVED)[0]
    assert (first.seat, first.cube_owner, first.bank_after) == (0, 2, bank2 + 1)


def test_remove_missing_cube_raises():
    table, _ = make_table()
    with pytest.raises(EngineError):
        table.remove(R.SOUTH, 1, by_seat=0, reason="negative_ad")


def test_move_cube_onto_voted_region():  # rules 15, 50
    table, _ = make_table(cubes={(R.WEST, 0): 1, (R.SOUTH, 0): 1}, voted=[R.NORTH])
    bank = table.state.players[0].bank
    table.move_cube(0, R.WEST, R.NORTH, by_seat=0, reason="swap", return_if_voted=True)
    assert table.state.cubes[R.WEST][0] == 0
    assert table.state.cubes[R.NORTH][0] == 0
    assert table.state.players[0].bank == bank + 1
    table.move_cube(0, R.SOUTH, R.NORTH, by_seat=0, reason="erin", return_if_voted=False)
    assert table.state.cubes[R.NORTH][0] == 1


def test_return_region_sends_every_cube_home():
    table, sink = make_table(cubes={(R.WEST, 0): 2, (R.WEST, 3): 1})
    banks = [p.bank for p in table.state.players]
    table.return_region(R.WEST)
    assert table.state.cubes[R.WEST] == [0, 0, 0, 0, 0]
    assert table.state.players[0].bank == banks[0] + 2
    assert table.state.players[3].bank == banks[3] + 1
    assert len(sink.of_type(EventType.CUBE_RETURNED)) == 2


def test_move_pawn_tracks_path():
    table, sink = make_table()
    table.move_pawn(0, R.MIDWEST, "bus")
    assert table.state.players[0].pawn is R.MIDWEST
    assert table.state.rnd.path[0] == [R.PACIFIC_NW, R.MIDWEST]
    move = sink.of_type(EventType.MOVE)[0]
    assert (move.pawn_before, move.pawn_after) == (R.PACIFIC_NW, R.MIDWEST)


def test_countermeasures_gain_only_when_an_opponent_has_cubes_there():  # rule 34
    table, _ = make_table(news=News.COUNTERMEASURES, cubes={(R.MIDWEST, 1): 1, (R.SOUTH, 0): 1})
    hand = table.state.players[0].hand
    table.move_pawn(0, R.MIDWEST, "bus")
    assert table.state.players[0].hand == hand + 1
    table.move_pawn(0, R.SOUTH, "bus")  # only our own cube there
    assert table.state.players[0].hand == hand + 1


def test_no_countermeasures_without_the_news_card():
    table, _ = make_table(cubes={(R.MIDWEST, 1): 1})
    hand = table.state.players[0].hand
    table.move_pawn(0, R.MIDWEST, "bus")
    assert table.state.players[0].hand == hand


def test_swap_pawn_with_cube():  # rule 50
    table, _ = make_table(cubes={(R.SOUTH, 3): 1})
    table.swap_pawn_with_cube(0, R.SOUTH, 3)
    assert table.state.players[0].pawn is R.SOUTH
    assert table.state.cubes[R.PACIFIC_NW][3] == 1
    assert table.state.cubes[R.SOUTH][3] == 0


def test_swap_into_voted_region_returns_the_cube():  # rule 50
    table, _ = make_table(cubes={(R.SOUTH, 3): 1}, voted=[R.PACIFIC_NW])
    bank = table.state.players[3].bank
    table.swap_pawn_with_cube(0, R.SOUTH, 3)
    assert table.state.cubes[R.PACIFIC_NW][3] == 0
    assert table.state.players[3].bank == bank + 1


def test_draw_news_redraws_frozen_assets_in_final_round():  # rule 40
    table, sink = make_table(round_=11)
    table.state.news_deck = [News.FROZEN_ASSETS, News.TAX_REFORM]
    assert table.draw_news() is News.TAX_REFORM
    assert table.state.news_history == [News.TAX_REFORM]
    assert [e.event_type for e in sink.events] == [EventType.NEWS_DISCARDED, EventType.NEWS_DRAWN]
    assert table.mods[0].immediate_gain == 2


def test_frozen_assets_is_kept_before_the_final_round():  # rule 40
    table, _ = make_table(round_=10)
    table.state.news_deck = [News.FROZEN_ASSETS, News.TAX_REFORM]
    assert table.draw_news() is News.FROZEN_ASSETS


def test_check_invariants_detects_a_lost_cube():
    state = make_state(cubes={(R.WEST, 0): 2})
    check_invariants(state)
    state.players[0].bank -= 1
    with pytest.raises(InvariantError):
        check_invariants(state)
