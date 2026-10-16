from primary.data.board import Region
from primary.data.candidates import Candidate
from primary.data.cards import Card
from primary.data.news import News
from primary.engine.actions import NO, YES, DecisionType, card_action, player_action, region_action
from primary.engine.events import EventType
from primary.engine.resolve import resolve_card
from tests.helpers import DEFAULT_CANDIDATES, drive, make_table

R = region_action
REG = Region
C = Candidate


def with_seat0(candidate):
    return (candidate, *DEFAULT_CANDIDATES[1:])


def hands(h0):
    return [h0, 3, 3, 3, 1]


# -- Super PAC and Fundraiser ----------------------------------------------------------------


def test_super_pac_is_counted_for_round_end():  # rule 8
    table, _ = make_table()
    drive(resolve_card(table, 0, Card.SUPER_PAC))
    assert table.state.rnd.super_pacs[0] == 1


def test_fundraiser_gains_one():  # rule 9
    table, _ = make_table(hands=hands(2))
    drive(resolve_card(table, 0, Card.FUNDRAISER))
    assert table.state.players[0].hand == 3
    assert table.state.rnd.fundraisers[0] == 1


def test_generous_donors_adds_one_per_fundraiser():  # rule 36
    table, sink = make_table(news=News.GENEROUS_DONORS, hands=hands(2))
    drive(resolve_card(table, 0, Card.FUNDRAISER))
    assert table.state.players[0].hand == 4
    assert [e.detail for e in sink.of_type(EventType.GAIN)] == ["fundraiser", "generous_donors"]


# -- Plane -----------------------------------------------------------------------------------


def test_plane_pays_one_and_moves_anywhere_else():  # rule 10
    table, _ = make_table(hands=hands(3))
    requests, _ = drive(resolve_card(table, 0, Card.PLANE), [R(REG.NEW_ENGLAND)])
    assert requests[0].decision is DecisionType.PLANE_TARGET
    assert R(REG.PACIFIC_NW) not in requests[0].legal_actions
    assert len(requests[0].legal_actions) == 13
    assert table.state.players[0].pawn is REG.NEW_ENGLAND
    assert table.state.players[0].hand == 2


def test_plane_with_empty_hand_has_no_effect():  # rule 10
    table, sink = make_table(hands=hands(0))
    requests, _ = drive(resolve_card(table, 0, Card.PLANE))
    assert requests == []
    assert table.state.players[0].pawn is REG.PACIFIC_NW
    assert sink.of_type(EventType.NO_EFFECT)[0].detail == "insufficient_hand"


def test_ryan_plane_is_free():  # rule 26
    table, _ = make_table(candidates=with_seat0(C.RYAN_GUPTA), hands=hands(0))
    drive(resolve_card(table, 0, Card.PLANE), [R(REG.SOUTH)])
    assert table.state.players[0].pawn is REG.SOUTH
    assert table.state.players[0].hand == 0


def test_frequent_flyer_plane_is_free():  # rule 51
    table, _ = make_table(news=News.FREQUENT_FLYER, hands=hands(0))
    drive(resolve_card(table, 0, Card.PLANE), [R(REG.SOUTH)])
    assert table.state.players[0].pawn is REG.SOUTH


def test_paparazzi_plane_may_place_a_bank_cube():  # rule 41
    table, _ = make_table(news=News.PAPARAZZI, hands=hands(3))
    bank = table.state.players[0].bank
    requests, _ = drive(resolve_card(table, 0, Card.PLANE), [R(REG.SOUTH), YES])
    assert requests[1].decision is DecisionType.OPTIONAL_BANK_PLACEMENT
    assert table.state.cubes[REG.SOUTH][0] == 1
    assert table.state.players[0].bank == bank + 1 - 1  # +1 plane payment, -1 placement


# -- Bus -------------------------------------------------------------------------------------


def test_bus_moves_one_step():  # rule 11
    table, _ = make_table()
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.NORTH)])
    assert set(requests[0].legal_actions) == {R(REG.NORTH), R(REG.SOUTH_WEST), R(REG.WEST)}
    assert table.state.players[0].pawn is REG.NORTH


def test_antonio_bus_moves_up_to_two_steps():  # rule 21
    table, _ = make_table(
        candidates=with_seat0(C.ANTONIO_PARADES),
        pawns=[REG.NEW_ENGLAND, REG.WEST, REG.SOUTH_WEST, REG.GREATER_TEXAS, REG.NORTH],
    )
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.APPALACHIA)])
    assert set(requests[0].legal_actions) == {
        R(REG.NORTH_EAST),
        R(REG.EARLY_MIDWEST),
        R(REG.APPALACHIA),
    }


def test_road_trip_extends_bus_by_one():  # rule 48
    table, _ = make_table(
        news=News.ROAD_TRIP,
        pawns=[REG.NEW_ENGLAND, REG.WEST, REG.SOUTH_WEST, REG.GREATER_TEXAS, REG.NORTH],
    )
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.NORTH_EAST)])
    assert len(requests[0].legal_actions) == 3


def test_antonio_with_road_trip_moves_up_to_three():  # rules 48, 52
    table, _ = make_table(
        candidates=with_seat0(C.ANTONIO_PARADES),
        news=News.ROAD_TRIP,
        pawns=[REG.NEW_ENGLAND, REG.WEST, REG.SOUTH_WEST, REG.GREATER_TEXAS, REG.NORTH],
    )
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.SOUTH)])
    assert len(requests[0].legal_actions) == 6


def test_calvin_bus_may_place_from_hand():  # rule 30
    table, _ = make_table(candidates=with_seat0(C.CALVIN_MATSUI), hands=hands(3))
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.WEST), YES])
    assert requests[1].decision is DecisionType.OPTIONAL_HAND_PLACEMENT
    assert table.state.cubes[REG.WEST][0] == 1
    assert table.state.players[0].hand == 2


def test_calvin_may_decline():  # rule 30
    table, _ = make_table(candidates=with_seat0(C.CALVIN_MATSUI), hands=hands(3))
    drive(resolve_card(table, 0, Card.BUS), [R(REG.WEST), NO])
    assert table.state.cubes[REG.WEST][0] == 0


def test_calvin_with_paparazzi_stacks():  # rule 52
    table, _ = make_table(
        candidates=with_seat0(C.CALVIN_MATSUI), news=News.PAPARAZZI, hands=hands(3)
    )
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.WEST), YES, YES])
    assert [r.decision for r in requests] == [
        DecisionType.BUS_TARGET,
        DecisionType.OPTIONAL_HAND_PLACEMENT,
        DecisionType.OPTIONAL_BANK_PLACEMENT,
    ]
    assert table.state.cubes[REG.WEST][0] == 2


# -- Ads -------------------------------------------------------------------------------------


def test_positive_ad_places_one_from_hand_anywhere():  # rule 12
    table, _ = make_table(hands=hands(2))
    requests, _ = drive(resolve_card(table, 0, Card.POSITIVE_AD), [R(REG.NEW_ENGLAND)])
    assert len(requests[0].legal_actions) == 14
    assert table.state.cubes[REG.NEW_ENGLAND][0] == 1
    assert table.state.players[0].hand == 1


def test_positive_ad_with_empty_hand_has_no_effect():  # rule 12
    table, sink = make_table(hands=hands(0))
    requests, _ = drive(resolve_card(table, 0, Card.POSITIVE_AD))
    assert requests == []
    assert len(sink.of_type(EventType.NO_EFFECT)) == 1


def test_extra_airtime_places_two_independently():  # rule 46
    table, _ = make_table(news=News.EXTRA_AIRTIME, hands=hands(2))
    drive(resolve_card(table, 0, Card.POSITIVE_AD), [R(REG.WEST), R(REG.SOUTH)])
    assert table.state.cubes[REG.WEST][0] == 1
    assert table.state.cubes[REG.SOUTH][0] == 1


def test_extra_airtime_with_one_cube_places_one():  # rule 46
    table, sink = make_table(news=News.EXTRA_AIRTIME, hands=hands(1))
    drive(resolve_card(table, 0, Card.POSITIVE_AD), [R(REG.WEST)])
    assert table.state.cubes[REG.WEST][0] == 1
    assert len(sink.of_type(EventType.NO_EFFECT)) == 1


def test_negative_ad_removes_the_chosen_cube_to_its_owner_bank():  # rule 13
    table, _ = make_table(cubes={(REG.SOUTH, 2): 2, (REG.SOUTH, 3): 1, (REG.WEST, 0): 1})
    bank3 = table.state.players[3].bank
    requests, _ = drive(resolve_card(table, 0, Card.NEGATIVE_AD), [R(REG.SOUTH), player_action(3)])
    assert set(requests[0].legal_actions) == {R(REG.SOUTH), R(REG.WEST)}
    assert requests[1].decision is DecisionType.NEGATIVE_AD_OWNER
    assert set(requests[1].legal_actions) == {player_action(2), player_action(3)}
    assert table.state.cubes[REG.SOUTH][3] == 0
    assert table.state.players[3].bank == bank3 + 1


def test_negative_ad_with_one_owner_is_forced():
    table, _ = make_table(cubes={(REG.SOUTH, 2): 1, (REG.WEST, 1): 1})
    requests, _ = drive(resolve_card(table, 0, Card.NEGATIVE_AD), [R(REG.SOUTH)])
    assert len(requests) == 1
    assert table.state.cubes[REG.SOUTH][2] == 0


def test_negative_ad_must_remove_own_cube_if_it_is_the_only_one():  # rule 13
    table, sink = make_table(cubes={(REG.WEST, 0): 1})
    requests, _ = drive(resolve_card(table, 0, Card.NEGATIVE_AD))
    assert requests == []
    assert table.state.cubes[REG.WEST][0] == 0
    assert all(e.forced for e in sink.of_type(EventType.DECISION))


def test_negative_ad_on_empty_map_has_no_effect():  # rule 13
    table, sink = make_table()
    drive(resolve_card(table, 0, Card.NEGATIVE_AD))
    assert sink.of_type(EventType.NO_EFFECT)[0].detail == "empty_map"


def test_spin_doctor_lets_positive_ad_act_as_negative_ad():  # rule 42
    table, _ = make_table(news=News.SPIN_DOCTOR, cubes={(REG.SOUTH, 1): 1})
    requests, _ = drive(
        resolve_card(table, 0, Card.POSITIVE_AD), [card_action(Card.NEGATIVE_AD)]
    )
    assert requests[0].decision is DecisionType.SPIN_DOCTOR_TYPE
    assert table.state.cubes[REG.SOUTH][1] == 0


# -- Rally -----------------------------------------------------------------------------------


def test_rally_places_two_on_the_pawn_region():  # rule 14
    table, _ = make_table(hands=hands(3))
    drive(resolve_card(table, 0, Card.RALLY))
    assert table.state.cubes[REG.PACIFIC_NW][0] == 2
    assert table.state.players[0].hand == 1


def test_rally_needs_two_cubes():  # rule 14
    table, sink = make_table(hands=hands(1))
    drive(resolve_card(table, 0, Card.RALLY))
    assert table.state.cubes[REG.PACIFIC_NW][0] == 0
    assert table.state.players[0].hand == 1
    assert sink.of_type(EventType.NO_EFFECT)[0].detail == "insufficient_hand"


def test_wall_street_first_rally_uses_bank():  # rule 47
    table, _ = make_table(news=News.WALL_STREET_SUPPORT, hands=hands(2))
    bank = table.state.players[0].bank
    drive(resolve_card(table, 0, Card.RALLY))
    assert (table.state.players[0].hand, table.state.players[0].bank) == (2, bank - 2)
    drive(resolve_card(table, 0, Card.RALLY))
    assert table.state.players[0].hand == 0
    assert table.state.cubes[REG.PACIFIC_NW][0] == 4


def test_wall_street_needs_two_bank_cubes():  # rule 47
    table, sink = make_table(news=News.WALL_STREET_SUPPORT, hands=hands(29))  # bank = 1
    drive(resolve_card(table, 0, Card.RALLY))
    assert sink.of_type(EventType.NO_EFFECT)[0].detail == "insufficient_bank"
    drive(resolve_card(table, 0, Card.RALLY))  # second Rally uses the hand
    assert table.state.players[0].hand == 27


def test_lily_extra_cube_on_first_effective_rally():  # rule 28
    table, _ = make_table(candidates=with_seat0(C.LILY_NGUYEN), hands=hands(5))
    requests, _ = drive(resolve_card(table, 0, Card.RALLY), [YES])
    assert requests[0].decision is DecisionType.OPTIONAL_HAND_PLACEMENT
    assert table.state.cubes[REG.PACIFIC_NW][0] == 3
    requests, _ = drive(resolve_card(table, 0, Card.RALLY))
    assert requests == []
    assert table.state.cubes[REG.PACIFIC_NW][0] == 5


def test_lily_extra_waits_for_an_effective_rally():  # rule 28
    table, _ = make_table(candidates=with_seat0(C.LILY_NGUYEN), hands=hands(1))
    assert drive(resolve_card(table, 0, Card.RALLY))[0] == []  # fails: hand 1
    table.gain(0, 2, "test")
    requests, _ = drive(resolve_card(table, 0, Card.RALLY), [YES])
    assert len(requests) == 1
    assert table.state.cubes[REG.PACIFIC_NW][0] == 3


def test_lily_extra_comes_from_hand_under_wall_street():  # rule 47
    table, _ = make_table(
        candidates=with_seat0(C.LILY_NGUYEN), news=News.WALL_STREET_SUPPORT, hands=hands(1)
    )
    drive(resolve_card(table, 0, Card.RALLY), [YES])
    assert table.state.cubes[REG.PACIFIC_NW][0] == 3
    assert table.state.players[0].hand == 0
