import pytest

from primary.data.board import Region
from primary.data.candidates import MAX_PLAYERS
from primary.data.cards import Card
from primary.engine.actions import (
    NUM_ACTIONS,
    PASS,
    STOP,
    YES,
    action_card,
    action_player,
    action_region,
    card_action,
    describe_action,
    player_action,
    region_action,
)


def test_action_layout():
    assert NUM_ACTIONS == 31
    assert {region_action(r) for r in Region} == set(range(14))
    assert {card_action(c) for c in Card} == set(range(14, 21))
    assert {player_action(s) for s in range(MAX_PLAYERS)} == set(range(21, 27))
    assert (YES, PASS, STOP) == (27, 29, 30)


def test_roundtrips():
    for region in Region:
        assert action_region(region_action(region)) is region
    for card in Card:
        assert action_card(card_action(card)) is card
    for seat in range(MAX_PLAYERS):
        assert action_player(player_action(seat)) == seat


def test_decoding_the_wrong_kind_raises():
    with pytest.raises(ValueError):
        action_region(card_action(Card.RALLY))
    with pytest.raises(ValueError):
        action_card(PASS)


def test_describe_action():
    assert describe_action(region_action(Region.NEW_ENGLAND)) == "region:NEW_ENGLAND"
    assert describe_action(card_action(Card.RALLY)) == "card:RALLY"
    assert describe_action(player_action(2)) == "player:2"
    assert describe_action(PASS) == "pass"
