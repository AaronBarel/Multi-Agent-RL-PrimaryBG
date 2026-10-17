from primary.data.board import (
    ADJACENCY,
    AWARDS,
    NUM_REGIONS,
    NUM_ROUNDS,
    REGION_NAMES,
    STATES,
    TOTAL_VOTES,
    VOTING_SLOTS,
    Region,
)
from primary.data.candidates import (
    ABILITY_TEXT,
    DISPLAY_NAME,
    STARTING_HAND,
    TOTAL_CUBES,
    Candidate,
)
from primary.data.cards import CARD_COUNTS, CARD_TEXT, CARDS_PER_ROUND, Card
from primary.data.news import NEWS_TEXT, News


def test_fourteen_regions_with_names_and_fifty_states():
    assert NUM_REGIONS == 14
    assert set(REGION_NAMES) == set(Region)
    all_states = [s for states in STATES.values() for s in states]
    assert len(all_states) == 50
    assert len(set(all_states)) == 50


def test_adjacency_is_symmetric_and_irreflexive():
    assert set(ADJACENCY) == set(Region)
    for region, neighbours in ADJACENCY.items():
        assert region not in neighbours
        assert neighbours, f"{region.name} has no neighbours"
        for other in neighbours:
            assert region in ADJACENCY[other], f"{region.name} -> {other.name} not symmetric"


def test_new_england_only_borders_north_east():
    assert ADJACENCY[Region.NEW_ENGLAND] == frozenset({Region.NORTH_EAST})


def test_awards_match_rulebook():
    assert set(AWARDS) == set(Region)
    assert AWARDS[Region.NEW_ENGLAND] == (18, 8)
    assert AWARDS[Region.NORTH] == (10,)
    assert sum(1 for a in AWARDS.values() if len(a) == 2) == 8
    assert TOTAL_VOTES == 240


def test_voting_slots_cover_every_region_once():
    assert set(VOTING_SLOTS) == set(range(1, NUM_ROUNDS + 1))
    assert sum(VOTING_SLOTS.values()) == NUM_REGIONS
    assert [rnd for rnd, n in VOTING_SLOTS.items() if n == 0] == [1, 4, 9]


def test_action_deck_has_fourteen_cards():
    assert sum(CARD_COUNTS.values()) == 14
    assert CARD_COUNTS[Card.SUPER_PAC] == 4
    assert CARD_COUNTS[Card.RALLY] == 3
    assert CARDS_PER_ROUND == 4
    assert set(CARD_TEXT) == set(Card)


def test_candidates_and_starting_hands():
    assert len(Candidate) == 12
    assert set(STARTING_HAND) == set(DISPLAY_NAME) == set(ABILITY_TEXT) == set(Candidate)
    assert STARTING_HAND[Candidate.ANTONIO_PARADES] == 5
    assert STARTING_HAND[Candidate.RYAN_GUPTA] == 1
    assert STARTING_HAND[Candidate.GEORGIA_BARNES] == 1
    assert TOTAL_CUBES == 30


def test_eighteen_news_cards():
    assert len(News) == 18
    assert set(NEWS_TEXT) == set(News)
