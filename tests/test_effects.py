import pytest

from primary.data.candidates import Candidate
from primary.data.news import News
from primary.engine.abilities import ABILITY_MODIFIERS
from primary.engine.effects import build_modifiers
from primary.engine.modifiers import NewsMove
from primary.engine.news_effects import NEWS_MODIFIERS

C, N = Candidate, News


def test_every_candidate_and_news_card_has_an_effect():
    assert set(ABILITY_MODIFIERS) == set(Candidate)
    assert set(NEWS_MODIFIERS) == set(News)


def test_base_rules():
    m = build_modifiers(C.GEORGIA_BARNES, None)
    assert (m.bus_range, m.plane_cost, m.max_cards, m.positive_ad_placements) == (1, 1, 4, 1)


@pytest.mark.parametrize(
    ("candidate", "field", "value"),
    [
        (C.ANTONIO_PARADES, "bus_range", 2),  # rule 21
        (C.KENNETH_WALKER, "two_fundraiser_bonus", 1),  # rule 22
        (C.BRYCE_MCKAY, "no_award_gain", 1),  # rule 23
        (C.KRISTEN_RAMIREZ, "unique_most_super_pac_bonus", 1),  # rule 24
        (C.DIANE_PROUDIAN, "can_skip_card", True),  # rule 25
        (C.RYAN_GUPTA, "plane_cost", 0),  # rule 26
        (C.CHESTER_FREEMAN, "colocation_gain", 1),  # rule 27
        (C.LILY_NGUYEN, "first_rally_extra", True),  # rule 28
        (C.ERIN_FIELDS, "erin_premove", True),  # rule 29
        (C.CALVIN_MATSUI, "bus_hand_placement", True),  # rule 30
        (C.ANNA_PATEL, "token_influence", 2),  # rule 31
        (C.GEORGIA_BARNES, "home_regions", 3),  # rule 32
    ],
)
def test_ability_modifiers(candidate, field, value):
    assert getattr(build_modifiers(candidate, None), field) == value


@pytest.mark.parametrize(
    ("news", "field", "value"),
    [
        (N.COUNTERMEASURES, "enter_contested_gain", 1),  # rule 34
        (N.OPPO_RESEARCH, "colocation_gain", 2),  # rule 35
        (N.GENEROUS_DONORS, "fundraiser_extra", 1),  # rule 36
        (N.TAX_REFORM, "immediate_gain", 2),  # rule 37
        (N.LAST_MINUTE_DEBATE, "news_move", NewsMove.DEBATE),  # rule 38
        (N.DETOUR, "news_move", NewsMove.DETOUR),  # rule 39
        (N.FROZEN_ASSETS, "freeze_cards", True),  # rule 40
        (N.PAPARAZZI, "move_bank_placement", True),  # rule 41
        (N.SPIN_DOCTOR, "spin_doctor", True),  # rule 42
        (N.FINAL_PUSH, "max_cards", 5),  # rule 43
        (N.LONG_TERM_PLANNING, "news_move", NewsMove.LONG_TERM),  # rule 44
        (N.DEREGULATION, "fundraiser_most_award", (2, 1)),  # rule 45
        (N.EXTRA_AIRTIME, "positive_ad_placements", 2),  # rule 46
        (N.WALL_STREET_SUPPORT, "first_rally_from_bank", True),  # rule 47
        (N.ROAD_TRIP, "bus_range", 2),  # rule 48
        (N.PLUTOCRACY, "extra_super_pac_award", (6, 3)),  # rule 49
        (N.NEW_CONSTITUENTS, "news_swap", True),  # rule 50
        (N.FREQUENT_FLYER, "plane_cost", 0),  # rule 51
    ],
)
def test_news_modifiers(news, field, value):
    assert getattr(build_modifiers(C.GEORGIA_BARNES, news), field) == value


def test_effects_stack():  # rule 52
    assert build_modifiers(C.ANTONIO_PARADES, N.ROAD_TRIP).bus_range == 3
    assert build_modifiers(C.CHESTER_FREEMAN, N.OPPO_RESEARCH).colocation_gain == 3
    assert build_modifiers(C.RYAN_GUPTA, N.FREQUENT_FLYER).plane_cost == 0
    calvin = build_modifiers(C.CALVIN_MATSUI, N.PAPARAZZI)
    assert calvin.bus_hand_placement and calvin.move_bank_placement
