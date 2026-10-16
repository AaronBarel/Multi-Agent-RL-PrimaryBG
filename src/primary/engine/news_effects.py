"""News cards as Modifiers edits (spec §3 rules 33–51)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from primary.data.board import NUM_ROUNDS, Region
from primary.data.news import News
from primary.engine.modifiers import Modifiers, NewsMove
from primary.engine.state import GameState

ModifierFn = Callable[[Modifiers], None]


def countermeasures(m: Modifiers) -> None:
    m.enter_contested_gain += 1


def oppo_research(m: Modifiers) -> None:
    m.colocation_gain += 2


def generous_donors(m: Modifiers) -> None:
    m.fundraiser_extra += 1


def tax_reform(m: Modifiers) -> None:
    m.immediate_gain += 2


def last_minute_debate(m: Modifiers) -> None:
    m.news_move = NewsMove.DEBATE


def detour(m: Modifiers) -> None:
    m.news_move = NewsMove.DETOUR


def frozen_assets(m: Modifiers) -> None:
    m.freeze_cards = True


def paparazzi(m: Modifiers) -> None:
    m.move_bank_placement = True


def spin_doctor(m: Modifiers) -> None:
    m.spin_doctor = True


def final_push(m: Modifiers) -> None:
    m.max_cards = 5


def long_term_planning(m: Modifiers) -> None:
    m.news_move = NewsMove.LONG_TERM


def deregulation(m: Modifiers) -> None:
    m.fundraiser_most_award = (2, 1)


def extra_airtime(m: Modifiers) -> None:
    m.positive_ad_placements = 2


def wall_street_support(m: Modifiers) -> None:
    m.first_rally_from_bank = True


def road_trip(m: Modifiers) -> None:
    m.bus_range += 1


def plutocracy(m: Modifiers) -> None:
    m.extra_super_pac_award = (6, 3)


def new_constituents(m: Modifiers) -> None:
    m.news_swap = True


def frequent_flyer(m: Modifiers) -> None:
    m.plane_cost = 0


NEWS_MODIFIERS: Mapping[News, ModifierFn] = MappingProxyType(
    {
        News.COUNTERMEASURES: countermeasures,
        News.OPPO_RESEARCH: oppo_research,
        News.GENEROUS_DONORS: generous_donors,
        News.TAX_REFORM: tax_reform,
        News.LAST_MINUTE_DEBATE: last_minute_debate,
        News.DETOUR: detour,
        News.FROZEN_ASSETS: frozen_assets,
        News.PAPARAZZI: paparazzi,
        News.SPIN_DOCTOR: spin_doctor,
        News.FINAL_PUSH: final_push,
        News.LONG_TERM_PLANNING: long_term_planning,
        News.DEREGULATION: deregulation,
        News.EXTRA_AIRTIME: extra_airtime,
        News.WALL_STREET_SUPPORT: wall_street_support,
        News.ROAD_TRIP: road_trip,
        News.PLUTOCRACY: plutocracy,
        News.NEW_CONSTITUENTS: new_constituents,
        News.FREQUENT_FLYER: frequent_flyer,
    }
)


def move_options(kind: NewsMove, state: GameState, seat: int) -> tuple[tuple[Region, ...], bool]:
    """Destinations offered by a News move and whether the player may decline (rules 38, 39, 44)."""
    here = state.players[seat].pawn
    others = [r for r in Region if r != here]
    if kind is NewsMove.DEBATE:
        rounds = (state.round, state.round + 1)
        return tuple(r for r in others if state.vote_round[r] in rounds), True
    if kind is NewsMove.LONG_TERM:
        return tuple(r for r in others if state.vote_round[r] == NUM_ROUNDS), True
    if kind is NewsMove.DETOUR:
        return tuple(others), False
    return (), True
