"""Per-player, per-round rule parameters. Defaults are the base rules; abilities and News edit them."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class NewsMove(StrEnum):
    NONE = "none"
    DEBATE = "last_minute_debate"
    DETOUR = "detour"
    LONG_TERM = "long_term_planning"


@dataclass
class Modifiers:
    # setup
    home_regions: int = 1
    # news phase
    immediate_gain: int = 0
    news_move: NewsMove = NewsMove.NONE
    news_swap: bool = False
    # commit phase
    max_cards: int = 4
    freeze_cards: bool = False
    # execute phase
    erin_premove: bool = False
    can_skip_card: bool = False
    plane_cost: int = 1
    bus_range: int = 1
    bus_hand_placement: bool = False
    move_bank_placement: bool = False
    fundraiser_extra: int = 0
    positive_ad_placements: int = 1
    spin_doctor: bool = False
    first_rally_from_bank: bool = False
    first_rally_extra: bool = False
    enter_contested_gain: int = 0
    # round end
    token_influence: int = 0
    extra_super_pac_award: tuple[int, int] = (0, 0)  # (unique most, tied most)
    unique_most_super_pac_bonus: int = 0
    fundraiser_most_award: tuple[int, int] = (0, 0)  # (unique most, tied most)
    two_fundraiser_bonus: int = 0
    no_award_gain: int = 0
    colocation_gain: int = 0
