"""The flat discrete action space, decision requests and outcomes (spec §4.2–4.3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from primary.data.board import NUM_REGIONS, Region
from primary.data.candidates import MAX_PLAYERS
from primary.data.cards import Card
from primary.engine.voting import TieBreak

REGION_BASE = 0
CARD_BASE = REGION_BASE + NUM_REGIONS
PLAYER_BASE = CARD_BASE + len(Card)
YES = PLAYER_BASE + MAX_PLAYERS
NO = YES + 1
PASS = NO + 1
STOP = PASS + 1
NUM_ACTIONS = STOP + 1


class EngineError(RuntimeError):
    """The engine was used incorrectly or reached an impossible state."""


class IllegalActionError(ValueError):
    """An action outside the current request's legal actions was submitted."""


class DecisionType(IntEnum):
    HOME_REGION = 0
    NEWS_MOVE = 1
    NEWS_SWAP_REGION = 2
    NEWS_SWAP_OWNER = 3
    COMMIT_CARD = 4
    ERIN_SOURCE = 5
    ERIN_DEST = 6
    DIANE_SKIP = 7
    PLANE_TARGET = 8
    BUS_TARGET = 9
    SPIN_DOCTOR_TYPE = 10
    POSITIVE_AD_TARGET = 11
    NEGATIVE_AD_REGION = 12
    NEGATIVE_AD_OWNER = 13
    OPTIONAL_HAND_PLACEMENT = 14
    OPTIONAL_BANK_PLACEMENT = 15


def region_action(region: Region) -> int:
    return REGION_BASE + int(region)


def card_action(card: Card) -> int:
    return CARD_BASE + int(card)


def player_action(seat: int) -> int:
    if not 0 <= seat < MAX_PLAYERS:
        raise ValueError(f"seat {seat} out of range")
    return PLAYER_BASE + seat


def action_region(action: int) -> Region:
    if not REGION_BASE <= action < CARD_BASE:
        raise ValueError(f"action {action} is not a region")
    return Region(action - REGION_BASE)


def action_card(action: int) -> Card:
    if not CARD_BASE <= action < PLAYER_BASE:
        raise ValueError(f"action {action} is not a card")
    return Card(action - CARD_BASE)


def action_player(action: int) -> int:
    if not PLAYER_BASE <= action < YES:
        raise ValueError(f"action {action} is not a player")
    return action - PLAYER_BASE


_NAMED = {YES: "yes", NO: "no", PASS: "pass", STOP: "stop"}


def describe_action(action: int) -> str:
    if action in _NAMED:
        return _NAMED[action]
    if REGION_BASE <= action < CARD_BASE:
        return f"region:{action_region(action).name}"
    if CARD_BASE <= action < PLAYER_BASE:
        return f"card:{action_card(action).name}"
    if PLAYER_BASE <= action < YES:
        return f"player:{action_player(action)}"
    return f"invalid:{action}"


@dataclass(frozen=True)
class DecisionRequest:
    """A player must choose one of ``legal_actions``. ``card``/``region`` give context."""

    seat: int
    decision: DecisionType
    legal_actions: tuple[int, ...]
    card: Card | None = None
    region: Region | None = None


@dataclass(frozen=True)
class GameOver:
    scores: tuple[int, ...]
    hands: tuple[int, ...]
    banks: tuple[int, ...]
    winners: tuple[int, ...]
    tiebreak: TieBreak

    @property
    def draw(self) -> bool:
        return self.tiebreak is TieBreak.DRAW
