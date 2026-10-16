"""Structured events emitted by the engine (spec §6). Logging never changes game behaviour."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from primary.data.board import Region
from primary.data.candidates import Candidate
from primary.data.cards import Card
from primary.data.news import News
from primary.engine.actions import DecisionType
from primary.engine.voting import Placement


class Phase(StrEnum):
    SETUP = "setup"
    NEWS = "news"
    COMMIT = "commit"
    EXECUTE = "execute"
    ROUND_END = "round_end"


class EventType(StrEnum):
    DECISION = "decision"
    NEWS_DRAWN = "news_drawn"
    NEWS_DISCARDED = "news_discarded"
    CARD_COMMITTED = "card_committed"
    CARD_REVEALED = "card_revealed"
    CARD_SKIPPED = "card_skipped"
    MOVE = "move"
    CUBE_PLACED = "cube_placed"
    CUBE_REMOVED = "cube_removed"
    CUBE_MOVED = "cube_moved"
    CUBE_RETURNED = "cube_returned"
    GAIN = "gain"
    PAYMENT = "payment"
    NO_EFFECT = "no_effect"
    VOTE_AWARDED = "vote_awarded"


class Source(StrEnum):
    HAND = "hand"
    BANK = "bank"


@dataclass(frozen=True)
class Event:
    """One atomic change. ``hand_*``/``bank_*`` describe ``cube_owner`` if set, else ``seat``."""

    seq: int
    round: int
    phase: Phase
    event_type: EventType
    seat: int | None = None
    candidate: Candidate | None = None
    card: Card | None = None
    news: News | None = None
    from_region: Region | None = None
    to_region: Region | None = None
    cube_owner: int | None = None
    amount: int | None = None
    source: Source | None = None
    hand_before: int | None = None
    hand_after: int | None = None
    bank_before: int | None = None
    bank_after: int | None = None
    pawn_before: Region | None = None
    pawn_after: Region | None = None
    decision_type: DecisionType | None = None
    action: int | None = None
    forced: bool | None = None
    detail: str | None = None


@dataclass(frozen=True)
class RegionOutcome:
    region: Region
    placement: Placement
    votes: int


@dataclass(frozen=True)
class RoundSummary:
    round: int
    seat: int
    candidate: Candidate
    start_region: Region | None
    end_region: Region | None
    path: tuple[Region, ...]
    hand_start: int
    hand_end: int
    bank_start: int
    bank_end: int
    cards: tuple[Card, ...]
    skipped_slot: int | None
    outcomes: tuple[RegionOutcome, ...]
    votes_gained: int


class EventSink(Protocol):
    def on_event(self, event: Event) -> None: ...

    def on_round(self, summary: RoundSummary) -> None: ...


class ListSink:
    """Keeps everything in memory. Used by tests and small tools."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.rounds: list[RoundSummary] = []

    def on_event(self, event: Event) -> None:
        self.events.append(event)

    def on_round(self, summary: RoundSummary) -> None:
        self.rounds.append(summary)

    def of_type(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.event_type is event_type]
