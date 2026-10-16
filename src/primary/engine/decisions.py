"""Asking a player for a decision (spec §4.2). Single-option decisions resolve automatically."""

from __future__ import annotations

from collections.abc import Generator, Iterable

from primary.data.board import Region
from primary.data.cards import Card
from primary.engine.actions import (
    DecisionRequest,
    DecisionType,
    EngineError,
    IllegalActionError,
    describe_action,
)
from primary.engine.events import EventType
from primary.engine.table import Table

Decide = Generator[DecisionRequest, int, int]
Flow = Generator[DecisionRequest, int, None]


def ask(
    table: Table,
    seat: int,
    decision: DecisionType,
    legal: Iterable[int],
    *,
    card: Card | None = None,
    region: Region | None = None,
) -> Decide:
    options = tuple(sorted(set(legal)))
    if not options:
        raise EngineError(f"{decision.name} for seat {seat} has no legal actions")
    if len(options) == 1:
        action, forced = options[0], True
    else:
        action = yield DecisionRequest(seat, decision, options, card, region)
        forced = False
        if action not in options:
            raise IllegalActionError(f"{action} is not one of {options} for {decision.name}")
    table.emit(
        EventType.DECISION,
        seat,
        decision_type=decision,
        action=action,
        forced=forced,
        card=card,
        detail=describe_action(action),
    )
    return action
