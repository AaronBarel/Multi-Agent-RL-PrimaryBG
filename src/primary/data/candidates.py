"""Candidates, their abilities and starting hands (CLAUDE.md: Candidate Information and Abilities)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType

TOTAL_CUBES = 30
MIN_PLAYERS = 3
MAX_PLAYERS = 6


class Candidate(IntEnum):
    ANTONIO_PARADES = 0
    KENNETH_WALKER = 1
    BRYCE_MCKAY = 2
    KRISTEN_RAMIREZ = 3
    DIANE_PROUDIAN = 4
    RYAN_GUPTA = 5
    CHESTER_FREEMAN = 6
    LILY_NGUYEN = 7
    ERIN_FIELDS = 8
    CALVIN_MATSUI = 9
    ANNA_PATEL = 10
    GEORGIA_BARNES = 11


_C = Candidate

STARTING_HAND: Mapping[Candidate, int] = MappingProxyType(
    {
        _C.ANTONIO_PARADES: 5,
        _C.KENNETH_WALKER: 2,
        _C.BRYCE_MCKAY: 3,
        _C.KRISTEN_RAMIREZ: 3,
        _C.DIANE_PROUDIAN: 3,
        _C.RYAN_GUPTA: 1,
        _C.CHESTER_FREEMAN: 3,
        _C.LILY_NGUYEN: 2,
        _C.ERIN_FIELDS: 5,
        _C.CALVIN_MATSUI: 3,
        _C.ANNA_PATEL: 3,
        _C.GEORGIA_BARNES: 1,
    }
)

DISPLAY_NAME: Mapping[Candidate, str] = MappingProxyType(
    {c: c.name.replace("_", " ").title().replace("Mckay", "McKay") for c in Candidate}
)

ABILITY_TEXT: Mapping[Candidate, str] = MappingProxyType(
    {
        _C.ANTONIO_PARADES: "Bus cards allow you to move up to 2 adjacent regions away",
        _C.KENNETH_WALKER: "If you play 2 or more Fundraiser cards in a round, gain 1 extra cube",
        _C.BRYCE_MCKAY: "After each vote where you do not win any delegates, gain 1 cube",
        _C.KRISTEN_RAMIREZ: "If you play the most Super PAC cards in a round, gain 1 extra cube",
        _C.DIANE_PROUDIAN: "Each round you may choose to not execute 1 Action card in your queue",
        _C.RYAN_GUPTA: "Plane cards do not cost you an Influence cube",
        _C.CHESTER_FREEMAN: "If you end a round in the same region as any opponent, gain 1 cube",
        _C.LILY_NGUYEN: "Your first Rally each round may place 1 extra cube from your hand",
        _C.ERIN_FIELDS: "Before your first Action card each round, you may move 1 of your cubes",
        _C.CALVIN_MATSUI: "Bus cards allow you to place 1 cube from your hand on the destination",
        _C.ANNA_PATEL: "Your character token counts as 2 Influence cubes in its current region",
        _C.GEORGIA_BARNES: "Start with 3 home regions and place 2 bank cubes in each",
    }
)
