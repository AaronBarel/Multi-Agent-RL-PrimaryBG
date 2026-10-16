"""Action cards (CLAUDE.md: Action Cards)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType


class Card(IntEnum):
    SUPER_PAC = 0
    FUNDRAISER = 1
    PLANE = 2
    BUS = 3
    POSITIVE_AD = 4
    NEGATIVE_AD = 5
    RALLY = 6


CARD_COUNTS: Mapping[Card, int] = MappingProxyType(
    {
        Card.SUPER_PAC: 4,
        Card.FUNDRAISER: 3,
        Card.PLANE: 1,
        Card.BUS: 1,
        Card.POSITIVE_AD: 1,
        Card.NEGATIVE_AD: 1,
        Card.RALLY: 3,
    }
)

CARDS_PER_ROUND = 4

CARD_TEXT: Mapping[Card, str] = MappingProxyType(
    {
        Card.SUPER_PAC: "If you play the most Super PAC cards, gain 4 Influence cubes. If tied, gain 2",
        Card.FUNDRAISER: "Gain 1 Influence cube",
        Card.PLANE: "Pay 1 Influence cube and move to any region",
        Card.BUS: "Move to any adjacent region",
        Card.POSITIVE_AD: "Place 1 Influence cube onto any region",
        Card.NEGATIVE_AD: "Remove any 1 Influence cube from any region",
        Card.RALLY: "Place 2 Influence cubes onto your current region",
    }
)
