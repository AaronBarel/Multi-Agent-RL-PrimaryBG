"""Regions, borders, vote awards and the voting schedule (CLAUDE.md: Primary Game Map Environment)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType


class Region(IntEnum):
    PACIFIC_NW = 0
    WEST = 1
    SOUTH_WEST = 2
    GREATER_TEXAS = 3
    NORTH = 4
    GREAT_LAKES = 5
    EARLY_MIDWEST = 6
    MIDWEST = 7
    DEEP_SOUTH = 8
    SOUTH_EAST = 9
    SOUTH = 10
    APPALACHIA = 11
    NORTH_EAST = 12
    NEW_ENGLAND = 13


NUM_REGIONS = len(Region)
NUM_ROUNDS = 11

REGION_NAMES: Mapping[Region, str] = MappingProxyType(
    {
        Region.PACIFIC_NW: "Pacific NW",
        Region.WEST: "West",
        Region.SOUTH_WEST: "South West",
        Region.GREATER_TEXAS: "Greater Texas",
        Region.NORTH: "North",
        Region.GREAT_LAKES: "Great Lakes",
        Region.EARLY_MIDWEST: "Early Midwest",
        Region.MIDWEST: "Midwest",
        Region.DEEP_SOUTH: "Deep South",
        Region.SOUTH_EAST: "South East",
        Region.SOUTH: "South",
        Region.APPALACHIA: "Appalachia",
        Region.NORTH_EAST: "North East",
        Region.NEW_ENGLAND: "New England",
    }
)

STATES: Mapping[Region, tuple[str, ...]] = MappingProxyType(
    {
        Region.PACIFIC_NW: ("Washington", "Oregon", "Idaho"),
        Region.WEST: ("California", "Nevada", "Alaska", "Hawaii"),
        Region.SOUTH_WEST: ("Colorado", "Utah", "Arizona", "New Mexico"),
        Region.GREATER_TEXAS: ("Texas", "Oklahoma"),
        Region.NORTH: ("Montana", "Wyoming", "North Dakota", "South Dakota"),
        Region.GREAT_LAKES: ("Minnesota", "Wisconsin", "Michigan"),
        Region.EARLY_MIDWEST: ("Illinois", "Ohio", "Indiana"),
        Region.MIDWEST: ("Nebraska", "Iowa", "Missouri", "Kansas"),
        Region.DEEP_SOUTH: ("Louisiana", "Mississippi", "Arkansas"),
        Region.SOUTH_EAST: ("Florida", "Alabama", "Georgia"),
        Region.SOUTH: ("Tennessee", "North Carolina", "South Carolina"),
        Region.APPALACHIA: ("Kentucky", "West Virginia", "Virginia"),
        Region.NORTH_EAST: ("Pennsylvania", "New Jersey", "Maryland", "Delaware"),
        Region.NEW_ENGLAND: (
            "New York",
            "Vermont",
            "New Hampshire",
            "Massachusetts",
            "Connecticut",
            "Maine",
            "Rhode Island",
        ),
    }
)

_R = Region
_BORDERS: dict[Region, tuple[Region, ...]] = {
    _R.PACIFIC_NW: (_R.NORTH, _R.SOUTH_WEST, _R.WEST),
    _R.WEST: (_R.SOUTH_WEST, _R.PACIFIC_NW),
    _R.SOUTH_WEST: (_R.NORTH, _R.PACIFIC_NW, _R.WEST, _R.MIDWEST, _R.GREATER_TEXAS),
    _R.GREATER_TEXAS: (_R.SOUTH_WEST, _R.MIDWEST, _R.DEEP_SOUTH),
    _R.NORTH: (_R.PACIFIC_NW, _R.SOUTH_WEST, _R.MIDWEST, _R.GREAT_LAKES),
    _R.GREAT_LAKES: (_R.NORTH, _R.MIDWEST, _R.EARLY_MIDWEST),
    _R.EARLY_MIDWEST: (_R.MIDWEST, _R.GREAT_LAKES, _R.NORTH_EAST, _R.APPALACHIA),
    _R.MIDWEST: (
        _R.EARLY_MIDWEST,
        _R.NORTH,
        _R.SOUTH_WEST,
        _R.GREATER_TEXAS,
        _R.DEEP_SOUTH,
        _R.GREAT_LAKES,
        _R.APPALACHIA,
        _R.SOUTH,
    ),
    _R.DEEP_SOUTH: (_R.MIDWEST, _R.GREATER_TEXAS, _R.SOUTH_EAST, _R.SOUTH),
    _R.SOUTH_EAST: (_R.DEEP_SOUTH, _R.SOUTH),
    _R.SOUTH: (_R.MIDWEST, _R.DEEP_SOUTH, _R.SOUTH_EAST, _R.APPALACHIA),
    _R.APPALACHIA: (_R.MIDWEST, _R.SOUTH, _R.EARLY_MIDWEST, _R.NORTH_EAST),
    _R.NORTH_EAST: (_R.NEW_ENGLAND, _R.EARLY_MIDWEST, _R.APPALACHIA),
    _R.NEW_ENGLAND: (_R.NORTH_EAST,),
}
ADJACENCY: Mapping[Region, frozenset[Region]] = MappingProxyType(
    {region: frozenset(neighbours) for region, neighbours in _BORDERS.items()}
)

# (1st place,) for winner-take-all regions; (1st, 2nd) for regions with a second-place award.
AWARDS: Mapping[Region, tuple[int, ...]] = MappingProxyType(
    {
        _R.PACIFIC_NW: (13,),
        _R.WEST: (17, 7),
        _R.SOUTH_WEST: (11, 5),
        _R.GREATER_TEXAS: (15, 6),
        _R.NORTH: (10,),
        _R.GREAT_LAKES: (14,),
        _R.EARLY_MIDWEST: (13, 5),
        _R.MIDWEST: (15,),
        _R.DEEP_SOUTH: (11,),
        _R.SOUTH_EAST: (16, 7),
        _R.SOUTH: (12, 5),
        _R.APPALACHIA: (12,),
        _R.NORTH_EAST: (14, 6),
        _R.NEW_ENGLAND: (18, 8),
    }
)

# Round number -> number of regions voting that round.
VOTING_SLOTS: Mapping[int, int] = MappingProxyType(
    {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2, 7: 1, 8: 2, 9: 0, 10: 2, 11: 3}
)

TOTAL_VOTES = sum(sum(awards) for awards in AWARDS.values())
