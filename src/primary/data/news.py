"""News cards (CLAUDE.md: News Cards)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType


class News(IntEnum):
    COUNTERMEASURES = 0
    OPPO_RESEARCH = 1
    GENEROUS_DONORS = 2
    TAX_REFORM = 3
    LAST_MINUTE_DEBATE = 4
    DETOUR = 5
    FROZEN_ASSETS = 6
    PAPARAZZI = 7
    SPIN_DOCTOR = 8
    FINAL_PUSH = 9
    LONG_TERM_PLANNING = 10
    DEREGULATION = 11
    EXTRA_AIRTIME = 12
    WALL_STREET_SUPPORT = 13
    ROAD_TRIP = 14
    PLUTOCRACY = 15
    NEW_CONSTITUENTS = 16
    FREQUENT_FLYER = 17


_N = News

NEWS_TEXT: Mapping[News, str] = MappingProxyType(
    {
        _N.COUNTERMEASURES: "Gain 1 cube each time you enter a region where any opponent has Influence",
        _N.OPPO_RESEARCH: "Players ending the round in the same region as an opponent gain 2 cubes",
        _N.GENEROUS_DONORS: "Gain 1 extra cube for each Fundraiser card played",
        _N.TAX_REFORM: "All players immediately gain 2 cubes",
        _N.LAST_MINUTE_DEBATE: "May move (free) to a region voting this round or next round",
        _N.DETOUR: "Must move (free) to a new region",
        _N.FROZEN_ASSETS: "Cards played this round cannot be used next round (redraw in round 11)",
        _N.PAPARAZZI: "Bus and Plane cards may place 1 bank cube on the destination",
        _N.SPIN_DOCTOR: "Positive Ad and Negative Ad cards may be used as either card type",
        _N.FINAL_PUSH: "All players may play 5 Action cards",
        _N.LONG_TERM_PLANNING: "May move (free) to a region voting in round 11",
        _N.DEREGULATION: "Most Fundraisers gains 2 cubes; tied (non-zero) players gain 1",
        _N.EXTRA_AIRTIME: "Positive Ad places 2 cubes from hand onto any region(s)",
        _N.WALL_STREET_SUPPORT: "Your first Rally uses 2 bank cubes instead of hand cubes",
        _N.ROAD_TRIP: "Bus cards move 1 adjacent region further",
        _N.PLUTOCRACY: "Most Super PACs gains 6 cubes; tied (non-zero) players gain 3",
        _N.NEW_CONSTITUENTS: "May swap your pawn with any Influence cube on the map",
        _N.FREQUENT_FLYER: "Plane cards do not cost an Influence cube",
    }
)
