"""Candidate abilities as Modifiers edits (spec §3 rules 21–32)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from primary.data.candidates import Candidate
from primary.engine.modifiers import Modifiers

ModifierFn = Callable[[Modifiers], None]


def antonio_parades(m: Modifiers) -> None:
    """Bus may move up to 2 adjacent regions away."""
    m.bus_range += 1


def kenneth_walker(m: Modifiers) -> None:
    """2+ Fundraisers in a round: gain 1 extra cube."""
    m.two_fundraiser_bonus += 1


def bryce_mckay(m: Modifiers) -> None:
    """After a vote where you win no delegates: gain 1 cube."""
    m.no_award_gain += 1


def kristen_ramirez(m: Modifiers) -> None:
    """Unique most Super PACs: gain 1 extra cube."""
    m.unique_most_super_pac_bonus += 1


def diane_proudian(m: Modifiers) -> None:
    """May skip executing 1 Action card per round."""
    m.can_skip_card = True


def ryan_gupta(m: Modifiers) -> None:
    """Plane cards are free."""
    m.plane_cost = 0


def chester_freeman(m: Modifiers) -> None:
    """End the round with an opponent's pawn: gain 1 cube."""
    m.colocation_gain += 1


def lily_nguyen(m: Modifiers) -> None:
    """First effective Rally may place 1 extra cube from hand."""
    m.first_rally_extra = True


def erin_fields(m: Modifiers) -> None:
    """Before the first Action card, may move 1 own cube anywhere."""
    m.erin_premove = True


def calvin_matsui(m: Modifiers) -> None:
    """Bus may place 1 cube from hand onto the destination."""
    m.bus_hand_placement = True


def anna_patel(m: Modifiers) -> None:
    """Pawn counts as 2 influence in its region."""
    m.token_influence += 2


def georgia_barnes(m: Modifiers) -> None:
    """Three home regions with 2 bank cubes each."""
    m.home_regions = 3


ABILITY_MODIFIERS: Mapping[Candidate, ModifierFn] = MappingProxyType(
    {
        Candidate.ANTONIO_PARADES: antonio_parades,
        Candidate.KENNETH_WALKER: kenneth_walker,
        Candidate.BRYCE_MCKAY: bryce_mckay,
        Candidate.KRISTEN_RAMIREZ: kristen_ramirez,
        Candidate.DIANE_PROUDIAN: diane_proudian,
        Candidate.RYAN_GUPTA: ryan_gupta,
        Candidate.CHESTER_FREEMAN: chester_freeman,
        Candidate.LILY_NGUYEN: lily_nguyen,
        Candidate.ERIN_FIELDS: erin_fields,
        Candidate.CALVIN_MATSUI: calvin_matsui,
        Candidate.ANNA_PATEL: anna_patel,
        Candidate.GEORGIA_BARNES: georgia_barnes,
    }
)
