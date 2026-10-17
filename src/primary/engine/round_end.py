"""End-of-round resolution (spec §5.4 step 4; rules 2, 8, 15, 19, 22–24, 27, 35, 40, 45, 49)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from enum import Enum

from primary.data.board import AWARDS, Region
from primary.engine.actions import GameOver
from primary.engine.events import EventType, Phase, RegionOutcome, RoundSummary
from primary.engine.modifiers import Modifiers
from primary.engine.state import GameState
from primary.engine.table import Table
from primary.engine.voting import Placement, RegionResult, compute_awards, determine_winners

SUPER_PAC_AWARD = (4, 2)  # (unique most, tied most)
_WON = (Placement.FIRST, Placement.SECOND)


class Tier(Enum):
    NONE = 0
    UNIQUE = 1
    TIED = 2


def most_tiers(counts: Sequence[int]) -> list[Tier]:
    """Who played the most of something. Zero is never 'the most'."""
    best = max(counts, default=0)
    if best <= 0:
        return [Tier.NONE] * len(counts)
    tier = Tier.UNIQUE if list(counts).count(best) == 1 else Tier.TIED
    return [tier if c == best else Tier.NONE for c in counts]


def influence(state: GameState, mods: Sequence[Modifiers], region: Region) -> list[int]:
    """Cubes on ``region`` per seat, plus Anna's token (rule 19)."""
    return [
        state.cubes[region][seat] + (mods[seat].token_influence if p.pawn == region else 0)
        for seat, p in enumerate(state.players)
    ]


def resolve_round_end(table: Table) -> None:
    table.phase = Phase.ROUND_END
    _card_bonuses(table)
    results = _vote(table)
    _no_award_bonus(table, results)
    _colocation_bonus(table)
    _return_cubes(table)
    _summarize(table, results)
    _freeze_cards(table)
    state = table.state
    state.start_seat = (state.start_seat + 1) % state.num_players


def finish_game(state: GameState) -> GameOver:
    scores = tuple(p.score for p in state.players)
    hands = tuple(p.hand for p in state.players)
    banks = tuple(p.bank for p in state.players)
    winners, tiebreak = determine_winners(scores, hands, banks)
    return GameOver(scores, hands, banks, winners, tiebreak)


def _bonus(table: Table, seat: int, amount: int, reason: str) -> None:
    if amount > 0:
        table.gain(seat, amount, reason)


def _card_bonuses(table: Table) -> None:
    state = table.state
    pacs = most_tiers(state.rnd.super_pacs)
    funds = most_tiers(state.rnd.fundraisers)
    for seat in state.turn_order():
        m = table.mods[seat]
        if pacs[seat] is Tier.UNIQUE:
            _bonus(table, seat, SUPER_PAC_AWARD[0], "super_pac")
            _bonus(table, seat, m.extra_super_pac_award[0], "plutocracy")
            _bonus(table, seat, m.unique_most_super_pac_bonus, "kristen")
        elif pacs[seat] is Tier.TIED:
            _bonus(table, seat, SUPER_PAC_AWARD[1], "super_pac_tie")
            _bonus(table, seat, m.extra_super_pac_award[1], "plutocracy_tie")
        if funds[seat] is Tier.UNIQUE:
            _bonus(table, seat, m.fundraiser_most_award[0], "deregulation")
        elif funds[seat] is Tier.TIED:
            _bonus(table, seat, m.fundraiser_most_award[1], "deregulation_tie")
        if state.rnd.fundraisers[seat] >= 2:
            _bonus(table, seat, m.two_fundraiser_bonus, "kenneth")


def _vote(table: Table) -> dict[Region, RegionResult]:
    state = table.state
    results: dict[Region, RegionResult] = {}
    for region in state.schedule.get(state.round, ()):
        result = compute_awards(influence(state, table.mods, region), AWARDS[region])
        for seat in state.turn_order():
            votes = result.votes[seat]
            if votes:
                state.players[seat].score += votes
                table.emit(
                    EventType.VOTE_AWARDED,
                    seat,
                    to_region=region,
                    amount=votes,
                    detail=result.placements[seat].value,
                )
        state.voted[region] = True
        results[region] = result
    return results


def _no_award_bonus(table: Table, results: dict[Region, RegionResult]) -> None:
    if not results:
        return
    for seat in table.state.turn_order():
        if all(r.placements[seat] not in _WON for r in results.values()):
            _bonus(table, seat, table.mods[seat].no_award_gain, "bryce")


def _colocation_bonus(table: Table) -> None:
    players = table.state.players
    for seat in table.state.turn_order():
        pawn = players[seat].pawn
        shared = any(o != seat and other.pawn == pawn for o, other in enumerate(players))
        if pawn is not None and shared:
            _bonus(table, seat, table.mods[seat].colocation_gain, "colocation")


def _return_cubes(table: Table) -> None:
    state = table.state
    for region in Region:
        if state.voted[region] and any(state.cubes[region]):
            table.return_region(region)


def _summarize(table: Table, results: dict[Region, RegionResult]) -> None:
    if table.sink is None:
        return
    state, rnd = table.state, table.state.rnd
    for seat, player in enumerate(state.players):
        outcomes = tuple(
            RegionOutcome(region, r.placements[seat], r.votes[seat])
            for region, r in results.items()
        )
        path = tuple(rnd.path[seat])
        table.sink.on_round(
            RoundSummary(
                round=state.round,
                seat=seat,
                candidate=player.candidate,
                start_region=path[0] if path else None,
                end_region=player.pawn,
                path=path,
                hand_start=rnd.hand_start[seat],
                hand_end=player.hand,
                bank_start=rnd.bank_start[seat],
                bank_end=player.bank,
                cards=tuple(rnd.queue[seat]),
                skipped_slot=rnd.skipped_slot[seat],
                outcomes=outcomes,
                votes_gained=sum(o.votes for o in outcomes),
            )
        )


def _freeze_cards(table: Table) -> None:
    state = table.state
    for seat in range(state.num_players):
        frozen = table.mods[seat].freeze_cards
        state.frozen[seat] = dict(Counter(state.rnd.queue[seat])) if frozen else {}
