"""Resolution of one Action card (spec §3 rules 8–14, 21, 26, 28, 30, 36, 41–42, 46–48, 51)."""

from __future__ import annotations

from primary.data.board import Region
from primary.data.cards import Card
from primary.engine.actions import (
    NO,
    YES,
    DecisionType,
    EngineError,
    action_card,
    action_player,
    action_region,
    card_action,
    player_action,
    region_action,
)
from primary.engine.decisions import Flow, ask
from primary.engine.events import Source
from primary.engine.movement import reachable
from primary.engine.table import Table


def resolve_card(table: Table, seat: int, card: Card) -> Flow:
    if card is Card.SUPER_PAC:
        table.state.rnd.super_pacs[seat] += 1  # paid out at round end
    elif card is Card.FUNDRAISER:
        _fundraiser(table, seat)
    elif card is Card.PLANE:
        yield from _plane(table, seat)
    elif card is Card.BUS:
        yield from _bus(table, seat)
    elif card in (Card.POSITIVE_AD, Card.NEGATIVE_AD):
        yield from _advertise(table, seat, card)
    elif card is Card.RALLY:
        yield from _rally(table, seat)


def _pawn(table: Table, seat: int) -> Region:
    pawn = table.state.players[seat].pawn
    if pawn is None:
        raise EngineError(f"seat {seat} has no pawn on the map")
    return pawn


def _fundraiser(table: Table, seat: int) -> None:
    table.state.rnd.fundraisers[seat] += 1
    table.gain(seat, 1, "fundraiser")
    extra = table.mods[seat].fundraiser_extra
    if extra:
        table.gain(seat, extra, "generous_donors")


def _optional_placement(
    table: Table, seat: int, card: Card, region: Region, source: Source, reason: str
) -> Flow:
    player = table.state.players[seat]
    available = player.hand if source is Source.HAND else player.bank
    if available < 1:
        return
    decision = (
        DecisionType.OPTIONAL_HAND_PLACEMENT
        if source is Source.HAND
        else DecisionType.OPTIONAL_BANK_PLACEMENT
    )
    choice = yield from ask(table, seat, decision, [YES, NO], card=card, region=region)
    if choice == YES:
        table.place(seat, region, 1, source, reason)


def _plane(table: Table, seat: int) -> Flow:
    mods = table.mods[seat]
    if mods.plane_cost and not table.pay(seat, mods.plane_cost, "plane"):
        table.no_effect(seat, Card.PLANE, "insufficient_hand")
        return
    here = _pawn(table, seat)
    options = [region_action(r) for r in Region if r != here]
    choice = yield from ask(table, seat, DecisionType.PLANE_TARGET, options, card=Card.PLANE)
    dst = action_region(choice)
    table.move_pawn(seat, dst, "plane")
    if mods.move_bank_placement:
        yield from _optional_placement(table, seat, Card.PLANE, dst, Source.BANK, "paparazzi")


def _bus(table: Table, seat: int) -> Flow:
    mods = table.mods[seat]
    options = [region_action(r) for r in reachable(_pawn(table, seat), mods.bus_range)]
    choice = yield from ask(table, seat, DecisionType.BUS_TARGET, options, card=Card.BUS)
    dst = action_region(choice)
    table.move_pawn(seat, dst, "bus")
    if mods.bus_hand_placement:
        yield from _optional_placement(table, seat, Card.BUS, dst, Source.HAND, "calvin")
    if mods.move_bank_placement:
        yield from _optional_placement(table, seat, Card.BUS, dst, Source.BANK, "paparazzi")


def _advertise(table: Table, seat: int, card: Card) -> Flow:
    effect = card
    if table.mods[seat].spin_doctor:
        both = [card_action(Card.POSITIVE_AD), card_action(Card.NEGATIVE_AD)]
        choice = yield from ask(table, seat, DecisionType.SPIN_DOCTOR_TYPE, both, card=card)
        effect = action_card(choice)
    if effect is Card.POSITIVE_AD:
        yield from _positive_ad(table, seat, card)
    else:
        yield from _negative_ad(table, seat, card)


def _positive_ad(table: Table, seat: int, card: Card) -> Flow:
    player = table.state.players[seat]
    for _ in range(table.mods[seat].positive_ad_placements):
        if player.hand < 1:
            table.no_effect(seat, card, "insufficient_hand")
            return
        options = [region_action(r) for r in Region]
        choice = yield from ask(table, seat, DecisionType.POSITIVE_AD_TARGET, options, card=card)
        table.place(seat, action_region(choice), 1, Source.HAND, "positive_ad")


def _negative_ad(table: Table, seat: int, card: Card) -> Flow:
    cubes = table.state.cubes
    regions = [r for r in Region if sum(cubes[r]) > 0]
    if not regions:
        table.no_effect(seat, card, "empty_map")
        return
    choice = yield from ask(
        table, seat, DecisionType.NEGATIVE_AD_REGION, [region_action(r) for r in regions], card=card
    )
    region = action_region(choice)
    owners = [player_action(o) for o, count in enumerate(cubes[region]) if count > 0]
    pick = yield from ask(
        table, seat, DecisionType.NEGATIVE_AD_OWNER, owners, card=card, region=region
    )
    table.remove(region, action_player(pick), by_seat=seat, reason="negative_ad")


def _rally(table: Table, seat: int) -> Flow:
    rnd, mods = table.state.rnd, table.mods[seat]
    first = rnd.rallies[seat] == 0
    rnd.rallies[seat] += 1
    source = Source.BANK if first and mods.first_rally_from_bank else Source.HAND
    here = _pawn(table, seat)
    if not table.place(seat, here, 2, source, "rally"):
        table.no_effect(seat, Card.RALLY, f"insufficient_{source.value}")
        return
    if mods.first_rally_extra and not rnd.rally_extra_used[seat]:
        rnd.rally_extra_used[seat] = True
        yield from _optional_placement(table, seat, Card.RALLY, here, Source.HAND, "lily")
