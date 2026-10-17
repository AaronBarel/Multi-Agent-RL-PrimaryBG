"""The phases of a round as decision flows (spec §5.4)."""

from __future__ import annotations

from collections import Counter

from primary.data.board import Region
from primary.data.cards import CARD_COUNTS, CARDS_PER_ROUND, Card
from primary.engine.actions import (
    NO,
    PASS,
    STOP,
    YES,
    DecisionType,
    action_card,
    action_player,
    action_region,
    card_action,
    player_action,
    region_action,
)
from primary.engine.decisions import Flow, ask
from primary.engine.events import EventType, Phase, Source
from primary.engine.modifiers import NewsMove
from primary.engine.news_effects import move_options
from primary.engine.resolve import resolve_card
from primary.engine.state import GameState
from primary.engine.table import Table


def setup_phase(table: Table) -> Flow:
    """Home regions in turn order from the starting player, 2 bank cubes each (rules 5, 32)."""
    table.phase = Phase.SETUP
    for seat in table.state.turn_order():
        chosen: list[Region] = []
        for _ in range(table.mods[seat].home_regions):
            options = [region_action(r) for r in Region if r not in chosen]
            choice = yield from ask(table, seat, DecisionType.HOME_REGION, options)
            region = action_region(choice)
            if not chosen:
                table.move_pawn(seat, region, "home")
            chosen.append(region)
            table.place(seat, region, 2, Source.BANK, "home")


def news_phase(table: Table) -> Flow:
    """Draw a News card and resolve its immediate effects in turn order (rule 33)."""
    table.phase = Phase.NEWS
    table.draw_news()
    for seat in table.state.turn_order():
        mods = table.mods[seat]
        if mods.immediate_gain:
            table.gain(seat, mods.immediate_gain, "tax_reform")
        if mods.news_move is not NewsMove.NONE:
            yield from _news_move(table, seat, mods.news_move)
        if mods.news_swap:
            yield from _news_swap(table, seat)


def _news_move(table: Table, seat: int, kind: NewsMove) -> Flow:
    regions, optional = move_options(kind, table.state, seat)
    if not regions:
        return
    options = [region_action(r) for r in regions] + ([PASS] if optional else [])
    choice = yield from ask(table, seat, DecisionType.NEWS_MOVE, options)
    if choice != PASS:
        table.move_pawn(seat, action_region(choice), kind.value)


def _news_swap(table: Table, seat: int) -> Flow:
    state = table.state
    here = state.players[seat].pawn
    regions = [r for r in Region if r != here and sum(state.cubes[r]) > 0]
    if not regions:
        return
    options = [region_action(r) for r in regions] + [PASS]
    choice = yield from ask(table, seat, DecisionType.NEWS_SWAP_REGION, options)
    if choice == PASS:
        return
    region = action_region(choice)
    owners = [player_action(o) for o, count in enumerate(state.cubes[region]) if count > 0]
    pick = yield from ask(table, seat, DecisionType.NEWS_SWAP_OWNER, owners, region=region)
    table.swap_pawn_with_cube(seat, region, action_player(pick))


def available_cards(state: GameState, seat: int) -> dict[Card, int]:
    """Copies of each card still available to commit this round (rules 6, 40)."""
    used = Counter(state.rnd.queue[seat])
    frozen = state.frozen[seat]
    return {card: CARD_COUNTS[card] - frozen.get(card, 0) - used[card] for card in Card}


def commit_phase(table: Table) -> Flow:
    """Each player commits an ordered queue of 4 cards (or 4–5 under Final Push, rule 43)."""
    table.phase = Phase.COMMIT
    state = table.state
    for seat in state.turn_order():
        for slot in range(table.mods[seat].max_cards):
            left = available_cards(state, seat)
            options = [card_action(c) for c in Card if left[c] > 0]
            if slot >= CARDS_PER_ROUND:
                options.append(STOP)
            choice = yield from ask(table, seat, DecisionType.COMMIT_CARD, options)
            if choice == STOP:
                break
            card = action_card(choice)
            state.rnd.queue[seat].append(card)
            table.emit(EventType.CARD_COMMITTED, seat, card=card)


def execute_phase(table: Table) -> Flow:
    """Reveal and execute cards slot by slot in turn order (rule 7)."""
    table.phase = Phase.EXECUTE
    state = table.state
    rnd = state.rnd
    longest = max((len(q) for q in rnd.queue), default=0)
    for slot in range(longest):
        for seat in state.turn_order():
            queue = rnd.queue[seat]
            if slot >= len(queue):
                continue
            mods = table.mods[seat]
            if slot == 0 and mods.erin_premove:
                yield from _erin_premove(table, seat)
            card = queue[slot]
            rnd.revealed[seat].append(card)
            table.emit(EventType.CARD_REVEALED, seat, card=card)
            if mods.can_skip_card and rnd.skipped_slot[seat] is None:
                choice = yield from ask(table, seat, DecisionType.DIANE_SKIP, [YES, NO], card=card)
                if choice == YES:
                    rnd.skipped_slot[seat] = slot
                    table.emit(EventType.CARD_SKIPPED, seat, card=card)
                    continue
            yield from resolve_card(table, seat, card)


def _erin_premove(table: Table, seat: int) -> Flow:
    state = table.state
    own = [r for r in Region if state.cubes[r][seat] > 0]
    if not own:
        return
    options = [region_action(r) for r in own] + [PASS]
    choice = yield from ask(table, seat, DecisionType.ERIN_SOURCE, options)
    if choice == PASS:
        return
    src = action_region(choice)
    destinations = [region_action(r) for r in Region if r != src]
    dst = yield from ask(table, seat, DecisionType.ERIN_DEST, destinations, region=src)
    table.move_cube(seat, src, action_region(dst), seat, "erin", return_if_voted=False)
