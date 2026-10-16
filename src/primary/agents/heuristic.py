"""A simple rule-based bot: pick the most valuable winnable upcoming region, go there, pile on cubes.

It reads only public information plus its own committed cards, so it is a fair baseline opponent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from primary.data.board import AWARDS, Region
from primary.data.cards import CARD_COUNTS, CARDS_PER_ROUND, Card
from primary.engine.actions import (
    CARD_BASE,
    NO,
    PASS,
    PLAYER_BASE,
    REGION_BASE,
    STOP,
    YES,
    DecisionRequest,
    DecisionType,
    action_player,
    action_region,
    card_action,
    player_action,
    region_action,
)
from primary.engine.game import Game
from primary.engine.modifiers import Modifiers
from primary.engine.movement import distance, reachable
from primary.engine.state import GameState

Chooser = Callable[[GameState, Modifiers, DecisionRequest], int]


def upcoming_value(state: GameState, region: Region) -> float:
    """Votes on offer, discounted by how many rounds away the vote is (0 once voted)."""
    if state.voted[region]:
        return 0.0
    wait = state.vote_round[region] - max(state.round, 1)
    if wait < 0:
        return 0.0
    return sum(AWARDS[region]) / (1 + wait)


def region_value(state: GameState, seat: int, region: Region) -> float:
    mine = state.cubes[region][seat]
    rival = max((state.cubes[region][o] for o in range(state.num_players) if o != seat), default=0)
    gap = max(0, rival - mine + 1)
    return upcoming_value(state, region) / (1 + gap)


def best_region(state: GameState, seat: int, regions: Iterable[Region]) -> Region:
    return max(regions, key=lambda r: (region_value(state, seat, r), -int(r)))


def plan_cards(state: GameState, seat: int, mods: Modifiers) -> list[Card]:
    """The 4 cards to commit: travel to the target if needed, then Rally/Fundraise/Advertise."""
    player = state.players[seat]
    target = best_region(state, seat, Region)
    left = {c: CARD_COUNTS[c] - state.frozen[seat].get(c, 0) for c in Card}
    plan: list[Card] = []
    hand = player.hand

    def take(card: Card) -> bool:
        if left[card] <= 0:
            return False
        left[card] -= 1
        plan.append(card)
        return True

    if player.pawn is not None and player.pawn != target:
        moved = target in reachable(player.pawn, mods.bus_range) and take(Card.BUS)
        if not moved and hand >= mods.plane_cost and take(Card.PLANE):
            hand -= mods.plane_cost
    while len(plan) < CARDS_PER_ROUND:
        if hand >= 2 and take(Card.RALLY):
            hand -= 2
        elif hand < 2 and take(Card.FUNDRAISER):
            hand += 1
        elif hand >= 1 and take(Card.POSITIVE_AD):
            hand -= 1
        elif take(Card.SUPER_PAC):
            pass
        elif take(Card.FUNDRAISER):
            hand += 1
        else:
            break
    return plan


def _regions(request: DecisionRequest) -> list[Region]:
    return [action_region(a) for a in request.legal_actions if REGION_BASE <= a < CARD_BASE]


def _choose_commit(state: GameState, mods: Modifiers, request: DecisionRequest) -> int:
    slot = len(state.rnd.queue[request.seat])
    plan = plan_cards(state, request.seat, mods)
    return card_action(plan[slot]) if slot < len(plan) else STOP


def _choose_best_legal_region(state: GameState, mods: Modifiers, request: DecisionRequest) -> int:
    return region_action(best_region(state, request.seat, _regions(request)))


def _choose_move(state: GameState, mods: Modifiers, request: DecisionRequest) -> int:
    target = best_region(state, request.seat, Region)
    regions = _regions(request)
    if target in regions:
        return region_action(target)
    if PASS in request.legal_actions:
        return PASS
    return region_action(min(regions, key=lambda r: (distance(r, target), int(r))))


def _choose_negative_region(state: GameState, mods: Modifiers, request: DecisionRequest) -> int:
    seat, regions = request.seat, _regions(request)
    rivals = range(state.num_players)
    contested = [r for r in regions if any(state.cubes[r][o] for o in rivals if o != seat)]
    if contested:
        return region_action(max(contested, key=lambda r: (upcoming_value(state, r), -int(r))))
    return region_action(min(regions, key=lambda r: (upcoming_value(state, r), int(r))))


def _choose_negative_owner(state: GameState, mods: Modifiers, request: DecisionRequest) -> int:
    seat, region = request.seat, request.region
    assert region is not None
    owners = [action_player(a) for a in request.legal_actions if PLAYER_BASE <= a < YES]
    rivals = [o for o in owners if o != seat]
    pick = max(rivals, key=lambda o: (state.cubes[region][o], -o)) if rivals else seat
    return player_action(pick)


def _choose_skip(state: GameState, mods: Modifiers, request: DecisionRequest) -> int:
    player = state.players[request.seat]
    if request.card is Card.RALLY:
        from_bank = mods.first_rally_from_bank and state.rnd.rallies[request.seat] == 0
        if (player.bank if from_bank else player.hand) < 2:
            return YES
    if request.card is Card.PLANE and player.hand < mods.plane_cost:
        return YES
    return NO


def _choose_spin(state: GameState, mods: Modifiers, request: DecisionRequest) -> int:
    has_cube = state.players[request.seat].hand >= 1
    return card_action(Card.POSITIVE_AD if has_cube else Card.NEGATIVE_AD)


def _always_yes(state: GameState, mods: Modifiers, request: DecisionRequest) -> int:
    return YES


def _always_pass(state: GameState, mods: Modifiers, request: DecisionRequest) -> int:
    return PASS


_CHOOSERS: dict[DecisionType, Chooser] = {
    DecisionType.HOME_REGION: _choose_best_legal_region,
    DecisionType.NEWS_MOVE: _choose_move,
    DecisionType.NEWS_SWAP_REGION: _always_pass,
    DecisionType.COMMIT_CARD: _choose_commit,
    DecisionType.ERIN_SOURCE: _always_pass,
    DecisionType.DIANE_SKIP: _choose_skip,
    DecisionType.PLANE_TARGET: _choose_move,
    DecisionType.BUS_TARGET: _choose_move,
    DecisionType.SPIN_DOCTOR_TYPE: _choose_spin,
    DecisionType.POSITIVE_AD_TARGET: _choose_best_legal_region,
    DecisionType.NEGATIVE_AD_REGION: _choose_negative_region,
    DecisionType.NEGATIVE_AD_OWNER: _choose_negative_owner,
    DecisionType.OPTIONAL_HAND_PLACEMENT: _always_yes,
    DecisionType.OPTIONAL_BANK_PLACEMENT: _always_yes,
}


class HeuristicAgent:
    name = "heuristic"

    def act(self, game: Game, request: DecisionRequest) -> int:
        chooser = _CHOOSERS.get(request.decision)
        if chooser is not None:
            action = chooser(game.state, game.table.mods[request.seat], request)
            if action in request.legal_actions:
                return action
        return request.legal_actions[0]
