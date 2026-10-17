"""Primitive state mutations. Every change to cubes, pawns and hands goes through here and is logged."""

from __future__ import annotations

from typing import Any

from primary.data.board import NUM_ROUNDS, Region
from primary.data.cards import Card
from primary.data.news import News
from primary.engine.actions import EngineError
from primary.engine.effects import build_modifiers
from primary.engine.events import Event, EventSink, EventType, Phase, Source
from primary.engine.modifiers import Modifiers
from primary.engine.state import GameState, RoundState


class Table:
    def __init__(self, state: GameState, sink: EventSink | None = None) -> None:
        self.state = state
        self.sink = sink
        self.phase = Phase.SETUP
        self.mods: list[Modifiers] = [build_modifiers(p.candidate, None) for p in state.players]
        self._seq = 0

    # -- logging -----------------------------------------------------------------------------

    def emit(self, event_type: EventType, seat: int | None = None, **fields: Any) -> None:
        seq = self._seq
        self._seq += 1
        if self.sink is None:
            return
        candidate = None if seat is None else self.state.players[seat].candidate
        self.sink.on_event(
            Event(
                seq=seq,
                round=self.state.round,
                phase=self.phase,
                event_type=event_type,
                seat=seat,
                candidate=candidate,
                **fields,
            )
        )

    def no_effect(self, seat: int, card: Card, reason: str) -> None:
        self.emit(EventType.NO_EFFECT, seat, card=card, detail=reason)

    # -- round and news ----------------------------------------------------------------------

    def set_news(self, news: News | None) -> None:
        self.state.current_news = news
        self.mods = [build_modifiers(p.candidate, news) for p in self.state.players]

    def begin_round(self, rnd: int) -> None:
        self.state.round = rnd
        self.state.rnd = RoundState.empty(self.state.players)

    def draw_news(self) -> News:
        """Draw the next News card; Frozen Assets is discarded and redrawn in round 11 (rule 40)."""
        state = self.state
        news = state.news_deck.pop(0)
        if news is News.FROZEN_ASSETS and state.round == NUM_ROUNDS:
            self.emit(EventType.NEWS_DISCARDED, news=news, detail="frozen_assets_in_final_round")
            news = state.news_deck.pop(0)
        state.news_history.append(news)
        self.emit(EventType.NEWS_DRAWN, news=news)
        self.set_news(news)
        return news

    # -- cubes -------------------------------------------------------------------------------

    def gain(self, seat: int, n: int, reason: str) -> int:
        """Move up to ``n`` cubes bank -> hand (capped by the bank, rule 16)."""
        player = self.state.players[seat]
        amount = min(n, player.bank)
        hand_before, bank_before = player.hand, player.bank
        player.bank -= amount
        player.hand += amount
        self.emit(
            EventType.GAIN,
            seat,
            amount=amount,
            source=Source.BANK,
            hand_before=hand_before,
            hand_after=player.hand,
            bank_before=bank_before,
            bank_after=player.bank,
            detail=reason,
        )
        return amount

    def pay(self, seat: int, n: int, reason: str) -> bool:
        """Move ``n`` cubes hand -> bank, or do nothing and return False if the hand is short."""
        player = self.state.players[seat]
        if player.hand < n:
            return False
        hand_before, bank_before = player.hand, player.bank
        player.hand -= n
        player.bank += n
        self.emit(
            EventType.PAYMENT,
            seat,
            amount=n,
            source=Source.HAND,
            hand_before=hand_before,
            hand_after=player.hand,
            bank_before=bank_before,
            bank_after=player.bank,
            detail=reason,
        )
        return True

    def place(self, seat: int, region: Region, n: int, source: Source, reason: str) -> bool:
        """Place ``n`` cubes from hand or bank onto ``region``; False (no change) if short."""
        player = self.state.players[seat]
        available = player.hand if source is Source.HAND else player.bank
        if available < n:
            return False
        hand_before, bank_before = player.hand, player.bank
        if source is Source.HAND:
            player.hand -= n
        else:
            player.bank -= n
        self.state.cubes[region][seat] += n
        self.emit(
            EventType.CUBE_PLACED,
            seat,
            to_region=region,
            cube_owner=seat,
            amount=n,
            source=source,
            hand_before=hand_before,
            hand_after=player.hand,
            bank_before=bank_before,
            bank_after=player.bank,
            detail=reason,
        )
        return True

    def remove(self, region: Region, owner: int, by_seat: int, reason: str) -> None:
        """Remove one of ``owner``'s cubes from ``region`` to ``owner``'s bank (rule 13)."""
        if self.state.cubes[region][owner] <= 0:
            raise EngineError(f"seat {owner} has no cube on {region.name}")
        victim = self.state.players[owner]
        bank_before = victim.bank
        self.state.cubes[region][owner] -= 1
        victim.bank += 1
        self.emit(
            EventType.CUBE_REMOVED,
            by_seat,
            from_region=region,
            cube_owner=owner,
            amount=1,
            hand_before=victim.hand,
            hand_after=victim.hand,
            bank_before=bank_before,
            bank_after=victim.bank,
            detail=reason,
        )

    def move_cube(
        self,
        owner: int,
        src: Region,
        dst: Region,
        by_seat: int,
        reason: str,
        *,
        return_if_voted: bool,
    ) -> None:
        """Move one of ``owner``'s cubes. With ``return_if_voted``, a voted ``dst`` sends it home."""
        if self.state.cubes[src][owner] <= 0:
            raise EngineError(f"seat {owner} has no cube on {src.name}")
        self.state.cubes[src][owner] -= 1
        player = self.state.players[owner]
        if return_if_voted and self.state.voted[dst]:
            bank_before = player.bank
            player.bank += 1
            self.emit(
                EventType.CUBE_RETURNED,
                by_seat,
                from_region=src,
                to_region=dst,
                cube_owner=owner,
                amount=1,
                bank_before=bank_before,
                bank_after=player.bank,
                detail=f"{reason}:already_voted",
            )
            return
        self.state.cubes[dst][owner] += 1
        self.emit(
            EventType.CUBE_MOVED,
            by_seat,
            from_region=src,
            to_region=dst,
            cube_owner=owner,
            amount=1,
            detail=reason,
        )

    def return_region(self, region: Region) -> None:
        """Return every cube on ``region`` to its owner's bank."""
        for owner, count in enumerate(self.state.cubes[region]):
            if count <= 0:
                continue
            player = self.state.players[owner]
            bank_before = player.bank
            self.state.cubes[region][owner] = 0
            player.bank += count
            self.emit(
                EventType.CUBE_RETURNED,
                owner,
                from_region=region,
                cube_owner=owner,
                amount=count,
                bank_before=bank_before,
                bank_after=player.bank,
                detail="round_end",
            )

    # -- pawns -------------------------------------------------------------------------------

    def move_pawn(self, seat: int, dst: Region, reason: str) -> None:
        player = self.state.players[seat]
        before = player.pawn
        player.pawn = dst
        self.state.rnd.path[seat].append(dst)
        self.emit(
            EventType.MOVE,
            seat,
            from_region=before,
            to_region=dst,
            pawn_before=before,
            pawn_after=dst,
            detail=reason,
        )
        bonus = self.mods[seat].enter_contested_gain
        rivals_here = any(
            count > 0 for owner, count in enumerate(self.state.cubes[dst]) if owner != seat
        )
        if bonus and rivals_here:
            self.gain(seat, bonus, "countermeasures")

    def swap_pawn_with_cube(self, seat: int, region: Region, owner: int) -> None:
        """New Constituents (rule 50): the cube goes to the pawn's region, the pawn to the cube's."""
        old = self.state.players[seat].pawn
        if old is None:
            raise EngineError(f"seat {seat} has no pawn on the map")
        self.move_cube(owner, region, old, seat, "new_constituents", return_if_voted=True)
        self.move_pawn(seat, region, "new_constituents")
