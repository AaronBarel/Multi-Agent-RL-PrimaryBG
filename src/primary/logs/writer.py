"""Buffered Parquet logging of games, events and per-round summaries (spec §6)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from primary import ENGINE_VERSION
from primary.engine.events import Event, RoundSummary
from primary.engine.game import Game
from primary.engine.voting import Placement
from primary.logs.schema import SCHEMAS, TABLE_NAMES

Row = dict[str, Any]


def _name(value: Enum | None) -> str | None:
    return None if value is None else value.name


def _value(value: Enum | None) -> str | None:
    return None if value is None else str(value.value)


def event_row(game_id: int, e: Event) -> Row:
    return {
        "game_id": game_id,
        "seq": e.seq,
        "round": e.round,
        "phase": _value(e.phase),
        "event_type": _value(e.event_type),
        "seat": e.seat,
        "candidate": _name(e.candidate),
        "card": _name(e.card),
        "news": _name(e.news),
        "from_region": _name(e.from_region),
        "to_region": _name(e.to_region),
        "cube_owner": e.cube_owner,
        "amount": e.amount,
        "source": _value(e.source),
        "hand_before": e.hand_before,
        "hand_after": e.hand_after,
        "bank_before": e.bank_before,
        "bank_after": e.bank_after,
        "pawn_before": _name(e.pawn_before),
        "pawn_after": _name(e.pawn_after),
        "decision_type": _name(e.decision_type),
        "action": e.action,
        "forced": e.forced,
        "detail": e.detail,
    }


def round_result(summary: RoundSummary) -> str:
    if not summary.outcomes:
        return "no_vote"
    placements = {o.placement for o in summary.outcomes}
    if placements & {Placement.FIRST, Placement.SECOND}:
        return "won"
    if placements == {Placement.ABSENT}:
        return "not_participated"
    return "lost"


def round_row(game_id: int, s: RoundSummary) -> Row:
    return {
        "game_id": game_id,
        "round": s.round,
        "seat": s.seat,
        "candidate": s.candidate.name,
        "start_region": _name(s.start_region),
        "end_region": _name(s.end_region),
        "path": [r.name for r in s.path],
        "hand_start": s.hand_start,
        "hand_end": s.hand_end,
        "bank_start": s.bank_start,
        "bank_end": s.bank_end,
        "cards": [c.name for c in s.cards],
        "skipped_slot": s.skipped_slot,
        "vote_regions": [o.region.name for o in s.outcomes],
        "vote_placements": [o.placement.value for o in s.outcomes],
        "vote_votes": [o.votes for o in s.outcomes],
        "votes_gained": s.votes_gained,
        "round_result": round_result(s),
    }


def game_row(game_id: int, game: Game, policies: Sequence[str]) -> Row:
    result, seed, state = game.result, game.seed, game.state
    if result is None or seed is None:
        raise ValueError(f"game {game_id} has not finished")
    pairs = [(rnd, region) for rnd in sorted(state.schedule) for region in state.schedule[rnd]]
    return {
        "game_id": game_id,
        "seed": seed,
        "engine_version": ENGINE_VERSION,
        "num_players": state.num_players,
        "candidates": [p.candidate.name for p in state.players],
        "policies": list(policies),
        "start_seat": state.initial_start_seat,
        "schedule_rounds": [rnd for rnd, _ in pairs],
        "schedule_regions": [region.name for _, region in pairs],
        "news_order": [n.name for n in state.news_history],
        "final_scores": list(result.scores),
        "final_hands": list(result.hands),
        "final_banks": list(result.banks),
        "winners": list(result.winners),
        "draw": result.draw,
        "tiebreak": result.tiebreak.value,
        "num_decisions": len(game.history),
    }


class GameRecorder:
    """An ``EventSink`` that turns one game's events into table rows."""

    def __init__(self, game_id: int) -> None:
        self.game_id = game_id
        self.events: list[Row] = []
        self.rounds: list[Row] = []

    def on_event(self, event: Event) -> None:
        self.events.append(event_row(self.game_id, event))

    def on_round(self, summary: RoundSummary) -> None:
        self.rounds.append(round_row(self.game_id, summary))


class RunLogger:
    """Collects finished games and appends them to ``games/events/rounds.parquet`` in batches."""

    def __init__(self, out_dir: Path | str, flush_every: int = 100) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._buffers: dict[str, list[Row]] = {name: [] for name in TABLE_NAMES}
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._unflushed = 0

    def recorder(self, game_id: int) -> GameRecorder:
        return GameRecorder(game_id)

    def finish(self, recorder: GameRecorder, game: Game, policies: Sequence[str]) -> None:
        self._buffers["events"].extend(recorder.events)
        self._buffers["rounds"].extend(recorder.rounds)
        self._buffers["games"].append(game_row(recorder.game_id, game, policies))
        self._unflushed += 1
        if self._unflushed >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        for name, rows in self._buffers.items():
            if not rows:
                continue
            table = pa.Table.from_pylist(rows, schema=SCHEMAS[name])
            writer = self._writers.get(name)
            if writer is None:
                writer = pq.ParquetWriter(self.out_dir / f"{name}.parquet", SCHEMAS[name])
                self._writers[name] = writer
            writer.write_table(table)
            rows.clear()
        self._unflushed = 0

    def close(self) -> None:
        self.flush()
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
