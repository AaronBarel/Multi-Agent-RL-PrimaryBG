"""Rebuild a logged game from its seed, candidates and requested decisions (spec §6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from primary.data.candidates import Candidate
from primary.engine.actions import DecisionRequest, GameOver
from primary.engine.events import EventSink, EventType
from primary.engine.game import Game


class ReplayError(RuntimeError):
    """The log and the engine disagree about how the game went."""


def load_decisions(run_dir: Path | str, game_id: int) -> tuple[dict[str, Any], list[int]]:
    run_dir = Path(run_dir)
    games = pq.read_table(
        run_dir / "games.parquet", filters=[("game_id", "==", game_id)]
    ).to_pylist()
    if not games:
        raise KeyError(f"game {game_id} was not logged in {run_dir}")
    events = pq.read_table(
        run_dir / "events.parquet",
        columns=["game_id", "seq", "event_type", "action", "forced"],
        filters=[("game_id", "==", game_id)],
    ).to_pylist()
    events.sort(key=lambda row: row["seq"])
    actions = [
        row["action"]
        for row in events
        if row["event_type"] == EventType.DECISION.value and not row["forced"]
    ]
    return games[0], actions


def replay_game(run_dir: Path | str, game_id: int, sink: EventSink | None = None) -> Game:
    meta, actions = load_decisions(run_dir, game_id)
    game = Game(meta["num_players"], sink=sink)
    outcome = game.reset(meta["seed"], [Candidate[name] for name in meta["candidates"]])
    for action in actions:
        if not isinstance(outcome, DecisionRequest):
            raise ReplayError(f"game {game_id}: the log has more decisions than the game")
        outcome = game.step(action)
    if not isinstance(outcome, GameOver):
        raise ReplayError(f"game {game_id}: the log ended before the game finished")
    return game


def verify_replay(run_dir: Path | str, game_id: int) -> bool:
    meta, _ = load_decisions(run_dir, game_id)
    result = replay_game(run_dir, game_id).result
    assert result is not None
    return (
        list(result.scores) == meta["final_scores"]
        and list(result.hands) == meta["final_hands"]
        and list(result.banks) == meta["final_banks"]
        and list(result.winners) == meta["winners"]
    )
