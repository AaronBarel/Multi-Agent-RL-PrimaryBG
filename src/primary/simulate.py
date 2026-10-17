"""Run many games between bots and optionally log them. Used by the ``primary`` CLI."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from primary.agents import make_agent
from primary.data.candidates import Candidate
from primary.engine.actions import DecisionRequest, GameOver
from primary.engine.game import Game
from primary.logs.schema import TABLE_NAMES
from primary.logs.writer import RunLogger


@dataclass
class SimConfig:
    num_games: int = 100
    num_players: int = 5
    seed: int = 0
    agents: list[str] = field(default_factory=lambda: ["random"])
    rotate_seats: bool = True
    log_every: int = 1  # 0 disables logging; N logs games 0, N, 2N, ...
    out_dir: str | None = None
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, path: Path | str) -> SimConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class GameRecord:
    game_id: int
    seed: int
    candidates: tuple[Candidate, ...]
    policies: tuple[str, ...]
    result: GameOver


@dataclass
class SimSummary:
    records: list[GameRecord]
    seconds: float

    @property
    def games_per_second(self) -> float:
        return len(self.records) / self.seconds if self.seconds > 0 else float("inf")

    @property
    def draw_rate(self) -> float:
        return sum(r.result.draw for r in self.records) / max(len(self.records), 1)

    def win_rates_by_policy(self) -> dict[str, float]:
        return self._win_rates(lambda rec, seat: rec.policies[seat])

    def win_rates_by_candidate(self) -> dict[str, float]:
        return self._win_rates(lambda rec, seat: rec.candidates[seat].name)

    def _win_rates(self, key: Callable[[GameRecord, int], str]) -> dict[str, float]:
        """Share of seats won; a k-way draw credits each tied seat 1/k."""
        wins: dict[str, float] = defaultdict(float)
        seats: dict[str, int] = defaultdict(int)
        for rec in self.records:
            share = 1.0 / len(rec.result.winners)
            for seat in range(len(rec.policies)):
                name = key(rec, seat)
                seats[name] += 1
                if seat in rec.result.winners:
                    wins[name] += share
        return {name: wins[name] / seats[name] for name in seats}


def seat_policies(cfg: SimConfig, game_id: int) -> tuple[str, ...]:
    names = list(cfg.agents) * cfg.num_players if len(cfg.agents) == 1 else list(cfg.agents)
    if len(names) != cfg.num_players:
        raise ValueError(
            f"agents must list 1 or {cfg.num_players} names, got {len(cfg.agents)}: {cfg.agents}"
        )
    k = game_id % cfg.num_players if cfg.rotate_seats else 0
    return tuple(names[-k:] + names[:-k]) if k else tuple(names)


def _prepare_out_dir(cfg: SimConfig) -> Path | None:
    if cfg.out_dir is None or cfg.log_every <= 0:
        return None
    out = Path(cfg.out_dir)
    if out.exists() and any(out.iterdir()):
        if not cfg.overwrite:
            raise FileExistsError(
                f"{out} already contains a run; pass --overwrite or choose another --out"
            )
        for name in TABLE_NAMES:
            (out / f"{name}.parquet").unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    saved = asdict(cfg) | {"overwrite": False}  # re-running a saved config must not delete this run
    (out / "config.yaml").write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")
    return out


def run_simulation(cfg: SimConfig) -> SimSummary:
    seat_policies(cfg, 0)
    for name in set(cfg.agents):
        make_agent(name)
    out = _prepare_out_dir(cfg)
    logger = RunLogger(out) if out is not None else None
    rng = np.random.default_rng(cfg.seed)
    records: list[GameRecord] = []
    start = time.perf_counter()
    try:
        for game_id in range(cfg.num_games):
            game_seed = int(rng.integers(0, 2**31 - 1))
            drawn = rng.choice(len(Candidate), cfg.num_players, replace=False)
            candidates = tuple(Candidate(int(c)) for c in drawn)
            policies = seat_policies(cfg, game_id)
            agents = [make_agent(name, seed=game_seed + seat) for seat, name in enumerate(policies)]
            logged = logger is not None and game_id % cfg.log_every == 0
            recorder = logger.recorder(game_id) if logger is not None and logged else None
            game = Game(cfg.num_players, sink=recorder)
            outcome = game.reset(game_seed, candidates)
            while isinstance(outcome, DecisionRequest):
                outcome = game.step(agents[outcome.seat].act(game, outcome))
            if logger is not None and recorder is not None:
                logger.finish(recorder, game, policies)
            records.append(GameRecord(game_id, game_seed, candidates, policies, outcome))
    finally:
        if logger is not None:
            logger.close()
    return SimSummary(records, time.perf_counter() - start)
