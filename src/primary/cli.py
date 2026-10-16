"""Command-line interface: ``primary simulate`` and ``primary replay``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from primary.logs.replay import load_decisions, verify_replay
from primary.simulate import SimConfig, SimSummary, run_simulation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="primary", description="Simulate and replay games of The Primary."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="play many games between bots")
    sim.add_argument("--config", type=Path, help="YAML run config (see configs/)")
    sim.add_argument("--games", type=int, help="override num_games")
    sim.add_argument("--seed", type=int, help="override seed")
    sim.add_argument("--out", help="override out_dir")
    sim.add_argument("--overwrite", action="store_true", help="replace an existing run in --out")
    sim.add_argument("--no-log", action="store_true", help="do not write Parquet logs")

    rep = sub.add_parser("replay", help="re-run a logged game and check it matches the log")
    rep.add_argument("--run", type=Path, required=True, help="run directory")
    rep.add_argument("--game-id", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "simulate":
        return _simulate(args)
    return _replay(args)


def _simulate(args: argparse.Namespace) -> int:
    cfg = SimConfig.from_yaml(args.config) if args.config else SimConfig()
    if args.games is not None:
        cfg.num_games = args.games
    if args.seed is not None:
        cfg.seed = args.seed
    if args.out is not None:
        cfg.out_dir = args.out
    if args.overwrite:
        cfg.overwrite = True
    if args.no_log:
        cfg.log_every = 0
    summary = run_simulation(cfg)
    _print_summary(summary)
    if cfg.out_dir and cfg.log_every > 0:
        print(f"logs written to {Path(cfg.out_dir).resolve()}")
    return 0


def _print_summary(summary: SimSummary) -> None:
    print(
        f"games: {len(summary.records)}  time: {summary.seconds:.1f}s  "
        f"games/sec: {summary.games_per_second:.1f}  draw rate: {summary.draw_rate:.3f}"
    )
    print("win rate by policy:")
    for name, rate in sorted(summary.win_rates_by_policy().items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} {rate:.3f}")
    print("win rate by candidate:")
    for name, rate in sorted(summary.win_rates_by_candidate().items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} {rate:.3f}")


def _replay(args: argparse.Namespace) -> int:
    meta, actions = load_decisions(args.run, args.game_id)
    ok = verify_replay(args.run, args.game_id)
    print(
        f"game {args.game_id}: {len(actions)} decisions, scores {meta['final_scores']}, "
        f"winners {meta['winners']} ({meta['tiebreak']})"
    )
    print("replay matches the log" if ok else "REPLAY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
