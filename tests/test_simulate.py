from math import comb
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from primary.cli import main
from primary.simulate import SimConfig, run_simulation, seat_policies

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
HEURISTIC_TABLE = ["heuristic", "random", "random", "random", "random"]


def test_seat_policies_rotate_through_every_seat():
    cfg = SimConfig(agents=HEURISTIC_TABLE)
    assert [seat_policies(cfg, g).index("heuristic") for g in range(5)] == [0, 1, 2, 3, 4]


def test_single_agent_name_fills_every_seat():
    assert seat_policies(SimConfig(agents=["random"]), 0) == ("random",) * 5


def test_wrong_number_of_agents_raises():
    with pytest.raises(ValueError, match="agents"):
        seat_policies(SimConfig(agents=["random", "random"]), 0)


def test_unknown_agent_raises():
    with pytest.raises(ValueError, match="unknown agent"):
        run_simulation(SimConfig(num_games=1, agents=["nope"], log_every=0))


def test_run_logs_to_a_path_with_spaces(tmp_path):
    out = tmp_path / "run with spaces"
    summary = run_simulation(SimConfig(num_games=3, seed=1, out_dir=str(out)))
    assert len(summary.records) == 3
    assert pq.read_table(out / "games.parquet").num_rows == 3
    assert (out / "config.yaml").exists()


def test_refuses_to_mix_runs_unless_overwrite(tmp_path):
    out = tmp_path / "run"
    run_simulation(SimConfig(num_games=3, out_dir=str(out)))
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_simulation(SimConfig(num_games=3, out_dir=str(out)))
    run_simulation(SimConfig(num_games=3, out_dir=str(out), overwrite=True))
    assert pq.read_table(out / "games.parquet").num_rows == 3


def test_log_every_n_games(tmp_path):
    run_simulation(SimConfig(num_games=4, log_every=2, out_dir=str(tmp_path / "r")))
    ids = pq.read_table(tmp_path / "r" / "games.parquet").column("game_id").to_pylist()
    assert ids == [0, 2]


def test_logging_off_writes_nothing(tmp_path):
    run_simulation(SimConfig(num_games=2, log_every=0, out_dir=str(tmp_path / "r")))
    assert not (tmp_path / "r").exists()


def test_simulation_is_reproducible():
    a = run_simulation(SimConfig(num_games=5, seed=9, log_every=0))
    b = run_simulation(SimConfig(num_games=5, seed=9, log_every=0))
    assert [r.result for r in a.records] == [r.result for r in b.records]


def test_from_yaml_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("num_games: 3\nnum_gmaes: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="num_gmaes"):
        SimConfig.from_yaml(path)


def test_shipped_configs_load():
    for path in CONFIGS.glob("*.yaml"):
        SimConfig.from_yaml(path)


def binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def test_heuristic_beats_random():  # spec §7 sanity baseline
    summary = run_simulation(
        SimConfig(num_games=100, seed=123, agents=HEURISTIC_TABLE, log_every=0)
    )
    wins = int(summary.win_rates_by_policy()["heuristic"] * 100)
    assert binomial_tail(wins, 100, 0.2) < 0.01, f"heuristic won only {wins}/100"


def test_cli_simulate_then_replay(tmp_path, capsys):
    out = tmp_path / "cli run"
    args = ["simulate", "--config", str(CONFIGS / "sim_random.yaml"), "--games", "2"]
    assert main([*args, "--out", str(out)]) == 0
    assert "games/sec" in capsys.readouterr().out
    assert main(["replay", "--run", str(out), "--game-id", "1"]) == 0
    assert "matches" in capsys.readouterr().out
