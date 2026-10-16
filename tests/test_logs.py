import pyarrow.parquet as pq
import pytest

from primary.engine.game import Game
from primary.logs.replay import replay_game, verify_replay
from primary.logs.writer import GameRecorder, RunLogger
from tests.helpers import play_random


def play_logged(out_dir, n_games=2):
    games = []
    with RunLogger(out_dir) as logger:
        for game_id in range(n_games):
            recorder = logger.recorder(game_id)
            game = Game(5, sink=recorder)
            play_random(game, seed=100 + game_id)
            logger.finish(recorder, game, policies=["random"] * 5)
            games.append(game)
    return games


def test_writes_three_tables(tmp_path):
    play_logged(tmp_path)
    assert pq.read_table(tmp_path / "games.parquet").num_rows == 2
    assert pq.read_table(tmp_path / "rounds.parquet").num_rows == 2 * 11 * 5
    events = pq.read_table(tmp_path / "events.parquet")
    kinds = set(events.column("event_type").to_pylist())
    assert {"decision", "card_committed", "card_revealed", "move", "cube_placed", "gain"} <= kinds


def test_logs_cover_the_claude_md_data_collection_list(tmp_path):
    play_logged(tmp_path, n_games=1)
    rounds = pq.read_table(tmp_path / "rounds.parquet").schema.names
    events = pq.read_table(tmp_path / "events.parquet").schema.names
    # locations (start, end, in between), hand before/after, cards played, vote outcome
    for column in (
        "start_region",
        "end_region",
        "path",
        "hand_start",
        "hand_end",
        "cards",
        "vote_regions",
        "vote_placements",
        "round_result",
    ):
        assert column in rounds
    # where cubes were placed or removed, and by whom
    for column in ("from_region", "to_region", "cube_owner", "amount", "hand_before", "hand_after"):
        assert column in events


def test_round_result_labels(tmp_path):
    play_logged(tmp_path, n_games=1)
    rows = pq.read_table(tmp_path / "rounds.parquet").to_pylist()
    assert {r["round_result"] for r in rows} <= {"no_vote", "won", "lost", "not_participated"}
    assert all(r["round_result"] == "no_vote" for r in rows if r["round"] in (1, 4, 9))


def test_replay_reproduces_the_logged_result(tmp_path):
    games = play_logged(tmp_path)
    assert replay_game(tmp_path, 1).result == games[1].result
    assert verify_replay(tmp_path, 1)


def test_replay_reproduces_the_event_stream(tmp_path):
    play_logged(tmp_path)
    recorder = GameRecorder(1)
    replay_game(tmp_path, 1, sink=recorder)
    logged = pq.read_table(tmp_path / "events.parquet", filters=[("game_id", "==", 1)]).to_pylist()
    logged.sort(key=lambda row: row["seq"])
    assert logged == recorder.events


def test_replaying_an_unlogged_game_raises(tmp_path):
    play_logged(tmp_path)
    with pytest.raises(KeyError, match="was not logged"):
        replay_game(tmp_path, 99)
