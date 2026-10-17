"""Column definitions for the three logged tables (spec §6)."""

from __future__ import annotations

import pyarrow as pa

_i8, _i16, _i32, _i64, _str, _bool = (
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.string(),
    pa.bool_(),
)

EVENTS = pa.schema(
    [
        ("game_id", _i64),
        ("seq", _i32),
        ("round", _i8),
        ("phase", _str),
        ("event_type", _str),
        ("seat", _i8),
        ("candidate", _str),
        ("card", _str),
        ("news", _str),
        ("from_region", _str),
        ("to_region", _str),
        ("cube_owner", _i8),
        ("amount", _i16),
        ("source", _str),
        ("hand_before", _i16),
        ("hand_after", _i16),
        ("bank_before", _i16),
        ("bank_after", _i16),
        ("pawn_before", _str),
        ("pawn_after", _str),
        ("decision_type", _str),
        ("action", _i16),
        ("forced", _bool),
        ("detail", _str),
    ]
)

ROUNDS = pa.schema(
    [
        ("game_id", _i64),
        ("round", _i8),
        ("seat", _i8),
        ("candidate", _str),
        ("start_region", _str),
        ("end_region", _str),
        ("path", pa.list_(_str)),
        ("hand_start", _i16),
        ("hand_end", _i16),
        ("bank_start", _i16),
        ("bank_end", _i16),
        ("cards", pa.list_(_str)),
        ("skipped_slot", _i8),
        ("vote_regions", pa.list_(_str)),
        ("vote_placements", pa.list_(_str)),
        ("vote_votes", pa.list_(_i16)),
        ("votes_gained", _i16),
        ("round_result", _str),
    ]
)

GAMES = pa.schema(
    [
        ("game_id", _i64),
        ("seed", _i64),
        ("engine_version", _str),
        ("num_players", _i8),
        ("candidates", pa.list_(_str)),
        ("policies", pa.list_(_str)),
        ("start_seat", _i8),
        ("schedule_rounds", pa.list_(_i8)),
        ("schedule_regions", pa.list_(_str)),
        ("news_order", pa.list_(_str)),
        ("final_scores", pa.list_(_i16)),
        ("final_hands", pa.list_(_i16)),
        ("final_banks", pa.list_(_i16)),
        ("winners", pa.list_(_i8)),
        ("draw", _bool),
        ("tiebreak", _str),
        ("num_decisions", _i32),
    ]
)

SCHEMAS: dict[str, pa.Schema] = {"games": GAMES, "events": EVENTS, "rounds": ROUNDS}
TABLE_NAMES = tuple(SCHEMAS)
