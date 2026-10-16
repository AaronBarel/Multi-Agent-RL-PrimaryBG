# The Primary — Game Engine, Environment & Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully tested, reproducible rules engine for *The Primary*, a PettingZoo AEC environment on top of it, random and heuristic bots, Parquet logging with replay, a `primary` CLI, CI on Linux + Windows, and a sanity-check notebook.

**Architecture:** A pure-Python engine written as generator "flows": rule code reads top-to-bottom and `yield`s a `DecisionRequest` whenever a player must choose; `Game.step(action)` sends the answer back in. All state mutation goes through a `Table` object that emits `Event`s to an optional sink (the logger). Candidate abilities and News cards are data: each one edits a per-player `Modifiers` dataclass that the core reads, so the core contains no candidate- or News-specific branches. The PettingZoo environment is a thin encoding layer over `Game`.

**Tech Stack:** Python ≥ 3.11, uv, numpy, pettingzoo, gymnasium, pyarrow, pyyaml, pytest, hypothesis, ruff, mypy; pandas/matplotlib/jupyter only in the optional `analysis` group.

**Spec:** `docs/superpowers/specs/2026-10-15-primary-engine-design.md` (rules clarifications are numbered §3 rules 1–52; tests cite them as `# rule N`).

## Global Constraints

- Python `>=3.11`; CI matrix: `ubuntu-latest` + `windows-latest` × Python 3.11, 3.12.
- Runtime dependencies are exactly: `numpy`, `gymnasium`, `pettingzoo`, `pyarrow`, `pyyaml` (all ship Windows wheels). No PyTorch or RL library in this sub-project.
- `primary.data` and `primary.engine` must not import `pettingzoo`, `gymnasium`, or `pyarrow`.
- Default 5 players; 3–6 must work (`MIN_PLAYERS = 3`, `MAX_PLAYERS = 6`).
- Every game must be exactly reproducible from `(seed, candidates, non-forced actions)`.
- Use `pathlib.Path` for all file paths; no shell-specific code (must run natively on Windows 10).
- Logged runs go under `data/runs/<run_id>/` (git-ignored).
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Work happens on branch `feature/game-engine`.

## Design notes (refinements of spec §5–6, decided while planning)

- **Hook points (spec §5.3) are implemented as `Modifiers`:** `abilities.py` and `news_effects.py` each map every Candidate/News card to one small function that edits a `Modifiers` instance. The engine reads modifier fields at the hook points (setup, news, commit, card resolution, enter-region, round-end). That fully data-driven design is simpler than a callback registry and keeps each card testable in isolation.
- **Event types:** news choices are logged as `decision` events (with `decision_type`), and bonuses as `gain` events whose `detail` names the reason (`super_pac`, `kristen`, `colocation`, …). The spec's `news_choice`/`bonus` names are covered by these.
- **Optional placements** are two decision types, `OPTIONAL_HAND_PLACEMENT` (Calvin, Lily) and `OPTIONAL_BANK_PLACEMENT` (Paparazzi), so an agent can tell which resource it is spending.
- **Cube moves onto already-voted regions:** New Constituents returns the cube to its owner's bank immediately (rule 50); everything else (placements, Erin's move) stays until round end and is then returned (rule 15).
- **Forced decisions** (exactly one legal action) are resolved by the engine without asking, are logged with `forced=True`, and are not part of `Game.history`.

## Review Focus

1. **3- and 6-player games** — someone configures `num_players: 3` or `6` and expects a valid game: covered by the Hypothesis test in Task 8 drawing `num_players` from 3–6.
2. **Running on Windows with a path that contains spaces, or re-running into an existing run directory** — expect it to work, and to refuse to silently mix two runs unless `--overwrite` is given: tests in Task 11.
3. **Late-game bank exhaustion** — gains silently capped at what's left in the bank, never negative counts: explicit test in Task 4 plus invariant checks on every step in Task 8.
4. **Stepping after the game ends or sending an illegal action** (engine and env) — expect a clear error naming the legal actions, not corrupted state: tests in Task 7 and Task 12.
5. **Replaying a game that was not logged** (`log_every > 1`) — expect a clear `KeyError` naming the run directory: test in Task 10.

## File structure

```
pyproject.toml  .python-version  .gitignore  uv.lock
configs/sim_random.yaml  configs/sim_heuristic.yaml
src/primary/
  __init__.py                 ENGINE_VERSION
  data/board.py               Region, ADJACENCY, AWARDS, VOTING_SLOTS, TOTAL_VOTES
  data/cards.py               Card, CARD_COUNTS, CARDS_PER_ROUND
  data/candidates.py          Candidate, STARTING_HAND, TOTAL_CUBES, MIN/MAX_PLAYERS
  data/news.py                News, NEWS_TEXT
  engine/voting.py            Placement, RegionResult, compute_awards, TieBreak, determine_winners
  engine/movement.py          distances_from, distance, reachable
  engine/actions.py           action layout, DecisionType, DecisionRequest, GameOver, errors
  engine/state.py             PlayerState, RoundState, GameState, make_schedule, new_game_state
  engine/events.py            Phase, EventType, Source, Event, RegionOutcome, RoundSummary, EventSink, ListSink
  engine/modifiers.py         NewsMove, Modifiers
  engine/abilities.py         ABILITY_MODIFIERS
  engine/news_effects.py      NEWS_MODIFIERS, move_options
  engine/effects.py           build_modifiers
  engine/table.py             Table (all primitive mutations + event emission)
  engine/invariants.py        InvariantError, check_invariants
  engine/decisions.py         ask, Decide, Flow
  engine/resolve.py           resolve_card
  engine/round_end.py         Tier, most_tiers, influence, resolve_round_end, finish_game
  engine/phases.py            setup_phase, news_phase, available_cards, commit_phase, execute_phase
  engine/game.py              Game, Outcome
  agents/base.py              Agent protocol
  agents/random_agent.py      RandomAgent
  agents/heuristic.py         HeuristicAgent, plan_cards, best_region
  agents/__init__.py          make_agent, AGENT_NAMES
  logs/schema.py              pyarrow schemas
  logs/writer.py              event_row, round_row, game_row, GameRecorder, RunLogger
  logs/replay.py              ReplayError, load_decisions, replay_game, verify_replay
  simulate.py                 SimConfig, GameRecord, SimSummary, seat_policies, run_simulation
  cli.py                      `primary simulate`, `primary replay`
  env/encoding.py             obs_size, encode_observation, encode_mask, to_absolute_action
  env/rewards.py              terminal_rewards, shaped_rewards
  env/aec_env.py              PrimaryEnv, env()
tests/  helpers.py + one test module per source module
notebooks/01_engine_sanity.ipynb
.github/workflows/ci.yml
```

---

### Task 1: Project scaffold and game data

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`
- Create: `src/primary/__init__.py`, `src/primary/data/__init__.py`, `src/primary/data/board.py`, `src/primary/data/cards.py`, `src/primary/data/candidates.py`, `src/primary/data/news.py`
- Create: `tests/__init__.py`, `tests/test_data.py`

**Interfaces:**
- Produces: `Region` (IntEnum 0–13), `NUM_REGIONS = 14`, `NUM_ROUNDS = 11`, `ADJACENCY: Mapping[Region, frozenset[Region]]`, `AWARDS: Mapping[Region, tuple[int, ...]]`, `VOTING_SLOTS: Mapping[int, int]`, `TOTAL_VOTES = 240`, `REGION_NAMES`, `STATES`; `Card` (IntEnum 0–6), `CARD_COUNTS`, `CARDS_PER_ROUND = 4`, `CARD_TEXT`; `Candidate` (IntEnum 0–11), `STARTING_HAND`, `TOTAL_CUBES = 30`, `MIN_PLAYERS = 3`, `MAX_PLAYERS = 6`, `DISPLAY_NAME`, `ABILITY_TEXT`; `News` (IntEnum 0–17), `NEWS_TEXT`; `primary.ENGINE_VERSION = "0.1.0"`.

- [ ] **Step 1: Create the project files**

`pyproject.toml`:

```toml
[project]
name = "primary-marl"
version = "0.1.0"
description = "Multi-agent reinforcement learning for The Primary board game"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = [
    "numpy>=1.26",
    "gymnasium>=0.29",
    "pettingzoo>=1.24.3",
    "pyarrow>=15",
    "pyyaml>=6.0",
]

[project.scripts]
primary = "primary.cli:main"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "hypothesis>=6.100",
    "ruff>=0.6",
    "mypy>=1.11",
    "types-PyYAML>=6.0",
]
analysis = [
    "pandas>=2.2",
    "matplotlib>=3.8",
    "jupyter>=1.0",
    "nbformat>=5.10",
]

[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/primary"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]  # line length is enforced by `ruff format`; long rule-text strings are fine

[tool.mypy]
python_version = "3.11"
strict = true
files = ["src/primary/data", "src/primary/engine", "src/primary/env"]

[[tool.mypy.overrides]]
module = ["pettingzoo.*", "gymnasium.*", "pyarrow.*"]
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["primary.env.*"]
disallow_subclassing_any = false
```

`.python-version`:

```
3.12
```

`.gitignore`:

```
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.hypothesis/
.ipynb_checkpoints/
dist/
build/
*.egg-info/
data/runs/
```

`src/primary/__init__.py`:

```python
"""Multi-agent reinforcement learning for The Primary board game."""

ENGINE_VERSION = "0.1.0"

__all__ = ["ENGINE_VERSION"]
```

`src/primary/data/__init__.py`:

```python
"""Static game data transcribed from CLAUDE.md (the single source of truth for rules data)."""
```

`tests/__init__.py`: empty file.

- [ ] **Step 2: Install the environment**

Run: `uv sync`
Expected: creates `.venv/` and `uv.lock`, installs the dev group.

- [ ] **Step 3: Write the failing data tests**

`tests/test_data.py`:

```python
from primary.data.board import (
    ADJACENCY,
    AWARDS,
    NUM_REGIONS,
    NUM_ROUNDS,
    REGION_NAMES,
    STATES,
    TOTAL_VOTES,
    VOTING_SLOTS,
    Region,
)
from primary.data.candidates import (
    ABILITY_TEXT,
    DISPLAY_NAME,
    STARTING_HAND,
    TOTAL_CUBES,
    Candidate,
)
from primary.data.cards import CARD_COUNTS, CARD_TEXT, CARDS_PER_ROUND, Card
from primary.data.news import NEWS_TEXT, News


def test_fourteen_regions_with_names_and_fifty_states():
    assert NUM_REGIONS == 14
    assert set(REGION_NAMES) == set(Region)
    all_states = [s for states in STATES.values() for s in states]
    assert len(all_states) == 50
    assert len(set(all_states)) == 50


def test_adjacency_is_symmetric_and_irreflexive():
    assert set(ADJACENCY) == set(Region)
    for region, neighbours in ADJACENCY.items():
        assert region not in neighbours
        assert neighbours, f"{region.name} has no neighbours"
        for other in neighbours:
            assert region in ADJACENCY[other], f"{region.name} -> {other.name} not symmetric"


def test_new_england_only_borders_north_east():
    assert ADJACENCY[Region.NEW_ENGLAND] == frozenset({Region.NORTH_EAST})


def test_awards_match_rulebook():
    assert set(AWARDS) == set(Region)
    assert AWARDS[Region.NEW_ENGLAND] == (18, 8)
    assert AWARDS[Region.NORTH] == (10,)
    assert sum(1 for a in AWARDS.values() if len(a) == 2) == 8
    assert TOTAL_VOTES == 240


def test_voting_slots_cover_every_region_once():
    assert set(VOTING_SLOTS) == set(range(1, NUM_ROUNDS + 1))
    assert sum(VOTING_SLOTS.values()) == NUM_REGIONS
    assert [rnd for rnd, n in VOTING_SLOTS.items() if n == 0] == [1, 4, 9]


def test_action_deck_has_fourteen_cards():
    assert sum(CARD_COUNTS.values()) == 14
    assert CARD_COUNTS[Card.SUPER_PAC] == 4
    assert CARD_COUNTS[Card.RALLY] == 3
    assert CARDS_PER_ROUND == 4
    assert set(CARD_TEXT) == set(Card)


def test_candidates_and_starting_hands():
    assert len(Candidate) == 12
    assert set(STARTING_HAND) == set(DISPLAY_NAME) == set(ABILITY_TEXT) == set(Candidate)
    assert STARTING_HAND[Candidate.ANTONIO_PARADES] == 5
    assert STARTING_HAND[Candidate.RYAN_GUPTA] == 1
    assert STARTING_HAND[Candidate.GEORGIA_BARNES] == 1
    assert TOTAL_CUBES == 30


def test_eighteen_news_cards():
    assert len(News) == 18
    assert set(NEWS_TEXT) == set(News)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primary.data.board'`.

- [ ] **Step 5: Write the data modules**

`src/primary/data/board.py`:

```python
"""Regions, borders, vote awards and the voting schedule (CLAUDE.md: Primary Game Map Environment)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType


class Region(IntEnum):
    PACIFIC_NW = 0
    WEST = 1
    SOUTH_WEST = 2
    GREATER_TEXAS = 3
    NORTH = 4
    GREAT_LAKES = 5
    EARLY_MIDWEST = 6
    MIDWEST = 7
    DEEP_SOUTH = 8
    SOUTH_EAST = 9
    SOUTH = 10
    APPALACHIA = 11
    NORTH_EAST = 12
    NEW_ENGLAND = 13


NUM_REGIONS = len(Region)
NUM_ROUNDS = 11

REGION_NAMES: Mapping[Region, str] = MappingProxyType(
    {
        Region.PACIFIC_NW: "Pacific NW",
        Region.WEST: "West",
        Region.SOUTH_WEST: "South West",
        Region.GREATER_TEXAS: "Greater Texas",
        Region.NORTH: "North",
        Region.GREAT_LAKES: "Great Lakes",
        Region.EARLY_MIDWEST: "Early Midwest",
        Region.MIDWEST: "Midwest",
        Region.DEEP_SOUTH: "Deep South",
        Region.SOUTH_EAST: "South East",
        Region.SOUTH: "South",
        Region.APPALACHIA: "Appalachia",
        Region.NORTH_EAST: "North East",
        Region.NEW_ENGLAND: "New England",
    }
)

STATES: Mapping[Region, tuple[str, ...]] = MappingProxyType(
    {
        Region.PACIFIC_NW: ("Washington", "Oregon", "Idaho"),
        Region.WEST: ("California", "Nevada", "Alaska", "Hawaii"),
        Region.SOUTH_WEST: ("Colorado", "Utah", "Arizona", "New Mexico"),
        Region.GREATER_TEXAS: ("Texas", "Oklahoma"),
        Region.NORTH: ("Montana", "Wyoming", "North Dakota", "South Dakota"),
        Region.GREAT_LAKES: ("Minnesota", "Wisconsin", "Michigan"),
        Region.EARLY_MIDWEST: ("Illinois", "Ohio", "Indiana"),
        Region.MIDWEST: ("Nebraska", "Iowa", "Missouri", "Kansas"),
        Region.DEEP_SOUTH: ("Louisiana", "Mississippi", "Arkansas"),
        Region.SOUTH_EAST: ("Florida", "Alabama", "Georgia"),
        Region.SOUTH: ("Tennessee", "North Carolina", "South Carolina"),
        Region.APPALACHIA: ("Kentucky", "West Virginia", "Virginia"),
        Region.NORTH_EAST: ("Pennsylvania", "New Jersey", "Maryland", "Delaware"),
        Region.NEW_ENGLAND: (
            "New York",
            "Vermont",
            "New Hampshire",
            "Massachusetts",
            "Connecticut",
            "Maine",
            "Rhode Island",
        ),
    }
)

_R = Region
_BORDERS: dict[Region, tuple[Region, ...]] = {
    _R.PACIFIC_NW: (_R.NORTH, _R.SOUTH_WEST, _R.WEST),
    _R.WEST: (_R.SOUTH_WEST, _R.PACIFIC_NW),
    _R.SOUTH_WEST: (_R.NORTH, _R.PACIFIC_NW, _R.WEST, _R.MIDWEST, _R.GREATER_TEXAS),
    _R.GREATER_TEXAS: (_R.SOUTH_WEST, _R.MIDWEST, _R.DEEP_SOUTH),
    _R.NORTH: (_R.PACIFIC_NW, _R.SOUTH_WEST, _R.MIDWEST, _R.GREAT_LAKES),
    _R.GREAT_LAKES: (_R.NORTH, _R.MIDWEST, _R.EARLY_MIDWEST),
    _R.EARLY_MIDWEST: (_R.MIDWEST, _R.GREAT_LAKES, _R.NORTH_EAST, _R.APPALACHIA),
    _R.MIDWEST: (
        _R.EARLY_MIDWEST,
        _R.NORTH,
        _R.SOUTH_WEST,
        _R.GREATER_TEXAS,
        _R.DEEP_SOUTH,
        _R.GREAT_LAKES,
        _R.APPALACHIA,
        _R.SOUTH,
    ),
    _R.DEEP_SOUTH: (_R.MIDWEST, _R.GREATER_TEXAS, _R.SOUTH_EAST, _R.SOUTH),
    _R.SOUTH_EAST: (_R.DEEP_SOUTH, _R.SOUTH),
    _R.SOUTH: (_R.MIDWEST, _R.DEEP_SOUTH, _R.SOUTH_EAST, _R.APPALACHIA),
    _R.APPALACHIA: (_R.MIDWEST, _R.SOUTH, _R.EARLY_MIDWEST, _R.NORTH_EAST),
    _R.NORTH_EAST: (_R.NEW_ENGLAND, _R.EARLY_MIDWEST, _R.APPALACHIA),
    _R.NEW_ENGLAND: (_R.NORTH_EAST,),
}
ADJACENCY: Mapping[Region, frozenset[Region]] = MappingProxyType(
    {region: frozenset(neighbours) for region, neighbours in _BORDERS.items()}
)

# (1st place,) for winner-take-all regions; (1st, 2nd) for regions with a second-place award.
AWARDS: Mapping[Region, tuple[int, ...]] = MappingProxyType(
    {
        _R.PACIFIC_NW: (13,),
        _R.WEST: (17, 7),
        _R.SOUTH_WEST: (11, 5),
        _R.GREATER_TEXAS: (15, 6),
        _R.NORTH: (10,),
        _R.GREAT_LAKES: (14,),
        _R.EARLY_MIDWEST: (13, 5),
        _R.MIDWEST: (15,),
        _R.DEEP_SOUTH: (11,),
        _R.SOUTH_EAST: (16, 7),
        _R.SOUTH: (12, 5),
        _R.APPALACHIA: (12,),
        _R.NORTH_EAST: (14, 6),
        _R.NEW_ENGLAND: (18, 8),
    }
)

# Round number -> number of regions voting that round.
VOTING_SLOTS: Mapping[int, int] = MappingProxyType(
    {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2, 7: 1, 8: 2, 9: 0, 10: 2, 11: 3}
)

TOTAL_VOTES = sum(sum(awards) for awards in AWARDS.values())
```

`src/primary/data/cards.py`:

```python
"""Action cards (CLAUDE.md: Action Cards)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType


class Card(IntEnum):
    SUPER_PAC = 0
    FUNDRAISER = 1
    PLANE = 2
    BUS = 3
    POSITIVE_AD = 4
    NEGATIVE_AD = 5
    RALLY = 6


CARD_COUNTS: Mapping[Card, int] = MappingProxyType(
    {
        Card.SUPER_PAC: 4,
        Card.FUNDRAISER: 3,
        Card.PLANE: 1,
        Card.BUS: 1,
        Card.POSITIVE_AD: 1,
        Card.NEGATIVE_AD: 1,
        Card.RALLY: 3,
    }
)

CARDS_PER_ROUND = 4

CARD_TEXT: Mapping[Card, str] = MappingProxyType(
    {
        Card.SUPER_PAC: "If you play the most Super PAC cards, gain 4 Influence cubes. If tied, gain 2",
        Card.FUNDRAISER: "Gain 1 Influence cube",
        Card.PLANE: "Pay 1 Influence cube and move to any region",
        Card.BUS: "Move to any adjacent region",
        Card.POSITIVE_AD: "Place 1 Influence cube onto any region",
        Card.NEGATIVE_AD: "Remove any 1 Influence cube from any region",
        Card.RALLY: "Place 2 Influence cubes onto your current region",
    }
)
```

`src/primary/data/candidates.py`:

```python
"""Candidates, their abilities and starting hands (CLAUDE.md: Candidate Information and Abilities)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType

TOTAL_CUBES = 30
MIN_PLAYERS = 3
MAX_PLAYERS = 6


class Candidate(IntEnum):
    ANTONIO_PARADES = 0
    KENNETH_WALKER = 1
    BRYCE_MCKAY = 2
    KRISTEN_RAMIREZ = 3
    DIANE_PROUDIAN = 4
    RYAN_GUPTA = 5
    CHESTER_FREEMAN = 6
    LILY_NGUYEN = 7
    ERIN_FIELDS = 8
    CALVIN_MATSUI = 9
    ANNA_PATEL = 10
    GEORGIA_BARNES = 11


_C = Candidate

STARTING_HAND: Mapping[Candidate, int] = MappingProxyType(
    {
        _C.ANTONIO_PARADES: 5,
        _C.KENNETH_WALKER: 2,
        _C.BRYCE_MCKAY: 3,
        _C.KRISTEN_RAMIREZ: 3,
        _C.DIANE_PROUDIAN: 3,
        _C.RYAN_GUPTA: 1,
        _C.CHESTER_FREEMAN: 3,
        _C.LILY_NGUYEN: 2,
        _C.ERIN_FIELDS: 5,
        _C.CALVIN_MATSUI: 3,
        _C.ANNA_PATEL: 3,
        _C.GEORGIA_BARNES: 1,
    }
)

DISPLAY_NAME: Mapping[Candidate, str] = MappingProxyType(
    {c: c.name.replace("_", " ").title().replace("Mckay", "McKay") for c in Candidate}
)

ABILITY_TEXT: Mapping[Candidate, str] = MappingProxyType(
    {
        _C.ANTONIO_PARADES: "Bus cards allow you to move up to 2 adjacent regions away",
        _C.KENNETH_WALKER: "If you play 2 or more Fundraiser cards in a round, gain 1 extra cube",
        _C.BRYCE_MCKAY: "After each vote where you do not win any delegates, gain 1 cube",
        _C.KRISTEN_RAMIREZ: "If you play the most Super PAC cards in a round, gain 1 extra cube",
        _C.DIANE_PROUDIAN: "Each round you may choose to not execute 1 Action card in your queue",
        _C.RYAN_GUPTA: "Plane cards do not cost you an Influence cube",
        _C.CHESTER_FREEMAN: "If you end a round in the same region as any opponent, gain 1 cube",
        _C.LILY_NGUYEN: "Your first Rally each round may place 1 extra cube from your hand",
        _C.ERIN_FIELDS: "Before your first Action card each round, you may move 1 of your cubes",
        _C.CALVIN_MATSUI: "Bus cards allow you to place 1 cube from your hand on the destination",
        _C.ANNA_PATEL: "Your character token counts as 2 Influence cubes in its current region",
        _C.GEORGIA_BARNES: "Start with 3 home regions and place 2 bank cubes in each",
    }
)
```

`src/primary/data/news.py`:

```python
"""News cards (CLAUDE.md: News Cards)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType


class News(IntEnum):
    COUNTERMEASURES = 0
    OPPO_RESEARCH = 1
    GENEROUS_DONORS = 2
    TAX_REFORM = 3
    LAST_MINUTE_DEBATE = 4
    DETOUR = 5
    FROZEN_ASSETS = 6
    PAPARAZZI = 7
    SPIN_DOCTOR = 8
    FINAL_PUSH = 9
    LONG_TERM_PLANNING = 10
    DEREGULATION = 11
    EXTRA_AIRTIME = 12
    WALL_STREET_SUPPORT = 13
    ROAD_TRIP = 14
    PLUTOCRACY = 15
    NEW_CONSTITUENTS = 16
    FREQUENT_FLYER = 17


_N = News

NEWS_TEXT: Mapping[News, str] = MappingProxyType(
    {
        _N.COUNTERMEASURES: "Gain 1 cube each time you enter a region where any opponent has Influence",
        _N.OPPO_RESEARCH: "Players ending the round in the same region as an opponent gain 2 cubes",
        _N.GENEROUS_DONORS: "Gain 1 extra cube for each Fundraiser card played",
        _N.TAX_REFORM: "All players immediately gain 2 cubes",
        _N.LAST_MINUTE_DEBATE: "May move (free) to a region voting this round or next round",
        _N.DETOUR: "Must move (free) to a new region",
        _N.FROZEN_ASSETS: "Cards played this round cannot be used next round (redraw in round 11)",
        _N.PAPARAZZI: "Bus and Plane cards may place 1 bank cube on the destination",
        _N.SPIN_DOCTOR: "Positive Ad and Negative Ad cards may be used as either card type",
        _N.FINAL_PUSH: "All players may play 5 Action cards",
        _N.LONG_TERM_PLANNING: "May move (free) to a region voting in round 11",
        _N.DEREGULATION: "Most Fundraisers gains 2 cubes; tied (non-zero) players gain 1",
        _N.EXTRA_AIRTIME: "Positive Ad places 2 cubes from hand onto any region(s)",
        _N.WALL_STREET_SUPPORT: "Your first Rally uses 2 bank cubes instead of hand cubes",
        _N.ROAD_TRIP: "Bus cards move 1 adjacent region further",
        _N.PLUTOCRACY: "Most Super PACs gains 6 cubes; tied (non-zero) players gain 3",
        _N.NEW_CONSTITUENTS: "May swap your pawn with any Influence cube on the map",
        _N.FREQUENT_FLYER: "Plane cards do not cost an Influence cube",
    }
)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: 8 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .python-version .gitignore uv.lock src tests
git commit -m "Add project scaffold and game data constants" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Movement and voting rules

**Files:**
- Create: `src/primary/engine/__init__.py`, `src/primary/engine/movement.py`, `src/primary/engine/voting.py`
- Test: `tests/test_movement.py`, `tests/test_voting.py`

**Interfaces:**
- Consumes: `Region`, `ADJACENCY` (Task 1).
- Produces: `distances_from(src: Region) -> tuple[int, ...]` (indexed by region), `distance(a: Region, b: Region) -> int`, `reachable(src: Region, max_steps: int) -> tuple[Region, ...]` (excludes `src`, sorted by region index); `Placement` (StrEnum: `FIRST="won_1st"`, `SECOND="won_2nd"`, `LOST="lost"`, `ABSENT="not_participated"`), `RegionResult(votes: tuple[int, ...], placements: tuple[Placement, ...])`, `compute_awards(influence: Sequence[int], awards: tuple[int, ...]) -> RegionResult`, `TieBreak` (StrEnum: `SCORE`, `HAND`, `BANK`, `DRAW`), `determine_winners(scores, hands, banks) -> tuple[tuple[int, ...], TieBreak]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_movement.py`:

```python
from primary.data.board import ADJACENCY, NUM_REGIONS, Region
from primary.engine.movement import distance, reachable


def test_one_step_is_adjacency():  # rule 11
    for region in Region:
        assert set(reachable(region, 1)) == set(ADJACENCY[region])


def test_new_england_within_two_steps():  # rule 21
    assert set(reachable(Region.NEW_ENGLAND, 2)) == {
        Region.NORTH_EAST,
        Region.EARLY_MIDWEST,
        Region.APPALACHIA,
    }


def test_new_england_within_three_steps():  # rule 48
    assert set(reachable(Region.NEW_ENGLAND, 3)) == {
        Region.NORTH_EAST,
        Region.EARLY_MIDWEST,
        Region.APPALACHIA,
        Region.MIDWEST,
        Region.GREAT_LAKES,
        Region.SOUTH,
    }


def test_reachable_excludes_start():
    assert Region.MIDWEST not in reachable(Region.MIDWEST, 3)


def test_distance_west_to_new_england():
    assert distance(Region.WEST, Region.NEW_ENGLAND) == 5
    assert distance(Region.WEST, Region.WEST) == 0


def test_map_is_connected():
    for region in Region:
        assert len(reachable(region, NUM_REGIONS)) == NUM_REGIONS - 1
```

`tests/test_voting.py`:

```python
import pytest

from primary.engine.voting import Placement, TieBreak, compute_awards, determine_winners

F, S, L, A = Placement.FIRST, Placement.SECOND, Placement.LOST, Placement.ABSENT


@pytest.mark.parametrize(
    ("influence", "awards", "votes", "placements"),
    [
        ([3, 1, 0], (10,), [10, 0, 0], [F, L, A]),  # rule 17: winner takes all
        ([2, 2, 0], (10,), [5, 5, 0], [F, F, A]),  # rule 17: tie splits
        ([1, 1, 1], (11,), [3, 3, 3], [F, F, F]),  # rule 17: rounded down
        ([0, 0, 0], (10,), [0, 0, 0], [A, A, A]),  # nobody has influence
        ([4, 2, 1], (18, 8), [18, 8, 0], [F, S, L]),  # rule 18: unique 1st and 2nd
        ([0, 3, 0], (18, 8), [0, 26, 0], [A, F, A]),  # rule 18: only one player present
        ([3, 3, 1], (18, 8), [13, 13, 0], [F, F, L]),  # rule 18: tie for 1st splits 1st+2nd
        ([2, 2, 2], (18, 8), [8, 8, 8], [F, F, F]),  # rule 18: 26 // 3
        ([5, 2, 2], (18, 8), [18, 4, 4], [F, S, S]),  # rule 18: tie for 2nd
        ([5, 1, 1, 1], (14, 6), [14, 2, 2, 2], [F, S, S, S]),
    ],
)
def test_compute_awards(influence, awards, votes, placements):
    result = compute_awards(influence, awards)
    assert list(result.votes) == votes
    assert list(result.placements) == placements


@pytest.mark.parametrize(
    ("scores", "hands", "banks", "winners", "tiebreak"),
    [
        ([50, 40, 30], [0, 0, 0], [0, 0, 0], (0,), TieBreak.SCORE),
        ([50, 50, 30], [2, 1, 9], [0, 0, 0], (0,), TieBreak.HAND),  # rule 20
        ([50, 50, 30], [2, 2, 9], [3, 4, 0], (1,), TieBreak.BANK),  # rule 20
        ([50, 50, 30], [2, 2, 9], [4, 4, 0], (0, 1), TieBreak.DRAW),  # rule 20
    ],
)
def test_determine_winners(scores, hands, banks, winners, tiebreak):
    assert determine_winners(scores, hands, banks) == (winners, tiebreak)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_movement.py tests/test_voting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primary.engine'`.

- [ ] **Step 3: Implement**

`src/primary/engine/__init__.py`:

```python
"""Rules engine for The Primary. Pure Python; no RL dependencies."""
```

`src/primary/engine/movement.py`:

```python
"""Graph distances on the region map (Bus range, heuristics)."""

from __future__ import annotations

from collections import deque
from functools import cache

from primary.data.board import ADJACENCY, NUM_REGIONS, Region


@cache
def distances_from(src: Region) -> tuple[int, ...]:
    """Breadth-first distances from ``src`` to every region, indexed by region."""
    dist = [-1] * NUM_REGIONS
    dist[src] = 0
    queue = deque([src])
    while queue:
        current = queue.popleft()
        for nxt in sorted(ADJACENCY[current]):
            if dist[nxt] < 0:
                dist[nxt] = dist[current] + 1
                queue.append(nxt)
    return tuple(dist)


def distance(a: Region, b: Region) -> int:
    return distances_from(a)[b]


def reachable(src: Region, max_steps: int) -> tuple[Region, ...]:
    """Regions 1..max_steps steps away from ``src`` (never ``src`` itself)."""
    dist = distances_from(src)
    return tuple(r for r in Region if 0 < dist[r] <= max_steps)
```

`src/primary/engine/voting.py`:

```python
"""Vote awards for a region and end-of-game winner selection (spec §3 rules 17–20)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class Placement(StrEnum):
    FIRST = "won_1st"
    SECOND = "won_2nd"
    LOST = "lost"
    ABSENT = "not_participated"


@dataclass(frozen=True)
class RegionResult:
    votes: tuple[int, ...]
    placements: tuple[Placement, ...]


def _split(
    votes: list[int], placements: list[Placement], seats: list[int], total: int, place: Placement
) -> None:
    share = total // len(seats)
    for seat in seats:
        votes[seat] = share
        placements[seat] = place


def compute_awards(influence: Sequence[int], awards: tuple[int, ...]) -> RegionResult:
    """Votes per seat for one region.

    ``awards`` is ``(first,)`` for winner-take-all or ``(first, second)``.
    """
    votes = [0] * len(influence)
    placements = [Placement.LOST if x > 0 else Placement.ABSENT for x in influence]
    present = [i for i, x in enumerate(influence) if x > 0]
    if present:
        top = max(influence[i] for i in present)
        firsts = [i for i in present if influence[i] == top]
        if len(awards) == 1:
            _split(votes, placements, firsts, awards[0], Placement.FIRST)
        elif len(present) == 1 or len(firsts) > 1:
            _split(votes, placements, firsts, awards[0] + awards[1], Placement.FIRST)
        else:
            _split(votes, placements, firsts, awards[0], Placement.FIRST)
            rest = [i for i in present if i not in firsts]
            runner_up = max(influence[i] for i in rest)
            seconds = [i for i in rest if influence[i] == runner_up]
            _split(votes, placements, seconds, awards[1], Placement.SECOND)
    return RegionResult(tuple(votes), tuple(placements))


class TieBreak(StrEnum):
    SCORE = "score"
    HAND = "hand"
    BANK = "bank"
    DRAW = "draw"


def determine_winners(
    scores: Sequence[int], hands: Sequence[int], banks: Sequence[int]
) -> tuple[tuple[int, ...], TieBreak]:
    """Most votes, then most cubes in hand, then most cubes in bank; otherwise a draw."""
    contenders = list(range(len(scores)))
    for key, label in ((scores, TieBreak.SCORE), (hands, TieBreak.HAND), (banks, TieBreak.BANK)):
        best = max(key[i] for i in contenders)
        contenders = [i for i in contenders if key[i] == best]
        if len(contenders) == 1:
            return tuple(contenders), label
    return tuple(contenders), TieBreak.DRAW
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_movement.py tests/test_voting.py -v`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add src/primary/engine tests/test_movement.py tests/test_voting.py
git commit -m "Add map distances and vote award rules" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Actions, game state, events and modifiers

**Files:**
- Create: `src/primary/engine/actions.py`, `src/primary/engine/state.py`, `src/primary/engine/events.py`, `src/primary/engine/modifiers.py`, `src/primary/engine/abilities.py`, `src/primary/engine/news_effects.py`, `src/primary/engine/effects.py`
- Test: `tests/test_actions.py`, `tests/test_state.py`, `tests/test_effects.py`

**Interfaces:**
- Consumes: Task 1 data, `TieBreak` and `Placement` (Task 2).
- Produces:
  - `actions.py`: `REGION_BASE = 0`, `CARD_BASE = 14`, `PLAYER_BASE = 21`, `YES = 27`, `NO = 28`, `PASS = 29`, `STOP = 30`, `NUM_ACTIONS = 31`; `region_action(Region) -> int`, `card_action(Card) -> int`, `player_action(int) -> int`, `action_region(int) -> Region`, `action_card(int) -> Card`, `action_player(int) -> int` (decoders raise `ValueError` on the wrong kind), `describe_action(int) -> str`; `DecisionType` IntEnum (16 members, listed in code); `DecisionRequest(seat, decision, legal_actions: tuple[int, ...], card: Card | None = None, region: Region | None = None)`; `GameOver(scores, hands, banks, winners, tiebreak)` with property `draw`; `EngineError(RuntimeError)`, `IllegalActionError(ValueError)`.
  - `state.py`: `PlayerState(candidate, hand, bank, score=0, pawn=None)`; `RoundState` (fields: `queue`, `revealed`, `skipped_slot`, `super_pacs`, `fundraisers`, `rallies`, `rally_extra_used`, `hand_start`, `bank_start`, `path`) with `RoundState.empty(players)`; `GameState` (fields: `players`, `cubes: list[list[int]]` indexed `[region][seat]`, `schedule: dict[int, tuple[Region, ...]]`, `vote_round: dict[Region, int]`, `news_deck`, `start_seat`, `initial_start_seat`, `frozen: list[dict[Card, int]]`, `rnd`, `round`, `voted: list[bool]`, `news_history`, `current_news`) with `num_players`, `turn_order()`, `cubes_on_map(seat)`, `set_schedule(schedule)`; `make_schedule(rng)`, `vote_rounds(schedule)`, `new_game_state(candidates, rng)`.
  - `events.py`: `Phase`, `EventType`, `Source` (StrEnums), `Event` (frozen dataclass), `RegionOutcome(region, placement, votes)`, `RoundSummary`, `EventSink` Protocol (`on_event`, `on_round`), `ListSink` (`events`, `rounds`, `of_type(EventType) -> list[Event]`).
  - `modifiers.py`: `NewsMove` StrEnum (`NONE`, `DEBATE`, `DETOUR`, `LONG_TERM`), `Modifiers` dataclass.
  - `abilities.py`: `ABILITY_MODIFIERS: Mapping[Candidate, Callable[[Modifiers], None]]`.
  - `news_effects.py`: `NEWS_MODIFIERS: Mapping[News, Callable[[Modifiers], None]]` (Task 7 adds `move_options`).
  - `effects.py`: `build_modifiers(candidate: Candidate, news: News | None) -> Modifiers`.

- [ ] **Step 1: Write the failing tests**

`tests/test_actions.py`:

```python
import pytest

from primary.data.board import Region
from primary.data.candidates import MAX_PLAYERS
from primary.data.cards import Card
from primary.engine.actions import (
    NUM_ACTIONS,
    PASS,
    STOP,
    YES,
    action_card,
    action_player,
    action_region,
    card_action,
    describe_action,
    player_action,
    region_action,
)


def test_action_layout():
    assert NUM_ACTIONS == 31
    assert {region_action(r) for r in Region} == set(range(14))
    assert {card_action(c) for c in Card} == set(range(14, 21))
    assert {player_action(s) for s in range(MAX_PLAYERS)} == set(range(21, 27))
    assert (YES, PASS, STOP) == (27, 29, 30)


def test_roundtrips():
    for region in Region:
        assert action_region(region_action(region)) is region
    for card in Card:
        assert action_card(card_action(card)) is card
    for seat in range(MAX_PLAYERS):
        assert action_player(player_action(seat)) == seat


def test_decoding_the_wrong_kind_raises():
    with pytest.raises(ValueError):
        action_region(card_action(Card.RALLY))
    with pytest.raises(ValueError):
        action_card(PASS)


def test_describe_action():
    assert describe_action(region_action(Region.NEW_ENGLAND)) == "region:NEW_ENGLAND"
    assert describe_action(card_action(Card.RALLY)) == "card:RALLY"
    assert describe_action(player_action(2)) == "player:2"
    assert describe_action(PASS) == "pass"
```

`tests/test_state.py`:

```python
import numpy as np
import pytest

from primary.data.board import VOTING_SLOTS, Region
from primary.data.candidates import TOTAL_CUBES, Candidate
from primary.engine.state import make_schedule, new_game_state

C = Candidate


def test_new_game_state_splits_cubes_between_hand_and_bank():  # rule 4
    state = new_game_state([C.GEORGIA_BARNES, C.ANTONIO_PARADES, C.RYAN_GUPTA], np.random.default_rng(0))
    assert [p.hand for p in state.players] == [1, 5, 1]
    assert all(p.hand + p.bank == TOTAL_CUBES for p in state.players)
    assert state.cubes == [[0, 0, 0] for _ in Region]


def test_schedule_votes_each_region_once():  # rule 3
    for seed in range(20):
        schedule = make_schedule(np.random.default_rng(seed))
        assert sorted(r for regions in schedule.values() for r in regions) == list(Region)
        for rnd, regions in schedule.items():
            assert len(regions) == VOTING_SLOTS[rnd]


def test_vote_round_matches_schedule():
    state = new_game_state(list(C)[:5], np.random.default_rng(3))
    for rnd, regions in state.schedule.items():
        for region in regions:
            assert state.vote_round[region] == rnd


def test_same_seed_same_setup():
    a = new_game_state(list(C)[:5], np.random.default_rng(42))
    b = new_game_state(list(C)[:5], np.random.default_rng(42))
    assert (a.schedule, a.news_deck, a.start_seat) == (b.schedule, b.news_deck, b.start_seat)


@pytest.mark.parametrize(
    "candidates",
    [[C.ANNA_PATEL, C.ANNA_PATEL, C.RYAN_GUPTA], [C.ANNA_PATEL, C.RYAN_GUPTA], list(C)[:7]],
)
def test_rejects_invalid_candidate_lists(candidates):  # rule 1
    with pytest.raises(ValueError):
        new_game_state(candidates, np.random.default_rng(0))


def test_turn_order_starts_at_token_holder():  # rule 2
    state = new_game_state(list(C)[:5], np.random.default_rng(0))
    state.start_seat = 3
    assert state.turn_order() == [3, 4, 0, 1, 2]
```

`tests/test_effects.py`:

```python
import pytest

from primary.data.candidates import Candidate
from primary.data.news import News
from primary.engine.abilities import ABILITY_MODIFIERS
from primary.engine.effects import build_modifiers
from primary.engine.modifiers import NewsMove
from primary.engine.news_effects import NEWS_MODIFIERS

C, N = Candidate, News


def test_every_candidate_and_news_card_has_an_effect():
    assert set(ABILITY_MODIFIERS) == set(Candidate)
    assert set(NEWS_MODIFIERS) == set(News)


def test_base_rules():
    m = build_modifiers(C.GEORGIA_BARNES, None)
    assert (m.bus_range, m.plane_cost, m.max_cards, m.positive_ad_placements) == (1, 1, 4, 1)


@pytest.mark.parametrize(
    ("candidate", "field", "value"),
    [
        (C.ANTONIO_PARADES, "bus_range", 2),  # rule 21
        (C.KENNETH_WALKER, "two_fundraiser_bonus", 1),  # rule 22
        (C.BRYCE_MCKAY, "no_award_gain", 1),  # rule 23
        (C.KRISTEN_RAMIREZ, "unique_most_super_pac_bonus", 1),  # rule 24
        (C.DIANE_PROUDIAN, "can_skip_card", True),  # rule 25
        (C.RYAN_GUPTA, "plane_cost", 0),  # rule 26
        (C.CHESTER_FREEMAN, "colocation_gain", 1),  # rule 27
        (C.LILY_NGUYEN, "first_rally_extra", True),  # rule 28
        (C.ERIN_FIELDS, "erin_premove", True),  # rule 29
        (C.CALVIN_MATSUI, "bus_hand_placement", True),  # rule 30
        (C.ANNA_PATEL, "token_influence", 2),  # rule 31
        (C.GEORGIA_BARNES, "home_regions", 3),  # rule 32
    ],
)
def test_ability_modifiers(candidate, field, value):
    assert getattr(build_modifiers(candidate, None), field) == value


@pytest.mark.parametrize(
    ("news", "field", "value"),
    [
        (N.COUNTERMEASURES, "enter_contested_gain", 1),  # rule 34
        (N.OPPO_RESEARCH, "colocation_gain", 2),  # rule 35
        (N.GENEROUS_DONORS, "fundraiser_extra", 1),  # rule 36
        (N.TAX_REFORM, "immediate_gain", 2),  # rule 37
        (N.LAST_MINUTE_DEBATE, "news_move", NewsMove.DEBATE),  # rule 38
        (N.DETOUR, "news_move", NewsMove.DETOUR),  # rule 39
        (N.FROZEN_ASSETS, "freeze_cards", True),  # rule 40
        (N.PAPARAZZI, "move_bank_placement", True),  # rule 41
        (N.SPIN_DOCTOR, "spin_doctor", True),  # rule 42
        (N.FINAL_PUSH, "max_cards", 5),  # rule 43
        (N.LONG_TERM_PLANNING, "news_move", NewsMove.LONG_TERM),  # rule 44
        (N.DEREGULATION, "fundraiser_most_award", (2, 1)),  # rule 45
        (N.EXTRA_AIRTIME, "positive_ad_placements", 2),  # rule 46
        (N.WALL_STREET_SUPPORT, "first_rally_from_bank", True),  # rule 47
        (N.ROAD_TRIP, "bus_range", 2),  # rule 48
        (N.PLUTOCRACY, "extra_super_pac_award", (6, 3)),  # rule 49
        (N.NEW_CONSTITUENTS, "news_swap", True),  # rule 50
        (N.FREQUENT_FLYER, "plane_cost", 0),  # rule 51
    ],
)
def test_news_modifiers(news, field, value):
    assert getattr(build_modifiers(C.GEORGIA_BARNES, news), field) == value


def test_effects_stack():  # rule 52
    assert build_modifiers(C.ANTONIO_PARADES, N.ROAD_TRIP).bus_range == 3
    assert build_modifiers(C.CHESTER_FREEMAN, N.OPPO_RESEARCH).colocation_gain == 3
    assert build_modifiers(C.RYAN_GUPTA, N.FREQUENT_FLYER).plane_cost == 0
    calvin = build_modifiers(C.CALVIN_MATSUI, N.PAPARAZZI)
    assert calvin.bus_hand_placement and calvin.move_bank_placement
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_actions.py tests/test_state.py tests/test_effects.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primary.engine.actions'`.

- [ ] **Step 3: Implement `actions.py`**

```python
"""The flat discrete action space, decision requests and outcomes (spec §4.2–4.3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from primary.data.board import NUM_REGIONS, Region
from primary.data.candidates import MAX_PLAYERS
from primary.data.cards import Card
from primary.engine.voting import TieBreak

REGION_BASE = 0
CARD_BASE = REGION_BASE + NUM_REGIONS
PLAYER_BASE = CARD_BASE + len(Card)
YES = PLAYER_BASE + MAX_PLAYERS
NO = YES + 1
PASS = NO + 1
STOP = PASS + 1
NUM_ACTIONS = STOP + 1


class EngineError(RuntimeError):
    """The engine was used incorrectly or reached an impossible state."""


class IllegalActionError(ValueError):
    """An action outside the current request's legal actions was submitted."""


class DecisionType(IntEnum):
    HOME_REGION = 0
    NEWS_MOVE = 1
    NEWS_SWAP_REGION = 2
    NEWS_SWAP_OWNER = 3
    COMMIT_CARD = 4
    ERIN_SOURCE = 5
    ERIN_DEST = 6
    DIANE_SKIP = 7
    PLANE_TARGET = 8
    BUS_TARGET = 9
    SPIN_DOCTOR_TYPE = 10
    POSITIVE_AD_TARGET = 11
    NEGATIVE_AD_REGION = 12
    NEGATIVE_AD_OWNER = 13
    OPTIONAL_HAND_PLACEMENT = 14
    OPTIONAL_BANK_PLACEMENT = 15


def region_action(region: Region) -> int:
    return REGION_BASE + int(region)


def card_action(card: Card) -> int:
    return CARD_BASE + int(card)


def player_action(seat: int) -> int:
    if not 0 <= seat < MAX_PLAYERS:
        raise ValueError(f"seat {seat} out of range")
    return PLAYER_BASE + seat


def action_region(action: int) -> Region:
    if not REGION_BASE <= action < CARD_BASE:
        raise ValueError(f"action {action} is not a region")
    return Region(action - REGION_BASE)


def action_card(action: int) -> Card:
    if not CARD_BASE <= action < PLAYER_BASE:
        raise ValueError(f"action {action} is not a card")
    return Card(action - CARD_BASE)


def action_player(action: int) -> int:
    if not PLAYER_BASE <= action < YES:
        raise ValueError(f"action {action} is not a player")
    return action - PLAYER_BASE


_NAMED = {YES: "yes", NO: "no", PASS: "pass", STOP: "stop"}


def describe_action(action: int) -> str:
    if action in _NAMED:
        return _NAMED[action]
    if REGION_BASE <= action < CARD_BASE:
        return f"region:{action_region(action).name}"
    if CARD_BASE <= action < PLAYER_BASE:
        return f"card:{action_card(action).name}"
    if PLAYER_BASE <= action < YES:
        return f"player:{action_player(action)}"
    return f"invalid:{action}"


@dataclass(frozen=True)
class DecisionRequest:
    """A player must choose one of ``legal_actions``. ``card``/``region`` give context."""

    seat: int
    decision: DecisionType
    legal_actions: tuple[int, ...]
    card: Card | None = None
    region: Region | None = None


@dataclass(frozen=True)
class GameOver:
    scores: tuple[int, ...]
    hands: tuple[int, ...]
    banks: tuple[int, ...]
    winners: tuple[int, ...]
    tiebreak: TieBreak

    @property
    def draw(self) -> bool:
        return self.tiebreak is TieBreak.DRAW
```

- [ ] **Step 4: Implement `state.py`**

```python
"""Mutable game state. Pure data plus small helpers; the rules live in other modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from primary.data.board import NUM_REGIONS, NUM_ROUNDS, VOTING_SLOTS, Region
from primary.data.candidates import (
    MAX_PLAYERS,
    MIN_PLAYERS,
    STARTING_HAND,
    TOTAL_CUBES,
    Candidate,
)
from primary.data.cards import Card
from primary.data.news import News

Schedule = dict[int, tuple[Region, ...]]


@dataclass
class PlayerState:
    candidate: Candidate
    hand: int
    bank: int
    score: int = 0
    pawn: Region | None = None


@dataclass
class RoundState:
    """Per-round bookkeeping, reset at the start of every round."""

    queue: list[list[Card]]
    revealed: list[list[Card]]
    skipped_slot: list[int | None]
    super_pacs: list[int]
    fundraisers: list[int]
    rallies: list[int]
    rally_extra_used: list[bool]
    hand_start: list[int]
    bank_start: list[int]
    path: list[list[Region]]

    @classmethod
    def empty(cls, players: Sequence[PlayerState]) -> RoundState:
        n = len(players)
        return cls(
            queue=[[] for _ in range(n)],
            revealed=[[] for _ in range(n)],
            skipped_slot=[None] * n,
            super_pacs=[0] * n,
            fundraisers=[0] * n,
            rallies=[0] * n,
            rally_extra_used=[False] * n,
            hand_start=[p.hand for p in players],
            bank_start=[p.bank for p in players],
            path=[[p.pawn] if p.pawn is not None else [] for p in players],
        )


@dataclass
class GameState:
    players: list[PlayerState]
    cubes: list[list[int]]  # cubes[region][seat]
    schedule: Schedule
    vote_round: dict[Region, int]
    news_deck: list[News]
    start_seat: int
    initial_start_seat: int
    frozen: list[dict[Card, int]]
    rnd: RoundState
    round: int = 0
    voted: list[bool] = field(default_factory=lambda: [False] * NUM_REGIONS)
    news_history: list[News] = field(default_factory=list)
    current_news: News | None = None

    @property
    def num_players(self) -> int:
        return len(self.players)

    def turn_order(self) -> list[int]:
        n = self.num_players
        return [(self.start_seat + i) % n for i in range(n)]

    def cubes_on_map(self, seat: int) -> int:
        return sum(row[seat] for row in self.cubes)

    def set_schedule(self, schedule: Schedule) -> None:
        self.schedule = schedule
        self.vote_round = vote_rounds(schedule)


def vote_rounds(schedule: Schedule) -> dict[Region, int]:
    return {region: rnd for rnd, regions in schedule.items() for region in regions}


def make_schedule(rng: np.random.Generator) -> Schedule:
    """Randomly assign the 14 regions to the 14 voting slots (rule 3)."""
    order = [Region(int(i)) for i in rng.permutation(NUM_REGIONS)]
    schedule: Schedule = {}
    k = 0
    for rnd in range(1, NUM_ROUNDS + 1):
        n = VOTING_SLOTS[rnd]
        schedule[rnd] = tuple(order[k : k + n])
        k += n
    return schedule


def new_game_state(candidates: Sequence[Candidate], rng: np.random.Generator) -> GameState:
    """Initial state before home regions are chosen. RNG draw order: schedule, news, token."""
    n = len(candidates)
    if not MIN_PLAYERS <= n <= MAX_PLAYERS:
        raise ValueError(f"need {MIN_PLAYERS}-{MAX_PLAYERS} players, got {n}")
    if len(set(candidates)) != n:
        raise ValueError(f"candidates must be distinct, got {[c.name for c in candidates]}")
    players = [PlayerState(c, STARTING_HAND[c], TOTAL_CUBES - STARTING_HAND[c]) for c in candidates]
    schedule = make_schedule(rng)
    news_deck = [News(int(i)) for i in rng.permutation(len(News))]
    start = int(rng.integers(n))
    return GameState(
        players=players,
        cubes=[[0] * n for _ in range(NUM_REGIONS)],
        schedule=schedule,
        vote_round=vote_rounds(schedule),
        news_deck=news_deck,
        start_seat=start,
        initial_start_seat=start,
        frozen=[{} for _ in range(n)],
        rnd=RoundState.empty(players),
    )
```

- [ ] **Step 5: Implement `events.py`**

```python
"""Structured events emitted by the engine (spec §6). Logging never changes game behaviour."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from primary.data.board import Region
from primary.data.candidates import Candidate
from primary.data.cards import Card
from primary.data.news import News
from primary.engine.actions import DecisionType
from primary.engine.voting import Placement


class Phase(StrEnum):
    SETUP = "setup"
    NEWS = "news"
    COMMIT = "commit"
    EXECUTE = "execute"
    ROUND_END = "round_end"


class EventType(StrEnum):
    DECISION = "decision"
    NEWS_DRAWN = "news_drawn"
    NEWS_DISCARDED = "news_discarded"
    CARD_COMMITTED = "card_committed"
    CARD_REVEALED = "card_revealed"
    CARD_SKIPPED = "card_skipped"
    MOVE = "move"
    CUBE_PLACED = "cube_placed"
    CUBE_REMOVED = "cube_removed"
    CUBE_MOVED = "cube_moved"
    CUBE_RETURNED = "cube_returned"
    GAIN = "gain"
    PAYMENT = "payment"
    NO_EFFECT = "no_effect"
    VOTE_AWARDED = "vote_awarded"


class Source(StrEnum):
    HAND = "hand"
    BANK = "bank"


@dataclass(frozen=True)
class Event:
    """One atomic change. ``hand_*``/``bank_*`` describe ``cube_owner`` if set, else ``seat``."""

    seq: int
    round: int
    phase: Phase
    event_type: EventType
    seat: int | None = None
    candidate: Candidate | None = None
    card: Card | None = None
    news: News | None = None
    from_region: Region | None = None
    to_region: Region | None = None
    cube_owner: int | None = None
    amount: int | None = None
    source: Source | None = None
    hand_before: int | None = None
    hand_after: int | None = None
    bank_before: int | None = None
    bank_after: int | None = None
    pawn_before: Region | None = None
    pawn_after: Region | None = None
    decision_type: DecisionType | None = None
    action: int | None = None
    forced: bool | None = None
    detail: str | None = None


@dataclass(frozen=True)
class RegionOutcome:
    region: Region
    placement: Placement
    votes: int


@dataclass(frozen=True)
class RoundSummary:
    round: int
    seat: int
    candidate: Candidate
    start_region: Region | None
    end_region: Region | None
    path: tuple[Region, ...]
    hand_start: int
    hand_end: int
    bank_start: int
    bank_end: int
    cards: tuple[Card, ...]
    skipped_slot: int | None
    outcomes: tuple[RegionOutcome, ...]
    votes_gained: int


class EventSink(Protocol):
    def on_event(self, event: Event) -> None: ...

    def on_round(self, summary: RoundSummary) -> None: ...


class ListSink:
    """Keeps everything in memory. Used by tests and small tools."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.rounds: list[RoundSummary] = []

    def on_event(self, event: Event) -> None:
        self.events.append(event)

    def on_round(self, summary: RoundSummary) -> None:
        self.rounds.append(summary)

    def of_type(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.event_type is event_type]
```

- [ ] **Step 6: Implement `modifiers.py`, `abilities.py`, `news_effects.py`, `effects.py`**

`src/primary/engine/modifiers.py`:

```python
"""Per-player, per-round rule parameters. Defaults are the base rules; abilities and News edit them."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class NewsMove(StrEnum):
    NONE = "none"
    DEBATE = "last_minute_debate"
    DETOUR = "detour"
    LONG_TERM = "long_term_planning"


@dataclass
class Modifiers:
    # setup
    home_regions: int = 1
    # news phase
    immediate_gain: int = 0
    news_move: NewsMove = NewsMove.NONE
    news_swap: bool = False
    # commit phase
    max_cards: int = 4
    freeze_cards: bool = False
    # execute phase
    erin_premove: bool = False
    can_skip_card: bool = False
    plane_cost: int = 1
    bus_range: int = 1
    bus_hand_placement: bool = False
    move_bank_placement: bool = False
    fundraiser_extra: int = 0
    positive_ad_placements: int = 1
    spin_doctor: bool = False
    first_rally_from_bank: bool = False
    first_rally_extra: bool = False
    enter_contested_gain: int = 0
    # round end
    token_influence: int = 0
    extra_super_pac_award: tuple[int, int] = (0, 0)  # (unique most, tied most)
    unique_most_super_pac_bonus: int = 0
    fundraiser_most_award: tuple[int, int] = (0, 0)  # (unique most, tied most)
    two_fundraiser_bonus: int = 0
    no_award_gain: int = 0
    colocation_gain: int = 0
```

`src/primary/engine/abilities.py`:

```python
"""Candidate abilities as Modifiers edits (spec §3 rules 21–32)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from primary.data.candidates import Candidate
from primary.engine.modifiers import Modifiers

ModifierFn = Callable[[Modifiers], None]


def antonio_parades(m: Modifiers) -> None:
    """Bus may move up to 2 adjacent regions away."""
    m.bus_range += 1


def kenneth_walker(m: Modifiers) -> None:
    """2+ Fundraisers in a round: gain 1 extra cube."""
    m.two_fundraiser_bonus += 1


def bryce_mckay(m: Modifiers) -> None:
    """After a vote where you win no delegates: gain 1 cube."""
    m.no_award_gain += 1


def kristen_ramirez(m: Modifiers) -> None:
    """Unique most Super PACs: gain 1 extra cube."""
    m.unique_most_super_pac_bonus += 1


def diane_proudian(m: Modifiers) -> None:
    """May skip executing 1 Action card per round."""
    m.can_skip_card = True


def ryan_gupta(m: Modifiers) -> None:
    """Plane cards are free."""
    m.plane_cost = 0


def chester_freeman(m: Modifiers) -> None:
    """End the round with an opponent's pawn: gain 1 cube."""
    m.colocation_gain += 1


def lily_nguyen(m: Modifiers) -> None:
    """First effective Rally may place 1 extra cube from hand."""
    m.first_rally_extra = True


def erin_fields(m: Modifiers) -> None:
    """Before the first Action card, may move 1 own cube anywhere."""
    m.erin_premove = True


def calvin_matsui(m: Modifiers) -> None:
    """Bus may place 1 cube from hand onto the destination."""
    m.bus_hand_placement = True


def anna_patel(m: Modifiers) -> None:
    """Pawn counts as 2 influence in its region."""
    m.token_influence += 2


def georgia_barnes(m: Modifiers) -> None:
    """Three home regions with 2 bank cubes each."""
    m.home_regions = 3


ABILITY_MODIFIERS: Mapping[Candidate, ModifierFn] = MappingProxyType(
    {
        Candidate.ANTONIO_PARADES: antonio_parades,
        Candidate.KENNETH_WALKER: kenneth_walker,
        Candidate.BRYCE_MCKAY: bryce_mckay,
        Candidate.KRISTEN_RAMIREZ: kristen_ramirez,
        Candidate.DIANE_PROUDIAN: diane_proudian,
        Candidate.RYAN_GUPTA: ryan_gupta,
        Candidate.CHESTER_FREEMAN: chester_freeman,
        Candidate.LILY_NGUYEN: lily_nguyen,
        Candidate.ERIN_FIELDS: erin_fields,
        Candidate.CALVIN_MATSUI: calvin_matsui,
        Candidate.ANNA_PATEL: anna_patel,
        Candidate.GEORGIA_BARNES: georgia_barnes,
    }
)
```

`src/primary/engine/news_effects.py`:

```python
"""News cards as Modifiers edits (spec §3 rules 33–51)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from primary.data.news import News
from primary.engine.modifiers import Modifiers, NewsMove

ModifierFn = Callable[[Modifiers], None]


def countermeasures(m: Modifiers) -> None:
    m.enter_contested_gain += 1


def oppo_research(m: Modifiers) -> None:
    m.colocation_gain += 2


def generous_donors(m: Modifiers) -> None:
    m.fundraiser_extra += 1


def tax_reform(m: Modifiers) -> None:
    m.immediate_gain += 2


def last_minute_debate(m: Modifiers) -> None:
    m.news_move = NewsMove.DEBATE


def detour(m: Modifiers) -> None:
    m.news_move = NewsMove.DETOUR


def frozen_assets(m: Modifiers) -> None:
    m.freeze_cards = True


def paparazzi(m: Modifiers) -> None:
    m.move_bank_placement = True


def spin_doctor(m: Modifiers) -> None:
    m.spin_doctor = True


def final_push(m: Modifiers) -> None:
    m.max_cards = 5


def long_term_planning(m: Modifiers) -> None:
    m.news_move = NewsMove.LONG_TERM


def deregulation(m: Modifiers) -> None:
    m.fundraiser_most_award = (2, 1)


def extra_airtime(m: Modifiers) -> None:
    m.positive_ad_placements = 2


def wall_street_support(m: Modifiers) -> None:
    m.first_rally_from_bank = True


def road_trip(m: Modifiers) -> None:
    m.bus_range += 1


def plutocracy(m: Modifiers) -> None:
    m.extra_super_pac_award = (6, 3)


def new_constituents(m: Modifiers) -> None:
    m.news_swap = True


def frequent_flyer(m: Modifiers) -> None:
    m.plane_cost = 0


NEWS_MODIFIERS: Mapping[News, ModifierFn] = MappingProxyType(
    {
        News.COUNTERMEASURES: countermeasures,
        News.OPPO_RESEARCH: oppo_research,
        News.GENEROUS_DONORS: generous_donors,
        News.TAX_REFORM: tax_reform,
        News.LAST_MINUTE_DEBATE: last_minute_debate,
        News.DETOUR: detour,
        News.FROZEN_ASSETS: frozen_assets,
        News.PAPARAZZI: paparazzi,
        News.SPIN_DOCTOR: spin_doctor,
        News.FINAL_PUSH: final_push,
        News.LONG_TERM_PLANNING: long_term_planning,
        News.DEREGULATION: deregulation,
        News.EXTRA_AIRTIME: extra_airtime,
        News.WALL_STREET_SUPPORT: wall_street_support,
        News.ROAD_TRIP: road_trip,
        News.PLUTOCRACY: plutocracy,
        News.NEW_CONSTITUENTS: new_constituents,
        News.FREQUENT_FLYER: frequent_flyer,
    }
)
```

`src/primary/engine/effects.py`:

```python
"""Combine base rules, a candidate's ability and the current News card (rule 52: effects stack)."""

from __future__ import annotations

from primary.data.candidates import Candidate
from primary.data.news import News
from primary.engine.abilities import ABILITY_MODIFIERS
from primary.engine.modifiers import Modifiers
from primary.engine.news_effects import NEWS_MODIFIERS


def build_modifiers(candidate: Candidate, news: News | None) -> Modifiers:
    m = Modifiers()
    ABILITY_MODIFIERS[candidate](m)
    if news is not None:
        NEWS_MODIFIERS[news](m)
    return m
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_actions.py tests/test_state.py tests/test_effects.py -v`
Expected: all pass (4 + 8 + 33 tests).

- [ ] **Step 8: Commit**

```bash
git add src/primary/engine tests/test_actions.py tests/test_state.py tests/test_effects.py
git commit -m "Add action space, game state, events and ability/News modifiers" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Table (primitive mutations), invariants and test helpers

**Files:**
- Create: `src/primary/engine/table.py`, `src/primary/engine/invariants.py`
- Create: `tests/helpers.py`
- Test: `tests/test_table.py`

**Interfaces:**
- Consumes: `GameState`, `RoundState` (Task 3), `Event`, `EventSink`, `EventType`, `Phase`, `Source`, `ListSink` (Task 3), `build_modifiers`, `Modifiers` (Task 3), `EngineError` (Task 3).
- Produces:
  - `Table(state: GameState, sink: EventSink | None = None)` with attributes `state`, `sink`, `phase: Phase`, `mods: list[Modifiers]` and methods:
    `emit(event_type, seat=None, **fields) -> None`; `set_news(news: News | None) -> None` (sets `state.current_news` and rebuilds `mods`); `begin_round(rnd: int) -> None`; `draw_news() -> News`;
    `gain(seat, n, reason) -> int` (bank→hand, capped, returns amount moved); `pay(seat, n, reason) -> bool` (hand→bank);
    `place(seat, region, n, source: Source, reason) -> bool`; `remove(region, owner, by_seat, reason) -> None` (to owner's bank; raises `EngineError` if no such cube);
    `move_cube(owner, src, dst, by_seat, reason, *, return_if_voted: bool) -> None`; `return_region(region) -> None`;
    `move_pawn(seat, dst, reason) -> None` (applies Countermeasures); `swap_pawn_with_cube(seat, region, owner) -> None`; `no_effect(seat, card, reason) -> None`.
  - `InvariantError(AssertionError)`, `check_invariants(state: GameState) -> None`.
  - `tests/helpers.py`: `DEFAULT_CANDIDATES`, `fixed_schedule(assign)`, `make_state(...) -> GameState`, `make_table(...) -> tuple[Table, ListSink]`, `drive(flow, answers) -> tuple[list[DecisionRequest], object]`.

- [ ] **Step 1: Write the test helpers**

`tests/helpers.py`:

```python
"""Scenario builders shared by the engine tests."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from primary.data.board import NUM_ROUNDS, VOTING_SLOTS, Region
from primary.data.candidates import STARTING_HAND, TOTAL_CUBES, Candidate
from primary.data.news import News
from primary.engine.actions import DecisionRequest
from primary.engine.events import ListSink, Phase
from primary.engine.state import GameState, RoundState, Schedule, new_game_state
from primary.engine.table import Table

# Abilities that rarely interfere with card-level tests. Seat 0 is Kenneth, seat 4 Georgia.
DEFAULT_CANDIDATES = (
    Candidate.KENNETH_WALKER,
    Candidate.KRISTEN_RAMIREZ,
    Candidate.BRYCE_MCKAY,
    Candidate.CHESTER_FREEMAN,
    Candidate.GEORGIA_BARNES,
)


def fixed_schedule(assign: Mapping[int, Sequence[Region]]) -> Schedule:
    """A valid schedule with the given regions in the given rounds; the rest fill in by index."""
    used = {r for regions in assign.values() for r in regions}
    pool = [r for r in Region if r not in used]
    schedule: Schedule = {}
    for rnd in range(1, NUM_ROUNDS + 1):
        regions = list(assign.get(rnd, ()))
        assert len(regions) <= VOTING_SLOTS[rnd], f"round {rnd} has {VOTING_SLOTS[rnd]} slots"
        while len(regions) < VOTING_SLOTS[rnd]:
            regions.append(pool.pop(0))
        schedule[rnd] = tuple(regions)
    return schedule


def make_state(
    candidates: Sequence[Candidate] = DEFAULT_CANDIDATES,
    *,
    round_: int = 2,
    pawns: Sequence[Region] | None = None,
    hands: Sequence[int] | None = None,
    cubes: Mapping[tuple[Region, int], int] | None = None,
    scores: Sequence[int] | None = None,
    voted: Iterable[Region] = (),
    schedule: Mapping[int, Sequence[Region]] | None = None,
    seed: int = 0,
) -> GameState:
    """A mid-game state. Pawns default to Region(seat); banks are derived so each player owns 30."""
    state = new_game_state(candidates, np.random.default_rng(seed))
    state.set_schedule(fixed_schedule(schedule or {}))
    state.round = round_
    state.start_seat = 0
    for (region, seat), count in (cubes or {}).items():
        state.cubes[region][seat] = count
    for seat, player in enumerate(state.players):
        player.pawn = pawns[seat] if pawns is not None else Region(seat)
        player.hand = hands[seat] if hands is not None else STARTING_HAND[player.candidate]
        player.score = scores[seat] if scores is not None else 0
        player.bank = TOTAL_CUBES - player.hand - state.cubes_on_map(seat)
        assert player.bank >= 0, f"seat {seat} has more than {TOTAL_CUBES} cubes"
    for region in voted:
        state.voted[region] = True
    state.rnd = RoundState.empty(state.players)
    return state


def make_table(
    state: GameState | None = None,
    *,
    news: News | None = None,
    phase: Phase = Phase.EXECUTE,
    **kwargs: Any,
) -> tuple[Table, ListSink]:
    if state is None:
        state = make_state(**kwargs)
    sink = ListSink()
    table = Table(state, sink)
    table.phase = phase
    table.set_news(news)
    return table, sink


def drive(
    flow: Generator[DecisionRequest, int, Any], answers: Iterable[int] = ()
) -> tuple[list[DecisionRequest], Any]:
    """Run an engine flow, answering each request from ``answers`` in order."""
    pending = list(answers)
    requests: list[DecisionRequest] = []
    try:
        request = next(flow)
        while True:
            requests.append(request)
            assert pending, f"no answer scripted for {request}"
            answer = pending.pop(0)
            assert answer in request.legal_actions, f"{answer} not legal for {request}"
            request = flow.send(answer)
    except StopIteration as stop:
        assert not pending, f"unused answers: {pending}"
        return requests, stop.value
```

- [ ] **Step 2: Write the failing tests**

`tests/test_table.py`:

```python
import pytest

from primary.data.board import Region
from primary.data.news import News
from primary.engine.actions import EngineError
from primary.engine.events import EventType, Source
from primary.engine.invariants import InvariantError, check_invariants
from tests.helpers import make_state, make_table

R = Region


def test_gain_moves_bank_to_hand_and_is_capped():  # rule 16
    table, sink = make_table(hands=[28, 3, 3, 3, 1])
    player = table.state.players[0]
    assert player.bank == 2
    assert table.gain(0, 5, "test") == 2
    assert (player.hand, player.bank) == (30, 0)
    event = sink.of_type(EventType.GAIN)[-1]
    assert (event.amount, event.hand_before, event.hand_after, event.bank_after) == (2, 28, 30, 0)


def test_gain_from_empty_bank_moves_nothing():  # rule 16
    table, _ = make_table(hands=[30, 3, 3, 3, 1])
    assert table.gain(0, 4, "test") == 0
    assert table.state.players[0].hand == 30


def test_pay_moves_hand_to_bank():
    table, _ = make_table(hands=[2, 3, 3, 3, 1])
    assert table.pay(0, 1, "plane")
    assert table.state.players[0].hand == 1
    assert not table.pay(0, 2, "plane")
    assert table.state.players[0].hand == 1


def test_place_requires_enough_cubes_in_source():  # rule 16
    table, _ = make_table(hands=[1, 3, 3, 3, 1])
    assert not table.place(0, R.WEST, 2, Source.HAND, "rally")
    assert table.state.cubes[R.WEST][0] == 0
    assert table.place(0, R.WEST, 1, Source.HAND, "ad")
    assert table.state.cubes[R.WEST][0] == 1
    assert table.state.players[0].hand == 0


def test_place_from_bank():
    table, _ = make_table()
    bank = table.state.players[1].bank
    assert table.place(1, R.SOUTH, 2, Source.BANK, "home")
    assert table.state.players[1].bank == bank - 2
    assert table.state.cubes[R.SOUTH][1] == 2


def test_remove_returns_cube_to_owner_bank_even_own_cube():  # rule 13
    table, sink = make_table(cubes={(R.SOUTH, 0): 1, (R.SOUTH, 2): 2})
    bank2 = table.state.players[2].bank
    table.remove(R.SOUTH, 2, by_seat=0, reason="negative_ad")
    assert table.state.cubes[R.SOUTH][2] == 1
    assert table.state.players[2].bank == bank2 + 1
    table.remove(R.SOUTH, 0, by_seat=0, reason="negative_ad")
    assert table.state.cubes[R.SOUTH][0] == 0
    first = sink.of_type(EventType.CUBE_REMOVED)[0]
    assert (first.seat, first.cube_owner, first.bank_after) == (0, 2, bank2 + 1)


def test_remove_missing_cube_raises():
    table, _ = make_table()
    with pytest.raises(EngineError):
        table.remove(R.SOUTH, 1, by_seat=0, reason="negative_ad")


def test_move_cube_onto_voted_region():  # rules 15, 50
    table, _ = make_table(cubes={(R.WEST, 0): 1, (R.SOUTH, 0): 1}, voted=[R.NORTH])
    bank = table.state.players[0].bank
    table.move_cube(0, R.WEST, R.NORTH, by_seat=0, reason="swap", return_if_voted=True)
    assert table.state.cubes[R.WEST][0] == 0
    assert table.state.cubes[R.NORTH][0] == 0
    assert table.state.players[0].bank == bank + 1
    table.move_cube(0, R.SOUTH, R.NORTH, by_seat=0, reason="erin", return_if_voted=False)
    assert table.state.cubes[R.NORTH][0] == 1


def test_return_region_sends_every_cube_home():
    table, sink = make_table(cubes={(R.WEST, 0): 2, (R.WEST, 3): 1})
    banks = [p.bank for p in table.state.players]
    table.return_region(R.WEST)
    assert table.state.cubes[R.WEST] == [0, 0, 0, 0, 0]
    assert table.state.players[0].bank == banks[0] + 2
    assert table.state.players[3].bank == banks[3] + 1
    assert len(sink.of_type(EventType.CUBE_RETURNED)) == 2


def test_move_pawn_tracks_path():
    table, sink = make_table()
    table.move_pawn(0, R.MIDWEST, "bus")
    assert table.state.players[0].pawn is R.MIDWEST
    assert table.state.rnd.path[0] == [R.PACIFIC_NW, R.MIDWEST]
    move = sink.of_type(EventType.MOVE)[0]
    assert (move.pawn_before, move.pawn_after) == (R.PACIFIC_NW, R.MIDWEST)


def test_countermeasures_gain_only_when_an_opponent_has_cubes_there():  # rule 34
    table, _ = make_table(news=News.COUNTERMEASURES, cubes={(R.MIDWEST, 1): 1, (R.SOUTH, 0): 1})
    hand = table.state.players[0].hand
    table.move_pawn(0, R.MIDWEST, "bus")
    assert table.state.players[0].hand == hand + 1
    table.move_pawn(0, R.SOUTH, "bus")  # only our own cube there
    assert table.state.players[0].hand == hand + 1


def test_no_countermeasures_without_the_news_card():
    table, _ = make_table(cubes={(R.MIDWEST, 1): 1})
    hand = table.state.players[0].hand
    table.move_pawn(0, R.MIDWEST, "bus")
    assert table.state.players[0].hand == hand


def test_swap_pawn_with_cube():  # rule 50
    table, _ = make_table(cubes={(R.SOUTH, 3): 1})
    table.swap_pawn_with_cube(0, R.SOUTH, 3)
    assert table.state.players[0].pawn is R.SOUTH
    assert table.state.cubes[R.PACIFIC_NW][3] == 1
    assert table.state.cubes[R.SOUTH][3] == 0


def test_swap_into_voted_region_returns_the_cube():  # rule 50
    table, _ = make_table(cubes={(R.SOUTH, 3): 1}, voted=[R.PACIFIC_NW])
    bank = table.state.players[3].bank
    table.swap_pawn_with_cube(0, R.SOUTH, 3)
    assert table.state.cubes[R.PACIFIC_NW][3] == 0
    assert table.state.players[3].bank == bank + 1


def test_draw_news_redraws_frozen_assets_in_final_round():  # rule 40
    table, sink = make_table(round_=11)
    table.state.news_deck = [News.FROZEN_ASSETS, News.TAX_REFORM]
    assert table.draw_news() is News.TAX_REFORM
    assert table.state.news_history == [News.TAX_REFORM]
    assert [e.event_type for e in sink.events] == [EventType.NEWS_DISCARDED, EventType.NEWS_DRAWN]
    assert table.mods[0].immediate_gain == 2


def test_frozen_assets_is_kept_before_the_final_round():  # rule 40
    table, _ = make_table(round_=10)
    table.state.news_deck = [News.FROZEN_ASSETS, News.TAX_REFORM]
    assert table.draw_news() is News.FROZEN_ASSETS


def test_check_invariants_detects_a_lost_cube():
    state = make_state(cubes={(R.WEST, 0): 2})
    check_invariants(state)
    state.players[0].bank -= 1
    with pytest.raises(InvariantError):
        check_invariants(state)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_table.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primary.engine.table'`.

- [ ] **Step 4: Implement `table.py`**

```python
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
```

- [ ] **Step 5: Implement `invariants.py`**

```python
"""Conservation checks run after every step in tests and in debug mode (spec §5.5)."""

from __future__ import annotations

from primary.data.candidates import TOTAL_CUBES
from primary.engine.state import GameState


class InvariantError(AssertionError):
    """The game state violates a rule that must always hold."""


def check_invariants(state: GameState) -> None:
    for seat, player in enumerate(state.players):
        column = [row[seat] for row in state.cubes]
        if player.hand < 0 or player.bank < 0 or min(column) < 0:
            raise InvariantError(f"seat {seat} has a negative cube count")
        total = player.hand + player.bank + sum(column)
        if total != TOTAL_CUBES:
            raise InvariantError(f"seat {seat} owns {total} cubes, expected {TOTAL_CUBES}")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_table.py -v`
Expected: 17 passed.

- [ ] **Step 7: Commit**

```bash
git add src/primary/engine/table.py src/primary/engine/invariants.py tests/helpers.py tests/test_table.py
git commit -m "Add logged state mutations, invariants and test scenario helpers" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Decisions and Action card resolution

**Files:**
- Create: `src/primary/engine/decisions.py`, `src/primary/engine/resolve.py`
- Test: `tests/test_decisions.py`, `tests/test_cards.py`

**Interfaces:**
- Consumes: `Table` (Task 4), `reachable` (Task 2), action helpers and `DecisionRequest`/`DecisionType`/`EngineError`/`IllegalActionError` (Task 3), `drive`/`make_table` (Task 4).
- Produces:
  - `Decide = Generator[DecisionRequest, int, int]`, `Flow = Generator[DecisionRequest, int, None]`.
  - `ask(table, seat, decision, legal: Iterable[int], *, card=None, region=None) -> Decide` — yields a `DecisionRequest` and returns the chosen action; a single legal action is returned without yielding. Every decision is logged as a `decision` event with `forced` set.
  - `resolve_card(table: Table, seat: int, card: Card) -> Flow`.

- [ ] **Step 1: Write the failing tests**

`tests/test_decisions.py`:

```python
import pytest

from primary.data.board import Region
from primary.engine.actions import DecisionType, EngineError, region_action
from primary.engine.decisions import ask
from primary.engine.events import EventType
from tests.helpers import drive, make_table

R = region_action


def test_ask_yields_request_and_returns_answer():
    table, sink = make_table()
    flow = ask(table, 0, DecisionType.PLANE_TARGET, [R(Region.WEST), R(Region.SOUTH)])
    requests, answer = drive(flow, [R(Region.SOUTH)])
    assert requests[0].legal_actions == (R(Region.WEST), R(Region.SOUTH))
    assert answer == R(Region.SOUTH)
    event = sink.of_type(EventType.DECISION)[0]
    assert (event.action, event.forced, event.detail) == (R(Region.SOUTH), False, "region:SOUTH")


def test_single_option_is_forced_without_asking():
    table, sink = make_table()
    requests, answer = drive(ask(table, 0, DecisionType.BUS_TARGET, [R(Region.WEST)]))
    assert requests == []
    assert answer == R(Region.WEST)
    assert sink.of_type(EventType.DECISION)[0].forced is True


def test_no_options_is_an_engine_error():
    table, _ = make_table()
    with pytest.raises(EngineError):
        drive(ask(table, 0, DecisionType.BUS_TARGET, []))
```

`tests/test_cards.py`:

```python
from primary.data.board import Region
from primary.data.candidates import Candidate
from primary.data.cards import Card
from primary.data.news import News
from primary.engine.actions import NO, YES, DecisionType, card_action, player_action, region_action
from primary.engine.events import EventType
from primary.engine.resolve import resolve_card
from tests.helpers import DEFAULT_CANDIDATES, drive, make_table

R = region_action
REG = Region
C = Candidate


def with_seat0(candidate):
    return (candidate, *DEFAULT_CANDIDATES[1:])


def hands(h0):
    return [h0, 3, 3, 3, 1]


# -- Super PAC and Fundraiser ----------------------------------------------------------------


def test_super_pac_is_counted_for_round_end():  # rule 8
    table, _ = make_table()
    drive(resolve_card(table, 0, Card.SUPER_PAC))
    assert table.state.rnd.super_pacs[0] == 1


def test_fundraiser_gains_one():  # rule 9
    table, _ = make_table(hands=hands(2))
    drive(resolve_card(table, 0, Card.FUNDRAISER))
    assert table.state.players[0].hand == 3
    assert table.state.rnd.fundraisers[0] == 1


def test_generous_donors_adds_one_per_fundraiser():  # rule 36
    table, sink = make_table(news=News.GENEROUS_DONORS, hands=hands(2))
    drive(resolve_card(table, 0, Card.FUNDRAISER))
    assert table.state.players[0].hand == 4
    assert [e.detail for e in sink.of_type(EventType.GAIN)] == ["fundraiser", "generous_donors"]


# -- Plane -----------------------------------------------------------------------------------


def test_plane_pays_one_and_moves_anywhere_else():  # rule 10
    table, _ = make_table(hands=hands(3))
    requests, _ = drive(resolve_card(table, 0, Card.PLANE), [R(REG.NEW_ENGLAND)])
    assert requests[0].decision is DecisionType.PLANE_TARGET
    assert R(REG.PACIFIC_NW) not in requests[0].legal_actions
    assert len(requests[0].legal_actions) == 13
    assert table.state.players[0].pawn is REG.NEW_ENGLAND
    assert table.state.players[0].hand == 2


def test_plane_with_empty_hand_has_no_effect():  # rule 10
    table, sink = make_table(hands=hands(0))
    requests, _ = drive(resolve_card(table, 0, Card.PLANE))
    assert requests == []
    assert table.state.players[0].pawn is REG.PACIFIC_NW
    assert sink.of_type(EventType.NO_EFFECT)[0].detail == "insufficient_hand"


def test_ryan_plane_is_free():  # rule 26
    table, _ = make_table(candidates=with_seat0(C.RYAN_GUPTA), hands=hands(0))
    drive(resolve_card(table, 0, Card.PLANE), [R(REG.SOUTH)])
    assert table.state.players[0].pawn is REG.SOUTH
    assert table.state.players[0].hand == 0


def test_frequent_flyer_plane_is_free():  # rule 51
    table, _ = make_table(news=News.FREQUENT_FLYER, hands=hands(0))
    drive(resolve_card(table, 0, Card.PLANE), [R(REG.SOUTH)])
    assert table.state.players[0].pawn is REG.SOUTH


def test_paparazzi_plane_may_place_a_bank_cube():  # rule 41
    table, _ = make_table(news=News.PAPARAZZI, hands=hands(3))
    bank = table.state.players[0].bank
    requests, _ = drive(resolve_card(table, 0, Card.PLANE), [R(REG.SOUTH), YES])
    assert requests[1].decision is DecisionType.OPTIONAL_BANK_PLACEMENT
    assert table.state.cubes[REG.SOUTH][0] == 1
    assert table.state.players[0].bank == bank + 1 - 1  # +1 plane payment, -1 placement


# -- Bus -------------------------------------------------------------------------------------


def test_bus_moves_one_step():  # rule 11
    table, _ = make_table()
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.NORTH)])
    assert set(requests[0].legal_actions) == {R(REG.NORTH), R(REG.SOUTH_WEST), R(REG.WEST)}
    assert table.state.players[0].pawn is REG.NORTH


def test_antonio_bus_moves_up_to_two_steps():  # rule 21
    table, _ = make_table(
        candidates=with_seat0(C.ANTONIO_PARADES),
        pawns=[REG.NEW_ENGLAND, REG.WEST, REG.SOUTH_WEST, REG.GREATER_TEXAS, REG.NORTH],
    )
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.APPALACHIA)])
    assert set(requests[0].legal_actions) == {
        R(REG.NORTH_EAST),
        R(REG.EARLY_MIDWEST),
        R(REG.APPALACHIA),
    }


def test_road_trip_extends_bus_by_one():  # rule 48
    table, _ = make_table(
        news=News.ROAD_TRIP,
        pawns=[REG.NEW_ENGLAND, REG.WEST, REG.SOUTH_WEST, REG.GREATER_TEXAS, REG.NORTH],
    )
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.NORTH_EAST)])
    assert len(requests[0].legal_actions) == 3


def test_antonio_with_road_trip_moves_up_to_three():  # rules 48, 52
    table, _ = make_table(
        candidates=with_seat0(C.ANTONIO_PARADES),
        news=News.ROAD_TRIP,
        pawns=[REG.NEW_ENGLAND, REG.WEST, REG.SOUTH_WEST, REG.GREATER_TEXAS, REG.NORTH],
    )
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.SOUTH)])
    assert len(requests[0].legal_actions) == 6


def test_calvin_bus_may_place_from_hand():  # rule 30
    table, _ = make_table(candidates=with_seat0(C.CALVIN_MATSUI), hands=hands(3))
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.WEST), YES])
    assert requests[1].decision is DecisionType.OPTIONAL_HAND_PLACEMENT
    assert table.state.cubes[REG.WEST][0] == 1
    assert table.state.players[0].hand == 2


def test_calvin_may_decline():  # rule 30
    table, _ = make_table(candidates=with_seat0(C.CALVIN_MATSUI), hands=hands(3))
    drive(resolve_card(table, 0, Card.BUS), [R(REG.WEST), NO])
    assert table.state.cubes[REG.WEST][0] == 0


def test_calvin_with_paparazzi_stacks():  # rule 52
    table, _ = make_table(
        candidates=with_seat0(C.CALVIN_MATSUI), news=News.PAPARAZZI, hands=hands(3)
    )
    requests, _ = drive(resolve_card(table, 0, Card.BUS), [R(REG.WEST), YES, YES])
    assert [r.decision for r in requests] == [
        DecisionType.BUS_TARGET,
        DecisionType.OPTIONAL_HAND_PLACEMENT,
        DecisionType.OPTIONAL_BANK_PLACEMENT,
    ]
    assert table.state.cubes[REG.WEST][0] == 2


# -- Ads -------------------------------------------------------------------------------------


def test_positive_ad_places_one_from_hand_anywhere():  # rule 12
    table, _ = make_table(hands=hands(2))
    requests, _ = drive(resolve_card(table, 0, Card.POSITIVE_AD), [R(REG.NEW_ENGLAND)])
    assert len(requests[0].legal_actions) == 14
    assert table.state.cubes[REG.NEW_ENGLAND][0] == 1
    assert table.state.players[0].hand == 1


def test_positive_ad_with_empty_hand_has_no_effect():  # rule 12
    table, sink = make_table(hands=hands(0))
    requests, _ = drive(resolve_card(table, 0, Card.POSITIVE_AD))
    assert requests == []
    assert len(sink.of_type(EventType.NO_EFFECT)) == 1


def test_extra_airtime_places_two_independently():  # rule 46
    table, _ = make_table(news=News.EXTRA_AIRTIME, hands=hands(2))
    drive(resolve_card(table, 0, Card.POSITIVE_AD), [R(REG.WEST), R(REG.SOUTH)])
    assert table.state.cubes[REG.WEST][0] == 1
    assert table.state.cubes[REG.SOUTH][0] == 1


def test_extra_airtime_with_one_cube_places_one():  # rule 46
    table, sink = make_table(news=News.EXTRA_AIRTIME, hands=hands(1))
    drive(resolve_card(table, 0, Card.POSITIVE_AD), [R(REG.WEST)])
    assert table.state.cubes[REG.WEST][0] == 1
    assert len(sink.of_type(EventType.NO_EFFECT)) == 1


def test_negative_ad_removes_the_chosen_cube_to_its_owner_bank():  # rule 13
    table, _ = make_table(cubes={(REG.SOUTH, 2): 2, (REG.SOUTH, 3): 1, (REG.WEST, 0): 1})
    bank3 = table.state.players[3].bank
    requests, _ = drive(resolve_card(table, 0, Card.NEGATIVE_AD), [R(REG.SOUTH), player_action(3)])
    assert set(requests[0].legal_actions) == {R(REG.SOUTH), R(REG.WEST)}
    assert requests[1].decision is DecisionType.NEGATIVE_AD_OWNER
    assert set(requests[1].legal_actions) == {player_action(2), player_action(3)}
    assert table.state.cubes[REG.SOUTH][3] == 0
    assert table.state.players[3].bank == bank3 + 1


def test_negative_ad_with_one_owner_is_forced():
    table, _ = make_table(cubes={(REG.SOUTH, 2): 1, (REG.WEST, 1): 1})
    requests, _ = drive(resolve_card(table, 0, Card.NEGATIVE_AD), [R(REG.SOUTH)])
    assert len(requests) == 1
    assert table.state.cubes[REG.SOUTH][2] == 0


def test_negative_ad_must_remove_own_cube_if_it_is_the_only_one():  # rule 13
    table, sink = make_table(cubes={(REG.WEST, 0): 1})
    requests, _ = drive(resolve_card(table, 0, Card.NEGATIVE_AD))
    assert requests == []
    assert table.state.cubes[REG.WEST][0] == 0
    assert all(e.forced for e in sink.of_type(EventType.DECISION))


def test_negative_ad_on_empty_map_has_no_effect():  # rule 13
    table, sink = make_table()
    drive(resolve_card(table, 0, Card.NEGATIVE_AD))
    assert sink.of_type(EventType.NO_EFFECT)[0].detail == "empty_map"


def test_spin_doctor_lets_positive_ad_act_as_negative_ad():  # rule 42
    table, _ = make_table(news=News.SPIN_DOCTOR, cubes={(REG.SOUTH, 1): 1})
    requests, _ = drive(
        resolve_card(table, 0, Card.POSITIVE_AD), [card_action(Card.NEGATIVE_AD)]
    )
    assert requests[0].decision is DecisionType.SPIN_DOCTOR_TYPE
    assert table.state.cubes[REG.SOUTH][1] == 0


# -- Rally -----------------------------------------------------------------------------------


def test_rally_places_two_on_the_pawn_region():  # rule 14
    table, _ = make_table(hands=hands(3))
    drive(resolve_card(table, 0, Card.RALLY))
    assert table.state.cubes[REG.PACIFIC_NW][0] == 2
    assert table.state.players[0].hand == 1


def test_rally_needs_two_cubes():  # rule 14
    table, sink = make_table(hands=hands(1))
    drive(resolve_card(table, 0, Card.RALLY))
    assert table.state.cubes[REG.PACIFIC_NW][0] == 0
    assert table.state.players[0].hand == 1
    assert sink.of_type(EventType.NO_EFFECT)[0].detail == "insufficient_hand"


def test_wall_street_first_rally_uses_bank():  # rule 47
    table, _ = make_table(news=News.WALL_STREET_SUPPORT, hands=hands(2))
    bank = table.state.players[0].bank
    drive(resolve_card(table, 0, Card.RALLY))
    assert (table.state.players[0].hand, table.state.players[0].bank) == (2, bank - 2)
    drive(resolve_card(table, 0, Card.RALLY))
    assert table.state.players[0].hand == 0
    assert table.state.cubes[REG.PACIFIC_NW][0] == 4


def test_wall_street_needs_two_bank_cubes():  # rule 47
    table, sink = make_table(news=News.WALL_STREET_SUPPORT, hands=hands(29))  # bank = 1
    drive(resolve_card(table, 0, Card.RALLY))
    assert sink.of_type(EventType.NO_EFFECT)[0].detail == "insufficient_bank"
    drive(resolve_card(table, 0, Card.RALLY))  # second Rally uses the hand
    assert table.state.players[0].hand == 27


def test_lily_extra_cube_on_first_effective_rally():  # rule 28
    table, _ = make_table(candidates=with_seat0(C.LILY_NGUYEN), hands=hands(5))
    requests, _ = drive(resolve_card(table, 0, Card.RALLY), [YES])
    assert requests[0].decision is DecisionType.OPTIONAL_HAND_PLACEMENT
    assert table.state.cubes[REG.PACIFIC_NW][0] == 3
    requests, _ = drive(resolve_card(table, 0, Card.RALLY))
    assert requests == []
    assert table.state.cubes[REG.PACIFIC_NW][0] == 5


def test_lily_extra_waits_for_an_effective_rally():  # rule 28
    table, _ = make_table(candidates=with_seat0(C.LILY_NGUYEN), hands=hands(1))
    assert drive(resolve_card(table, 0, Card.RALLY))[0] == []  # fails: hand 1
    table.gain(0, 2, "test")
    requests, _ = drive(resolve_card(table, 0, Card.RALLY), [YES])
    assert len(requests) == 1
    assert table.state.cubes[REG.PACIFIC_NW][0] == 3


def test_lily_extra_comes_from_hand_under_wall_street():  # rule 47
    table, _ = make_table(
        candidates=with_seat0(C.LILY_NGUYEN), news=News.WALL_STREET_SUPPORT, hands=hands(1)
    )
    drive(resolve_card(table, 0, Card.RALLY), [YES])
    assert table.state.cubes[REG.PACIFIC_NW][0] == 3
    assert table.state.players[0].hand == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_decisions.py tests/test_cards.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primary.engine.decisions'`.

- [ ] **Step 3: Implement `decisions.py`**

```python
"""Asking a player for a decision (spec §4.2). Single-option decisions resolve automatically."""

from __future__ import annotations

from collections.abc import Generator, Iterable

from primary.data.board import Region
from primary.data.cards import Card
from primary.engine.actions import (
    DecisionRequest,
    DecisionType,
    EngineError,
    IllegalActionError,
    describe_action,
)
from primary.engine.events import EventType
from primary.engine.table import Table

Decide = Generator[DecisionRequest, int, int]
Flow = Generator[DecisionRequest, int, None]


def ask(
    table: Table,
    seat: int,
    decision: DecisionType,
    legal: Iterable[int],
    *,
    card: Card | None = None,
    region: Region | None = None,
) -> Decide:
    options = tuple(sorted(set(legal)))
    if not options:
        raise EngineError(f"{decision.name} for seat {seat} has no legal actions")
    if len(options) == 1:
        action, forced = options[0], True
    else:
        action = yield DecisionRequest(seat, decision, options, card, region)
        forced = False
        if action not in options:
            raise IllegalActionError(f"{action} is not one of {options} for {decision.name}")
    table.emit(
        EventType.DECISION,
        seat,
        decision_type=decision,
        action=action,
        forced=forced,
        card=card,
        detail=describe_action(action),
    )
    return action
```

- [ ] **Step 4: Implement `resolve.py`**

```python
"""Resolution of one Action card (spec §3 rules 8–14, 21, 26, 28, 30, 36, 41–42, 46–48, 51)."""

from __future__ import annotations

from primary.data.board import Region
from primary.data.cards import Card
from primary.engine.actions import (
    NO,
    YES,
    DecisionType,
    EngineError,
    action_card,
    action_player,
    action_region,
    card_action,
    player_action,
    region_action,
)
from primary.engine.decisions import Flow, ask
from primary.engine.events import Source
from primary.engine.movement import reachable
from primary.engine.table import Table


def resolve_card(table: Table, seat: int, card: Card) -> Flow:
    if card is Card.SUPER_PAC:
        table.state.rnd.super_pacs[seat] += 1  # paid out at round end
    elif card is Card.FUNDRAISER:
        _fundraiser(table, seat)
    elif card is Card.PLANE:
        yield from _plane(table, seat)
    elif card is Card.BUS:
        yield from _bus(table, seat)
    elif card in (Card.POSITIVE_AD, Card.NEGATIVE_AD):
        yield from _advertise(table, seat, card)
    elif card is Card.RALLY:
        yield from _rally(table, seat)


def _pawn(table: Table, seat: int) -> Region:
    pawn = table.state.players[seat].pawn
    if pawn is None:
        raise EngineError(f"seat {seat} has no pawn on the map")
    return pawn


def _fundraiser(table: Table, seat: int) -> None:
    table.state.rnd.fundraisers[seat] += 1
    table.gain(seat, 1, "fundraiser")
    extra = table.mods[seat].fundraiser_extra
    if extra:
        table.gain(seat, extra, "generous_donors")


def _optional_placement(
    table: Table, seat: int, card: Card, region: Region, source: Source, reason: str
) -> Flow:
    player = table.state.players[seat]
    available = player.hand if source is Source.HAND else player.bank
    if available < 1:
        return
    decision = (
        DecisionType.OPTIONAL_HAND_PLACEMENT
        if source is Source.HAND
        else DecisionType.OPTIONAL_BANK_PLACEMENT
    )
    choice = yield from ask(table, seat, decision, [YES, NO], card=card, region=region)
    if choice == YES:
        table.place(seat, region, 1, source, reason)


def _plane(table: Table, seat: int) -> Flow:
    mods = table.mods[seat]
    if mods.plane_cost and not table.pay(seat, mods.plane_cost, "plane"):
        table.no_effect(seat, Card.PLANE, "insufficient_hand")
        return
    here = _pawn(table, seat)
    options = [region_action(r) for r in Region if r != here]
    choice = yield from ask(table, seat, DecisionType.PLANE_TARGET, options, card=Card.PLANE)
    dst = action_region(choice)
    table.move_pawn(seat, dst, "plane")
    if mods.move_bank_placement:
        yield from _optional_placement(table, seat, Card.PLANE, dst, Source.BANK, "paparazzi")


def _bus(table: Table, seat: int) -> Flow:
    mods = table.mods[seat]
    options = [region_action(r) for r in reachable(_pawn(table, seat), mods.bus_range)]
    choice = yield from ask(table, seat, DecisionType.BUS_TARGET, options, card=Card.BUS)
    dst = action_region(choice)
    table.move_pawn(seat, dst, "bus")
    if mods.bus_hand_placement:
        yield from _optional_placement(table, seat, Card.BUS, dst, Source.HAND, "calvin")
    if mods.move_bank_placement:
        yield from _optional_placement(table, seat, Card.BUS, dst, Source.BANK, "paparazzi")


def _advertise(table: Table, seat: int, card: Card) -> Flow:
    effect = card
    if table.mods[seat].spin_doctor:
        both = [card_action(Card.POSITIVE_AD), card_action(Card.NEGATIVE_AD)]
        choice = yield from ask(table, seat, DecisionType.SPIN_DOCTOR_TYPE, both, card=card)
        effect = action_card(choice)
    if effect is Card.POSITIVE_AD:
        yield from _positive_ad(table, seat, card)
    else:
        yield from _negative_ad(table, seat, card)


def _positive_ad(table: Table, seat: int, card: Card) -> Flow:
    player = table.state.players[seat]
    for _ in range(table.mods[seat].positive_ad_placements):
        if player.hand < 1:
            table.no_effect(seat, card, "insufficient_hand")
            return
        options = [region_action(r) for r in Region]
        choice = yield from ask(table, seat, DecisionType.POSITIVE_AD_TARGET, options, card=card)
        table.place(seat, action_region(choice), 1, Source.HAND, "positive_ad")


def _negative_ad(table: Table, seat: int, card: Card) -> Flow:
    cubes = table.state.cubes
    regions = [r for r in Region if sum(cubes[r]) > 0]
    if not regions:
        table.no_effect(seat, card, "empty_map")
        return
    choice = yield from ask(
        table, seat, DecisionType.NEGATIVE_AD_REGION, [region_action(r) for r in regions], card=card
    )
    region = action_region(choice)
    owners = [player_action(o) for o, count in enumerate(cubes[region]) if count > 0]
    pick = yield from ask(
        table, seat, DecisionType.NEGATIVE_AD_OWNER, owners, card=card, region=region
    )
    table.remove(region, action_player(pick), by_seat=seat, reason="negative_ad")


def _rally(table: Table, seat: int) -> Flow:
    rnd, mods = table.state.rnd, table.mods[seat]
    first = rnd.rallies[seat] == 0
    rnd.rallies[seat] += 1
    source = Source.BANK if first and mods.first_rally_from_bank else Source.HAND
    here = _pawn(table, seat)
    if not table.place(seat, here, 2, source, "rally"):
        table.no_effect(seat, Card.RALLY, f"insufficient_{source.value}")
        return
    if mods.first_rally_extra and not rnd.rally_extra_used[seat]:
        rnd.rally_extra_used[seat] = True
        yield from _optional_placement(table, seat, Card.RALLY, here, Source.HAND, "lily")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_decisions.py tests/test_cards.py -v`
Expected: 3 + 31 passed.

- [ ] **Step 6: Commit**

```bash
git add src/primary/engine/decisions.py src/primary/engine/resolve.py tests/test_decisions.py tests/test_cards.py
git commit -m "Add decision requests and Action card resolution" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Round end — bonuses, voting, cube returns, summaries

**Files:**
- Create: `src/primary/engine/round_end.py`
- Test: `tests/test_round_end.py`

**Interfaces:**
- Consumes: `Table` (Task 4), `compute_awards`, `determine_winners`, `Placement`, `RegionResult` (Task 2), `GameOver`, `RegionOutcome`, `RoundSummary` (Task 3).
- Produces: `SUPER_PAC_AWARD = (4, 2)`; `Tier` enum (`NONE`, `UNIQUE`, `TIED`); `most_tiers(counts: Sequence[int]) -> list[Tier]`; `influence(state, mods, region) -> list[int]`; `resolve_round_end(table: Table) -> None`; `finish_game(state: GameState) -> GameOver`.

- [ ] **Step 1: Write the failing tests**

`tests/test_round_end.py`:

```python
from primary.data.board import Region
from primary.data.candidates import Candidate
from primary.data.cards import Card
from primary.data.news import News
from primary.engine.events import RegionOutcome
from primary.engine.round_end import Tier, finish_game, most_tiers, resolve_round_end
from primary.engine.voting import Placement, TieBreak
from tests.helpers import make_state, make_table

REG = Region


def hands_of(table):
    return [p.hand for p in table.state.players]


def test_most_tiers():
    assert most_tiers([0, 0, 0]) == [Tier.NONE] * 3
    assert most_tiers([1, 0, 0]) == [Tier.UNIQUE, Tier.NONE, Tier.NONE]
    assert most_tiers([2, 2, 1]) == [Tier.TIED, Tier.TIED, Tier.NONE]


# -- Super PAC, Plutocracy, Kristen, Deregulation, Kenneth (round 1 has no vote) -------------


def test_super_pac_unique_most_gains_four():  # rule 8
    table, _ = make_table(round_=1)
    table.state.rnd.super_pacs[:] = [1, 0, 0, 0, 0]
    resolve_round_end(table)
    assert hands_of(table) == [6, 3, 3, 3, 1]


def test_super_pac_lone_card_beats_zeros():  # rule 8
    table, _ = make_table(round_=1)
    table.state.rnd.super_pacs[:] = [0, 0, 1, 0, 0]
    resolve_round_end(table)
    assert hands_of(table) == [2, 3, 7, 3, 1]


def test_super_pac_tie_gains_two_each():  # rules 8, 24
    table, _ = make_table(round_=1)
    table.state.rnd.super_pacs[:] = [2, 2, 0, 0, 0]
    resolve_round_end(table)
    assert hands_of(table) == [4, 5, 3, 3, 1]  # Kristen (seat 1) gets no bonus on a tie


def test_everyone_tied_everyone_gains_two():  # rule 8
    table, _ = make_table(round_=1)
    table.state.rnd.super_pacs[:] = [4, 4, 4, 4, 4]
    resolve_round_end(table)
    assert hands_of(table) == [4, 5, 5, 5, 3]


def test_kristen_bonus_when_unique_most():  # rule 24
    table, _ = make_table(round_=1)
    table.state.rnd.super_pacs[:] = [0, 2, 0, 0, 0]
    resolve_round_end(table)
    assert hands_of(table)[1] == 3 + 4 + 1


def test_plutocracy_adds_to_the_super_pac_award():  # rule 49
    table, _ = make_table(round_=1, news=News.PLUTOCRACY)
    table.state.rnd.super_pacs[:] = [3, 1, 0, 0, 0]
    resolve_round_end(table)
    assert hands_of(table)[:2] == [2 + 4 + 6, 3]


def test_plutocracy_tie():  # rule 49
    table, _ = make_table(round_=1, news=News.PLUTOCRACY)
    table.state.rnd.super_pacs[:] = [2, 2, 0, 0, 0]
    resolve_round_end(table)
    assert hands_of(table)[:2] == [2 + 2 + 3, 3 + 2 + 3]


def test_deregulation_unique_and_tied():  # rule 45
    table, _ = make_table(round_=1, news=News.DEREGULATION)
    table.state.rnd.fundraisers[:] = [1, 2, 0, 0, 0]
    resolve_round_end(table)
    assert hands_of(table)[:2] == [2, 5]
    table, _ = make_table(round_=1, news=News.DEREGULATION)
    table.state.rnd.fundraisers[:] = [1, 1, 0, 0, 0]
    resolve_round_end(table)
    assert hands_of(table)[:2] == [3, 4]


def test_kenneth_bonus_needs_two_fundraisers():  # rule 22
    table, _ = make_table(round_=1)
    table.state.rnd.fundraisers[:] = [2, 0, 0, 0, 0]
    resolve_round_end(table)
    assert hands_of(table)[0] == 3
    table, _ = make_table(round_=1)
    table.state.rnd.fundraisers[:] = [1, 0, 0, 0, 0]
    resolve_round_end(table)
    assert hands_of(table)[0] == 2


# -- Voting and cube returns ---------------------------------------------------------------


def test_voting_scores_and_returns_cubes_but_others_persist():  # rules 15, 17
    table, _ = make_table(
        schedule={2: (REG.NORTH,)},
        cubes={(REG.NORTH, 0): 3, (REG.NORTH, 1): 1, (REG.SOUTH, 0): 2},
    )
    banks = [p.bank for p in table.state.players]
    resolve_round_end(table)
    state = table.state
    assert [p.score for p in state.players] == [10, 0, 0, 0, 0]
    assert state.voted[REG.NORTH]
    assert state.cubes[REG.NORTH] == [0, 0, 0, 0, 0]
    assert state.players[0].bank == banks[0] + 3
    assert state.players[1].bank == banks[1] + 1
    assert state.cubes[REG.SOUTH][0] == 2


def test_cubes_on_a_previously_voted_region_return():  # rule 15
    table, _ = make_table(round_=1, voted=[REG.WEST], cubes={(REG.WEST, 0): 2})
    bank = table.state.players[0].bank
    resolve_round_end(table)
    assert table.state.cubes[REG.WEST][0] == 0
    assert table.state.players[0].bank == bank + 2


def test_anna_token_counts_as_two_influence():  # rules 19, 31
    table, _ = make_table(
        candidates=(
            Candidate.ANNA_PATEL,
            Candidate.KENNETH_WALKER,
            Candidate.KRISTEN_RAMIREZ,
            Candidate.BRYCE_MCKAY,
            Candidate.CHESTER_FREEMAN,
        ),
        schedule={2: (REG.NORTH,)},
        pawns=[REG.NORTH, REG.WEST, REG.SOUTH_WEST, REG.GREATER_TEXAS, REG.PACIFIC_NW],
        cubes={(REG.NORTH, 1): 1},
    )
    resolve_round_end(table)
    assert table.state.players[0].score == 10


def test_anna_token_alone_takes_both_awards():  # rules 18, 19
    table, _ = make_table(
        candidates=(
            Candidate.ANNA_PATEL,
            Candidate.KENNETH_WALKER,
            Candidate.KRISTEN_RAMIREZ,
            Candidate.BRYCE_MCKAY,
            Candidate.CHESTER_FREEMAN,
        ),
        schedule={2: (REG.WEST,)},
        pawns=[REG.WEST, REG.PACIFIC_NW, REG.SOUTH_WEST, REG.GREATER_TEXAS, REG.NORTH],
    )
    resolve_round_end(table)
    assert table.state.players[0].score == 24


def test_bryce_gains_after_a_vote_he_does_not_win():  # rule 23
    table, _ = make_table(schedule={2: (REG.NORTH,)}, cubes={(REG.NORTH, 0): 1})
    resolve_round_end(table)
    assert table.state.players[2].hand == 4


def test_bryce_second_place_counts_as_a_win():  # rule 23
    table, _ = make_table(schedule={2: (REG.WEST,)}, cubes={(REG.WEST, 0): 3, (REG.WEST, 2): 1})
    resolve_round_end(table)
    assert table.state.players[2].score == 7
    assert table.state.players[2].hand == 3


def test_bryce_does_not_trigger_in_a_non_voting_round():  # rule 23
    table, _ = make_table(round_=1)
    resolve_round_end(table)
    assert table.state.players[2].hand == 3


def test_chester_gains_when_sharing_a_region():  # rule 27
    table, _ = make_table(
        round_=1, pawns=[REG.PACIFIC_NW, REG.WEST, REG.SOUTH_WEST, REG.MIDWEST, REG.MIDWEST]
    )
    resolve_round_end(table)
    assert hands_of(table)[3:] == [4, 1]


def test_oppo_research_stacks_with_chester():  # rules 35, 52
    table, _ = make_table(
        round_=1,
        news=News.OPPO_RESEARCH,
        pawns=[REG.PACIFIC_NW, REG.WEST, REG.SOUTH_WEST, REG.MIDWEST, REG.MIDWEST],
    )
    resolve_round_end(table)
    assert hands_of(table) == [2, 3, 3, 6, 3]


# -- Frozen Assets, token, summaries, final scoring -----------------------------------------


def test_frozen_assets_freezes_committed_cards():  # rule 40
    table, _ = make_table(round_=1, news=News.FROZEN_ASSETS)
    table.state.rnd.queue[0].extend([Card.RALLY, Card.RALLY, Card.FUNDRAISER, Card.BUS])
    resolve_round_end(table)
    assert table.state.frozen[0] == {Card.RALLY: 2, Card.FUNDRAISER: 1, Card.BUS: 1}
    assert table.state.frozen[1] == {}


def test_frozen_cards_clear_after_one_round():  # rule 40
    table, _ = make_table(round_=1)
    table.state.frozen[0] = {Card.PLANE: 1}
    resolve_round_end(table)
    assert table.state.frozen[0] == {}


def test_starting_player_token_passes_clockwise():  # rule 2
    table, _ = make_table(round_=1)
    table.state.start_seat = 4
    resolve_round_end(table)
    assert table.state.start_seat == 0


def test_round_summary_is_emitted_for_every_player():
    table, sink = make_table(schedule={2: (REG.NORTH,)}, cubes={(REG.NORTH, 0): 1})
    resolve_round_end(table)
    assert len(sink.rounds) == 5
    mine = sink.rounds[0]
    assert mine.outcomes == (RegionOutcome(REG.NORTH, Placement.FIRST, 10),)
    assert (mine.votes_gained, mine.start_region, mine.path) == (10, REG.PACIFIC_NW, (REG.PACIFIC_NW,))


def test_finish_game_applies_tie_breakers():  # rule 20
    state = make_state(scores=[10, 10, 0, 0, 0], hands=[2, 2, 3, 3, 1])
    result = finish_game(state)
    assert result.winners == (0, 1)
    assert result.tiebreak is TieBreak.DRAW
    assert result.draw
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_round_end.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primary.engine.round_end'`.

- [ ] **Step 3: Implement `round_end.py`**

```python
"""End-of-round resolution (spec §5.4 step 4; rules 2, 8, 15, 19, 22–24, 27, 35, 40, 45, 49)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from enum import Enum

from primary.data.board import AWARDS, Region
from primary.engine.actions import GameOver
from primary.engine.events import EventType, Phase, RegionOutcome, RoundSummary
from primary.engine.modifiers import Modifiers
from primary.engine.state import GameState
from primary.engine.table import Table
from primary.engine.voting import Placement, RegionResult, compute_awards, determine_winners

SUPER_PAC_AWARD = (4, 2)  # (unique most, tied most)
_WON = (Placement.FIRST, Placement.SECOND)


class Tier(Enum):
    NONE = 0
    UNIQUE = 1
    TIED = 2


def most_tiers(counts: Sequence[int]) -> list[Tier]:
    """Who played the most of something. Zero is never 'the most'."""
    best = max(counts, default=0)
    if best <= 0:
        return [Tier.NONE] * len(counts)
    tier = Tier.UNIQUE if list(counts).count(best) == 1 else Tier.TIED
    return [tier if c == best else Tier.NONE for c in counts]


def influence(state: GameState, mods: Sequence[Modifiers], region: Region) -> list[int]:
    """Cubes on ``region`` per seat, plus Anna's token (rule 19)."""
    return [
        state.cubes[region][seat] + (mods[seat].token_influence if p.pawn == region else 0)
        for seat, p in enumerate(state.players)
    ]


def resolve_round_end(table: Table) -> None:
    table.phase = Phase.ROUND_END
    _card_bonuses(table)
    results = _vote(table)
    _no_award_bonus(table, results)
    _colocation_bonus(table)
    _return_cubes(table)
    _summarize(table, results)
    _freeze_cards(table)
    state = table.state
    state.start_seat = (state.start_seat + 1) % state.num_players


def finish_game(state: GameState) -> GameOver:
    scores = tuple(p.score for p in state.players)
    hands = tuple(p.hand for p in state.players)
    banks = tuple(p.bank for p in state.players)
    winners, tiebreak = determine_winners(scores, hands, banks)
    return GameOver(scores, hands, banks, winners, tiebreak)


def _bonus(table: Table, seat: int, amount: int, reason: str) -> None:
    if amount > 0:
        table.gain(seat, amount, reason)


def _card_bonuses(table: Table) -> None:
    state = table.state
    pacs = most_tiers(state.rnd.super_pacs)
    funds = most_tiers(state.rnd.fundraisers)
    for seat in state.turn_order():
        m = table.mods[seat]
        if pacs[seat] is Tier.UNIQUE:
            _bonus(table, seat, SUPER_PAC_AWARD[0], "super_pac")
            _bonus(table, seat, m.extra_super_pac_award[0], "plutocracy")
            _bonus(table, seat, m.unique_most_super_pac_bonus, "kristen")
        elif pacs[seat] is Tier.TIED:
            _bonus(table, seat, SUPER_PAC_AWARD[1], "super_pac_tie")
            _bonus(table, seat, m.extra_super_pac_award[1], "plutocracy_tie")
        if funds[seat] is Tier.UNIQUE:
            _bonus(table, seat, m.fundraiser_most_award[0], "deregulation")
        elif funds[seat] is Tier.TIED:
            _bonus(table, seat, m.fundraiser_most_award[1], "deregulation_tie")
        if state.rnd.fundraisers[seat] >= 2:
            _bonus(table, seat, m.two_fundraiser_bonus, "kenneth")


def _vote(table: Table) -> dict[Region, RegionResult]:
    state = table.state
    results: dict[Region, RegionResult] = {}
    for region in state.schedule.get(state.round, ()):
        result = compute_awards(influence(state, table.mods, region), AWARDS[region])
        for seat in state.turn_order():
            votes = result.votes[seat]
            if votes:
                state.players[seat].score += votes
                table.emit(
                    EventType.VOTE_AWARDED,
                    seat,
                    to_region=region,
                    amount=votes,
                    detail=result.placements[seat].value,
                )
        state.voted[region] = True
        results[region] = result
    return results


def _no_award_bonus(table: Table, results: dict[Region, RegionResult]) -> None:
    if not results:
        return
    for seat in table.state.turn_order():
        if all(r.placements[seat] not in _WON for r in results.values()):
            _bonus(table, seat, table.mods[seat].no_award_gain, "bryce")


def _colocation_bonus(table: Table) -> None:
    players = table.state.players
    for seat in table.state.turn_order():
        pawn = players[seat].pawn
        shared = any(o != seat and other.pawn == pawn for o, other in enumerate(players))
        if pawn is not None and shared:
            _bonus(table, seat, table.mods[seat].colocation_gain, "colocation")


def _return_cubes(table: Table) -> None:
    state = table.state
    for region in Region:
        if state.voted[region] and any(state.cubes[region]):
            table.return_region(region)


def _summarize(table: Table, results: dict[Region, RegionResult]) -> None:
    if table.sink is None:
        return
    state, rnd = table.state, table.state.rnd
    for seat, player in enumerate(state.players):
        outcomes = tuple(
            RegionOutcome(region, r.placements[seat], r.votes[seat]) for region, r in results.items()
        )
        path = tuple(rnd.path[seat])
        table.sink.on_round(
            RoundSummary(
                round=state.round,
                seat=seat,
                candidate=player.candidate,
                start_region=path[0] if path else None,
                end_region=player.pawn,
                path=path,
                hand_start=rnd.hand_start[seat],
                hand_end=player.hand,
                bank_start=rnd.bank_start[seat],
                bank_end=player.bank,
                cards=tuple(rnd.queue[seat]),
                skipped_slot=rnd.skipped_slot[seat],
                outcomes=outcomes,
                votes_gained=sum(o.votes for o in outcomes),
            )
        )


def _freeze_cards(table: Table) -> None:
    state = table.state
    for seat in range(state.num_players):
        frozen = table.mods[seat].freeze_cards
        state.frozen[seat] = dict(Counter(state.rnd.queue[seat])) if frozen else {}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_round_end.py -v`
Expected: 24 passed.

- [ ] **Step 5: Commit**

```bash
git add src/primary/engine/round_end.py tests/test_round_end.py
git commit -m "Add round-end bonuses, voting, cube returns and round summaries" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Round phases and the `Game` driver

**Files:**
- Modify: `src/primary/engine/news_effects.py` (add `move_options`)
- Create: `src/primary/engine/phases.py`, `src/primary/engine/game.py`
- Modify: `tests/helpers.py` (add `play_random`)
- Test: `tests/test_phases.py`, `tests/test_game.py`

**Interfaces:**
- Consumes: everything from Tasks 2–6.
- Produces:
  - `move_options(kind: NewsMove, state: GameState, seat: int) -> tuple[tuple[Region, ...], bool]` (destinations, may decline).
  - `setup_phase(table) -> Flow`, `news_phase(table) -> Flow`, `available_cards(state, seat) -> dict[Card, int]`, `commit_phase(table) -> Flow`, `execute_phase(table) -> Flow`.
  - `Outcome = DecisionRequest | GameOver`; `Game(num_players: int = 5, sink: EventSink | None = None, check_invariants: bool = False)` with `reset(seed: int, candidates: Sequence[Candidate]) -> Outcome`, `step(action: int) -> Outcome`, properties `table`, `state`, attributes `num_players`, `seed`, `pending: DecisionRequest | None`, `result: GameOver | None`, `history: list[int]` (non-forced actions only).
  - `tests/helpers.play_random(game, seed, candidates=DEFAULT_CANDIDATES) -> GameOver`.

- [ ] **Step 1: Add `play_random` to the test helpers**

Append to `tests/helpers.py`:

```python
def play_random(game: Any, seed: int, candidates: Sequence[Candidate] = DEFAULT_CANDIDATES) -> Any:
    """Play a whole game with uniformly random legal actions (engine RNG seeded by ``seed``)."""
    rng = np.random.default_rng(seed + 1)
    outcome = game.reset(seed, candidates)
    while isinstance(outcome, DecisionRequest):
        outcome = game.step(int(rng.choice(outcome.legal_actions)))
    return outcome
```

- [ ] **Step 2: Write the failing tests**

`tests/test_phases.py`:

```python
import numpy as np

from primary.data.board import Region
from primary.data.candidates import Candidate
from primary.data.cards import Card
from primary.data.news import News
from primary.engine.actions import NO, PASS, STOP, YES, DecisionType, card_action, region_action
from primary.engine.events import EventType, ListSink, Phase
from primary.engine.phases import commit_phase, execute_phase, news_phase, setup_phase
from primary.engine.state import new_game_state
from primary.engine.table import Table
from tests.helpers import DEFAULT_CANDIDATES, drive, make_table

R, K, REG = region_action, card_action, Region


# -- setup ---------------------------------------------------------------------------------


def test_setup_runs_in_turn_order_and_georgia_picks_three():  # rules 5, 32
    state = new_game_state(DEFAULT_CANDIDATES, np.random.default_rng(0))
    state.start_seat = 2
    table = Table(state, ListSink())
    answers = [
        R(REG.SOUTH),  # seat 2
        R(REG.SOUTH),  # seat 3 shares it
        R(REG.WEST),  # seat 4 Georgia x3
        R(REG.NORTH),
        R(REG.NEW_ENGLAND),
        R(REG.MIDWEST),  # seat 0
        R(REG.SOUTH),  # seat 1
    ]
    requests, _ = drive(setup_phase(table), answers)
    assert [r.seat for r in requests] == [2, 3, 4, 4, 4, 0, 1]
    assert R(REG.WEST) not in requests[3].legal_actions
    assert [state.cubes[REG.SOUTH][s] for s in (1, 2, 3)] == [2, 2, 2]
    georgia = state.players[4]
    assert georgia.pawn is REG.WEST
    assert [state.cubes[r][4] for r in (REG.WEST, REG.NORTH, REG.NEW_ENGLAND)] == [2, 2, 2]
    assert (georgia.hand, georgia.bank) == (1, 23)


# -- news ----------------------------------------------------------------------------------


def news_table(news, **kwargs):
    table, sink = make_table(phase=Phase.NEWS, **kwargs)
    table.state.news_deck = [news]
    return table, sink


def test_tax_reform_gives_everyone_two():  # rules 33, 37
    table, _ = news_table(News.TAX_REFORM)
    drive(news_phase(table))
    assert [p.hand for p in table.state.players] == [4, 5, 5, 5, 3]
    assert table.state.news_history == [News.TAX_REFORM]


def test_detour_forces_everyone_to_move():  # rule 39
    table, _ = news_table(News.DETOUR)
    requests, _ = drive(news_phase(table), [R(REG.SOUTH)] * 5)
    assert all(PASS not in r.legal_actions for r in requests)
    assert R(REG.PACIFIC_NW) not in requests[0].legal_actions
    assert all(p.pawn is REG.SOUTH for p in table.state.players)


def test_last_minute_debate_offers_this_and_next_round_regions():  # rule 38
    table, _ = news_table(
        News.LAST_MINUTE_DEBATE, schedule={2: (REG.NORTH,), 3: (REG.SOUTH, REG.WEST)}
    )
    requests, _ = drive(news_phase(table), [PASS, R(REG.NORTH), PASS, PASS, PASS])
    assert set(requests[0].legal_actions) == {R(REG.NORTH), R(REG.SOUTH), R(REG.WEST), PASS}
    assert set(requests[1].legal_actions) == {R(REG.NORTH), R(REG.SOUTH), PASS}
    assert set(requests[4].legal_actions) == {R(REG.SOUTH), R(REG.WEST), PASS}
    assert table.state.players[1].pawn is REG.NORTH


def test_long_term_planning_offers_final_round_regions():  # rule 44
    table, _ = news_table(
        News.LONG_TERM_PLANNING, schedule={11: (REG.SOUTH, REG.NEW_ENGLAND, REG.MIDWEST)}
    )
    requests, _ = drive(news_phase(table), [R(REG.NEW_ENGLAND), PASS, PASS, PASS, PASS])
    assert set(requests[0].legal_actions) == {
        R(REG.SOUTH),
        R(REG.NEW_ENGLAND),
        R(REG.MIDWEST),
        PASS,
    }
    assert table.state.players[0].pawn is REG.NEW_ENGLAND


def test_new_constituents_swaps_pawn_with_any_cube():  # rule 50
    table, _ = news_table(News.NEW_CONSTITUENTS, cubes={(REG.SOUTH, 3): 2})
    requests, _ = drive(news_phase(table), [R(REG.SOUTH), PASS, PASS, PASS, PASS])
    assert requests[0].decision is DecisionType.NEWS_SWAP_REGION
    state = table.state
    assert state.players[0].pawn is REG.SOUTH
    assert state.cubes[REG.PACIFIC_NW][3] == 1
    assert state.cubes[REG.SOUTH][3] == 1


def test_new_constituents_with_empty_map_asks_nobody():  # rule 50
    table, _ = news_table(News.NEW_CONSTITUENTS)
    requests, _ = drive(news_phase(table))
    assert requests == []


# -- commit --------------------------------------------------------------------------------


def test_commit_four_cards_in_turn_order():  # rule 6
    table, sink = make_table(phase=Phase.COMMIT)
    table.state.start_seat = 1
    requests, _ = drive(commit_phase(table), [K(Card.SUPER_PAC)] * 20)
    assert [r.seat for r in requests] == [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [0] * 4
    assert table.state.rnd.queue[1] == [Card.SUPER_PAC] * 4
    assert len(sink.of_type(EventType.CARD_COMMITTED)) == 20


def test_card_copies_limit_choices():  # rule 6
    table, _ = make_table(phase=Phase.COMMIT)
    answers = [K(Card.RALLY)] * 3 + [K(Card.FUNDRAISER)] + [K(Card.SUPER_PAC)] * 16
    requests, _ = drive(commit_phase(table), answers)
    assert K(Card.RALLY) not in requests[3].legal_actions
    assert K(Card.RALLY) in requests[4].legal_actions


def test_frozen_cards_are_unavailable():  # rule 40
    table, _ = make_table(phase=Phase.COMMIT)
    table.state.frozen[0] = {Card.PLANE: 1, Card.RALLY: 2}
    requests, _ = drive(commit_phase(table), [K(Card.RALLY)] + [K(Card.SUPER_PAC)] * 19)
    assert K(Card.PLANE) not in requests[0].legal_actions
    assert K(Card.RALLY) not in requests[1].legal_actions


def test_final_push_allows_a_fifth_card_or_stop():  # rule 43
    table, _ = make_table(phase=Phase.COMMIT, news=News.FINAL_PUSH)
    answers = [K(Card.SUPER_PAC)] * 4 + [K(Card.RALLY)] + ([K(Card.SUPER_PAC)] * 4 + [STOP]) * 4
    requests, _ = drive(commit_phase(table), answers)
    assert STOP not in requests[3].legal_actions
    assert STOP in requests[4].legal_actions
    assert len(table.state.rnd.queue[0]) == 5
    assert len(table.state.rnd.queue[1]) == 4


# -- execute -------------------------------------------------------------------------------


def test_cards_execute_slot_by_slot_in_turn_order():  # rule 7
    table, sink = make_table()
    table.state.start_seat = 3
    for queue in table.state.rnd.queue:
        queue.extend([Card.FUNDRAISER, Card.SUPER_PAC, Card.SUPER_PAC, Card.SUPER_PAC])
    drive(execute_phase(table))
    revealed = [(e.seat, e.card) for e in sink.of_type(EventType.CARD_REVEALED)]
    assert [seat for seat, _ in revealed[:5]] == [3, 4, 0, 1, 2]
    assert {card for _, card in revealed[:5]} == {Card.FUNDRAISER}
    assert len(revealed) == 20


def test_diane_may_skip_one_card_per_round():  # rule 25
    table, sink = make_table(candidates=(Candidate.DIANE_PROUDIAN, *DEFAULT_CANDIDATES[1:]))
    table.state.rnd.queue[0].extend(
        [Card.FUNDRAISER, Card.SUPER_PAC, Card.SUPER_PAC, Card.FUNDRAISER]
    )
    requests, _ = drive(execute_phase(table), [NO, YES])
    assert [r.decision for r in requests] == [DecisionType.DIANE_SKIP] * 2
    rnd = table.state.rnd
    assert (rnd.super_pacs[0], rnd.fundraisers[0], rnd.skipped_slot[0]) == (1, 2, 1)
    assert [e.card for e in sink.of_type(EventType.CARD_SKIPPED)] == [Card.SUPER_PAC]


def test_erin_moves_a_cube_before_her_first_card():  # rule 29
    table, sink = make_table(
        candidates=(Candidate.ERIN_FIELDS, *DEFAULT_CANDIDATES[1:]), cubes={(REG.SOUTH, 0): 1}
    )
    table.state.rnd.queue[0].extend([Card.SUPER_PAC] * 4)
    requests, _ = drive(execute_phase(table), [R(REG.SOUTH), R(REG.NEW_ENGLAND)])
    assert [r.decision for r in requests] == [DecisionType.ERIN_SOURCE, DecisionType.ERIN_DEST]
    assert R(REG.SOUTH) not in requests[1].legal_actions
    assert table.state.cubes[REG.SOUTH][0] == 0
    assert table.state.cubes[REG.NEW_ENGLAND][0] == 1
    types = [e.event_type for e in sink.events]
    assert types.index(EventType.CUBE_MOVED) < types.index(EventType.CARD_REVEALED)


def test_erin_without_cubes_on_the_map_is_not_asked():  # rule 29
    table, _ = make_table(candidates=(Candidate.ERIN_FIELDS, *DEFAULT_CANDIDATES[1:]))
    table.state.rnd.queue[0].extend([Card.SUPER_PAC] * 4)
    requests, _ = drive(execute_phase(table))
    assert requests == []
```

`tests/test_game.py`:

```python
import pytest

from primary.engine.actions import (
    NUM_ACTIONS,
    DecisionRequest,
    EngineError,
    GameOver,
    IllegalActionError,
)
from primary.engine.events import EventType, ListSink
from primary.engine.game import Game
from tests.helpers import DEFAULT_CANDIDATES, play_random


def test_full_random_game_reaches_game_over():  # rules 3, 33
    sink = ListSink()
    game = Game(5, sink=sink, check_invariants=True)
    result = play_random(game, seed=7)
    assert isinstance(result, GameOver)
    state = game.state
    assert state.round == 11
    assert len(state.news_history) == 11
    assert all(state.voted)
    assert len(sink.rounds) == 55
    assert result.winners
    assert game.result == result
    assert game.pending is None


def test_step_after_game_over_raises():
    game = Game(5)
    play_random(game, seed=1)
    with pytest.raises(EngineError):
        game.step(0)


def test_illegal_action_is_rejected_and_the_game_continues():
    game = Game(5)
    request = game.reset(3, DEFAULT_CANDIDATES)
    assert isinstance(request, DecisionRequest)
    illegal = next(a for a in range(NUM_ACTIONS) if a not in request.legal_actions)
    with pytest.raises(IllegalActionError, match="legal"):
        game.step(illegal)
    assert game.pending == request
    game.step(request.legal_actions[0])


def test_same_seed_and_actions_replay_identically():
    a, b = ListSink(), ListSink()
    result_a = play_random(Game(5, sink=a), seed=11)
    result_b = play_random(Game(5, sink=b), seed=11)
    assert result_a == result_b
    assert a.events == b.events
    assert a.rounds == b.rounds


def test_history_holds_only_requested_decisions():
    sink = ListSink()
    game = Game(5, sink=sink)
    play_random(game, seed=5)
    requested = [e.action for e in sink.of_type(EventType.DECISION) if not e.forced]
    assert game.history == requested


def test_invalid_player_counts_and_candidates_are_rejected():  # rule 1
    with pytest.raises(ValueError):
        Game(5).reset(0, DEFAULT_CANDIDATES[:4])
    with pytest.raises(ValueError):
        Game(7)


def test_state_before_reset_raises():
    with pytest.raises(EngineError):
        _ = Game(5).state
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_phases.py tests/test_game.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primary.engine.phases'`.

- [ ] **Step 4: Add `move_options` to `news_effects.py`**

Add these imports at the top of `src/primary/engine/news_effects.py` (after the existing imports):

```python
from primary.data.board import NUM_ROUNDS, Region
from primary.engine.state import GameState
```

Append at the end of the file:

```python
def move_options(kind: NewsMove, state: GameState, seat: int) -> tuple[tuple[Region, ...], bool]:
    """Destinations offered by a News move and whether the player may decline (rules 38, 39, 44)."""
    here = state.players[seat].pawn
    others = [r for r in Region if r != here]
    if kind is NewsMove.DEBATE:
        rounds = (state.round, state.round + 1)
        return tuple(r for r in others if state.vote_round[r] in rounds), True
    if kind is NewsMove.LONG_TERM:
        return tuple(r for r in others if state.vote_round[r] == NUM_ROUNDS), True
    if kind is NewsMove.DETOUR:
        return tuple(others), False
    return (), True
```

- [ ] **Step 5: Implement `phases.py`**

```python
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
```

- [ ] **Step 6: Implement `game.py`**

```python
"""Public engine API: reset/step a whole game (spec §5.2)."""

from __future__ import annotations

from collections.abc import Generator, Sequence

import numpy as np

from primary.data.board import NUM_ROUNDS
from primary.data.candidates import MAX_PLAYERS, MIN_PLAYERS, Candidate
from primary.engine.actions import (
    DecisionRequest,
    EngineError,
    GameOver,
    IllegalActionError,
    describe_action,
)
from primary.engine.events import EventSink
from primary.engine.invariants import check_invariants
from primary.engine.phases import commit_phase, execute_phase, news_phase, setup_phase
from primary.engine.round_end import finish_game, resolve_round_end
from primary.engine.state import GameState, new_game_state
from primary.engine.table import Table

Outcome = DecisionRequest | GameOver
GameFlow = Generator[DecisionRequest, int, GameOver]


class Game:
    """One game of The Primary.

    ``reset`` returns the first ``DecisionRequest``; answer each with ``step(action)`` until a
    ``GameOver`` is returned. Decisions with a single legal action are taken automatically.
    """

    def __init__(
        self, num_players: int = 5, sink: EventSink | None = None, check_invariants: bool = False
    ) -> None:
        if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
            raise ValueError(f"need {MIN_PLAYERS}-{MAX_PLAYERS} players, got {num_players}")
        self.num_players = num_players
        self.sink = sink
        self.check_invariants = check_invariants
        self.seed: int | None = None
        self.pending: DecisionRequest | None = None
        self.result: GameOver | None = None
        self.history: list[int] = []
        self._table: Table | None = None
        self._flow: GameFlow | None = None

    @property
    def table(self) -> Table:
        if self._table is None:
            raise EngineError("call reset() first")
        return self._table

    @property
    def state(self) -> GameState:
        return self.table.state

    def reset(self, seed: int, candidates: Sequence[Candidate]) -> Outcome:
        if len(candidates) != self.num_players:
            raise ValueError(f"expected {self.num_players} candidates, got {len(candidates)}")
        self.seed = seed
        self._table = Table(new_game_state(candidates, np.random.default_rng(seed)), self.sink)
        self._flow = self._play(self._table)
        self.pending = None
        self.result = None
        self.history = []
        return self._advance(None)

    def step(self, action: int) -> Outcome:
        request = self.pending
        if request is None:
            raise EngineError("the game is over (or not started); call reset()")
        if action not in request.legal_actions:
            legal = ", ".join(describe_action(a) for a in request.legal_actions)
            raise IllegalActionError(
                f"{describe_action(action)} is not legal for {request.decision.name}; "
                f"legal actions: {legal}"
            )
        self.history.append(action)
        return self._advance(action)

    def _advance(self, action: int | None) -> Outcome:
        assert self._flow is not None
        try:
            request = next(self._flow) if action is None else self._flow.send(action)
        except StopIteration as stop:
            result: GameOver = stop.value
            self.pending = None
            self.result = result
            if self.check_invariants:
                check_invariants(self.state)
            return result
        if self.check_invariants:
            check_invariants(self.state)
        self.pending = request
        return request

    @staticmethod
    def _play(table: Table) -> GameFlow:
        yield from setup_phase(table)
        for rnd in range(1, NUM_ROUNDS + 1):
            table.begin_round(rnd)
            yield from news_phase(table)
            yield from commit_phase(table)
            yield from execute_phase(table)
            resolve_round_end(table)
        return finish_game(table.state)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_phases.py tests/test_game.py -v`
Expected: 15 + 7 passed.

- [ ] **Step 8: Run the whole suite**

Run: `uv run pytest`
Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/primary/engine tests/helpers.py tests/test_phases.py tests/test_game.py
git commit -m "Add round phases and the Game reset/step driver" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Property-based invariant and determinism tests

**Files:**
- Test: `tests/test_properties.py`

**Interfaces:**
- Consumes: `Game`, `ListSink`, `DecisionRequest`, `GameOver` (Tasks 3, 7). Produces nothing new; this task exists to catch rule-interaction bugs the unit tests miss, across 3–6 players and all 12 candidates.

- [ ] **Step 1: Write the tests**

`tests/test_properties.py`:

```python
"""Property-based checks over many random games (spec §5.5, §7)."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from primary.data.board import Region
from primary.data.candidates import Candidate
from primary.engine.actions import DecisionRequest, GameOver
from primary.engine.events import ListSink
from primary.engine.game import Game


def random_candidates(rng: np.random.Generator, n: int) -> list[Candidate]:
    return [Candidate(int(c)) for c in rng.choice(len(Candidate), n, replace=False)]


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 2**31 - 2), num_players=st.integers(3, 6))
def test_random_games_respect_invariants(seed, num_players):
    rng = np.random.default_rng(seed)
    sink = ListSink()
    game = Game(num_players, sink=sink, check_invariants=True)  # cube conservation every step
    outcome = game.reset(seed, random_candidates(rng, num_players))
    previous = [0] * num_players
    while isinstance(outcome, DecisionRequest):
        assert outcome.legal_actions
        outcome = game.step(int(rng.choice(outcome.legal_actions)))
        scores = [p.score for p in game.state.players]
        assert all(now >= before for now, before in zip(scores, previous, strict=True))
        previous = scores
    assert isinstance(outcome, GameOver)
    state = game.state
    assert state.round == 11
    assert sorted(r for regions in state.schedule.values() for r in regions) == list(Region)
    assert all(state.voted)
    assert all(sum(row) == 0 for row in state.cubes)  # every region voted, so all cubes went home
    assert len(state.news_history) == 11
    assert len(sink.rounds) == 11 * num_players
    assert sum(r.votes_gained for r in sink.rounds) == sum(outcome.scores)
    assert outcome.winners


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 2))
def test_games_are_deterministic(seed):
    def run():
        rng = np.random.default_rng(seed)
        sink = ListSink()
        game = Game(5, sink=sink)
        outcome = game.reset(seed, random_candidates(rng, 5))
        while isinstance(outcome, DecisionRequest):
            outcome = game.step(int(rng.choice(outcome.legal_actions)))
        return outcome, sink.events, sink.rounds

    assert run() == run()
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_properties.py -v`
Expected: 2 passed. If a Hypothesis example fails, it prints the falsifying `seed`/`num_players`; reproduce it with `play_random`-style stepping, fix the engine (not the test), and add a focused unit test for the rule involved in the relevant engine test module.

- [ ] **Step 3: Commit**

```bash
git add tests/test_properties.py
git commit -m "Add property-based invariant and determinism tests" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Random and heuristic bots

**Files:**
- Create: `src/primary/agents/__init__.py`, `src/primary/agents/base.py`, `src/primary/agents/random_agent.py`, `src/primary/agents/heuristic.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `Game` (Task 7), `DecisionRequest`, action helpers (Task 3), `Modifiers` (Task 3), `available_cards`-style logic via `CARD_COUNTS` and `state.frozen`, `reachable`/`distance` (Task 2).
- Produces: `Agent` Protocol (`name: str`, `act(game: Game, request: DecisionRequest) -> int`); `RandomAgent(seed: int | None = None)`; `HeuristicAgent()`; `plan_cards(state, seat, mods) -> list[Card]`; `best_region(state, seat, regions) -> Region`; `AGENT_NAMES = ("random", "heuristic")`; `make_agent(name: str, seed: int | None = None) -> Agent` (raises `ValueError` for unknown names).

- [ ] **Step 1: Write the failing tests**

`tests/test_agents.py`:

```python
import pytest

from primary.agents import AGENT_NAMES, make_agent
from primary.agents.heuristic import HeuristicAgent, plan_cards
from primary.agents.random_agent import RandomAgent
from primary.data.board import Region
from primary.data.candidates import Candidate
from primary.data.cards import Card
from primary.engine.actions import DecisionRequest, GameOver
from primary.engine.effects import build_modifiers
from primary.engine.game import Game
from tests.helpers import make_state

KENNETH = build_modifiers(Candidate.KENNETH_WALKER, None)
F, RALLY, BUS, PLANE = Card.FUNDRAISER, Card.RALLY, Card.BUS, Card.PLANE


def play(agents, candidates, seed):
    game = Game(len(candidates), check_invariants=True)
    outcome = game.reset(seed, candidates)
    while isinstance(outcome, DecisionRequest):
        action = agents[outcome.seat].act(game, outcome)
        assert action in outcome.legal_actions
        outcome = game.step(action)
    return outcome


def test_random_agent_completes_a_game():
    agents = [RandomAgent(seed=s) for s in range(5)]
    assert isinstance(play(agents, list(Candidate)[:5], seed=0), GameOver)


def test_heuristic_completes_games_with_every_candidate():
    for start in range(len(Candidate)):
        candidates = [Candidate((start + k) % len(Candidate)) for k in range(5)]
        agents = [HeuristicAgent() for _ in range(5)]
        assert isinstance(play(agents, candidates, seed=start), GameOver)


def test_mixed_table_completes_games():
    agents = [HeuristicAgent(), RandomAgent(1), HeuristicAgent(), RandomAgent(2), RandomAgent(3)]
    for seed in range(5):
        assert isinstance(play(agents, list(Candidate)[5:10], seed=seed), GameOver)


def test_make_agent():
    assert set(AGENT_NAMES) == {"random", "heuristic"}
    assert make_agent("heuristic").name == "heuristic"
    assert make_agent("random", seed=1).name == "random"
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("nope")


def _state_near_new_england(pawn, hand):
    return make_state(
        hands=[hand, 3, 3, 3, 1],
        pawns=[pawn, Region.WEST, Region.SOUTH_WEST, Region.GREATER_TEXAS, Region.NORTH],
        schedule={2: (Region.NEW_ENGLAND,)},
    )


def test_plan_fundraises_before_rallying_with_an_empty_hand():
    state = _state_near_new_england(Region.NEW_ENGLAND, hand=0)
    assert plan_cards(state, 0, KENNETH) == [F, F, RALLY, F]


def test_plan_takes_the_bus_to_an_adjacent_target():
    state = _state_near_new_england(Region.NORTH_EAST, hand=3)
    assert plan_cards(state, 0, KENNETH) == [BUS, RALLY, F, RALLY]


def test_plan_flies_when_the_bus_is_frozen():
    state = _state_near_new_england(Region.NORTH_EAST, hand=3)
    state.frozen[0] = {Card.BUS: 1}
    assert plan_cards(state, 0, KENNETH) == [PLANE, RALLY, F, F]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agents.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primary.agents'`.

- [ ] **Step 3: Implement the agents**

`src/primary/agents/base.py`:

```python
"""The interface every bot and (later) every trained policy implements."""

from __future__ import annotations

from typing import Protocol

from primary.engine.actions import DecisionRequest
from primary.engine.game import Game


class Agent(Protocol):
    name: str

    def act(self, game: Game, request: DecisionRequest) -> int:
        """Return one of ``request.legal_actions``.

        Agents may read ``game.state`` but must only use public information plus their own
        committed cards (``state.rnd.queue[request.seat]``); opponents' queues are hidden.
        """
        ...
```

`src/primary/agents/random_agent.py`:

```python
"""Uniformly random legal play: the weakest baseline."""

from __future__ import annotations

import numpy as np

from primary.engine.actions import DecisionRequest
from primary.engine.game import Game


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, game: Game, request: DecisionRequest) -> int:
        return int(self.rng.choice(request.legal_actions))
```

`src/primary/agents/heuristic.py`:

```python
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
```

`src/primary/agents/__init__.py`:

```python
"""Baseline bots. Trained RL policies (sub-project 2) implement the same ``Agent`` protocol."""

from __future__ import annotations

from primary.agents.base import Agent
from primary.agents.heuristic import HeuristicAgent
from primary.agents.random_agent import RandomAgent

AGENT_NAMES = ("random", "heuristic")


def make_agent(name: str, seed: int | None = None) -> Agent:
    if name == "random":
        return RandomAgent(seed)
    if name == "heuristic":
        return HeuristicAgent()
    raise ValueError(f"unknown agent {name!r}; choose from {AGENT_NAMES}")


__all__ = ["AGENT_NAMES", "Agent", "HeuristicAgent", "RandomAgent", "make_agent"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_agents.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/primary/agents tests/test_agents.py
git commit -m "Add random and heuristic baseline agents" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Parquet logging and replay

**Files:**
- Create: `src/primary/logs/__init__.py`, `src/primary/logs/schema.py`, `src/primary/logs/writer.py`, `src/primary/logs/replay.py`
- Test: `tests/test_logs.py`

**Interfaces:**
- Consumes: `Event`, `RoundSummary`, `EventType` (Task 3), `Game` (Task 7), `Placement` (Task 2), `play_random`/`DEFAULT_CANDIDATES` (test helpers).
- Produces:
  - `schema.py`: `EVENTS`, `ROUNDS`, `GAMES` (pyarrow schemas), `SCHEMAS: dict[str, pa.Schema]`, `TABLE_NAMES = ("games", "events", "rounds")`.
  - `writer.py`: `event_row(game_id, event) -> dict`, `round_row(game_id, summary) -> dict`, `round_result(summary) -> str` (`"no_vote" | "won" | "lost" | "not_participated"`), `game_row(game_id, game, policies) -> dict`, `GameRecorder(game_id)` (an `EventSink` with `events`/`rounds` row lists), `RunLogger(out_dir, flush_every=100)` with `recorder(game_id)`, `finish(recorder, game, policies)`, `flush()`, `close()`, and context-manager support. Files: `<out_dir>/games.parquet`, `events.parquet`, `rounds.parquet`.
  - `replay.py`: `ReplayError(RuntimeError)`, `load_decisions(run_dir, game_id) -> tuple[dict, list[int]]` (raises `KeyError` if the game was not logged), `replay_game(run_dir, game_id, sink=None) -> Game`, `verify_replay(run_dir, game_id) -> bool`.

- [ ] **Step 1: Write the failing tests**

`tests/test_logs.py`:

```python
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
    for column in ("start_region", "end_region", "path", "hand_start", "hand_end", "cards",
                   "vote_regions", "vote_placements", "round_result"):
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_logs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primary.logs'`.

- [ ] **Step 3: Implement the schemas**

`src/primary/logs/__init__.py`:

```python
"""Parquet logging of simulated games and exact replay from the logs."""
```

`src/primary/logs/schema.py`:

```python
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
```

- [ ] **Step 4: Implement the writer**

`src/primary/logs/writer.py`:

```python
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
```

- [ ] **Step 5: Implement replay**

`src/primary/logs/replay.py`:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_logs.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add src/primary/logs tests/test_logs.py
git commit -m "Add Parquet logging of games, events and rounds, with exact replay" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Simulation runner, CLI, run configs and the heuristic-vs-random sanity check

**Files:**
- Create: `src/primary/simulate.py`, `src/primary/cli.py`
- Create: `configs/sim_random.yaml`, `configs/sim_heuristic.yaml`
- Test: `tests/test_simulate.py`

**Interfaces:**
- Consumes: `make_agent` (Task 9), `Game` (Task 7), `RunLogger`, `TABLE_NAMES` (Task 10), `replay_game`, `verify_replay` (Task 10).
- Produces: `SimConfig` (fields `num_games=100`, `num_players=5`, `seed=0`, `agents=["random"]`, `rotate_seats=True`, `log_every=1`, `out_dir=None`, `overwrite=False`; `SimConfig.from_yaml(path)` rejects unknown keys with `ValueError`); `GameRecord(game_id, seed, candidates, policies, result)`; `SimSummary(records, seconds)` with `games_per_second`, `draw_rate`, `win_rates_by_policy()`, `win_rates_by_candidate()`; `seat_policies(cfg, game_id) -> tuple[str, ...]`; `run_simulation(cfg) -> SimSummary` (raises `FileExistsError` when `out_dir` holds a run and `overwrite` is false; writes `config.yaml` into the run directory); `cli.main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write the run configs**

`configs/sim_random.yaml`:

```yaml
# Five random bots. A smoke test of the engine and logging.
seed: 0
num_games: 1000
num_players: 5
agents: [random]
rotate_seats: true
log_every: 1
out_dir: data/runs/sanity_random
```

`configs/sim_heuristic.yaml`:

```yaml
# One heuristic bot against four random bots; the heuristic rotates through every seat.
seed: 0
num_games: 1000
num_players: 5
agents: [heuristic, random, random, random, random]
rotate_seats: true
log_every: 1
out_dir: data/runs/sanity_heuristic
```

- [ ] **Step 2: Write the failing tests**

`tests/test_simulate.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_simulate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primary.cli'`.

- [ ] **Step 4: Implement `simulate.py`**

```python
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
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
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
```

- [ ] **Step 5: Implement `cli.py`**

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_simulate.py -v`
Expected: 13 passed. If `test_heuristic_beats_random` fails, the heuristic is too weak: improve `plan_cards`/the choosers in `heuristic.py`. Never lower the test's threshold.

- [ ] **Step 7: Try the CLI by hand**

Run: `uv run primary simulate --config configs/sim_heuristic.yaml --games 200 --out data/runs/smoke --overwrite`
Expected: a summary with `games/sec`, the heuristic's win rate well above 0.2, and `logs written to .../data/runs/smoke`.

Run: `uv run primary replay --run data/runs/smoke --game-id 7`
Expected: `replay matches the log`.

- [ ] **Step 8: Commit**

```bash
git add src/primary/simulate.py src/primary/cli.py configs tests/test_simulate.py
git commit -m "Add simulation runner, primary CLI and run configs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: PettingZoo AEC environment

**Files:**
- Create: `src/primary/env/__init__.py`, `src/primary/env/encoding.py`, `src/primary/env/rewards.py`, `src/primary/env/aec_env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: `Game`, `GameState`, `DecisionRequest`, `GameOver`, `DecisionType`, `NUM_ACTIONS`, `PLAYER_BASE` (Tasks 3, 7), `TOTAL_VOTES`, `TOTAL_CUBES`.
- Produces:
  - `encoding.py`: `MAX_QUEUE = 5`; `obs_size(n: int) -> int`; `encode_observation(state, me, request) -> np.ndarray` (float32, all values in [0, 1], observer is relative seat 0); `encode_mask(request, me, n) -> np.ndarray` (int8, length `NUM_ACTIONS`); `to_absolute_action(action, me, n) -> int`; `relative_seat`, `absolute_seat`.
  - `rewards.py`: `terminal_rewards(result: GameOver, n) -> list[float]`, `shaped_rewards(before, after) -> list[float]`.
  - `aec_env.py`: `PrimaryEnv(num_players=5, candidates=None, shaped_reward=False, render_mode=None)` (a `pettingzoo.AECEnv`, metadata name `primary_v0`); `env(**kwargs)` returns it wrapped in `OrderEnforcingWrapper`. Agents are `player_0..player_{n-1}`; observations are `{"observation", "action_mask"}`; `reset(seed=None, options=None)` accepts `options={"candidates": [...]}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_env.py`:

```python
import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from primary.data.candidates import TOTAL_CUBES, Candidate
from primary.data.cards import Card
from primary.engine.actions import (
    PLAYER_BASE,
    DecisionRequest,
    DecisionType,
    GameOver,
    IllegalActionError,
    player_action,
)
from primary.engine.voting import TieBreak
from primary.env.aec_env import env
from primary.env.encoding import encode_mask, encode_observation, obs_size, to_absolute_action
from primary.env.rewards import shaped_rewards, terminal_rewards
from tests.helpers import make_state


def test_pettingzoo_api_compliance():
    api_test(env(), num_cycles=1000, verbose_progress=False)


def test_seeded_resets_are_reproducible():
    seed_test(env, num_cycles=200)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_observation_size(n):
    state = make_state(candidates=tuple(Candidate)[:n])
    assert encode_observation(state, 0, None).shape == (obs_size(n),)


def test_opponents_face_down_cards_are_hidden():
    a, b = make_state(), make_state()
    a.rnd.queue[1].extend([Card.RALLY] * 3 + [Card.BUS])
    b.rnd.queue[1].extend([Card.SUPER_PAC] * 4)
    assert np.array_equal(encode_observation(a, 0, None), encode_observation(b, 0, None))
    a.rnd.revealed[1].append(Card.RALLY)
    b.rnd.revealed[1].append(Card.SUPER_PAC)
    assert not np.array_equal(encode_observation(a, 0, None), encode_observation(b, 0, None))


def test_own_queue_is_visible():
    a, b = make_state(), make_state()
    a.rnd.queue[0].append(Card.RALLY)
    b.rnd.queue[0].append(Card.BUS)
    assert not np.array_equal(encode_observation(a, 0, None), encode_observation(b, 0, None))


def test_observation_is_egocentric():
    state = make_state(hands=[2, 9, 3, 3, 1])
    hands_offset = 2 * 14 * 5  # after the cube and pawn blocks
    assert encode_observation(state, 1, None)[hands_offset] == pytest.approx(9 / TOTAL_CUBES)
    assert encode_observation(state, 0, None)[hands_offset] == pytest.approx(2 / TOTAL_CUBES)


def test_player_actions_are_rotated_to_the_observer():
    request = DecisionRequest(
        seat=1,
        decision=DecisionType.NEGATIVE_AD_OWNER,
        legal_actions=(player_action(0), player_action(2)),
    )
    mask = encode_mask(request, me=1, n=5)
    assert set(np.flatnonzero(mask)) == {PLAYER_BASE + 1, PLAYER_BASE + 4}
    assert to_absolute_action(PLAYER_BASE + 1, me=1, n=5) == player_action(2)
    assert to_absolute_action(PLAYER_BASE + 4, me=1, n=5) == player_action(0)


def test_terminal_and_shaped_rewards():
    win = GameOver((10, 5, 0), (0, 0, 0), (0, 0, 0), (0,), TieBreak.SCORE)
    assert terminal_rewards(win, 3) == [1.0, 0.0, 0.0]
    draw = GameOver((10, 5, 10), (0, 0, 0), (0, 0, 0), (0, 2), TieBreak.DRAW)
    assert terminal_rewards(draw, 3) == [0.5, 0.0, 0.5]
    assert shaped_rewards([0, 0], [24, 0]) == [pytest.approx(0.1), 0.0]


def run_episode(e, seed):
    rng = np.random.default_rng(seed)
    e.reset(seed=seed)
    totals = dict.fromkeys(e.possible_agents, 0.0)
    for agent in e.agent_iter():
        obs, reward, termination, truncation, _ = e.last()
        totals[agent] += reward
        assert 0.0 <= obs["observation"].min() and obs["observation"].max() <= 1.0
        done = termination or truncation
        e.step(None if done else int(rng.choice(np.flatnonzero(obs["action_mask"]))))
    return totals


def test_unshaped_episode_hands_out_exactly_one_point():
    totals = run_episode(env(), seed=3)
    assert sum(totals.values()) == pytest.approx(1.0)
    assert min(totals.values()) >= 0.0


def test_shaped_episode_adds_vote_share():
    e = env(shaped_reward=True)
    totals = run_episode(e, seed=3)
    scores = e.unwrapped.game.result.scores
    assert sum(totals.values()) == pytest.approx(1.0 + sum(scores) / 240)


def test_illegal_action_raises():
    e = env()
    e.reset(seed=0)
    mask = e.observe(e.agent_selection)["action_mask"]
    illegal = int(np.flatnonzero(mask == 0)[0])
    with pytest.raises(IllegalActionError):
        e.step(illegal)


def test_fixed_candidates_option():
    e = env()
    chosen = list(Candidate)[:5]
    e.reset(seed=1, options={"candidates": chosen})
    assert [p.candidate for p in e.unwrapped.game.state.players] == chosen
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primary.env'`.

- [ ] **Step 3: Implement the encoding and rewards**

`src/primary/env/__init__.py`:

```python
"""PettingZoo environment over the rules engine."""

from primary.env.aec_env import PrimaryEnv, env

__all__ = ["PrimaryEnv", "env"]
```

`src/primary/env/encoding.py`:

```python
"""Observation and action-mask encoding (spec §4.3–4.4). The observer is always seat 0."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from primary.data.board import NUM_REGIONS, NUM_ROUNDS, TOTAL_VOTES
from primary.data.candidates import TOTAL_CUBES, Candidate
from primary.data.cards import CARD_COUNTS, Card
from primary.data.news import News
from primary.engine.actions import NUM_ACTIONS, PLAYER_BASE, DecisionRequest, DecisionType
from primary.engine.state import GameState

MAX_QUEUE = 5
N_CARDS = len(Card)
FloatArray = npt.NDArray[np.float32]


def relative_seat(seat: int, me: int, n: int) -> int:
    return (seat - me) % n


def absolute_seat(rel: int, me: int, n: int) -> int:
    return (rel + me) % n


def obs_size(n: int) -> int:
    return (
        2 * NUM_REGIONS * n  # cubes, pawns
        + 3 * n  # hand, bank, score
        + (NUM_ROUNDS + 1)  # round one-hot (0 = setup)
        + NUM_REGIONS * NUM_ROUNDS  # voting schedule
        + NUM_REGIONS  # already voted
        + 2 * len(News)  # current News, used News
        + len(Candidate) * n  # candidate per seat
        + n  # starting-player token
        + MAX_QUEUE * N_CARDS + MAX_QUEUE  # own queue, own executed flags
        + (n - 1) * MAX_QUEUE * N_CARDS  # opponents' revealed cards
        + 2 * n  # committed / revealed counts
        + N_CARDS  # own frozen cards
        + n  # skip used
        + len(DecisionType) + N_CARDS + NUM_REGIONS  # decision type and context
    )


def _one_hot(index: int | None, size: int) -> FloatArray:
    vec = np.zeros(size, dtype=np.float32)
    if index is not None:
        vec[index] = 1.0
    return vec


def _cards_one_hot(cards: Sequence[Card]) -> FloatArray:
    grid = np.zeros((MAX_QUEUE, N_CARDS), dtype=np.float32)
    for slot, card in enumerate(cards[:MAX_QUEUE]):
        grid[slot, card] = 1.0
    return grid.ravel()


def encode_observation(state: GameState, me: int, request: DecisionRequest | None) -> FloatArray:
    """What seat ``me`` can see. ``request`` is the decision ``me`` must make now, if any."""
    n = state.num_players
    order = [absolute_seat(k, me, n) for k in range(n)]
    players = [state.players[s] for s in order]
    rnd = state.rnd

    cubes = np.asarray(state.cubes, dtype=np.float32)[:, order] / TOTAL_CUBES
    pawns = np.zeros((NUM_REGIONS, n), dtype=np.float32)
    for k, player in enumerate(players):
        if player.pawn is not None:
            pawns[player.pawn, k] = 1.0
    schedule = np.zeros((NUM_REGIONS, NUM_ROUNDS), dtype=np.float32)
    for region, rnd_number in state.vote_round.items():
        schedule[region, rnd_number - 1] = 1.0
    used_news = np.zeros(len(News), dtype=np.float32)
    for news in state.news_history:
        used_news[news] = 1.0
    candidates = np.zeros((n, len(Candidate)), dtype=np.float32)
    for k, player in enumerate(players):
        candidates[k, player.candidate] = 1.0
    executed = np.zeros(MAX_QUEUE, dtype=np.float32)
    executed[: len(rnd.revealed[me])] = 1.0
    counts = [
        [len(rnd.queue[s]) / MAX_QUEUE, len(rnd.revealed[s]) / MAX_QUEUE] for s in order
    ]
    frozen = [state.frozen[me].get(card, 0) / CARD_COUNTS[card] for card in Card]
    skipped = [rnd.skipped_slot[s] is not None for s in order]

    parts = [
        cubes.ravel(),
        pawns.ravel(),
        np.array([p.hand / TOTAL_CUBES for p in players], dtype=np.float32),
        np.array([p.bank / TOTAL_CUBES for p in players], dtype=np.float32),
        np.array([p.score / TOTAL_VOTES for p in players], dtype=np.float32),
        _one_hot(state.round, NUM_ROUNDS + 1),
        schedule.ravel(),
        np.array(state.voted, dtype=np.float32),
        _one_hot(state.current_news, len(News)),
        used_news,
        candidates.ravel(),
        _one_hot(relative_seat(state.start_seat, me, n), n),
        _cards_one_hot(rnd.queue[me]),
        executed,
        *[_cards_one_hot(rnd.revealed[s]) for s in order[1:]],
        np.array(counts, dtype=np.float32).ravel(),
        np.array(frozen, dtype=np.float32),
        np.array(skipped, dtype=np.float32),
        _one_hot(None if request is None else request.decision, len(DecisionType)),
        _one_hot(None if request is None else request.card, N_CARDS),
        _one_hot(None if request is None else request.region, NUM_REGIONS),
    ]
    return np.concatenate(parts).astype(np.float32)


def encode_mask(request: DecisionRequest, me: int, n: int) -> npt.NDArray[np.int8]:
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    for action in request.legal_actions:
        if PLAYER_BASE <= action < PLAYER_BASE + n:
            action = PLAYER_BASE + relative_seat(action - PLAYER_BASE, me, n)
        mask[action] = 1
    return mask


def to_absolute_action(action: int, me: int, n: int) -> int:
    """Map an observer-relative action back to the engine's absolute seat numbering."""
    if PLAYER_BASE <= action < PLAYER_BASE + n:
        return PLAYER_BASE + absolute_seat(action - PLAYER_BASE, me, n)
    return action
```

`src/primary/env/rewards.py`:

```python
"""Reward functions (spec §4.5)."""

from __future__ import annotations

from collections.abc import Sequence

from primary.data.board import TOTAL_VOTES
from primary.engine.actions import GameOver


def terminal_rewards(result: GameOver, n: int) -> list[float]:
    """+1 for an outright win; a k-way draw gives 1/k to each tied player."""
    share = 1.0 / len(result.winners)
    return [share if seat in result.winners else 0.0 for seat in range(n)]


def shaped_rewards(before: Sequence[int], after: Sequence[int]) -> list[float]:
    """Optional shaping: votes gained this step as a share of all votes in the game."""
    return [(a - b) / TOTAL_VOTES for a, b in zip(after, before, strict=True)]
```

- [ ] **Step 4: Implement the environment**

`src/primary/env/aec_env.py`:

```python
"""PettingZoo AEC environment for The Primary (spec §4)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from primary.data.candidates import Candidate
from primary.engine.actions import NUM_ACTIONS, GameOver
from primary.engine.game import Game, Outcome
from primary.env.encoding import encode_mask, encode_observation, obs_size, to_absolute_action
from primary.env.rewards import shaped_rewards, terminal_rewards


class PrimaryEnv(AECEnv):
    metadata = {"name": "primary_v0", "render_modes": ["ansi"], "is_parallelizable": False}

    def __init__(
        self,
        num_players: int = 5,
        candidates: Sequence[Candidate] | None = None,
        shaped_reward: bool = False,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.num_players = num_players
        self.fixed_candidates = tuple(candidates) if candidates is not None else None
        self.shaped_reward = shaped_reward
        self.render_mode = render_mode
        self.game = Game(num_players)
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self._seat = {agent: i for i, agent in enumerate(self.possible_agents)}
        size = obs_size(num_players)
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(0.0, 1.0, (size,), dtype=np.float32),
                    "action_mask": spaces.Box(0, 1, (NUM_ACTIONS,), dtype=np.int8),
                }
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {agent: spaces.Discrete(NUM_ACTIONS) for agent in self.possible_agents}

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        candidates = (options or {}).get("candidates") or self.fixed_candidates
        if candidates is None:
            rng = np.random.default_rng([seed, 1])  # independent of the engine's stream
            drawn = rng.choice(len(Candidate), self.num_players, replace=False)
            candidates = tuple(Candidate(int(c)) for c in drawn)
        self.agents = self.possible_agents[:]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[0]
        self._handle(self.game.reset(seed, list(candidates)))

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        me = self._seat[agent]
        pending = self.game.pending
        request = pending if pending is not None and pending.seat == me else None
        mask = (
            encode_mask(request, me, self.num_players)
            if request is not None
            else np.zeros(NUM_ACTIONS, dtype=np.int8)
        )
        return {
            "observation": encode_observation(self.game.state, me, request),
            "action_mask": mask,
        }

    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        assert action is not None
        me = self._seat[agent]
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        before = [p.score for p in self.game.state.players]
        outcome = self.game.step(to_absolute_action(int(action), me, self.num_players))
        if self.shaped_reward:
            after = [p.score for p in self.game.state.players]
            for seat, reward in enumerate(shaped_rewards(before, after)):
                self.rewards[self.possible_agents[seat]] += reward
        self._handle(outcome)
        self._accumulate_rewards()

    def _handle(self, outcome: Outcome) -> None:
        if isinstance(outcome, GameOver):
            for seat, reward in enumerate(terminal_rewards(outcome, self.num_players)):
                self.rewards[self.possible_agents[seat]] += reward
            for agent in self.agents:
                self.terminations[agent] = True
                self.infos[agent] = {
                    "winners": list(outcome.winners),
                    "scores": list(outcome.scores),
                    "tiebreak": outcome.tiebreak.value,
                }
        else:
            self.agent_selection = self.possible_agents[outcome.seat]

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        state = self.game.state
        news = state.current_news.name if state.current_news is not None else "-"
        lines = [f"Round {state.round}  News: {news}"]
        for seat, p in enumerate(state.players):
            pawn = p.pawn.name if p.pawn is not None else "-"
            lines.append(
                f"  P{seat} {p.candidate.name:<16} votes={p.score:>3} "
                f"hand={p.hand:>2} bank={p.bank:>2} at {pawn}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        pass


def env(**kwargs: Any) -> AECEnv:
    """The environment wrapped with PettingZoo's call-order checks (the recommended entry point)."""
    return wrappers.OrderEnforcingWrapper(PrimaryEnv(**kwargs))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_env.py -v`
Expected: 15 passed. If `api_test` fails, read its assertion message: it names the exact AEC contract that was broken (most often reward bookkeeping or `agent_selection` after termination). Fix `aec_env.py` to satisfy it; do not skip the test.

- [ ] **Step 6: Commit**

```bash
git add src/primary/env tests/test_env.py
git commit -m "Add PettingZoo AEC environment with egocentric observations and action masks" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: CI, README quick start, sanity notebook and the final quality pass

**Files:**
- Create: `.github/workflows/ci.yml`, `notebooks/01_engine_sanity.ipynb`
- Modify: `README.md`
- Modify: any source file that `ruff`/`mypy` flag

**Interfaces:**
- Consumes: everything above. Produces the CI gate and the documentation a reviewer reads first.

- [ ] **Step 1: Lint, format and type-check the whole project**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run ruff check .`
Run: `uv run mypy`
Expected: `Success: no issues found`. Fix every reported error in the source (add precise types; use `# type: ignore[<code>]` only on third-party boundaries such as `AECEnv`, each with the error code). Re-run `uv run pytest` after fixes; everything must still pass.

- [ ] **Step 2: Add the CI workflow**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python: ["3.11", "3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: ${{ matrix.python }}
      - run: uv sync --locked
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy
      - run: uv run pytest
```

- [ ] **Step 3: Update the README**

In `README.md`, replace the line `- Negative Ad x 1: Remove one opponent's currency cube from anywhere on the map` with:

```markdown
- Negative Ad x 1: Remove any one currency cube (an opponent's or your own) from anywhere on the map; it returns to its owner's bank
```

Insert the following directly after the first paragraph (the one starting "This repo is for a multi-agent RL project"):

````markdown
## Status

| Part | What | State |
|---|---|---|
| 1 | Rules engine, PettingZoo environment, bots, logging & replay | ✅ this branch |
| 2 | Training: PPO, Dueling DQN, NFSP, PSRO (PyTorch) | planned |
| 3 | Evaluation: win rates, matchups, strategy summaries | planned |

## Quick start

Works natively on Linux, macOS and Windows 10/11. Install [uv](https://docs.astral.sh/uv/) once:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone https://github.com/AaronBarel/Multi-Agent-RL-PrimaryBG.git
cd Multi-Agent-RL-PrimaryBG
uv sync
uv run primary simulate --config configs/sim_heuristic.yaml   # 1,000 games, logs to data/runs/
uv run primary replay --run data/runs/sanity_heuristic --game-id 0
uv run pytest
```

Explore the logs with `uv run --group analysis jupyter lab notebooks/01_engine_sanity.ipynb`.

## How the game maps to an MDP

- **Agents:** one per seat (`player_0` … `player_4`), each playing a candidate; turn-based PettingZoo AEC API.
- **Actions:** one flat `Discrete(31)` space with a legal-action mask. Each round an agent commits its 4 Action cards one at a time (hidden from opponents), then chooses targets (where to fly, which cube to remove, …) when each card is revealed.
- **Observations:** an egocentric vector (675 values for 5 players): board cubes, pawns, hands, banks, scores, the public voting schedule, News, revealed cards, and the current decision. Opponents' face-down cards are never observed.
- **Rewards:** +1 for a win (1/k each in a k-way draw), optional vote-share shaping.
- **Randomness:** voting schedule, News deck, starting player — all derived from one seed, so every game replays exactly.

Full design: [`docs/superpowers/specs/2026-10-15-primary-engine-design.md`](docs/superpowers/specs/2026-10-15-primary-engine-design.md), including every rules clarification (numbered rules 1–52, each covered by tests).

## Logged data

Each run writes three Parquet tables to `data/runs/<run>/`:

- `games` — seed, candidates, policies, voting schedule, News order, final scores, winners, tie-breaker
- `rounds` — per player per round: start/end/intermediate locations, hand and bank before and after, cards played, vote outcome per region
- `events` — every atomic change: moves, cubes placed/removed/returned (where, whose, from hand or bank), gains, decisions

## Repository layout

```
src/primary/data      game data from the rulebook
src/primary/engine    rules engine (pure Python, no RL dependencies)
src/primary/env       PettingZoo environment
src/primary/agents    random and heuristic baseline bots
src/primary/logs      Parquet logging and replay
configs/              simulation run configs
notebooks/            analysis notebooks
tests/                pytest + Hypothesis test suite
```

## Attribution

*The Primary* is a self-published board game by Matt Quock. This project is an independent, non-commercial study of the game's balance and is not affiliated with the designer.
````

- [ ] **Step 4: Generate the sanity-check run**

Run: `uv run primary simulate --config configs/sim_heuristic.yaml --overwrite`
Expected: 1,000 games logged to `data/runs/sanity_heuristic`; the heuristic's win rate is well above 0.2.

- [ ] **Step 5: Build the notebook**

Run this one-off builder (it is not committed; the notebook is):

```bash
uv run --group analysis python - <<'EOF'
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Engine sanity check\n\n"
       "Loads a logged simulation run (one heuristic bot vs. four random bots, rotating seats) "
       "and checks that the engine and logs behave sensibly before any RL training.\n\n"
       "Generate the data with `uv run primary simulate --config configs/sim_heuristic.yaml`; "
       "the first cell creates it if it is missing."),
    code("from pathlib import Path\n\n"
         "import matplotlib.pyplot as plt\n"
         "import pandas as pd\n\n"
         "from primary.simulate import SimConfig, run_simulation\n\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "cfg = SimConfig.from_yaml(ROOT / 'configs' / 'sim_heuristic.yaml')\n"
         "RUN = ROOT / cfg.out_dir\n"
         "if not (RUN / 'games.parquet').exists():\n"
         "    cfg.out_dir = str(RUN)\n"
         "    run_simulation(cfg)\n"
         "games = pd.read_parquet(RUN / 'games.parquet')\n"
         "rounds = pd.read_parquet(RUN / 'rounds.parquet')\n"
         "events = pd.read_parquet(RUN / 'events.parquet')\n"
         "print(f'{len(games)} games, {len(rounds)} player-rounds, {len(events)} events')"),
    md("## Win rates\n\nA seat's win counts 1/k in a k-way draw. With 5 players, chance is 0.20."),
    code("rows = []\n"
         "for g in games.itertuples():\n"
         "    share = 1 / len(g.winners)\n"
         "    for seat, (cand, pol, score) in enumerate(zip(g.candidates, g.policies, g.final_scores)):\n"
         "        rows.append({'game_id': g.game_id, 'seat': seat, 'candidate': cand, 'policy': pol,\n"
         "                     'score': score, 'win': share if seat in g.winners else 0.0})\n"
         "seats = pd.DataFrame(rows)\n"
         "seats.groupby('policy')['win'].agg(['mean', 'count']).rename(columns={'mean': 'win rate'})"),
    code("by_candidate = seats[seats.policy == 'random'].groupby('candidate')['win'].mean().sort_values()\n"
         "fig, ax = plt.subplots(figsize=(7, 4))\n"
         "ax.barh(by_candidate.index, by_candidate.values, color='#4C72B0')\n"
         "ax.axvline(0.2, color='grey', linestyle='--', linewidth=1, label='chance (1/5)')\n"
         "ax.set_xlabel('win rate (random policy only)')\n"
         "ax.set_title('Candidate strength under random play')\n"
         "ax.legend(frameon=False)\n"
         "fig.tight_layout()"),
    md("## Final scores"),
    code("fig, ax = plt.subplots(figsize=(7, 4))\n"
         "for policy, group in seats.groupby('policy'):\n"
         "    ax.hist(group.score, bins=30, alpha=0.6, label=policy)\n"
         "ax.set_xlabel('final votes')\n"
         "ax.set_ylabel('seats')\n"
         "ax.legend(frameon=False)\n"
         "fig.tight_layout()"),
    md("## Round outcomes and cube flow"),
    code("merged = rounds.merge(seats[['game_id', 'seat', 'policy']], on=['game_id', 'seat'])\n"
         "pd.crosstab(merged.policy, merged.round_result, normalize='index').round(3)"),
    code("gains = events[events.event_type == 'gain'].groupby('detail')['amount'].sum()\n"
         "gains.sort_values(ascending=False).rename('cubes gained')"),
    md("## Draws and tie-breakers"),
    code("games.tiebreak.value_counts()"),
    md("**What to look for:** the heuristic should win far more than 20% of seats; every "
       "non-voting round (1, 4, 9) should be `no_vote`; draws should be rare."),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/01_engine_sanity.ipynb")
EOF
uv run --group analysis jupyter nbconvert --to notebook --execute --inplace notebooks/01_engine_sanity.ipynb
```

Expected: the notebook executes without errors and contains the tables and two charts.

- [ ] **Step 6: Final verification**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`
Expected: all four succeed. Paste the pytest summary line into the commit body.

- [ ] **Step 7: Commit**

```bash
git add .github README.md notebooks src tests
git commit -m "Add CI, README quick start and engine sanity notebook" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage map

| Spec section | Task(s) |
|---|---|
| §1 success criteria: every rule tested | 2–8 (tests cite `# rule N`) |
| §1 `api_test` | 12 |
| §1 reproducible from seed + actions | 7, 8, 10 |
| §1 data-collection list | 10 (`test_logs_cover_the_claude_md_data_collection_list`) |
| §1 Linux + Windows, `uv run primary simulate` | 11, 13 |
| §1 heuristic beats random | 11 |
| §2 libraries | 1 |
| §3 rules 1–52 | 2 (17–20), 3 (1, 3, 4, 21–51 modifiers), 4 (13, 15, 16, 34, 40, 50), 5 (8–14, 21, 26, 28, 30, 36, 41, 42, 46–48, 51, 52), 6 (2, 8, 15, 19, 20, 22–24, 27, 31, 35, 40, 45, 49), 7 (5–7, 25, 29, 32, 33, 37–39, 43, 44, 50) |
| §4 MDP (decisions, action space, observation, rewards) | 3, 5, 7, 12 |
| §5 architecture, hook points, resolution order, invariants | 3–8 |
| §6 logging and replay | 10, 11 |
| §7 testing | every task; 8 (property), 11 (sanity), 12 (api_test) |
| §8 CI and Windows quick start | 13 |
| §9 deliverables (incl. notebook) | 11–13 |
