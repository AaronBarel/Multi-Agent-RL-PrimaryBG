# The Primary — Game Engine, Environment & Logging (Sub-project 1) Design

- **Date:** 2026-10-15
- **Status:** Draft for review
- **Branch:** `feature/game-engine`
- **Source rules:** `CLAUDE.md` (authoritative game data) + the clarifications in §3 below

## 1. Context and goals

This repository is a public portfolio project: multi-agent reinforcement learning (MARL) for
*The Primary*, a self-published board game by Matt Quock. Each of the 12 candidates is an agent; we want
optimal policies per candidate, global and matchup win rates, strategy summaries, and a comparison of RL
methods (PPO, NFSP, Dueling DQN, plus PSRO as our own pick, with an ISMCTS search baseline).

The work is decomposed into three sub-projects, each with its own spec → plan → implementation cycle:

1. **Game engine, environment, and logging** (this document)
2. Training: four algorithms in PyTorch, self-play across the candidate pool, checkpoints, Windows GPU runs
3. Evaluation: tournaments, win-rate/matchup statistics with confidence intervals, strategy summaries, notebooks

**Success criteria for sub-project 1**

- Every rule in `CLAUDE.md` and §3 is implemented and covered by at least one test.
- The PettingZoo AEC environment passes `pettingzoo.test.api_test`.
- Any game can be exactly reproduced from its seed + logged actions.
- Logs capture every item in the `CLAUDE.md` "Data Collection" list.
- `uv sync && uv run primary simulate ...` works on Linux and on Windows 10 natively; CI covers both.
- A heuristic bot beats a random bot by a statistically significant margin (sanity check of the engine and logs).

**Out of scope:** any RL training code, PyTorch dependency, win-rate/matchup analysis (sub-projects 2–3).

## 2. Library decisions

- **Environment API:** PettingZoo (AEC API) with action masking.
- **RL (sub-project 2):** CleanRL-style single-file PyTorch implementations of PPO and Dueling DQN; NFSP and
  PSRO implemented in PyTorch with tests and paper citations. Chosen over RLlib/OpenSpiel because OpenSpiel has
  no Windows wheels and RLlib is heavy and lacks NFSP; a single shared codebase also makes the method
  comparison fair.
- **Engine:** pure Python (approach 1 of 3 considered). A vectorized NumPy/Numba engine was rejected for
  readability; a differential-tested fast engine is a possible later upgrade only if profiling shows the
  engine is the bottleneck.
- **Tooling:** Python ≥ 3.11 (developed on 3.12), `uv`, `pytest`, `hypothesis`, `ruff`, `mypy`, `pyarrow`,
  `numpy`, `pyyaml`, `gymnasium`, `pettingzoo`.

## 3. Rules clarifications (resolved with the project owner)

These supplement `CLAUDE.md`. Items marked *(assumption)* were not explicitly discussed and are chosen for
consistency; they are the first things to revisit if a test or game result looks wrong.

### Setup

1. **Players:** default 5 per game, candidates sampled without replacement from the 12. Player count is
   configurable (3–6) for analysis.
2. **Seats and turn order:** seats are fixed clockwise. The starting-player token is assigned randomly at
   setup and passes clockwise at the end of each round.
3. **Voting schedule:** the 14 regions are randomly permuted into the 14 voting slots
   (R2:1, R3:2, R5:1, R6:2, R7:1, R8:2, R10:2, R11:3). Each region votes exactly once. The schedule is public.
4. **Cubes:** each player owns 30. Hand = candidate's starting cubes; bank = the rest.
5. **Home regions:** in turn order starting with the starting player, each player chooses a home region,
   places their pawn there and places 2 cubes **from the bank** on it. Players may share a home region.
   Georgia Barnes chooses 3 distinct home regions with 2 bank cubes each; her pawn starts on the first one
   she chooses *(assumption)*.

### Action cards

6. Every player has all 14 Action cards available every round (except under Frozen Assets / Final Push) and
   commits an ordered sequence of 4.
7. **Order is committed face-down at the start of the Action phase; targets are chosen when each card is
   executed**, with full knowledge of what has already been revealed.
8. **Super PAC:** compared across all players by number of Super PAC cards played. A strict unique maximum
   gains 4; players tied for the maximum gain 2 each. Zero is never "the most". Gains come from the bank.
9. **Fundraiser:** move 1 cube bank → hand.
10. **Plane:** pay 1 cube hand → bank and move to any region other than the current one. With 0 cubes in hand
    the card is executed with no effect (no move).
11. **Bus:** move to an adjacent region (mandatory if the card takes effect).
12. **Positive Ad:** place 1 cube from hand onto any region. With 0 cubes in hand, no effect.
13. **Negative Ad:** remove any 1 cube from any region (an opponent's or your own) to its **owner's bank**.
    The player must choose a cube if any exist on the map; if the map has no cubes, no effect.
14. **Rally:** requires 2 cubes in hand; places 2 cubes from hand onto the pawn's region. With fewer than 2,
    the card is executed with no effect.
15. **Cubes persist** on regions that did not vote this round. At round end, cubes on regions that voted this
    round return to owners' banks. Cubes on any region that **already voted in an earlier round** (e.g. placed
    there later, or moved there by Erin / New Constituents) also return to their owners' banks at round end
    *(assumption, consistent with the New Constituents ruling)*.
16. **Empty sources:** every gain is capped by the remaining bank. Every placement requires its stated source
    (hand or bank); if empty, the effect does not happen and a `no_effect` event is logged.

### Voting

17. **Winner-take-all regions:** most influence wins all votes; a tie splits the votes evenly, rounded down.
18. **Two-award regions (1st/2nd):**
    - Unique 1st and unique 2nd: each receives their award.
    - Only one player has influence: that player receives 1st + 2nd.
    - Tie for 1st: tied players split (1st + 2nd) evenly, rounded down; no 2nd place is awarded.
    - Unique 1st, tie for 2nd: tied players split the 2nd award evenly, rounded down.
19. **Influence** = cubes on the region, plus 2 for Anna Patel if her pawn is there (her token counts as
    influence even with zero cubes, including for the "only one player has influence" rule).
20. **Winning:** most votes after round 11; tie-breaker 1 = most cubes in hand; tie-breaker 2 = most cubes in
    bank; otherwise a draw (tracked).

### Candidate abilities

21. **Antonio Parades:** Bus may move to any region within 1 or 2 adjacency steps (not the current one).
22. **Kenneth Walker:** if he executes ≥ 2 Fundraisers in a round, gain 1 extra cube at round end.
23. **Bryce McKay:** once per **voting round**, if he wins no award (1st or 2nd place counts as a win) in any
    region voted that round, gain 1 cube. Does not trigger in non-voting rounds.
24. **Kristen Ramirez:** if she plays the **unique** most Super PACs, gain 1 extra cube *(assumption: the
    card text distinguishes "most" from "tied")*.
25. **Diane Proudian:** once per round, when one of her cards is revealed, she may skip executing it. A
    skipped card still counts as played for Frozen Assets but not for Super PAC/Fundraiser counts.
26. **Ryan Gupta:** Plane costs nothing.
27. **Chester Freeman:** if his pawn ends the round in the same region as any opponent's pawn, gain 1 cube.
28. **Lily Nguyen:** the first Rally **that takes effect** each round may place 1 extra cube from hand.
29. **Erin Fields:** before executing her first Action card each round, she may move 1 of her own cubes from
    any region to any region.
30. **Calvin Matsui:** Bus may place 1 cube from hand onto the destination (optional).
31. **Anna Patel:** see rule 19.
32. **Georgia Barnes:** see rule 5.

### News cards

33. A News card is drawn every round, including non-voting rounds 1, 4, 9 (without replacement from 18).
    Immediate choices are made in turn order starting with the starting player.
34. **Countermeasures:** gain 1 cube each time your pawn enters a region where any opponent has cubes
    (checked at the destination, before any placement there). For a multi-step Bus move only the final
    destination counts *(assumption)*.
35. **Oppo Research:** pawns ending the round in the same region as any opponent's pawn gain 2.
36. **Generous Donors:** +1 extra cube per Fundraiser executed.
37. **Tax Reform:** every player immediately gains 2.
38. **Last Minute Debate:** each player may move (free) to a region voting this round or the literal next
    round.
39. **Detour:** each player must move (free) to any region other than the current one.
40. **Frozen Assets:** cards committed this round are unavailable next round. Redrawn if drawn in round 11.
41. **Paparazzi:** Bus and Plane (when they take effect) may place 1 cube from the bank onto the destination.
42. **Spin Doctor:** each Positive/Negative Ad chooses which effect to use when executed.
43. **Final Push:** each player commits 4 or 5 cards (choosing to stop after the 4th is a legal action).
44. **Long Term Planning:** each player may move (free) to a region voting in round 11.
45. **Deregulation:** unique most Fundraisers executed (non-zero) gains 2; tied (non-zero) players gain 1.
46. **Extra Airtime:** Positive Ad places 2 cubes from hand, each onto any region (two independent
    placements; each requires a cube in hand) *(assumption)*.
47. **Wall Street Support:** each player's first Rally uses 2 cubes from the bank instead of the hand (needs
    2 in bank). Lily's extra cube still comes from her hand.
48. **Road Trip:** Bus moves 1 adjacent region further (normal Bus: up to 2 steps; Antonio: up to 3).
49. **Plutocracy:** unique most Super PACs gains 6, ties gain 3 — **in addition** to the normal Super PAC
    award *(assumption)*.
50. **New Constituents:** each player may swap their pawn with any single cube on the map (anyone's): the
    cube moves to the pawn's old region and the pawn moves to the cube's region. A cube landing on a region
    that already voted returns to its owner's bank immediately. A swap within the same region is not offered.
51. **Frequent Flyer:** Planes cost nothing.
52. **Stacking:** abilities and News effects stack additively (e.g. Calvin + Paparazzi = 1 hand cube + 1 bank
    cube on the Bus destination).

## 4. MDP formulation

### 4.1 Agents and turn structure

PettingZoo AEC environment: at every step exactly one agent makes one decision. Agents are named by seat
(`player_0` … `player_{P-1}`); the seat→candidate mapping is part of the observation.

### 4.2 Decision types

| Decision | When | Choices |
|---|---|---|
| `HOME_REGION` | Setup, in turn order | 14 regions (Georgia: 3 distinct) |
| `NEWS_CHOICE` | After the News card is revealed, in turn order, only for cards with a choice | region / cube (region → owner) / pass |
| `COMMIT_CARD` | Commit phase, slot k = 1..4 (5 with Final Push) | 7 card types, masked by remaining copies and frozen cards; `STOP` after slot 4 under Final Push |
| `ERIN_PREMOVE` | Before Erin's first card each round | source region of one of her cubes or pass → destination region |
| `DIANE_SKIP` | When each of Diane's cards is revealed (until she uses her skip) | execute / skip |
| `CARD_TARGET` | When a card executes | Plane: any other region · Bus: reachable regions · Positive Ad: region (×2 under Extra Airtime) · Negative Ad: region → owner · Spin Doctor: Ad type · optional bonuses (Lily, Calvin, Paparazzi): yes/no |

Commits are made one card at a time, but opponents' commits are hidden, so this is strategically equivalent to
simultaneous commitment while giving 7-way choices instead of ~2,000 sequence choices.

### 4.3 Action space

A single flat `Discrete(N)` space (N ≈ 30): 14 region slots, 7 card slots, up to 6 player slots, yes/no, pass,
stop. Every observation includes a legal-action mask and a decision-type one-hot. Multi-part choices
(Negative Ad region → owner, Erin source → destination, New Constituents cube region → owner) are split into
consecutive decisions with their own decision types.

### 4.4 Observation (egocentric: the acting agent is always seat 0)

- Cubes per region per player (14 × P), pawn locations (14 × P one-hot)
- Hand, bank, score per player
- Round number; per-region voting round (one-hot over 11) and already-voted flag
- Current News card (one-hot 18) and used News cards
- Candidate per seat (12 one-hot × P), seat offset from the starting-player token
- Own committed queue and executed flags; opponents' revealed cards this round and count still face-down
- Frozen cards (Frozen Assets carry-over); Diane-skip-used flag
- Decision type one-hot + decision context (e.g. Negative Ad step 2 with the chosen region)
- Legal-action mask (separate key, PettingZoo convention: `{"observation", "action_mask"}`)

Approximate size ~600 floats. Hidden information: opponents' face-down cards and the News deck order.

### 4.5 Rewards

- **Default:** terminal only. +1 for an outright win, 0 otherwise; a k-way draw (after both tie-breakers)
  gives 1/k to each tied player.
- **Optional (config flag, off by default):** per-round shaping = votes gained ÷ total votes available. Shaped
  vs. unshaped training is one of the sub-project 2 experiments.
- γ = 1 (fixed 11-round horizon).

### 4.6 Stochasticity

Voting schedule, News deck shuffle, starting-player token, and candidate draw, all derived from a single
per-game seed via `numpy.random.Generator`. Opponent policies are part of each agent's environment.

## 5. Architecture

### 5.1 Package layout

```
pyproject.toml            uv-managed; console script `primary`
configs/                  YAML run configs (seed, players, agents, logging)
src/primary/
  data/       board.py candidates.py cards.py news.py   # typed constants mirroring CLAUDE.md
  engine/
    state.py              GameState dataclass (numpy arrays + small Python fields)
    game.py               round state machine
    effects.py            hook points and modifier registry
    news_effects.py       one function per News card
    abilities.py          one function per candidate ability
    voting.py             award computation
    movement.py           adjacency / reachability (BFS)
  env/
    aec_env.py            PettingZoo AECEnv
    encoding.py           observation + mask encoding
    rewards.py
  agents/
    random_agent.py       uniform over legal actions
    heuristic.py          rule-based bot (e.g. rally/advertise into soon-voting regions it can win)
  logs/
    schema.py             pyarrow schemas
    writer.py             buffered Parquet writer
    replay.py             rebuild a game from logs
  cli.py                  `primary simulate`, `primary replay`
tests/
notebooks/01_engine_sanity.ipynb
```

### 5.2 Engine interface

The engine has no RL dependencies:

```python
game = Game(config)
req = game.reset(seed, candidates)          # -> DecisionRequest(player, decision_type, legal_actions, context)
while not isinstance(req, GameOver):
    req = game.step(action)                 # -> DecisionRequest | GameOver
```

The PettingZoo env, bots, tests, CLI, and replay all drive this same interface. Event records are emitted by
the engine to an optional sink (the logger), so logging never changes game behaviour.

### 5.3 Hook points

`round_start`, `news_resolve`, `commit_constraints`, `before_first_card`, `card_resolve` (modifiers can change
cost, source, range, extra placements), `enter_region`, `round_end_bonuses`, `vote_result`, `round_end`.
Each News card and ability registers functions at these points; the engine core contains no
card- or candidate-specific `if` statements.

### 5.4 Resolution order per round

1. **News:** draw (redraw Frozen Assets in round 11); resolve immediate effects/choices in turn order.
2. **Commit:** each player commits 4 (or 4–5) cards.
3. **Execute:** cards are revealed and executed one per player in turn order, slot by slot. Erin's pre-move
   happens before her first card; Diane's skip decision is made at reveal.
4. **Round end:**
   1. Super PAC, Plutocracy, Deregulation, Kenneth, Kristen bonuses
   2. Voting on this round's regions
   3. Bryce bonus
   4. Chester and Oppo Research bonuses (pawn co-location)
   5. Return cubes on regions that voted this round or earlier to owners' banks
   6. Pass the starting-player token clockwise

### 5.5 Invariants (asserted after every step in debug mode and in tests)

- For each player: hand + bank + cubes on map = 30
- No negative counts; every `DecisionRequest` has ≥ 1 legal action
- Scores are non-decreasing; the game ends after exactly 11 rounds; each region votes exactly once

## 6. Data collection

Parquet files under `data/runs/<run_id>/` (git-ignored), written in buffered row groups.

| Table | Grain | Key columns |
|---|---|---|
| `games` | one per game | game_id, seed, engine_version, seat→candidate, policy ids, voting schedule, news order, final score/hand/bank per seat, winner(s), draw flag, deciding tie-breaker |
| `events` | one per atomic event | game_id, round, phase, step, seat, candidate, event_type, card, from_region, to_region, cube_owner, amount, source (hand/bank), hand/bank before & after, pawn before & after, decision_type, action |
| `rounds` | one per seat per round | start/end location, intermediate locations, hand/bank at start and end, ordered cards played, per-region vote outcome (won_1st / won_2nd / lost / not_participated), votes gained |

`event_type` ∈ {`news_drawn`, `news_choice`, `card_committed`, `card_revealed`, `card_skipped`, `move`,
`cube_placed`, `cube_removed`, `cube_moved`, `cube_returned`, `gain`, `payment`, `no_effect`,
`vote_awarded`, `bonus`}.

Logging is configurable: off, every Nth game, or all games (evaluation default). `replay.py` reconstructs any
game from `games` + `events` decision rows; a test verifies the reconstructed final state equals the original.

## 7. Testing

- **Rule tests:** a scenario builder (`make_state(cubes=..., pawns=..., hand=..., news=...)`) sets up exact
  positions. Each News card and ability has at least one positive test and one edge-case test (empty
  bank/hand, stacking, Frozen Assets in round 11, etc.). Each numbered rule in §3 maps to at least one test.
- **Voting tests:** table-driven cases for rules 17–20.
- **Property tests (Hypothesis):** thousands of seeded random games checking §5.5 invariants.
- **Determinism & replay:** same seed + actions ⇒ identical logs; replay round-trip equality.
- **PettingZoo `api_test`** compliance.
- **Sanity baseline:** heuristic vs. random win rate, with a binomial test (p < 0.01) over a fixed seed set.
- **Benchmark:** games/second printed (not gating).

## 8. Quality, CI, and running on Windows

- `ruff` (lint + format) and `mypy` on `engine/` and `env/`.
- GitHub Actions matrix: `ubuntu-latest` and `windows-latest`, Python 3.11 and 3.12; lint + tests.
- README quick start (Windows PowerShell):

  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  git clone <repo> ; cd Multi-Agent-RL-PrimaryBG
  uv sync
  uv run primary simulate --config configs/sim_heuristic.yaml
  ```

- Runs are fully specified by a YAML config + seed. The CUDA PyTorch extra arrives in sub-project 2.
- Game attribution: README credits Matt Quock and links the existing reference.

## 9. Deliverables

Engine, PettingZoo environment, random and heuristic agents, logging and replay, CLI, tests, CI workflow,
README quick start, `notebooks/01_engine_sanity.ipynb` loading a simulation run's logs to demonstrate the data
end-to-end.
