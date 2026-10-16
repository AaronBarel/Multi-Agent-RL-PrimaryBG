# Multi-Agent-RL-PrimaryBG
This repo is for a multi-agent RL project that targets finding optimal policies for different characters in a non production board game.

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

# The Primary
The Primary board game is a self publish board game by Matt Quock (https://tompetgames.wordpress.com/2018/11/19/matt-quock-the-creator-of-the-primary/). The game board is a map of the United States that is broken up into 14 distinct regions. Each player chooses a candidate with a special ability and a different amount of initial currency. The objective of the game is to earn the most votes by the end of the game. The game board is broken up into 11 rounds. Rounds 1, 4, and 9 are non-voting rounds while the rest of the turns are randomly assigned to different regions for eligible voting. For example, round 3 has two tokens that can be assigned to two different regions. When round 3 is active, players can try to win the votes from that region. Each region is either winner take all votes as a reward or a first and second place voting reward. In the beginning of the game, each player, starting with the player with the initial possession of The Primary token, chooses their starting region and two initial currency cubes on the region. The round begins with the player who currently has The Primary Token. The Primary Token is randomly assigned in the beginning of the game and then transitions clockwise. Prior to placing the action cards, a News event card is randomly selected in the beginning of each round. Each player has the same set of action cards which the players set face down and execute in the order of which they placed the cards. The players take turns in revealing each consecutive action. The News event introduces an element of randomness to the environmnet which the players have to account for the immediate impact of their strategy as well as their long term strategy. Regions are won by having the most currency cubes on that region (or second most in the case of first and second place regions). At the end of each round, the winner of the round's eligible voting regions receives their votes and advances their point tracking token on the outer rim of the board.

## Game Pieces
- Round tokens: Assigned randomly in the beginning of the game to different regions. Round tokens 1, 4, and 9 are placed in the non-voting round section
- Eligible voting round tokens: These tokens are placed on the regions up for voting on the round (1-3 regions are elgible in each voting round)
- Candidate token: Tracks the players current location on the game board
- Votes tracker: Tracks the players current vote total
- The Primary: Determines player to execute first an action card
- Currency cubes: Cubes placed on regions to win votes or to execute specific actions. Cubes in the players hand are eligible for consumption while the cubes in the bank act as a reserve to draw from

## Action Cards
- Fundraiser x 3: Place one currency cube from your bank into your hand
- Super Pac x 4: The player who plays the most Super Pac cards in the round receives 4 currency cubes from their bank to their hand (2 in the case of ties)
- Rally x 3: Players place two cubes from their hand onto the region where their candidate token is
- Plane x 1: Spend one token from your hand to move anywhere on the map
- Bus x 1: Move to an adjacent region
- Positive Ad x 1: Place one currency cube on any region on the map
- Negative Ad x 1: Remove any one currency cube (an opponent's or your own) from anywhere on the map; it returns to its owner's bank

## Example Round
- A random News card is activated and the rules of the world slightly change
- - If a News card forces players to decide on actions immediately, the Primary token holder will go first and then go clockwise for the other players decisions
- Each of the players set their four Action cards face down in the order of which they want to execute (left to right in front of the player)
- The Primary token holder executes their first Action card and then the next player (clockwise) will execute theirs and this will continue until all players have executed their Action cards.
- The winner(s) of the rounds eligible voting regions receives their votes and advances their votes tracker
- Any News implication that applies at the end of the round gets resolved as well as candidates' abilities

# Multi Agent Reinforcement Learning
The objective of this project is to have 5 agents each find their own optimal policy to acheive the best strategy to win the game. This project will be set up in an Environment-Agent-Simulation framework (EAS). The agent's only mode of control is the Action cards to play and the order in which to execute them. We hope to find optimal policies for each candidate character in the game depending on their own specific ability and match-ups with the other candidates. From playing this game with my friends, we've noticed their are certain imbalances that make a candidate stronger or weaker in a 5 player format. This makes sense since it is an independently developed board game, but many production level board games and video games from the most successful companies also face imbalance issues. We hope that the agents can reveal optimal policies depending on the candidate ability and we also hope to simulate enough games to estimate a candidates strength. The agents will have to deal with randomness introduced to the environment via News cards and the other agent's actions in order to develop a strategy to win the game. The agent's will have to evaluate immediate rewards versus a long-term winning strategy.
