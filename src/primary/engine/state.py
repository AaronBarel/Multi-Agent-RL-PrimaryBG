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
