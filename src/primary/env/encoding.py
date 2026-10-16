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
        + MAX_QUEUE * N_CARDS
        + MAX_QUEUE  # own queue, own executed flags
        + (n - 1) * MAX_QUEUE * N_CARDS  # opponents' revealed cards
        + 2 * n  # committed / revealed counts
        + N_CARDS  # own frozen cards
        + n  # skip used
        + len(DecisionType)
        + N_CARDS
        + NUM_REGIONS  # decision type and context
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
    counts = [[len(rnd.queue[s]) / MAX_QUEUE, len(rnd.revealed[s]) / MAX_QUEUE] for s in order]
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
