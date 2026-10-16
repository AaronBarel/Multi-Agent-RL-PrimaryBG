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
