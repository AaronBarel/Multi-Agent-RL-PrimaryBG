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
