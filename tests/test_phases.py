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
