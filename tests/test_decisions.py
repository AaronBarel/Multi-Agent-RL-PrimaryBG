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
