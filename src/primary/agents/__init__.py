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
