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
