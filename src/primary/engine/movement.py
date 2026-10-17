"""Graph distances on the region map (Bus range, heuristics)."""

from __future__ import annotations

from collections import deque
from functools import cache

from primary.data.board import ADJACENCY, NUM_REGIONS, Region


@cache
def distances_from(src: Region) -> tuple[int, ...]:
    """Breadth-first distances from ``src`` to every region, indexed by region."""
    dist = [-1] * NUM_REGIONS
    dist[src] = 0
    queue = deque([src])
    while queue:
        current = queue.popleft()
        for nxt in sorted(ADJACENCY[current]):
            if dist[nxt] < 0:
                dist[nxt] = dist[current] + 1
                queue.append(nxt)
    return tuple(dist)


def distance(a: Region, b: Region) -> int:
    return distances_from(a)[b]


def reachable(src: Region, max_steps: int) -> tuple[Region, ...]:
    """Regions 1..max_steps steps away from ``src`` (never ``src`` itself)."""
    dist = distances_from(src)
    return tuple(r for r in Region if 0 < dist[r] <= max_steps)
