"""A valid expert sort, built on the host, as the grouped GEMM checks state it.

The grouped products (`gather_gemm`, `route_gemm`) read the tables `moe_sort`
writes (``crates/ops/src/moe/sort.rs``): routes grouped by expert, each
expert's rows cut into tiles of `TILE_ROWS`; `tile_to_expert` and
`tile_to_limit` describing the tiles, the limit being the permuted row one
past a tile's last live row; `permuted_to_expanded`, the route
`token · top + slot` of each permuted row, padded to whole tiles with `-1`;
`expanded_to_permuted`, each route's permuted row, `-1` for a route no expert
holds; and the count of live tiles. The combine (`moe_unpermute`) finds a
route's product at its `expanded_to_permuted` row, so that map is how a check
reads a product's rows as the engine does.

Building the sort here, rather than assuming one, is what lets a case hold the
two shapes a product must read from the tables instead of assuming: an expert
with no routes, after which a tile's index is no longer its expert's, and
tiles whose live rows end before their last row.
"""

from typing import NamedTuple

import torch

#: The descriptors' M tile: the rows of one tile of a sorted expert.
TILE_ROWS = 128


class Sort(NamedTuple):
    """The tables `moe_sort` writes for one batch's routes, as host lists."""

    #: `[tiles]`: the expert each tile's rows belong to.
    tile_to_expert: list[int]
    #: `[tiles]`: the permuted row one past each tile's last live row,
    #: counted from the first permuted row of the whole sort.
    tile_to_limit: list[int]
    #: `[tokens · top]`: each route's permuted row, `-1` for a route no
    #: expert holds.
    expanded_to_permuted: list[int]
    #: `[tiles · TILE_ROWS]`: each permuted row's route, `-1` on a tile's
    #: padding.
    permuted_to_expanded: list[int]
    #: The tiles that hold live rows.
    live_tiles: int


def routed(tokens, experts, top, emptied):
    """`[tokens, top]` routes, each token's `top` experts distinct, with the
    expert `emptied` given no route and route `[0, 0]` dead (`-1`).

    At these sizes every expert expects routes, so a dropped route empties
    none: the emptied expert's routes move to its neighbour, which keeps every
    token's route count. The dead route leaves one tile only partly filled.
    """  # noqa: D205
    routes = torch.stack([torch.randperm(experts)[:top] for _ in range(tokens)])
    routes[routes == emptied] = (emptied + 1) % experts
    routes[0, 0] = -1
    return routes


def tiles_of(routes, experts):
    """The sort's view of the routes: one list of `(token, slot)` per expert,
    in order.
    """  # noqa: D205
    per_expert = [[] for _ in range(experts)]
    for token, slots in enumerate(routes.tolist()):
        for slot, expert in enumerate(slots):
            if expert >= 0:
                per_expert[expert].append((token, slot))
    return per_expert


def sort_metadata(per_expert, tokens, top):
    """The :class:`Sort` of `tokens` tokens' `top` routes grouped as
    `per_expert` (:func:`tiles_of`), as `moe_sort` writes it: the experts'
    tiles follow each other in expert order, an expert without routes owns
    none.
    """  # noqa: D205
    tile_to_expert, tile_to_limit, permuted_to_expanded = [], [], []
    expanded_to_permuted = [-1] * (tokens * top)
    for expert, members in enumerate(per_expert):
        for start in range(0, len(members), TILE_ROWS):
            chunk = members[start : start + TILE_ROWS]
            first = len(permuted_to_expanded)
            tile_to_expert.append(expert)
            # A row index, not a count: the products compare a permuted row
            # with it.
            tile_to_limit.append(first + len(chunk))
            for place, (token, slot) in enumerate(chunk):
                expanded_to_permuted[token * top + slot] = first + place
                permuted_to_expanded.append(token * top + slot)
            # The sort pads every tile to whole rows and names no route there.
            permuted_to_expanded.extend([-1] * (TILE_ROWS - len(chunk)))
    if not tile_to_expert:
        tile_to_expert, tile_to_limit = [-1], [0]
        permuted_to_expanded = [-1] * TILE_ROWS
    live = sum(
        1
        for tile, limit in enumerate(tile_to_limit)
        if limit > tile * TILE_ROWS
    )
    return Sort(
        tile_to_expert,
        tile_to_limit,
        expanded_to_permuted,
        permuted_to_expanded,
        live,
    )
