"""A valid expert sort, built on the host, as the grouped GEMM checks state it.

The grouped products (`gather_gemm`, `route_gemm`) read the tables `moe_sort`
writes: routes grouped by expert, each expert's rows cut into tiles of
`TILE_ROWS`, `tile_to_expert` and `tile_to_limit` describing the tiles,
`permuted_to_expanded` the map from a permuted row to its route
`token · top + slot`, padded to whole tiles, and the count of live tiles.
Building the sort here, rather than assuming one, is what lets a case hold an
expert with no routes and a last tile that is only partly filled: the two
places a product that ignores `tile_to_limit` writes outside its tile.
"""

import torch

#: The descriptors' M tile: the rows of one tile of a sorted expert.
TILE_ROWS = 128


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


def sort_metadata(per_expert, top):
    """`tile_to_expert`, `tile_to_limit`, `permuted_to_expanded` and the live
    tile count, as `moe_sort` writes them.
    """  # noqa: D205
    tile_to_expert, tile_to_limit, permuted_to_expanded = [], [], []
    for expert, members in enumerate(per_expert):
        for start in range(0, len(members), TILE_ROWS):
            chunk = members[start : start + TILE_ROWS]
            tile_to_expert.append(expert)
            tile_to_limit.append(len(chunk))
            permuted_to_expanded.extend(
                token * top + slot for token, slot in chunk
            )
            # The sort pads every tile to whole rows; the map names each.
            permuted_to_expanded.extend([0] * (TILE_ROWS - len(chunk)))
    if not tile_to_expert:
        tile_to_expert, tile_to_limit = [-1], [0]
        permuted_to_expanded = [0] * TILE_ROWS
    live = sum(1 for limit in tile_to_limit if limit > 0)
    return tile_to_expert, tile_to_limit, permuted_to_expanded, live
