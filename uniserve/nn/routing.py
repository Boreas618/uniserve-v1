"""Mathematical token routes independent of request or device ownership.

A routed layer (a mixture-of-transformers layer) gives each route its own
modules over one packed token domain. :class:`IndexedRoutes` assigns tokens
to routes with static shapes: every token takes the *dense* route, and each
*sparse* route then takes over a fixed-size list of token rows, whose
results replace the dense route's. The lists are device tensors, so which
tokens a sparse route takes may change between evaluations of one captured
computation while every shape stays the same.

A sparse route evaluates only its listed rows; the dense route evaluates
every token, the listed rows included, and its results there are
overwritten before any operation mixes tokens.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["IndexedRoutes", "SparseRoute"]

# A routed module's per-token arguments: a [tokens, ...] tensor, or a tuple
# of them (such as the cosine factors of every rotary axis).
_Tokenwise = torch.Tensor | tuple[torch.Tensor, ...]


def _rows(value: _Tokenwise, rows: torch.Tensor) -> _Tokenwise:
    """Select token rows of a tensor, or of every tensor of a tuple."""
    if isinstance(value, tuple):
        return tuple(part.index_select(0, rows) for part in value)
    return value.index_select(0, rows)


@dataclass(frozen=True, slots=True)
class SparseRoute:
    """The token rows one route takes over from the dense route.

    Attributes:
        route: The route's name.
        rows: ``int64 [S]`` packed token rows; ``S`` is static.
        keep: ``bool [S]`` the entries that route their row. Kept entries
            name distinct rows. Entries not kept stand for absent rows (such
            as the markers of padding rows); once the routes are a shard's
            (:meth:`IndexedRoutes.shard`), each of them names a row that no
            kept entry names, so it leaves that row's dense result as it is.
    """

    route: str
    rows: torch.Tensor
    keep: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.route, str) or not self.route:
            raise ValueError("route names must be nonempty strings")
        if (
            self.rows.ndim != 1
            or self.rows.dtype != torch.int64
            or self.keep.shape != self.rows.shape
            or self.keep.dtype != torch.bool
            or self.keep.device != self.rows.device
        ):
            raise ValueError(
                "a sparse route holds int64 token rows and a boolean keep "
                "mask of one shape on one device"
            )

    @classmethod
    def of(cls, route: str, rows: torch.Tensor) -> SparseRoute:
        """A route over ``rows``, where -1 marks an absent entry."""
        return cls(route, rows.clamp_min(0), rows >= 0)


@dataclass(frozen=True, slots=True)
class IndexedRoutes:
    """Assign one packed token domain to routes with static shapes.

    Every token takes ``dense``; each of ``sparse`` replaces the results of
    its kept rows with its own. A layer applies the routes of its token
    shard (:meth:`shard`), whose rows are shard-local.

    Attributes:
        dense: The route every token takes.
        sparse: Routes taking over listed rows, each named once and none
            named ``dense``.
    """

    dense: str
    sparse: tuple[SparseRoute, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.dense, str) or not self.dense:
            raise ValueError("route names must be nonempty strings")
        names = [route.route for route in self.sparse]
        if self.dense in names or len(set(names)) != len(names):
            raise ValueError(
                "sparse routes must be distinct and differ from the dense route"
            )

    @property
    def names(self) -> frozenset[str]:
        """Every route the tokens take."""
        return frozenset((self.dense, *(route.route for route in self.sparse)))

    def shard(self, start: int, count: int) -> IndexedRoutes:
        """The routes of the shard of ``count`` tokens from row ``start``.

        Rows become shard-local. An entry whose row lies outside the shard
        is not kept, and every entry not kept is pointed at the first row of
        the shard that no kept entry names, so applying the routes leaves
        that row's dense result unchanged. Such a row exists while a shard's
        kept entries are fewer than its tokens; a framed image sequence has
        more tokens than markers, which keeps every shard of a whole number
        of sequences so. A shard without tokens keeps no entry. The work is
        device-side and reads no device value on the host.
        """
        start, count = int(start), int(count)
        sparse = []
        for route in self.sparse:
            if count == 0:
                # An empty shard routes nothing; its sparse modules still
                # evaluate (on no rows), as the other shards' do.
                empty = route.rows[:0]
                sparse.append(SparseRoute(route.route, empty, route.keep[:0]))
                continue
            local = route.rows - start
            keep = route.keep & (local >= 0) & (local < count)

            # Rows a kept entry names; the extra last slot absorbs the
            # entries not kept, so the scatter has a static shape.
            taken = torch.zeros(
                count + 1, dtype=torch.int8, device=local.device
            )
            taken.index_fill_(0, torch.where(keep, local, count), 1)
            # ``argmin`` returns the first row no kept entry names.
            free = torch.argmin(taken[:count])
            rows = torch.where(keep, local, free)
            sparse.append(SparseRoute(route.route, rows, keep))
        return IndexedRoutes(self.dense, tuple(sparse))

    def apply(
        self,
        modules: Mapping[str, nn.Module] | nn.ModuleDict,
        value: torch.Tensor,
        *tokenwise: _Tokenwise,
    ):
        """Evaluate each route's module over its tokens.

        The dense route's module receives ``value`` and the ``tokenwise``
        arguments whole; a sparse route's module receives its rows of each.
        Every module returns a tensor, or a tuple of tensors, with a leading
        token axis; the result is the dense module's with each sparse
        route's kept rows replaced in place. The modules must return fresh
        tensors, which the replacement writes.

        Raises:
            ValueError: A route has no module.
        """
        if self.names.difference(modules):
            raise ValueError(
                "every numerical route requires a corresponding module"
            )
        result = modules[self.dense](value, *tokenwise)
        outputs = result if isinstance(result, tuple) else (result,)
        for route in self.sparse:
            routed = modules[route.route](
                value.index_select(0, route.rows),
                *(_rows(argument, route.rows) for argument in tokenwise),
            )
            for output, values in zip(
                outputs,
                routed if isinstance(routed, tuple) else (routed,),
                strict=True,
            ):
                # Entries not kept write back the dense result of a row no
                # kept entry names, so their order among the copies does
                # not matter.
                keep = route.keep.view(-1, *(1,) * (values.ndim - 1))
                current = output.index_select(0, route.rows)
                output.index_copy_(
                    0, route.rows, torch.where(keep, values, current)
                )
        return result
