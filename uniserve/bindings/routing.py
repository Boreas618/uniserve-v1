"""Runtime storage for expert routing and numerical redistribution layouts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from uniserve.loading.retained import Layout
from uniserve.nn.moe import FusedMoE


class ExpertRouting:
    """Keep graph-addressable logical-to-physical expert maps for a model.

    Each layer borrows its table. The caller moves weights into the requested
    physical slots before committing at a drained computation boundary; the
    commit updates table contents without changing their addresses. An order
    lists logical expert IDs in global physical-slot order, retaining the
    existing number of experts on every rank.
    """

    def __init__(
        self,
        module: nn.Module,
        orders: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        self.module = module
        self.layers = {
            path: layer
            for path, layer in module.named_modules()
            if isinstance(layer, FusedMoE)
        }
        device = next(
            (
                value.device
                for value in module.parameters()
                if not value.is_meta
            ),
            torch.device("cpu"),
        )
        self.tables: dict[str, torch.Tensor] = {}
        self.orders: dict[str, tuple[int, ...]] = {}
        for path, layer in self.layers.items():
            weight = layer.up_gate.weight
            table = torch.arange(
                layer.num_experts + 1,
                dtype=torch.int32,
                device=device if weight.is_meta else weight.device,
            )
            self.tables[path] = table
            self.orders[path] = tuple(range(layer.num_experts))
            layer.logical_to_physical = table
        if orders:
            # Loading validated the complete skeleton before checkpoint
            # mappings pruned modules outside this pipeline stage.
            self.commit(
                json.dumps(
                    {
                        path: ids
                        for path, ids in orders.items()
                        if path in self.layers
                    }
                )
            )

    def resolve(
        self, orders: Mapping[str, Sequence[int]]
    ) -> dict[str, tuple[int, ...]]:
        """Resolve selected layers' orders, rejecting shape-changing plans."""
        resolved = dict(self.orders)
        for path, order in orders.items():
            layer = self.layers.get(path)
            if layer is None:
                raise ValueError(f"{path!r} is not a routed expert layer")
            ids = tuple(order)
            if any(type(index) is not int for index in ids) or sorted(
                ids
            ) != list(range(layer.num_experts)):
                raise ValueError(
                    f"{path!r} requires a permutation of its experts"
                )
            resolved[path] = ids
        return resolved

    def describe(
        self,
        loaded,
        orders: str,
        host_paths: Sequence[str] = (),
        resident: Layout | None = None,
    ) -> Layout:
        """Describe requested weights without changing live routing."""
        from uniserve.loading import host_weights

        resolved = self.resolve(json.loads(orders))
        logical = {}
        for path, ids in resolved.items():
            layer = self.layers[path]
            for name, part in layer.logical_tensors(
                ids[layer.expert_slice]
            ).items():
                logical[id(layer.get_parameter(name))] = part
        return Layout.describe(
            loaded.root,
            host=frozenset(
                id(parameter)
                for path in host_paths
                for parameter in host_weights.streamed(loaded, str(path))
            ),
            origins=loaded.origins,
            logical=logical,
            resident=resident,
        )

    def select(self, orders: str) -> str:
        """Resolve the layers this rank holds from a deployment-wide order.

        The operation's coordinator checks the union of participating ranks'
        layer names, since pipeline stages and dedicated workers hold subsets.
        """
        selected = {
            path: ids
            for path, ids in json.loads(orders).items()
            if path in self.layers
        }
        return json.dumps(self.resolve(selected), sort_keys=True)

    def commit(self, orders: str) -> None:
        """Publish routing after the caller remapped every rank's weights.

        The caller must keep computations drained until all ranks complete
        this call. CUDA copies are synchronized here before they may resume.
        """
        resolved = self.resolve(json.loads(orders))
        devices = set()
        for path, ids in resolved.items():
            if ids == self.orders[path]:
                continue
            table = self.tables[path]
            # Expert transports use E as the unused-row sentinel. It keeps
            # that value through the map and contributes no expert output.
            inverse = [0] * len(ids) + [len(ids)]
            for physical, logical in enumerate(ids):
                inverse[logical] = physical
            table.copy_(
                torch.tensor(inverse, dtype=table.dtype, device=table.device)
            )
            self.layers[path].expert_ids = ids[self.layers[path].expert_slice]
            if table.is_cuda:
                devices.add(table.device)
        for device in devices:
            torch.cuda.current_stream(device).synchronize()
        self.orders = resolved


def bind(loaded) -> ExpertRouting:
    """Bind a serving rank's tables before its numerical graphs are captured."""
    if loaded.routing is None:
        loaded.routing = ExpertRouting(loaded.root)
    return loaded.routing


def select_experts(
    module: nn.Module, orders: Mapping[str, Sequence[int]]
) -> None:
    """Select checkpoint experts before loading their physical slots.

    The loader binds the corresponding routing tensors once materialization
    finishes. The supplied orders are complete global permutations; slicing
    uses the numerical partition already bound on each layer.
    """
    for path, order in orders.items():
        layer = module.get_submodule(path)
        if not isinstance(layer, FusedMoE):
            raise ValueError(f"{path!r} is not a routed expert layer")
        ids = tuple(order)
        if any(type(index) is not int for index in ids) or sorted(ids) != list(
            range(layer.num_experts)
        ):
            raise ValueError(f"{path!r} requires a permutation of its experts")
        layer.expert_ids = ids[layer.expert_slice]
