"""Numerical manifests over distributed expert owners' resident shards."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence

import torch

from uniserve.loading.retained import Layout, field_parts, restore_tensor
from uniserve.nn.moe import FusedMoE
from uniserve.quantization import QuantizedTensor


class ResidentExperts:
    """Retain numerical descriptions over the owner's borrowed shard views.

    Capture after kernel representation preparation and before full composite
    views replace local parameters. ``bind`` then replaces every distributed
    field's backing with its actual resident allocation. No model parameter
    or execution buffer is changed by describing another expert order.
    """

    def __init__(self, loaded) -> None:
        self.layers = [
            (path, layer, layer.expert_slice)
            for path, layer in loaded.module.named_modules()
            if isinstance(layer, FusedMoE)
            and layer.expert_group.size > 1
            and not layer.up_gate.weight.is_meta
        ]
        self.layouts = []
        self.logical = []
        self.names: dict[int, tuple[str, ...]] = {}
        for path, layer, _ in self.layers:
            self.logical.append(layer.logical_tensors())
            layout = Layout.describe(layer, origins=loaded.origins)
            for entry in layout.entries:
                kind, name = entry.id.split(":", 1)
                entry.id = f"{kind}:{path}.{name}" if path else entry.id
            self.layouts.append(layout)

    def bind(
        self,
        layer: int,
        names: Sequence[str],
        offsets: Sequence[int],
        *fields: torch.Tensor,
    ) -> None:
        """Bind one layer's fields to the owner's resident byte views."""
        path = self.layers[int(layer)][0]
        layout = self.layouts[int(layer)]
        values = {
            f"{entry.id}/{name}": value
            for entry in layout.entries
            for name, value in entry.fields.items()
        }
        replacements = {}
        for name, offset, storage in zip(names, offsets, fields, strict=True):
            projection, field = str(name).split(".", 1)
            field = "data" if field == "weight" else field
            prefix = f"{path}." if path else ""
            identity = f"p:{prefix}{projection}.weight/{field}"
            original = values[identity]
            size = original.numel() * original.element_size()
            replacements[identity] = (
                storage[int(offset) : int(offset) + size]
                .view(original.dtype)
                .view(original.shape)
            )
        self.layouts[int(layer)] = layout.replace_fields(replacements)
        self.names[int(layer)] = tuple(str(name) for name in names)

    def fields(self, retained) -> list:
        """Borrow transferred shards in this owner's collective field order."""
        counts, names, fields = [], [], []
        for index, (path, _, _) in enumerate(self.layers):
            selected = self.names[index]
            counts.append(len(selected))
            for name in selected:
                projection, field = name.split(".", 1)
                prefix = f"{path}." if path else ""
                value = retained.tensors[f"p:{prefix}{projection}.weight"]
                names.append(name)
                fields.append(
                    value.buffers()[field]
                    if isinstance(value, QuantizedTensor)
                    else value
                )
        return [counts, names, fields]

    def describe(self, orders: str) -> Layout:
        """Describe resident shards in the requested global expert order."""
        orders = json.loads(orders)
        result = Layout([])
        for (path, layer, interval), layout, original in zip(
            self.layers, self.layouts, self.logical, strict=True
        ):
            logical = (
                layer.logical_tensors(orders[path][interval])
                if path in orders
                else original
            )
            entries = []
            for entry in layout.entries:
                prefix = f"p:{path}." if path else "p:"
                name = entry.id.removeprefix(prefix)
                value = restore_tensor(entry.representation, entry.fields)
                entries.append(
                    dataclasses.replace(
                        entry,
                        parts=field_parts(entry, value, logical.get(name)),
                    )
                )
            result = result.merge(Layout(entries))
        return result
