"""Unit-pool layout and borrowed views of paged K/V state with column planes.

The pool is ``num_units`` equally sized allocation units. Every unit is
ordinary memory; a serving owner reserves unit zero as the padding sentinel
that padding rows may read, and assigns the others. Each unit holds
``columns`` columns, and each column a key plane and a value plane of
``plane_bytes`` bytes. Every plane ``(column, field)`` is one contiguous
``[num_units, plane_bytes]`` tensor, and a field's planes are stacked on a
leading column axis in one allocation.

Layers that share a history window and a K/V page shape form a group. A
group's logical page holds ``page_tokens = plane_bytes / row_bytes`` tokens
of every layer in the group and occupies ``units_per_page = layers /
columns`` units: the group's ``k``-th layer in configuration order reads
column ``k % columns`` of the page's ``k // columns``-th unit. A layer's key
view is therefore ``plane(column, key)`` read as a contiguous ``[num_units,
page_tokens, heads, dim]`` tensor whose block id is the unit id, and the
units of one page position ``k // columns`` of every page form one block
table. Groups read the same bytes differently; the unit allocator guarantees
that a unit belongs to one group at a time.

A cache whose layers all share one group has one unit per page with one
column per layer, ``page_tokens`` equal to ``block_size`` and one table, so
each layer's view and the stacked allocation of each field are the
``[layers, blocks, block_size, heads, dim]`` layout of a per-layer stack.

The serving worker allocates the fields itself and binds a ``UnitPool`` over
them with ``UnitPool.from_plan``; standalone library callers allocate with
``UnitPool.allocate``. Either way the pool only borrows its fields: unit
assignment, retirement and the lifetime of the backing belong to its owner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import gcd

import torch

from uniserve_models.quantization import Quantizer
from uniserve_models.tensors import BufferConfig

from . import mha
from .state import Config, State

_FIELDS = ("key", "value")


def async_tensor_h2d(
    values: Sequence[int], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Copy host integers into a new tensor on ``device`` without a host wait.

    On CUDA the values are written to pinned memory from PyTorch's caching
    host allocator and copied non-blocking on the current stream. The
    allocator reuses that pinned block only after the copy completes, so the
    calling thread never waits for device work queued ahead of the copy, and
    the result is ordered before every later operation on the stream. Other
    devices receive an ordinary copy.
    """
    pinned = device.type == "cuda"
    host = torch.tensor(values, dtype=dtype, pin_memory=pinned)
    return host.to(device, non_blocking=pinned)


@dataclass(frozen=True, slots=True)
class CacheGroup:
    """Cache layers that share a history window and a K/V page shape.

    Attributes:
        window: History tokens any reader of the group needs; ``None`` keeps
            the whole history.
        page_tokens: Tokens stored per logical page; a power of two.
        units_per_page: Units one logical page occupies.
        layers: Cache layer names in column order; layer ``k`` lies in
            column ``k % columns`` of a page's unit ``k // columns``.
        num_kv_heads: Local K/V heads stored per layer.
        head_dim: Elements per head.
        dtype: Stored element dtype (``float8_e4m3fn`` for FP8 storage).
    """

    window: int | None
    page_tokens: int
    units_per_page: int
    layers: tuple[str, ...]
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    @property
    def row_bytes(self) -> int:
        """Bytes one token of one layer's key (or value) occupies."""
        return (
            self.num_kv_heads
            * self.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )


@dataclass(frozen=True, slots=True)
class CacheTable:
    """One block table: the units at one position of every page of a group.

    ``row`` indexes a unit within each logical page of group ``group``; every
    layer whose column lies in that unit reads through this table.
    """

    group: int
    row: int


@dataclass(frozen=True, slots=True)
class Planes:
    """The unit and column-plane organization of one K/V unit pool.

    Attributes:
        columns: Columns per unit, the greatest common divisor of the
            groups' layer counts.
        plane_bytes: Bytes of one ``(column, field)`` plane of one unit.
        groups: Cache groups in table order.
        quantized: Whether the planes hold FP8 values with one FP32 scale
            per unit, column and field.
    """

    columns: int
    plane_bytes: int
    groups: tuple[CacheGroup, ...]
    quantized: bool

    @property
    def unit_bytes(self) -> int:
        """Bytes one unit occupies: its planes, flags and FP8 scales."""
        metadata = 1 + (4 if self.quantized else 0)
        return self.columns * len(_FIELDS) * (self.plane_bytes + metadata)

    @property
    def tables(self) -> tuple[CacheTable, ...]:
        """Block tables in table-id order: every group's rows, group-major."""
        return tuple(
            CacheTable(group, row)
            for group, value in enumerate(self.groups)
            for row in range(value.units_per_page)
        )

    def buffers(self, num_units: int) -> dict[str, BufferConfig]:
        """Declare every backing field the pool allocates for ``num_units``.

        Values are raw bytes that each group reads in its own element type;
        initialization flags and FP8 scales hold one entry per unit, column
        and field.
        """
        if type(num_units) is not int or num_units < 1:
            raise ValueError("a K/V unit pool requires at least one unit")
        result = {}
        for field in _FIELDS:
            result[f"{field}.values"] = BufferConfig(
                (self.columns, num_units, self.plane_bytes), torch.uint8
            )
            if self.quantized:
                result[f"{field}.scale"] = BufferConfig(
                    (self.columns, num_units), torch.float32
                )
            result[f"{field}.initialized"] = BufferConfig(
                (self.columns, num_units), torch.bool
            )
        return result


def plan_units(
    config: Config,
    *,
    block_size: int,
    dtype: torch.dtype | None = None,
    quantization: Mapping[str, Quantizer | None] | None = None,
) -> Planes:
    """Group cache layers and derive the unit pool's column planes.

    Layers group by history window, logical and local K/V heads, head width
    and stored element type, in order of first appearance. ``block_size`` is
    the page size of the group with the widest token rows; its rows fill one
    plane, and every other group's page holds as many tokens as fit that
    plane.

    Args:
        config: MHA cache layers by module path.
        block_size: Tokens per page of the widest group; a power of two.
        dtype: Logical storage dtype, or ``None`` for each layer's compute
            dtype.
        quantization: Optional per-layer quantizer; every layer must use the
            same FP8 block quantizer or none.

    Raises:
        ValueError: A layer is not an MHA layout, quantization is mixed or
            unsupported, ``block_size`` is not a positive power of two, or a
            group's rows do not divide the plane into a power-of-two page.
    """
    layers = config.layers
    if not layers or any(
        not isinstance(layout, mha.Config) for layout in layers.values()
    ):
        raise ValueError("a K/V unit pool requires MHA cache layers")
    if quantization is not None and set(quantization) - set(layers):
        raise ValueError(
            "cache quantization names must identify resident state layers"
        )
    if (
        type(block_size) is not int
        or block_size < 1
        or block_size & (block_size - 1)
    ):
        raise ValueError("the K/V page size must be a power of two")

    quantizers = {
        name: None if quantization is None else quantization.get(name)
        for name in layers
    }
    if len(set(quantizers.values())) != 1:
        raise ValueError("every K/V layer must share one storage quantizer")
    quantizer = next(iter(quantizers.values()))

    # Group layers in configuration order by retention and page shape. The
    # layout validates each layer's dtype and quantizer through its buffers.
    members: dict[tuple, list[str]] = {}
    for name, layout in layers.items():
        fields = layout.buffers(
            num_blocks=1, block_size=1, dtype=dtype, quantizer=quantizer
        )
        stored = fields["key.values"].dtype
        key = (
            layout.window,
            layout.num_kv_heads,
            layout.head_indices,
            layout.head_dim,
            stored,
        )
        members.setdefault(key, []).append(name)

    columns = 0
    for names in members.values():
        columns = gcd(columns, len(names))
    rows = {
        key: len(key[2]) * key[3] * torch.empty((), dtype=key[4]).element_size()
        for key in members
    }
    plane_bytes = block_size * max(rows.values())

    groups = []
    for key, names in members.items():
        window, _, head_indices, head_dim, stored = key
        heads = len(head_indices)
        page_tokens, remainder = divmod(plane_bytes, rows[key])
        if remainder or page_tokens & (page_tokens - 1):
            raise ValueError(
                "K/V row sizes must divide the plane into power-of-two pages"
            )
        groups.append(
            CacheGroup(
                window=window,
                page_tokens=page_tokens,
                units_per_page=len(names) // columns,
                layers=tuple(names),
                num_kv_heads=heads,
                head_dim=head_dim,
                dtype=stored,
            )
        )
    return Planes(columns, plane_bytes, tuple(groups), quantizer is not None)


_STORAGE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


def _dtype(name: str) -> torch.dtype:
    """Resolve a storage dtype name (a torch name without its prefix)."""
    try:
        return _STORAGE_DTYPES[name.removeprefix("torch.")]
    except KeyError:
        raise ValueError(f"unsupported K/V storage dtype {name!r}") from None


class UnitPool:
    """Borrowed views of a K/V unit pool's fields and its layers' states.

    The fields are ``{key,value}.values`` uint8 ``[columns, num_units,
    plane_bytes]``, ``{key,value}.initialized`` bool ``[columns,
    num_units]`` and, for FP8 storage, ``{key,value}.scale`` float32
    ``[columns, num_units]``. No layer, attention method or numerical input
    receives the pool itself: layers borrow their states, and block tables
    name units by id.
    """

    def __init__(
        self,
        planes: Planes,
        fields: Mapping[str, torch.Tensor],
        configs: Mapping[str, mha.Config],
        *,
        num_units: int,
    ) -> None:
        """Bind every layer's state over already allocated ``fields``.

        Use ``from_plan`` or ``allocate``. ``configs`` gives each resident
        layer's MHA layout. Dense layers bind their group's stored dtype as
        their logical storage dtype; FP8 layers encode their compute dtype.
        """
        expected = planes.buffers(num_units)
        for name, requirement in expected.items():
            tensor = fields.get(name)
            if (
                tensor is None
                or tuple(tensor.shape) != requirement.shape
                or tensor.dtype != requirement.dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"K/V unit field {name!r} disagrees with its layout"
                )
        self.planes = planes
        self.num_units = num_units
        self._fields = {name: fields[name] for name in expected}
        self.device = self._fields["key.values"].device
        # Groups storing different element types reinterpret each other's
        # stale bytes, which need not decode as finite (`recycle_units`).
        self._mixed_types = len({group.dtype for group in planes.groups}) > 1

        quantizer = Quantizer("fp8", axis=0) if planes.quantized else None
        self._configs = dict(configs)
        self._states: dict[str, State] = {}
        self._tables: dict[str, int] = {}
        columns = planes.columns
        first_table = 0
        for group in planes.groups:
            for index, name in enumerate(group.layers):
                row, column = divmod(index, columns)
                views = {}
                for field in _FIELDS:
                    # One plane read as [units, page tokens, heads, dim]; the
                    # block id of the view is the unit id.
                    views[f"{field}.values"] = (
                        self._fields[f"{field}.values"][column]
                        .view(group.dtype)
                        .view(
                            num_units,
                            group.page_tokens,
                            group.num_kv_heads,
                            group.head_dim,
                        )
                    )
                    views[f"{field}.initialized"] = self._fields[
                        f"{field}.initialized"
                    ][column]
                    if planes.quantized:
                        # One FP32 scale per unit of this column.
                        views[f"{field}.scale"] = self._fields[
                            f"{field}.scale"
                        ][column].view(num_units, 1, 1, 1)
                self._states[name] = configs[name].bind(
                    views,
                    block_size=group.page_tokens,
                    dtype=None if planes.quantized else group.dtype,
                    quantizer=quantizer,
                )
                self._tables[name] = first_table + row
            first_table += group.units_per_page

    @classmethod
    def from_plan(
        cls, plan: Mapping, tensors: Mapping[str, torch.Tensor]
    ) -> UnitPool:
        """Bind a pool over fields its owner allocated for ``plan``.

        Args:
            plan: The cache description ``interface.cache.cache_plan``
                returns, plus ``"num_units"``, the units the fields hold
                (sentinel included).
            tensors: The unit fields by name; other entries, such as block
                tables, are ignored. Nothing is allocated or initialized.

        Raises:
            ValueError: A field disagrees with the plan's layout.
        """
        quantized = bool(plan["quantized"])
        groups = []
        configs = {}
        for value in plan["groups"]:
            window = int(value["window"])
            offset = int(value["kv_head_offset"])
            heads = int(value["num_kv_heads"])
            layers = tuple(str(name) for name in value["layers"])
            groups.append(
                CacheGroup(
                    window=None if window < 0 else window,
                    page_tokens=int(value["page_tokens"]),
                    units_per_page=int(value["units_per_page"]),
                    layers=layers,
                    num_kv_heads=heads,
                    head_dim=int(value["head_dim"]),
                    dtype=_dtype(str(value["dtype"])),
                )
            )
            config = mha.Config(
                int(value["total_kv_heads"]),
                int(value["head_dim"]),
                tuple(range(offset, offset + heads)),
                _dtype(str(value["compute_dtype"])),
                window=None if window < 0 else window,
            )
            configs.update(dict.fromkeys(layers, config))
        planes = Planes(
            int(plan["columns"]),
            int(plan["plane_bytes"]),
            tuple(groups),
            quantized,
        )
        if planes.unit_bytes != int(plan["unit_bytes"]):
            raise ValueError("the cache plan's unit bytes disagree with it")
        return cls(planes, tensors, configs, num_units=int(plan["num_units"]))

    @classmethod
    def allocate(
        cls,
        config: Config,
        *,
        num_units: int,
        block_size: int,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        quantization: Mapping[str, Quantizer | None] | None = None,
    ) -> UnitPool:
        """Allocate and reset a pool's fields for standalone library use.

        ``block_size``, ``dtype`` and ``quantization`` are interpreted as by
        ``plan_units``; ``num_units`` counts every unit, sentinel included.
        Scales start at one and every other field at zero.
        """
        planes = plan_units(
            config,
            block_size=block_size,
            dtype=dtype,
            quantization=quantization,
        )
        fields = {
            # Multiplicative scales restart at one; encoded bytes and flags
            # restart at zero.
            name: (torch.ones if name.endswith(".scale") else torch.zeros)(
                requirement.shape, dtype=requirement.dtype, device=device
            )
            for name, requirement in planes.buffers(num_units).items()
        }
        configs = {}
        for name, layout in config.layers.items():
            if not isinstance(layout, mha.Config):
                raise ValueError("a K/V unit pool requires MHA cache layers")
            configs[name] = layout
        return cls(planes, fields, configs, num_units=num_units)

    @property
    def groups(self) -> tuple[CacheGroup, ...]:
        """Cache groups in table order."""
        return self.planes.groups

    @property
    def tables(self) -> tuple[CacheTable, ...]:
        """Block tables in table-id order."""
        return self.planes.tables

    def state(self, name: str) -> State:
        """Borrow one layer's state; the fields' owner outlives its use."""
        return self._states[name]

    def table(self, name: str) -> int:
        """Return the ID of the block table addressing one layer's units.

        The table holds the units at the layer's position within every page
        of its group (``tables``). ``AttentionBatch`` entries are keyed by
        these IDs.
        """
        self.state(name)
        return self._tables[name]

    def head_offset(self, name: str) -> int:
        """Return one layer's first global K/V head; local heads follow it.

        Raises:
            ValueError: The layer's local heads are not consecutive.
        """
        heads = self._configs[name].head_indices
        if heads != tuple(range(heads[0], heads[0] + len(heads))):
            raise ValueError("the layer's local K/V heads are not consecutive")
        return heads[0]

    def planes_of(self, group: int, field: str) -> torch.Tensor:
        """Borrow one field's planes as one group reads them.

        Returns ``[columns, num_units, page_tokens, heads, dim]`` in the
        group's stored dtype for ``field`` ``"key"`` or ``"value"``: the
        group's layer ``k`` of a page's unit ``u`` is ``[k % columns, u]``.
        Writes through a layer's state are visible here and vice versa.
        """
        values = self.planes.groups[group]
        return (
            self._fields[f"{field}.values"]
            .view(values.dtype)
            .view(
                self.planes.columns,
                self.num_units,
                values.page_tokens,
                values.num_kv_heads,
                values.head_dim,
            )
        )

    def scales_of(self, field: str) -> torch.Tensor:
        """Borrow one field's FP8 scales as ``[columns, num_units]``.

        Raises:
            ValueError: The pool stores unquantized values.
        """
        if not self.planes.quantized:
            raise ValueError("unquantized K/V planes carry no scales")
        return self._fields[f"{field}.scale"]

    def _index(self, units: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Return unit ids as an int64 index on the pool's device.

        A tensor index is used as given: the serving owner validated its
        units, and reading them back would synchronize the device.
        """
        if isinstance(units, torch.Tensor):
            if units.dtype != torch.int64 or units.device != self.device:
                raise ValueError(
                    "unit indices must be int64 on the pool's device"
                )
            return units
        if any(
            type(unit) is not int or not 0 <= unit < self.num_units
            for unit in units
        ):
            raise ValueError("unit ids must lie within the pool")
        return torch.tensor(tuple(units), dtype=torch.int64).to(
            self.device, non_blocking=True
        )

    def zero_units(self, units: torch.Tensor | Sequence[int]) -> None:
        """Reset the selected units in every column and field.

        Resets the units' bytes, initialization flags and FP8 scales, so a
        unit leaving one group can join any other. Each field resets with
        one indexed fill over its unit axis, however the units are
        scattered, enqueued on the current stream without a host read.
        Repeated ids are allowed.
        """
        index = self._index(units)
        if not index.numel():
            return
        for name, tensor in self._fields.items():
            # Every field is laid out [columns, num_units, ...].
            tensor.index_fill_(1, index, int(name.endswith(".scale")))

    def recycle_units(self, units: torch.Tensor | Sequence[int]) -> None:
        """Prepare units that held earlier tokens for a new owner.

        Readers never weigh tokens a unit's owner has not written: every
        attention kernel replaces the scores of unwritten positions, so
        their values contribute exactly zero while they are finite. The pool
        starts zeroed and writers store finite values, so when every group
        stores one element type, stale bytes decode as finite values in
        every group's view and stay in place. What a writer reads is reset:
        the FP8 scales and initialization flags that decide whether a write
        grows a block's scale. Unquantized writes only set flags, so an
        unquantized pool of one element type needs no device work. A pool
        whose groups store different element types is reset completely
        (``zero_units``), since stale bytes of one type need not decode as
        finite values of another.
        """
        if self._mixed_types:
            self.zero_units(units)
            return
        if not self.planes.quantized:
            return
        self.reset_flags(units)

    def reset_flags(self, units: torch.Tensor | Sequence[int]) -> None:
        """Clear the selected units' initialization flags and FP8 scales.

        This is the device work of ``recycle_units`` for a quantized pool of
        one element type.
        """
        index = self._index(units)
        if not index.numel():
            return
        for field in _FIELDS:
            self._fields[f"{field}.initialized"].index_fill_(1, index, False)
            if self.planes.quantized:
                self._fields[f"{field}.scale"].index_fill_(1, index, 1)

    def mark_initialized(self, units: torch.Tensor | Sequence[int]) -> None:
        """Commit externally transferred values and scales of whole units.

        Every column of a unit holds a layer of the unit's group, so the
        flags of both fields in every column are set with one indexed fill
        per field, without a host read.
        """
        index = self._index(units)
        if not index.numel():
            return
        for field in _FIELDS:
            self._fields[f"{field}.initialized"].index_fill_(1, index, True)
