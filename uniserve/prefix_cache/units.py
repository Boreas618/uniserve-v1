"""Unit-pool layout and borrowed views of paged K/V state with column planes.

The pool is ``num_units`` equally sized allocation units. Every unit is
ordinary storage; a serving owner reserves unit zero as the padding sentinel
that padding rows may read, and assigns the others. Each unit holds
``columns`` columns, and each column a key plane and a value plane of
``plane_bytes`` bytes. Every plane ``(column, field)`` is one contiguous
``[num_units, plane_bytes]`` tensor, and a field's planes are stacked on a
leading column axis in one allocation.

Layers that share a history window and a K/V page shape form a group. A
group's logical page holds ``block_size = plane_bytes / row_bytes`` tokens
of every layer in the group and occupies ``units_per_page = layers /
columns`` units: the group's ``k``-th layer in configuration order reads
column ``k % columns`` of the page's ``k // columns``-th unit. A layer's key
view is therefore ``plane(column, key)`` read as a contiguous ``[num_units,
block_size, heads, dim]`` tensor whose block id is the unit id, and the
units of one page position ``k // columns`` of every page form one block
table. Groups read the same bytes differently; the unit allocator guarantees
that a unit belongs to one group at a time.

A cache whose layers all share one group has one unit per page with one
column per layer, the group's ``block_size`` equal to the pool's and one
table, so each layer's view and the stacked allocation of each field are
the ``[layers, blocks, block_size, heads, dim]`` layout of a per-layer
stack.

Layers kept per request slot (``StateConfig.indexing == "states"``) form
the pool's state group, after the K/V groups: its page ``k`` holds every
such layer's checkpoint, the state after token ``(k + 1) * block_size -
1``, where ``block_size`` is the largest K/V block size, so every page
boundary of the state group is a page boundary of every K/V group. A
page's planes are numbered in the order a host page stores them (field
major, then unit, then column: plane ``p`` is field ``p // (units_per_page
* columns)``, unit ``(p // columns) % units_per_page`` of the page, column
``p % columns``); each layer's checkpoint is a run of consecutive planes
holding its slot fields' bytes (``CheckpointLayout``), read and written
through ``uniserve.prefix_cache.checkpoints``.

A drafter's layers (``uniserve.models.Drafter``) form the pool's draft
group, after the state group: the draft cache, optional derived state a
request may lack. It keeps the pool's columns, so adding it changes no
other group's pages. A draft group of ``L`` layers, fewer than the pool's
``columns``, lays each layer over ``span = columns / L`` consecutive
columns of a unit: column ``j`` of layer ``k``'s span holds token block
``j`` of the page, so its page holds ``span`` blocks of ``plane_bytes /
row_bytes`` tokens. Layer ``k`` reads its columns as one contiguous
``[span * num_units, block, heads, dim]`` tensor whose block ``j *
num_units + u`` is token block ``j`` of unit ``u``.

The serving worker allocates the fields itself and constructs a ``UnitPool``
over them; standalone library callers allocate with ``UnitPool.allocate``.
Either way the pool only borrows its fields: unit assignment, retirement and
the lifetime of the backing belong to its owner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import gcd

import torch

from uniserve.quantization import Quantizer
from uniserve.tensors import BufferConfig

from . import mha
from .checkpoints import Checkpoints, Lookups
from .state import Config, State, StateConfig

_FIELDS = ("key", "value")


@dataclass(frozen=True, slots=True)
class CheckpointLayout:
    """Where one layer's checkpoint lies in a page of the state group.

    Attributes:
        first_plane: The first of the page planes the checkpoint occupies,
            in the page's plane order (see the module documentation).
        planes: The consecutive planes it occupies.
        fields: Each slot field the checkpoint copies, in the order the
            layer's state type declares them: its bank field name
            (``"<field>.values"``, ``"<field>.initialized"``), the byte
            offset within the run (a multiple of ``CHECKPOINT_ALIGNMENT``),
            one slot's extent and the element type.
    """

    first_plane: int
    planes: int
    fields: tuple[tuple[str, int, tuple[int, ...], torch.dtype], ...]


# Byte alignment of each field of a checkpoint, a multiple of every element
# size, so each field's bytes view as its element type in place.
CHECKPOINT_ALIGNMENT = 16


@dataclass(frozen=True, slots=True)
class CacheGroup:
    """Cache layers sharing a history window and a K/V page shape.

    The pool's state group instead holds the checkpoints of the layers kept
    per request slot.

    Attributes:
        window: History tokens any reader of the group needs; ``None`` keeps
            the whole history. ``None`` for the state group.
        block_size: Tokens stored per logical page; a power of two. The
            state group's pages are checkpoints ``block_size`` tokens apart.
        units_per_page: Units one logical page occupies.
        layers: Cache layer names in column order; layer ``k`` lies in
            column ``k % columns`` of a page's unit ``k // columns``. The
            state group's layers lie in the planes of ``checkpoints``.
        num_kv_heads: Local K/V heads stored per layer; for the state
            group, the local heads of its layers' state partition.
        head_dim: Elements per head; one for the state group, whose pages
            hold bytes rather than K/V rows.
        dtype: Stored element dtype (``float8_e4m3fn`` for FP8 storage);
            ``uint8`` for the state group.
        checkpoints: The state group's layers' checkpoints, in ``layers``
            order; empty for a K/V group.
        draft: The group holds the draft cache: a drafter's K/V, which a
            request may lack and whose page's last entry depends on the
            token after the page.
        span: Consecutive columns of a unit each layer's page spans, one
            token block of ``block_size // span`` tokens per column; one for
            every group but a draft group of fewer layers than columns.
    """

    window: int | None
    block_size: int
    units_per_page: int
    layers: tuple[str, ...]
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    checkpoints: tuple[CheckpointLayout, ...] = ()
    draft: bool = False
    span: int = 1

    @property
    def states(self) -> bool:
        """Whether the group holds checkpoints of layers kept per request
        slot rather than K/V rows.
        """  # noqa: D205
        return bool(self.checkpoints)

    @property
    def block(self) -> int:
        """Tokens of one column's plane: the block a kernel reads."""
        return self.block_size // self.span

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

    @property
    def attention(self) -> tuple[CacheGroup, ...]:
        """The K/V groups, which lead the groups."""
        return tuple(group for group in self.groups if not group.states)

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
    draft: Config | None = None,
) -> Planes:
    """Group cache layers and derive the unit pool's column planes.

    K/V layers group by history window, logical and local K/V heads, head
    width and stored element type, in order of first appearance.
    ``block_size`` is the block size of the group with the widest token
    rows; its rows fill one plane, and every other group's page holds as
    many tokens as fit that plane. Layers kept per request slot form the
    state group, last, whose pages hold their checkpoints at every multiple
    of the largest K/V block size (see the module documentation).
    ``draft`` layers, a drafter's MHA layers, form the draft group after
    them, with the pool's columns and planes (see the module
    documentation).

    Args:
        config: Cache layers by module path: MHA layers, and layers whose
            type is indexed by request slot.
        block_size: Tokens per page of the widest group; a power of two.
        dtype: Logical storage dtype, or ``None`` for each layer's compute
            dtype.
        quantization: Optional per-layer quantizer of the MHA layers; every
            layer must use the same FP8 block quantizer or none, and the
            draft layers store as the others do.
        draft: The draft cache's MHA layers, or ``None`` for no draft group.

    Raises:
        ValueError: There is no MHA layer, a layer is neither an MHA layout
            nor indexed by request slot, quantization is mixed or
            unsupported, ``block_size`` is not a positive power of two, a
            group's rows do not divide the plane into a power-of-two page,
            or the draft layers neither divide nor fill whole units.
    """
    states = {
        name: layout
        for name, layout in config.layers.items()
        if layout.indexing == "states"
    }
    layers = {
        name: layout
        for name, layout in config.layers.items()
        if name not in states
    }
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
        raise ValueError("the K/V block size must be a power of two")

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
        # Every layer was validated as an MHA layout above.
        assert isinstance(layout, mha.Config)
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
        group_block_size, remainder = divmod(plane_bytes, rows[key])
        if remainder or group_block_size & (group_block_size - 1):
            raise ValueError(
                "K/V row sizes must divide the plane into power-of-two pages"
            )
        groups.append(
            CacheGroup(
                window=window,
                block_size=group_block_size,
                units_per_page=len(names) // columns,
                layers=tuple(names),
                num_kv_heads=heads,
                head_dim=head_dim,
                dtype=stored,
            )
        )
    if states:
        groups.append(
            _state_group(
                states,
                columns=columns,
                plane_bytes=plane_bytes,
                block_size=max(group.block_size for group in groups),
            )
        )
    if draft is not None and draft.layers:
        groups.append(
            _draft_group(
                draft,
                columns=columns,
                plane_bytes=plane_bytes,
                dtype=dtype,
                quantizer=quantizer,
            )
        )
    return Planes(columns, plane_bytes, tuple(groups), quantizer is not None)


def _draft_group(
    draft: Config,
    *,
    columns: int,
    plane_bytes: int,
    dtype: torch.dtype | None,
    quantizer: Quantizer | None,
) -> CacheGroup:
    """The draft group of a drafter's MHA layers over the pool's planes.

    Its layers share one page shape. With ``L`` layers and ``columns`` a
    multiple of ``L``, each layer's page spans ``columns / L`` columns of
    one unit; with ``L`` a multiple of ``columns``, a page takes ``L /
    columns`` units of whole-column layers, as any K/V group does.

    Raises:
        ValueError: A draft layer is no MHA layout, the layers differ in
            page shape or bound their history, their rows do not divide the
            plane into a power-of-two block, their count neither divides
            nor is a multiple of the columns, or the page's tokens are no
            power of two.
    """
    shapes = set()
    for layout in draft.layers.values():
        if not isinstance(layout, mha.Config):
            raise ValueError("draft cache layers are MHA layers")
        fields = layout.buffers(
            num_blocks=1, block_size=1, dtype=dtype, quantizer=quantizer
        )
        shapes.add(
            (
                layout.window,
                layout.num_kv_heads,
                layout.head_indices,
                layout.head_dim,
                fields["key.values"].dtype,
            )
        )
    if len(shapes) != 1:
        raise ValueError("draft cache layers share one page shape")
    window, _, head_indices, head_dim, stored = next(iter(shapes))
    if window is not None:
        raise ValueError("draft cache layers read the whole history")
    row = len(head_indices) * head_dim * stored.itemsize
    block, remainder = divmod(plane_bytes, row)
    if remainder or block & (block - 1):
        raise ValueError(
            "draft K/V rows must divide the plane into power-of-two blocks"
        )
    layers = len(draft.layers)
    if columns % layers == 0:
        span, units_per_page = columns // layers, 1
    elif layers % columns == 0:
        span, units_per_page = 1, layers // columns
    else:
        raise ValueError(
            "draft cache layers must divide the pool's columns or fill whole "
            "units"
        )
    page = span * block
    if page & (page - 1):
        raise ValueError("a draft cache page must hold a power-of-two tokens")
    return CacheGroup(
        window=None,
        block_size=page,
        units_per_page=units_per_page,
        layers=tuple(draft.layers),
        num_kv_heads=len(head_indices),
        head_dim=head_dim,
        dtype=stored,
        draft=True,
        span=span,
    )


def _state_group(
    states: Mapping[str, StateConfig],
    *,
    columns: int,
    plane_bytes: int,
    block_size: int,
) -> CacheGroup:
    """The state group of the layers kept per request slot.

    Each layer's checkpoint copies every slot field its type declares, at
    ``CHECKPOINT_ALIGNMENT``-byte offsets, into a run of whole planes that
    follows the previous layer's; a page holds the units every run needs.

    Raises:
        ValueError: The layers partition their states by different heads.
    """
    from . import slots

    partitions = {
        (layout.heads, layout.state_heads) for layout in states.values()
    }
    if len(partitions) != 1:
        raise ValueError(
            "the layers kept per request slot partition their states by "
            "different heads"
        )
    _, heads = next(iter(partitions))

    checkpoints = []
    first = 0
    for layout in states.values():
        fields = []
        offset = 0
        for name, (shape, dtype) in slots.fields(layout).items():
            offset = -(-offset // CHECKPOINT_ALIGNMENT) * CHECKPOINT_ALIGNMENT
            fields.append((name, offset, shape, dtype))
            count = 1
            for extent in shape:
                count *= extent
            offset += count * torch.empty((), dtype=dtype).element_size()
        planes = max(1, -(-offset // plane_bytes))
        checkpoints.append(CheckpointLayout(first, planes, tuple(fields)))
        first += planes

    # Every unit of a page holds a key and a value plane per column.
    units_per_page = -(-first // (2 * columns))
    return CacheGroup(
        window=None,
        block_size=block_size,
        units_per_page=units_per_page,
        layers=tuple(states),
        num_kv_heads=len(heads),
        head_dim=1,
        dtype=torch.uint8,
        checkpoints=tuple(checkpoints),
    )


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
        states: Mapping[str, State] | None = None,
    ) -> None:
        """Bind every layer's state over already allocated ``fields``.

        ``configs`` gives each resident layer's MHA layout; ``allocate``
        also allocates the fields. Dense layers bind their group's stored
        dtype as their logical storage dtype; FP8 layers encode their
        compute dtype. ``states`` are the states of the layers kept per
        request slot (``uniserve.prefix_cache.slots``), which ``state``
        serves too; they have no block table, and their checkpoints in the
        pages of the state group are served by ``checkpoints``.

        Raises:
            ValueError: A field disagrees with the planes' layout.
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
        quantizer = Quantizer("fp8", axis=0) if planes.quantized else None
        self._configs = dict(configs)
        self._states: dict[str, State] = {}
        self._tables: dict[str, int] = {}
        self._checkpoints: dict[str, Checkpoints] = {}
        # The state group's layers read the same tables in a forward.
        lookups = Lookups()
        columns = planes.columns
        first_table = 0
        for group in planes.groups:
            if group.states:
                # The state group's pages hold checkpoints of the layers
                # kept per request slot; its tables address them.
                for name, layout in zip(
                    group.layers, group.checkpoints, strict=True
                ):
                    self._checkpoints[name] = Checkpoints(
                        (
                            self._fields["key.values"],
                            self._fields["value.values"],
                        ),
                        layout,
                        tables=range(
                            first_table, first_table + group.units_per_page
                        ),
                        interval=group.block_size,
                        lookups=lookups,
                    )
                first_table += group.units_per_page
                continue
            span = group.span
            for index, name in enumerate(group.layers):
                # A layer's page spans columns ``[column, column + span)``
                # of the page's unit ``row``.
                row, column = divmod(index * span, columns)
                spanned = slice(column, column + span)
                blocks = span * num_units
                views = {}
                for field in _FIELDS:
                    # The planes read as [span * units, block, heads, dim];
                    # block ``j * units + u`` is column ``column + j`` of
                    # unit ``u``: the unit id itself for a span of one.
                    views[f"{field}.values"] = (
                        self._fields[f"{field}.values"][spanned]
                        .view(group.dtype)
                        .view(
                            blocks,
                            group.block,
                            group.num_kv_heads,
                            group.head_dim,
                        )
                    )
                    views[f"{field}.initialized"] = self._fields[
                        f"{field}.initialized"
                    ][spanned].view(blocks)
                    if planes.quantized:
                        # One FP32 scale per unit of each column.
                        views[f"{field}.scale"] = self._fields[
                            f"{field}.scale"
                        ][spanned].view(blocks, 1, 1, 1)
                self._states[name] = configs[name].bind(
                    views,
                    block_size=group.block,
                    dtype=None if planes.quantized else group.dtype,
                    quantizer=quantizer,
                )
                self._tables[name] = first_table + row
            first_table += group.units_per_page
        self._states.update(states or {})

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
        request_slots: int = 0,
        draft: Config | None = None,
    ) -> UnitPool:
        """Allocate and reset a pool's fields for standalone library use.

        ``block_size``, ``dtype``, ``quantization`` and ``draft`` are
        interpreted as by ``plan_units``; ``num_units`` counts every unit,
        sentinel included.
        Scales start at one and every other field at zero. The layers kept
        per request slot get banks of ``request_slots`` slots after the
        padding slot 0, zeroed.
        """
        from . import slots

        planes = plan_units(
            config,
            block_size=block_size,
            dtype=dtype,
            quantization=quantization,
            draft=draft,
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
        states = {}
        for name, layout in config.layers.items():
            if layout.indexing == "states":
                states[name] = layout
                continue
            if not isinstance(layout, mha.Config):
                raise ValueError("a K/V unit pool requires MHA cache layers")
            configs[name] = layout
        if draft is not None:
            configs.update(draft.layers)
        banks = {
            f"{name}/{field}": torch.zeros(
                (request_slots + 1, *shape), dtype=bank_dtype, device=device
            )
            for name, layout in states.items()
            for field, (shape, bank_dtype) in slots.fields(layout).items()
        }
        return cls(
            planes,
            fields,
            configs,
            num_units=num_units,
            states=slots.bind(states, banks),
        )

    def checkpoints(self, name: str) -> Checkpoints | None:
        """The checkpoints of a layer kept per request slot in the state
        group's pages; ``None`` for a K/V layer.
        """  # noqa: D205
        self.state(name)
        return self._checkpoints.get(name)

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

    def holds(self, name: str) -> bool:
        """Whether the pool keeps cache layer ``name``: a drafter's layers
        are held only by a pool planned with the draft group.
        """  # noqa: D205
        return name in self._states

    def draft_group(self) -> int | None:
        """The draft group's index, ``None`` without one."""
        for index, group in enumerate(self.planes.groups):
            if group.draft:
                return index
        return None

    def paged(self, name: str) -> bool:
        """Whether one layer's state lies in the pool's pages (it has a
        block table) rather than in its request slots.
        """  # noqa: D205
        self.state(name)
        return name in self._tables

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

        Returns ``[columns, num_units, block, heads, dim]`` in the group's
        stored dtype for ``field`` ``"key"`` or ``"value"``: the group's
        layer ``k`` of a page's unit ``u`` is ``[k % columns, u]``, and a
        draft group's span lays a layer's token blocks over consecutive
        columns. Writes through a layer's state are visible here and vice
        versa.
        """
        values = self.planes.groups[group]
        return (
            self._fields[f"{field}.values"]
            .view(values.dtype)
            .view(
                self.planes.columns,
                self.num_units,
                values.block,
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


def stream_scope(stream: int, device: torch.device) -> Callable:
    """Return a factory of scopes enqueuing on the Rust-owned ``stream``.

    Host tensors need no stream. The external stream is wrapped once, at
    bind time; each call enters a fresh scope.
    """
    from contextlib import nullcontext

    if device.type != "cuda":
        return nullcontext
    external = torch.cuda.ExternalStream(stream, device=device)
    return lambda: torch.cuda.stream(external)


def named_fields(
    names: Sequence[str], fields: Sequence[torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The pool's unit fields by name."""
    return {str(name): field for name, field in zip(names, fields, strict=True)}


class UnitResets:
    """Scattered unit resets over Rust-owned unit fields, on one stream.

    Args:
        units: An int64 ``[num_units]`` column on the pool's device. Rust
            writes the unit ids into ``units[:count]`` before each call.
        stream: The stream handle every reset is enqueued on.
        names: The name of each of ``fields``: ``{key,value}.values``,
            ``{key,value}.initialized`` and, for FP8, ``{key,value}.scale``.
        fields: The pool's unit fields, each with a ``[columns, num_units]``
            leading layout.

    Each method takes ``count`` and is one indexed fill per field, without
    a host read.
    """

    def __init__(
        self,
        units: torch.Tensor,
        stream: int,
        names: Sequence[str],
        *fields: torch.Tensor,
    ) -> None:
        named = named_fields(names, fields)
        for name in ("key.values", "value.values"):
            if name not in named:
                raise ValueError(f"the unit fields lack {name!r}")
        self._fields = named
        self._units = units
        self._scope = stream_scope(int(stream), units.device)
        self._values = [name for name in named if name.endswith(".values")]
        self._flags = [name for name in named if name.endswith(".initialized")]
        self._scales = [name for name in named if name.endswith(".scale")]

    def _fill(self, count: int, selected: Sequence[str], value) -> None:
        index = self._units[: int(count)]
        with self._scope():
            for name in selected:
                self._fields[name].index_fill_(1, index, value)

    def zero(self, count: int) -> None:
        """Clear values and flags and reset scales to one."""
        self._fill(count, self._values + self._flags, 0)
        self._fill(count, self._scales, 1)

    def recycle(self, count: int) -> None:
        """Clear flags and reset scales.

        The reset of a quantized pool of one element type; Rust zeroes
        mixed-type pools and skips dense pools of one type.
        """
        self._fill(count, self._flags, 0)
        self._fill(count, self._scales, 1)

    def mark_initialized(self, count: int) -> None:
        """Set both fields' flags."""
        self._fill(count, self._flags, 1)
