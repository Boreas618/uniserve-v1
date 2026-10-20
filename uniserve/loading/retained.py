"""A worker's weights as the host's Storage Server retains them.

A worker keeps its loaded parameters and buffers at the Storage Server, so a
restarted worker binds them instead of reading the checkpoint. The server
stores bytes; this module defines what they are:

- :func:`fingerprint` digests everything that determines a model's loaded
  bytes besides the worker's placement and GPUs: the checkpoint's files,
  the normalized configuration, the module selection, the weight choices,
  the load mode, which kernels read the routed experts, and the numerics
  digest (:func:`numerics`) of the code that loads and converts weights.
  The worker adds its placement and GPUs to form the keys.
- :class:`Layout` describes a loaded model's device tensors for publishing:
  each parameter and buffer as its fields (a dense tensor's data, or a
  quantized tensor's encoded fields), grouped by the torch storage they
  view, and a manifest naming each tensor's representation. The worker
  allocates retained memory per storage; :meth:`Layout.publish` copies one
  storage into it and rebinds every field that viewed it, so branches that
  loading coalesced into one storage stay adjacent.
- :class:`Retained` rebuilds a worker's retained tensors as views of its
  retained storages under the manifest's representations, and installs
  them on a freshly constructed model in place of reading the checkpoint.

A tensor is named by its first path in ``named_modules(remove_duplicate=
False)`` order, which the model a worker constructs before loading repeats
exactly: ``p:<path>`` for a parameter and ``b:<path>`` for a buffer.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from uniserve.quantization import QuantizedTensor, Quantizer
from uniserve.quantization.tensor import RowOrder, ScaleLayout

# The manifest's format; a manifest of another version is not bound.
MANIFEST_VERSION = 1

# The field name of a dense tensor's data.
_DATA = "data"

# Source files of the numerical packages that determine loaded bytes.
_SOURCES = frozenset(
    (".py", ".cu", ".cuh", ".h", ".hpp", ".inl", ".cpp", ".json")
)

# Distributions whose code converts weights or whose kernels' presence
# selects the provider, and so the layout, a weight is stored in.
_LIBRARIES = (
    "torch",
    "triton",
    "flashinfer-python",
    "nvidia-cutlass-dsl",
    "safetensors",
)

# Torch allocator cache a publish may hold before returning it to the
# driver, which the retained allocations that follow draw from.
_RELEASE_BYTES = 1 << 30


def _hash(digest, text: str) -> None:
    """Add one length-prefixed text to ``digest``."""
    data = text.encode()
    digest.update(struct.pack("<Q", len(data)))
    digest.update(data)


@functools.cache
def numerics() -> str:
    """The digest of the code that loads, converts and lays out weights.

    Covers the sources of the installed ``uniserve`` and
    ``uniserve_kernels`` packages as they are on disk, an editable install
    included, and the versions of the kernel libraries in ``_LIBRARIES``.
    Computed once per process.
    """
    digest = hashlib.sha256()
    _hash(digest, "uniserve-numerics")
    for package in ("uniserve", "uniserve_kernels"):
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            _hash(digest, f"{package}: absent")
            continue
        root = Path(spec.origin).parent
        for path in sorted(root.rglob("*")):
            if (
                path.suffix not in _SOURCES
                or "__pycache__" in path.parts
                or not path.is_file()
            ):
                continue
            _hash(digest, f"{package}/{path.relative_to(root).as_posix()}")
            contents = path.read_bytes()
            digest.update(struct.pack("<Q", len(contents)))
            digest.update(contents)
    for name in _LIBRARIES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "absent"
        _hash(digest, f"{name}=={version}")
    return digest.hexdigest()


def _name(value: Any) -> str:
    """The qualified name of a class or function."""
    module = getattr(value, "__module__", "")
    return f"{module}.{getattr(value, '__qualname__', repr(value))}"


def _canonical(value: Any) -> Any:
    """A deterministic, JSON-encodable form of a configuration value.

    Sets are sorted and mappings ordered by key, so the form does not depend
    on hash randomization; a dataclass carries its class name.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__class__": _name(type(value)),
            **{
                item.name: _canonical(getattr(value, item.name))
                for item in dataclasses.fields(value)
            },
        }
    if isinstance(value, enum.Enum):
        return f"{type(value).__qualname__}.{value.name}"
    if isinstance(value, Mapping):
        items = [
            [_canonical(key), _canonical(item)] for key, item in value.items()
        ]
        return sorted(items, key=lambda pair: json.dumps(pair[0]))
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical(item) for item in value),
            key=lambda item: json.dumps(item),
        )
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, type) or callable(value):
        return _name(value)
    return repr(value)


def _header(path: Path) -> str:
    """A weight file's header.

    A safetensors file's JSON header names every tensor's dtype, shape and
    offsets; the size alone identifies any other file.
    """
    if path.suffix != ".safetensors":
        return ""
    with open(path, "rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
        return file.read(length).decode("utf-8", errors="replace")


def fingerprint(source, io, model_dtype: str, expert_kernels: str) -> str:
    """The digest of what determines a model's loaded bytes.

    The worker's placement and GPUs aside, which the worker adds.

    Args:
        source: The resolved ``registry.Config`` of this rank's modules,
            with its weight configuration.
        io: The load options; their mode (``dummy``) changes the bytes.
        model_dtype: The compute element type's name.
        expert_kernels: Which kernels read the routed experts
            (``uniserve.loading.layout.ExpertKernels``).

    Returns:
        A lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    _hash(digest, "uniserve-model-fingerprint")
    _hash(digest, numerics())
    for checkpoint in source.checkpoint:
        _hash(digest, checkpoint.name)
        _hash(digest, checkpoint.prefix)
        for path in sorted(checkpoint.files):
            _hash(digest, path.relative_to(checkpoint.root).as_posix())
            _hash(digest, str(path.stat().st_size))
            _hash(digest, _header(path))
        _hash(digest, json.dumps(_canonical(checkpoint.headers)))
    for part in (
        source.model_class,
        source.model,
        source.checkpoint_format,
        source.modules,
        source.exclude_modules,
        source.weights,
        io.mode,
        str(model_dtype),
        str(expert_kernels),
    ):
        _hash(digest, json.dumps(_canonical(part)))
    return digest.hexdigest()


@dataclass
class _Entry:
    """One parameter or buffer of a loaded model, as its fields."""

    id: str
    representation: dict[str, Any]
    fields: dict[str, torch.Tensor]


def _entry(identity: str, tensor: torch.Tensor) -> _Entry:
    if isinstance(tensor, QuantizedTensor):
        quantizer = tensor.quantizer
        representation = {
            "kind": "quantized",
            "format": quantizer.format,
            "axis": quantizer.axis,
            "calibrated_scale": quantizer.calibrated_scale,
            "scale_layout": tensor.scale_layout.value,
            "row_order": tensor.row_order.value,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        return _Entry(identity, representation, dict(tensor.buffers()))
    return _Entry(identity, {"kind": "dense"}, {_DATA: tensor})


def _tensors(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Every parameter and buffer of ``module`` with its identity, once each.

    In ``named_modules(remove_duplicate=False)`` order; an alias is named by
    its first path.
    """
    seen: set[int] = set()
    for path, child in module.named_modules(remove_duplicate=False):
        prefix = f"{path}." if path else ""
        for kind, items in (
            ("p", child._parameters.items()),
            ("b", child._buffers.items()),
        ):
            for name, tensor in items:
                if tensor is None or id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                yield f"{kind}:{prefix}{name}", tensor


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unknown dtype {name!r}")
    return value


@dataclass
class Layout:
    """A loaded model's device tensors, grouped by the torch storage they view.

    The worker reads the manifest and the flat lists below, allocates
    retained memory for each storage on its device, and calls
    :meth:`publish` once per storage, in order. Lists over fields run
    storage by storage.

    The projection matrices of a stack streamed from host memory
    (``uniserve.loading.host_weights``) are manifest entries whose data
    lies in the stack's host region instead of a device storage:
    :meth:`host_fields` lists them for each stack.
    """

    entries: list[_Entry]
    # Each storage, and the fields that view it with their entries and
    # their byte offsets in it as loaded (publishing rebinds the fields).
    _storages: list[torch.UntypedStorage] = field(default_factory=list)
    _views: list[list[tuple[_Entry, str, torch.Tensor, int]]] = field(
        default_factory=list
    )
    # Bytes freed into the torch allocator cache since it last returned
    # memory to the driver.
    _cached: int = 0
    # The entries of host-streamed matrices, by parameter identity.
    _host: dict[int, _Entry] = field(default_factory=dict)

    @classmethod
    def describe(
        cls, module: nn.Module, host: frozenset[int] = frozenset()
    ) -> Layout:
        """The layout of ``module``'s parameters and buffers.

        Meta tensors (modules this rank does not hold) are left out.
        ``host`` names the parameters (by object identity) a stack streams
        from host memory, which no device storage holds.

        Raises:
            ValueError: A tensor lies outside CUDA memory, which retention
                does not hold.
        """
        layout = cls([])
        index: dict[tuple[str, int], int] = {}
        for identity, tensor in _tensors(module):
            if tensor.is_meta:
                continue
            if tensor.device.type != "cuda":
                raise ValueError(
                    f"{identity} lies in {tensor.device.type} memory; "
                    "only CUDA weights are retained"
                )
            entry = _entry(identity, tensor)
            layout.entries.append(entry)
            if id(tensor) in host:
                layout._host[id(tensor)] = entry
                continue
            for name, value in entry.fields.items():
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr())
                if key not in index:
                    index[key] = len(layout._storages)
                    layout._storages.append(storage)
                    layout._views.append([])
                offset = value.storage_offset() * value.element_size()
                layout._views[index[key]].append((entry, name, value, offset))
        return layout

    def manifest(self) -> str:
        """Each tensor's identity, representation and field names, as JSON."""
        return json.dumps(
            {
                "version": MANIFEST_VERSION,
                "tensors": [
                    {
                        "id": entry.id,
                        "representation": entry.representation,
                        "fields": list(entry.fields),
                    }
                    for entry in self.entries
                ],
            },
            sort_keys=True,
        )

    def storage_devices(self) -> list[int]:
        """Each storage's CUDA device ordinal."""
        return [int(views[0][2].device.index or 0) for views in self._views]

    def storage_bytes(self) -> list[int]:
        """Each storage's bytes."""
        return [storage.nbytes() for storage in self._storages]

    def _fields(self) -> Iterator[tuple[int, _Entry, str, torch.Tensor]]:
        for index, views in enumerate(self._views):
            for entry, name, tensor, _ in views:
                yield index, entry, name, tensor

    def field_ids(self) -> list[str]:
        """Each field's identity, ``<tensor id>/<field>``."""
        return [f"{entry.id}/{name}" for _, entry, name, _ in self._fields()]

    def field_storages(self) -> list[int]:
        """The storage each field views."""
        return [index for index, _, _, _ in self._fields()]

    def field_offsets(self) -> list[int]:
        """Each field's byte offset in its storage as loaded."""
        return [offset for views in self._views for *_, offset in views]

    def field_dtypes(self) -> list[str]:
        """Each field's element type name."""
        return [
            str(tensor.dtype).removeprefix("torch.")
            for _, _, _, tensor in self._fields()
        ]

    def field_ranks(self) -> list[int]:
        """Each field's dimension count, which splits the flat shapes."""
        return [tensor.dim() for _, _, _, tensor in self._fields()]

    def field_shapes(self) -> list[int]:
        """Every field's shape, back to back."""
        return [
            extent
            for _, _, _, tensor in self._fields()
            for extent in tensor.shape
        ]

    def field_strides(self) -> list[int]:
        """Every field's strides in elements, back to back."""
        return [
            stride
            for _, _, _, tensor in self._fields()
            for stride in tensor.stride()
        ]

    @torch.no_grad()
    def publish(self, index: int, memory: torch.Tensor, offset: int) -> None:
        """Move storage ``index`` into ``memory`` at byte ``offset``.

        Copies the storage's bytes and rebinds every field that viewed it to
        the same place in ``memory``, a one-dimensional ``uint8`` tensor
        over a whole retained allocation on the storage's device. Each field
        keeps its tensor object, so every module that holds it reads the
        retained bytes. The loaded storage returns to the driver with the
        allocator cache once the cache holds ``_RELEASE_BYTES``, and after
        the last storage, so the next retained allocation can take its
        memory; the copies have then completed.
        """
        storage = self._storages[index]
        size = storage.nbytes()
        device = memory.device
        source = torch.empty(0, dtype=torch.uint8, device=device)
        source.set_(storage, 0, (size,), (1,))
        target = memory.untyped_storage()
        base = memory.storage_offset() + offset
        memory[offset : offset + size].copy_(source)
        for _, _, tensor, position in self._views[index]:
            element = tensor.element_size()
            start = base + position
            view = torch.empty(0, dtype=tensor.dtype, device=device)
            view.set_(
                target,
                start // element,
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
            tensor.data = view
        del source
        self._storages[index] = torch.UntypedStorage(0)
        self._cached += size
        if self._cached >= _RELEASE_BYTES or index == len(self._storages) - 1:
            torch.cuda.current_stream(device).synchronize()
            torch.cuda.empty_cache()
            self._cached = 0

    def host_fields(self, loaded, path: str) -> list[Any]:
        """The fields of the stack under ``path`` in its host region.

        Returns ``[ids, offsets, dtypes, ranks, shapes, strides]``, each a
        flat list as the ``field_*`` methods return them, with byte offsets
        in the region (``host_weights.placements``). Call before the stack
        is bound to its slots.

        Raises:
            ValueError: A streamed matrix is not a dense host-streamed entry
                of this layout.
        """
        from uniserve.loading import host_weights

        ids, offsets, dtypes, ranks, shapes, strides = [], [], [], [], [], []
        for parameter, offset in host_weights.placements(loaded, str(path)):
            entry = self._host.get(id(parameter))
            if entry is None or entry.representation["kind"] != "dense":
                raise ValueError(
                    "a streamed matrix is not a dense host entry of the layout"
                )
            ids.append(f"{entry.id}/{_DATA}")
            offsets.append(offset)
            dtypes.append(str(parameter.dtype).removeprefix("torch."))
            ranks.append(parameter.dim())
            shapes.extend(parameter.shape)
            strides.extend(parameter.stride())
        return [ids, offsets, dtypes, ranks, shapes, strides]


def describe(loaded, host_paths: Sequence[str] = ()) -> Layout:
    """The publish layout of a rank's loaded model (``rank.Loaded``).

    ``host_paths`` are the module paths whose layer stacks stream from host
    memory (``uniserve.loading.host_weights``).
    """
    from uniserve.loading import host_weights

    host = {
        id(parameter)
        for path in host_paths
        for parameter in host_weights.streamed(loaded, str(path))
    }
    return Layout.describe(loaded.root, frozenset(host))


@dataclass
class Retained:
    """A worker's retained tensors, by identity, ready to install."""

    tensors: dict[str, torch.Tensor]

    @classmethod
    def build(
        cls,
        manifest: str,
        storages: Sequence[torch.Tensor],
        field_ids: Sequence[str],
        field_storages: Sequence[int],
        field_offsets: Sequence[int],
        field_dtypes: Sequence[str],
        field_ranks: Sequence[int],
        field_shapes: Sequence[int],
        field_strides: Sequence[int],
    ) -> Retained:
        """Rebuild every tensor the manifest names.

        Args:
            manifest: The JSON :meth:`Layout.manifest` wrote.
            storages: Each retained storage as a one-dimensional ``uint8``
                tensor over the whole allocation.
            field_ids: Each field's identity, ``<tensor id>/<field>``.
            field_storages: The storage each field views.
            field_offsets: Each field's byte offset in its storage.
            field_dtypes: Each field's element type name.
            field_ranks: Each field's dimension count.
            field_shapes: Every field's shape, back to back.
            field_strides: Every field's strides in elements, back to back.

        Raises:
            ValueError: The manifest is of another version, or names a field
                without a record.
        """
        described = json.loads(str(manifest))
        if described.get("version") != MANIFEST_VERSION:
            raise ValueError("the retained manifest is of another version")

        fields: dict[str, torch.Tensor] = {}
        position = 0
        for index, identity in enumerate(field_ids):
            rank = int(field_ranks[index])
            bounds = slice(position, position + rank)
            shape = tuple(int(extent) for extent in field_shapes[bounds])
            strides = tuple(int(stride) for stride in field_strides[bounds])
            position += rank
            memory = storages[int(field_storages[index])]
            dtype = _dtype(str(field_dtypes[index]))
            element = torch.empty(0, dtype=dtype).element_size()
            start = memory.storage_offset() + int(field_offsets[index])
            view = torch.empty(0, dtype=dtype, device=memory.device)
            view.set_(
                memory.untyped_storage(), start // element, shape, strides
            )
            fields[str(identity)] = view

        tensors: dict[str, torch.Tensor] = {}
        for entry in described["tensors"]:
            identity = entry["id"]
            names = entry["fields"]
            missing = [f"{identity}/{name}" for name in names]
            missing = [name for name in missing if name not in fields]
            if missing:
                raise ValueError(f"retained fields {missing} have no record")
            parts = {name: fields[f"{identity}/{name}"] for name in names}
            representation = entry["representation"]
            if representation["kind"] == "quantized":
                quantizer = Quantizer(
                    representation["format"],
                    axis=representation["axis"],
                    calibrated_scale=representation["calibrated_scale"],
                )
                value: torch.Tensor = quantizer.from_tensors(
                    parts,
                    shape=tuple(representation["shape"]),
                    dtype=_dtype(representation["dtype"]),
                    scale_layout=ScaleLayout(representation["scale_layout"]),
                    row_order=RowOrder(representation["row_order"]),
                )
            else:
                value = parts[_DATA]
            tensors[identity] = value
        return cls(tensors)

    def install(self, model: nn.Module, selected: set[int]) -> None:
        """Install the retained tensors on ``model``.

        Every parameter and buffer of a module in ``selected`` (module object
        identities) takes its retained tensor; an alias takes the same one,
        so shared parameters stay shared. A buffer registered as ``None``
        takes the tensor retained at its path, if any. A tensor without a
        retained one keeps its placeholder, as it stayed on meta when the
        model was published. Parameters keep their placeholder's
        ``requires_grad``.
        """
        identities = {id(tensor): name for name, tensor in _tensors(model)}
        resident: dict[int, torch.Tensor] = {}
        # A module under several paths is visited again holding the tensors
        # already installed on it.
        installed: set[int] = set()
        for path, module in model.named_modules(remove_duplicate=False):
            if id(module) not in selected:
                continue
            prefix = f"{path}." if path else ""
            for name, parameter in list(module._parameters.items()):
                if parameter is None or id(parameter) in installed:
                    continue
                key = id(parameter)
                if key not in resident:
                    value = self.tensors.get(identities[key])
                    if value is None:
                        continue
                    resident[key] = nn.Parameter(
                        value, requires_grad=parameter.requires_grad
                    )
                    installed.add(id(resident[key]))
                module._parameters[name] = resident[key]
            for name, buffer in list(module._buffers.items()):
                if buffer is None:
                    value = self.tensors.get(f"b:{prefix}{name}")
                    if value is not None:
                        module._buffers[name] = value
                        installed.add(id(value))
                    continue
                key = id(buffer)
                if key in installed:
                    continue
                if key not in resident:
                    value = self.tensors.get(identities[key])
                    if value is None:
                        continue
                    resident[key] = value
                    installed.add(id(value))
                module._buffers[name] = resident[key]
