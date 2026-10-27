"""A worker's weights as the host's Storage Server retains them.

A worker keeps its loaded parameters and buffers at the Storage Server, so a
restarted worker binds them instead of reading the checkpoint. The server
stores bytes; this module defines what they are:

- :func:`fingerprints` digests what determines a model's loaded bytes
  besides the worker's placement and GPUs. The structure fingerprint covers
  the normalized configuration, the module selection, the weight choices,
  the load mode, which kernels read the routed experts, the revisions of
  how the model's weights derive from its checkpoint
  (``uniserve.numerics.loading``) and what selects the providers whose
  layouts encoded weights are stored in: the kernel libraries' versions and
  the kernel providers plugins add. A change of code that keeps those
  revisions keeps the bytes. Each checkpoint source's fingerprint adds that
  source's files. A tensor loaded from a source is keyed by its source's
  fingerprint, so a change to one source's files reloads only its tensors;
  the manifest and the tensors no source loads are keyed by the structure.
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
exactly: ``p:<path>`` for a parameter and ``b:<path>`` for a buffer. A
worker that finds only some of its tensors binds those and loads the rest
(``uniserve.loading.loader.load_model``'s ``load``).
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import importlib.metadata
import json
import re
import struct
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from uniserve import numerics, plugins
from uniserve.quantization import QuantizedTensor, Quantizer
from uniserve.quantization.tensor import RowOrder, ScaleLayout

# The manifest's format; a manifest of another version is not bound.
MANIFEST_VERSION = 4

# The field name of a dense tensor's data.
_DATA = "data"

# Distributions whose code converts weights or whose kernels' presence
# selects the provider, and so the layout, a weight is stored in.
_LIBRARIES = (
    "torch",
    "triton",
    "flashinfer-python",
    "nvidia-cutlass-dsl",
    "safetensors",
)

# How long ago a checkpoint file must have last changed for its identity to
# name its contents: longer than the kernel's timestamp resolution.
_SETTLED_NS = 1_000_000_000

# The name of a Hugging Face cache blob: the SHA-256 or git SHA-1 of its
# contents, in lowercase hexadecimal.
_BLOB_NAME = re.compile(r"[0-9a-f]{64}|[0-9a-f]{40}")

# Torch allocator cache a publish may hold before returning it to the
# driver, which the retained allocations that follow draw from.
_RELEASE_BYTES = 1 << 30


def _hash(digest, text: str) -> None:
    """Add one length-prefixed text to ``digest``."""
    data = text.encode()
    digest.update(struct.pack("<Q", len(data)))
    digest.update(data)


@functools.cache
def _providers() -> str:
    """The digest of what selects the providers whose layouts encoded
    weights are stored in: the versions of the kernel libraries in
    ``_LIBRARIES``, and each kernel provider a plugin adds
    (``uniserve.plugins``) with the version of the distribution it comes
    from. A plugin that adds no provider selects no layout. Computed once
    per process.
    """  # noqa: D205
    digest = hashlib.sha256()
    _hash(digest, "uniserve-weight-providers")
    for name in _LIBRARIES:
        _hash(digest, f"{name}=={_version(name)}")
    distributions = importlib.metadata.packages_distributions()
    for family, added in sorted(plugins.registry().providers.items()):
        for name, factory in sorted(added.items()):
            package = factory.__module__.partition(".")[0]
            versions = [
                f"{distribution}=={_version(distribution)}"
                for distribution in sorted(distributions.get(package, ()))
            ]
            _hash(digest, f"{family}/{name}: {factory.__module__} {versions}")
    return digest.hexdigest()


def _version(distribution: str) -> str:
    """The installed version of ``distribution``; ``absent`` without one."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def _encodings(source) -> set[str]:
    """The encodings of the weights ``source`` loads: those its weight
    choices and its components' precisions quantize to.
    """  # noqa: D205
    choices = [source.weights, *source.precisions.values()]
    return {
        config.weight.format
        for choice in choices
        for config in choice.quantization.values()
        if config is not None
    }


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


def _identity(path: Path) -> str:
    """The identity of the file ``path`` names, a link followed.

    A file of a Hugging Face cache is a link to a blob named by a digest of
    its contents (SHA-256 for a large file, the git blob SHA-1 otherwise),
    and the cache writes a blob once: the blob's name is its identity.

    Any other file's identity is its device, inode, modification time and
    status change time. The kernel sets a file's status change time on
    every write, and no process can set it back, so a file rewritten after
    it was read has another identity, and a file at another path is
    another file. The time has the resolution of the kernel's clock tick,
    though: a file written within the last :data:`_SETTLED_NS` may be
    written again under the same time, so its identity is unique to this
    read and its weights match no later load.
    """
    resolved = path.resolve()
    if resolved.parent.name == "blobs" and _BLOB_NAME.fullmatch(resolved.name):
        return f"blob:{resolved.name}"
    status = resolved.stat()
    if time.time_ns() - status.st_ctime_ns < _SETTLED_NS:
        return f"unsettled:{uuid.uuid4().hex}"
    return ":".join(
        str(value)
        for value in (
            status.st_dev,
            status.st_ino,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )
    )


def fingerprints(
    source,
    io,
    model_dtype: str,
    expert_kernels: str,
) -> dict[str, Any]:
    """The digests of what determines a model's loaded bytes.

    The worker's placement and GPUs aside, which the worker adds.

    Args:
        source: The resolved ``registry.Config`` of this rank's modules,
            with its weight configuration.
        io: The load options; their mode (``dummy``) changes the bytes.
        model_dtype: The compute element type's name.
        expert_kernels: Which kernels read the routed experts
            (``uniserve.loading.layout.ExpertKernels``).

    Returns:
        ``{"structure": digest, "sources": {name: digest}, "values":
        digest}``: the structure fingerprint, each checkpoint source's, and
        that of what determines the weights' values regardless of the
        layouts their bytes take, which keys what forwards compute from
        them; as lowercase hexadecimal SHA-256 digests. A source's digest
        covers each of its files' name, size, header and identity
        (:func:`_identity`).

    Raises:
        SetupError: The model's package or a quantizer it loads with
            declares no revision (``uniserve.numerics.loading``).
    """
    # What determines the weights' values: the configuration, the weight
    # choices and the revisions of how they derive from the checkpoint.
    values = hashlib.sha256()
    _hash(values, "uniserve-model-values")
    _hash(values, numerics.loading(source.model_class, _encodings(source)))
    for part in (
        source.model_class,
        source.model,
        source.checkpoint_format,
        source.modules,
        source.exclude_modules,
        source.weights,
        io.mode,
        str(model_dtype),
    ):
        _hash(values, json.dumps(_canonical(part)))
    for checkpoint in source.checkpoint:
        _hash(values, checkpoint.name)
        _hash(values, checkpoint.prefix)
    # Their bytes besides: the layouts the providers that read them select.
    structure = hashlib.sha256()
    _hash(structure, "uniserve-model-structure")
    _hash(structure, values.hexdigest())
    _hash(structure, _providers())
    _hash(structure, str(expert_kernels))
    structure_digest = structure.hexdigest()

    sources = {}
    for checkpoint in source.checkpoint:
        files = hashlib.sha256()
        _hash(files, "uniserve-checkpoint-files")
        _hash(files, checkpoint.name)
        for path in sorted(checkpoint.files):
            _hash(files, path.relative_to(checkpoint.root).as_posix())
            _hash(files, str(path.stat().st_size))
            _hash(files, _header(path))
            _hash(files, _identity(path))
        if checkpoint.stream is not None:
            # A plugin's weight source names what it streams.
            _hash(files, f"stream:{checkpoint.identity}")
        _hash(files, json.dumps(_canonical(checkpoint.headers)))
        files_digest = files.hexdigest()
        _hash(values, files_digest)
        digest = hashlib.sha256()
        _hash(digest, "uniserve-checkpoint-source")
        _hash(digest, structure_digest)
        _hash(digest, files_digest)
        sources[checkpoint.name] = digest.hexdigest()
    return {
        "structure": structure_digest,
        "sources": sources,
        "values": values.hexdigest(),
    }


@dataclass
class _Entry:
    """One parameter or buffer of a loaded model, as its fields.

    ``source`` names the checkpoint source the tensor is loaded from, empty
    for a tensor no source loads; ``host`` marks a matrix streamed from
    host memory.
    """

    id: str
    representation: dict[str, Any]
    fields: dict[str, torch.Tensor]
    source: str = ""
    host: bool = False


def tensor_entry(identity: str, tensor: torch.Tensor) -> _Entry:
    """The fields and representation of tensor ``identity``."""
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


def tensors(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
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

    The manifest names every tensor; the storages and fields cover only the
    tensors to publish, which a worker that bound the others narrows to the
    tensors it read.
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
        cls,
        module: nn.Module,
        host: frozenset[int] = frozenset(),
        origins: Mapping[int, str] | None = None,
        only: frozenset[str] | None = None,
    ) -> Layout:
        """The layout of ``module``'s parameters and buffers.

        Meta tensors (modules this rank does not hold) are left out.
        ``host`` names the parameters (by object identity) a stack streams
        from host memory, which no device storage holds. ``origins`` names
        each loaded tensor's checkpoint source by object identity
        (``uniserve.loading.loader.Result.origins``). ``only`` narrows the
        tensors to publish to those identities; ``None`` publishes all.

        Raises:
            ValueError: A tensor lies outside CUDA memory, which retention
                does not hold.
        """
        origins = {} if origins is None else origins
        layout = cls([])
        index: dict[tuple[str, int], int] = {}
        for identity, tensor in tensors(module):
            if tensor.is_meta:
                continue
            if tensor.device.type != "cuda":
                raise ValueError(
                    f"{identity} lies in {tensor.device.type} memory; "
                    "only CUDA weights are retained"
                )
            entry = tensor_entry(identity, tensor)
            entry.source = origins.get(id(tensor), "")
            entry.host = id(tensor) in host
            layout.entries.append(entry)
            if only is not None and identity not in only:
                continue
            if entry.host:
                layout._host[id(tensor)] = entry
                continue
            for name, value in entry.fields.items():
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr())
                if key not in index:
                    index[key] = len(layout._storages)
                    layout._storages.append(storage)
                    layout._views.append([])
                offset = int(value.storage_offset()) * value.element_size()
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
                        "source": entry.source,
                        "host": entry.host,
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

    def field_sources(self) -> list[str]:
        """The checkpoint source of each field's tensor; empty for none."""
        return [entry.source for _, entry, _, _ in self._fields()]

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

        Returns ``[ids, offsets, dtypes, ranks, shapes, strides, sources]``,
        each a flat list as the ``field_*`` methods return them, with byte
        offsets in the region (``host_weights.placements``). Call before the
        stack is bound to its slots.

        Raises:
            ValueError: A streamed matrix is not a dense host-streamed entry
                of this layout.
        """
        from uniserve.loading import host_weights

        ids: list[str] = []
        offsets: list[int] = []
        dtypes: list[str] = []
        ranks: list[int] = []
        shapes: list[int] = []
        strides: list[int] = []
        sources: list[str] = []
        for parameter, offset in host_weights.placements(loaded, str(path)):
            entry = self._host.get(id(parameter))
            if entry is None or entry.representation["kind"] != "dense":
                raise ValueError(
                    "a streamed matrix is not a dense host entry of the layout"
                )
            ids.append(f"{entry.id}/{_DATA}")
            sources.append(entry.source)
            offsets.append(offset)
            dtypes.append(str(parameter.dtype).removeprefix("torch."))
            ranks.append(parameter.dim())
            shapes.extend(parameter.shape)
            strides.extend(parameter.stride())
        return [ids, offsets, dtypes, ranks, shapes, strides, sources]


def describe(
    loaded, host_paths: Sequence[str] = (), only: Sequence[str] | None = None
) -> Layout:
    """The publish layout of a rank's loaded model (``rank.Loaded``).

    ``host_paths`` are the module paths whose layer stacks stream from host
    memory (``uniserve.loading.host_weights``). ``only`` names the tensors
    to publish, those the rank read while it bound the others; ``None``
    publishes all.
    """
    from uniserve.loading import host_weights

    host = {
        id(parameter)
        for path in host_paths
        for parameter in host_weights.streamed(loaded, str(path))
    }
    return Layout.describe(
        loaded.root,
        frozenset(host),
        loaded.origins,
        None if only is None else frozenset(str(name) for name in only),
    )


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
        load: Sequence[str],
        field_storages: Sequence[int],
        field_offsets: Sequence[int],
        field_dtypes: Sequence[str],
        field_ranks: Sequence[int],
        field_shapes: Sequence[int],
        field_strides: Sequence[int],
    ) -> Retained:
        """Rebuild every tensor the manifest names but those in ``load``.

        Args:
            manifest: The JSON :meth:`Layout.manifest` wrote.
            storages: Each retained storage as a one-dimensional ``uint8``
                tensor over the whole allocation.
            field_ids: Each field's identity, ``<tensor id>/<field>``.
            load: The tensors the worker reads from the checkpoint instead,
                whose fields are not among ``field_ids``.
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
        loaded = frozenset(str(name) for name in load)
        for entry in described["tensors"]:
            identity = entry["id"]
            if identity in loaded:
                continue
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

    def install(
        self,
        model: nn.Module,
        selected: set[int],
        skip: frozenset[str] = frozenset(),
    ) -> None:
        """Install the retained tensors on ``model``.

        Every parameter and buffer of a module in ``selected`` (module object
        identities) takes its retained tensor, but those ``skip`` names
        (identities), which loading reads; an alias takes the same one, so
        shared parameters stay shared. A buffer registered as ``None`` takes
        the tensor retained at its path, if any. A tensor without a retained
        one keeps its placeholder, as it stayed on meta when the model was
        published. Parameters keep their placeholder's ``requires_grad``.
        """
        identities = {id(tensor): name for name, tensor in tensors(model)}
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
                    if identities[key] in skip:
                        continue
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
                    if f"b:{prefix}{name}" in skip:
                        continue
                    value = self.tensors.get(f"b:{prefix}{name}")
                    if value is not None:
                        module._buffers[name] = value
                        installed.add(id(value))
                    continue
                key = id(buffer)
                if key in installed:
                    continue
                if key not in resident:
                    if identities[key] in skip:
                        continue
                    value = self.tensors.get(identities[key])
                    if value is None:
                        continue
                    resident[key] = value
                    installed.add(id(value))
                module._buffers[name] = resident[key]
