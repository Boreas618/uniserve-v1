"""Plugins: kernel providers and model packages from other packages.

A plugin package names a function in the entry point group
``uniserve.plugins``, as vLLM's ``vllm.general_plugins``. A process that
resolves a provider or a model architecture calls every such function once,
first, with the :class:`Registry`, through which the function adds:

- providers of the kernel families (``attention``, ``moe``, ``matmul``)
  under names of its own, each a factory of the family's ``Backend``;
- model packages under the architecture names checkpoints declare;
- worker hooks (``uniserve.hooks``): the operations the admin API runs on
  every worker (``load_lora_adapter`` and ``unload_lora_adapter`` of a LoRA
  plugin, ``uniserve.adapters``, or a name of its own);
- communication buffers: memory a plugin's kernels communicate through
  (symmetric memory, multicast objects), which every GPU worker allocates
  before it sizes the KV cache (:func:`allocate_buffers`);
- weight sources: the tensors of a checkpoint streamed from elsewhere than
  its files (a tensor stream, CUDA IPC handles), named by a URI of the
  plugin's scheme (:func:`weight_source`);
- a CUDA graph policy: the graph mode of a worker and the buckets each of
  its execution paths captures (:func:`graph_policy`).

Configuration then selects a plugin's provider by its name, as it selects a
built-in one (``auto`` never selects a plugin's), and a checkpoint of a
plugin's architecture loads through its package. A plugin's name may not
shadow a built-in provider or architecture.

The installed plugins' distributions and versions enter the weights'
fingerprint (:func:`versions`), since a provider declares the layout the
weights it serves are stored in: installing another version is an engine
update, which loads its weights anew.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# The entry point group plugin packages register in.
GROUP = "uniserve.plugins"
# The kernel families a plugin may add providers to.
FAMILIES = ("attention", "moe", "matmul")


@dataclass
class Registry:
    """What the installed plugins add."""

    providers: dict[str, dict[str, Callable[[], Any]]] = field(
        default_factory=lambda: {family: {} for family in FAMILIES}
    )
    models: dict[str, str] = field(default_factory=dict)
    hooks: dict[str, Callable[[Any, list[Any]], Any]] = field(
        default_factory=dict
    )
    buffers: dict[str, Callable[[list[Any], int], Any]] = field(
        default_factory=dict
    )
    weight_sources: dict[str, Callable[[str], Any]] = field(
        default_factory=dict
    )
    graph_policies: list[Callable[[list[Any], dict[str, Any]], Any]] = field(
        default_factory=list
    )

    def provider(
        self, family: str, name: str, factory: Callable[[], Any]
    ) -> None:
        """Add ``factory``, which makes a ``Backend`` of ``family``, as the
        provider ``name``.

        Raises:
            ValueError: ``family`` is not a kernel family, or a plugin
                already added a provider of that name.
        """  # noqa: D205
        if family not in self.providers:
            raise ValueError(
                f"{family!r} is not a kernel family; the families are "
                + ", ".join(FAMILIES)
            )
        if name in self.providers[family]:
            raise ValueError(f"two plugins add the {family} provider {name!r}")
        self.providers[family][name] = factory

    def model(self, architecture: str, package: str) -> None:
        """Load checkpoints that declare ``architecture`` through the model
        package ``package`` (an importable module path).

        Raises:
            ValueError: A plugin already added that architecture.
        """  # noqa: D205
        if architecture in self.models:
            raise ValueError(
                f"two plugins add the architecture {architecture!r}"
            )
        self.models[architecture] = package

    def hook(
        self, name: str, function: Callable[[Any, list[Any]], Any]
    ) -> None:
        """Implement the worker hook ``name`` with ``function``, called with
        the request's JSON payload and the worker's models
        (``uniserve.hooks.models``) and returning a JSON value.

        Raises:
            ValueError: A plugin already implements that hook, or the name
                begins with ``uniserve.``, which names the engine's own
                hooks.
        """  # noqa: D205
        if name.startswith("uniserve."):
            raise ValueError(
                f"the worker hook {name!r} is the engine's own: "
                "plugin hooks take other names"
            )
        if name in self.hooks:
            raise ValueError(f"two plugins implement the worker hook {name!r}")
        self.hooks[name] = function

    def buffer(
        self, name: str, factory: Callable[[list[Any], int], Any]
    ) -> None:
        """Allocate the communication buffer ``name`` on every GPU worker
        before it sizes the KV cache: ``factory`` is called with the
        worker's models (``uniserve.hooks.models``) and its CUDA device
        ordinal, allocates the memory and returns an object that holds it,
        which the worker keeps for its lifetime. The memory counts against
        the KV cache, memory NVML attributes to no process (a VMM
        allocation's) included.

        Raises:
            ValueError: A plugin already allocates a buffer of that name.
        """  # noqa: D205
        if name in self.buffers:
            raise ValueError(f"two plugins allocate the buffer {name!r}")
        self.buffers[name] = factory

    def weight_source(self, scheme: str, factory: Callable[[str], Any]) -> None:
        """Stream the weights a URI ``scheme://...`` names: ``factory(uri)``
        returns the source, with
        ``identity()``: what it streams, which keys the weights retained
        from it as a checkpoint's file identities key theirs;
        ``tensors(source)``: ``(name, shape, dtype)`` of every tensor of the
        checkpoint source named ``source``, named as in its files;
        ``read(source, name)``: one of those tensors, on the host or the
        device.
        The model's configuration and tokenizer stay those of its checkpoint
        directory; the tensors load through the same layout conversion as a
        checkpoint's files (``/admin/update_weights``).

        Raises:
            ValueError: A plugin already streams that scheme.
        """  # noqa: D205
        if scheme in self.weight_sources:
            raise ValueError(f"two plugins stream the weight scheme {scheme!r}")
        self.weight_sources[scheme] = factory

    def graph_policy(
        self, function: Callable[[list[Any], dict[str, Any]], Any]
    ) -> None:
        """Choose each worker's CUDA graphs: ``function`` is called with
        the worker's models and its configured policy, a dict of ``mode``
        (``auto``, ``full`` or ``off``) and the buckets of each execution
        path: ``decode`` batch sizes, ``prefill`` token sizes,
        ``flow_batches`` and ``flow_shapes`` (``[height, width]`` pixels) of
        image denoising; it returns a dict of the entries it changes. The
        worker checks the result as it checks the configuration.

        Raises:
            ValueError: Another plugin already chooses the graph policy.
        """  # noqa: D205
        if self.graph_policies:
            raise ValueError("two plugins choose the CUDA graph policy")
        self.graph_policies.append(function)


_registry: Registry | None = None


def registry() -> Registry:
    """The installed plugins' additions, registered on the first call."""
    global _registry
    if _registry is None:
        added = Registry()
        for entry in sorted(
            importlib.metadata.entry_points(group=GROUP),
            key=lambda entry: entry.name,
        ):
            entry.load()(added)
        for family, names in _built_in().items():
            shadowed = sorted(names & added.providers[family].keys())
            if shadowed:
                raise ValueError(
                    f"plugins add the built-in {family} providers "
                    + ", ".join(shadowed)
                )
        _registry = added
    return _registry


def weight_source(uri: str) -> Any:
    """The weight source a plugin streams ``uri`` (``scheme://...``) from.

    Raises:
        ValueError: ``uri`` names no scheme, or one no plugin streams.
    """
    scheme, separator, _ = str(uri).partition("://")
    if not separator:
        raise ValueError(f"the weight source {uri!r} names no scheme")
    factory = registry().weight_sources.get(scheme)
    if factory is None:
        raise ValueError(f"no plugin streams weights of the scheme {scheme!r}")
    return factory(str(uri))


def graph_policy(configured: str) -> str:
    """The CUDA graph policy of a worker whose configured policy is the
    JSON object ``configured`` (``Registry.graph_policy``), as JSON: the
    configured policy with the plugin's choices, or ``""`` without a plugin
    that chooses.
    """  # noqa: D205
    import json

    from uniserve import hooks

    policies = registry().graph_policies
    if not policies:
        return ""
    policy = json.loads(str(configured))
    chosen = policies[0](hooks.models(), dict(policy))
    unknown = sorted(set(chosen) - set(policy))
    if unknown:
        raise ValueError(
            "a plugin's graph policy chooses unknown entries "
            + ", ".join(unknown)
        )
    policy.update(chosen)
    return json.dumps(policy)


def allocate_buffers(device: int) -> list[Any]:
    """Allocate every plugin's communication buffer on CUDA device
    ``device``, in name order; the objects that hold them, which the caller
    keeps for as long as the buffers serve.
    """  # noqa: D205
    from uniserve import hooks

    models = hooks.models()
    return [
        factory(models, int(device))
        for _, factory in sorted(registry().buffers.items())
    ]


def _built_in() -> dict[str, frozenset[str]]:
    """Each kernel family's built-in provider names, ``auto`` included."""
    from uniserve.backends import attention, moe
    from uniserve.backends.matmul import base as matmul

    return {
        "attention": frozenset(("auto", *attention.PROVIDERS)),
        "moe": frozenset(("auto", *moe.PROVIDERS)),
        "matmul": frozenset(("auto", *matmul.PROVIDERS)),
    }


def provider(family: str, name: str) -> Callable[[], Any] | None:
    """The factory a plugin added as ``family``'s provider ``name``."""
    return registry().providers[family].get(name)


def versions() -> list[str]:
    """``distribution==version`` of every installed plugin, sorted."""
    found = set()
    for entry in importlib.metadata.entry_points(group=GROUP):
        distribution = entry.dist
        found.add(
            f"{distribution.name}=={distribution.version}"
            if distribution is not None
            else f"{entry.value}==unknown"
        )
    return sorted(found)
