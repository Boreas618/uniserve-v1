"""Capture what every component of a causal language model computes.

This program runs one revision of UniServe: it imports ``uniserve`` from
``sys.path``, so its caller selects the revision by placing that tree, and
its ``uniserve_kernels/src``, first on ``PYTHONPATH`` (``compare.py`` does).
It imports nothing else from the repository, so one copy of it drives any
revision whose public library interfaces it uses.

It loads a checkpoint through the public loader
(``uniserve.models.registry``), binds the model to an ``ExecutionContext``
over a ``UnitPool`` it allocates, as a Python caller of the library does, and
evaluates a fixed, seeded workload eagerly, each forward with its paged
attention batch:

- ``prefill``: a probe prompt of ``--prompt`` tokens, computed alone, alone
  again, and packed with other prompts into forwards of up to ``--tokens``
  tokens, at their front, middle and end; each request's last row is
  projected to logits.
- ``decode``: one token after a probe context of ``--context`` tokens,
  computed alone, alone again, and among up to ``--rows`` decode rows of
  other requests, whose contexts are cached beforehand; every row is
  projected to logits.

Each set of requests evaluated together is a *composition*. A forward hook
on every module records the probe's rows of each tensor the module returns,
its *leaves*: a leaf whose leading dimension is the forward's row count is
attributed row by row. ``<output>.safetensors`` holds the probe's rows of
every leaf in every composition, keyed ``<phase>/<composition>/<leaf>``;
``<output>.json`` describes the revision, the workload, every component's
declared function and identity (``uniserve.numerics``), the identity chain
of each module call, the leaves no composition attributes to rows, and the
kernels the context selected.

A module call's chain digests the identities of everything computed before
it: every module call that completed before it began, every module
registered before it along its path (whose parameters a functional call
may read), and its own. Two revisions whose chains agree for a call computed
that call's inputs with the same declared functions and formats.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

import uniserve
from uniserve import numerics
from uniserve.bindings.context import ExecutionContext
from uniserve.models import CausalLM, TextInput, TextSize, registry
from uniserve.nn.attention import AttentionBatch, PagedInput
from uniserve.prefix_cache.units import UnitPool, plan_units

# Tokens per page of the widest cache group: serving's default page
# (``kv_cache_config.block_size``, the largest supported up to 64).
PAGE = 64
# Seeds every token id and length of the workload.
SEED = 1027
# The longest companion prompt of a packed prefill composition.
COMPANION_TOKENS = 1024


@dataclasses.dataclass(frozen=True)
class Request:
    """A token sequence with pages of its own in every block table.

    ``tokens`` is ``[length]`` int64 on the host; ``blocks[t]`` lists the
    units of block table ``t`` holding its pages, in page order.
    """

    tokens: torch.Tensor
    blocks: tuple[tuple[int, ...], ...]


@dataclasses.dataclass(frozen=True)
class Rows:
    """The rows ``[start, start + count)`` of one request in a forward.

    The request's tokens before ``start`` are already in its pages.
    """

    request: Request
    start: int
    count: int


@dataclasses.dataclass(frozen=True)
class Composition:
    """Forwards whose rows of one request, the probe, are recorded.

    ``probe`` indexes the probe's entry in ``rows``.
    """

    name: str
    rows: tuple[Rows, ...]
    probe: int


class Units:
    """Hands out the pool's units: unit 0 is the padding sentinel."""

    def __init__(self, pages: tuple[int, ...]) -> None:
        # Tokens per page of each block table.
        self.pages = pages
        self.count = 1

    def request(self, tokens: torch.Tensor) -> Request:
        blocks = []
        for page in self.pages:
            needed = math.ceil(tokens.numel() / page)
            blocks.append(tuple(range(self.count, self.count + needed)))
            self.count += needed
        return Request(tokens, tuple(blocks))


@dataclasses.dataclass(frozen=True)
class Workload:
    """The fixed inputs: what is cached first, then what is recorded."""

    # Forwards that fill caches the recorded compositions read.
    setup: tuple[tuple[Rows, ...], ...]
    prefill: tuple[Composition, ...]
    decode: tuple[Composition, ...]

    def forwards(self) -> Iterator[tuple[Rows, ...]]:
        yield from self.setup
        for composition in (*self.prefill, *self.decode):
            yield composition.rows


def _split(total: int, generator: torch.Generator, limit: int) -> list[int]:
    """Seeded lengths in ``[1, limit]`` adding up to ``total``."""
    lengths = []
    while total > 0:
        length = int(torch.randint(1, limit + 1, (1,), generator=generator))
        lengths.append(min(length, total))
        total -= lengths[-1]
    return lengths


def _packed(chunks: list[tuple[Rows, ...]], rows: list[Rows], tokens: int):
    """Pack ``rows`` into forwards of at most ``tokens`` tokens each."""
    current: list[Rows] = []
    for entry in rows:
        if current and sum(r.count for r in current) + entry.count > tokens:
            chunks.append(tuple(current))
            current = []
        current.append(entry)
    if current:
        chunks.append(tuple(current))


def workload(
    vocab: int,
    pages: tuple[int, ...],
    *,
    prompt: int,
    context: int,
    tokens: int,
    rows: int,
) -> tuple[Workload, int]:
    """The seeded workload and the number of pool units it needs.

    Raises:
        ValueError: The sizes leave no room for other requests.
    """
    if prompt < 1 or context < 1 or tokens < 4 * prompt or rows < 4:
        raise ValueError(
            "the workload needs a prompt and a context, a token budget of at "
            "least four prompts and at least four decode rows"
        )
    generator = torch.Generator().manual_seed(SEED)
    units = Units(pages)

    def draw(length: int) -> torch.Tensor:
        return torch.randint(0, vocab, (length,), generator=generator)

    def companions(total: int) -> list[Rows]:
        limit = min(COMPANION_TOKENS, tokens)
        return [
            Rows(units.request(draw(length)), 0, length)
            for length in _split(total, generator, limit)
        ]

    # Prefill: the probe prompt alone, and packed into forwards of a 16th,
    # a quarter and all of the token budget, at the front, in the middle and
    # at the end. Companions are fresh prompts with pages of their own.
    probe = Rows(units.request(draw(prompt)), 0, prompt)
    prefill = [
        Composition("alone", (probe,), 0),
        Composition("repeat", (probe,), 0),
    ]
    front = companions(max(1, tokens // 16 - prompt))
    prefill.append(Composition("front", (probe, *front), 0))
    before = companions(max(1, (tokens // 4 - prompt) // 2))
    after = companions(
        max(1, tokens // 4 - prompt - sum(r.count for r in before))
    )
    prefill.append(Composition("middle", (*before, probe, *after), len(before)))
    back = companions(tokens - prompt)
    prefill.append(Composition("back", (*back, probe), len(back)))

    # Decode: one token after the probe context, among other requests'
    # decode rows. Each other request's context is drawn up to a quarter of
    # the probe's and cached, with the probe's, by the setup forwards.
    target = units.request(draw(context + 1))
    others = []
    for _ in range(rows - 1):
        length = int(
            torch.randint(
                1, max(1, context // 4) + 1, (1,), generator=generator
            )
        )
        others.append(units.request(draw(length + 1)))
    setup: list[tuple[Rows, ...]] = []
    _packed(
        setup,
        [
            Rows(request, 0, request.tokens.numel() - 1)
            for request in (target, *others)
        ],
        tokens,
    )

    def step(request: Request) -> Rows:
        return Rows(request, request.tokens.numel() - 1, 1)

    decode = [
        Composition("alone", (step(target),), 0),
        Composition("repeat", (step(target),), 0),
        Composition("pair", (step(others[0]), step(target)), 1),
    ]
    for name, count, at in (
        ("small", max(3, rows // 32), 0),
        ("medium", max(4, rows // 4), max(4, rows // 4) // 2),
        ("full", rows, rows - 1),
    ):
        batch = [step(request) for request in others[: count - 1]]
        batch.insert(at, step(target))
        decode.append(Composition(name, tuple(batch), at))
    return Workload(tuple(setup), tuple(prefill), tuple(decode)), units.count


def leaves(value, name: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    """The tensors of a module's output, named by their place in it."""
    if isinstance(value, torch.Tensor):
        yield name, value
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{name}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{name}[{key}]")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from leaves(
                getattr(value, field.name), f"{name}.{field.name}"
            )


class Recorder:
    """Forward hooks recording the probe's rows of every module's output.

    ``forward`` starts a model forward: call numbers restart and the
    record of completed calls empties. ``attribute`` names the row count of
    the calls that follow and the probe's rows among them, or ``None`` to
    record nothing.
    """

    def __init__(self, model: nn.Module) -> None:
        self.identities: dict[str, str] = {}
        self.functions: dict[str, str] = {}
        paths: dict[int, str] = {}
        for path, module in model.named_modules():
            paths[id(module)] = path
            self.functions[path] = numerics.function(type(module))
            self.identities[path] = numerics.identity(module)
        self.preceding = self._preceding(model)

        # Chains and the component of every module call, by call key.
        self.calls: dict[str, dict[str, str]] = {}
        # The probe's rows of every leaf the current composition recorded.
        self.captured: dict[str, torch.Tensor] = {}
        # Leaves whose leading dimension is not the forward's row count.
        self.unattributed: dict[str, list[int]] = {}
        self._rows: int | None = None
        self._probe = slice(0)
        self._numbers: dict[str, int] = {}
        self._completed = hashlib.sha256()
        self._active: list[str] = []
        self._hooks = []
        for module in model.modules():
            path = paths[id(module)]
            self._hooks.append(
                module.register_forward_pre_hook(self._starter(path))
            )
            self._hooks.append(
                module.register_forward_hook(self._finisher(path))
            )

    def _preceding(self, model: nn.Module) -> dict[str, bytes]:
        """For every module path, the digest of the identities of the
        modules registered before it under each of its ancestors.
        """  # noqa: D205
        result: dict[str, bytes] = {"": b""}

        def visit(module: nn.Module, path: str, inherited: bytes) -> None:
            earlier = hashlib.sha256(inherited)
            for name, child in module.named_children():
                child_path = f"{path}.{name}" if path else name
                if child_path in result:
                    continue
                result[child_path] = earlier.digest()
                visit(child, child_path, earlier.digest())
                earlier.update(self.identities[child_path].encode())

        visit(model, "", b"")
        return result

    def forward(self) -> None:
        self._numbers.clear()
        self._completed = hashlib.sha256()

    def attribute(self, rows: int | None, probe: slice = slice(0)) -> None:
        self._rows, self._probe = rows, probe

    def _starter(self, path: str):
        def start(module, args):
            number = self._numbers.get(path, 0)
            self._numbers[path] = number + 1
            key = f"{path}#{number}"
            self._active.append(key)
            if key not in self.calls:
                chain = hashlib.sha256(self._completed.digest())
                chain.update(self.preceding[path])
                chain.update(self.identities[path].encode())
                self.calls[key] = {
                    "component": path,
                    "chain": chain.hexdigest(),
                }

        return start

    def _finisher(self, path: str):
        def finish(module, args, output):
            key = self._active.pop()
            self._completed.update(self.identities[path].encode())
            if self._rows is None:
                return
            for suffix, tensor in leaves(output):
                name = key + suffix
                if tensor.ndim and tensor.shape[0] == self._rows:
                    self.captured[name] = tensor[self._probe].detach().clone()
                else:
                    self.unattributed[name] = list(tensor.shape)

        return finish

    def take(self) -> dict[str, torch.Tensor]:
        """The current composition's leaves, on the host."""
        captured, self.captured = self.captured, {}
        return {name: value.cpu() for name, value in captured.items()}


class Runner:
    """Evaluates forwards of packed rows over the model's paged cache."""

    def __init__(
        self,
        model: CausalLM,
        context: ExecutionContext,
        pages: tuple[int, ...],
        device: torch.device,
    ) -> None:
        self.model, self.context = model, context
        self.pages, self.device = pages, device

    def attention(self, rows: tuple[Rows, ...]) -> AttentionBatch:
        """One paged entry per block table, sharing the query lengths."""
        entries: dict[int, PagedInput] = {}
        for table, page in enumerate(self.pages):
            entry = PagedInput.from_blocks(
                blocks=tuple(r.request.blocks[table] for r in rows),
                query_lengths=tuple(r.count for r in rows),
                prefix_lengths=tuple(r.start for r in rows),
                block_size=page,
                causal=True,
                device=self.device,
            )
            if entries:
                entry = dataclasses.replace(
                    entry,
                    queries=entries[0].queries,
                    prefixes=entries[0].prefixes,
                )
            entries[table] = entry
        return AttentionBatch(entries, entries[0].queries)

    def __call__(
        self,
        rows: tuple[Rows, ...],
        recorder: Recorder,
        probe: int | None = None,
    ) -> None:
        """Evaluate ``rows`` and project each entry's last row to logits,
        recording entry ``probe``'s rows unless it is ``None``.
        """  # noqa: D205
        ids = torch.cat(
            [r.request.tokens[r.start : r.start + r.count] for r in rows]
        ).to(self.device)
        positions = torch.cat(
            [torch.arange(r.start, r.start + r.count) for r in rows]
        ).to(self.device)
        offsets = [0]
        for entry in rows:
            offsets.append(offsets[-1] + entry.count)
        attention = self.attention(rows)
        self.context.bind_attention(attention)

        recorder.forward()
        if probe is not None:
            recorder.attribute(
                offsets[-1], slice(offsets[probe], offsets[probe + 1])
            )
        hidden = self.model(TextInput(ids, positions, attention))

        # Logits of each entry's last row, as serving projects them.
        last = torch.tensor(offsets[1:], device=self.device) - 1
        if probe is not None:
            recorder.attribute(len(rows), slice(probe, probe + 1))
        self.model.compute_logits(hidden, token_indices=last)
        recorder.attribute(None)


def _revision(tree: Path) -> dict[str, object]:
    """The commit of the tree ``uniserve`` was imported from, if any."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(tree), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "tree": str(tree),
            "commit": git("rev-parse", "HEAD"),
            "modified": bool(
                git("status", "--porcelain", "--untracked-files=no")
            ),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"tree": str(tree), "commit": None, "modified": None}


@torch.inference_mode()
def capture(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    config = registry.read_config(args.checkpoint)
    model = registry.load_model(config, device=device).model
    if not isinstance(model, CausalLM):
        raise SystemExit(
            f"{type(model).__name__} is not a causal language model"
        )
    model.eval()

    planes = plan_units(model.cache_config, block_size=PAGE)
    pages = tuple(
        planes.groups[table.group].block_size for table in planes.tables
    )
    plan, count = workload(
        int(model.backbone.vocab_size),
        pages,
        prompt=args.prompt,
        context=args.context,
        tokens=args.tokens,
        rows=args.rows,
    )
    cache = UnitPool.allocate(
        model.cache_config, num_units=count, block_size=PAGE, device=device
    )
    recorder = Recorder(model)
    captured: dict[str, torch.Tensor] = {}
    # Every provider is the one serving selects by default for the device
    # (``auto``), whose choices the description records as ``kernels``.
    with ExecutionContext(model, cache=cache) as context:
        # The operators are prepared once, for the largest forward, as a
        # serving runner prepares them.
        forwards = list(plan.forwards())
        context.prepare(
            TextSize(
                max(sum(r.count for r in rows) for rows in forwards),
                max(len(rows) for rows in forwards),
            )
        )
        run = Runner(model, context, pages, device)
        for rows in plan.setup:
            run(rows, recorder)
        for phase, compositions in (
            ("prefill", plan.prefill),
            ("decode", plan.decode),
        ):
            for composition in compositions:
                run(composition.rows, recorder, composition.probe)
                for name, value in recorder.take().items():
                    captured[f"{phase}/{composition.name}/{name}"] = value
        kernels = context.kernels()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.contiguous() for name, value in captured.items()},
        output.with_suffix(".safetensors"),
    )
    tree = Path(uniserve.__file__).resolve().parents[1]
    description = {
        "revision": _revision(tree),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "torch": torch.__version__,
        "workload": {
            "seed": SEED,
            "page": PAGE,
            "prompt": args.prompt,
            "context": args.context,
            "tokens": args.tokens,
            "rows": args.rows,
            "compositions": {
                phase: {
                    item.name: [r.count for r in item.rows]
                    for item in compositions
                }
                for phase, compositions in (
                    ("prefill", plan.prefill),
                    ("decode", plan.decode),
                )
            },
        },
        "components": {
            path: {
                "function": recorder.functions[path],
                "identity": recorder.identities[path],
            }
            for path in recorder.functions
        },
        "calls": recorder.calls,
        "unattributed": recorder.unattributed,
        "kernels": kernels,
    }
    output.with_suffix(".json").write_text(json.dumps(description, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output", required=True, help="path without suffix of the files"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", type=int, default=64)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=512)
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
