"""A kernel launched as a programmatic dependent reads what its producer wrote.

A module compiled for PDL (`launch_pdl`) may start while the grid before it on
the stream is still running: the producer's `gdc_launch_dependents` lets it
begin, and only its own `gdc_wait` holds its reads back until the producer's
writes are complete and visible. :func:`check` runs an operator behind a
producer that releases its dependent at once and writes the operator's inputs
only after a delay, so an operator that reads before it waits computes from
the inputs' previous contents. An operator's check module runs its case
through :func:`check` in a test marked `gpu` and asserts that no round
differed. Every target the Build Server describes launches dependently
(its target description's `dependent_launch`), so a device a module is built
for runs the case.

The delay makes an early read likely, not certain: a dependent that becomes
resident only after the producer wrote reads the inputs' final contents. A
correct operator never differs, so a round that differs is a defect, and the
rounds repeat the race to make a missing wait show.

A case states one operator: the inputs it reads, the buffers it writes (with
their starting contents, since some operators read them too) and a launch. The
inputs live in one byte arena that one producer grid writes, so the operator's
launch follows exactly one grid. The reference is the same launch after a
stream synchronization; the dependent launch must match it bit for bit in
every round.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
import triton
import triton.language as tl

#: The producer's delay before it writes, in nanoseconds of the global timer:
#: long enough that a dependent which does not wait reads its inputs first.
DELAY_NS = 200_000

#: The bytes one producer program copies per step.
PRODUCER_BLOCK = 1024

#: The producer's programs. They are few, so the dependent grid finds the
#: device's resources free while the producer waits: a dependent that cannot
#: become resident cannot read early either.
PRODUCER_PROGRAMS = 4

#: The alignment of every input in the arena, which any vector load accepts.
ALIGNMENT = 256

#: How many dependent launches each case runs.
ROUNDS = 20


@triton.jit
def _late_copy_kernel(
    source_ptr,
    destination_ptr,
    size,
    delay,
    BLOCK: tl.constexpr,  # noqa: N803
):
    """Release the dependent grid, wait `delay` ns, then copy `size` bytes.

    Program `p` of `n` copies the blocks `p, p + n, ...`. The wait reads the
    device's global timer, in nanoseconds (Triton's
    `tl.extra.cuda.globaltimer`).
    """
    tl.extra.cuda.gdc_launch_dependents()
    start = tl.extra.cuda.globaltimer()
    while tl.extra.cuda.globaltimer() - start < delay:
        pass
    programs = tl.num_programs(0)
    for block in range(tl.program_id(0), tl.cdiv(size, BLOCK), programs):
        offsets = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        live = offsets < size
        tl.store(
            destination_ptr + offsets,
            tl.load(source_ptr + offsets, mask=live),
            mask=live,
        )


@dataclasses.dataclass
class Case:
    """One operator's dependent launch.

    `inputs` are the tensors the operator reads, with their values; `outputs`
    the buffers it writes, with the contents they hold before the launch.
    `launch(inputs, outputs, pdl)` launches the operator once on device views
    of both, compiled and launched as a dependent when `pdl` is true, and
    answers the tensors whose contents are its result.
    """

    name: str
    inputs: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]
    launch: Callable[[dict, dict, bool], list[torch.Tensor]]


def _arena(
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One zeroed byte arena on the device and a typed view of it per input."""
    offsets, size = {}, 0
    for name, tensor in inputs.items():
        offsets[name] = size
        length = tensor.numel() * tensor.element_size()
        size += -(-length // ALIGNMENT) * ALIGNMENT
    arena = torch.zeros(size, dtype=torch.uint8, device="cuda")
    views = {}
    for name, tensor in inputs.items():
        length = tensor.numel() * tensor.element_size()
        start = offsets[name]
        typed = arena[start : start + length].view(tensor.dtype)
        views[name] = typed.view(tensor.shape)
    return arena, views


def _bytes(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """The results as bytes, so NaNs and signed zeros compare as written."""
    return [
        tensor.detach().contiguous().view(torch.uint8).clone()
        for tensor in tensors
    ]


def _equal(results: list[torch.Tensor], reference: list[torch.Tensor]) -> bool:
    return all(torch.equal(a, b) for a, b in zip(_bytes(results), reference))


def check(case: Case) -> int:
    """Run `case` behind a late producer; answer the rounds that differed.

    A case whose result on zeroed inputs equals its result on its stated
    inputs cannot tell an early read from a correct one, and is refused.
    """
    arena, views = _arena(case.inputs)
    for name, tensor in case.inputs.items():
        views[name].copy_(tensor)
    staged = arena.clone()
    outputs = {
        name: tensor.cuda().clone() for name, tensor in case.outputs.items()
    }

    def reset():
        for name, tensor in case.outputs.items():
            outputs[name].copy_(tensor)

    # The reference: the inputs are complete before the operator launches.
    reset()
    torch.cuda.synchronize()
    reference = _bytes(case.launch(views, outputs, False))

    arena.zero_()
    reset()
    torch.cuda.synchronize()
    if _equal(case.launch(views, outputs, False), reference):
        raise AssertionError(
            f"{case.name}: the result on zeroed inputs is the reference"
        )

    blocks = triton.cdiv(arena.numel(), PRODUCER_BLOCK)
    grid = (min(PRODUCER_PROGRAMS, blocks),)
    failures = 0
    for _ in range(ROUNDS):
        arena.zero_()
        reset()
        torch.cuda.synchronize()
        # The producer launches after the stream's earlier work as usual; the
        # operator right behind it is the dependent launch under test.
        _late_copy_kernel[grid](
            staged, arena, arena.numel(), DELAY_NS, BLOCK=PRODUCER_BLOCK
        )
        result = case.launch(views, outputs, True)
        torch.cuda.synchronize()
        if not _equal(result, reference):
            failures += 1
    return failures
