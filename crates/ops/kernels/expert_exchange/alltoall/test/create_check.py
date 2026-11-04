#!/usr/bin/env python3
"""Check the expert exchange's lifecycle boundary: what it refuses, and how.

    python create_check.py <module_dir>

``<module_dir>`` is what ``resolve`` hands a runner: the directory holding
``module.so``. The lifecycle entries are called by the executor, not through
``ops``, so this is the contract between them, and the parts of it that need no
device are checked here on any host:

* ``create`` with a group of several ranks is refused **by name**, before it
  touches CUDA, with a message saying what the group would need. A package that
  cannot serve a topology must say so; a crash, a silent single-rank fallback or a
  half-built transport would each turn a boundary into a defect somewhere else.
* ``create`` with no positive extent, or an element type the exchange cannot
  carry, is refused by name as well.
* ``suspend``, ``restore``, ``release`` and ``info`` refuse a null transport
  rather than dereferencing it, which is what lets the executor tear down after a
  refused create without a second code path.

With a device present it also drives the whole lifecycle at one rank: create,
``info``'s four views, suspend and restore over the same addresses (that is what
keeps a captured graph valid), release. With none, it says so and runs the
refusals only.

Run it with the Program Server's environment
(``crates/program-server/python/.venv/bin/python``), which has torch and triton's
bundled CUDA; the module itself needs only ``libcudart``, which ``ldd`` names.
"""

import argparse
import ctypes
import pathlib
import sys

#: The operator crate's `DType::as_i64()` numbering, by element type name.
DTYPES = {"bf16": 4, "f16": 3, "f32": 5}


class Lifecycle(ctypes.Structure):
    """`uniserve_exchange_lifecycle`, field for field."""

    _fields_ = [
        ("transport", ctypes.c_void_p),
        ("group", ctypes.c_void_p),
        ("ranks", ctypes.c_int64),
        ("rank", ctypes.c_int64),
        ("experts", ctypes.c_int64),
        ("hidden", ctypes.c_int64),
        ("max_tokens", ctypes.c_int64),
        ("top", ctypes.c_int64),
        ("dtype", ctypes.c_int64),
        ("device", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


class Info(ctypes.Structure):
    """`uniserve_exchange_info`, field for field."""

    _fields_ = [
        ("transport", ctypes.c_void_p),
        ("states", ctypes.c_void_p),
        ("ids", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("counts", ctypes.c_void_p),
        ("ranks", ctypes.c_int64),
        ("rank", ctypes.c_int64),
        ("capacity", ctypes.c_int64),
        ("hidden", ctypes.c_int64),
        ("top", ctypes.c_int64),
        ("dtype", ctypes.c_int64),
        ("device", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


def message(call):
    """A refusal's message, as the entry left it."""
    raw = bytes(call.error)
    return raw.split(b"\0", 1)[0].decode(errors="replace")


def lifecycle(
    ranks=1, hidden=2048, top=8, max_tokens=1024, dtype="bf16", experts=128
):
    """A lifecycle call with the extents a plan would carry."""
    call = Lifecycle()
    call.ranks = ranks
    call.rank = 0
    call.experts = experts
    call.hidden = hidden
    call.max_tokens = max_tokens
    call.top = top
    call.dtype = DTYPES[dtype]
    call.device = 0
    return call


def refusals(library):
    """Every refusal that must hold without a device."""
    cases = [
        (
            "a group of two ranks",
            lifecycle(ranks=2),
            "peer-mapping transport",
        ),
        (
            "a group of eight ranks",
            lifecycle(ranks=8),
            "peer-mapping transport",
        ),
        ("no hidden extent", lifecycle(hidden=0), "positive hidden"),
        ("no top routes", lifecycle(top=0), "positive hidden"),
        ("no row budget", lifecycle(max_tokens=0), "positive hidden"),
    ]
    failures = []
    for name, call, expected in cases:
        status = library.uniserve_expert_exchange_create(
            ctypes.byref(call), None
        )
        text = message(call)
        if status == 0:
            failures.append(f"{name}: create succeeded; it must be refused")
            library.uniserve_expert_exchange_release(ctypes.byref(call), None)
            continue
        if expected and expected not in text:
            failures.append(
                f"{name}: refused with {text!r}, expected {expected!r}"
            )
        if call.transport:
            failures.append(f"{name}: a refused create left a transport behind")
        print(f"  {name}: refused — {text}")

    # An element type outside the operator's numbering is refused by the same
    # extent check: it has no row size, so it carries no rows to move. The types
    # the exchange can lay out include the packed ones (FP8 by the byte, FP4 by
    # the nibble); whether a module serves one is its descriptor's business, not
    # this entry's, so those are not refusals and are not tested as such.
    unknown = lifecycle()
    unknown.dtype = 42
    status = library.uniserve_expert_exchange_create(
        ctypes.byref(unknown), None
    )
    if status == 0:
        failures.append(
            "an element type outside the numbering: create succeeded"
        )
        library.uniserve_expert_exchange_release(ctypes.byref(unknown), None)
    else:
        print(
            f"  an element type outside the numbering: refused — {message(unknown)}"
        )

    # The four remaining entries take a transport and must refuse a null one.
    for entry in ("suspend", "restore", "release"):
        status = getattr(library, f"uniserve_expert_exchange_{entry}")(
            ctypes.byref(lifecycle()), None
        )
        if status == 0:
            failures.append(f"{entry} with a null transport returned success")
    info = Info()
    status = library.uniserve_expert_exchange_info(ctypes.byref(info), None)
    if status == 0:
        failures.append("info with a null transport returned success")
    print("  a null transport: suspend, restore, release and info all refused")
    return failures


def lifecycle_at_one_rank(library):
    """The whole lifecycle on a device: the part that needs one."""
    call = lifecycle()
    status = library.uniserve_expert_exchange_create(ctypes.byref(call), None)
    if status != 0:
        return [f"create at one rank failed: {message(call)}"]
    failures = []
    try:
        info = Info()
        info.transport = call.transport
        if library.uniserve_expert_exchange_info(ctypes.byref(info), None) != 0:
            failures.append(f"info failed: {message(info)}")
        else:
            missing = [
                name
                for name in ("states", "ids", "weights", "counts")
                if not getattr(info, name)
            ]
            if missing:
                failures.append(f"info left these views null: {missing}")
            print(
                f"  info: ranks {info.ranks}, rank {info.rank}, capacity "
                f"{info.capacity}, views at {info.states:#x}"
            )
            addresses = (info.states, info.ids, info.weights, info.counts)

        if (
            library.uniserve_expert_exchange_suspend(ctypes.byref(call), None)
            != 0
        ):
            failures.append("suspend failed")
        if (
            library.uniserve_expert_exchange_restore(ctypes.byref(call), None)
            != 0
        ):
            failures.append("restore failed")
        after = Info()
        after.transport = call.transport
        if (
            library.uniserve_expert_exchange_info(ctypes.byref(after), None)
            != 0
        ):
            failures.append("info after restore failed")
        elif (
            after.states,
            after.ids,
            after.weights,
            after.counts,
        ) != addresses:
            failures.append(
                "restore moved the views: a captured graph addresses the old ones"
            )
        else:
            print("  suspend and restore: the views kept their addresses")
    finally:
        if (
            library.uniserve_expert_exchange_release(ctypes.byref(call), None)
            != 0
        ):
            failures.append("release failed")
        else:
            print("  release: the transport is gone")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "module", type=pathlib.Path, help="the module directory"
    )
    arguments = parser.parse_args()

    library = ctypes.CDLL(str(arguments.module / "module.so"))
    for name in ("create", "suspend", "restore", "release", "info"):
        entry = getattr(library, f"uniserve_expert_exchange_{name}")
        entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        entry.restype = ctypes.c_int32

    failures = refusals(library)
    has_device = False
    try:
        import torch

        has_device = torch.cuda.is_available()
    except ImportError:
        pass
    if has_device:
        failures += lifecycle_at_one_rank(library)
    else:
        print("no GPU on this host: the lifecycle at one rank was not driven")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
