#!/usr/bin/env python3
"""Checks steps of the Rust buffer pool against the design model's steps.

    steps.py [--jobs J] [--batch B] [--expect fail] STEPS WORK

STEPS is what the enumeration of L1 (`agreement_steps.sh`) or the replay of
a trace (`replay.sh`) wrote: layouts, the pool's states in the model's terms
(the lines agreement.py reads) and its steps, each from a state to the next
by one input. Every step is checked from its recorded state: TLC must find
a path of MC_Pool's steps from the first state to the second (KvTrace.tla).

A step is checked on the part of the pool it touches, which is how a state
of thousands of blocks is checked. The model's steps read and write the
pool through the blocks, owners' entries and keys they name, and through
nothing else but the free units of a region or of a group's slot rows when
there are none, the `completed` words and the count of copies. So a step's
window is each part of the pool closed under what holds what: a changed
fact (a unit freed or taken, an entry, a key, a block's memories, a piece
of waiting work, a copy or a task), with every holder of each of its units
(owners' entries and slots, keys, waiting work, copies, tasks), every unit,
owner entry and key a piece of waiting work, a copy or a task names, the
units of a key, the owner and the work of an owner the input names, all of
an owner's entries when its group, slot or state changes or it is moved,
and the tasks of a key the input names. Parts that share no changed fact
are separate windows; a fact no step changes may be in several. Each window
keeps one free unit of each region and one free slot row of each group that
the step leaves free and no window holds (the path takes no unit the next
state has free, KvTrace's `Toward`); its stamps and `completed` values are
renumbered in order (the model only compares them), its copies are counted
from those in flight in it, and its units, owners, keys and positions are
renamed in order (the model tells none of them apart but by what holds
what). The paths of a step's windows, taken one after another, are a path
of the whole pool from its state to the next: each reads and writes only
its own part and leaves every fact it shares as the next state has it; with
the copies counted across the windows, which steps.py checks, they reach
the next state. Windows of one layout and of like size are checked in one
TLC run, with units, owners, keys and positions up to the largest of them;
a window's others are in no region, never attached and never cached, so no
step but a resize's grow and shrink touches them, and a path to the next
state shrinks again what it grows, with nothing between that reads it.

A step the model has no terms for is counted as outside the model and not
checked: an open, a pool's image, a close that closes a KV region (its
evictions and shrinks, one step to the pool, are as many to the model,
in every order), or one that changes the layout otherwise than by its
session's groups and slot rows leaving it; a request of a class other
than 0; a load or a store of several keys (the
model makes one per key; a prefetch of several is checked as one per key
in order, and a state states a task of several keys as one per key); a
request of no open session; a state while no session is open, with units
of a region the model lacks held (another class's), with units of a region
it has held by an owner in a class other than 0 or a key cached in such a
class (the model has one class; `# held` and `# class` lines state them),
or with a key cached on the device in several regions (the model caches
one device block per key; `# device` lines state them); a step that takes
or frees units of a region the model lacks; a resize over KV regions of
two unit sizes in one memory space, which the model's chunks cannot
relate; a state with an update on trial of an engine epoch that has several
sessions (the model commits and rolls back per session). Several open
sessions are checked as the model's epochs, one per session (KvCore's
`Epochs`): owners and groups are named under their session (`e<epoch>`,
or `e<epoch>-<n>` for an engine epoch's n-th session, as an engine with a
parked standby has one per engine), and each epoch's trial, whether its
session is closed, and each copy's visibility (R8) are stated; a session's
end is the model's `Lose` of its epoch, checked in the layout it ends, so
a standby's session that takes over is followed.

Prints one verdict line: ok; FAIL with the first step no path follows,
whose window is kept in WORK/failure.txt (with the first 50 such); or
INCONCLUSIVE when a TLC run stopped before its search was complete. With
--expect fail the verdict is ok only when some step is not followed. WORK keeps each run's MC_Trace and
TLC's output; TLC's state files are kept under $TMPDIR (default /tmp).
"""
import argparse
import bisect
import collections
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

# The modules a window's TLC run reads: the model, the pool alone and the trace.
MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
TOOLS = os.environ.get("TLA_TOOLS", "/workspace/dev/tools/tla")
UNIT = re.compile(r"\b(\d+):(\d+)\b")


class Rec(dict):
    """A TLA+ record."""


class Fn(dict):
    """A TLA+ function."""


class Set(list):
    """A TLA+ set."""


class Seq(list):
    """A TLA+ sequence."""


class Raw(str):
    """TLA+ text as it is: the name of a definition."""


def tla(value):
    if isinstance(value, Raw):
        return value
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return '"' + value + '"'
    if isinstance(value, Rec):
        return "[" + ", ".join(f"{name} |-> {tla(item)}" for name, item in value.items()) + "]"
    if isinstance(value, Fn):
        # A function as `@@` of its pairs `k :> v`, in a balanced tree: TLC
        # builds each as a table at once, and a tree's depth is the log of
        # its size where a chain as long as a function overflows TLC's stack.
        pairs = [f"{tla(key)} :> {tla(item)}" for key, item in sorted(value.items())]
        if not pairs:
            return "[x \\in {} |-> 0]"

        def tree(low, high):
            if high - low == 1:
                return "(" + pairs[low] + ")"
            middle = (low + high) // 2
            return "(" + tree(low, middle) + " @@ " + tree(middle, high) + ")"
        return tree(0, len(pairs))
    if isinstance(value, Seq):
        return "<<" + ", ".join(tla(item) for item in value) + ">>"
    if isinstance(value, tuple):
        return "<<" + ", ".join(tla(item) for item in value) + ">>"
    if isinstance(value, (Set, set, frozenset)):
        return "{" + ", ".join(sorted({tla(item) for item in value})) + "}"
    raise TypeError(value)


class Outside(Exception):
    """A state or an input the model has no terms for."""


# ---------------------------------------------------------------------------
# Reading a file of states and steps.


def parse_layout(block):
    """Per group, named by its session (`e<epoch>.g<group>`), its memory, its
    slot rows' region, its KV regions and the host parts it copies with; the
    host regions; per region its memory space and the bytes of a unit; the
    open sessions, with groups or none. None while no session is open."""
    if block == ["layout none"]:
        return None
    groups, hosts, regions, sessions = {}, set(), {}, set()
    for line in block:
        parts = line.split()
        if parts[1] == "group":
            kv = parts[8].split(",") if len(parts) > 8 and parts[8] != "parts" else []
            named = parts[parts.index("parts") + 1:] if "parts" in parts else []
            copied = tuple(named[0].split(",")) if named else ()
            groups[parts[2]] = (parts[4], parts[6], tuple(kv), copied)
        elif parts[1] == "host":
            hosts.add(parts[2])
        elif parts[1] == "region":
            regions[parts[2]] = (parts[4], parts[6])
        elif parts[1] == "session":
            sessions.add(parts[2])
    return (tuple(sorted(groups.items())), tuple(sorted(hosts)), tuple(sorted(regions.items())),
            tuple(sorted(sessions, key=epoch_key)))


def epoch_of(name):
    """The epoch an owner or a group belongs to: `e1` of `e1.5` and `e1.g0`,
    `e1-2` of `e1-2.g0`."""
    return name.split(".")[0]


def unstated(state, owner):
    """What an owner that `state` does not state is: never attached while
    its session is open, gone once its session is closed (a closed
    session's owners are not stated), as KvTrace's `Matches` reads it."""
    return "new" if state.epochs.get(epoch_of(owner), ("closed",))[0] == "open" else "gone"


def epoch_key(name):
    """The order of the model's epochs: by engine epoch, then by session
    (`e1`, `e1-2`, `e2`; an engine with a parked standby has a session per
    engine, each one of the model's epochs)."""
    number, _, nth = name[1:].partition("-")
    return int(number), int(nth or 1)


def ended(before, after, text):
    """Why a change of layout is outside the model, or None when it is a
    session's end beside the sessions that stay (the model's `Lose` of its
    epoch): the closing session's groups leave the layout, the others stay
    as they were, no region appears, and every KV region stays, named by a
    session that stays (an engine's standby shares its KV region), so that
    only the closing session's slot rows close with it; or the last
    session's end where it named no KV region. A close that closes
    a KV region evicts every block cached in it and shrinks each of its
    units, one step to the pool and as many to the model, in every order."""
    kv = lambda layout: {region for _, (_, _, regions, _) in layout[0] for region in regions}
    closing = fields(text).get("e")
    if before is None or not text.startswith("close ") or closing not in before[3]:
        return "a change of layout"
    if after is None:
        # The last session's close closes every region it named.
        return "a close that closes a KV region" if kv(before) else None
    kept = tuple(item for item in before[0] if epoch_of(item[0]) != closing)
    if after[3] != tuple(name for name in before[3] if name != closing) or after[0] != kept:
        return "a change of layout"
    if not set(after[1]) <= set(before[1]) or not set(after[2]) <= set(before[2]):
        return "a change of layout"
    if kv(after) != kv(before):
        return "a close that closes a KV region"
    return None


def kinds(layout):
    """What each region is to the model: the KV units, a group's slot rows, or host units."""
    if layout is None:
        raise Outside("no session open")
    regions = {}
    for group, (_, rows, kv, _) in layout[0]:
        if len(kv) != 1:
            raise Outside("a group with other than one KV region")
        if regions.get(rows, ("rows", group)) != ("rows", group) or regions.get(kv[0], ("kv", None)) != ("kv", None):
            raise Outside("a region of two kinds or of two groups' slot rows")
        regions[kv[0]] = ("kv", None)
        regions[rows] = ("rows", group)
    for region in layout[1]:
        regions.setdefault(region, ("host", None))
    return regions


def layouts(path):
    """Where each layout of the file begins, and its lines, in order (`grep`
    reads a file of gigabytes faster than Python)."""
    found = subprocess.run(["grep", "-b", "^layout", path], capture_output=True, text=True).stdout
    blocks = []
    for line in found.splitlines():
        offset, _, text = line.partition(":")
        offset = int(offset)
        if blocks and blocks[-1][2] == offset:
            blocks[-1][1].append(text)
            blocks[-1][2] = offset + len(text.encode()) + 1
        else:
            blocks.append([offset, [text], offset + len(text.encode()) + 1])
    return [(offset, parse_layout(block)) for offset, block, _ in blocks]


def records(path, begin, end):
    """The states and steps from the first state at or after byte `begin` to
    the first state at or after byte `end`, in order: ("state", number,
    offset, lines) and ("step", source, target, input)."""
    with open(path, "rb") as stream:
        stream.seek(begin)
        if begin:
            stream.readline()
        offset = stream.tell()
        line = stream.readline()
        while line and not line.startswith(b"state "):
            offset = stream.tell()
            line = stream.readline()
        while line:
            text = line.decode().rstrip("\n")
            if text.startswith("state "):
                number, body = int(text.split()[1]), []
                for item in stream:
                    item = item.decode().rstrip("\n")
                    if item == "":
                        break
                    body.append(item.rstrip())
                yield ("state", number, offset, body)
                if offset >= end:
                    return
            elif text.startswith("step "):
                parts = text.split(" ", 3)
                yield ("step", int(parts[1]), int(parts[2]), parts[3] if len(parts) > 3 else "")
            offset = stream.tell()
            line = stream.readline()


def fields(text):
    return dict(item.split("=", 1) for item in text.split() if "=" in item)


def stamps(text):
    """A list of stamps `g0:5,g1:3` as a dict."""
    return {group: int(at) for group, _, at in (item.partition(":") for item in text.split(",") if item)}


# ---------------------------------------------------------------------------
# A state: its facts, and what holds what.

RECORDS = ("copy", "commit", "finish", "release", "transfer", "replicate", "task")


def record(line):
    """A piece of waiting work, a copy or a task: its kind, units, owners
    with the positions it names, and keys."""
    word, _, rest = line.partition(" ")
    found = fields(rest)
    units = {f"{region}:{unit}" for region, unit in UNIT.findall(rest)}
    owner = found.get("o", "-")
    owners = {} if owner == "-" else {owner: {found["p"]} if found.get("p", "-") != "-" else set()}
    key = found.get("k", "-")
    kind = word if word != "task" else "task " + rest.split()[0]
    return kind, units, owners, set() if key == "-" else {key}


class Free:
    """The free units of a state as the line states them: per region, runs
    of units `(first, last)` in order. A region of tens of thousands of
    units is a few runs, so nothing here walks every free unit."""

    def __init__(self, text=""):
        self.text = text
        self.runs = collections.defaultdict(list)
        for item in text.split():
            region, _, span = item.partition(":")
            first, _, last = span.partition("-")
            self.runs[region].append((int(first), int(last or first)))
        for runs in self.runs.values():
            runs.sort()
        self.starts = {region: [first for first, _ in runs] for region, runs in self.runs.items()}

    def __contains__(self, unit):
        region, _, number = unit.partition(":")
        runs = self.runs.get(region)
        if not runs:
            return False
        number = int(number)
        at = bisect.bisect_right(self.starts[region], number) - 1
        return at >= 0 and runs[at][1] >= number

    def regions(self):
        return set(self.runs)

    def without(self, other):
        """The units free here and not in `other`, one by one."""
        found = []
        for region, runs in self.runs.items():
            theirs = other.runs.get(region, [])
            at = 0
            for first, last in runs:
                while at < len(theirs) and theirs[at][1] < first:
                    at += 1
                position, scan = first, at
                while position <= last:
                    if scan < len(theirs) and theirs[scan][0] <= last:
                        low, high = theirs[scan]
                        found += [f"{region}:{unit}" for unit in range(position, min(low, last + 1))]
                        position = max(position, high + 1)
                        scan += 1
                    else:
                        found += [f"{region}:{unit}" for unit in range(position, last + 1)]
                        break
        return found

    def changed(self, other):
        """The units free in one state and not in the other."""
        if self.text == other.text:
            return []
        return self.without(other) + other.without(self)

    def first_common(self, other, region, held):
        """The lowest unit of `region` free in both and not in `held`, or None."""
        ours, theirs = self.runs.get(region, []), other.runs.get(region, [])
        at = 0
        for first, last in ours:
            while at < len(theirs) and theirs[at][1] < first:
                at += 1
            scan = at
            while scan < len(theirs) and theirs[scan][0] <= last:
                low, high = max(first, theirs[scan][0]), min(last, theirs[scan][1])
                for unit in range(low, high + 1):
                    if f"{region}:{unit}" not in held:
                        return f"{region}:{unit}"
                scan += 1
        return None


class State:
    """A state's facts by what they name. `lines` holds every line but the
    free units, `ncopy` and `completed`."""

    def __init__(self):
        self.free = Free()
        self.ncopy = 0
        self.completed = ""
        self.owners = {}          # owner -> (is, group, slot, {position: (kind, unit)})
        self.owner_lines = {}
        self.keys = {}            # key -> (d, h, line)
        self.valid = {}           # unit -> line
        self.records = {}         # line -> record(line)
        self.classes = {}         # key -> class
        self.devices = {}         # key -> its device copies, where more than one
        self.foreign = set()      # the regions of the units owners hold in a class other than 0
        self.epochs = {}          # epoch -> (open or closed, trial)
        self.makers = {}          # key -> (the epochs its device and host copies are visible to alone)
        self.by_maker = collections.defaultdict(set)  # epoch -> {key}
        self.holders = collections.defaultdict(set)   # unit -> {fact}
        self.by_owner = collections.defaultdict(set)  # owner -> {record line}
        self.by_key = collections.defaultdict(set)    # key -> {task line}
        self.lines = set()

    def add(self, line):
        word, _, rest = line.partition(" ")
        if word == "ncopy":
            self.ncopy = int(rest)
        elif word == "completed":
            self.completed = rest
        elif word == "free":
            self.free = Free(rest)
        elif word == "#":
            parts = rest.split()
            if parts[0] == "class":
                self.classes[parts[1]] = parts[2]
            elif parts[0] == "device":
                self.devices[parts[1]] = parts[2]
            elif parts[0] == "held":
                self.foreign.update(unit.split(":")[0] for unit in parts[3:])
        else:
            self.lines.add(line)
            if word == "owner":
                parts = rest.split()
                owner = parts[0]
                if parts[1] == "live":
                    entries = {}
                    for entry in parts[4:]:
                        position, _, held = entry[1:].partition("=")
                        kind, _, unit = held.partition(":")
                        entries[position] = (kind, unit)
                        self.holders[unit].add(("e", owner, position))
                    slot = parts[3][len("slot="):]
                    self.holders[slot].add(("h", owner))
                    self.owners[owner] = ("live", parts[2], slot, entries)
                else:
                    self.owners[owner] = (parts[1], "-", "-", {})
                self.owner_lines[owner] = line
            elif word == "key":
                found = fields(rest)
                key = rest.split()[0]
                self.keys[key] = (found["d"], found["h"], line)
                for unit in (found["d"], found["h"]):
                    if unit != "-":
                        self.holders[unit].add(("k", key))
                self.makers[key] = (found["vd"], found["vh"])
                for epoch in self.makers[key]:
                    if epoch != "-":
                        self.by_maker[epoch].add(key)
            elif word == "epoch":
                parts = rest.split()
                self.epochs[parts[0]] = (parts[1], parts[2][len("trial="):])
            elif word == "valid":
                self.valid[rest.split()[0]] = line
            elif word in RECORDS:
                found = record(line)
                self.records[line] = found
                for unit in found[1]:
                    self.holders[unit].add(("l", line))
                for owner in found[2]:
                    self.by_owner[owner].add(line)
                if word == "task":
                    for key in found[3]:
                        self.by_key[key].add(line)
            else:
                raise ValueError(f"a line the model has no terms for: {line}")

    def remove(self, line):
        word, _, rest = line.partition(" ")
        self.lines.discard(line)
        if word == "owner":
            owner = rest.split()[0]
            _, _, slot, entries = self.owners.pop(owner)
            for position, (_, unit) in entries.items():
                self.holders[unit].discard(("e", owner, position))
            if slot != "-":
                self.holders[slot].discard(("h", owner))
            del self.owner_lines[owner]
        elif word == "key":
            key = rest.split()[0]
            d, h, _ = self.keys.pop(key)
            for unit in (d, h):
                if unit != "-":
                    self.holders[unit].discard(("k", key))
            for epoch in self.makers.pop(key):
                self.by_maker[epoch].discard(key)
        elif word == "epoch":
            del self.epochs[rest.split()[0]]
        elif word == "valid":
            del self.valid[rest.split()[0]]
        elif word in RECORDS:
            found = self.records.pop(line)
            for unit in found[1]:
                self.holders[unit].discard(("l", line))
            for owner in found[2]:
                self.by_owner[owner].discard(line)
            if word == "task":
                for key in found[3]:
                    self.by_key[key].discard(line)


def outside_state(state, regions):
    """Why a state is outside the model's terms, or None."""
    if any(cls != "0" for cls in state.classes.values()):
        return "a key cached in a class other than 0"
    if state.devices:
        return "a key cached on the device in several regions"
    # The model commits or rolls back an update per epoch, which is a
    # session to it; an engine epoch with a standby has several.
    for epoch, (_, trial) in state.epochs.items():
        if "-" in epoch and trial != "no":
            return "an update on trial of an engine epoch with several sessions"
    # A unit of the model's region held in another class is neither free
    # nor held in its terms (one class); another class's region is none of
    # its business while no step takes or frees a unit of it.
    if state.foreign & set(regions):
        return "units of the model's region held in a class other than 0"
    for line in state.lines:
        for region, _ in UNIT.findall(line):
            if region not in regions:
                return "units of a region the model lacks"
    return None


# ---------------------------------------------------------------------------
# The inputs.

ARRIVE = {"release", "detach", "commit", "finish", "transfer", "replicate"}
STUTTER = {"victims", "list", "unreadable", "limits", "hold", "release-holds", "grant", "withdraw"}
OUTSIDE = {"open", "image"}
ACTS = {"attach": ["Attach"], "allocate": ["Allocate"], "pin": ["Pin"], "clone": ["Clone"],
        "load": ["Load"], "save": ["Save", "EvictHost"], "restore": ["Restore"], "evict": ["Evict"],
        "completed": ["Advance"], "tick": ["Deadline"], "done": ["PrefetchEnd", "PutEnd"],
        "ended": ["TransferEnd", "ReplicateEnd", "PrefetchEnd", "PutEnd"], "publish": ["CommitUpdate"],
        "roll-back": ["Rollback", "Evict", "EvictHost"],
        # A session's end: its engine is lost (every stamp of its groups
        # counts as reached, as `Advance`s of their words), and the device
        # regions no other session names close, their cached blocks evicted.
        "close": ["Lose", "Advance", "Evict", "Shrink"],
        # A resize of a KV region, a group's slot region or the host region:
        # a shrink of a KV or host region evicts in the installed order, the
        # device copies or the host copies of the region it shrinks.
        "resize": ["Evict", "EvictHost", "Shrink", "Grow"]}


def acts(text):
    """The actions an input allows, or why the model has no terms for it."""
    kind, found = text.split(" ")[0], fields(text)
    if kind in OUTSIDE:
        raise Outside(kind)
    if found.get("class", "0") != "0":
        raise Outside("a class other than 0")
    keys = found.get("ks", "").split(",")
    if kind == "load" and len(keys) > 1:
        raise Outside("a load of several keys")
    if kind in STUTTER:
        return []
    if kind in ARRIVE:
        return ["Arrive"]
    if kind == "store":
        if len(keys) > 1:
            raise Outside("a store of several keys")
        return {"host": ["Store", "EvictHost"], "remote": ["StoreRemote"]}.get(found["tier"], [])
    if kind == "prefetch":
        # A prefetch makes room for its block as a store does, by evicting host copies.
        return ["Prefetch", "EvictHost"]
    if kind not in ACTS:
        raise ValueError(f"an input the model has no terms for: {text}")
    return ACTS[kind]


# ---------------------------------------------------------------------------
# A step's windows.


def changes(before, after):
    """What a step changes: the facts of either state that differ."""
    changed = set()
    changed |= {("u", unit) for unit in before.free.changed(after.free)}
    removed, added = before.lines - after.lines, after.lines - before.lines
    for line in removed | added:
        word, _, rest = line.partition(" ")
        if word == "owner":
            # An owner of a session that closes is not stated after it: one
            # gone before the close is gone after it, and does not change.
            owner = rest.split()[0]
            old = before.owners.get(owner, (unstated(before, owner), "-", "-", {}))
            new = after.owners.get(owner, (unstated(after, owner), "-", "-", {}))
            if old[:3] != new[:3]:
                changed.add(("h", owner))
                changed.add(("f", owner))
            for position in set(old[3]) | set(new[3]):
                if old[3].get(position) != new[3].get(position):
                    changed.add(("e", owner, position))
                    for entry in (old[3].get(position), new[3].get(position)):
                        if entry:
                            changed.add(("u", entry[1]))
        elif word == "key":
            key = rest.split()[0]
            changed.add(("k", key))
            for state in (before, after):
                if key in state.keys:
                    changed |= {("u", unit) for unit in state.keys[key][:2] if unit != "-"}
        elif word == "valid":
            changed.add(("u", rest.split()[0]))
        elif word == "epoch":
            changed.add(("E", rest.split()[0]))
        else:
            changed.add(("l", line))
            changed |= {("u", unit) for unit in record(line)[1]}
    return changed


def named(text):
    """The facts an input names, which one window holds together."""
    kind, found = text.split(" ")[0], fields(text)
    facts = set()
    if "o" in found:
        owner = found["o"]
        facts |= {("h", owner), ("r", owner)}
        facts |= {("e", owner, position) for position in filter(None, found.get("ps", "").split(","))}
        if kind in ("detach", "transfer"):
            facts.add(("f", owner))
    for key in filter(None, found.get("ks", "").split(",")):
        facts |= {("k", key), ("K", key)}
    return facts


def closure(fact, states, partners):
    """The facts a fact brings into its window, in either state. `partners`
    gives, per KV region, the KV regions of other layouts whose units of one
    number occupy the same memory (R10)."""
    kind = fact[0]
    found = set()
    if kind == "u":
        region, _, number = fact[1].partition(":")
        found |= {("u", f"{other}:{number}") for other in partners.get(region, ())}
    for state in states:
        if kind == "u":
            found |= state.holders.get(fact[1], set())
        elif kind == "E":
            found |= {("k", key) for key in state.by_maker.get(fact[1], set())}
        elif kind == "e":
            owner, position = fact[1], fact[2]
            found.add(("h", owner))
            entry = state.owners.get(owner, (None, None, None, {}))[3].get(position)
            if entry:
                found.add(("u", entry[1]))
        elif kind == "h":
            owner = fact[1]
            if owner in state.owners and state.owners[owner][2] != "-":
                found.add(("u", state.owners[owner][2]))
        elif kind == "f":
            owner = fact[1]
            found.add(("h", owner))
            found.add(("r", owner))
            found |= {("e", owner, position) for position in state.owners.get(owner, (0, 0, 0, {}))[3]}
        elif kind == "r":
            found.add(("h", fact[1]))
            found |= {("l", line) for line in state.by_owner.get(fact[1], set())}
        elif kind == "k":
            if fact[1] in state.keys:
                found |= {("u", unit) for unit in state.keys[fact[1]][:2] if unit != "-"}
        elif kind == "K":
            found.add(("k", fact[1]))
            found |= {("l", line) for line in state.by_key.get(fact[1], set())}
        elif kind == "l":
            line = fact[1]
            known = state.records.get(line)
            if known is None:
                continue
            word, units, owners, keys = known
            found |= {("u", unit) for unit in units}
            for owner, positions in owners.items():
                found.add(("h", owner))
                found |= {("e", owner, position) for position in positions}
                if word in ("transfer", "task transfer"):
                    found |= {("f", owner), ("r", owner)}
            found |= {("k", key) for key in keys}
            if word.startswith("task"):
                found |= {("K", key) for key in keys}
    return found


def windows_of(before, after, text, partners):
    """The parts of the pool a step touches that share no changed fact:
    each a set of facts. A fact that does not change may be in several
    parts: no path changes it for good, since each part's path ends in the
    next state's facts, and the paths are taken one after another."""
    changed = changes(before, after)
    states = (before, after)
    parent = {fact: fact for fact in changed}

    def find(fact):
        while parent[fact] != fact:
            parent[fact] = parent[parent[fact]]
            fact = parent[fact]
        return fact

    def reach(start):
        seen, pending = set(start), collections.deque(start)
        while pending:
            for other in closure(pending.popleft(), states, partners):
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        return seen

    reached, done = [], set()
    # The input's facts are one part, whatever they reach.
    starts = [named(text)] + [{fact} for fact in sorted(changed, key=repr)]
    for start in starts:
        if not start or (len(start) == 1 and next(iter(start)) in done):
            continue
        found = reach(start)
        touched = sorted(found & changed, key=repr)
        if not touched:
            continue
        done.update(touched)
        for fact in touched[1:]:
            one, other = find(touched[0]), find(fact)
            if one != other:
                parent[other] = one
        reached.append((touched[0], found))
    parts = collections.defaultdict(set)
    for fact, found in reached:
        parts[find(fact)] |= found
    # An epoch's trial is read by any step: a step that changes one (a commit
    # or a rollback of an update) is one window.
    if any(fact[0] == "E" for fact in changed) and len(parts) > 1:
        return [set().union(*parts.values())]
    return list(parts.values())


# ---------------------------------------------------------------------------
# A window in the model's terms.


def projected(state, part, witnesses, rename):
    """The lines of `state` within `part`, renamed, with every epoch."""
    units = {fact[1] for fact in part if fact[0] == "u"} | witnesses
    full = {fact[1] for fact in part if fact[0] == "f"}
    lines = [f"free {' '.join(sorted(rename(unit) for unit in units if unit in state.free))}"]
    lines += [f"epoch {rename(epoch, 'e')} {is_} trial={trial}" for epoch, (is_, trial) in sorted(state.epochs.items())]
    # An epoch the window knows that the state does not state: its session
    # is closed, and its update was committed or rolled back.
    lines += [f"epoch {rename(epoch, 'e')} closed trial=unstated" for epoch in rename.epochs
              if epoch not in state.epochs]
    for owner in sorted({fact[1] for fact in part if fact[0] == "h"}):
        if owner not in state.owners:
            lines.append(f"owner {rename(owner, 'o')} {unstated(state, owner)}")
            continue
        is_, group, slot, entries = state.owners[owner]
        if is_ != "live":
            lines.append(f"owner {rename(owner, 'o')} {is_}")
            continue
        kept = [f"p{rename(position, 'p')}={kind}:{rename(unit)}"
                for position, (kind, unit) in sorted(entries.items())
                if owner in full or ("e", owner, position) in part]
        lines.append(" ".join([f"owner {rename(owner, 'o')} live {rename(group, 'g')} slot={rename(slot)}"] + kept))
    for key in sorted({fact[1] for fact in part if fact[0] == "k"}):
        if key in state.keys:
            d, h, line = state.keys[key]
            remote = fields(line)["remote"]
            made = [rename(epoch, "e") for epoch in state.makers[key]]
            lines.append(f"key {rename(key, 'k')} d={rename(d)} h={rename(h)} remote={remote} "
                         f"vd={made[0]} vh={made[1]}")
    for unit in sorted(units):
        if unit in state.valid:
            lines.append(rename_units(state.valid[unit], rename))
    for line in sorted(fact[1] for fact in part if fact[0] == "l"):
        if line in state.records:
            lines.append(rename_record(line, rename))
    return lines


def rename_units(text, rename):
    return UNIT.sub(lambda match: rename(match.group(0)), text)


def rename_stamps(text, rename):
    """Stamps `<group>:<sequence>,...` with their groups renamed."""
    return ",".join(f"{rename(group, 'g')}:{at}" for group, _, at in (item.partition(":") for item in text.split(",")
                                                                      if item))


def rename_record(line, rename):
    """A record line with its units, owners, groups, positions, keys and
    sessions renamed."""
    def item(token):
        name, sep, value = token.partition("=")
        if not sep:
            return rename_units(token, rename)
        if value == "-":
            return token
        if name == "o":
            return f"o={rename(value, 'o')}"
        if name == "p":
            return f"p={rename(value, 'p')}"
        if name == "k":
            return f"k={rename(value, 'k')}"
        if name == "to":
            return f"to={rename(value, 'g')}"
        if name == "s":
            return f"s={rename(value, 'e')}"
        if name == "after":
            return f"after={rename_stamps(value, rename)}"
        return f"{name}={rename_units(value, rename)}"
    return " ".join(item(token) for token in line.split(" "))


def rename_input(text, rename):
    kind, found = text.split(" ")[0], fields(text)
    parts = [kind]
    for name, value in found.items():
        if name == "o":
            value = rename(value, "o")
        elif name in ("g", "to"):
            value = rename(value, "g")
        elif name == "e":
            value = rename(value, "e")
        elif name == "after":
            value = rename_stamps(value, rename)
        elif name == "ps":
            value = ",".join(rename(position, "p") for position in value.split(",") if position)
        elif name == "ks":
            value = ",".join(rename(key, "k") for key in value.split(",") if key)
        parts.append(f"{name}={value}")
    return " ".join(parts)


class Renaming:
    """Names in order, per region for units: what a window's model knows of
    its units, owners, keys and positions; epochs in order, those with
    groups in the layout first, and owners in order within their epoch
    (`e1.o1`); a group keeps its number under its epoch's name."""

    def __init__(self, units, owners, keys, positions, epochs):
        self.names = {}
        by_region = collections.defaultdict(list)
        for unit in units:
            region, _, number = unit.partition(":")
            by_region[region].append(int(number))
        for region, numbers in by_region.items():
            for index, number in enumerate(sorted(numbers)):
                self.names[("u", f"{region}:{number}")] = f"{region}:{index + 1}"
        self.epochs = list(epochs)
        for index, epoch in enumerate(epochs):
            self.names[("e", epoch)] = f"e{index + 1}"
        by_epoch = collections.defaultdict(list)
        for owner in owners:
            by_epoch[epoch_of(owner)].append(owner)
        self.counts = {region: len(numbers) for region, numbers in by_region.items()}
        for epoch, named in by_epoch.items():
            for index, owner in enumerate(sorted(named, key=lambda item: int(item.split(".")[1]))):
                self.names[("o", owner)] = f"{self.names[('e', epoch)]}.o{index + 1}"
            self.counts[f"owners {self.names[('e', epoch)]}"] = len(named)
        for prefix, items in (("k", keys), ("p", positions)):
            ordered = sorted(items, key=lambda item: (len(item), item))
            for index, item in enumerate(ordered):
                self.names[(prefix, item)] = f"{prefix}{index + 1}" if prefix != "p" else str(index + 1)
        self.counts.update(keys=len(keys), positions=len(positions), epochs=len(epochs))

    def __call__(self, item, prefix="u"):
        if item == "-" or (prefix == "e" and item == "none"):
            return item
        if prefix == "g":
            epoch, _, group = item.partition(".")
            return f"{self.names[('e', epoch)]}.{group}"
        return self.names[(prefix, item)]


def window(step, before, after, text, part, witnesses, layout, chunks):
    """A window as plain data: its renamed states, input and witnesses, and
    its size."""
    units = {fact[1] for fact in part if fact[0] == "u"} | witnesses
    owners = {fact[1] for fact in part if fact[0] in ("h", "f", "r")}
    keys = {fact[1] for fact in part if fact[0] in ("k", "K")}
    positions = {fact[2] for fact in part if fact[0] == "e"}
    found = fields(text)
    owners |= {found["o"]} if "o" in found else set()
    keys |= set(filter(None, found.get("ks", "").split(",")))
    positions |= set(filter(None, found.get("ps", "").split(",")))
    for state in (before, after):
        for fact in part:
            if fact[0] == "f" and fact[1] in state.owners:
                positions |= set(state.owners[fact[1]][3])
            elif fact[0] == "l" and fact[1] in state.records:
                _, _, named_owners, named_keys = state.records[fact[1]]
                owners |= set(named_owners)
                keys |= named_keys
                for named_positions in named_owners.values():
                    positions |= named_positions
    # Every epoch: those of the layout's groups in order, then the others.
    sessions = sorted({epoch_of(group) for group, _ in layout[0]}, key=epoch_key)
    others = (set(before.epochs) | set(after.epochs)) - set(sessions)
    epochs = sessions + sorted(others, key=epoch_key)
    rename = Renaming(units, owners, keys, positions, epochs)
    lines = [projected(state, part, witnesses, rename) for state in (before, after)]

    # The memory each KV unit occupies, where units of two regions share one
    # (another layout's region over the same memory): their chunk, named in
    # order; no entry where a unit shares its memory with none of the window.
    shared = collections.defaultdict(list)
    for unit in units:
        region, _, number = unit.partition(":")
        if region in chunks:
            shared[(chunks[region], number)].append(rename(unit))
    occupied = {}
    for index, (_, named) in enumerate(sorted((chunk, sorted(named)) for chunk, named in shared.items()
                                                if len(named) > 1)):
        occupied.update(dict.fromkeys(named, f"c{index + 1}"))

    # The copies of the window: those in flight in it are numbered first.
    def identity(line):
        found = fields(line)
        return found["kind"], found["write"], found["mem"]
    copies = [{identity(line) for line in body if line.startswith("copy ")} for body in lines]
    made = len(copies[1] - copies[0])

    # The order in which a request takes its positions: a pin's in order,
    # each with its key; an allocation's whose units are free first, then
    # those whose units it evicts, as the model evicts only from a region
    # with no free unit (the Rust buffer pool evicts what it lacks and then
    # takes units from the region).
    order = [position for position in found.get("ps", "").split(",") if position]
    if text.startswith("allocate") and found["o"] in after.owners:
        entries = after.owners[found["o"]][3]
        index = {position: at for at, position in enumerate(order)}
        order.sort(key=lambda position: (position not in entries or entries[position][1] not in before.free,
                                         index[position]))
    korder = [key for key in found.get("ks", "").split(",") if key]

    # The keys whose device or host copy the window's next state no longer
    # caches, which its evictions take in this order.
    def dropped(index):
        gone = []
        for key in keys:
            if key in before.keys and before.keys[key][index] != "-":
                if key not in after.keys or after.keys[key][index] == "-":
                    gone.append(rename(key, "k"))
        return sorted(gone, key=lambda name: int(name[1:]))
    return {"step": step, "input": rename_input(text, rename), "raw": text,
            "order": [rename(position, "p") for position in order],
            "korder": [rename(key, "k") for key in korder],
            "evict": dropped(0), "evicthost": dropped(1),
            "lines": lines, "ncopy": (len(copies[0]), len(copies[0]) + made), "made": made,
            "completed": tuple(" ".join(f"{rename(group, 'g')}={at}" for group, _, at in
                                        (entry.partition("=") for entry in state.completed.split()))
                               for state in (before, after)),
            "witnesses": sorted(rename(unit) for unit in witnesses),
            "layout": rename_layout(layout, rename), "chunks": occupied,
            "counts": rename.counts}


def rename_layout(layout, rename):
    """A layout with its groups named by the window's epochs."""
    groups = tuple(sorted((rename(group, "g"), info) for group, info in layout[0]))
    return (groups, layout[1], layout[2])


def witnesses_of(before, after, parts, layout):
    """One free unit of each KV region and one free slot row of each group
    that the step leaves free and no window holds."""
    held = set()
    for part in parts:
        held |= {fact[1] for fact in part if fact[0] == "u"}
    chosen = set()
    for _, (_, rows, kv, _) in layout[0]:
        for region in (kv[0], rows):
            unit = before.free.first_common(after.free, region, held)
            if unit:
                chosen.add(unit)
    return chosen


def state_of(body):
    state = State()
    for line in body:
        state.add(line)
    return state


def memory_of(layout):
    """Per KV region whose memory space holds another KV region of units of
    the same size, that space, and per such region the others: their unit of
    one number is one chunk of memory (R10). And whether a space holds KV
    regions of units of two sizes, whose memory the model's chunks cannot
    relate."""
    spaces = collections.defaultdict(list)
    sizes = dict(layout[2])
    for _, (_, _, kv, _) in layout[0]:
        for region in kv:
            if region in sizes and region not in spaces[sizes[region][0]]:
                spaces[sizes[region][0]].append(region)
    chunks, partners, mixed = {}, {}, False
    for space, regions in spaces.items():
        if len(regions) < 2:
            continue
        if len({sizes[region][1] for region in regions}) > 1:
            mixed = True
            continue
        for region in regions:
            chunks[region] = space
            partners[region] = [other for other in regions if other != region]
    return chunks, partners, mixed


def step_windows(step, before, after, layouts_of, text):
    """A step's windows; or why it is outside the model; or none when it
    changes nothing; and a failure the windows cannot show."""
    try:
        acts(text)
        if "e?" in text.split(".")[0] or any(value.startswith("e?") for value in fields(text).values()):
            raise Outside("a request of no open session")
        # A session's end is checked in the layout it ends: its groups are
        # then those of a lost epoch, which the next state no longer states.
        if layouts_of[0] != layouts_of[1]:
            reason = ended(layouts_of[0], layouts_of[1], text)
            if reason:
                raise Outside(reason)
        regions = kinds(layouts_of[0])
        for state in (before, after):
            reason = outside_state(state, regions)
            if reason:
                raise Outside(reason)
        # Free units of a region the model lacks (another class's) are none
        # of its business while they stay free.
        lacked = (before.free.regions() | after.free.regions()) - set(regions)
        if any(before.free.runs.get(region) != after.free.runs.get(region) for region in lacked):
            raise Outside("units of a region the model lacks taken or freed")
        chunks, partners, mixed = memory_of(layouts_of[0])
        if mixed and text.startswith("resize"):
            raise Outside("a resize over KV regions of two unit sizes in one memory")
    except Outside as reason:
        return [], str(reason), None
    parts = windows_of(before, after, text, partners)
    if not parts and before.ncopy == after.ncopy and before.completed == after.completed:
        return [], None, None
    spare = witnesses_of(before, after, parts, layouts_of[0])
    # The `completed` words and the count of copies alone, when no fact
    # changes with them.
    found = [window(step, before, after, text, part, spare, layouts_of[0], chunks) for part in (parts or [set()])]
    made = sum(item["made"] for item in found)
    failure = None
    if made != after.ncopy - before.ncopy:
        failure = (step, text, f"the count of copies goes from {before.ncopy} to {after.ncopy} "
                               f"while {made} copies appear")
    return found, None, failure


def scan(job):
    """The windows of the steps of a byte range of a file whose states
    follow one another, each after the step that leads to it."""
    path, begin, end, marks = job
    windows, outside, unchanged, failures = [], collections.Counter(), 0, []
    current, pending = None, None

    def layout_at(offset):
        found = None
        for mark, layout in marks:
            if mark <= offset:
                found = layout
        return found

    for item in records(path, begin, end):
        if item[0] == "step":
            pending = item
            continue
        _, number, offset, body = item
        state, layout = state_of(body), layout_at(offset)
        if current is not None and pending is not None and pending[1] == current[0] and pending[2] == number:
            found, reason, failure = step_windows((current[0], number), current[1], state,
                                                  (current[2], layout), pending[3])
            windows.extend(found)
            if reason:
                outside[reason] += 1
            elif not found:
                unchanged += 1
            if failure:
                failures.append(failure)
        current, pending = (number, state, layout), None
    return windows, outside, unchanged, failures


def scan_whole(path):
    """The windows of the steps of a file read whole: the enumeration's,
    whose steps follow all its states."""
    states, steps, layout = {}, [], None
    lines = open(path).read().split("\n")
    at = 0
    while at < len(lines):
        line = lines[at]
        if line.startswith("layout"):
            block = []
            while at < len(lines) and lines[at].startswith("layout"):
                block.append(lines[at])
                at += 1
            layout = parse_layout(block)
            continue
        if line.startswith("state "):
            number, body = int(line.split()[1]), []
            at += 1
            while at < len(lines) and lines[at] != "":
                body.append(lines[at].rstrip())
                at += 1
            states[number] = (layout, body)
            continue
        if line.startswith("step "):
            parts = line.split(" ", 3)
            steps.append((int(parts[1]), int(parts[2]), parts[3] if len(parts) > 3 else ""))
        at += 1
    parsed = {}
    windows, outside, unchanged, failures = [], collections.Counter(), 0, []
    for source, target, text in steps:
        for number in (source, target):
            if number not in parsed:
                parsed[number] = state_of(states[number][1])
        found, reason, failure = step_windows((source, target), parsed[source], parsed[target],
                                              (states[source][0], states[target][0]), text)
        windows.extend(found)
        if reason:
            outside[reason] += 1
        elif not found:
            unchanged += 1
        if failure:
            failures.append(failure)
    return windows, outside, unchanged, failures


def split(path, jobs):
    """Byte ranges of the file for the scans."""
    size = os.path.getsize(path)
    bounds = [size * index // jobs for index in range(jobs)] + [size]
    return [(bounds[index], bounds[index + 1]) for index in range(jobs) if bounds[index] < bounds[index + 1]]


# ---------------------------------------------------------------------------
# A TLC run over windows of one layout.


def parse_lines(lines, rank):
    """A window's lines as the record KvTrace's `Matches` compares."""
    state = Rec(ncopy=0, completed=Fn(), free=Set(), epochs=Fn(), owners=Fn(), keys=Set(), valid=Set(),
                copies=Set(), commits=Set(), transfers=Set(), replicates=Set(), finishes=Set(),
                releases=Set(), tasks=Set())
    after = lambda text: Fn((group, rank[at]) for group, at in stamps(text).items())
    for line in lines:
        word, _, rest = line.partition(" ")
        if word == "free":
            state["free"] = Set(rest.split())
        elif word == "epoch":
            parts = rest.split()
            state["epochs"][parts[0]] = Rec(**{"is": parts[1]}, trial=parts[2][len("trial="):])
        elif word == "owner":
            parts = rest.split()
            owner = Rec(**{"is": parts[1]}, g="-", slot="-", t=Set())
            if parts[1] == "live":
                owner["g"], owner["slot"] = parts[2], parts[3][len("slot="):]
                for entry in parts[4:]:
                    position, _, held = entry[1:].partition("=")
                    kind, _, unit = held.partition(":")
                    owner["t"].append((position, kind, unit))
            state["owners"][parts[0]] = owner
        elif word == "key":
            parts = rest.split()
            found = fields(rest)
            state["keys"].append(Rec(k=parts[0], d=found["d"], h=found["h"], r=int(found["remote"]),
                                     vd=found["vd"], vh=found["vh"]))
        elif word == "valid":
            parts = rest.split()
            state["valid"].append((parts[0], Set(parts[1:])))
        elif word == "copy":
            state["copies"].append(copy_line(rest))
        elif word == "commit":
            found = fields(rest)
            state["commits"].append(Rec(o=found["o"], p=found["p"], u=found["u"], k=found["k"],
                                        after=after(found["after"])))
        elif word == "transfer":
            found = fields(rest)
            state["transfers"].append(Rec(o=found["o"], to=found["to"], after=after(found["after"])))
        elif word == "replicate":
            found = fields(rest)
            state["replicates"].append(Rec(o=found["o"], p=found["p"], to=found["to"], u=found["u"],
                                           after=after(found["after"])))
        elif word == "finish":
            ok, waits, copy = rest.split(" ", 2)
            state["finishes"].append(Rec(ok=int(ok[len("ok="):]), after=after(waits[len("after="):]),
                                         copy=copy_line(copy)))
        elif word == "release":
            waits, _, units = rest.partition(" ")
            state["releases"].append(Rec(after=after(waits[len("after="):]), units=Set(units.split())))
        elif word == "task":
            kind, _, rest = rest.partition(" ")
            if kind == "transfer":
                match = re.match(r"o=(\S+) to=(\S+) slot=(\S+) units=(.*?) ?writes=(.*)$", rest)
                o, to, slot, units, writes = match.groups()
                state["tasks"].append(Rec(kind=kind, o=o, p="-", to=to, slot=slot, units=Set(units.split()),
                                          writes=Set(writes.split()), h="-", expired=0, k="-", parts=Set(),
                                          s="-"))
            elif kind == "replicate":
                found = fields(rest)
                state["tasks"].append(Rec(kind=kind, o=found["o"], p=found["p"], to=found["to"], slot="-",
                                          units=Set([found["u"]]), writes=Set([found["u"]]), h="-",
                                          expired=0, k="-", parts=Set(), s="-"))
            else:
                found = fields(rest)
                state["tasks"].append(Rec(kind=kind, o="-", p="-", to="-", slot="-", units=Set(), writes=Set(),
                                          h=found["h"], expired=int(found["expired"]), k=found["k"],
                                          parts=Set(filter(None, found["parts"].split(","))), s=found["s"]))
        else:
            raise ValueError(f"a line the model has no terms for: {line}")
    return state


def copy_line(text):
    found = fields(text)
    return Rec(kind=found["kind"], o=found["o"], p=found["p"], read=found["read"],
               write=found["write"], fill=int(found["fill"]), mem=found["mem"], k=found["k"],
               parts=Set(filter(None, found["parts"].split(","))), s=found["s"])


def values(item):
    """The stamps and `completed` values a window names, which are renumbered."""
    found = {0}
    for completed in item["completed"]:
        found |= {int(entry.split("=")[1]) for entry in completed.split()}
    for body in item["lines"]:
        for line in body:
            for match in re.finditer(r"after=(\S*)", line):
                found |= set(stamps(match.group(1)).values())
    named_values = fields(item["input"])
    if named_values.get("after"):
        found |= {int(entry.split(":")[1]) for entry in named_values["after"].split(",")}
    if "value" in named_values:
        found.add(int(named_values["value"]))
    return sorted(found)


def check(job):
    """One TLC run over windows of one layout: for each window, whether its
    next state was reached; and the count of distinct states."""
    batch, work, run, heap, scratch = job
    layout = batch[0]["layout"]
    regions = kinds(layout)
    groups = [group for group, _ in layout[0]]
    universe = collections.Counter()
    for item in batch:
        for name, count in item["counts"].items():
            universe[name] = max(universe[name], count)
    units = {kind: [] for kind in ("kv", "rows", "host")}
    for region, (kind, _) in sorted(regions.items()):
        units[kind] += [f"{region}:{index + 1}" for index in range(universe.get(region, 0))]
    keys = [f"k{index + 1}" for index in range(universe["keys"])]
    positions = [str(index + 1) for index in range(universe["positions"])]
    # Epochs and owners up to the most of the windows, an owner under its epoch.
    epochs = [f"e{index + 1}" for index in range(universe["epochs"])]
    owners = [f"{epoch}.o{index + 1}" for epoch in epochs for index in range(universe[f"owners {epoch}"])]

    names, definitions = {}, []

    def define(prefix, value):
        text = tla(value)
        if text not in names:
            names[text] = f"{prefix}{len(names) + 1}"
            definitions.append(f"{names[text]} == {text}")
        return names[text]

    entries, batches, tasks, copies = [], 0, 0, 0
    for item in batch:
        numbers = values(item)
        rank = {value: index for index, value in enumerate(numbers)}
        batches = max(batches, len(numbers))
        states = []
        for body, ncopy, completed in zip(item["lines"], item["ncopy"], item["completed"]):
            state = parse_lines(body, rank)
            state["ncopy"] = ncopy
            state["completed"] = Fn((group, rank[int(at)]) for group, at in
                                    (entry.split("=") for entry in completed.split()))
            # Owners and epochs the window has no use for: owners never
            # attached, epochs neither on trial nor closed.
            for owner in owners:
                state["owners"].setdefault(owner, Rec(**{"is": "new"}, g="-", slot="-", t=Set()))
            for epoch in epochs:
                state["epochs"].setdefault(epoch, Rec(**{"is": "open"}, trial="no"))
            states.append(state)
            tasks = max(tasks, sum(1 for task in state["tasks"] if task["kind"] in ("prefetch", "put")))
            copies = max(copies, ncopy)
        start = states[0]
        ordered = Seq(sorted(start["copies"], key=tla))
        finishes = Set(Rec(ok=line["ok"], at=ordered.index(line["copy"]) + 1, after=line["after"])
                       for line in start["finishes"])
        # The device units, slot rows and host units in their regions: those
        # the first state names, free or held. A unit of the window it does
        # not name is outside its region (one a resize grows), or one the
        # run's universe has beyond this window's.
        member = Set(unit for unit in units["kv"] + units["rows"] + units["host"]
                     if re.search(rf"(?<![\d:]){re.escape(unit)}(?![\d])", "\n".join(item["lines"][0])))
        begin = Rec(state=Raw(define("S", start)), member=member, copies=ordered, finishes=finishes)

        text = item["input"]
        jump = 0
        if text.startswith("completed"):
            found = fields(text)
            now = start["completed"].get(found["g"], 0)
            jump = max(0, rank[int(found["value"])] - now)
        # Steps enough for each changed line and each changed entry of an
        # owner's line (a pin of 128 keys changes one line and 128 entries).
        changed = set(item["lines"][0]) ^ set(item["lines"][1])
        held = [{entry for line in body if line.startswith("owner ") for entry in
                 (f"{line.split()[1]} {position}" for position in line.split()[5:])} for body in item["lines"]]
        bound = 4 + 2 * (len(changed) + len(held[0] ^ held[1])) + jump
        entries.append(f"[start |-> {tla(begin)}, "
                       f"input |-> {define('I', input_record(item, groups, owners, positions, keys, rank))}, "
                       f"state |-> {define('S', states[1])}, bound |-> {bound}]")

    directory = os.path.join(work, f"run{run}")
    os.makedirs(directory, exist_ok=True)
    for module in ("KvCore.tla", "MC_Pool.tla", "KvTrace.tla"):
        shutil.copy(os.path.join(MODEL, module), directory)
    slot_group = Fn((slot, regions[slot.split(":")[0]][1]) for slot in units["rows"])
    memory = Fn((group, info[0]) for group, info in layout[0])
    kv_region = Fn((group, "r" + info[2][0]) for group, info in layout[0])
    # The host parts each group copies with, by their bits; a crossing may
    # go to any other group of its session, each stating the class's regions.
    group_parts = Fn((group, Set(info[3])) for group, info in layout[0])
    host_parts = Set(sorted({part for _, info in layout[0] for part in info[3]}))
    readers = Fn((group, Set(other for other in groups if other != group and epoch_of(other) == epoch_of(group)))
                 for group in groups)
    # The memory a KV unit occupies: its own, or the chunk it shares with a
    # unit of another layout's region (the windows of a run share these).
    chunk = {unit: unit for unit in units["kv"]}
    chunk.update(batch[0]["chunks"])
    # Each constant is the name of a definition of its value: TLC evaluates a
    # definition that substitutes a constant (`<-`) at each use, and one it
    # names once, at the start.
    values_of = {
        "TraceUnits": tla(Set(units["kv"])),
        "TraceSlots": tla(Set(units["rows"])),
        "TraceHostUnits": tla(Set(units["host"])),
        "TraceKeys": tla(Set(keys)),
        "TraceOwners": tla(Set(owners)),
        "TraceGroups": tla(Set(groups)),
        "TracePositions": tla(Set(positions)),
        "TraceSlotGroup": tla(slot_group),
        "TraceEpochs": tla(Set(epochs)),
        "TraceGroupEpoch": tla(Fn((group, epoch_of(group)) for group in groups)),
        "TraceOwnerEpoch": tla(Fn((owner, epoch_of(owner)) for owner in owners)),
        "TraceGroupMemory": tla(memory),
        "TraceUnitRegion": tla(Fn((unit, "r" + unit.split(":")[0]) for unit in units["kv"])),
        "TraceUnitChunk": tla(Fn(chunk)),
        "TraceGroupRegion": tla(kv_region),
        "TraceHostParts": tla(host_parts),
        "TraceGroupParts": tla(group_parts),
        "TraceReaders": tla(readers),
        "TraceRules": "AllRules",
        # Every device unit, slot row and host unit: a window's start states
        # which are in their regions (KvTrace's `Start`), and `Resize` may
        # move any.
        "TraceRegionUnits": "TraceUnits \\cup TraceSlots \\cup TraceHostUnits",
    }
    module = [
        "---- MODULE MC_Trace ----",
        "EXTENDS KvTrace",
        *(f"{name}Value == {value}\n{name} == {name}Value" for name, value in values_of.items()),
        *definitions,
        "TraceWindowsValue == <<" + ",\n    ".join(entries) + ">>",
        "TraceWindows == TraceWindowsValue",
        "====",
    ]
    with open(os.path.join(directory, "MC_Trace.tla"), "w") as out:
        out.write("\n".join(module) + "\n")
    constants = {
        "Units": "<- TraceUnits", "Slots": "<- TraceSlots", "HostUnits": "<- TraceHostUnits",
        "Keys": "<- TraceKeys", "Owners": "<- TraceOwners", "Groups": "<- TraceGroups",
        "Positions": "<- TracePositions", "Epochs": "<- TraceEpochs", "Transfers": "= TRUE",
        "Stamping": "<- TraceGroups", "SlotGroup": "<- TraceSlotGroup",
        "GroupEpoch": "<- TraceGroupEpoch", "OwnerEpoch": "<- TraceOwnerEpoch",
        "GroupMemory": "<- TraceGroupMemory", "UnitRegion": "<- TraceUnitRegion",
        "UnitChunk": "<- TraceUnitChunk", "GroupRegion": "<- TraceGroupRegion",
        "InitialUnits": "<- TraceRegionUnits", "Movable": "<- TraceRegionUnits", "TrialEpochs": "= {}",
        "Losable": "<- TraceEpochs",
        "MaxBatch": f"= {batches}", "MaxRequests": "= 1", "MaxCopies": f"= {copies}",
        "MaxTasks": f"= {tasks}", "Hazards": "= {}", "Eager": "= TRUE",
        "Rules": "<- TraceRules", "Windows": "<- TraceWindows", "HostParts": "<- TraceHostParts",
        "GroupParts": "<- TraceGroupParts", "Readers": "<- TraceReaders",
    }
    # MC_Pool's model values: the names on the first line of its CONSTANTS.
    pool = open(os.path.join(MODEL, "MC_Pool.tla")).read()
    declared = re.search(r"^CONSTANTS ([^\n]*)", pool, flags=re.M).group(1)
    constants.update({name: f"= {name}" for name in re.findall(r"\w+", declared)})
    with open(os.path.join(directory, "MC_Trace.cfg"), "w") as out:
        out.write("INIT TraceInit\nNEXT TraceNext\nVIEW TraceView\nCONSTANTS\n")
        out.writelines(f"    {name} {value}\n" for name, value in constants.items())
        out.write("CHECK_DEADLOCK FALSE\n")

    java = os.path.join(TOOLS, "jdk-21.0.13+9", "bin", "java")
    states_dir = tempfile.mkdtemp(prefix=f"run{run}-", dir=scratch)
    command = [java, "-XX:+UseParallelGC", "-XX:ParallelGCThreads=2", f"-Xmx{heap}", "-Xss64m",
               f"-DTLA-Library={TOOLS}/tlaps/lib/tlapm/stdlib", "-cp", f"{TOOLS}/tla2tools.jar",
               "tlc2.TLC", "-workers", "1", "-noGenerateSpecTE", "-metadir", states_dir,
               "-config", "MC_Trace.cfg", "MC_Trace"]
    with open(os.path.join(directory, "tlc.log"), "w") as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=False)
    shutil.rmtree(states_dir, ignore_errors=True)
    output = open(os.path.join(directory, "tlc.log")).read()
    if "Model checking completed" not in output:
        errors = [line for line in output.split("\n") if line.startswith("Error") or "Exception" in line]
        return None, f"TLC did not complete ({directory}/tlc.log): {' '.join(errors[:3])}"
    reached = {int(window) for window in re.findall(r'<<"reached", (\d+)>>', output)}
    # The final count: the progress lines before it state counts so far.
    distinct = re.findall(r"([\d,]+) distinct states found", output)[-1]
    return [index + 1 in reached for index in range(len(batch))], int(distinct.replace(",", ""))


def input_record(item, groups, owners, positions, keys, rank):
    text = item["input"]
    kind, named_values = text.split(" ")[0], fields(text)
    after = Fn((group, 0) for group in groups)
    for stamp in filter(None, named_values.get("after", "").split(",")):
        group, at = stamp.split(":")
        after[group] = rank[int(at)]
    listed = lambda name, every: Set(named_values[name].split(",")) if named_values.get(name) else Set(every)
    request = Rec(kind=kind if kind in ARRIVE else "none", o=named_values.get("o", "none"),
                  ps=listed("ps", []), p=named_values.get("ps", "none"), k=named_values.get("ks", "none"),
                  to=named_values.get("to", "none"), ok=named_values.get("ok") == "1", after=after,
                  e=named_values.get("e", "none"))
    completed = Fn((group, 0) for group in groups)
    if kind == "completed":
        completed[named_values["g"]] = rank[int(named_values["value"])]
    if kind == "close":
        # The groups of the session that ends: every stamp of theirs reached.
        for group in groups:
            if epoch_of(group) == named_values["e"]:
                completed[group] = max(rank.values())
    order, korder = Seq(item["order"]), Seq(item["korder"])
    evict, evicthost = Seq(item["evict"]), Seq(item["evicthost"])
    return Rec(acts=Set(acts(text)), os=Set([named_values["o"]]) if "o" in named_values else Set(owners),
               gs=Set([named_values["g"]]) if "g" in named_values else Set(groups),
               ps=listed("ps", positions), ks=listed("ks", keys), order=order, korder=korder,
               evict=evict, evicthost=evicthost,
               req=request, completed=completed)


def weight(item):
    """How large a window's states are: its units, keys and owners' entries."""
    counts = item["counts"]
    owners = sum(count for name, count in counts.items() if name.startswith("owners "))
    units = sum(count for name, count in counts.items()
                if not name.startswith("owners ") and name not in ("keys", "positions", "epochs"))
    return units + counts["keys"] + owners * (counts["positions"] + 1)


def batches_of(windows, limit):
    """Windows of one layout, of the same units sharing memory, and alike in
    size, at most `limit` per run."""
    grouped = collections.defaultdict(list)
    for item in windows:
        grouped[(repr(item["layout"]), repr(sorted(item["chunks"].items())))].append(item)
    found = []
    for items in grouped.values():
        items.sort(key=weight)
        current = []
        for item in items:
            if current and (len(current) >= limit or weight(item) > 2 * weight(current[0]) + 16):
                found.append(current)
                current = []
            current.append(item)
        if current:
            found.append(current)
    return found


def sequential(path):
    """Whether each state of the file follows the step that leads to it, as
    a replay writes them (the enumeration writes its steps last)."""
    with open(path) as stream:
        for line in stream:
            if line.startswith("step "):
                return True
            if line.startswith("state ") and line.split()[1] != "0":
                return False
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("STEPS_JOBS", "16")),
                        help="scans and TLC runs at once")
    parser.add_argument("--batch", type=int, default=400, help="windows per TLC run")
    parser.add_argument("--expect", choices=["pass", "fail"], default="pass")
    parser.add_argument("steps")
    parser.add_argument("work")
    arguments = parser.parse_args()
    os.makedirs(arguments.work, exist_ok=True)
    name = os.path.basename(arguments.steps)
    begun = time.time()
    heap = os.environ.get("TLC_HEAP", "3g")
    scratch = tempfile.mkdtemp(prefix="steps-", dir=os.environ.get("TMPDIR", "/tmp"))

    windows, outside, unchanged, failures = [], collections.Counter(), 0, []
    if sequential(arguments.steps):
        marks = layouts(arguments.steps)
        jobs = [(arguments.steps, begin, end, marks) for begin, end in split(arguments.steps, arguments.jobs)]
        with multiprocessing.Pool(arguments.jobs) as workers:
            scans = list(workers.imap(scan, jobs))
    else:
        scans = [scan_whole(arguments.steps)]
    for found, left, same, failed in scans:
        windows.extend(found)
        outside.update(left)
        unchanged += same
        failures.extend(failed)
    scanned = time.time() - begun

    batches = batches_of(windows, arguments.batch)
    runs = [(batch, arguments.work, index, heap, scratch) for index, batch in enumerate(batches)]
    distinct, errors = 0, []
    with multiprocessing.Pool(arguments.jobs) as workers:
        for batch, (reached, found) in zip(batches, workers.imap(check, runs)):
            if reached is None:
                errors.append(found)
                continue
            distinct += found
            failures.extend((item["step"], item["raw"], item) for item, ok in zip(batch, reached) if not ok)
    shutil.rmtree(scratch, ignore_errors=True)

    seconds = time.time() - begun
    steps = len({item["step"] for item in windows}) + unchanged
    left = ", ".join(f"{count} {reason}" for reason, count in sorted(outside.items()))
    summary = (f"{steps} steps ({unchanged} that change nothing) in {len(windows)} windows, "
               f"{len(batches)} TLC runs ({distinct} distinct states), scanned in {scanned:.0f} s, "
               f"{seconds:.0f} s in all" + (f"; outside the model: {left}" if left else ""))
    detail = ""
    if failures:
        failures.sort(key=lambda failure: failure[0])
        with open(os.path.join(arguments.work, "failure.txt"), "w") as out:
            for (source, target), text, item in failures[:50]:
                out.write(f"=== step {source} -> {target} by `{text}` not followed\n")
                if isinstance(item, dict):
                    out.write(f"--- window (renamed), input `{item['input']}`, witnesses {item['witnesses']}\n")
                    out.write("--- before\n" + "\n".join(item["lines"][0]) + "\n")
                    out.write("--- after\n" + "\n".join(item["lines"][1]) + "\n")
                else:
                    out.write(item + "\n")
        (source, target), text, item = failures[0]
        detail = (f"{len(failures)} windows not followed; the first: step {source} -> {target} by `{text}` "
                  f"({arguments.work}/failure.txt)")
    if failures and arguments.expect == "pass":
        verdict = f"FAIL  {name}: {detail}; {summary}"
        if errors:
            verdict += f"; and {len(errors)} TLC runs did not complete, the first: {errors[0]}"
    elif errors:
        # A run that stops before its search is complete settles nothing
        # (specs/buffer-pool.md §18).
        verdict = (f"INCONCLUSIVE  {name}: {len(errors)} TLC runs did not complete, the first: {errors[0]}"
                   f"{'; ' + detail if failures else ''}; {summary}")
    elif arguments.expect == "pass":
        verdict = f"FAIL  {name}: {detail}; {summary}" if failures else f"ok    {name}: every step followed: {summary}"
    else:
        verdict = (f"ok    {name}: reported as expected: {detail}" if failures
                   else f"FAIL  {name}: expected a step not followed, and every step was: {summary}")
    print(verdict)
    return 0 if verdict.startswith("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
