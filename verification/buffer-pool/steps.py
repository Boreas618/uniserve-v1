#!/usr/bin/env python3
"""Checks steps of the Rust buffer pool against the design model's steps.

    steps.py [--chain N] [--chunk C] [--expect fail] STEPS WORK

STEPS is what the enumeration of L1 (`agreement_steps.sh`) or the replay of
a trace (`replay.sh`) wrote: layouts, the pool's states in the model's terms
(the lines agreement.py reads) and its steps, each from a state to the next
by one input. Each step is a window, or with --chain each run of at most N
consecutive steps: TLC must find, from the window's first state, a path of
MC_Pool's steps through each of its states in turn (KvTrace.tla). The
windows are checked in runs of at most C windows of one layout, each run
with the constants of its windows: their units, slot rows, host units,
owners, positions and keys, and their stamps and `completed` values
renumbered in order (the model only compares them), with one value above
them for the stamps of waiting work, which no state states.

A step the model has no terms for ends a window and is counted as outside
the model: an open, a close or a pool's image; a request of a class other
than 0; a prefetch from the peers; a copy, a prefetch or a put of several
keys (the model makes one per key); a state while no session or several are
open, with units of a region the model lacks, or with a key cached in a
class other than 0 (the model has one class; `# class` lines state them).

Prints one verdict line: ok, or FAIL with the first step no path follows,
whose states are kept in WORK/failure.txt; with --expect fail the verdict is
ok only when some step is not followed. WORK keeps each run's MC_Trace and
TLC's output.
"""
import argparse
import collections
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.environ.get("TLA_TOOLS", "/workspace/dev/tools/tla")
WORKERS = os.environ.get("TLC_WORKERS", "16")
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
        if not value:
            return "[x \\in {} |-> 0]"
        return "(" + " @@ ".join(f"{tla(key)} :> {tla(item)}" for key, item in sorted(value.items())) + ")"
    if isinstance(value, Seq):
        return "<<" + ", ".join(tla(item) for item in value) + ">>"
    if isinstance(value, tuple):
        return "<<" + ", ".join(tla(item) for item in value) + ">>"
    if isinstance(value, (Set, set, frozenset)):
        return "{" + ", ".join(sorted({tla(item) for item in value})) + "}"
    raise TypeError(value)


class Outside(Exception):
    """A state or an input the model has no terms for."""


def read(path):
    """The states of a file by number, each with its lines and the layout it
    was written under, and the steps in the order written."""
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
    return states, steps


def parse_layout(block):
    """Per group its memory, its slot rows' region and its KV regions; the
    host regions. None while no session or several are open."""
    if block == ["layout none"]:
        return None
    groups, hosts = {}, set()
    for line in block:
        parts = line.split()
        if parts[1] == "group":
            kv = parts[8].split(",") if len(parts) > 8 else []
            groups[parts[2]] = (parts[4], parts[6], tuple(kv))
        elif parts[1] == "host":
            hosts.add(parts[2])
    return (tuple(sorted(groups.items())), tuple(sorted(hosts)))


def kinds(layout):
    """What each region is to the model: the KV units, a group's slot rows, or host units."""
    if layout is None:
        raise Outside("no session or several open")
    regions = {}
    for group, (_, rows, kv) in layout[0]:
        if len(kv) != 1:
            raise Outside("a group with other than one KV region")
        regions[kv[0]] = ("kv", None)
        regions[rows] = ("rows", group)
    for region in layout[1]:
        regions.setdefault(region, ("host", None))
    return regions


def fields(text):
    return dict(item.split("=", 1) for item in text.split() if "=" in item)


def copy_line(text):
    found = fields(text)
    return Rec(kind=found["kind"], o=found["o"], p=found["p"], read=found["read"],
               write=found["write"], fill=int(found["fill"]), mem=found["mem"])


def parse_state(lines, regions):
    """A state's lines as the record KvTrace's `Matches` compares."""
    for line in lines:
        for region, _ in UNIT.findall(line):
            if region not in regions:
                raise Outside("units of a region the model lacks")
    state = Rec(ncopy=0, completed=Fn(), free=Set(), owners=Fn(), keys=Set(), valid=Set(),
                copies=Set(), commits=Set(), transfers=Set(), finishes=Set(), released=Set(),
                tasks=Set())
    for line in lines:
        word, _, rest = line.partition(" ")
        if word == "#":
            # What the model has no terms for: a key cached in a class other than 0.
            parts = rest.split()
            if parts[0] == "class" and parts[2] != "0":
                raise Outside("a key cached in a class other than 0")
            continue
        if word == "ncopy":
            state["ncopy"] = int(rest)
        elif word == "completed":
            state["completed"] = Fn((group, int(at)) for group, at in (item.split("=") for item in rest.split()))
        elif word == "free":
            state["free"] = Set(rest.split())
        elif word == "owner":
            parts = rest.split()
            owner = Rec(**{"is": parts[1]}, g="-", slot="-", t=Set())
            if parts[1] == "live":
                owner["g"], owner["slot"] = parts[2], parts[3][len("slot="):]
                for entry in parts[4:]:
                    position, _, held = entry[1:].partition("=")
                    kind, _, unit = held.partition(":")
                    owner["t"].append((position, kind, unit))
            if parts[0] in state["owners"]:
                raise Outside("an owner named twice")
            state["owners"][parts[0]] = owner
        elif word == "key":
            parts = rest.split()
            found = fields(rest)
            state["keys"].append(Rec(k=parts[0], d=found["d"], h=found["h"], r=int(found["remote"])))
        elif word == "valid":
            parts = rest.split()
            state["valid"].append((parts[0], Set(parts[1:])))
        elif word == "copy":
            state["copies"].append(copy_line(rest))
        elif word == "commit":
            found = fields(rest)
            state["commits"].append(Rec(o=found["o"], p=found["p"], u=found["u"]))
        elif word == "transfer":
            found = fields(rest)
            state["transfers"].append(Rec(o=found["o"], to=found["to"]))
        elif word == "finish":
            ok, _, copy = rest.partition(" ")
            state["finishes"].append(Rec(ok=int(ok[len("ok="):]), copy=copy_line(copy)))
        elif word == "release":
            state["released"] = Set(rest.split())
        elif word == "task":
            kind, _, rest = rest.partition(" ")
            if kind == "transfer":
                match = re.match(r"o=(\S+) to=(\S+) slot=(\S+) units=(.*?) ?writes=(.*)$", rest)
                o, to, slot, units, writes = match.groups()
                state["tasks"].append(Rec(kind=kind, o=o, to=to, slot=slot, units=Set(units.split()),
                                          writes=Set(writes.split()), h="-", expired=0))
            else:
                found = fields(rest)
                state["tasks"].append(Rec(kind=kind, o="-", to="-", slot="-", units=Set(), writes=Set(),
                                          h=found["h"], expired=int(found["expired"])))
        else:
            raise ValueError(f"a line the model has no terms for: {line}")
    return state


ARRIVE = {"release", "detach", "commit", "finish", "transfer"}
STUTTER = {"victims", "list", "unreadable", "limits", "hold", "release-peer", "grant", "withdraw"}
OUTSIDE = {"open", "close", "image"}
ACTS = {"attach": ["Attach"], "allocate": ["Allocate"], "pin": ["Pin"], "clone": ["Clone"],
        "load": ["Load"], "save": ["Save", "EvictHost"], "restore": ["Restore"], "evict": ["Evict"],
        "completed": ["Advance"], "tick": ["Deadline"], "done": ["PrefetchEnd", "PutEnd"],
        "ended": ["TransferEnd", "PrefetchEnd", "PutEnd"], "publish": ["CommitUpdate"],
        "roll-back": ["Rollback"], "resize": ["Evict", "Shrink", "Grow"]}


def acts(text):
    """The actions an input allows, or why the model has no terms for it."""
    kind, found = text.split(" ")[0], fields(text)
    if kind in OUTSIDE:
        raise Outside(kind)
    if found.get("class", "0") != "0":
        raise Outside("a class other than 0")
    keys = found.get("ks", "").split(",")
    if kind in ("load", "prefetch") and len(keys) > 1:
        raise Outside(f"a {kind} of several keys")
    if kind in STUTTER:
        return []
    if kind in ARRIVE:
        return ["Arrive"]
    if kind == "store":
        if len(keys) > 1:
            raise Outside("a store of several keys")
        return {"host": ["Store", "EvictHost"], "remote": ["StoreRemote"]}.get(found["tier"], [])
    if kind == "prefetch":
        if found["source"] != "remote":
            raise Outside("a prefetch from the peers")
        return ["Prefetch"]
    if kind not in ACTS:
        raise ValueError(f"an input the model has no terms for: {text}")
    return ACTS[kind]


def values(state_lines, input_text):
    """The stamps and `completed` values a step names, which are renumbered."""
    found = {0}
    for line in state_lines:
        if line.startswith("completed"):
            found |= {int(item.split("=")[1]) for item in line.split()[1:]}
    named = fields(input_text)
    if "after" in named and named["after"]:
        found |= {int(item.split(":")[1]) for item in named["after"].split(",")}
    if "value" in named:
        found.add(int(named["value"]))
    return found


def check(windows, states, work, run):
    """One TLC run over windows of one layout: the furthest step each window reached."""
    layout = states[windows[0][0][0]][0]
    regions = kinds(layout)
    groups = [group for group, _ in layout[0]]
    numbers = sorted(set().union(*(values(states[s][1] + states[t][1], text)
                                   for window in windows for s, t, text in window)))
    # The values renumbered in order, and one more above them all: the
    # stamp, which no state states, of work that waits for a later batch.
    rank = {value: index for index, value in enumerate(numbers)}

    parsed, units, owners, positions, keys = {}, set(), set(), set(), set()
    copies = tasks = 0
    for window in windows:
        for number in {window[0][0]} | {t for _, t, _ in window}:
            if number in parsed:
                continue
            state = parse_state(states[number][1], regions)
            state["completed"] = Fn((group, rank[at]) for group, at in state["completed"].items())
            parsed[number] = state
            for line in states[number][1]:
                units |= {f"{region}:{unit}" for region, unit in UNIT.findall(line)}
            owners |= set(state["owners"])
            positions |= {entry[0] for owner in state["owners"].values() for entry in owner["t"]}
            positions |= {line["p"] for line in state["commits"]}
            positions |= {line["p"] for line in state["copies"] if line["p"] != "-"}
            keys |= {line["k"] for line in state["keys"]}
            copies = max(copies, state["ncopy"])
            tasks = max(tasks, sum(1 for task in state["tasks"] if task["kind"] != "transfer"))
        for _, _, text in window:
            named = fields(text)
            positions |= set(filter(None, named.get("ps", "").split(",")))
            keys |= set(filter(None, named.get("ks", "").split(",")))
            if "o" in named:
                owners.add(named["o"])
    of = lambda kind: sorted(unit for unit in units if regions[unit.split(":")[0]][0] == kind)
    kv, slots, hosts = of("kv"), of("rows"), of("host")

    def input_record(text):
        kind, named = text.split(" ")[0], fields(text)
        after = Fn((group, 0) for group in groups)
        for item in filter(None, named.get("after", "").split(",")):
            group, at = item.split(":")
            after[group] = rank[int(at)]
        listed = lambda name, every: Set(named[name].split(",")) if named.get(name) else Set(every)
        request = Rec(kind=kind if kind in ARRIVE else "none", o=named.get("o", "none"),
                      ps=listed("ps", []), p=named.get("ps", "none"), k=named.get("ks", "none"),
                      to=named.get("to", "none"), ok=named.get("ok") == "1", after=after)
        completed = Fn((group, 0) for group in groups)
        if kind == "completed":
            completed[named["g"]] = rank[int(named["value"])]
        return Rec(acts=Set(acts(text)), os=Set([named["o"]]) if "o" in named else Set(owners),
                   gs=Set([named["g"]]) if "g" in named else Set(groups), ps=listed("ps", positions),
                   ks=listed("ks", keys), req=request, completed=completed)

    names, definitions = {}, []

    def named(prefix, value):
        text = tla(value)
        if text not in names:
            names[text] = f"{prefix}{len(names) + 1}"
            definitions.append(f"{names[text]} == {text}")
        return names[text]

    entries = []
    for window in windows:
        start = window[0][0]
        state = parsed[start]
        lines = Seq(state["copies"])
        finishes = Set(Rec(ok=item["ok"], at=lines.index(item["copy"]) + 1) for item in state["finishes"])
        member = Set(unit for unit in kv if re.search(rf"(?<![\d:]){re.escape(unit)}(?![\d])",
                                                       "\n".join(states[start][1])))
        begin = Rec(state=Raw(named("S", state)), member=member, copies=lines, finishes=finishes)
        steps = []
        for source, target, text in window:
            stated = lambda number: {line for line in states[number][1] if not line.startswith("#")}
            before, after = stated(source), stated(target)
            jump = 0
            if text.startswith("completed"):
                found = fields(text)
                now = parsed[source]["completed"].get(found["g"], 0)
                jump = max(0, rank[int(found["value"])] - now)
            bound = 4 + 2 * len(before ^ after) + jump
            steps.append(f"[input |-> {named('I', input_record(text))}, "
                         f"state |-> {named('S', parsed[target])}, bound |-> {bound}]")
        entries.append(f"[start |-> {tla(begin)}, "
                       f"steps |-> <<{', '.join(steps)}>>]")

    directory = os.path.join(work, f"run{run}")
    os.makedirs(directory, exist_ok=True)
    for module in ("KvCore.tla", "MC_Pool.tla", "KvTrace.tla"):
        shutil.copy(os.path.join(HERE, module), directory)
    slot_group = Fn((slot, regions[slot.split(":")[0]][1]) for slot in slots)
    memory = Fn((group, info[0]) for group, info in layout[0])
    kv_region = Fn((group, "r" + info[2][0]) for group, info in layout[0])
    module = [
        "---- MODULE MC_Trace ----",
        "EXTENDS KvTrace",
        f"TraceUnits == {tla(Set(kv))}",
        f"TraceSlots == {tla(Set(slots))}",
        f"TraceHostUnits == {tla(Set(hosts))}",
        f"TraceKeys == {tla(Set(keys))}",
        f"TraceOwners == {tla(Set(owners))}",
        f"TraceGroups == {tla(Set(groups))}",
        f"TracePositions == {tla(Set(positions))}",
        f"TraceSlotGroup == {tla(slot_group)}",
        "TraceGroupEpoch == [g \\in TraceGroups |-> e1]",
        "TraceOwnerEpoch == [o \\in TraceOwners |-> e1]",
        f"TraceGroupMemory == {tla(memory)}",
        f"TraceUnitRegion == {tla(Fn((unit, 'r' + unit.split(':')[0]) for unit in kv))}",
        "TraceUnitChunk == [u \\in TraceUnits |-> u]",
        f"TraceGroupRegion == {tla(kv_region)}",
        *definitions,
        "TraceWindows == <<" + ",\n    ".join(entries) + ">>",
        "====",
    ]
    with open(os.path.join(directory, "MC_Trace.tla"), "w") as out:
        out.write("\n".join(module) + "\n")
    constants = {
        "Units": "<- TraceUnits", "Slots": "<- TraceSlots", "HostUnits": "<- TraceHostUnits",
        "Keys": "<- TraceKeys", "Owners": "<- TraceOwners", "Groups": "<- TraceGroups",
        "Positions": "<- TracePositions", "Epochs": "= {e1}", "Transfers": "= TRUE",
        "Stamping": "<- TraceGroups", "SlotGroup": "<- TraceSlotGroup",
        "GroupEpoch": "<- TraceGroupEpoch", "OwnerEpoch": "<- TraceOwnerEpoch",
        "GroupMemory": "<- TraceGroupMemory", "UnitRegion": "<- TraceUnitRegion",
        "UnitChunk": "<- TraceUnitChunk", "GroupRegion": "<- TraceGroupRegion",
        "InitialUnits": "<- TraceUnits", "Movable": "<- TraceUnits", "TrialEpochs": "= {}",
        "MaxBatch": f"= {len(numbers)}", "MaxRequests": "= 1", "MaxCopies": f"= {copies}",
        "MaxTasks": f"= {tasks}", "Losable": "= {}", "Hazards": "= {}", "Eager": "= TRUE",
        "Rules": "<- AllRules", "Windows": "<- TraceWindows",
    }
    constants.update({name: f"= {name}" for name in ("ga", "gb", "sa", "sb", "sc", "sd", "e1")})
    with open(os.path.join(directory, "MC_Trace.cfg"), "w") as out:
        out.write("INIT TraceInit\nNEXT TraceNext\nVIEW TraceView\nCONSTANTS\n")
        out.writelines(f"    {name} {value}\n" for name, value in constants.items())
        out.write("CHECK_DEADLOCK FALSE\n")

    java = os.path.join(TOOLS, "jdk-21.0.13+9", "bin", "java")
    command = [java, "-XX:+UseParallelGC", f"-Xmx{os.environ.get('TLC_HEAP', '16g')}",
               f"-DTLA-Library={TOOLS}/tlaps/lib/tlapm/stdlib", "-cp", f"{TOOLS}/tla2tools.jar",
               "tlc2.TLC", "-workers", WORKERS, "-noGenerateSpecTE", "-metadir",
               os.path.join(directory, "states"), "-config", "MC_Trace.cfg", "MC_Trace"]
    with open(os.path.join(directory, "tlc.log"), "w") as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=False)
    output = open(os.path.join(directory, "tlc.log")).read()
    if "Model checking completed" not in output:
        errors = [line for line in output.split("\n") if line.startswith("Error") or "Exception" in line]
        raise RuntimeError(f"TLC did not complete ({directory}/tlc.log): {' '.join(errors[:3])}")
    reached = collections.defaultdict(int)
    for window, at in re.findall(r'<<"reached", (\d+), (\d+)>>', output):
        reached[int(window)] = max(reached[int(window)], int(at))
    distinct = re.search(r"([\d,]+) distinct states found", output)
    return [reached[index + 1] for index in range(len(windows))], distinct.group(1) if distinct else "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", type=int, default=0, help="windows of up to N consecutive steps")
    parser.add_argument("--chunk", type=int, default=2000, help="windows per TLC run")
    parser.add_argument("--expect", choices=["pass", "fail"], default="pass")
    parser.add_argument("steps")
    parser.add_argument("work")
    arguments = parser.parse_args()
    os.makedirs(arguments.work, exist_ok=True)
    name = os.path.basename(arguments.steps)
    begun = time.time()

    states, steps = read(arguments.steps)
    windows, outside = [], collections.Counter()
    current = []
    for source, target, text in steps:
        try:
            acts(text)
            for number in (source, target):
                parse_state(states[number][1], kinds(states[number][0]))
            if states[source][0] != states[target][0]:
                raise Outside("a change of layout")
        except Outside as reason:
            outside[str(reason)] += 1
            if current:
                windows.append(current)
            current = []
            continue
        joined = current and arguments.chain and current[-1][1] == source and len(current) < arguments.chain
        if current and not joined:
            windows.append(current)
            current = []
        current.append((source, target, text))
    if current:
        windows.append(current)

    by_layout = collections.defaultdict(list)
    for window in windows:
        by_layout[states[window[0][0]][0]].append(window)
    failures, runs, distinct = [], 0, []
    for layout, group in by_layout.items():
        for first in range(0, len(group), arguments.chunk):
            chunk = group[first:first + arguments.chunk]
            reached, found = check(chunk, states, arguments.work, runs)
            runs += 1
            distinct.append(found)
            failures.extend((window, at) for window, at in zip(chunk, reached) if at < len(window))
    seconds = time.time() - begun
    checked = sum(len(window) for window in windows)
    left = ", ".join(f"{count} {reason}" for reason, count in sorted(outside.items()))
    summary = (f"{checked} steps in {len(windows)} windows, {runs} TLC runs "
               f"({' + '.join(distinct)} distinct states), {seconds:.0f} s"
               + (f"; outside the model: {left}" if left else ""))

    if failures:
        window, at = failures[0]
        source, target, text = window[at]
        with open(os.path.join(arguments.work, "failure.txt"), "w") as out:
            out.write(f"window from state {window[0][0]}; step {source} -> {target} by `{text}` not followed\n")
            out.write(f"--- state {source}\n" + "\n".join(states[source][1]) + "\n")
            out.write(f"--- state {target}\n" + "\n".join(states[target][1]) + "\n")
        detail = (f"{len(failures)} of {len(windows)} windows not followed; the first: step {source} -> {target} "
                  f"by `{text}` (window from state {window[0][0]}, {arguments.work}/failure.txt)")
    if arguments.expect == "pass":
        verdict = f"FAIL  {name}: {detail}; {summary}" if failures else f"ok    {name}: every step followed: {summary}"
    else:
        verdict = (f"ok    {name}: reported as expected: {detail}" if failures
                   else f"FAIL  {name}: expected a step not followed, and every step was: {summary}")
    print(verdict)
    return 0 if verdict.startswith("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
