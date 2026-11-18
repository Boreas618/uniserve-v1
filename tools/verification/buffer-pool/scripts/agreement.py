#!/usr/bin/env python3
"""Compares the states the Rust buffer pool reaches with the model's (L1).

    agreement.py SHAPE MODEL_DUMP RUST_STATES

MODEL_DUMP is what TLC's -dump wrote for MC_Pool under the shape's
configuration; RUST_STATES is what the crate's enumeration wrote for the
shape. Both are projected onto the same facts, one line each: the units that
are free (as runs), each owner's group, slot and table, each key's copies, the
memories in which each device block an owner, the cache or a waiting commit
holds is valid and the host parts that hold each host copy, what waits for
a stamp with the stamps it still waits for, the copies and the tasks in
flight with their keys, and the `completed` words.

Passes when every state the Rust buffer pool reaches is a state of the
model; prints the first that is not, with a state it was reached from. The
dump may be a named pipe that TLC writes while this reads: a configuration's
states are then never stored.
"""
import hashlib
import multiprocessing
import os
import re
import sys

# The names of the model's constants in the Rust configuration of each shape:
# region:unit. Region 0 holds the KV units; each group's slot rows follow; the
# host units are last.
NAMES = {
    "core": {"u1": "0:1", "u2": "0:2", "sa": "1:1", "sb": "1:2"},
    "copies": {"u1": "0:1", "u2": "0:2", "sa": "1:1", "sb": "1:2", "h1": "2:0"},
    "handoff": {"u1": "0:1", "u2": "0:2", "sa": "1:1", "sc": "2:1"},
    "tiers": {"u1": "0:1", "sa": "1:1", "h1": "2:0"},
    "cross": {"u1": "0:1", "sa": "1:1", "sc": "2:1"},
    "parts": {"u1": "0:1", "sa": "1:1", "sc": "2:1", "h1": "3:0"},
    # The serving session's KV unit (region 0) and slot row (region 1), the
    # target's KV unit over the same memory (region 2) and its slot row (3).
    "trial": {"a1": "0:1", "b1": "2:1", "sa": "1:1", "sc": "3:1"},
}
# The bits of a block's validity in the Rust configuration of each shape: each
# group's memory (the model's `GroupMemory[g] = g`), then the host parts.
MEMORIES = {
    "core": {"ga": "m0", "hp": "m1"},
    "copies": {"ga": "m0", "hp": "m1"},
    "handoff": {"ga": "m0", "gb": "m1", "hp": "m2"},
    "tiers": {"ga": "m0", "hp": "m1"},
    "cross": {"ga": "m0", "gb": "m1", "hp": "m2"},
    "parts": {"ga": "m0", "gb": "m1", "pa": "m2", "pb": "m3"},
    "trial": {"ga": "m0", "gb": "m1", "hp": "m2"},
}
# The host parts each group's ranks copy with in each shape (the model's
# `GroupParts`): the one part in all but `parts`.
GROUP_PARTS = {shape: {"ga": ["hp"], "gb": ["hp"]}
               for shape in ("core", "copies", "handoff", "tiers", "cross", "trial")}
GROUP_PARTS["parts"] = {"ga": ["pa"], "gb": ["pa", "pb"]}
# The enumeration's session is of the epoch 1 (the model's `e1`), its groups
# and owners named under it; in `trial` the target's session is of the epoch
# 2, with the decode group gb as its group and o2 as its owner 1.
GROUPS = {shape: {"ga": "e1.g0", "gb": "e1.g1"} for shape in NAMES}
GROUPS["trial"] = {"ga": "e1.g0", "gb": "e2.g0"}
OWNERS = {shape: {"o1": "e1.1", "o2": "e1.2"} for shape in NAMES}
OWNERS["trial"] = {"o1": "e1.1", "o2": "e2.1"}
MEMORY_BITS = {"ga": 0, "gb": 1}
# The model's slot rows and their groups (`MCSlotGroup` of MC_Pool).
SLOT_GROUPS = {"sa": "ga", "sb": "ga", "sc": "gb", "sd": "gb"}

TOKEN = re.compile(r'\s*(<<|>>|\|->|:>|@@|[{}\[\](),]|"[^"]*"|[A-Za-z0-9_]+)')


def parse(text):
    """A TLA+ value as TLC prints it: sets and tuples become lists, records
    and functions dicts, and everything else a string."""
    tokens = TOKEN.findall(text)
    position = 0

    def value():
        nonlocal position
        token = tokens[position]
        position += 1
        if token == "{":
            return sequence("}")
        if token == "<<":
            return sequence(">>")
        if token == "[":
            record = {}
            while tokens[position] != "]":
                name = tokens[position]
                position += 2
                record[name] = value()
                if tokens[position] == ",":
                    position += 1
            position += 1
            return record
        if token == "(":
            function = {}
            while True:
                argument = value()
                position += 1
                function[argument] = value()
                closing = tokens[position]
                position += 1
                if closing == ")":
                    return function
        return token.strip('"')

    def sequence(closing):
        nonlocal position
        items = []
        while tokens[position] != closing:
            items.append(value())
            if tokens[position] == ",":
                position += 1
        position += 1
        return items

    return value()


# The enumeration's one key, named by its 32 bytes as the Rust buffer pool names keys.
KEY = "01" * 32

# The variables of the model that the projection reads.
READ = {"valid", "ogroup", "deferred", "hcache", "tasks", "free", "copies", "ncopy", "launched", "table",
        "remote", "oslot", "ostate", "cache", "trial", "lost", "vis"}


def epoch_of(name):
    """The epoch a group belongs to: `e1` of `e1.g0`."""
    return name.split(".")[0]


def runs(found):
    """Units as the Rust buffer pool states its free ones: runs as long as
    they can be, `region:first-last` or `region:unit` for one, by region and
    first unit."""
    numbers = sorted(tuple(map(int, item.split(":"))) for item in found)
    joined = []
    for region, number in numbers:
        if joined and joined[-1][0] == region and joined[-1][2] + 1 == number:
            joined[-1][2] = number
        else:
            joined.append([region, number, number])
    return " ".join(f"{region}:{first}" if first == last else f"{region}:{first}-{last}"
                    for region, first, last in joined)


def project(state, shape):
    """The lines of a model state, as the Rust enumeration writes its own."""
    names, memory, parts_of = NAMES[shape], MEMORIES[shape], GROUP_PARTS[shape]
    bits = lambda found: ",".join(sorted((memory[name] for name in found), key=lambda bit: int(bit[1:])))
    unit = lambda name: names[name]
    units = lambda found: " ".join(sorted(unit(name) for name in found))
    groups_of, owners_of = GROUPS[shape], OWNERS[shape]
    group = lambda name: groups_of[name]
    # A lost epoch's session has ended: its groups, their slot rows and its
    # owners are no longer stated.
    lost = set(state["lost"])
    gone = lambda name: epoch_of(name) in lost
    lines = [f"ncopy {state['ncopy']}"]

    launched = state["launched"]
    lines.append("completed " + " ".join(f"{group(name)}={launched[name]}" for name in sorted(launched)
                                         if not gone(group(name))))
    free = [name for name in state["free"] if name not in SLOT_GROUPS or not gone(group(SLOT_GROUPS[name]))]
    lines.append(f"free {runs(unit(name) for name in free)}".rstrip())
    # Each epoch stated: an open session's, one still on trial, or one that
    # made a copy still cached (R8); with its trial and whether its session
    # is closed.
    made = {state["vis"][key]["d"] for key in state["cache"] if state["cache"][key] != "none"}
    made |= {state["vis"][key]["h"] for key in state["hcache"] if state["hcache"][key] != "none"}
    for epoch in sorted(state["trial"]):
        if epoch in lost and state["trial"][epoch] != "yes" and epoch not in made:
            continue
        lines.append(f"epoch {epoch} {'closed' if epoch in lost else 'open'} trial={state['trial'][epoch]}")

    held, hosts = set(), set()
    for owner in sorted(state["ostate"]):
        number = owners_of[owner]
        if gone(number):
            continue
        if state["ostate"][owner] != "live":
            lines.append(f"owner {number} {state['ostate'][owner]}")
            continue
        line = f"owner {number} live {group(state['ogroup'][owner])} slot={unit(state['oslot'][owner])}"
        entry = state["table"][owner]["p1"]
        if entry["u"] != "none":
            line += f" p0={entry['k']}:{unit(entry['u'])}"
            held.add(entry["u"])
        lines.append(line)

    for key in state["cache"]:
        device, host = state["cache"][key], state["hcache"][key]
        remote = int(key in state["remote"])
        if device != "none" or host != "none" or remote:
            shown = lambda name: "-" if name == "none" else unit(name)
            # The epoch on trial each copy is visible to alone (R8), or `-`.
            # A copy that is not there is visible to no one in particular.
            maker = state["vis"][key]
            vd = "-" if device == "none" or maker["d"] == "none" else maker["d"]
            vh = "-" if host == "none" or maker["h"] == "none" else maker["h"]
            lines.append(f"key {KEY} d={shown(device)} h={shown(host)} remote={remote} vd={vd} vh={vh}")
        if device != "none":
            held.add(device)
        if host != "none":
            hosts.add(host)
    # A commit that waits holds its block, whose memories it keeps.
    held |= {block for item in state["deferred"] if item["kind"] == "commit" for block in item["units"]}
    for name in held | hosts:
        memories = " ".join(sorted(memory[found] for found in state["valid"][name]))
        lines.append(f"valid {unit(name)} {memories}".rstrip())

    def copy_line(copy):
        kind = copy["kind"]
        owner = position = memory = "-"
        if kind in ("clone", "load"):
            owner, position = str(owners_of[copy["o"]]), "0"
            memory = f"m{MEMORY_BITS[copy['g']]}"
        elif kind == "restore":
            owner = str(owners_of[copy["o"]])
        elif kind == "save":
            # The owner whose slot it reads, while that owner is there.
            live = state["ostate"][copy["o"]] == "live" and state["oslot"][copy["o"]] == copy["read"]
            owner = str(owners_of[copy["o"]]) if live else "-"
        fill = int(copy["fill"] == "TRUE" and kind in ("load", "store"))
        # The key a copy caches its block under; a clone and a restore cache none.
        key = KEY if kind in ("load", "store", "save") else "-"
        # The host parts a store or a save writes: a store's fill those its
        # host block lacks, otherwise those of its group.
        parts = ""
        if kind == "store" and copy["fill"] == "TRUE":
            parts = bits(copy["parts"])
        elif kind in ("store", "save"):
            parts = bits(parts_of[copy["g"]])
        return (f"kind={kind} o={owner} p={position} read={unit(copy['read'])} "
                f"write={unit(copy['write'])} fill={fill} mem={memory} k={key} parts={parts} "
                f"s={epoch_of(group(copy['g']))}")

    copies = state["copies"]
    lines.extend("copy " + copy_line(copy) for copy in copies)

    def pending(after):
        """The stamps `completed` has not reached, by group."""
        waits = sorted((group(name), int(at)) for name, at in after.items() if int(at) > int(launched[name]))
        return ",".join(f"{group}:{at}" for group, at in waits)

    for item in state["deferred"]:
        kind, after = item["kind"], pending(item["after"])
        if kind == "release":
            # A release waits once per group whose stamp it has not reached.
            for stamp in filter(None, after.split(",")):
                lines.append(f"release after={stamp} {units(item['units'])}")
        elif kind == "commit":
            (block,) = item["units"]
            lines.append(f"commit o={owners_of[item['o']]} p=0 u={unit(block)} k={KEY} after={after}")
        elif kind == "lend":
            lines.append(f"lend o={owners_of[item['o']]} after={after}")
        elif kind == "replicate":
            (block,) = item["units"]
            lines.append(f"replicate o={owners_of[item['o']]} p=0 to={group(item['to'])} u={unit(block)} "
                         f"after={after}")
        elif kind == "finish":
            ok = int(item["ok"] == "TRUE")
            lines.extend(f"finish ok={ok} after={after} {copy_line(copy)}"
                         for copy in copies if copy["id"] == item["c"])

    for task in state["tasks"]:
        expired = int(task["expired"] == "TRUE")
        if task["kind"] == "import":
            # The block each position takes, in the order of the positions (the
            # one position p1 is the Rust buffer pool's 0), and the lent blocks
            # it reads.
            places = " ".join(f"p0={unit(block)}" for _, block in task["places"])
            reads = sorted(unit(name) for name in task["units"] if name not in task["writes"])
            lines.append(" ".join(filter(None, [f"task import o={owners_of[task['o']]}", places,
                                                f"reads={','.join(reads) or '-'}"])))
        elif task["kind"] == "lend":
            lines.append(f"lent o={owners_of[task['o']]}")
        elif task["kind"] == "borrow":
            lines.append(f"borrow o={owners_of[task['o']]} {units(task['units'])}".rstrip())
        elif task["kind"] == "replicate":
            (block,) = task["units"]
            lines.append(f"task replicate o={owners_of[task['o']]} p=0 to={group(task['to'])} u={unit(block)}")
        else:
            # A prefetch writes the host parts of the group that asked for it,
            # of its session.
            prefetch = task["kind"] == "prefetch"
            parts = bits(parts_of[task["to"]]) if prefetch else ""
            asked = epoch_of(group(task["to"])) if prefetch else "-"
            lines.append(f"task {task['kind']} h={unit(task['h'])} expired={expired} k={KEY} parts={parts} "
                         f"s={asked}")
    return "\n".join(sorted({line.rstrip() for line in lines}))


def digest(state):
    return hashlib.md5(state.encode()).digest()


def project_chunk(job):
    """The digests of the projections of some states of the dump."""
    shape, states = job
    found = set()
    for text in states:
        state, variable = {}, None
        for line in text.split("\n"):
            if line.startswith("/\\ "):
                variable, _, value = line[3:].partition(" = ")
                state[variable] = value
            elif variable and line.strip():
                state[variable] += " " + line
        found.add(digest(project({name: parse(value) for name, value in state.items() if name in READ}, shape)))
    return found


def chunks(path, shape, size=2000):
    """The dump's states, each the text between two `State` lines."""
    states, lines = [], []
    with open(path) as dump:
        for line in dump:
            if line.startswith("State "):
                if lines:
                    states.append("".join(lines))
                    lines = []
                if len(states) >= size:
                    yield shape, states
                    states = []
            else:
                lines.append(line)
    if lines:
        states.append("".join(lines))
    if states:
        yield shape, states


def model_states(path, shape):
    """The digests of the projections of every state of the dump."""
    projections = set()
    # $AGREEMENT_JOBS processes project the dump (default: one per CPU).
    with multiprocessing.Pool(int(os.environ.get("AGREEMENT_JOBS", os.cpu_count()))) as workers:
        for found in workers.imap_unordered(project_chunk, chunks(path, shape)):
            projections |= found
    return projections


def rust_states(path):
    states, steps = {}, []
    with open(path) as written:
        blocks = written.read().split("\n\n")
    for block in blocks:
        lines = block.strip("\n").split("\n")
        if lines[0].startswith("state "):
            body = "\n".join(sorted({line.rstrip() for line in lines[1:]}))
            states[int(lines[0].split()[1])] = body
        else:
            steps.extend(tuple(map(int, line.split()[1:])) for line in lines if line.startswith("step "))
    return states, steps


def main():
    shape, dump, written = sys.argv[1:4]
    model = model_states(dump, shape)
    states, steps = rust_states(written)
    missing = [number for number, state in sorted(states.items()) if digest(state) not in model]
    print(f"{shape}: the model has {len(model)} projected states, the Rust buffer pool reaches {len(states)}")
    if missing:
        print(f"{shape}: FAIL: {len(missing)} states of the Rust buffer pool are not states of the model; the first:")
        print(states[missing[0]])
        # A step that leads to it from a state the model has.
        known = set(states) - set(missing)
        for source, target in steps:
            if target == missing[0] and source in known:
                print("--- reached in one step from:")
                print(states[source])
                break
        return 1
    print(f"{shape}: ok: every state the Rust buffer pool reaches is a state of the model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
