#!/usr/bin/env python3
"""Compares the states the Rust buffer pool reaches with the model's (L1).

    agreement.py SHAPE MODEL_DUMP RUST_STATES

MODEL_DUMP is what TLC's -dump wrote for MC_Pool under the shape's
configuration; RUST_STATES is what the crate's enumeration wrote for the
shape. Both are projected onto the same facts, one line each: the units that
are free, each owner's group, slot and table, each key's copies, the
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
import re
import sys

# The names of the model's constants in the Rust configuration of each shape:
# region:unit. Region 0 holds the KV units; each group's slot rows follow; the
# host units are last.
NAMES = {
    "core": {"u1": "0:1", "u2": "0:2", "sa": "1:1", "sb": "1:2"},
    "copies": {"u1": "0:1", "u2": "0:2", "sa": "1:1", "sb": "1:2", "h1": "2:0"},
    "transfer": {"u1": "0:1", "sa": "1:1", "sb": "1:2", "sc": "2:1", "sd": "2:2"},
    "tiers": {"u1": "0:1", "sa": "1:1", "h1": "2:0"},
    "cross": {"u1": "0:1", "sa": "1:1", "sc": "2:1"},
    "parts": {"u1": "0:1", "sa": "1:1", "sc": "2:1", "h1": "3:0"},
}
# The bits of a block's validity in the Rust configuration of each shape: each
# group's memory (the model's `GroupMemory[g] = g`), then the host parts.
MEMORIES = {
    "core": {"ga": "m0", "hp": "m1"},
    "copies": {"ga": "m0", "hp": "m1"},
    "transfer": {"ga": "m0", "gb": "m1", "hp": "m2"},
    "tiers": {"ga": "m0", "hp": "m1"},
    "cross": {"ga": "m0", "gb": "m1", "hp": "m2"},
    "parts": {"ga": "m0", "gb": "m1", "pa": "m2", "pb": "m3"},
}
# The host parts each group's ranks copy with in each shape (the model's
# `GroupParts`): the one part in all but `parts`.
GROUP_PARTS = {shape: {"ga": ["hp"], "gb": ["hp"]} for shape in ("core", "copies", "transfer", "tiers", "cross")}
GROUP_PARTS["parts"] = {"ga": ["pa"], "gb": ["pa", "pb"]}
GROUPS = {"ga": 0, "gb": 1}
OWNERS = {"o1": 1, "o2": 2}

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
        "remote", "oslot", "ostate", "cache"}


def project(state, shape):
    """The lines of a model state, as the Rust enumeration writes its own."""
    names, memory, parts_of = NAMES[shape], MEMORIES[shape], GROUP_PARTS[shape]
    bits = lambda found: ",".join(sorted((memory[name] for name in found), key=lambda bit: int(bit[1:])))
    unit = lambda name: names[name]
    units = lambda found: " ".join(sorted(unit(name) for name in found))
    group = lambda name: f"g{GROUPS[name]}"
    lines = [f"ncopy {state['ncopy']}"]

    launched = state["launched"]
    lines.append("completed " + " ".join(f"{group(name)}={launched[name]}" for name in sorted(launched)))
    lines.append(f"free {units(state['free'])}".rstrip() + ("" if state["free"] else " "))

    held, hosts = set(), set()
    for owner in sorted(state["ostate"]):
        number = OWNERS[owner]
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
            lines.append(f"key {KEY} d={shown(device)} h={shown(host)} remote={remote}")
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
            owner, position = str(OWNERS[copy["o"]]), "0"
            memory = f"m{GROUPS[copy['g']]}"
        elif kind == "restore":
            owner = str(OWNERS[copy["o"]])
        elif kind == "save":
            # The owner whose slot it reads, while that owner is there.
            live = state["ostate"][copy["o"]] == "live" and state["oslot"][copy["o"]] == copy["read"]
            owner = str(OWNERS[copy["o"]]) if live else "-"
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
                f"write={unit(copy['write'])} fill={fill} mem={memory} k={key} parts={parts}")

    copies = state["copies"]
    lines.extend("copy " + copy_line(copy) for copy in copies)

    def pending(after):
        """The stamps `completed` has not reached, by group."""
        waits = sorted((GROUPS[name], int(at)) for name, at in after.items() if int(at) > int(launched[name]))
        return ",".join(f"g{group}:{at}" for group, at in waits)

    for item in state["deferred"]:
        kind, after = item["kind"], pending(item["after"])
        if kind == "release":
            # A release waits once per group whose stamp it has not reached.
            for stamp in filter(None, after.split(",")):
                lines.append(f"release after={stamp} {units(item['units'])}")
        elif kind == "commit":
            (block,) = item["units"]
            lines.append(f"commit o={OWNERS[item['o']]} p=0 u={unit(block)} k={KEY} after={after}")
        elif kind == "transfer":
            lines.append(f"transfer o={OWNERS[item['o']]} to={group(item['to'])} after={after}")
        elif kind == "replicate":
            (block,) = item["units"]
            lines.append(f"replicate o={OWNERS[item['o']]} p=0 to={group(item['to'])} u={unit(block)} "
                         f"after={after}")
        elif kind == "finish":
            ok = int(item["ok"] == "TRUE")
            lines.extend(f"finish ok={ok} after={after} {copy_line(copy)}"
                         for copy in copies if copy["id"] == item["c"])

    for task in state["tasks"]:
        expired = int(task["expired"] == "TRUE")
        if task["kind"] == "transfer":
            line = (f"task transfer o={OWNERS[task['o']]} to={group(task['to'])} slot={unit(task['slot'])} "
                    f"units={units(task['units'])} writes={units(task['writes'])}")
            lines.append(line.rstrip())
        elif task["kind"] == "replicate":
            (block,) = task["units"]
            lines.append(f"task replicate o={OWNERS[task['o']]} p=0 to={group(task['to'])} u={unit(block)}")
        else:
            # A prefetch writes the host parts of the group that asked for it.
            parts = bits(parts_of[task["to"]]) if task["kind"] == "prefetch" else ""
            lines.append(f"task {task['kind']} h={unit(task['h'])} expired={expired} k={KEY} parts={parts}")
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
    with multiprocessing.Pool() as workers:
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
