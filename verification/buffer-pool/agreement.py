#!/usr/bin/env python3
"""Compares the states the Rust buffer pool reaches with the model's (L1).

    agreement.py SHAPE MODEL_DUMP RUST_STATES

MODEL_DUMP is what TLC's -dump wrote for MC_Pool under the shape's
configuration; RUST_STATES is what the crate's enumeration wrote for the
shape. Both are projected onto the same facts, one line each: the units that
are free, each owner's group, slot and table, each key's copies, the
memories in which each held block is valid, what waits for a stamp, the
copies and the tasks in flight, and the `completed` words.

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
}
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


# The variables of the model that the projection reads.
READ = {"valid", "ogroup", "deferred", "hcache", "tasks", "free", "copies", "ncopy", "launched", "table",
        "remote", "oslot", "ostate", "cache"}


def project(state, names):
    """The lines of a model state, as the Rust enumeration writes its own."""
    unit = lambda name: names[name]
    units = lambda found: " ".join(sorted(unit(name) for name in found))
    group = lambda name: f"g{GROUPS[name]}"
    lines = [f"ncopy {state['ncopy']}"]

    launched = state["launched"]
    lines.append("completed " + " ".join(f"{group(name)}={launched[name]}" for name in sorted(launched)))
    lines.append(f"free {units(state['free'])}".rstrip() + ("" if state["free"] else " "))

    held = set()
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
            lines.append(f"key 01 d={shown(device)} h={shown(host)} remote={remote}")
        if device != "none":
            held.add(device)
    for name in held:
        memories = " ".join(sorted(f"m{GROUPS[memory]}" for memory in state["valid"][name]))
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
        fill = int(copy["fill"] == "TRUE" and kind == "load")
        return (f"kind={kind} o={owner} p={position} read={unit(copy['read'])} "
                f"write={unit(copy['write'])} fill={fill} mem={memory}")

    copies = state["copies"]
    lines.extend("copy " + copy_line(copy) for copy in copies)

    released = set()
    for item in state["deferred"]:
        kind = item["kind"]
        if kind == "release":
            released |= set(item["units"])
        elif kind == "commit":
            (block,) = item["units"]
            lines.append(f"commit o={OWNERS[item['o']]} p=0 u={unit(block)}")
        elif kind == "transfer":
            lines.append(f"transfer o={OWNERS[item['o']]} to={group(item['to'])}")
        elif kind == "finish":
            ok = int(item["ok"] == "TRUE")
            lines.extend(f"finish ok={ok} {copy_line(copy)}" for copy in copies if copy["id"] == item["c"])
    if released:
        lines.append(f"release {units(released)}")

    for task in state["tasks"]:
        expired = int(task["expired"] == "TRUE")
        if task["kind"] == "transfer":
            line = (f"task transfer o={OWNERS[task['o']]} to={group(task['to'])} slot={unit(task['slot'])} "
                    f"units={units(task['units'])} writes={units(task['writes'])}")
            lines.append(line.rstrip())
        else:
            lines.append(f"task {task['kind']} h={unit(task['h'])} expired={expired}")
    return "\n".join(sorted(line.rstrip() for line in lines))


def digest(state):
    return hashlib.md5(state.encode()).digest()


def project_chunk(job):
    """The digests of the projections of some states of the dump."""
    names, states = job
    found = set()
    for text in states:
        state, variable = {}, None
        for line in text.split("\n"):
            if line.startswith("/\\ "):
                variable, _, value = line[3:].partition(" = ")
                state[variable] = value
            elif variable and line.strip():
                state[variable] += " " + line
        found.add(digest(project({name: parse(value) for name, value in state.items() if name in READ}, names)))
    return found


def chunks(path, names, size=2000):
    """The dump's states, each the text between two `State` lines."""
    states, lines = [], []
    with open(path) as dump:
        for line in dump:
            if line.startswith("State "):
                if lines:
                    states.append("".join(lines))
                    lines = []
                if len(states) >= size:
                    yield names, states
                    states = []
            else:
                lines.append(line)
    if lines:
        states.append("".join(lines))
    if states:
        yield names, states


def model_states(path, names):
    """The digests of the projections of every state of the dump."""
    projections = set()
    with multiprocessing.Pool() as workers:
        for found in workers.imap_unordered(project_chunk, chunks(path, names)):
            projections |= found
    return projections


def rust_states(path):
    states, steps = {}, []
    with open(path) as written:
        blocks = written.read().split("\n\n")
    for block in blocks:
        lines = block.strip("\n").split("\n")
        if lines[0].startswith("state "):
            body = "\n".join(sorted(line.rstrip() for line in lines[1:]))
            states[int(lines[0].split()[1])] = body
        else:
            steps.extend(tuple(map(int, line.split()[1:])) for line in lines if line.startswith("step "))
    return states, steps


def main():
    shape, dump, written = sys.argv[1:4]
    model = model_states(dump, NAMES[shape])
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
