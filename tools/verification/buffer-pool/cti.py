#!/usr/bin/env python3
"""Prints Apalache's counterexample to induction: the state before, what the
step changed, and the violated formula.   cti.py OUT_DIR"""
import sys, json, os, glob, re
d = sorted(glob.glob(os.path.join(sys.argv[1], "MC_Apalache.tla", "*", "")))[-1]
j = json.load(open(os.path.join(d, "violation1.itf.json")))
def show(v):
    if isinstance(v, dict):
        if "#set" in v: return "{" + ", ".join(show(x) for x in v["#set"]) + "}"
        if "#map" in v: return "(" + ", ".join(show(k) + ":>" + show(x) for k, x in v["#map"]) + ")"
        if "#tup" in v: return "<<" + ", ".join(show(x) for x in v["#tup"]) + ">>"
        if "#bigint" in v: return v["#bigint"]
        return "[" + ", ".join(f"{k}|->{show(x)}" for k, x in v.items()) + "]"
    return json.dumps(v) if isinstance(v, str) else str(v)
skip = set("Units Slots HostUnits Keys Owners Groups Positions Epochs SlotGroup GroupEpoch OwnerEpoch GroupMemory UnitRegion UnitChunk GroupRegion Rules InitialUnits Movable Losable Hazards TrialEpochs MaxBatch MaxRequests MaxCopies MaxTasks #meta".split())
s0, s1 = j["states"][-2], j["states"][-1]
print("Eager", show(s0.get("Eager")))
for k in sorted(s0):
    if k in skip or k == "Eager": continue
    a, b = show(s0[k]), show(s1[k])
    if a != b: print("CHANGED", k, "\n   from", a[:1500], "\n   to  ", b[:1500])
    elif "dbt" not in k: print("same", k, "=", a[:700])
    else: print("same dbt: entries", re.findall(r'"(\w+)":>\("(\w+)":>\(([^)]*)\)', a)[:0], [m for m in re.findall(r'\[u\|->"(\w+)", w\|->(\w+)\]', a) if m[0] != "none"])
t = open(os.path.join(d, "violation1.tla")).read()
m = re.search(r"InvariantViolation ==(.*?)\n=====", t, flags=re.S)
print("VIOLATION:", " ".join(m.group(1).split())[:1800] if m else None)
