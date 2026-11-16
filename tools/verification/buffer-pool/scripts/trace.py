#!/usr/bin/env python3
"""Prints a TLC counterexample as its actions with the variables each changed.

    trace.py TLC_LOG
"""
import re
import sys

text = open(sys.argv[1]).read()
states = re.split(r"^State (\d+): ", text, flags=re.M)[1:]
previous = {}
for number, body in zip(states[::2], states[1::2]):
    header, _, rest = body.partition("\n")
    rest = rest.split("\n\n")[0]
    values = {}
    for part in re.split(r"^/\\ ", rest, flags=re.M)[1:]:
        name, _, value = part.partition(" = ")
        values[name.strip()] = " ".join(value.split())
    action = re.sub(r" line .*", ">", header)
    print(f"{number}: {action}")
    for name, value in values.items():
        if previous and previous.get(name) != value:
            print(f"     {name} = {value}")
    previous = values
