#!/bin/sh
# Check the NCCL a serving process will load, before it starts, and print its directory.
#
# `crates/platform` opens `libnccl.so.2` by name at run time, so nothing at build time decides which
# library a host hands it, and a host's system NCCL can be older than the one this project pins
# (`nvidia-nccl-cu13` in `pyproject.toml`). System NCCL 2.27.7 crashes on the first replay of a graph
# that holds a collective, so an old library shows up as a crash after the model load. This script
# states it before the load, by name.
#
# Usage, from any directory, before `uniserve serve`:
#
#   nccl=$(tools/check_nccl.sh) || exit 1
#   export LD_LIBRARY_PATH=$nccl:$LD_LIBRARY_PATH
#
# `UNISERVE_NCCL_DIR` names the directory that holds `libnccl.so.2`. Unset, the project environment's
# copy answers, which `uv sync --extra gpu` installs, and then the loader's own search. The directory
# is the only thing written to standard output; a refusal goes to standard error with exit status 1.

set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
pin=$(sed -n 's/.*"nvidia-nccl-cu13==\([0-9.]*\).*/\1/p' "$root/pyproject.toml" | head -n 1)
[ -n "$pin" ] || { echo "check_nccl: pyproject.toml pins no nvidia-nccl-cu13" >&2; exit 1; }

directory=${UNISERVE_NCCL_DIR:-}
if [ -z "$directory" ]; then
    for candidate in "$root"/.venv/lib/python*/site-packages/nvidia/nccl/lib; do
        [ -e "$candidate/libnccl.so.2" ] && directory=$candidate
    done
fi
if [ -z "$directory" ]; then
    library=$(ldconfig -p | sed -n 's/.*libnccl\.so\.2 .*=> //p' | head -n 1)
    [ -n "$library" ] || { echo "check_nccl: no libnccl.so.2 on this host; set UNISERVE_NCCL_DIR" >&2; exit 1; }
    directory=$(dirname -- "$library")
fi
[ -e "$directory/libnccl.so.2" ] || { echo "check_nccl: $directory holds no libnccl.so.2" >&2; exit 1; }

python3 - "$directory/libnccl.so.2" "$pin" <<'P' >&2
import ctypes
import sys

library, pin = sys.argv[1:3]
code = ctypes.c_int()
if ctypes.CDLL(library).ncclGetVersion(ctypes.byref(code)) != 0:
    sys.exit(f"check_nccl: {library} does not state its version")
# NCCL's version code is major * 10000 + minor * 100 + patch from 2.9 on.
found = (code.value // 10000, code.value // 100 % 100, code.value % 100)
wanted = tuple(int(part) for part in pin.split("."))
named = ".".join(str(part) for part in found)
if found < wanted:
    sys.exit(
        f"check_nccl: {library} is NCCL {named}, older than the {pin} this project pins; "
        "set UNISERVE_NCCL_DIR to a directory that holds that version"
    )
print(f"check_nccl: {library} is NCCL {named}")
P
echo "$directory"
