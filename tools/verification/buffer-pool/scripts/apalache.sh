#!/bin/bash
# Runs Apalache on MC_Apalache with the given arguments of `apalache-mc check`.
#   apalache.sh [ARGUMENT...]
# Apalache reads a module's imports only from the module's own directory, so the
# run reads MC_Apalache and every module it may extend (model/, properties/,
# proof/) as links in one directory of its own, under $APALACHE_WORK (default: a
# temporary directory), which it is run in; a relative `--out-dir` is that
# directory's.
set -u
T=${TLA_TOOLS:-/workspace/dev/tools/tla}
export JAVA_HOME=$T/jdk-21.0.13+9 PATH=$T/jdk-21.0.13+9/bin:$PATH
here=$(cd "$(dirname "$0")/.." && pwd)
work=${APALACHE_WORK:-$(mktemp -d)}
mkdir -p "$work"
ln -sf "$here"/model/*.tla "$here"/properties/*.tla "$here"/proof/*.tla "$here/configs/MC_Apalache.tla" "$work"/
cd "$work"
exec "$T"/apalache/bin/apalache-mc check "$@" MC_Apalache.tla
