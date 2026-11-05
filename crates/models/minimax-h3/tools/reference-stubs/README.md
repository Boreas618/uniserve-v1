# Import stubs for the reference's host-only arithmetic

The diagnosis producers under `tools/` read the *reference* implementation's arithmetic, and several
of its modules import `uniserve_kernels` - the compiled package - at load time. Where that package is
not built, the import fails before a single value is computed, even though the code these producers
reach calls no kernel at all.

These two modules are that import made satisfiable and nothing more: every name resolves to a
function that raises, so a producer that keeps to host arithmetic runs, and a producer that reaches a
kernel fails with a message that says the kernels are missing rather than returning a number.

```bash
PYTHONPATH=.:crates/models/minimax-h3/tools/reference-stubs \
    .venv/bin/python crates/models/minimax-h3/tools/dump_encoder_tables.py \
    --case case.json --out ref
```

**Never put this directory on the path of a run that must compute**: a serving process, a fixture
build, or anything that installs programs. It is for the diagnosis producers alone, and a build host
has the real package and does not need it.
