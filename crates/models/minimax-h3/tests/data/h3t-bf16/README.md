# A BF16 `H3D1` container, written by the reference producer

Two files for the same six numbers: `stack_out.h3t` (element code 4, the halves' bits as little-endian
`u16`) and `stack_out.npy` (the same tensor widened to FP32), both written by
`crates/models/minimax-h3/tools/dump_tables.py`'s `dump` - the writer the reference side uses, not a
writer this crate owns. Committed so that `tools/compare_stack.py` can prove its own reader against the
**other** implementation's bytes with NumPy alone, on a host that has neither a device nor the reference
model.

The values are `[[0.5, -2.25, 0.0], [3.0, 0.0625, -7.5]]`, each already exact in BF16, so the expected
FP32 reading is the value itself: a reader that rounds or shifts the halves wrongly fails on `0.0625`
before it fails on anything else.

## Regenerating it

The fixture is a **record of the reference writer's output**, not a statement of the `H3D1` format, and
it is replaced by running the writer rather than by editing bytes:

```bash
.venv/bin/python - <<'PY'
import importlib.util, pathlib, torch
spec = importlib.util.spec_from_file_location("dump_tables", "crates/models/minimax-h3/tools/dump_tables.py")
dump_tables = importlib.util.module_from_spec(spec); spec.loader.exec_module(dump_tables)
values = torch.tensor([[0.5, -2.25, 0.0], [3.0, 0.0625, -7.5]], dtype=torch.bfloat16)
dump_tables.dump(pathlib.Path("crates/models/minimax-h3/tests/data/h3t-bf16"), "stack_out", values)
PY
```

A change to the writer that changes what it writes shows up here as a two-file diff, which is the point
of committing bytes rather than describing them: the diff is the review, and `--self-test` fails
against the old pair until it is regenerated.
