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
