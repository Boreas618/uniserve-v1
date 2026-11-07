# The deployment probes

Two tools behind the text encoder's figures in `notes/h3/deployments.md`. Each builds the checkpoint's
Qwen3-VL text encoder on the meta device from `text_encoder/config.json` with Transformers, so no weight
file is read and no device is needed. They are measurement tools, not deliverables: they print, they
assert nothing, and no code path in the crate calls them.

```
.venv/bin/python crates/models/minimax-h3/tools/deployment/<probe>.py <checkpoint root>
```

`<checkpoint root>` is a MiniMax-H3 export. Nothing defaults to a path on one machine.

| probe | what it measures | rows it produces |
| --- | --- | --- |
| `measure_qwen.py` | the text encoder whole: language model and vision tower | §7's total against the checkpoint's index |
| `measure_qwen_split.py` | the same, split by retained layer count and by the sharded parameter class | §1's `text_encoder` row (50 of 64 layers) and §7's TP1/TP2 figures |

The figures are a count of parameters a component holds, not a file's size. The per-stage sizes of every
deployment are `examples/deployment_sizes.rs`.
