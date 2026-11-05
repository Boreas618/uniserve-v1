# The deployment probes

The tools behind the memory and parameter figures in `notes/h3/deployments.md`. Each constructs the
**reference implementation in this tree** (`uniserve/models/minimax_h3/`) on the meta device from the
shipped sidecars, with the kernel registry stubbed and the worker library standing in for the Rust rule
it exports, so no weight file is read and no device is needed. They are measurement tools, not
deliverables: they print, they assert nothing, and no code path in the crate calls them.

Run from the repository root, with the tree's interpreter and the package importable:

```
PYTHONPATH=. .venv/bin/python crates/models/minimax-h3/tools/deployment/<probe>.py <checkpoint root>
```

`<checkpoint root>` is a MiniMax-H3 export: `vae/config.json`, `audio_vae/config.json` and
`text_encoder/config.json` under it, plus `transformer/diffusion_pytorch_model.safetensors.index.json`
for `reconcile_index.py`. Nothing defaults to a path on one machine.

| probe | what it measures | rows it produces |
| --- | --- | --- |
| `measure_params.py` | bytes per component at the deployment's representation (BF16), and element counts with `--components` | `deployments.md` §1, the `weights` column of every stage row |
| `measure_shard.py` | how much of a component a tensor-parallel rank does not hold; no root needed | §1's rule for the sharded rows, §6.1's TP2 figures |
| `measure_qwen.py` | the text encoder whole: language model and vision tower | §7's total against the checkpoint's index |
| `measure_qwen_split.py` | the same, split by retained layer count and by the sharded parameter class | §1's `text_encoder` row (50 of 64 layers) and §7's TP1/TP2 figures |
| `reconcile_index.py` | the index's tensor names against the module's declared parameters | §7, the settled index |

The figures are a count of parameters or bytes a component holds, not a file's size: a component whose
checkpoint stores some tensors at another representation differs from its count by exactly those
tensors, which is what §7.1 is about.

`measure_params.py` sets the default dtype to BF16 before constructing anything, so its GiB column is
the note's figure directly. A row printed as `every component together` is a sum and is not a
deployment: a deployment places some of these, and §5 states which.
