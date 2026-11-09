# minimax-h3

MiniMax-H3 in Rust: the model library for the three tasks `t2va`, `fl2va` and `ref2va`, built on
`nn` (layers, weights, capabilities) and `ops`. This crate owns configuration interpretation,
checkpoint classification, the packed layout of a request, the model blocks, and the host tables the
runner writes to a device; it owns no device, no stream, no scheduler and no request state
(`AGENTS.md`, Model Library Boundary).

## Loading

```rust
let checkpoint = nn::Checkpoint::open(&root)?;          // or open_roots(&[&export, &base])
let partition = Partition::single().placing(Components::ALL.without(Component::TransformerRef));
let model = minimax_h3::load(&checkpoint, partition, None, &canvases, &mut loader)?;
```

`load` reads the checkpoint's own statements, normalizes them into a [`Config`](src/config) once at
this boundary, and builds the blocks the [`Partition`](src/partition.rs) places, each registered under the name its runner
addresses. With `dtype = "auto"` nothing is converted at load: each block keeps the representation
its weights are stored in (BF16 for the DiT and the text encoder, FP16 for the video VAE decoder), and
each layer declares the precision it computes in.

Block names are capabilities, not directories: `text_encoder`, `denoiser`, `video_decoder`,
`audio_decoder`, `latent_encoder`. Both DiT partitions (`transformer` for `t2va`/`fl2va`,
`transformer_ref` for `ref2va`) are the one `denoiser` block, and a rank holds exactly one of them.
A block this build cannot construct is refused by name at load rather than omitted.

A checkpoint may span an export and the base release it continues. The engine opens the export first
(the earlier root shadows), and every sidecar is read from the component that holds it, so the text
encoder, the processor and the VAEs resolve through the base while the DiT and schedulers come from
the export.
