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
let model = minimax_h3::load(&checkpoint, partition, &mut loader)?;
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

## Reading a checkpoint without a device

```sh
cargo run -p minimax-h3 --example inspect_checkpoint -- /opt/uv/models/MiniMax-H3
```

It prints the contract, the DiT partitions and the tasks each serves, the text encoder's retained and
stored depth, both head widths in the checkpoint (the language stack's and the vision tower's), the
video VAE's depth and latent channels, the attention kind, the canvases, and the block name and
activation of every component. It opens no tensor, so it is the cheap first step before a load whose
weights cost the machine.

## The packed layout, and its diagnosis

A request's layout - frames aligned to `17n + 5`, the latent raster, the video and audio rows, the
prompt and condition segments, the tile order, the kept tiles, and every table the denoiser reads -
is host arithmetic in [`packing`](src/packing.rs). Two implementations of that arithmetic exist by design: this one
and today's `uniserve/models/minimax_h3/packing.py`. They are compared, and the comparison is a
diagnosis rather than an acceptance gate (`specs/worker.md` 10):

```sh
.venv/bin/python crates/models/minimax-h3/tools/dump_tables.py \
    --case crates/models/minimax-h3/tools/cases/dense-fl2va.json --out /tmp/dump
cargo run -p minimax-h3 --example compare_blocks -- --dumps /tmp/dump --out /tmp/report.md
```

The five committed cases cover a dense two-keyframe plan, both sparse contracts (`t2va` at tile 64,
`ref2va` at tile 128 with a reference keep), the 480p DMD canvas, the frame rule's floor at 22
frames, and a `ref2va` request whose references carry audio. Every integer table of all five agrees
exactly; the coordinates differ only in the last places of FP64, which is the declared
pairwise-versus-sequential sum of the rotary time series. What a correct report looks like, and how
the dumper must be run for a *recorded* run (in a full environment, without stubbing imports), is in
`notes/h3/numeric-diagnosis.md`.

The same example compares **block tensors** in a second mode, file by file between two dump
directories - one from the reference implementation's shadow pass and one from this crate's traced
views:

```text
cargo run -p minimax-h3 --example compare_blocks -- --reference /tmp/ref --candidate /tmp/ours
```

Integer tensors are compared exactly, floats by their largest absolute and relative difference, and
half-precision tensors are read in the dtype they were computed in (`4` bf16, `5` f16 - the reader
widens, so a trace is not doubled on disk). Nothing here is a gate: a report states what the two
sides computed, and a difference is diagnosed before it is fixed.

## What is not built here yet

- The video decoder, the audio decoder and the condition encoders are not yet capability blocks, so
  `load` refuses a partition that places one, naming it.
- A Ulysses-sharded sparse deployment is refused by the sparse attention layer until the compression
  is composed across members.
