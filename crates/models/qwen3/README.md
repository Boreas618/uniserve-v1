# qwen3

The Qwen3 family in Rust, on the tensor library (`uniserve-tensor`) and the neural-network library (`nn`). The crate holds what a vLLM or candle model file holds: the configuration read from a checkpoint, a module tree whose `forward`s compose tensor operations and `nn` modules, the checkpoint's names, and the family values. It owns no device, stream, buffer, scheduler or request state (`AGENTS.md`, Model Library Boundary), and it depends on `nn`, `uniserve-tensor` and `processing` only.

## What it holds

- [`config`](src/config.rs): `Config`, one typed configuration per checkpoint, composed of `TextConfig` (the decoder stack, its layer types and its gated delta network), `MoeConfig` (routed experts and an optional shared expert) and `VisionConfig` (the Qwen3-VL tower and its placeholder tokens), with the quantization declaration and the processor rules the checkpoint's other sidecars state. It reads `Qwen3MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` and `Qwen3VLForConditionalGeneration` checkpoints.
- [`decoder`](src/decoder.rs): `DecoderLayer`, the one decoder layer every Qwen3 model composes: softmax attention or a gated delta network, then a gated MLP or routed experts with a shared expert, in vLLM's `(hidden, residual)` form.
- [`causal_lm`](src/causal_lm.rs): `Qwen3ForCausalLm`, Qwen3-MoE and the Qwen3.5-MoE language model, implementing `nn::CausalLm` and `nn::family::TextModel`.
- [`mtp`](src/mtp.rs): `Mtp`, the Qwen3.5 multi-token-prediction drafter (`nn::Drafter`).
- [`vision`](src/vision.rs): `VisionTower` with its `VisionBlock`s and `PatchMerger`s (the final merger and the DeepStack mergers), implementing `nn::Vision`, and how the tower reads pixels (`patch_format`).
- [`vl`](src/vl.rs): `Qwen3VlModel`, Qwen3-VL as an encoder over a prefix of its layers, with DeepStack, implementing `nn::Encoder`; MiniMax-H3 uses it as its text encoder.
- [`positions`](src/positions.rs): the position tables of a forward as CPU tensor expressions: M-RoPE coordinates, placeholder rows, the tower's learned-position taps and weights, rotary coordinates and sequence offsets.
- [`family`](src/family.rs): the family registrations `FAMILY` (`Qwen3MoeForCausalLM`) and `HYBRID_FAMILY` (`Qwen3_5MoeForConditionalGeneration`), one implementation behind both.

The Qwen-VL processor rules that size media to the tower's patch grid (`smart_resize`, frame sampling) and the block a request's context gives an image are request processing's: `processing::qwen_vl` in `crates/processing`.

## Types

Every type is a configuration read from the checkpoint, a module with a `forward`, the family value or the error.

| Type | Kind |
| --- | --- |
| `Config`, `TextConfig`, `MoeConfig`, `VisionConfig` | configuration |
| `DecoderLayer`, `Qwen3ForCausalLm`, `Mtp`, `VisionBlock`, `PatchMerger`, `VisionTower`, `Qwen3VlModel` | module |
| `family::Qwen3` | family value |
| `Error` | error |

## Forwards

A forward computes on tensors: on CPU tensors it computes its formulas, on a device it launches kernels through the execution context execution installed, and on meta tensors it states the kernels it would launch, which is a profile run. The values a forward needs beyond its inputs (positions, row indices, sequence offsets) are CPU tensors moved with `to_device` to the device of the tensor they meet; their shapes follow from the batch's or layout's counts alone, so every request of one shape replays one captured graph. The only tensors a forward writes without having made them are the ones execution lends for that purpose: key/value pages, recurrent state, an image's feature buffer and an encoder's output rows.

## Tests

- `cargo test -p qwen3` runs the unit tests (configuration parsing, positions) and the reference vectors of [`tests/reference.rs`](tests/reference.rs): tiny random checkpoints under [`tests/data`](tests/data), loaded through the family onto the CPU device, against outputs transformers computed from the same weights in PyTorch. [`tests/generate_reference.py`](tests/generate_reference.py) writes the fixtures.
- `cargo test -p qwen3 -- --ignored` runs the profile runs of [`tests/profile.rs`](tests/profile.rs) on the meta device over the headers of the released checkpoints under `/workspace/models`.
