# qwen3

The Qwen3 family in Rust, built on `nn` (layers, blocks, weights, capabilities) and `ops`. The crate owns configuration interpretation, checkpoint names, the mathematical layouts and the neural composition of every Qwen3 model UniServe computes; it owns no device, stream, buffer, scheduler or request state (`AGENTS.md`, Model Library Boundary).

## What it holds

- [`config`](src/config.rs): `Config`, one typed configuration per checkpoint, composed of `TextConfig` (the decoder stack), the per-layer `Mixer`, `MoeConfig` (routed experts and an optional shared expert) and `VisionConfig` (the Qwen3-VL tower and its placeholder tokens). It reads `Qwen3MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` and `Qwen3VLForConditionalGeneration` checkpoints.
- [`decoder`](src/decoder.rs): `DecoderLayer` and `DecoderStack`, the one decoder implementation every Qwen3 model composes; the residual stream is the caller's tensor, updated in place, and each layer's intermediates live in its own forward scope.
- [`causal_lm`](src/causal_lm.rs): `Qwen3ForCausalLm`, Qwen3-MoE and the Qwen3.5-MoE language model, implementing `nn::capability::CausalLm` and `nn::family::TextModel`.
- [`mtp`](src/mtp.rs): the Qwen3.5 multi-token-prediction drafter (`nn::family::Drafter`).
- [`vision`](src/vision.rs): `VisionTower`, the Qwen3-VL vision tower with its patch mergers and DeepStack mergers, implementing `nn::capability::Vision`, and how the tower reads pixels (`patch_format`, `pixel_norm`, `open_processor`).
- [`vl`](src/vl.rs): `Qwen3VlModel`, Qwen3-VL as an encoder over a prefix of its layers, with DeepStack, implementing `nn::capability::Encoder`; MiniMax-H3 uses it as its text encoder.
- [`positions`](src/positions.rs): the host arithmetic a forward uploads: M-RoPE coordinates, placeholder rows, the tower's position samples, rotary coordinates and sequence offsets.
The Qwen-VL processor rules that size media to the tower's patch grid (`smart_resize`, frame sampling) and the image block a request's context gives an image are request processing's: `processing::qwen_vl` in `crates/processing`.

- [`family`](src/family.rs): the family registrations `FAMILY` (`Qwen3MoeForCausalLM`) and `HYBRID_FAMILY` (`Qwen3_5MoeForConditionalGeneration`), one implementation behind both.

## Forwards

Every forward computes its host values (positions, packing rows, taps) and places them with `nn::Forward::upload`; their shapes follow from the batch's or layout's counts alone, so a staging forward before each graph replay writes the replay's values where the captured copies read them. Every tensor a forward computes is reserved from the forward; the only caller-provided locations are the capability boundaries the runtime owns (the hidden rows, logits, encoder outputs and caches).

## Boundaries

A model receives borrowed `nn::Forward` resources and tensor views, and every device operation is a call to an `nn` layer. The tensor-parallel and expert-parallel splits are mathematical partitions; the worker creates the communicators, streams, arenas, staging regions and graphs, and schedules the requests.
