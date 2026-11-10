# qwen3

The Qwen3 family in Rust, built on `nn` (layers, blocks, weights, capabilities) and `ops`. This crate owns configuration interpretation, checkpoint mapping, mathematical layouts and neural composition for every Qwen3 model UniServe computes; it owns no device, no stream, no scheduler and no request state (`AGENTS.md`, Model Library Boundary).

## What it holds

- [`config`](src/config.rs): the typed configurations read once at the loading boundary: `TextConfig` (the decoder stack), `MoeConfig` (the routed experts of `Qwen3MoeForCausalLM`), `VisionConfig` and `VlConfig` (Qwen3-VL).
- [`decoder`](src/decoder.rs): `DecoderLayer` and `DecoderStack`, the one decoder implementation every Qwen3 model composes, whatever its attention, rounding and rotary precision.
- [`moe`](src/moe/mod.rs): `Qwen3MoeForCausalLM` (`Qwen3Moe`), the decoder stack with routed experts under a vocabulary-parallel embedding and head, implementing `nn::capability::CausalLm`.
- [`vl`](src/vl/mod.rs): the Qwen3-VL language model (`Qwen3VlModel`), its vision tower (`VisionTower`), patch mergers, DeepStack features and the host position tables a request decides.
- [`processor`](src/processor.rs): the Qwen-VL processor that sizes media to patch grids, samples video frames and lays pixels out as patch rows.
- [`family`](src/family.rs): the family as execution loads it and request processing serves it.

## Public entry

The crate's entry is its family value, `qwen3::family::FAMILY` (an `nn::family::Family`), which `src/main.rs` registers once: its main processes load the family through the `nn::family::Catalogue` and its daemon serves the family's request side (`Family::request`, a `processing::Family`) through the `processing::Catalogue`, both built from that one list. It claims the `qwen3` and `qwen3_moe` model types and the `Qwen3MoeForCausalLM` class and states both contracts:

- to execution, a text family (`nn::family::TextFamily`): `open` reads the checkpoint's configuration and quantization declaration once from its sidecars, and `load` places the model by the deployment's tensor and expert splits (`nn::family::TextPlacement`), returning an `nn::family::TextModel` that states its widths, its key/value cache and its routed experts;
- to request processing, its chat profile (`processing::ChatProfile`): outputs delimit reasoning between `<think>` and `</think>` and write tool calls as JSON between `<tool_call>` and `</tool_call>`.

The Qwen3-VL model and the processor have no family of their own here: `crates/models/minimax-h3` composes them as its text encoder.

## Boundaries

A model receives borrowed `nn::Forward` resources and tensor views, and every device operation is a call to an `nn` layer. The tensor-parallel and expert-parallel splits a model applies are mathematical partitions; the worker creates the communicators, streams, arenas and graphs, and schedules the requests.
