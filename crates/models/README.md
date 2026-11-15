# models

The model families in Rust, one module each, on the tensor library (`tensor`) and the neural-network library (`nn`). A family holds what a vLLM, candle or diffusers model file holds: the configuration read from a checkpoint, a module tree whose `forward`s compose tensor operations and `nn` modules, the checkpoint's names, and the family values execution loads and request processing serves. It owns no device, stream, buffer, scheduler or request state and names no kernel, context, stream or allocation (`AGENTS.md`, Model Library Boundary); the crate depends on `nn`, `tensor` and `processing` only.

| Module | Family |
| --- | --- |
| `qwen3` | The Qwen3 decoder, `Qwen3MoeForCausalLM`, the language model of `Qwen3_5MoeForConditionalGeneration` with its MTP drafter, and Qwen3-VL as an encoder with its vision tower |
| `qwen4_exp` | Qwen3.8 Flash Next (`Qwen4ExpForConditionalGeneration`), on the vision tower and configuration readers of `qwen3` |
| `minimax_h3` | MiniMax-H3 and FastH3, whose text encoder is the Qwen3-VL model of `qwen3` |

Every type is a configuration read from the checkpoint, a module with a `forward`, a family value or a family's error.

## Forwards

A forward computes on tensors: on CPU tensors it computes its formulas, on a device it launches kernels through the execution context execution installed, and on meta tensors it states the kernels it would launch, which is a profile run. The values a forward needs beyond its inputs (positions, row indices, sequence offsets) are CPU tensors moved with `to_device` to the device of the tensor they meet; their shapes follow from the batch's or layout's counts alone, so every request of one shape replays one captured graph. The only tensors a forward writes without having made them are the ones execution lends for that purpose: key/value pages, recurrent state, an image's feature buffer and an encoder's output rows.

## `qwen3`

- [`config`](src/qwen3/config.rs): `Config`, one typed configuration per checkpoint, composed of `TextConfig` (the decoder stack, its layer types and its gated delta network), `MoeConfig` (routed experts and an optional shared expert) and `VisionConfig` (the Qwen3-VL tower and its placeholder tokens), with the quantization declaration and the processor rules the checkpoint's other sidecars state. It reads `Qwen3MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` and `Qwen3VLForConditionalGeneration` checkpoints.
- [`decoder`](src/qwen3/decoder.rs): `DecoderLayer`, the one decoder layer every Qwen3 model composes: softmax attention or a gated delta network, then a gated MLP or routed experts with a shared expert, in vLLM's `(hidden, residual)` form.
- [`causal_lm`](src/qwen3/causal_lm.rs): `Qwen3ForCausalLm`, Qwen3-MoE and the Qwen3.5-MoE language model, implementing `nn::CausalLm` and `nn::family::TextModel`.
- [`mtp`](src/qwen3/mtp.rs): `Mtp`, the Qwen3.5 multi-token-prediction drafter (`nn::Drafter`).
- [`vision`](src/qwen3/vision.rs): `VisionTower` with its `VisionBlock`s and `PatchMerger`s (the final merger and the DeepStack mergers), implementing `nn::Vision`, and how the tower reads pixels (`patch_format`).
- [`vl`](src/qwen3/vl.rs): `Qwen3VlModel`, Qwen3-VL as an encoder over a prefix of its layers, with DeepStack, implementing `nn::Encoder`; MiniMax-H3 uses it as its text encoder.
- [`positions`](src/qwen3/positions.rs): the position tables of a forward as CPU tensor expressions: M-RoPE coordinates, placeholder rows, the tower's learned-position taps and weights, rotary coordinates and sequence offsets.
- [`family`](src/qwen3/family.rs): the family registrations `FAMILY` (`Qwen3MoeForCausalLM`) and `HYBRID_FAMILY` (`Qwen3_5MoeForConditionalGeneration`), one implementation behind both.

The Qwen-VL processor rules that size media to the tower's patch grid (`smart_resize`, frame sampling) and the block a request's context gives an image are request processing's: `processing::qwen_vl` in `crates/processing`.

## `qwen4_exp`

The configuration (`Config`), the hyper-connections around each block (`HyperConnection`), the hashed n-gram embedding and its PLE layer (`NgramEmbedding`, `PleLayer`), indexed attention (`IndexedAttention`), the decoder layer (`DecoderLayer`), the causal language model (`Qwen4ExpForCausalLm`, an `nn::CausalLm`), the drafter over streams (`Mtp`, an `nn::Drafter`) and the family value `FAMILY`, following vLLM's implementation of the family. The vision tower and the M-RoPE coordinates of a prompt are `qwen3`'s.

## `minimax_h3`

The family of the three tasks `t2va`, `fl2va` and `ref2va`. Its entry is its family value, `minimax_h3::FAMILY` (an `nn::family::Family`), which claims the diffusers modular pipeline class `MiniMaxH3ModularPipeline`. To execution it is a video family (`nn::family::VideoFamily`): `open` composes the deployment's checkpoint, base release and adapter into one checkpoint and reads and normalizes its sidecars once into a `Config`, and `load` builds the stages a deployment places on a rank from that configuration, the deployment's sequence and tensor splits and its serving bounds, returning an `nn::family::VideoModel`. To request processing it is `processing::minimax_h3::FAMILY`, whose rules (resolutions, frame counts, prompts, prepared conditions) this family consumes and does not redefine.

| Module | Holds |
| --- | --- |
| `config` | `Config` and one struct per numerical submodule: `TransformerConfig`, `ScheduleConfig`, `VsaConfig`, `DenoiserConfig`, `VideoVaeConfig`, `AudioVaeConfig`; the text encoder's is `qwen3::Config`; the pipeline's components (`Component`) and a FastH3 student's contract (`Contract`) |
| `adapter` | `FastVideoLora`, FastVideo's adapter container, as `nn::adapter::Format` |
| `packing` | The packed sequence of a request as CPU tensor expressions: the document's coordinates, tags, timestep groups and conditioning rows, the dense and the tiled layout, the sparse attention's tile tables |
| `transformer` | `Transformer`, `Block`, `Attention` (dense or video sparse, inside the sequence-parallel projections), `OutputHead` |
| `refiner`, `modulation` | `TokenRefiner`; `Modulation`, which fills the schedule's `nn::modules::TimestepTable` |
| `denoiser` | `Denoiser`, the `nn::Denoiser` capability: layout, schedule, noise, conditioning and the denoising step |
| `video_vae`, `audio_vae`, `splice` | `VideoDecoder`, `VideoEncoder`; `AudioDecoder`, `AudioEncoder`; `VideoPostprocessor`, the `nn::Splicer` |
| `family` | `FAMILY`, the composition of a deployment's sources (`compose`), the loaded pipeline `MinimaxH3`, and the stages a rank builds |

### Stages

A deployment places stages under the names of its deployment file: `text_encoder` (the Qwen3-VL model at its first 50 decoder layers, an `nn::Encoder`), `transformer` (the network of `t2va` and `fl2va`) or `transformer_ref` (the network of `ref2va`), of which a rank holds one, `video_decoder`, `audio_decoder`, `latent_encoder` (the condition encoders, which load with the denoiser that reads them) and `video_postprocessor` (the clip splice, which holds no weights and is placed wherever `video_decoder` is). A rank builds the stages its tasks need that its group places.

### Checkpoints

A deployment names a checkpoint, and may name the base release it continues and an adapter directory (`nn::Sources`); `family::compose` composes them by what the checkpoint declares, each component (`config::Component`) from exactly one source, its directory the one the checkpoint's pipeline index declares. A checkpoint is the base release (no contract: the networks its pipeline index declares, among `transformer` and `transformer_ref`, beside `text_encoder`, `processor`, `vae`, `audio_vae` and the two schedulers), a FastH3 export (its contract, `fastvideo_inference.json` read by `config::Contract`, states its network, its DMD rungs or parallel-decoding blocks, its attention, and the base release it continues when it holds only its network and schedulers), or the base release with a FastH3 adapter (FastVideo's `fastvideo-lora-v2` container, `adapter::FastVideoLora`) fused into the network the contract shipped beside it names. With no element type stated, each stage keeps the representation its weights are stored in (BF16 for the networks and the text encoder, FP32 for the end projections, the VAE's own for the decoders).

A ModelOpt NVFP4 export of the family (a FastH3 export with a `modelopt_manifest.json` at its root) stores its calibrated projections packed, as `weight_packed`, `weight_scale` and `weight_tensor_scale`: the fused gate/up and the down projection of every block's feed-forward in the network, and the query, key, value, output and feed-forward projections of every block of the video decoder, the latter with their biases. The manifest lists each calibrated projection by its path in the exporter's module tree with the activation magnitude its calibration measured; `config` maps each path to the module its stage stores (`config/modelopt.rs` states the table) and declares the stage's quantization, and `nn::modules::Linear` loads the packed fields and encodes its activations at the manifest's scale (`nn::quantization`). The text encoder and the audio VAE of the released exports are not calibrated and load dense. A manifest that lists a path of another module tree, or a calibration for a projection the stage stores dense, is refused by name.

## Tests

`cargo test -p models` runs the unit tests and the integration tests of [`tests/it`](tests/it), one module per family, on CPU tensors without a GPU. Each reference test loads a family through its own loading path and compares it with outputs the reference implementation computed from the same weights. A test embeds its committed fixtures in its binary (`nn_testing::fixture!`, from paths relative to the test's source file) and reads them from a temporary directory with the readers the families load with, a reference's tensors through `nn::Checkpoint::open_file`; what the tests share (checkpoint writing, released checkpoints, meta forwards and their checks) is the development crate [`nn-testing`](../nn/testing). Each generator states the reference classes it runs.

The generators of [`python`](python) are console programs of one project, which imports each reference as a package pinned to one revision by its own lock (`python/uv.lock`): its `reference` group holds vLLM-Omni, the vLLM release it is written against and their Transformers. Each generator writes under `--out`, runs on `--device` and draws from `--seed` under `torch.use_deterministic_algorithms(True)` (`model_fixtures.run`); the committed fixtures are those of the default arguments on the CPU. The environment is installed apart from the root project's, whose `test` extra pins another PyTorch:

```bash
UV_PROJECT_ENVIRONMENT=<directory> uv sync --locked --project crates/models/python --group reference
<directory>/bin/qwen3-fixtures --out crates/models/tests/it/qwen3/data
<directory>/bin/minimax-h3-packing-fixtures --out crates/models/tests/it/minimax_h3/fixtures/packing
```

- `qwen3::reference`: tiny random checkpoints under [`tests/it/qwen3/data`](tests/it/qwen3/data), loaded onto the CPU device, against outputs transformers computed in PyTorch (`qwen3-fixtures` writes the fixtures).
- `qwen4_exp::reference`: the family's modules and the whole model against the references of [`generate_reference.py`](tests/it/qwen4_exp/generate_reference.py), and prefill against steps.
- `minimax_h3::packing_reference` compares every packed tensor exactly with fixtures of vLLM-Omni's packing (`minimax-h3-packing-fixtures`); `minimax_h3::transformer_reference` and `minimax_h3::vae_reference` compare the modules over tiny checkpoints with the reference modules run on CPU (`fixtures/generate_transformer.py`, `fixtures/generate_vae.py`); `minimax_h3::denoiser` steps a tiny denoiser through `nn::Denoiser` in several layouts of one request.

The profile runs (`qwen3::profile`, `qwen4_exp::profile`, `minimax_h3::denoiser_profile`, `minimax_h3::vae_profile`) run on the meta device over the headers of released checkpoints and print their kernel keys and allocator peaks. Each release is a Hugging Face repository pinned at a commit ([`tests/it/releases.rs`](tests/it/releases.rs)):

| Repository | Commit | Read by |
| --- | --- | --- |
| `Qwen/Qwen3-30B-A3B` | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | `qwen3::profile` |
| `nvidia/Qwen3-30B-A3B-NVFP4` | `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3` | `qwen3::profile` |
| `Qwen/Qwen3.6-35B-A3B` | `345d467b117d1fd5eda9b879a5988de9ae5f47a7` | `qwen3::profile` |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | `1355db6a052410cfd62085d94b58866fd0f2c3c5` | `qwen3::profile` |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | `fc694b54fb0174e0913e6adf86691ef85a4ead47` | `qwen4_exp::profile` |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` | `qwen4_exp::profile` |
| `Qwen/Qwen3.8-Flash-Next-FP8` | `236dfdf285828023ca3bcd3f37366c58a3469b13` | `qwen4_exp::profile` |
| `MiniMaxAI/MiniMax-H3` | `42ed227ee7df40d41602854ae760620d6eb651fe` | `qwen3::profile`, `minimax_h3::denoiser_profile`, `minimax_h3::vae_profile` |
| `FastVideo/FastVideo-FastH3-8-Step-V2` | `3da2ddfe1954d9cda4c05b643dc0f26007a655c5` | `minimax_h3::denoiser_profile` |
| `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` | `5ea076f35b84da4c3c82217112fa733d8eea2ae1` | `minimax_h3::denoiser_profile` |
| `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4` | `3e7ccea5cb400e802cdb8aa1848707a6b0c82b06` | `minimax_h3::denoiser_profile`, `minimax_h3::vae_profile` |

A run resolves its releases through the standard Hugging Face cache (`nn_testing::Release`): `HF_HUB_CACHE`, or `HF_HOME/hub`, or `~/.cache/huggingface/hub`, with `HF_ENDPOINT` for a mirror and `HF_TOKEN` (or the token file under `HF_HOME`) for a gated repository. Online, the files of the pinned commit the cache lacks are downloaded into it first; a host that sets `HF_HUB_OFFLINE=1` reads the cache alone, populated beforehand with `hf download <repository> --revision <commit>`. A release that cannot be resolved fails the run, naming the repository and the commit. The runs are ignored by default and run with `cargo test -p models --test it -- --ignored --nocapture`, or one family's with its module path as the filter.
