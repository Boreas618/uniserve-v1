# minimax-h3

MiniMax-H3 and FastH3 in Rust: the model crate of the three tasks `t2va`, `fl2va` and `ref2va`, on the tensor library (`uniserve-tensor`) and `nn`, with the Qwen3-VL text encoder of `crates/models/qwen3`. It holds what a diffusers or vLLM-Omni model file holds: configuration structs read from the checkpoint, a module tree whose `forward`s compose tensor operations and `nn` modules, checkpoint names, and the family value. It depends on `nn`, `uniserve-tensor`, `processing` and `qwen3`, and names no kernel, context, stream or allocation (`AGENTS.md`, Model Library Boundary).

## Public entry

The crate's entry is its family value, `minimax_h3::FAMILY` (an `nn::family::Family`), which claims the diffusers modular pipeline class `MiniMaxH3ModularPipeline`. To execution it is a video family (`nn::family::VideoFamily`): `open` reads and normalizes the checkpoint's sidecars once into a `Config`, and `load` builds the stages a deployment places on a rank from that configuration, the deployment's sequence and tensor splits and its serving bounds, returning an `nn::family::VideoModel`. To request processing it is `processing::minimax_h3::FAMILY`, whose rules (resolutions, frame counts, prompts, prepared conditions) this crate consumes and does not redefine.

## Modules

| Module | Holds |
| --- | --- |
| `config` | `Config` and one struct per numerical submodule: `TransformerConfig`, `ScheduleConfig`, `VsaConfig`, `DenoiserConfig`, `VideoVaeConfig`, `AudioVaeConfig`; the text encoder's is `qwen3::Config` |
| `packing` | The packed sequence of a request as CPU tensor expressions: the document's coordinates, tags, timestep groups and conditioning rows, the dense and the tiled layout, the sparse attention's tile tables |
| `transformer` | `Transformer`, `Block`, `Attention` (dense or video sparse, inside the sequence-parallel projections), `OutputHead` |
| `refiner`, `modulation` | `TokenRefiner`; `Modulation`, which fills the schedule's `nn::modules::TimestepTable` |
| `denoiser` | `Denoiser`, the `nn::Denoiser` capability: layout, schedule, noise, conditioning and the denoising step |
| `video_vae`, `audio_vae`, `splice` | `VideoDecoder`, `VideoEncoder`; `AudioDecoder`, `AudioEncoder`; `VideoPostprocessor`, the `nn::Splicer` |
| `family` | `FAMILY`, the loaded pipeline `MinimaxH3`, and the stages a rank builds |

## Stages

A deployment places stages under the names of its deployment file: `text_encoder` (the Qwen3-VL model at its first 50 decoder layers, an `nn::Encoder`), `transformer` (the network of `t2va` and `fl2va`) or `transformer_ref` (the network of `ref2va`), of which a rank holds one, `video_decoder`, `audio_decoder`, `latent_encoder` (the condition encoders, which load with the denoiser that reads them) and `video_postprocessor` (the clip splice, which holds no weights). A rank builds the stages its tasks need that its group places.

## Checkpoints

A checkpoint is the base release (a pipeline index over `transformer`, `transformer_ref`, `text_encoder`, `processor`, `vae` and `audio_vae`), a FastH3 export (one network whose `fastvideo_inference.json` states its task, its DMD rungs or parallel-decoding blocks, and its sparse attention), or the base release with a FastH3 adapter fused into `transformer` by `nn::adapter`. A checkpoint may span an export and the base release it continues (`nn::Checkpoint::open_roots`): each sidecar is read from the first root that holds it. With no element type stated, each stage keeps the representation its weights are stored in (BF16 for the networks and the text encoder, FP32 for the end projections, the VAE's own for the decoders).

A ModelOpt NVFP4 export of the family (a FastH3 export with a `modelopt_manifest.json` at its root) stores its calibrated projections packed, as `weight_packed`, `weight_scale` and `weight_tensor_scale`: the fused gate/up and the down projection of every block's feed-forward in the network, and the query, key, value, output and feed-forward projections of every block of the video decoder, the latter with their biases. The manifest lists each calibrated projection by its path in the exporter's module tree with the activation magnitude its calibration measured; `config` maps each path to the module its stage stores (`config/modelopt.rs` states the table) and declares the stage's quantization, and `nn::modules::Linear` loads the packed fields and encodes its activations at the manifest's scale (`nn::quantization`). The text encoder and the audio VAE of the released exports are not calibrated and load dense. A manifest that lists a path of another module tree, or a calibration for a projection the stage stores dense, is refused by name.

## Tests

`cargo test -p minimax-h3` runs on CPU tensors and needs no GPU:

- `tests/packing_reference.rs` compares every packed tensor exactly with fixtures of vLLM-Omni's packing (`tests/fixtures/generate_packing.py`).
- `tests/transformer_reference.rs` and `tests/vae_reference.rs` compare the modules over tiny checkpoints with the reference modules run on CPU (`tests/fixtures/generate_transformer.py`, `tests/fixtures/generate_vae.py`).
- `tests/denoiser.rs` steps a tiny denoiser through `nn::Denoiser` in several layouts of one request.
- `tests/denoiser_profile.rs` and `tests/vae_profile.rs` run the stages on the meta device over the released checkpoints' headers and print their kernel keys and allocator peaks; they are ignored by default and run with `-- --ignored --nocapture`.

Each generator states the reference classes it runs and what stands in for the serving framework they import.
