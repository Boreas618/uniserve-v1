# minimax-h3

MiniMax-H3 and FastH3 in Rust: the model library for the three tasks `t2va`, `fl2va` and `ref2va`, built on `nn` (layers, weights, capabilities) and `ops`, with the Qwen3-VL text encoder of `crates/models/qwen3`. This crate owns configuration interpretation, checkpoint classification, the mathematical partition of a deployment, the packed layout of a request, the model blocks, the host tables a runner writes to a device, and the family's request processing; it owns no device, no stream, no scheduler and no request state (`AGENTS.md`, Model Library Boundary).

## Public entry

The crate's entry is its family value, `minimax_h3::family::FAMILY` (an `nn::family::Family`), which the engine executable (`crates/engine/src/main.rs`) registers once: its main processes load the family through the `nn::family::Catalogue`. The daemon serves the family's request side (`Family::request`, the `processing::Family` that `processing::minimax_h3::FAMILY` states) through the `processing::Registry` of `processing::FAMILIES`, and links no model. It claims the diffusers modular pipeline class `MiniMaxH3ModularPipeline` and states both sides:

- to execution, a video family (`nn::family::VideoFamily`): `open` reads and normalizes the checkpoint's sidecars once into a [`Config`](src/config/mod.rs), and `load` builds the blocks the deployment places on a rank from that configuration, its [`Partition`](src/partition.rs) (the denoiser's Ulysses sequence split and the text encoder's tensor-parallel position) and its serving bounds, returning an `nn::family::VideoModel`;
- to request processing, its video processor (`minimax_h3::processing`), which holds a request to the model's canvas, frame and condition rules over the family-neutral rules of `crates/processing`.

With `dtype = "auto"` nothing is converted at load: each block keeps the representation its weights are stored in (BF16 for the DiT and the text encoder, FP16 for the video VAE decoder), and each layer declares the precision it computes in.

## Blocks

A deployment places blocks under the component names of its deployment file: `text_encoder`, `transformer` (the DiT of `t2va` and `fl2va`), `transformer_ref` (the DiT of `ref2va`), `video_decoder`, `audio_decoder`, `latent_encoder` and `video_postprocessor`. Each is a block implementing an `nn` capability: the text encoder is an `nn::capability::Encoder`, the placed DiT an `nn::capability::Denoiser` (a rank holds exactly one of the two networks), the video and audio decoders `nn::capability::Decoder`s, and the video postprocessor (`output::VideoPostprocessor`, the clip splice, which holds no weights) an `nn::capability::Splicer`. A block this build cannot construct is refused by name at load rather than omitted.

## Checkpoints

A checkpoint may span an export and the base release it continues (`nn::Checkpoint::open_roots`). The export is opened first (the earlier root shadows), and every sidecar is read from the component that holds it, so the text encoder, the processor and the VAEs resolve through the base while the DiT and schedulers come from the export.
