//! MiniMax-H3 and FastH3: the video-and-audio diffusion family, as a model file holds it.
//!
//! The module holds what a diffusers or vLLM-Omni model file holds, on the tensor library and
//! [`nn`]:
//!
//! - **configuration** ([`config`]): the checkpoint's sidecars, read and normalized once at the
//!   loading boundary into one typed, immutable struct per numerical submodule;
//! - **the module tree**: the diffusion transformer with its dense or video-sparse attention
//!   ([`transformer`]), the denoiser around it ([`denoiser`]), the video VAE ([`video_vae`]),
//!   the audio VAE with its vocoder and its encoder ([`audio_vae`]) and the clip splice
//!   ([`splice`]). Every `forward` composes tensor operations and `nn` modules;
//! - **the packed sequence** ([`packing`]): the row layout a request's latents, prompt and
//!   conditions are laid out in, as CPU tensor expressions a forward moves to its device;
//! - **the family value** ([`family`]): how a deployment's checkpoint, base release and adapter
//!   ([`adapter`]) compose, which stages ([`stage`]) a deployment places on a rank, and what the
//!   loaded pipeline states to execution.
//!
//! The request side of the family (the resolution and frame rules, the prompt, the prepared
//! conditions) is [`processing::minimax_h3`]'s; a model consumes those values and redefines
//! none. A model holds no runtime object: the execution context a forward computes under is the
//! thread's, installed by execution (`AGENTS.md`, Model Library Boundary).
//!
//! ## References
//!
//! A citation `name:path:lines` names a file of a reference implementation at the commit
//! this table pins, its path below the directory the table states, and the lines a rule is
//! read from:
//!
//! | Name | Repository | Commit | Paths below |
//! | --- | --- | --- | --- |
//! | `vllm-omni` | <https://github.com/vllm-project/vllm-omni> | `9f95053f95b65d072b37a8bbaea9ff8551266e9a` | `vllm_omni/diffusion/` |
//! | `sglang` | <https://github.com/sgl-project/sglang> | `505120f0c3026b198ee0dd81a486389a9f82ce27` | the repository |

pub mod adapter;
pub mod audio_vae;
pub mod config;
pub mod denoiser;
pub mod family;
#[cfg(test)]
pub(crate) mod fixtures;
pub mod modulation;
pub mod packing;
pub mod refiner;
pub mod splice;
pub mod stage;
pub mod transformer;
pub mod video_vae;

pub use config::{
    AudioVaeConfig, Config, DenoiserConfig, ScheduleConfig, TransformerConfig, VideoVaeConfig,
    VsaConfig,
};
pub use family::{FAMILY, MinimaxH3};
pub use processing::minimax_h3::VideoTask;

/// The id this family's refusals name it by ([`nn::Error::Model`]): its registration's.
pub(crate) const ID: nn::ModelId = nn::ModelId(processing::minimax_h3::FAMILY.id);
