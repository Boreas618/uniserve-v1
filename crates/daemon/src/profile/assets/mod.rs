//! Model asset discovery and tokenizer metadata loading.
//!
//! [`model_files`] locates checkpoint files in a local directory, the local
//! Hugging Face Hub cache, or the Hub itself; [`PipelineCheckpoint`] reads
//! the root index that identifies a diffusers pipeline checkpoint and
//! locates its component files, including those a component
//! export draws from the base checkpoint it pins; [`checkpoint`]
//! reads tensor shapes from a safetensors header without loading weights;
//! [`config`] deserializes the JSON metadata that profile resolution in the
//! parent module consumes. Every failure is reported as this module's
//! [`Error`].

pub mod checkpoint;
pub mod config;
pub mod error;
#[cfg(test)]
mod hub_stub;
pub mod model_files;
pub mod pipeline;
mod pipeline_index;

pub use checkpoint::resolve_tensor_shape;
pub use config::{
    DiffusionSamplerConfig, GenerationConfig, HfSpecialTokens, HfTokenizerConfig, ModelConfig,
    NamedSpecialToken, OneOrManyTokenIds, load_generation_config, load_model_config,
    load_tokenizer_config, read_json,
};
pub use error::{Error, Result};
pub use model_files::ResolvedModelFiles;
pub use pipeline::PipelineCheckpoint;
