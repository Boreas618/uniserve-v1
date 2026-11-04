//! The generation bounds a checkpoint's architecture allows, which the
//! deployment's engine is configured from before its workers report their own
//! (`specs/worker.md` §4.2).

use models::omni::GenerationControls;
use proto::request::GenerationLimits;

use super::{BagelProfile, ModelConfig, ModelParameters};

impl ModelConfig {
    pub fn generation_controls(&self) -> Option<&models::omni::GenerationControls> {
        match &self.parameters {
            ModelParameters::SenseNova(value) => Some(&value.controls),
            ModelParameters::Bagel(value) => Some(&value.controls),
            ModelParameters::Qwen3
            | ModelParameters::MiniMaxH3
            | ModelParameters::DiffusionGemma(_) => None,
        }
    }

    /// Model facts used at startup before the loaded workers' capacities and
    /// served features are applied (`resolve_generation_limits`).
    pub fn generation_limits(
        &self,
        model_dtype: proto::request::ModelDtype,
    ) -> proto::request::GenerationLimits {
        match &self.parameters {
            ModelParameters::Qwen3 | ModelParameters::MiniMaxH3 => {
                proto::request::GenerationLimits {
                    latent_downsample: 1,
                    max_cfg_branches: 1,
                    ..Default::default()
                }
            }
            ModelParameters::SenseNova(profile) => profile.runtime_limits(model_dtype),
            ModelParameters::Bagel(_) => BagelProfile::runtime_limits(model_dtype),
            // A readout prefills text and encoded images and denoises its
            // canvases once over them. An image writes one cache entry per soft
            // token; the worker's encoder entries bound the feature bytes and
            // the number of cached encodings, so the model states no bound of
            // its own for either.
            ModelParameters::DiffusionGemma(profile) => proto::request::GenerationLimits {
                latent_downsample: 1,
                max_vit_grid_tokens: profile.images.max_soft_tokens,
                max_vision_feature_bytes: u64::MAX,
                max_cfg_branches: 1,
                encoder_cache_entries: u32::MAX,
                ..Default::default()
            },
        }
    }
}
