//! The generation bounds a checkpoint's architecture allows, which the
//! deployment's engine is configured from before its workers report their own
//! (`specs/worker.md` §4.2).

use super::ModelConfig;

impl ModelConfig {
    /// Model facts used at startup before the loaded workers' capacities and
    /// served features are applied (`resolve_generation_limits`).
    pub fn generation_limits(
        &self,
        _model_dtype: proto::request::ModelDtype,
    ) -> proto::request::GenerationLimits {
        // Both lower bounds are divisors and counts that admit nothing at zero;
        // no family this daemon serves states a capacity before its workers do.
        proto::request::GenerationLimits {
            latent_downsample: 1,
            max_cfg_branches: 1,
            ..Default::default()
        }
    }
}
