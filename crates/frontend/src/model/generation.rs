//! The generation bounds a checkpoint's architecture allows, which the
//! deployment's engine is configured from before its workers report their own.

use super::ModelConfig;

impl ModelConfig {
    /// Model facts used at startup before the loaded workers' capacities and
    /// served features are applied (`resolve_generation_limits`).
    pub fn generation_limits(
        &self,
        _model_dtype: proto::request::ModelDtype,
    ) -> proto::request::GenerationLimits {
        // No family this daemon serves states a capacity before its workers do.
        proto::request::GenerationLimits::default()
    }
}
