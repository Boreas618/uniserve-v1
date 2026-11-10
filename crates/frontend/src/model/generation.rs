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
        let mut limits = proto::request::GenerationLimits::default();
        // A family that takes images states no bound of its own on their encodings: the
        // engine takes each limit as the smaller of this one and its workers', so the workers'
        // feature storage and encoder cache bound them.
        if self.chat().is_some_and(|profile| profile.images.is_some()) {
            limits.max_vision_feature_bytes = u64::MAX;
            limits.max_vit_grid_tokens = u32::MAX;
            limits.encoder_cache_entries = u32::MAX;
        }
        limits
    }
}
