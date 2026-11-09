//! The Qwen3 family as execution loads it ([`nn::family`]).
//!
//! [`FAMILY`] claims the `Qwen3MoeForCausalLM` class. A checkpoint opens to its configuration and
//! its quantization declaration, both read once from its sidecars; a load places the model by the
//! deployment's tensor and expert splits; the loaded [`Qwen3Moe`] states its widths, its cache and
//! its experts to the execution that serves it.

use std::sync::Arc;

use nn::family::{
    CacheShape, ExpertOrder, ExpertShape, Family, TextCheckpoint, TextFamily, TextModel,
    TextPlacement,
};
use nn::{Checkpoint, DType, WeightLoader};

use nn::quantization::Declaration;

use crate::config::MoeConfig;
use crate::moe::Qwen3Moe;

/// The Qwen3 family.
pub const FAMILY: Family = Family::Text(&Qwen3MoeFamily);

/// The family's one value, which [`FAMILY`] refers to.
struct Qwen3MoeFamily;

impl TextFamily for Qwen3MoeFamily {
    fn classes(&self) -> &'static [&'static str] {
        &["Qwen3MoeForCausalLM"]
    }

    fn open(&self, checkpoint: &Checkpoint) -> nn::Result<Box<dyn TextCheckpoint>> {
        let config = MoeConfig::open(checkpoint)?;
        let declaration = Declaration::read(checkpoint)?;
        Ok(Box::new(Qwen3MoeCheckpoint {
            config,
            declaration,
        }))
    }
}

/// A Qwen3-MoE checkpoint: the configuration and the quantization declaration read from it.
struct Qwen3MoeCheckpoint {
    config: MoeConfig,
    declaration: Declaration,
}

impl TextCheckpoint for Qwen3MoeCheckpoint {
    fn expert_layers(&self) -> std::collections::BTreeMap<String, u32> {
        let experts = self.config.num_experts;
        crate::moe::load::expert_modules(&self.config)
            .into_iter()
            .map(|module| (module, experts))
            .collect()
    }

    fn end_of_text(&self) -> Option<&'static str> {
        // Qwen3 states no beginning-of-sequence token: its vocabulary's end of text begins a
        // sequence, and a completion that emits it stops.
        Some("<|endoftext|>")
    }

    fn load(
        &self,
        _checkpoint: &Checkpoint,
        loader: &mut WeightLoader<'_>,
        placement: &TextPlacement,
        experts: &ExpertOrder,
    ) -> nn::Result<Arc<dyn TextModel>> {
        // The checkpoint states its own representation: a calibrated export is read packed and a
        // dense one dense.
        let model = crate::moe::load::load(
            &self.config,
            &self.declaration,
            placement.tensor,
            placement.experts,
            loader,
            experts,
        )?;
        Ok(Arc::new(model))
    }
}

impl TextModel for Qwen3Moe {
    fn hidden(&self) -> u32 {
        self.config().text.hidden_size
    }

    fn vocabulary(&self) -> u32 {
        self.config().text.vocab_size
    }

    fn dtype(&self) -> DType {
        Qwen3Moe::dtype(self)
    }

    fn cache(&self) -> CacheShape {
        let text = &self.config().text;
        CacheShape {
            layers: text.num_hidden_layers,
            // A tensor rank stores the heads its attention holds; the load checked the split
            // divides them.
            kv_heads: text.num_key_value_heads / self.tensor().size,
            head_dim: text.head_dim,
        }
    }

    fn experts(&self) -> Option<ExpertShape> {
        let config = self.config();
        (config.num_experts > 0).then(|| ExpertShape {
            experts: config.num_experts,
            intermediate: config.moe_intermediate_size,
            top: config.num_experts_per_tok,
            encoded: self.experts_encoded(),
        })
    }
}
