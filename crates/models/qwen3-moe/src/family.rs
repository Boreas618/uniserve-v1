//! The Qwen3-MoE family as execution loads it ([`nn::family`]).
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

use crate::representation::Representation;
use crate::{Config, LoadOptions, Partition, Qwen3Moe};

/// The Qwen3-MoE family.
pub const FAMILY: Family = Family::Text(&Qwen3MoeFamily);

/// The family's one value, which [`FAMILY`] refers to.
struct Qwen3MoeFamily;

impl TextFamily for Qwen3MoeFamily {
    fn classes(&self) -> &'static [&'static str] {
        &["Qwen3MoeForCausalLM"]
    }

    fn open(&self, checkpoint: &Checkpoint) -> nn::Result<Box<dyn TextCheckpoint>> {
        let config = Config::open(checkpoint)?;
        let declaration = Representation::read(checkpoint.root())?;
        Ok(Box::new(Qwen3MoeCheckpoint {
            config,
            declaration,
        }))
    }
}

/// A Qwen3-MoE checkpoint: the configuration and the quantization declaration read from it.
struct Qwen3MoeCheckpoint {
    config: Config,
    declaration: Representation,
}

impl TextCheckpoint for Qwen3MoeCheckpoint {
    fn expert_layers(&self) -> std::collections::BTreeMap<String, u32> {
        let experts = self.config.num_experts;
        crate::weights::expert_sources(&self.config)
            .into_iter()
            .map(|source| (source.module, experts))
            .collect()
    }

    fn end_of_text(&self) -> Option<&'static str> {
        // Qwen3 states no beginning-of-sequence token: its vocabulary's end of text begins a
        // sequence, and a completion that emits it stops.
        Some("<|endoftext|>")
    }

    fn load(
        &self,
        checkpoint: &Checkpoint,
        loader: &mut WeightLoader<'_>,
        placement: &TextPlacement,
        experts: &ExpertOrder,
    ) -> nn::Result<Arc<dyn TextModel>> {
        let partition = Partition::new(
            &self.config,
            placement.tensor.size,
            placement.tensor.rank,
            placement.experts.size,
            placement.experts.rank,
        )?;

        // The model's own defaults stand for the page and the attention workspace.
        let options = LoadOptions::default();

        // The checkpoint states its own representation: a calibrated export is read packed and a
        // dense one dense.
        let model = crate::load::load(
            &self.config,
            &self.declaration,
            checkpoint,
            partition,
            loader,
            options,
            experts,
        )?;
        Ok(Arc::new(model))
    }
}

impl TextModel for Qwen3Moe {
    fn hidden(&self) -> u32 {
        self.config().hidden_size
    }

    fn vocabulary(&self) -> u32 {
        self.config().vocab_size
    }

    fn dtype(&self) -> DType {
        Qwen3Moe::dtype(self)
    }

    fn cache(&self) -> CacheShape {
        let config = self.config();
        CacheShape {
            page_tokens: self.page_tokens(),
            layers: config.num_hidden_layers,
            // A tensor rank stores the heads its attention holds.
            kv_heads: self.partition().local_key_value_heads(config),
            head_dim: config.head_dim,
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
