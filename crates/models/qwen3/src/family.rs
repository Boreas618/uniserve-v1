//! The Qwen3 family as execution loads it and request processing serves it ([`nn::family`]).
//!
//! One implementation serves every class the family registers: a checkpoint opens to its
//! [`Config`], its quantization declaration and, when it holds a vision tower and the processor
//! files its images are sized by, its processor rules; a load builds [`Qwen3ForCausalLm`]. Two
//! registrations state what request processing serves of each class, because their chat profiles
//! differ:
//!
//! - [`FAMILY`]: `Qwen3MoeForCausalLM`, whose outputs delimit reasoning between `<think>` and
//!   `</think>` and write tool calls as JSON between `<tool_call>` and `</tool_call>`;
//! - [`HYBRID_FAMILY`]: `Qwen3_5MoeForConditionalGeneration` (Qwen3.6-35B-A3B), which reasons the
//!   same way and takes images; its tool calls are `<function=…>` and `<parameter=…>` tags
//!   between `<tool_call>` and `</tool_call>`.

use std::collections::BTreeMap;
use std::sync::Arc;

use nn::capability::PixelNorm;
use nn::family::{
    ExpertOrder, Family, Model, TextCheckpoint, TextFamily, TextModel, TextPlacement,
};
use nn::quantization::Declaration;
use nn::{Checkpoint, WeightLoader};

use crate::Qwen3ForCausalLm;
use crate::causal_lm::expert_layers;
use crate::config::Config;
use processing::qwen_vl::ProcessorConfig;

/// The `Qwen3MoeForCausalLM` registration.
pub const FAMILY: Family = Family {
    request: processing::qwen::QWEN3,
    model: Model::Text(&Qwen3),
};

/// The `Qwen3_5MoeForConditionalGeneration` registration.
pub const HYBRID_FAMILY: Family = Family {
    request: processing::qwen::QWEN3_5_MOE,
    model: Model::Text(&Qwen3),
};

/// The family: how a checkpoint opens.
struct Qwen3;

/// A tensor every Qwen3-VL and Qwen3.5 vision tower holds: the patch embedding.
const TOWER_WEIGHT: &str = "model.visual.patch_embed.proj.weight";

impl TextFamily for Qwen3 {
    fn open(&self, checkpoint: &Checkpoint) -> nn::Result<Box<dyn TextCheckpoint>> {
        let config = Config::open(checkpoint)?;
        // The tower serves when the checkpoint holds both its weights and its processor files;
        // otherwise the checkpoint serves text alone.
        let images = match checkpoint.tensor(TOWER_WEIGHT) {
            Some(_) => crate::vision::open_processor(checkpoint)?,
            None => None,
        };
        Ok(Box::new(Opened {
            declaration: Declaration::read(checkpoint)?,
            config,
            images,
        }))
    }
}

/// An opened checkpoint: what its files state, read once.
struct Opened {
    config: Config,
    declaration: Declaration,
    /// The processor rules of a checkpoint whose vision tower serves.
    images: Option<(ProcessorConfig, PixelNorm)>,
}

impl TextCheckpoint for Opened {
    fn expert_layers(&self) -> BTreeMap<String, u32> {
        expert_layers(&self.config)
    }

    fn end_of_text(&self) -> Option<&'static str> {
        // Qwen vocabularies state no beginning-of-sequence token: the end of text begins a
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
        Ok(Arc::new(Qwen3ForCausalLm::load(
            &self.config,
            &self.declaration,
            placement,
            loader,
            experts,
            self.images,
        )?))
    }
}
