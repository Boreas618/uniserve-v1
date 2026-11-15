//! The Qwen3 family as execution loads it and request processing serves it ([`nn::family`]).
//!
//! One implementation serves every class the family registers: a checkpoint opens to its
//! [`Config`], which holds what its sidecars state (its configuration and its quantization
//! declaration), and a load builds [`Qwen3ForCausalLm`] from it, through the scaffolding every
//! causal language model family shares ([`nn::causal_lm`]). Two registrations state what
//! request processing serves of each class, because their chat profiles differ:
//!
//! - [`FAMILY`]: `Qwen3MoeForCausalLM`, whose outputs delimit reasoning between `<think>` and
//!   `</think>` and write tool calls as JSON between `<tool_call>` and `</tool_call>`;
//! - [`HYBRID_FAMILY`]: `Qwen3_5MoeForConditionalGeneration` (Qwen3.6-35B-A3B), which reasons
//!   the same way and takes images; its tool calls are `<function=…>` and `<parameter=…>` tags
//!   between `<tool_call>` and `</tool_call>`.

use std::sync::Arc;

use nn::causal_lm::{CausalLmConfig, CausalLmFamily, MoeConfig};
use nn::family::{ExpertOrder, Family, Model, TextModel, TextPlacement};
use nn::quantization::Declaration;
use nn::{Checkpoint, ModelId, Result, VarBuilder};

use crate::qwen3::Qwen3ForCausalLm;
use crate::qwen3::config::Config;

/// The `Qwen3MoeForCausalLM` registration.
pub const FAMILY: Family = Family {
    request: processing::qwen::QWEN3,
    model: Model::Text(&CAUSAL_LM),
    step_stack: crate::qwen3::causal_lm::TRANSFORMER,
};

/// The `Qwen3_5MoeForConditionalGeneration` registration.
pub const HYBRID_FAMILY: Family = Family {
    request: processing::qwen::QWEN3_5_MOE,
    model: Model::Text(&CAUSAL_LM),
    step_stack: crate::qwen3::causal_lm::TRANSFORMER,
};

/// The family value both registrations load through.
const CAUSAL_LM: CausalLmFamily<Config> = CausalLmFamily::new();

impl CausalLmConfig for Config {
    const ID: ModelId = crate::qwen3::ID;

    fn open(checkpoint: &Checkpoint) -> Result<Self> {
        Config::open(checkpoint)
    }

    fn quantization(&self) -> &Declaration {
        &self.quantization
    }

    fn layers(&self) -> u32 {
        self.text.num_hidden_layers
    }

    fn experts(&self) -> Option<&MoeConfig> {
        self.moe.as_ref()
    }

    fn build(
        &self,
        checkpoint: &Checkpoint,
        vb: VarBuilder<'_, '_>,
        placement: &TextPlacement,
        order: &ExpertOrder,
    ) -> Result<Arc<dyn TextModel>> {
        Ok(Arc::new(Qwen3ForCausalLm::load(
            self, checkpoint, vb, placement, order,
        )?))
    }
}
