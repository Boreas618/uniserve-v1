//! The family as execution loads it and request processing serves it ([`nn::family`]).
//!
//! A checkpoint opens to its [`Config`], which holds what its sidecars state (its
//! configuration and its quantization declaration), and a load builds [`Qwen4ExpForCausalLm`]
//! from it, through the scaffolding every causal language model family shares
//! ([`nn::causal_lm`]).
//!
//! Request processing serves `Qwen4ExpForConditionalGeneration` as
//! [`processing::qwen::QWEN4_EXP`]: reasoning between `<think>` and `</think>`, tool calls as
//! `<function=…>` and `<parameter=…>` tags between `<tool_call>` and `</tool_call>`, and
//! images through the Qwen-VL processor.

use std::sync::Arc;

use nn::causal_lm::{CausalLmConfig, CausalLmFamily, MoeConfig};
use nn::family::{ExpertOrder, Family, Model, TextModel, TextPlacement};
use nn::quantization::Declaration;
use nn::{Checkpoint, ModelId, Result, VarBuilder};

use crate::qwen4_exp::Qwen4ExpForCausalLm;
use crate::qwen4_exp::config::Config;

/// The `Qwen4ExpForConditionalGeneration` registration.
pub const FAMILY: Family = Family {
    request: processing::qwen::QWEN4_EXP,
    model: Model::Text(&CAUSAL_LM),
    step_stack: crate::qwen4_exp::causal_lm::TRANSFORMER,
};

/// The family value the registration loads through.
const CAUSAL_LM: CausalLmFamily<Config> = CausalLmFamily::new();

impl CausalLmConfig for Config {
    const ID: ModelId = crate::qwen4_exp::ID;

    fn open(checkpoint: &Checkpoint) -> Result<Self> {
        Config::open(checkpoint)
    }

    fn quantization(&self) -> &Declaration {
        &self.quantization
    }

    fn layers(&self) -> u32 {
        self.text.num_hidden_layers
    }

    /// Every layer routes.
    fn experts(&self) -> Option<&MoeConfig> {
        Some(&self.moe)
    }

    fn build(
        &self,
        checkpoint: &Checkpoint,
        vb: VarBuilder<'_, '_>,
        placement: &TextPlacement,
        order: &ExpertOrder,
    ) -> Result<Arc<dyn TextModel>> {
        Ok(Arc::new(Qwen4ExpForCausalLm::load(
            self, checkpoint, vb, placement, order,
        )?))
    }
}
