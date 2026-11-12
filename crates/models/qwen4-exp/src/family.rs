//! The family as execution loads it and request processing serves it ([`nn::family`]).
//!
//! A checkpoint opens to its [`Config`], which holds what its sidecars state (its
//! configuration, its quantization declaration and, when it holds the vision tower and the
//! processor files its images are sized by, its processor rules), and a load builds
//! [`Qwen4ExpForCausalLm`] from it.
//!
//! Request processing serves `Qwen4ExpForConditionalGeneration` as
//! [`processing::qwen::QWEN4_EXP`]: reasoning between `<think>` and `</think>`, tool calls as
//! `<function=…>` and `<parameter=…>` tags between `<tool_call>` and `</tool_call>`, and
//! images through the Qwen-VL processor.

use std::collections::BTreeMap;
use std::sync::Arc;

use nn::family::{
    ExpertOrder, Family, Model, TextCheckpoint, TextFamily, TextModel, TextPlacement,
};
use nn::{Checkpoint, WeightLoader};

use crate::Qwen4ExpForCausalLm;
use crate::causal_lm::expert_layers;
use crate::config::Config;

/// The `Qwen4ExpForConditionalGeneration` registration.
pub const FAMILY: Family = Family {
    request: processing::qwen::QWEN4_EXP,
    model: Model::Text(&Qwen4Exp),
};

/// The family: how a checkpoint opens.
struct Qwen4Exp;

impl TextFamily for Qwen4Exp {
    fn open(&self, checkpoint: &Checkpoint) -> nn::Result<Box<dyn TextCheckpoint>> {
        Ok(Box::new(Config::open(checkpoint)?))
    }
}

impl TextCheckpoint for Config {
    fn expert_layers(&self) -> BTreeMap<String, u32> {
        expert_layers(self)
    }

    fn end_of_text(&self) -> Option<&'static str> {
        // The vocabulary states no beginning-of-sequence token: the end of text begins a
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
        // Every later weight is read as the export declares it stored.
        loader.declare(&self.quantization)?;

        Ok(Arc::new(Qwen4ExpForCausalLm::load(
            self,
            loader.root(),
            placement,
            experts,
        )?))
    }
}
