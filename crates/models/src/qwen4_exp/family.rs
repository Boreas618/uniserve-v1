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
use proto::config::Quantization;

use crate::qwen4_exp::Qwen4ExpForCausalLm;
use crate::qwen4_exp::causal_lm::expert_layers;
use crate::qwen4_exp::config::Config;

/// The `Qwen4ExpForConditionalGeneration` registration.
pub const FAMILY: Family = Family {
    request: processing::qwen::QWEN4_EXP,
    model: Model::Text(&Qwen4Exp),
    step_stack: crate::qwen4_exp::causal_lm::TRANSFORMER,
};

/// The family: how a checkpoint opens.
struct Qwen4Exp;

impl TextFamily for Qwen4Exp {
    fn open(
        &self,
        checkpoint: &Checkpoint,
        quantization: &Quantization,
    ) -> nn::Result<Box<dyn TextCheckpoint>> {
        // The family serves each checkpoint as its export declares it, BF16 or quantized.
        nn::quantization::own_representation("qwen4_exp", quantization)?;

        Ok(Box::new(Config::open(checkpoint)?))
    }
}

impl TextCheckpoint for Config {
    fn expert_layers(&self) -> BTreeMap<String, u32> {
        expert_layers(self)
    }

    fn load(
        &self,
        checkpoint: &Checkpoint,
        loader: &mut WeightLoader<'_>,
        placement: &TextPlacement,
        experts: &ExpertOrder,
    ) -> nn::Result<Arc<dyn TextModel>> {
        // Every later weight is read as the export declares it stored.
        loader.declare(&self.quantization)?;

        Ok(Arc::new(Qwen4ExpForCausalLm::load(
            self,
            checkpoint,
            loader.root(),
            placement,
            experts,
        )?))
    }
}
