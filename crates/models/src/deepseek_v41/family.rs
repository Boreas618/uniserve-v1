//! The family as execution loads it and request processing serves it ([`nn::family`]).
//!
//! A checkpoint opens to its [`Config`], which holds what its sidecars state (its configuration
//! and its block-scaled quantization declaration), and a load builds [`DeepseekV41ForCausalLm`]
//! from it, through the scaffolding every causal language model family shares
//! ([`nn::causal_lm`]). Request processing serves `DeepseekV41ForCausalLM` as
//! [`processing::deepseek_v41::FAMILY`].

use std::collections::BTreeMap;
use std::sync::Arc;

use nn::causal_lm::{CausalLmConfig, CausalLmFamily, MoeConfig};
use nn::family::{ExpertOrder, Family, Model, TextModel, TextPlacement};
use nn::quantization::Declaration;
use nn::{Checkpoint, ModelId, Result, VarBuilder};

use crate::deepseek_v41::causal_lm::{DeepseekV41ForCausalLm, TRANSFORMER};
use crate::deepseek_v41::config::Config;
use crate::deepseek_v41::decoder::expert_module;

/// The `DeepseekV41ForCausalLM` registration.
pub const FAMILY: Family = Family {
    request: processing::deepseek_v41::FAMILY,
    model: Model::Text(&CAUSAL_LM),
    step_stack: TRANSFORMER,
};

/// The family value the registration loads through.
const CAUSAL_LM: CausalLmFamily<Config> = CausalLmFamily::new();

impl CausalLmConfig for Config {
    const ID: ModelId = crate::deepseek_v41::ID;

    fn open(checkpoint: &Checkpoint) -> Result<Self> {
        Config::open(checkpoint)
    }

    fn quantization(&self) -> &Declaration {
        &self.quantization
    }

    fn layers(&self) -> u32 {
        self.num_hidden_layers
    }

    /// The routed experts are DeepSeek's (`ffn.experts`), not transformers' sparse MoE block:
    /// the layers that route are stated by [`Self::expert_layers`].
    fn experts(&self) -> Option<&MoeConfig> {
        None
    }

    /// Every layer routes over its experts, named by their module path.
    fn expert_layers(&self) -> BTreeMap<String, u32> {
        (0..self.num_hidden_layers)
            .map(|layer| (expert_module(layer), self.experts.experts))
            .collect()
    }

    fn build(
        &self,
        checkpoint: &Checkpoint,
        vb: VarBuilder<'_, '_>,
        placement: &TextPlacement,
        order: &ExpertOrder,
    ) -> Result<Arc<dyn TextModel>> {
        Ok(Arc::new(DeepseekV41ForCausalLm::load(
            self, checkpoint, vb, placement, order,
        )?))
    }
}
