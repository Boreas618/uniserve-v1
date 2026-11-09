//! The Qwen3-MoE family as execution loads it ([`nn::family`]).
//!
//! [`FAMILY`] claims the `Qwen3MoeForCausalLM` class. A checkpoint opens to its configuration, and a
//! load places the model by the deployment's tensor and expert splits; the loaded [`Qwen3Moe`]
//! states its widths, its cache and its experts to the execution that serves it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nn::family::{
    CacheShape, ExpertOrder, ExpertShape, Family, TextCheckpoint, TextFamily, TextModel,
    TextPlacement,
};
use nn::{DType, WeightLoader};

use crate::{Config, LoadOptions, Partition, Qwen3Moe};

/// The Qwen3-MoE family.
pub const FAMILY: Family = Family::Text(&Qwen3MoeFamily);

/// The family's one value, which [`FAMILY`] refers to.
struct Qwen3MoeFamily;

impl TextFamily for Qwen3MoeFamily {
    fn classes(&self) -> &'static [&'static str] {
        &["Qwen3MoeForCausalLM"]
    }

    fn open(&self, checkpoint: &Path) -> nn::Result<Box<dyn TextCheckpoint>> {
        let config = Config::read(checkpoint).map_err(model_error)?;

        // Every dimension execution reads is a protocol width, checked once here so that the
        // model's statements below cannot be truncated.
        for (name, value) in [
            ("hidden_size", config.hidden_size),
            ("vocab_size", config.vocab_size),
            ("num_hidden_layers", config.num_hidden_layers),
            ("num_key_value_heads", config.num_key_value_heads),
            ("head_dim", config.head_dim),
            ("num_experts", config.num_experts),
            ("moe_intermediate_size", config.moe_intermediate_size),
            ("num_experts_per_tok", config.num_experts_per_tok),
        ] {
            if u32::try_from(value).is_err() {
                return Err(nn::Error::Checkpoint {
                    path: checkpoint.join("config.json"),
                    problem: format!("{name} of {value} exceeds the protocol's width"),
                });
            }
        }

        Ok(Box::new(Qwen3MoeCheckpoint {
            root: checkpoint.to_path_buf(),
            config,
        }))
    }
}

/// A Qwen3-MoE checkpoint: its root and the configuration read from it.
struct Qwen3MoeCheckpoint {
    root: PathBuf,
    config: Config,
}

impl TextCheckpoint for Qwen3MoeCheckpoint {
    fn expert_layers(&self) -> std::collections::BTreeMap<String, u32> {
        let experts = width(self.config.num_experts);
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
        loader: &mut WeightLoader<'_>,
        placement: &TextPlacement,
        experts: &ExpertOrder,
    ) -> nn::Result<Arc<dyn TextModel>> {
        let partition = Partition::new(
            &self.config,
            placement.tensor.size as usize,
            placement.tensor.rank as usize,
            placement.experts.size as usize,
            placement.experts.rank as usize,
        )
        .map_err(model_error)?;

        // The model's own defaults stand for the page and the attention workspace; a deployment's
        // exchange sizes the expert step's buffers.
        let exchange = placement.exchange.unwrap_or(nn::family::Exchange {
            capacity: 0,
            attention_ranks: 0,
        });
        let options = LoadOptions {
            exchange_capacity: exchange.capacity,
            exchange_attention: exchange.attention_ranks,
            ..LoadOptions::default()
        };

        // The checkpoint states its own representation: a calibrated export is read packed and a
        // dense one dense. The scale layout is the one the serving programs are built for.
        let model = crate::load::load_placed(
            &self.root,
            partition,
            loader,
            options,
            ops::fp4::ScaleLayout::Linear,
            experts,
        )
        .map_err(model_error)?;
        Ok(Arc::new(model))
    }
}

impl TextModel for Qwen3Moe {
    fn hidden(&self) -> u32 {
        width(self.config().hidden_size)
    }

    fn vocabulary(&self) -> u32 {
        width(self.config().vocab_size)
    }

    fn dtype(&self) -> DType {
        Qwen3Moe::dtype(self)
    }

    fn cache(&self) -> CacheShape {
        let config = self.config();
        CacheShape {
            page_tokens: self.page_tokens(),
            layers: width(config.num_hidden_layers),
            // A tensor rank stores the heads its attention holds.
            kv_heads: width(self.partition().local_key_value_heads(config)),
            head_dim: width(config.head_dim),
        }
    }

    fn experts(&self) -> Option<ExpertShape> {
        let config = self.config();
        (config.num_experts > 0).then(|| ExpertShape {
            experts: width(config.num_experts),
            intermediate: width(config.moe_intermediate_size),
            top: width(config.num_experts_per_tok),
            encoded: self.experts_encoded(),
        })
    }
}

/// A configuration dimension as the protocol's width; [`Qwen3MoeFamily::open`] refused every
/// checkpoint whose dimensions do not fit it.
fn width(value: usize) -> u32 {
    value as u32
}

/// The crate's error as the family contract states it.
fn model_error(error: crate::Error) -> nn::Error {
    nn::Error::Model(Box::new(error))
}
