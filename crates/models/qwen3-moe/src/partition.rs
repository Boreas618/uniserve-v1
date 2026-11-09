//! The mathematical partition of a Qwen3-MoE deployment.
//!
//! A deployment fixes how the model is split: tensor parallelism shards the vocabulary, the
//! attention heads and every dense projection; expert parallelism shards the expert set, so each
//! rank holds a subset and an exchange carries tokens to the experts that own them
//! (specs/worker.md 4.3, 9). The partition is a *mathematical* description the layers apply; it is
//! not a runner, a communicator or a rank layout, and it holds no runtime object (`AGENTS.md`, Model
//! Library Boundary).

use crate::{Error, Result};

/// How one deployment splits the model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Partition {
    /// Tensor-parallel degree: vocabulary, attention heads and dense projections are sharded.
    pub tensor_parallel: usize,
    /// This rank's position in the tensor-parallel group.
    pub tensor_rank: usize,
    /// Expert-parallel degree: each rank owns `num_experts / expert_parallel` experts.
    pub expert_parallel: usize,
    /// This rank's position in the expert-parallel group.
    pub expert_rank: usize,
}

impl Partition {
    /// An unsharded deployment: one rank owns the whole model.
    pub const fn single() -> Self {
        Self {
            tensor_parallel: 1,
            tensor_rank: 0,
            expert_parallel: 1,
            expert_rank: 0,
        }
    }

    /// A partition of `tensor_parallel` ranks and `expert_parallel` ranks, checked against the
    /// model's head count and expert count.
    pub fn new(
        config: &crate::Config,
        tensor_parallel: usize,
        tensor_rank: usize,
        expert_parallel: usize,
        expert_rank: usize,
    ) -> Result<Self> {
        let partition = Self {
            tensor_parallel,
            tensor_rank,
            expert_parallel,
            expert_rank,
        };
        partition.validate(config)?;
        Ok(partition)
    }

    /// Check every degree and rank against the model's dimensions.
    pub fn validate(&self, config: &crate::Config) -> Result<()> {
        if self.tensor_parallel == 0 || self.expert_parallel == 0 {
            return Err(Error::Config(
                "partition degrees must be positive".to_owned(),
            ));
        }
        if self.tensor_rank >= self.tensor_parallel {
            return Err(Error::Config(format!(
                "tensor rank {} is outside a degree of {}",
                self.tensor_rank, self.tensor_parallel
            )));
        }
        if self.expert_rank >= self.expert_parallel {
            return Err(Error::Config(format!(
                "expert rank {} is outside a degree of {}",
                self.expert_rank, self.expert_parallel
            )));
        }
        if !config
            .num_attention_heads
            .is_multiple_of(self.tensor_parallel)
        {
            return Err(Error::Config(format!(
                "tensor parallel degree {} does not divide {} attention heads",
                self.tensor_parallel, config.num_attention_heads
            )));
        }
        if !config
            .num_key_value_heads
            .is_multiple_of(self.tensor_parallel)
        {
            return Err(Error::Config(format!(
                "tensor parallel degree {} does not divide {} key/value heads",
                self.tensor_parallel, config.num_key_value_heads
            )));
        }
        if !config.num_experts.is_multiple_of(self.expert_parallel) {
            return Err(Error::Config(format!(
                "expert parallel degree {} does not divide {} experts",
                self.expert_parallel, config.num_experts
            )));
        }
        Ok(())
    }

    /// Query heads this rank's attention computes.
    pub const fn local_attention_heads(&self, config: &crate::Config) -> usize {
        config.num_attention_heads / self.tensor_parallel
    }

    /// Key/value heads this rank's attention reads.
    pub const fn local_key_value_heads(&self, config: &crate::Config) -> usize {
        config.num_key_value_heads / self.tensor_parallel
    }

    /// The vocabulary shard this rank's embedding and head own, as `(first id, rows)`.
    ///
    /// The shard is `ceil(vocab / degree)` rows except the last, which is short when the
    /// vocabulary does not divide evenly.
    pub fn vocabulary_shard(&self, vocab_size: usize) -> (usize, usize) {
        let rows = vocab_size.div_ceil(self.tensor_parallel);
        let first = self.tensor_rank * rows;
        let rows = rows.min(vocab_size.saturating_sub(first));
        (first, rows)
    }

    /// The width of this rank's vocabulary shard as the head stores and computes it.
    ///
    /// The head multiplies a whole number of eight-row tiles per rank, so a shard shorter than its
    /// tile is stored padded; the vocabulary's own last rank can be short, and then the padding is
    /// what the shard's rows are rounded up to. The same width is the head's output width and the
    /// width of one rank's block in the gathered logits matrix.
    pub fn padded_vocabulary_shard(&self, vocab_size: usize) -> usize {
        self.vocabulary_shard(vocab_size).1.next_multiple_of(8)
    }

    /// The intermediate width of one routed expert on this rank: the tensor group splits every
    /// expert's `moe_intermediate_size` and sums the partial outputs once after the experts, as
    /// the Python worker's tensor-parallel binding does (`uniserve/nn/moe.py`, `FusedMoE`).
    pub const fn local_expert_intermediate(&self, config: &crate::Config) -> usize {
        config.moe_intermediate_size / self.tensor_parallel
    }

    /// The intermediate width of a dense feed-forward layer on this rank.
    pub const fn local_intermediate(&self, config: &crate::Config) -> usize {
        config.intermediate_size / self.tensor_parallel
    }

    /// The experts this rank owns, as a half-open range of the checkpoint's expert ids.
    pub fn expert_range(&self, config: &crate::Config) -> std::ops::Range<usize> {
        let per_rank = config.num_experts / self.expert_parallel;
        let first = self.expert_rank * per_rank;
        first..first + per_rank
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_rank_owns_everything() {
        let partition = Partition::single();
        assert_eq!(partition.vocabulary_shard(151936), (0, 151936));
    }

    #[test]
    fn the_last_vocabulary_shard_is_short() {
        let partition = Partition {
            tensor_parallel: 4,
            tensor_rank: 3,
            expert_parallel: 1,
            expert_rank: 0,
        };
        // 151936 / 4 = 37984 exactly.
        assert_eq!(partition.vocabulary_shard(151936), (113952, 37984));
        // A shard is `ceil(vocabulary / degree)` rows: an uneven vocabulary rounds the width up,
        // and the last rank keeps the remainder. (Such a checkpoint is refused at load, because
        // the head's rows must be its padded width; the rule still has one meaning here.)
        let partition = Partition {
            tensor_rank: 3,
            ..partition
        };
        assert_eq!(partition.vocabulary_shard(151937), (113955, 37982));
    }

    #[test]
    fn expert_ranges_partition_the_experts() {
        let partition = Partition {
            tensor_parallel: 1,
            tensor_rank: 0,
            expert_parallel: 4,
            expert_rank: 2,
        };
        assert_eq!(
            partition.expert_range(&crate::Config {
                vocab_size: 1,
                hidden_size: 1,
                intermediate_size: 1,
                num_hidden_layers: 1,
                num_attention_heads: 1,
                num_key_value_heads: 1,
                head_dim: 2,
                hidden_act: crate::Activation::Silu,
                rms_norm_eps: 1e-6,
                rope_theta: 10_000.0,
                rope_scaling: None,
                max_position_embeddings: 1,
                attention_bias: false,
                tie_word_embeddings: false,
                num_experts: 128,
                num_experts_per_tok: 8,
                moe_intermediate_size: 1,
                norm_topk_prob: true,
                decoder_sparse_step: 1,
                mlp_only_layers: Vec::new(),
            }),
            64..96
        );
    }
}
