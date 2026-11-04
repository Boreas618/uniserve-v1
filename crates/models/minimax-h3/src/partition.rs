//! The mathematical partition of a MiniMax-H3 deployment.
//!
//! A deployment fixes how the model is split. The denoiser is sharded over the packed sequence by
//! Ulysses sequence parallelism: every rank holds a contiguous slice of the sequence's tokens and
//! all of one rank's heads, and an all-to-all turns one into the other around each attention. The
//! text encoder is sharded over the vocabulary, the attention heads and every dense projection by
//! tensor parallelism. The values here are a *mathematical* description the layers apply; they are
//! not a runner, a communicator, a stream or a placement (`AGENTS.md`, Model Library Boundary).
//!
//! A sequence rank's own token slice and the order ranks are gathered in are the runner's business:
//! the model is told how many ranks the exchange runs over, and the layer that shards a tensor
//! reads only the degree and the rank's position in it.

use crate::{Error, Result, config::Config};

/// How one deployment splits the model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Partition {
    /// Ulysses sequence-parallel degree of the denoiser: the heads each rank attends and the
    /// ranks the all-to-all runs over.
    pub ulysses_degree: usize,
    /// Tensor-parallel degree of the text encoder: vocabulary, attention heads and dense
    /// projections are sharded over it.
    pub tensor_parallel: usize,
    /// This rank's position in the text encoder's tensor-parallel group.
    pub tensor_rank: usize,
}

impl Partition {
    /// An unsharded deployment: one rank holds the whole model.
    pub const fn single() -> Self {
        Self {
            ulysses_degree: 1,
            tensor_parallel: 1,
            tensor_rank: 0,
        }
    }

    /// A deployment's partition, checked against the checkpoint's head counts.
    pub fn new(
        config: &Config,
        ulysses_degree: usize,
        tensor_parallel: usize,
        tensor_rank: usize,
    ) -> Result<Self> {
        let partition = Self {
            ulysses_degree,
            tensor_parallel,
            tensor_rank,
        };
        partition.validate(config)?;
        Ok(partition)
    }

    /// Check every degree and rank against the model's dimensions.
    ///
    /// The denoiser's Ulysses degree splits the attention heads between the ranks of the
    /// exchange, so it must divide every DiT partition's head count and the group must be a power
    /// of two, which the exchange layer also refuses. The text encoder's tensor degree splits the
    /// query heads and the key/value heads, so it must divide both.
    pub fn validate(&self, config: &Config) -> Result<()> {
        if self.ulysses_degree == 0 || self.tensor_parallel == 0 {
            return Err(Error::config(
                "partition degrees must be positive".to_owned(),
            ));
        }
        if !self.ulysses_degree.is_power_of_two() {
            return Err(Error::config(format!(
                "the Ulysses degree {} is not a power of two",
                self.ulysses_degree
            )));
        }
        if self.tensor_rank >= self.tensor_parallel {
            return Err(Error::config(format!(
                "tensor rank {} is outside a degree of {}",
                self.tensor_rank, self.tensor_parallel
            )));
        }
        for (dit, denoiser) in &config.denoisers {
            let heads = denoiser.transformer.num_attention_heads as usize;
            if heads % self.ulysses_degree != 0 {
                return Err(Error::config(format!(
                    "the Ulysses degree {} does not divide the {} attention heads of {}",
                    self.ulysses_degree,
                    heads,
                    dit.directory()
                )));
            }
        }
        let text = &config.text_encoder;
        if text.num_attention_heads as usize % self.tensor_parallel != 0 {
            return Err(Error::config(format!(
                "the tensor-parallel degree {} does not divide the text encoder's {} query heads",
                self.tensor_parallel, text.num_attention_heads
            )));
        }
        if text.num_key_value_heads as usize % self.tensor_parallel != 0 {
            return Err(Error::config(format!(
                "the tensor-parallel degree {} does not divide the text encoder's {} key/value \
                 heads",
                self.tensor_parallel, text.num_key_value_heads
            )));
        }
        Ok(())
    }

    /// The attention heads one rank of the denoiser's exchange attends.
    pub const fn denoiser_heads(&self, heads: u32) -> u32 {
        heads / self.ulysses_degree as u32
    }

    /// The query heads one tensor-parallel rank of the text encoder holds.
    pub const fn text_query_heads(&self, config: &Config) -> u32 {
        config.text_encoder.num_attention_heads / self.tensor_parallel as u32
    }

    /// The key/value heads one tensor-parallel rank of the text encoder holds.
    pub const fn text_key_value_heads(&self, config: &Config) -> u32 {
        config.text_encoder.num_key_value_heads / self.tensor_parallel as u32
    }

    /// The vocabulary rows one tensor-parallel rank of the text encoder holds, as the half-open
    /// range `[first, stop)`.
    ///
    /// The rows are split into `tensor_parallel` contiguous shards, the last taking the remainder;
    /// the head pads its shard to a multiple of the descriptor's row alignment, and the gathered
    /// logits drop the padding.
    pub fn vocabulary_shard(&self, vocab_size: u32) -> (u32, u32) {
        let degree = self.tensor_parallel as u32;
        let rows = vocab_size.div_ceil(degree);
        let first = rows * self.tensor_rank as u32;
        (first, (first + rows).min(vocab_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released checkpoint's dimensions.
    fn config() -> Config {
        crate::config::tests_support::released_config()
    }

    #[test]
    fn an_unsharded_partition_holds_the_whole_model() {
        let config = config();
        let partition = Partition::single();
        partition.validate(&config).expect("one rank");
        assert_eq!(partition.denoiser_heads(56), 56);
        assert_eq!(partition.text_query_heads(&config), 64);
        assert_eq!(partition.text_key_value_heads(&config), 8);
        assert_eq!(partition.vocabulary_shard(151_936), (0, 151_936));
    }

    #[test]
    fn four_way_ulysses_and_text_tensor_parallelism_split_the_heads() {
        let config = config();
        let partition = Partition::new(&config, 4, 4, 2).expect("four ranks");
        assert_eq!(partition.denoiser_heads(56), 14);
        assert_eq!(partition.text_query_heads(&config), 16);
        assert_eq!(partition.text_key_value_heads(&config), 2);
    }

    #[test]
    fn eight_way_ulysses_splits_fifty_six_heads_into_seven() {
        let config = config();
        let partition = Partition::new(&config, 8, 8, 0).expect("eight ranks");
        assert_eq!(partition.denoiser_heads(56), 7);
        assert_eq!(partition.tensor_rank, 0);
    }

    #[test]
    fn a_partition_that_does_not_divide_the_heads_is_refused() {
        let config = config();
        let error = Partition::new(&config, 3, 4, 0).expect_err("three is not a power of two");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("power of two")),
            "{error}"
        );
        let error = Partition::new(&config, 5, 1, 0).expect_err("five does not divide 56");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("Ulysses")),
            "{error}"
        );
        let error = Partition::new(&config, 1, 3, 0).expect_err("three does not divide 64 heads");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("query heads")),
            "{error}"
        );
        let error = Partition::new(&config, 1, 1, 1).expect_err("rank outside the degree");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("outside")),
            "{error}"
        );
    }

    #[test]
    fn the_vocabulary_shard_gives_the_last_rank_the_remainder() {
        let partition = Partition {
            ulysses_degree: 1,
            tensor_parallel: 3,
            tensor_rank: 2,
        };
        assert_eq!(partition.vocabulary_shard(151_936), (101_292, 151_936));
        let partition = Partition {
            ulysses_degree: 1,
            tensor_parallel: 2,
            tensor_rank: 0,
        };
        assert_eq!(partition.vocabulary_shard(151_936), (0, 75_968));
        // A vocabulary the degree does not divide leaves the last shard short.
        let partition = Partition {
            ulysses_degree: 1,
            tensor_parallel: 4,
            tensor_rank: 3,
        };
        assert_eq!(partition.vocabulary_shard(10), (9, 10));
    }
}
