//! The blocks a worker group runs and how each divides over its ranks.
//!
//! A block is a named part of the model that implements one capability: the
//! text encoder, the denoiser, the video decoder, the language model
//! (`specs/worker.md` §0). A worker group names each block it runs and the
//! ordered ranks that run it (the engine's worker group configuration).
//! A [`BlockConfig`] either forms one model-parallel group over its ranks, of
//! its tensor, pipeline and sequence degrees, or, with a
//! [`BlockDistribution`], runs an independent local copy on every rank and
//! deals independent units among them.
//!
//! [`BlockConfig::validate`] checks membership and parallel degrees for launch
//! and worker registration; process-world bounds belong to the owner of that
//! world. The same value crosses worker IPC in the worker's capability message.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Invalid block membership or parallel degrees.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct BlockConfigError(pub &'static str);

/// Serde default for an omitted degree and for `units_per_rank`.
const fn one() -> usize {
    1
}

/// How a Ulysses block's merged head projections reach each member's heads.
/// Both produce the same values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadProjection {
    /// Every member gathers all members' input rows and projects its own
    /// head shard with head-partitioned weights.
    #[default]
    Gather,
    /// Every member projects its own rows for all heads with whole weights
    /// and sends each member that member's head shard.
    Exchange,
}

/// One mutually exclusive sequence attention strategy.
///
/// Serialized with a `kind` tag (`local`, `ulysses`, `allgather`, `hybrid`);
/// an omitted degree defaults to one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SequenceParallel {
    /// No sequence parallelism.
    #[default]
    Local,
    /// Ulysses sequence parallelism over `ulysses_degree` ranks.
    Ulysses {
        #[serde(default = "one")]
        ulysses_degree: usize,
        /// Head projection mode; an omitted mode gathers.
        #[serde(default)]
        projection: HeadProjection,
        /// Rows each member projects and sends per overlapped exchange step;
        /// absent, the whole shard is exchanged once. Exchange only.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        projection_rows: Option<usize>,
    },
    /// Context parallelism that gathers every context partition's keys and
    /// values onto each of `allgather_degree` ranks.
    Allgather {
        #[serde(default = "one")]
        allgather_degree: usize,
    },
    /// Ulysses and all-gather context parallelism composed; the sequence
    /// degree is the product of both.
    Hybrid {
        #[serde(default = "one")]
        ulysses_degree: usize,
        #[serde(default = "one")]
        allgather_degree: usize,
    },
}

impl SequenceParallel {
    /// Context ranks precede the Ulysses ranks, which vary fastest.
    pub fn dimensions(&self) -> [(&'static str, usize); 2] {
        let (context, ulysses) = match *self {
            Self::Local => (1, 1),
            Self::Ulysses { ulysses_degree, .. } => (1, ulysses_degree),
            Self::Allgather { allgather_degree } => (allgather_degree, 1),
            Self::Hybrid {
                ulysses_degree,
                allgather_degree,
            } => (allgather_degree, ulysses_degree),
        };
        [("cp", context), ("ulysses", ulysses)]
    }

    /// Derives sequence degree from the selected strategy's independent dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`BlockConfigError`] when a degree is zero, the product
    /// overflows, or a Ulysses projection step is zero or steps a gathered
    /// projection.
    pub fn size(&self) -> Result<usize, BlockConfigError> {
        if let Self::Ulysses {
            projection,
            projection_rows: Some(rows),
            ..
        } = *self
            && (rows == 0 || projection != HeadProjection::Exchange)
        {
            return Err(BlockConfigError(
                "projection_rows is a positive step of an exchange projection",
            ));
        }
        let degrees = match *self {
            Self::Local => vec![1],
            Self::Ulysses { ulysses_degree, .. } => vec![ulysses_degree],
            Self::Allgather { allgather_degree } => vec![allgather_degree],
            Self::Hybrid {
                ulysses_degree,
                allgather_degree,
            } => vec![ulysses_degree, allgather_degree],
        };
        checked_product(&degrees)
    }
}

/// Multiplies degrees, rejecting a zero degree and overflow.
fn checked_product(degrees: &[usize]) -> Result<usize, BlockConfigError> {
    degrees.iter().try_fold(1usize, |size, &degree| {
        if degree == 0 {
            return Err(BlockConfigError("parallel degrees must be positive"));
        }
        size.checked_mul(degree)
            .ok_or(BlockConfigError("parallel degree product overflow"))
    })
}

/// Blocks may distribute independent temporal decode units over local decoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockDistribution {
    /// Every member rank runs its own local copy, and independent media
    /// units are dealt to the ranks in `ranks` order, `units_per_rank`
    /// consecutive units each.
    TemporalUnits,
}

/// Where a block keeps its weights while it serves.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockWeights {
    /// Every weight stays in device storage.
    #[default]
    Device,
    /// The dense projection weights of the block's repeated layer stack
    /// stay in page-locked host storage, and each rank copies them into a few
    /// layer-sized device slots ahead of each layer's use. Every other
    /// weight stays in device storage. The rank serves the block's calls
    /// without CUDA graphs.
    Host,
}

/// Rank membership and parallel degrees of one named block in a worker group.
///
/// Without a `distribution`, `ranks` form one model-parallel group and must
/// number exactly [`Self::world_size`]. With a distribution, the parallel world
/// size must be one. Call [`Self::validate`] after parsing and before using
/// the rank-selection methods.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockConfig {
    /// Indices into the worker group's ordered rank list; their order assigns
    /// each rank its position in the parallel degrees or unit schedule. The
    /// deployment file may leave them out, for every rank of the group.
    #[serde(default)]
    pub ranks: Vec<usize>,
    /// Tensor-parallel degree.
    #[serde(default = "one")]
    pub tensor_parallel_size: usize,
    /// Pipeline-parallel degree.
    #[serde(default = "one")]
    pub pipeline_parallel_size: usize,
    /// Sequence attention strategy and its degree.
    #[serde(default)]
    pub sequence_parallel: SequenceParallel,
    /// Consecutive units each rank takes under `distribution`; must be
    /// positive.
    #[serde(default = "one")]
    pub units_per_rank: usize,
    /// Residency of the block's weights; the rank refuses `Host` for a block
    /// without a repeated layer stack. Always written, so the positional
    /// encoding keeps `distribution` last.
    #[serde(default)]
    pub weights: BlockWeights,
    /// How independent units are spread over `ranks`; `None` runs the block
    /// as one model-parallel group.
    ///
    /// An absent distribution is omitted when serialized. This field is
    /// declared last so that a positional encoding, which writes a struct as
    /// its fields in order, loses only its final element and still decodes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distribution: Option<BlockDistribution>,
}

impl BlockConfig {
    /// An undistributed block over `ranks` at degree one in every dimension;
    /// a caller states the degrees its ranks form with struct update syntax.
    pub fn parallel(ranks: Vec<usize>) -> Self {
        Self {
            ranks,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            sequence_parallel: SequenceParallel::Local,
            units_per_rank: 1,
            weights: BlockWeights::Device,
            distribution: None,
        }
    }

    /// Row-major rank order shared by rank layout, numerical meshes and routing.
    pub fn dimensions(&self) -> [(&'static str, usize); 4] {
        let [context, ulysses] = self.sequence_parallel.dimensions();
        [
            ("pp", self.pipeline_parallel_size),
            context,
            ("tp", self.tensor_parallel_size),
            ulysses,
        ]
    }

    /// The rank count one model-parallel group of this block needs: the product
    /// of the tensor, pipeline, and sequence degrees.
    ///
    /// # Errors
    ///
    /// Returns [`BlockConfigError`] when a degree is zero or the product
    /// overflows.
    pub fn world_size(&self) -> Result<usize, BlockConfigError> {
        checked_product(&[
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
            self.sequence_parallel.size()?,
        ])
    }

    /// Check a block's membership and its independent parallel degrees.
    /// Process-world bounds are checked by the caller that owns that world.
    pub fn validate(&self) -> Result<(), BlockConfigError> {
        if self.ranks.is_empty()
            || self.ranks.iter().collect::<HashSet<_>>().len() != self.ranks.len()
        {
            return Err(BlockConfigError("block ranks must be unique and non-empty"));
        }
        if self.units_per_rank == 0 {
            return Err(BlockConfigError("units_per_rank must be positive"));
        }

        let degree = self.world_size()?;
        if self.distribution.is_some() {
            if degree != 1 {
                return Err(BlockConfigError(
                    "temporal unit distribution requires local decoder parallelism",
                ));
            }
        } else if degree != self.ranks.len() {
            return Err(BlockConfigError(
                "block membership must equal TP × sequence × pipeline degrees",
            ));
        }
        Ok(())
    }

    /// Final-stage output ranks with tensor replicas counted once.
    pub fn output_ranks(&self) -> Vec<usize> {
        if self.distribution.is_some() {
            return self.ranks.clone();
        }
        let stage = self.ranks.len() / self.pipeline_parallel_size;
        let ulysses = self.sequence_parallel.dimensions()[1].1;
        self.ranks
            .iter()
            .enumerate()
            .filter_map(|(index, &rank)| {
                (index >= self.ranks.len() - stage
                    && (index / ulysses).is_multiple_of(self.tensor_parallel_size))
                .then_some(rank)
            })
            .collect()
    }

    /// Reports whether every member rank publishes its own copy of the
    /// block's products.
    ///
    /// A distributed block's ranks each publish the units dealt to them, and
    /// sequence-parallel ranks each publish their own shard. A tensor replica
    /// publishes once, from tensor-parallel coordinate zero, and only the
    /// final pipeline stage publishes, so a block with either degree above one
    /// leaves some member ranks without a copy. [`Self::output_ranks`] selects
    /// the publishing ranks by the same rule.
    pub fn publishes_on_every_rank(&self) -> bool {
        self.distribution.is_some()
            || (self.tensor_parallel_size == 1 && self.pipeline_parallel_size == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_replicated_or_pipelined_blocks_leave_ranks_without_a_copy() {
        let block = |tensor_parallel_size, pipeline_parallel_size, sequence_parallel| BlockConfig {
            tensor_parallel_size,
            pipeline_parallel_size,
            sequence_parallel,
            ..BlockConfig::parallel(vec![0, 1, 2, 3])
        };
        let ulysses = |ulysses_degree| SequenceParallel::Ulysses {
            ulysses_degree,
            projection: HeadProjection::Gather,
            projection_rows: None,
        };

        assert!(block(1, 1, ulysses(4)).publishes_on_every_rank());
        assert!(!block(4, 1, SequenceParallel::Local).publishes_on_every_rank());
        assert!(!block(2, 1, ulysses(2)).publishes_on_every_rank());
        assert!(!block(1, 4, SequenceParallel::Local).publishes_on_every_rank());

        let mut distributed = BlockConfig::parallel(vec![0, 1, 2, 3]);
        distributed.distribution = Some(BlockDistribution::TemporalUnits);
        assert!(distributed.publishes_on_every_rank());
    }

    #[test]
    fn a_deployment_names_where_a_block_keeps_its_weights() {
        let host: BlockConfig =
            serde_json::from_str(r#"{"ranks": [0], "weights": "host"}"#).unwrap();
        assert_eq!(host.weights, BlockWeights::Host);
        let device: BlockConfig = serde_json::from_str(r#"{"ranks": [0]}"#).unwrap();
        assert_eq!(device.weights, BlockWeights::Device);
        assert!(
            serde_json::from_str::<BlockConfig>(r#"{"ranks": [0], "weights": "disk"}"#).is_err()
        );
    }

    #[test]
    fn ulysses_selects_its_head_projection() {
        let parse = |text: &str| serde_json::from_str::<SequenceParallel>(text);

        let gathered = parse(r#"{"kind": "ulysses", "ulysses_degree": 8}"#).unwrap();
        assert_eq!(
            gathered,
            SequenceParallel::Ulysses {
                ulysses_degree: 8,
                projection: HeadProjection::Gather,
                projection_rows: None,
            }
        );

        let stepped = parse(
            r#"{"kind": "ulysses", "ulysses_degree": 8, "projection": "exchange", "projection_rows": 4096}"#,
        )
        .unwrap();
        assert_eq!(stepped.size().unwrap(), 8);
        let text = serde_json::to_string(&stepped).unwrap();
        assert_eq!(parse(&text).unwrap(), stepped);

        for invalid in [
            r#"{"kind": "ulysses", "projection": "exchange", "projection_rows": 0}"#,
            r#"{"kind": "ulysses", "projection_rows": 4096}"#,
        ] {
            assert!(parse(invalid).unwrap().size().is_err());
        }
        assert!(parse(r#"{"kind": "ulysses", "projection": "scatter"}"#).is_err());
        assert!(parse(r#"{"kind": "hybrid", "projection": "exchange"}"#).is_err());
        assert!(parse(r#"{"kind": "allgather", "projection": "gather"}"#).is_err());
    }
}
