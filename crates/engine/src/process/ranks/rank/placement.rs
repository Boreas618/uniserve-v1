//! A rank's place in its deployment's parallel shape.

use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use nn::Split;
use nn::family::{
    DenoiserSplit, GATHER_AXIS, LAYER_AXIS, Model, Projection, SequenceSplit, TENSOR_AXIS,
    TextPlacement, ULYSSES_AXIS, VideoCheckpoint,
};
use proto::config::{BlockConfig, EngineArgs, HeadProjection, WorkerGroupConfig};

use super::ModelRunner;

impl ModelRunner {
    /// This rank's place in the denoiser's partition and the text encoder's tensor split, as
    /// its group's blocks state them for its checkpoint ([`video_split`]).
    ///
    /// # Errors
    ///
    /// A checkpoint that does not open, or a split the deployment's groups do not state.
    pub(crate) fn partition(&self) -> Result<(DenoiserSplit, Split)> {
        let opened = self.open_video_checkpoint()?;
        self.video_split(&*self.open_video_family(&opened)?)
    }

    /// This rank's video checkpoint, opened over its roots: the checkpoint the launch names and the
    /// base release it continues, with the deployment's adapter fused into the block the
    /// family says it edits.
    pub(super) fn open_video_checkpoint(&self) -> Result<nn::Checkpoint> {
        let model = &self.engine_config().model;
        let mut roots = vec![self.checkpoint().to_path_buf()];
        if let Some(base) = &model.base {
            roots.push(PathBuf::from(base));
        }
        let roots: Vec<&std::path::Path> = roots.iter().map(PathBuf::as_path).collect();
        let checkpoint = nn::Checkpoint::open_roots(&roots)?;
        let Some(path) = &model.lora_path else {
            return Ok(checkpoint);
        };
        let Model::Video(family) = self.family.model else {
            bail!("a causal language model fuses no adapter");
        };
        let adapter = nn::Adapter::open(std::path::Path::new(path))
            .with_context(|| format!("reading the adapter {path}"))?;
        let stage = family.adapter_stage(&checkpoint, &adapter)?;
        checkpoint
            .attach(stage, adapter)
            .with_context(|| format!("fusing the adapter {path}"))
    }

    /// The video family's reading of `checkpoint`, which [`Self::open_video_checkpoint`] opened,
    /// under the epoch's precision policy.
    pub(super) fn open_video_family(
        &self,
        checkpoint: &nn::Checkpoint,
    ) -> Result<Box<dyn VideoCheckpoint>> {
        let Model::Video(family) = self.family.model else {
            bail!("a causal language model holds no video pipeline");
        };
        Ok(family.open(checkpoint, &self.engine_config().model.quantization)?)
    }

    /// The denoiser's partition and the text encoder's tensor split of this rank, as the
    /// deployment page states them for `checkpoint`'s blocks: its own group's, at its position
    /// there.
    pub(super) fn video_split(
        &self,
        checkpoint: &dyn VideoCheckpoint,
    ) -> Result<(DenoiserSplit, Split)> {
        video_split(
            self.capability_groups(),
            &self.group,
            u32::try_from(self.position()?).context("a rank past what the protocol numbers")?,
            Split {
                size: self.mapping.tensor_parallel,
                rank: self.mapping.tensor_rank,
            },
            self.world_size,
            checkpoint.denoisers(),
            checkpoint.text_encoder(),
        )
    }

    /// The epoch's worker groups when they partition the model by capability, that is when some
    /// group places a block other than the whole model ([`proto::config::DEFAULT_BLOCK`]).
    ///
    /// The command-line shorthand places the whole model on every group, and a deployment file may
    /// as well. Such a deployment states no block to split, so it states no sequence degree and
    /// chooses no network of a checkpoint that holds two.
    pub(super) fn capability_groups(&self) -> Option<&[WorkerGroupConfig]> {
        let groups = &self.engine_config().parallel.worker_groups;
        groups
            .iter()
            .any(|group| {
                group
                    .blocks
                    .keys()
                    .any(|name| name != proto::config::DEFAULT_BLOCK)
            })
            .then_some(groups.as_slice())
    }
}

/// How this rank takes part in the deployment's parallel shape, as the build config
/// fixes it and the daemon's launch places it.
#[derive(Clone)]
pub(crate) struct Mapping {
    pub(super) tensor_parallel: u32,
    pub(super) tensor_rank: u32,
    /// The stages of a text group's pipeline, and this rank's stage.
    pub(super) pipeline_parallel: u32,
    pub(super) pipeline_rank: u32,
    /// A text group's Ulysses sequence split and this rank's member of it.
    pub(super) sequence: SequenceSplit,
    /// This rank's position in its worker group's parallel order (`parallel_order`): its place
    /// along the group's one axis and its part of every unit of the session's regions.
    pub(super) position: u32,
    /// The ranks of the expert union that share the routed experts, and which share this rank
    /// keeps; one rank outside a union.
    pub(super) expert_parallel: u32,
    pub(super) expert_rank: u32,
    /// This rank's place in the expert union, attention ranks first.
    pub(super) union_rank: u32,
    /// Leading ranks of the union that hold no experts.
    pub(super) attention_ranks: u32,
    /// Whether expert parallelism shards the routed experts over this rank's own worker group
    /// rather than an expert union: each of the group's tensor-parallel ranks holds whole
    /// experts and runs them over every token of the group's batches
    /// ([`proto::config::EngineConfig::expert_union`]).
    pub(super) group_experts: bool,
}

impl Mapping {
    /// This rank's part of the epoch's parallel shape.
    ///
    /// Tensor parallelism splits every layer of the group's ranks. Expert parallelism splits the
    /// routed experts of the deployment's schedulers when they form an expert union, and
    /// otherwise those of each scheduler's group over its tensor-parallel ranks. A rank's
    /// ordinal places it in both: its tensor rank is its position inside its scheduler's group,
    /// and its expert rank in a union is which scheduler's share of the experts it keeps.
    pub(super) fn of(launch: &EngineArgs, build: &proto::config::EngineConfig) -> Result<Self> {
        let tensor_parallel = u32::try_from(build.parallel.tensor_parallel_size.get())
            .context("a tensor-parallel degree larger than u32::MAX")?;
        if tensor_parallel == 0 {
            bail!("a tensor-parallel degree of zero splits nothing");
        }
        let data_parallel = u32::try_from(build.parallel.data_parallel_size.get())
            .context("a data-parallel degree larger than u32::MAX")?;
        let union = build.expert_union();
        let expert_parallel = if union { data_parallel } else { 1 };
        let tensor_rank = launch.rank % tensor_parallel;
        Ok(Self {
            tensor_parallel,
            tensor_rank,
            pipeline_parallel: 1,
            pipeline_rank: 0,
            sequence: SequenceSplit::WHOLE,
            position: tensor_rank,
            expert_parallel,
            expert_rank: (launch.rank / tensor_parallel) % expert_parallel,
            union_rank: 0,
            attention_ranks: 0,
            group_experts: build.moe.expert_parallel && !union,
        })
    }

    /// The placement this rank's causal language model loads at, for a deployment that drafts
    /// `draft_tokens` tokens a step: its tensor, pipeline and sequence splits, its share of the
    /// expert union's experts, and, when expert parallelism shards its own group's experts,
    /// whole experts of its tensor-parallel group's slots (`docs/expert_parallelism.md`).
    pub(super) fn text_placement(&self, draft_tokens: u32) -> TextPlacement {
        TextPlacement {
            tensor: Split {
                size: self.tensor_parallel,
                rank: self.tensor_rank,
            },
            pipeline: Split {
                size: self.pipeline_parallel,
                rank: self.pipeline_rank,
            },
            sequence: self.sequence,
            experts: Split {
                size: self.expert_parallel,
                rank: self.expert_rank,
            },
            tensor_experts: self.group_experts,
            // The drafter's layers load when the deployment speculates with them, and a model
            // whose state keeps a window per request sizes it by the drafts.
            drafter: draft_tokens > 0,
            draft_tokens,
        }
    }
}

/// This rank's place in the denoiser's partition and the text encoder's tensor split, as the
/// deployment page states them.
///
/// **The page is the one statement of the parallel shape, and this reads it rather than a second
/// copy.** A main process knows its own devices and its rank number, and the build config states the
/// tensor-parallel width; the denoiser's degrees and the head projection are the deployment file's,
/// written beside the block they split. So the group is found by the name the launch states, its
/// denoiser block is read, and the rank's place in that block's order is its coordinate on each
/// degree ([`DenoiserSplit`]: the Ulysses degree fastest, then the tensor degree, then the
/// all-gather degree, the layer degree slowest).
///
/// It refuses what would otherwise be assumed: a sequence split on another block than the
/// denoiser's, and a denoiser block whose ranks do not number its degrees' product.
pub(super) fn video_split(
    groups: Option<&[WorkerGroupConfig]>,
    group: &str,
    rank: u32,
    tensor: Split,
    world_size: usize,
    denoisers: &[&str],
    text_encoder: &str,
) -> Result<(DenoiserSplit, Split)> {
    let tensor_parallel = tensor.size as usize;
    let tensor_rank = tensor.rank as usize;
    let rank = usize::try_from(rank).context("a rank number the process addresses")?;
    let split = |size: usize, rank: usize| -> Result<Split> {
        Ok(Split {
            size: u32::try_from(size).context("a split wider than the protocol addresses")?,
            rank: u32::try_from(rank).context("a split rank the protocol addresses")?,
        })
    };
    let denoiser = denoisers.first().copied().unwrap_or_default();
    // A deployment that places the whole model states no block to split. With one rank there is no
    // other rank for a shape to disagree with, so the width the build config states is the whole of it.
    let Some(groups) = groups else {
        if world_size == 1 {
            return Ok((DenoiserSplit::WHOLE, Split::WHOLE));
        }
        bail!(
            "this deployment places the whole model on its groups and serves {world_size} ranks, so \
             no sequence degree is stated for them: a deployment file that places the {denoiser:?} \
             block states a rank's group and its split"
        );
    };
    // **The name the launch states is the daemon's process group, not the page's id.** Every engine's
    // main process is launched into one process group (`ENGINE_GROUP = "engine"`), which is the
    // epoch's bookkeeping name, while the page names the groups it places (`model`, `text`, `flow`).
    // A rank that looked itself up by the name it was launched under found neither and refused - so
    // the name is preferred when the page has it, and otherwise the page is asked the question it can
    // answer: **which group places this class's block?** A media rank serves the denoiser, and a
    // group that places no transformer is another engine's.
    let named = groups.iter().find(|candidate| candidate.id.0 == group);
    let placing: Vec<&WorkerGroupConfig> = groups
        .iter()
        .filter(|candidate| {
            denoisers
                .iter()
                .any(|block| candidate.blocks.contains_key(*block))
        })
        .collect();
    // Both sides as a pair in either refusal: the launch's name is the daemon's roster label and the
    // page's ids are the deployment's groups, and a reader who sees only one side cannot tell which
    // of the two statements to correct.
    let placed = match (named, placing.as_slice()) {
        (Some(named), _) => named,
        (None, [only]) => *only,
        (None, []) => {
            bail!(
                "the launch file states worker group {group:?} and the deployment page places \
                 neither it nor a group that places {denoiser:?}: the page places {}",
                groups
                    .iter()
                    .map(|group| format!("{:?}", group.id.0))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        (None, many) => {
            bail!(
                "this rank serves worker group {group:?}, which the deployment does not place, \
                 and {} groups place {denoiser:?} ({}): one partition needs one of them",
                many.len(),
                many.iter()
                    .map(|group| format!("{:?}", group.id.0))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    };
    // The page states the text encoder's degree beside it, so the group's width is the page's and
    // the rank's place in it is the rank's place in the block. A rank the block does not hold, a
    // denoising rank beside a text encoder split over fewer ranks, loads no part of the encoder
    // (`ModelRunner::stages`): its split is the whole, which nothing it loads reads.
    let (tensor_parallel, tensor_rank) = match placed.blocks.get(text_encoder) {
        Some(text) if text.tensor_parallel_size > 1 => {
            let position = if text.ranks.is_empty() {
                Some(rank)
            } else {
                text.ranks.iter().position(|member| *member == rank)
            };
            match position {
                Some(position) => (text.tensor_parallel_size, position),
                None => (1, 0),
            }
        }
        _ => (tensor_parallel, tensor_rank),
    };
    let text = split(tensor_parallel, tensor_rank)?;

    // The one block a sequence splits is the denoiser's: a split stated on another would serve
    // it at degree one on every member.
    let network = placed
        .blocks
        .iter()
        .find(|(name, _)| denoisers.contains(&name.as_str()));
    let local = |block: &proto::config::BlockConfig| {
        matches!(
            block.sequence_parallel,
            proto::config::SequenceParallel::Local
        )
    };
    if let Some((other, _)) = placed
        .blocks
        .iter()
        .find(|(name, block)| !local(block) && network.is_none_or(|(network, _)| network != *name))
    {
        bail!(
            "worker group {group:?} states a sequence split on block {other:?}, and only the \
             denoiser's block ({}) splits its sequence",
            network.map_or(denoiser, |(name, _)| name.as_str())
        );
    }
    let Some((name, block)) = network else {
        return Ok((DenoiserSplit::WHOLE, text));
    };

    // A block that states no member list holds every rank of its replica, so a rank's place in it
    // is the rank's own ordinal. A rank outside the block loads no part of the denoiser.
    let place = if block.ranks.is_empty() {
        Some(rank)
    } else {
        block.ranks.iter().position(|member| *member == rank)
    };
    let Some(place) = place else {
        return Ok((DenoiserSplit::WHOLE, text));
    };

    let (degrees, strides) = denoiser_axes(block);
    let [ulysses, gather, tensor, layers] = degrees;
    let projection = match &block.sequence_parallel {
        proto::config::SequenceParallel::Ulysses { projection, .. } => head_projection(*projection),
        // A hybrid split states no projection, and its Ulysses members gather, as an omitted
        // mode does.
        proto::config::SequenceParallel::Hybrid { .. } => Projection::Gather,
        proto::config::SequenceParallel::Local
        | proto::config::SequenceParallel::Allgather { .. } => Projection::Exchange,
    };
    let width = [ulysses, gather, tensor, layers]
        .iter()
        .try_fold(1_usize, |product, degree| product.checked_mul(*degree))
        .filter(|width| *width > 0)
        .context("the denoiser's degrees multiply past what a block numbers")?;
    let members = if block.ranks.is_empty() {
        placed.ranks.len()
    } else {
        block.ranks.len()
    };
    if members != width || place >= width {
        bail!(
            "block {name:?} of worker group {group:?} holds {members} ranks, and its degrees \
             (Ulysses {ulysses}, all-gather {gather}, tensor {tensor}, layers {layers}) number \
             {width}"
        );
    }

    // The rank's coordinate on each axis.
    let coordinate = |axis: usize| (place / strides[axis]) % degrees[axis];
    Ok((
        DenoiserSplit {
            ulysses: SequenceSplit {
                split: split(ulysses, coordinate(ULYSSES_AXIS))?,
                projection,
            },
            gather: split(gather, coordinate(GATHER_AXIS))?,
            tensor: split(tensor, coordinate(TENSOR_AXIS))?,
            layers: split(layers, coordinate(LAYER_AXIS))?,
        },
        text,
    ))
}

/// A denoiser block's degree along each axis of its partition, in the order the denoiser
/// addresses them (`nn::family::ULYSSES_AXIS` and after: Ulysses, all-gather, tensor, layers),
/// and the distance in the block's order between two consecutive coordinates of each: the
/// Ulysses degree varies fastest, then the tensor degree, then the all-gather degree, and the
/// layer degree slowest ([`DenoiserSplit`]). A caller refuses a degree of zero.
pub(super) fn denoiser_axes(block: &BlockConfig) -> ([usize; 4], [usize; 4]) {
    let (ulysses, gather) = match block.sequence_parallel {
        proto::config::SequenceParallel::Local => (1, 1),
        proto::config::SequenceParallel::Ulysses { ulysses_degree, .. } => (ulysses_degree, 1),
        proto::config::SequenceParallel::Allgather { allgather_degree } => (1, allgather_degree),
        proto::config::SequenceParallel::Hybrid {
            ulysses_degree,
            allgather_degree,
        } => (ulysses_degree, allgather_degree),
    };
    let tensor = block.tensor_parallel_size;

    let degrees = [ulysses, gather, tensor, block.pipeline_parallel_size];
    let strides = [
        1,
        ulysses.saturating_mul(tensor),
        ulysses,
        ulysses.saturating_mul(tensor).saturating_mul(gather),
    ];
    (degrees, strides)
}

/// The head projection a deployment states for a Ulysses split, as a model takes it.
pub(super) fn head_projection(stated: HeadProjection) -> Projection {
    match stated {
        HeadProjection::Gather => Projection::Gather,
        HeadProjection::Exchange => Projection::Exchange,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`video_split`] for a checkpoint whose networks are `transformer` and `transformer_ref` and
    /// whose text encoder is `text_encoder`, as a deployment file names them.
    fn split(
        groups: Option<&[WorkerGroupConfig]>,
        group: &str,
        rank: u32,
        tensor_parallel: u32,
        tensor_rank: u32,
        world_size: usize,
    ) -> Result<(DenoiserSplit, Split)> {
        video_split(
            groups,
            group,
            rank,
            Split {
                size: tensor_parallel,
                rank: tensor_rank,
            },
            world_size,
            &["transformer", "transformer_ref"],
            "text_encoder",
        )
    }

    /// A deployment page as written, parsed and placed on this host.
    fn page(toml: &str) -> Vec<WorkerGroupConfig> {
        proto::config::WorkerGroupConfig::parse_file(toml, "localhost").expect("the page places")
    }

    /// A deployment page holding one group named `model`, with the blocks it names.
    fn groups(blocks: &str) -> Vec<WorkerGroupConfig> {
        page(&format!(
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cuda:0-3\"]\nqueue_depth = 2\n{blocks}"
        ))
    }

    /// **The name the launch states is the daemon's process group, and the page's ids are its own.**
    ///
    /// Every engine's main process is launched into one process group (`engine`), which is the
    /// epoch's bookkeeping name rather than the deployment's; the page names what it places (`model`,
    /// `text`, `flow`). The page answers the question it states: the group that places the denoiser
    /// is this rank's. A page where two groups place a denoiser does not say which rank is which, and
    /// is refused rather than guessed at.
    #[test]
    fn the_group_is_the_one_that_places_the_denoiser_when_the_launch_name_is_not_on_the_page() {
        let placed = groups(
            "[worker_groups.blocks.transformer]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 4, projection = \"exchange\" }\n",
        );
        let (sequence, _) = split(Some(&placed), "engine", 1, 4, 1, 4).expect(
            "the group that places the denoiser is this rank's, whatever the launch calls it",
        );
        assert_eq!(
            sequence.sequence().size,
            4,
            "its split is the one this rank serves"
        );
        assert_eq!(sequence.sequence().rank, 1);

        let two = page(
            "[[worker_groups]]\nid = \"text\"\nranks = [\"cuda:0-3\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.text_encoder]\ntensor_parallel_size = 4\n\
             [[worker_groups]]\nid = \"flow\"\nranks = [\"cuda:0-3\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.transformer]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 4 }\n\
             [[worker_groups]]\nid = \"other\"\nranks = [\"cuda:0-3\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.transformer]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 4 }\n",
        );
        let error = split(Some(&two), "engine", 0, 1, 0, 1)
            .expect_err("two groups placing a denoiser is not one partition");
        assert!(
            format!("{error:#}").contains("flow") && format!("{error:#}").contains("other"),
            "the refusal names both groups: {error:#}"
        );
    }

    /// The rank's sequence split is the deployment's, and it is read from the page that states it:
    /// the build config carries no sequence degree, and a split of one over four ranks would serve
    /// four whole denoisers.
    #[test]
    fn the_sequence_split_comes_from_the_group_the_rank_serves() {
        let placed = groups(
            "[worker_groups.blocks.transformer]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 4, projection = \"exchange\" }\n\
             [worker_groups.blocks.text_encoder]\ntensor_parallel_size = 4\n",
        );
        let (sequence, tensor) =
            split(Some(&placed), "model", 2, 4, 2, 4).expect("a four-way exchange partition");
        assert_eq!(sequence.sequence().size, 4, "the page states the degree");
        assert_eq!(
            sequence.ulysses.projection,
            Projection::Exchange,
            "the page states the projection"
        );
        assert_eq!(
            (tensor.size, tensor.rank),
            (4, 2),
            "the tensor place is the rank's"
        );
        assert_eq!(
            sequence.sequence().rank,
            2,
            "a block of every rank places the rank at its ordinal"
        );
    }

    /// A text encoder split over some of a group's ranks, as
    /// `configs/minimax_h3/ulysses8-reference-two-host.toml` splits it over the first four of
    /// eight: its members take their places in the block, and a rank outside it takes the whole
    /// split, since it loads no part of the encoder, while its sequence place is its own.
    #[test]
    fn a_rank_outside_the_text_encoder_s_block_holds_no_part_of_it() {
        let placed = page(
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cuda:0-7\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.transformer_ref]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 8, projection = \"exchange\" }\n\
             [worker_groups.blocks.text_encoder]\nranks = [0, 1, 2, 3]\ntensor_parallel_size = 4\n",
        );
        let (_, tensor) = split(Some(&placed), "model", 2, 1, 0, 8).expect("a member's split");
        assert_eq!((tensor.size, tensor.rank), (4, 2));

        let (sequence, tensor) =
            split(Some(&placed), "model", 5, 1, 0, 8).expect("a denoising rank outside the block");
        assert_eq!(tensor, Split::WHOLE);
        assert_eq!((sequence.sequence().size, sequence.sequence().rank), (8, 5));
    }

    /// A denoiser block's order places each rank on its four degrees, the Ulysses degree
    /// fastest and the layer degree slowest: a tensor two by Ulysses two block's Ulysses pairs
    /// are its consecutive places, an all-gather degree or a hybrid one gives each place its
    /// interval of the sequence, and a layer split one stage per place. A block whose ranks do
    /// not number the product of its degrees is refused.
    #[test]
    fn a_denoiser_block_places_each_rank_on_its_degrees() {
        let placed = |transformer: &str| {
            page(&format!(
                "[[worker_groups]]\nid = \"model\"\nranks = [\"cuda:0-3\"]\nqueue_depth = 2\n\
                 [worker_groups.blocks.transformer]\nranks = [3, 1, 2, 0]\n{transformer}\n"
            ))
        };
        let place = |groups: &[WorkerGroupConfig], rank: u32| {
            let (denoiser, _) = split(Some(groups), "model", rank, 1, 0, 4).unwrap();
            (
                (denoiser.sequence().size, denoiser.sequence().rank),
                denoiser.ulysses.split.size,
                (denoiser.tensor.size, denoiser.tensor.rank),
                (denoiser.layers.size, denoiser.layers.rank),
            )
        };

        // Group rank 1 is the block's place 1, group rank 0 its place 3.
        let mixed = placed(
            "tensor_parallel_size = 2\nsequence_parallel = { kind = \"ulysses\", ulysses_degree = 2 }",
        );
        assert_eq!(place(&mixed, 1), ((2, 1), 2, (2, 0), (1, 0)));
        assert_eq!(place(&mixed, 0), ((2, 1), 2, (2, 1), (1, 0)));

        let gather = placed("sequence_parallel = { kind = \"allgather\", allgather_degree = 4 }");
        assert_eq!(place(&gather, 2), ((4, 2), 1, (1, 0), (1, 0)));

        let hybrid = placed(
            "sequence_parallel = { kind = \"hybrid\", ulysses_degree = 2, allgather_degree = 2 }",
        );
        let (denoiser, _) = split(Some(&hybrid), "model", 0, 1, 0, 4).unwrap();
        assert_eq!((denoiser.ulysses.split.rank, denoiser.gather.rank), (1, 1));
        assert_eq!(denoiser.ulysses.projection, Projection::Gather);

        let pipeline = placed("pipeline_parallel_size = 4");
        assert_eq!(place(&pipeline, 3), ((1, 0), 1, (1, 0), (4, 0)));
        assert_eq!(place(&pipeline, 0), ((1, 0), 1, (1, 0), (4, 3)));

        let tensor = placed("tensor_parallel_size = 4");
        assert_eq!(place(&tensor, 2).2, (4, 2));
    }

    /// A deployment that places the whole model serves one rank, and refuses to guess above one.
    ///
    /// The shorthand deployment (no deployment file) places the whole model, so it states a
    /// tensor-parallel width and no sequence degree. With one rank no other rank can disagree with the shape, so one rank's
    /// sequence is the whole of it; with several, a shape read from the width alone would be an
    /// assumption, and the refusal names the file that would state it.
    #[test]
    fn a_deployment_without_a_file_serves_one_rank_or_refuses() {
        let (sequence, tensor) = split(None, "model", 0, 1, 0, 1).expect("one rank's sequence");
        assert_eq!(sequence, DenoiserSplit::WHOLE);
        assert_eq!(tensor, Split::WHOLE);
        let error = split(None, "model", 2, 4, 2, 4).expect_err("four ranks need a file");
        assert!(
            format!("{error:#}").contains("deployment file"),
            "the refusal names what would state the shape: {error:#}"
        );
    }

    /// A page that states no split serves one rank's sequence, which is not a refusal. The group is
    /// one rank: a block without a split on several ranks would be several whole denoisers, and
    /// the page's own validation refuses that.
    #[test]
    fn a_group_with_no_sequence_split_serves_one_rank() {
        let placed = page(
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cuda:0\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.transformer]\n",
        );
        let (sequence, _) = split(Some(&placed), "model", 0, 1, 0, 1).expect("one rank's sequence");
        assert_eq!(sequence.sequence(), Split::WHOLE);
    }

    /// A page with two sequence-split blocks is refused by name: a partition describes one split,
    /// and serving the second at degree one would hold whole copies of it on every member. Both
    /// projections are served, so neither is a refusal.
    #[test]
    fn a_second_sequence_split_is_refused_by_name() {
        for projection in ["gather", "exchange"] {
            let placed = page(&format!(
                "[[worker_groups]]\nid = \"model\"\nranks = [\"cuda:0-1\"]\nqueue_depth = 2\n\
                 [worker_groups.blocks.transformer]\nsequence_parallel = {{ kind = \"ulysses\", \
                 ulysses_degree = 2, projection = \"{projection}\" }}\n"
            ));
            let (sequence, _) = split(Some(&placed), "model", 0, 1, 0, 2)
                .unwrap_or_else(|error| panic!("the {projection} projection is served: {error:#}"));
            assert_eq!(sequence.sequence().size, 2);
        }
        let two = page(
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cuda:0-1\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.transformer]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 2 }\n\
             [worker_groups.blocks.text_encoder]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 2 }\n",
        );
        let error = split(Some(&two), "model", 0, 2, 0, 2).expect_err("one partition, one split");
        assert!(
            format!("{error:#}").contains("transformer")
                && format!("{error:#}").contains("text_encoder"),
            "the refusal names both blocks: {error:#}"
        );
    }
}
