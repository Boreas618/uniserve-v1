//! A rank's place in its deployment's parallel shape.

use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use nn::family::{Family, Projection, SequenceSplit, Split, VideoCheckpoint};
use proto::config::EngineArgs;

use super::{ModelRunner, ParallelShape};

impl ModelRunner {
    /// The parallel groups this rank joins: `(sequence group's ranks, tensor group's ranks)` and its
    /// place in each. A rank of a media deployment is in both; the text class joins neither here.
    pub(crate) fn parallel_shape(&self) -> Result<ParallelShape> {
        let Family::Video(_) = self.family else {
            // The tensor group is the text model's only collective group: an expert union
            // exchanges over its own transport (`union_member`).
            return Ok(ParallelShape {
                tensor: self.mapping.tensor_parallel as usize,
                tensor_rank: self.mapping.tensor_rank as usize,
                ..ParallelShape::default()
            });
        };
        let (sequence, tensor) = self.video_split(&*self.open_video()?)?;
        Ok(ParallelShape {
            sequence: sequence.split.size as usize,
            sequence_rank: sequence.split.rank as usize,
            tensor: tensor.size as usize,
            tensor_rank: tensor.rank as usize,
        })
    }

    /// This rank's video checkpoint, read over its roots: the checkpoint the launch names and the
    /// base release it continues.
    pub(super) fn open_video(&self) -> Result<Box<dyn VideoCheckpoint>> {
        let Family::Video(family) = self.family else {
            bail!("a causal language model holds no video pipeline");
        };
        let mut roots = vec![self.checkpoint().to_path_buf()];
        if let Some(base) = &self.engine_config().model.base {
            roots.push(PathBuf::from(base));
        }
        Ok(family.open(&roots)?)
    }

    /// The denoiser's sequence split and the text encoder's tensor split of this rank, as the
    /// deployment page states them for `checkpoint`'s components.
    pub(super) fn video_split(
        &self,
        checkpoint: &dyn VideoCheckpoint,
    ) -> Result<(SequenceSplit, Split)> {
        video_split(
            self.capability_groups(),
            &self.launch.group,
            self.rank,
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
    pub(super) fn capability_groups(&self) -> Option<&[scheduler::WorkerGroupConfig]> {
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
    pub(super) expert_parallel: u32,
    pub(super) expert_rank: u32,
    /// This rank's place in the expert union, attention ranks first.
    pub(super) union_rank: u32,
    /// Leading ranks of the union that hold no experts.
    pub(super) attention_ranks: u32,
}

impl Mapping {
    /// This rank's part of the epoch's parallel shape.
    ///
    /// Tensor parallelism splits every layer of the group's ranks; expert parallelism splits the
    /// routed experts of the deployment's schedulers. A rank's ordinal places it in both: its
    /// tensor rank is its position inside its scheduler's group, and its expert rank is which
    /// scheduler's share of the experts it keeps.
    pub(super) fn of(launch: &EngineArgs, build: &proto::config::EngineConfig) -> Result<Self> {
        let tensor_parallel = u32::try_from(build.parallel.tensor_parallel_size.get())
            .context("a tensor-parallel degree larger than u32::MAX")?;
        if tensor_parallel == 0 {
            bail!("a tensor-parallel degree of zero splits nothing");
        }
        let data_parallel = u32::try_from(build.parallel.data_parallel_size.get())
            .context("a data-parallel degree larger than u32::MAX")?;
        let expert_parallel = if build.moe.expert_parallel {
            data_parallel
        } else {
            1
        };
        Ok(Self {
            tensor_parallel,
            tensor_rank: launch.rank % tensor_parallel,
            expert_parallel,
            expert_rank: (launch.rank / tensor_parallel) % expert_parallel,
            union_rank: 0,
            attention_ranks: 0,
        })
    }
}

/// This rank's sequence split, and its place in it, as the deployment page states them.
///
/// **The page is the one statement of the parallel shape, and this reads it rather than a second
/// copy.** A main process knows its own devices and its rank number, and the build config states the
/// tensor-parallel width; the sequence degree and the head projection are the deployment file's,
/// written beside the component they split. So the group is found by the name the launch states, its
/// one sequence-split component is read, and the rank's position inside that component is its
/// sequence rank.
///
/// It refuses what would otherwise be assumed: a group with two sequence-split components is not one
/// partition, a projection this stack does not implement is refused rather than served at degree
/// one, and a mode that is not Ulysses is refused by name because the denoiser's sequence exchange
/// is Ulysses' alone.
#[expect(
    clippy::too_many_arguments,
    reason = "the page, the rank's place in it, and the components the checkpoint names"
)]
pub(super) fn video_split(
    groups: Option<&[scheduler::WorkerGroupConfig]>,
    group: &str,
    rank: u32,
    tensor: Split,
    world_size: usize,
    denoisers: &[&str],
    text_encoder: &str,
) -> Result<(SequenceSplit, Split)> {
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
            return Ok((SequenceSplit::WHOLE, Split::WHOLE));
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
    // answer: **which group places this class's component?** A media rank serves the denoiser, and a
    // group that places no transformer is another engine's.
    let named = groups.iter().find(|candidate| candidate.id.0 == group);
    let denoisers: Vec<&scheduler::WorkerGroupConfig> = groups
        .iter()
        .filter(|candidate| {
            denoisers
                .iter()
                .any(|component| candidate.blocks.contains_key(*component))
        })
        .collect();
    // Both sides as a pair in either refusal: the launch's name is the daemon's roster label and the
    // page's ids are the deployment's groups, and a reader who sees only one side cannot tell which
    // of the two statements to correct.
    let placed = match (named, denoisers.as_slice()) {
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
    // the rank's place in it is the rank's place in the component.
    let (tensor_parallel, tensor_rank) = match placed.blocks.get(text_encoder) {
        Some(text) if text.tensor_parallel_size > 1 => {
            let position = if text.ranks.is_empty() {
                rank
            } else {
                text.ranks
                    .iter()
                    .position(|member| *member == rank)
                    .with_context(|| {
                        format!(
                            "this rank is {rank} and the group's text encoder holds ranks {:?}",
                            text.ranks
                        )
                    })?
            };
            (text.tensor_parallel_size, position)
        }
        _ => (tensor_parallel, tensor_rank),
    };
    let mut splits = placed.blocks.iter().filter(|(_, component)| {
        !matches!(
            component.sequence_parallel,
            proto::config::SequenceParallel::Local
        )
    });
    let (_, component) = match (splits.next(), splits.next()) {
        // No split: the sequence is this rank's own, and the tensor degree is the whole shape.
        (None, _) => {
            return Ok((
                SequenceSplit {
                    split: split(1, 0)?,
                    projection: Projection::Exchange,
                },
                split(tensor_parallel, tensor_rank)?,
            ));
        }
        (Some(one), None) => one,
        (Some(first), Some(second)) => bail!(
            "worker group {group:?} states a sequence split on two components ({:?} and {:?}) and \
             one partition describes one split",
            first.0,
            second.0
        ),
    };
    // A component that states no member list holds every rank of its replica, so a rank's position
    // in it is the rank's own ordinal.
    let sequence_rank = if component.ranks.is_empty() {
        rank
    } else {
        component
            .ranks
            .iter()
            .position(|member| *member == rank)
            .with_context(|| {
                format!(
                    "this rank is {rank} and the group's sequence-split component holds ranks {:?}",
                    component.ranks
                )
            })?
    };
    let sequence = match &component.sequence_parallel {
        proto::config::SequenceParallel::Ulysses {
            ulysses_degree,
            projection,
            ..
        } => SequenceSplit {
            split: split(*ulysses_degree, sequence_rank)?,
            projection: match projection {
                proto::config::HeadProjection::Exchange => Projection::Exchange,
                proto::config::HeadProjection::Gather => Projection::Gather,
            },
        },
        other => bail!(
            "worker group {group:?} states the sequence split {other:?}, and a video denoiser \
             exchanges its sequence over Ulysses ranks"
        ),
    };
    Ok((sequence, split(tensor_parallel, tensor_rank)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`video_split`] for a checkpoint whose networks are `transformer` and `transformer_ref` and
    /// whose text encoder is `text_encoder`, as a deployment file names them.
    fn split(
        groups: Option<&[scheduler::WorkerGroupConfig]>,
        group: &str,
        rank: u32,
        tensor_parallel: u32,
        tensor_rank: u32,
        world_size: usize,
    ) -> Result<(SequenceSplit, Split)> {
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
    fn page(toml: &str) -> Vec<scheduler::WorkerGroupConfig> {
        proto::config::WorkerGroupConfig::parse_file(toml, "localhost").expect("the page places")
    }

    /// A deployment page holding one group named `model`, with the components it names.
    fn groups(components: &str) -> Vec<scheduler::WorkerGroupConfig> {
        page(&format!(
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cuda:0-3\"]\nqueue_depth = 2\n{components}"
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
            sequence.split.size, 4,
            "its split is the one this rank serves"
        );
        assert_eq!(sequence.split.rank, 1);

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
        assert_eq!(sequence.split.size, 4, "the page states the degree");
        assert_eq!(
            sequence.projection,
            Projection::Exchange,
            "the page states the projection"
        );
        assert_eq!(
            (tensor.size, tensor.rank),
            (4, 2),
            "the tensor place is the rank's"
        );
        assert_eq!(
            sequence.split.rank, 2,
            "a component of every rank places the rank at its ordinal"
        );
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
        assert_eq!(sequence, SequenceSplit::WHOLE);
        assert_eq!(tensor, Split::WHOLE);
        let error = split(None, "model", 2, 4, 2, 4).expect_err("four ranks need a file");
        assert!(
            format!("{error:#}").contains("deployment file"),
            "the refusal names what would state the shape: {error:#}"
        );
    }

    /// A page that states no split serves one rank's sequence, which is not a refusal. The group is
    /// one rank: a component without a split on several ranks would be several whole denoisers, and
    /// the page's own validation refuses that.
    #[test]
    fn a_group_with_no_sequence_split_serves_one_rank() {
        let placed = page(
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cuda:0\"]\nqueue_depth = 2\n\
             [worker_groups.blocks.transformer]\n",
        );
        let (sequence, _) = split(Some(&placed), "model", 0, 1, 0, 1).expect("one rank's sequence");
        assert_eq!(sequence.split, Split::WHOLE);
    }

    /// A page with two sequence-split components is refused by name: a partition describes one split,
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
            assert_eq!(sequence.split.size, 2);
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
            "the refusal names both components: {error:#}"
        );
    }
}
