//! Image trajectories of admitted requests: guidance plans and the cache
//! prefixes each guidance branch conditions on.
//!
//! An image request denoises one trajectory per generated image. When its
//! latent is prepared, its batch opens an [`ImageTrajectory`]: the
//! guidance plan the model's schedules and guidance select per step (asked
//! of the library once per trajectory; the solver update rebuilds the
//! schedules and guidance from the image's parameters, which the plan
//! keeps), and the request's guidance prefixes.
//!
//! Each guidance branch attends to a cache prefix: the conditioned branch to
//! the request's own context; the unconditional branches to prefixes the
//! scheduler placed in other request slots, which the runtime prefills the
//! first time a branch is evaluated. A prefix's tokens are resolved once
//! per source when the trajectory opens and kept with it: the negative
//! source's are the request's admitted negative tokens, which the engine
//! sized that prefix's cache from ([`BranchSource::admitted`]), and the
//! model frames the others (`image.guidance_prefix`, its prompt framing
//! and tokenizer). A prefix's physical
//! coordinates are resolved again for every submitted denoising call,
//! because the scheduler may place a later interval's prefixes in other
//! slots.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use uniserve_ipc::RequestKey;

use uniserve_core::ImageParams;

use crate::{Error, Result};

/// A classifier-free guidance branch (`uniserve.diffusion.Branch`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Branch {
    Conditioned,
    TextUnconditional,
    ImageUnconditional,
}

impl Branch {
    /// Every branch in evaluation order: conditioned first.
    pub(crate) const ALL: [Self; 3] = [
        Self::Conditioned,
        Self::TextUnconditional,
        Self::ImageUnconditional,
    ];

    const fn bit(self) -> u8 {
        match self {
            Self::Conditioned => 1,
            Self::TextUnconditional => 2,
            Self::ImageUnconditional => 4,
        }
    }

    /// The prefix this branch conditions on; `image_unconditional` is the
    /// denoiser's source for its image-unconditional branch.
    pub(crate) fn source(self, image_unconditional: BranchSource) -> BranchSource {
        match self {
            Self::Conditioned => BranchSource::Conditioning,
            Self::TextUnconditional => BranchSource::NegativeOrStart,
            Self::ImageUnconditional => image_unconditional,
        }
    }
}

/// The branches one denoising step evaluates, in evaluation order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct BranchSet(u8);

impl BranchSet {
    /// A set from its mask: bit 0 conditioned, bit 1 text-unconditional,
    /// bit 2 image-unconditional. The conditioned branch is always
    /// evaluated.
    pub(crate) fn from_mask(mask: u8) -> Result<Self> {
        if mask & Branch::Conditioned.bit() == 0 || mask > 7 {
            return Err(Error::Interface(format!(
                "guidance branch mask {mask:#b} omits the conditioned branch"
            )));
        }
        Ok(Self(mask))
    }

    pub(crate) fn mask(self) -> u8 {
        self.0
    }

    pub(crate) fn contains(self, branch: Branch) -> bool {
        self.0 & branch.bit() != 0
    }

    /// The branches in evaluation order.
    pub(crate) fn iter(self) -> impl Iterator<Item = Branch> {
        Branch::ALL
            .into_iter()
            .filter(move |&branch| self.contains(branch))
    }

    pub(crate) fn len(self) -> usize {
        self.0.count_ones() as usize
    }
}

/// What a guidance branch conditions on (`uniserve.processing.BranchSource`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BranchSource {
    /// The image prompt, or the request's own context when blank.
    Conditioning,
    /// The negative prompt. In serving its prefix is the request's admitted
    /// negative tokens ([`Self::admitted`]); the model frames an empty
    /// negative prompt as the start of a conversation.
    NegativeOrStart,
    /// The bare start of a conversation.
    Start,
}

impl BranchSource {
    pub(crate) fn parse(name: &str) -> Result<Self> {
        match name {
            "conditioning" => Ok(Self::Conditioning),
            "negative_or_start" => Ok(Self::NegativeOrStart),
            "start" => Ok(Self::Start),
            other => Err(Error::Interface(format!(
                "unknown guidance branch source {other:?}"
            ))),
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Conditioning => "conditioning",
            Self::NegativeOrStart => "negative_or_start",
            Self::Start => "start",
        }
    }

    /// The prefix a branch of this source conditions on that the request's
    /// admission fixes, given its admitted negative tokens `negative`; `None`
    /// for a source the model frames.
    ///
    /// The negative source conditions on exactly the admitted tokens, also
    /// when none were admitted. They are the request's final negative
    /// conditioning tokens (`GenerationRequest::negative_prompt_token_ids`),
    /// and the engine sizes the prefix's cache and its prefill from their
    /// count, so a prefix the rank framed itself would not fit the slot the
    /// engine placed it in.
    pub(crate) fn admitted(self, negative: &[u32]) -> Option<Prefix> {
        (self == Self::NegativeOrStart).then(|| Prefix {
            tokens: Arc::from(negative),
            reuses_conditioning: false,
        })
    }
}

/// A guidance prefix: its tokens, or the request's own context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Prefix {
    pub tokens: Arc<[u32]>,
    /// The branch attends to the request's own conditioning prefix cache.
    pub reuses_conditioning: bool,
}

/// Cache coordinates of a request slot: its visible tokens and its allocated
/// token capacity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Coordinates {
    pub slot: u32,
    pub visible: u64,
    pub capacity: u32,
}

impl From<(u32, u64, u32)> for Coordinates {
    fn from((slot, visible, capacity): (u32, u64, u32)) -> Self {
        Self {
            slot,
            visible,
            capacity,
        }
    }
}

/// One scheduler forward descriptor of a denoising call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ForwardSlot {
    pub slot: u32,
    pub seq_len: u32,
    pub query_len: u32,
}

/// A prefix prefill the runtime submits before a step's denoising rows:
/// `tokens` written as prefix cache from position zero of `slot`, causally, with the
/// hidden states selected (and discarded).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrefixFill {
    pub slot: u32,
    pub tokens: Arc<[u32]>,
}

/// The guidance plan of one trajectory: per-step evaluation branches and
/// timesteps, with the image parameters its schedules and guidance are made
/// from, which the solver update combines predictions with.
#[derive(Clone, Debug)]
pub(crate) struct GuidancePlan {
    /// The image parameters (`DenoiserRunner.plan`).
    pub params: Arc<ImageParams>,
    /// The model timestep of each solver step.
    pub timesteps: Vec<f32>,
    /// The branches each step evaluates.
    pub branches: Vec<BranchSet>,
}

impl GuidancePlan {
    pub(crate) fn steps(&self) -> usize {
        self.branches.len()
    }

    /// The branches and timestep of `step`.
    pub(crate) fn step(&self, step: usize) -> Result<(BranchSet, f32)> {
        match (self.branches.get(step), self.timesteps.get(step)) {
            (Some(&branches), Some(&time)) => Ok((branches, time)),
            _ => Err(Error::Invalid(format!(
                "denoising step {step} is outside its {}-step schedule",
                self.steps()
            ))),
        }
    }
}

/// One image request's open trajectory.
#[derive(Clone, Debug)]
pub(crate) struct ImageTrajectory {
    pub key: RequestKey,
    /// Image height and width in pixels.
    pub size: (u32, u32),
    pub plan: GuidancePlan,
    /// The prefix tokens of each source a branch may condition on,
    /// resolved when the trajectory opened.
    prefixes: HashMap<BranchSource, Prefix>,
    /// The request's own conditioning coordinates for the current call.
    pub conditioning: Coordinates,
    /// Each branch's prefix coordinates for the current call.
    branches: HashMap<Branch, Coordinates>,
}

impl ImageTrajectory {
    /// A trajectory of `plan` whose guidance branches condition on
    /// `prefixes`, by source.
    pub(crate) fn new(
        key: RequestKey,
        size: (u32, u32),
        plan: GuidancePlan,
        prefixes: HashMap<BranchSource, Prefix>,
    ) -> Self {
        Self {
            key,
            size,
            plan,
            prefixes,
            conditioning: Coordinates::default(),
            branches: HashMap::new(),
        }
    }

    /// Start a denoising call: the request's conditioning coordinates are
    /// current and every branch's coordinates resolve again.
    pub(crate) fn begin_call(&mut self, conditioning: Coordinates) {
        self.conditioning = conditioning;
        self.branches.clear();
    }

    /// Resolve the coordinates of every branch of `set` not yet resolved in
    /// this call, returning the prefix prefills to submit first.
    ///
    /// `descriptors` are the call's forward descriptors: optional prefix
    /// prefills, then one denoising descriptor per branch of `set` in
    /// evaluation order. A non-reusing branch's prefix lives in its
    /// denoising descriptor's slot; it is empty (visible length zero) when
    /// the call also carries a prefill of exactly the prefix's tokens for
    /// that slot, else its visible length is the descriptor's context.
    /// `coordinates` reads a slot's allocated coordinates. Branches sharing
    /// a physical prefix submit one prefill. A branch whose source the
    /// trajectory was opened without is refused.
    pub(crate) fn resolve_branches(
        &mut self,
        set: BranchSet,
        image_unconditional: BranchSource,
        descriptors: &[ForwardSlot],
        coordinates: impl Fn(u32, u64) -> Result<Coordinates>,
    ) -> Result<Vec<PrefixFill>> {
        let count = set.len();
        if descriptors.len() < count {
            return Err(Error::Invalid(
                "media denoise has incomplete forward-row metadata".into(),
            ));
        }
        let (prefills, denoising) = descriptors.split_at(descriptors.len() - count);
        let mut fills = Vec::new();
        let mut filled = HashSet::new();
        for (index, branch) in set.iter().enumerate() {
            if self.branches.contains_key(&branch) {
                continue;
            }
            let source = branch.source(image_unconditional);
            let prefix = self.prefixes.get(&source).cloned().ok_or_else(|| {
                Error::Invariant(format!(
                    "an image trajectory was opened without its {} prefix",
                    source.name()
                ))
            })?;

            let (located, length) = if prefix.reuses_conditioning {
                (self.conditioning, self.conditioning.visible)
            } else {
                let descriptor = denoising[index];
                let tokens = prefix.tokens.len() as u32;
                let prefilled = prefills.iter().any(|row| {
                    row.slot == descriptor.slot
                        && row.seq_len == row.query_len
                        && row.query_len == tokens
                });
                let visible = if prefilled {
                    0
                } else {
                    u64::from(descriptor.seq_len.saturating_sub(descriptor.query_len))
                };
                (coordinates(descriptor.slot, visible)?, u64::from(tokens))
            };
            if length > u64::from(located.capacity) {
                return Err(Error::Invalid(
                    "guidance prefix exceeds scheduler params".into(),
                ));
            }
            if located.visible != 0 && located.visible != length {
                return Err(Error::Invalid(
                    "guidance branch prefix disagrees with its initialized physical state".into(),
                ));
            }
            self.branches.insert(branch, located);
            if located.visible == 0 && !prefix.tokens.is_empty() && filled.insert(located.slot) {
                fills.push(PrefixFill {
                    slot: located.slot,
                    tokens: Arc::clone(&prefix.tokens),
                });
            }
        }
        Ok(fills)
    }

    /// Record that a prefix prefill made `count` tokens visible in `slot`.
    pub(crate) fn prefilled(&mut self, slot: u32, count: u64) {
        for coordinates in self.branches.values_mut() {
            if coordinates.slot == slot && coordinates.visible == 0 {
                coordinates.visible = count;
            }
        }
    }

    /// The denoising rows of `set`: each branch's slot and visible prefix,
    /// with its temporal position. The conditioned branch continues the
    /// request's logical position; an unconditional branch follows its
    /// prefix.
    pub(crate) fn branch_rows(
        &self,
        set: BranchSet,
        logical_position: u64,
    ) -> Result<Vec<(Branch, Coordinates, u64)>> {
        set.iter()
            .map(|branch| {
                let coordinates =
                    self.branches.get(&branch).copied().ok_or_else(|| {
                        Error::Invalid("guidance branch has no cache prefix".into())
                    })?;
                let temporal = if branch == Branch::Conditioned {
                    logical_position
                } else {
                    coordinates.visible
                };
                Ok((branch, coordinates, temporal))
            })
            .collect()
    }
}

/// The open trajectory of each request slot.
#[derive(Debug, Default)]
pub(crate) struct Trajectories {
    slots: HashMap<usize, ImageTrajectory>,
}

impl Trajectories {
    /// The open trajectory of the request at `slot`, if it is still that
    /// request's (a reused slot never sees an earlier request's trajectory).
    pub(crate) fn get(&mut self, slot: usize, key: RequestKey) -> Option<&mut ImageTrajectory> {
        self.slots
            .get_mut(&slot)
            .filter(|trajectory| trajectory.key == key)
    }

    pub(crate) fn open(
        &mut self,
        slot: usize,
        trajectory: ImageTrajectory,
    ) -> &mut ImageTrajectory {
        self.slots.entry(slot).insert_entry(trajectory).into_mut()
    }

    /// Forget the trajectories of slots whose requests retired or restarted.
    pub(crate) fn release(&mut self, slots: &[usize]) {
        for slot in slots {
            self.slots.remove(slot);
        }
    }
}

impl crate::executor::lifecycle::RequestStorage for Trajectories {
    fn admit_slots(
        &mut self,
        slots: &[usize],
        _jobs: &crate::executor::lifecycle::LaneJobs,
    ) -> Result<()> {
        // A reused slot never continues an earlier request's trajectory.
        self.release(slots);
        Ok(())
    }

    fn release_buffers(
        &mut self,
        _buffers: &std::collections::HashSet<uniserve_ipc::BufferId>,
        _jobs: &crate::executor::lifecycle::LaneJobs,
    ) -> Result<()> {
        Ok(())
    }

    fn release_requests(
        &mut self,
        _requests: &std::collections::HashSet<uniserve_ipc::RequestKey>,
        _retained: &std::collections::HashSet<uniserve_ipc::BufferId>,
        _jobs: &crate::executor::lifecycle::LaneJobs,
    ) -> Result<()> {
        Ok(())
    }

    fn retirement_ready(
        &mut self,
        _buffers: &std::collections::HashSet<uniserve_ipc::BufferId>,
        _requests: &std::collections::HashSet<uniserve_ipc::RequestKey>,
        _retained: &std::collections::HashSet<uniserve_ipc::BufferId>,
    ) -> Result<bool> {
        Ok(true)
    }

    fn retire_slots(
        &mut self,
        slots: &[usize],
        _jobs: &crate::executor::lifecycle::LaneJobs,
    ) -> Result<()> {
        self.release(slots);
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use uniserve_core::RequestId;

    use super::*;

    fn plan(masks: &[u8]) -> GuidancePlan {
        GuidancePlan {
            params: Arc::new(ImageParams::default()),
            timesteps: (0..masks.len()).map(|step| step as f32).collect(),
            branches: masks
                .iter()
                .map(|&mask| BranchSet::from_mask(mask).unwrap())
                .collect(),
        }
    }

    fn trajectory(masks: &[u8]) -> ImageTrajectory {
        let sources = [
            BranchSource::Conditioning,
            BranchSource::NegativeOrStart,
            BranchSource::Start,
        ];
        let mut trajectory = ImageTrajectory::new(
            RequestKey::new(1, RequestId(7), 1),
            (64, 64),
            plan(masks),
            sources
                .into_iter()
                .map(|source| (source, prefix(source)))
                .collect(),
        );
        trajectory.begin_call(Coordinates {
            slot: 3,
            visible: 40,
            capacity: 64,
        });
        trajectory
    }

    fn prefix(source: BranchSource) -> Prefix {
        match source {
            BranchSource::Conditioning => Prefix {
                tokens: Arc::from(Vec::new()),
                reuses_conditioning: true,
            },
            BranchSource::NegativeOrStart => Prefix {
                tokens: Arc::from(vec![1, 2, 3]),
                reuses_conditioning: false,
            },
            BranchSource::Start => Prefix {
                tokens: Arc::from(vec![1]),
                reuses_conditioning: false,
            },
        }
    }

    fn table(slot: u32, visible: u64) -> Result<Coordinates> {
        Ok(Coordinates {
            slot,
            visible,
            capacity: 16,
        })
    }

    #[test]
    fn the_negative_branch_conditions_on_exactly_the_admitted_tokens() -> Result<()> {
        let given = BranchSource::NegativeOrStart.admitted(&[4, 5]);
        assert_eq!(given.map(|prefix| prefix.tokens.to_vec()), Some(vec![4, 5]));
        assert_eq!(BranchSource::Conditioning.admitted(&[4, 5]), None);
        assert_eq!(BranchSource::Start.admitted(&[4, 5]), None);

        // A request admitted without negative tokens: the engine reserves its
        // alternative prefix no tokens and gives that branch's denoising row
        // no context. The branch conditions on an empty prefix, which fits
        // the slot and needs no prefill.
        let empty = BranchSource::NegativeOrStart
            .admitted(&[])
            .ok_or_else(|| Error::State("the admission fixes no prefix for the negative source"))?;
        assert!(empty.tokens.is_empty() && !empty.reuses_conditioning);
        let mut trajectory = ImageTrajectory::new(
            RequestKey::new(1, RequestId(7), 1),
            (64, 64),
            plan(&[3]),
            [
                (
                    BranchSource::Conditioning,
                    prefix(BranchSource::Conditioning),
                ),
                (BranchSource::NegativeOrStart, empty),
            ]
            .into(),
        );
        trajectory.begin_call(Coordinates {
            slot: 3,
            visible: 40,
            capacity: 64,
        });
        let rows = [
            ForwardSlot {
                slot: 3,
                seq_len: 41,
                query_len: 1,
            },
            ForwardSlot {
                slot: 5,
                seq_len: 1,
                query_len: 1,
            },
        ];
        let fills = trajectory.resolve_branches(
            BranchSet::from_mask(3)?,
            BranchSource::Start,
            &rows,
            |slot, visible| {
                Ok(Coordinates {
                    slot,
                    visible,
                    capacity: 0,
                })
            },
        )?;
        assert!(fills.is_empty());
        Ok(())
    }

    #[test]
    fn branch_sets_keep_evaluation_order_and_require_the_conditioned_branch() {
        let all = BranchSet::from_mask(7).unwrap();
        assert_eq!(all.iter().collect::<Vec<_>>(), Branch::ALL.to_vec());
        assert_eq!(BranchSet::from_mask(5).unwrap().len(), 2);
        assert!(BranchSet::from_mask(2).is_err());
        assert!(BranchSet::from_mask(9).is_err());
    }

    #[test]
    fn a_first_interval_prefills_each_new_prefix_once_per_slot() -> Result<()> {
        let mut trajectory = trajectory(&[7]);
        let (set, _) = trajectory.plan.step(0)?;
        // Two prefix prefills (slots 5 and 6), then three denoising rows.
        let descriptors = [
            ForwardSlot {
                slot: 5,
                seq_len: 3,
                query_len: 3,
            },
            ForwardSlot {
                slot: 6,
                seq_len: 1,
                query_len: 1,
            },
            ForwardSlot {
                slot: 3,
                seq_len: 40,
                query_len: 0,
            },
            ForwardSlot {
                slot: 5,
                seq_len: 3,
                query_len: 3,
            },
            ForwardSlot {
                slot: 6,
                seq_len: 1,
                query_len: 1,
            },
        ];
        let fills = trajectory.resolve_branches(set, BranchSource::Start, &descriptors, table)?;
        assert_eq!(
            fills,
            vec![
                PrefixFill {
                    slot: 5,
                    tokens: Arc::from(vec![1, 2, 3]),
                },
                PrefixFill {
                    slot: 6,
                    tokens: Arc::from(vec![1]),
                },
            ]
        );
        trajectory.prefilled(5, 3);
        trajectory.prefilled(6, 1);

        // The conditioned branch continues the request; the others follow
        // their prefixes.
        let rows = trajectory.branch_rows(set, 41)?;
        let coordinates: Vec<_> = rows
            .iter()
            .map(|(branch, at, temporal)| (*branch, at.slot, at.visible, *temporal))
            .collect();
        assert_eq!(
            coordinates,
            vec![
                (Branch::Conditioned, 3, 40, 41),
                (Branch::TextUnconditional, 5, 3, 3),
                (Branch::ImageUnconditional, 6, 1, 1),
            ]
        );
        Ok(())
    }

    #[test]
    fn a_later_interval_reads_resident_prefixes_without_prefill() -> Result<()> {
        let mut trajectory = trajectory(&[3, 3]);
        let (set, _) = trajectory.plan.step(1)?;
        let descriptors = [
            ForwardSlot {
                slot: 3,
                seq_len: 40,
                query_len: 0,
            },
            ForwardSlot {
                slot: 5,
                seq_len: 3,
                query_len: 0,
            },
        ];
        let fills = trajectory.resolve_branches(set, BranchSource::Start, &descriptors, table)?;
        assert!(fills.is_empty());
        assert_eq!(trajectory.branch_rows(set, 41)?[1].1.visible, 3);

        // A resident prefix whose length disagrees with the tokens is
        // refused.
        trajectory.begin_call(trajectory.conditioning);
        let stale = [
            descriptors[0],
            ForwardSlot {
                slot: 5,
                seq_len: 2,
                query_len: 0,
            },
        ];
        assert!(
            trajectory
                .resolve_branches(set, BranchSource::Start, &stale, table)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn prefixes_beyond_the_slot_capacity_and_short_descriptors_are_refused() {
        let mut trajectory = trajectory(&[3]);
        let set = BranchSet::from_mask(3).unwrap();
        let descriptors = [ForwardSlot {
            slot: 3,
            seq_len: 40,
            query_len: 0,
        }];
        assert!(
            trajectory
                .resolve_branches(set, BranchSource::Start, &descriptors, table)
                .is_err()
        );
        let small = |slot, visible| {
            Ok(Coordinates {
                slot,
                visible,
                capacity: 2,
            })
        };
        let descriptors = [
            descriptors[0],
            ForwardSlot {
                slot: 5,
                seq_len: 3,
                query_len: 3,
            },
        ];
        assert!(
            trajectory
                .resolve_branches(set, BranchSource::Start, &descriptors, small)
                .is_err()
        );
    }

    #[test]
    fn reused_slots_never_see_an_earlier_requests_trajectory() {
        let mut trajectories = Trajectories::default();
        let first = RequestKey::new(1, RequestId(7), 1);
        trajectories.open(
            3,
            ImageTrajectory::new(first, (64, 64), plan(&[1]), HashMap::new()),
        );
        assert!(trajectories.get(3, first).is_some());
        assert!(
            trajectories
                .get(3, RequestKey::new(1, RequestId(8), 1))
                .is_none()
        );
        trajectories.release(&[3]);
        assert!(trajectories.get(3, first).is_none());
    }
}
