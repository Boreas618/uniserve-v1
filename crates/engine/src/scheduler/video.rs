//! A video request's part of the scheduler (`specs/scheduler-design.md` §3.3, §3.8, §4.3, §4.4,
//! §7.3): what it asks of the deployment, how many calls of each phase it makes and when each is
//! ready, the products its calls read and write, and the trajectory it stores when it is
//! preempted.
//!
//! A video request runs a fixed chain of media phases, vllm-omni's step scheduler with the stages
//! around it: a conditioned request reads its condition media (`MediaRead`); every request
//! encodes its prompt (`TextEncode`), whose text encoder computes the vision features of the
//! conditions' vision blocks inside the call, prepares its latents (`LatentPrepare`), which
//! encodes the condition latents inside the call (the model's `nn::Encoder::encode` and
//! `nn::Denoiser::prepare_latents`), denoises them one step per call (`Denoise`), decodes the
//! video in rounds of media units, every rank of the decoder group decoding its share of a round,
//! and the audio once (`VideoDecode`, `AudioDecode`), encodes each decoded round and the audio
//! track (`VideoEncode`, `AudioEncode`) and muxes the encoded rounds into the artifact (`Mux`). No
//! graph object holds this: the calls of a phase are [`Request::video_needed`], whether one more
//! may be given now is [`Request::video_ready`], and both read only the request's counters and
//! its groups.
//!
//! Every product is a block of the buffer pool: one per declared result, allocated at admission
//! for the owner on the group whose phase writes it, in the class of its product kind (the
//! products the model family's pipeline declares, `nn::pipeline::Pipeline::products`, which the
//! groups state, [`pipeline`]). A request's declared results list each kind's results together,
//! in kind order, and a result's position in the request's product table is its index in that
//! list; a media row names the results its call reads and writes by these positions
//! ([`Request::results`]).
//!
//! The conditioning is the request's encoder output (SD §3.3, §4.3): keyed by what its text
//! encoder reads (its prompt and the conditions its vision blocks present) and the encoder's
//! identity, pinned at admission where cached instead of allocated, and committed once encoded.
//! A request whose admission found it makes no text encoding, and its latent preparation reads
//! the pinned block at the same position.

use std::iter;
use std::ops::Range;

use crate::batch::{Extent, Phase, Shape};
use crate::runner::decoder::RoundConfig;
use nn::pipeline::{Extent as Blocks, Pipeline, Product, ProductSpec};
use proto::request::{ConditionExtent, DiffusionRequest, Event, MigrationCause, RequestId};
use storage::Key;
use storage::key::latent_content;
use storage::runtime::{self, Cause, Completion, GroupId, OwnerId};

use super::update::migratable;
use super::{Request, RequestStatus, Scheduler, WorkerGroup};

/// The media pipeline the groups serve parts of, as their ranks state it
/// (`Capacity::pipeline`; every group of a media deployment states the same, which the scheduler
/// checks as it starts): the phases a video request offers, its products in the order of their
/// classes and the blocks the deployment places. `None` for a text deployment.
pub(super) fn pipeline(groups: &[WorkerGroup]) -> Option<&Pipeline> {
    groups
        .iter()
        .find_map(|group| group.capacity.pipeline.as_ref())
}

/// The products of the groups' pipeline, in the order of their classes; none for a text
/// deployment.
pub(super) fn products(groups: &[WorkerGroup]) -> &[ProductSpec] {
    pipeline(groups).map_or(&[], |pipeline| pipeline.products.as_slice())
}

/// A video request's calls, products and trajectory.
impl Request {
    /// The video request this request is; `None` for a token request.
    pub(super) fn video(&self) -> Option<&DiffusionRequest> {
        match &self.input {
            proto::request::Request::Diffusion(video) => Some(video),
            proto::request::Request::Generation(_) => None,
        }
    }

    /// Whether this is a video request.
    pub(super) fn is_video(&self) -> bool {
        self.video().is_some()
    }

    /// How many calls of `phase` a video request makes in all (SD §3.3); 0 for a token request
    /// and for a phase a video request does not run.
    ///
    /// The prompt is encoded once, unless the admission's `Pin` found its conditioning. The
    /// denoising steps are those the group that denoises states. A video decode round covers the
    /// decoder group's width in media units. Each decoded round is encoded once and handed to the
    /// muxer once, before a final muxing call that assembles the artifact.
    pub(super) fn video_needed(&self, phase: Phase, groups: &[WorkerGroup]) -> u32 {
        let Some(video) = self.video() else {
            return 0;
        };
        match phase {
            Phase::MediaRead => u32::from(!video.conditions.is_empty()),
            Phase::TextEncode => u32::from(!self.found_conditioning()),
            Phase::LatentPrepare | Phase::AudioDecode | Phase::AudioEncode => 1,
            Phase::Denoise => self
                .serving(Phase::Denoise, groups)
                .map_or(0, |group| group.capacity.denoise_steps),
            Phase::VideoDecode | Phase::VideoEncode => {
                video.sampling.video_units.div_ceil(self.width(groups))
            }
            Phase::Mux => self
                .video_needed(Phase::VideoEncode, groups)
                .saturating_add(1),
            _ => 0,
        }
    }

    /// Whether a video request may be given one more call of `phase` now (SD §3.3, with the rules
    /// of `S/execution/media.rs:269-403` and the splice tail of gap G11). It has calls of the
    /// phase left, every crossing of a product the call reads into its group was answered
    /// `Replicated` (BP §15 "A product read in another process"), and:
    ///
    /// - every call but the media read waits until the read completed: the text encoding reads
    ///   the vision pixels it wrote and the latent preparation the condition pixels and samples,
    ///   and nothing later is offered before them;
    /// - the latent preparation waits until the text encoding is handed over (at once when the
    ///   admission found the conditioning), and a denoising step until the preparation is, with
    ///   at most the denoising group's call window of steps in flight (`Capacity::call_window`:
    ///   the step its denoiser runs and those its queue depth holds behind it, so that the
    ///   denoiser never waits for the host between steps and a preemption meets the request at a
    ///   step boundary within that many steps);
    /// - the decoders wait until every denoising step is handed over, and a video decode round
    ///   until the round before it completed: its first unit splices onto the tail of that
    ///   round's last unit;
    /// - a video encode round waits until its decode round completed, the audio encode until the
    ///   audio decode did, and a muxing call until the encode round it hands over did, one muxing
    ///   call at a time; the final one waits for every encode round and the audio encode.
    ///
    /// A predecessor is handed over once its calls are scheduled when it runs on the same group
    /// as `phase` and every rank of that group that runs `phase` runs it too ([`ordered`]): each
    /// such rank's worker runs the two in submission order. It is handed over once its calls
    /// completed otherwise: nothing orders another group's write before the read, nor the write
    /// of a rank whose own worker passes the read (a decoder rank that does not denoise receives
    /// its latents from its process's first denoising rank, `BufferMemory::share`).
    pub(super) fn video_ready(&self, phase: Phase, groups: &[WorkerGroup]) -> bool {
        if self.video().is_none() || phase.index() >= Phase::COUNT {
            return false;
        }
        let needed = |phase: Phase| self.video_needed(phase, groups);
        let scheduled = |phase: Phase| self.scheduled[phase.index()];
        let completed = |phase: Phase| self.completed[phase.index()];
        let in_flight = |phase: Phase| scheduled(phase).saturating_sub(completed(phase));
        let done = |phase: Phase| completed(phase) >= needed(phase);
        let handed = |before: Phase| {
            let group = self.placement[before.index()];
            let along = group.is_some()
                && group == self.placement[phase.index()]
                && self
                    .serving(phase, groups)
                    .is_some_and(|group| ordered(group, before, phase));
            let count = if along {
                scheduled(before)
            } else {
                completed(before)
            };
            count >= needed(before)
        };

        if scheduled(phase) >= needed(phase) {
            return false;
        }
        if phase != Phase::MediaRead && !done(Phase::MediaRead) {
            return false;
        }
        if !self.crossed(phase, groups) {
            return false;
        }
        match phase {
            Phase::MediaRead | Phase::TextEncode => true,
            Phase::LatentPrepare => handed(Phase::TextEncode),
            Phase::Denoise => {
                let window = |group: &WorkerGroup| group.capacity.call_window.max(1);
                handed(Phase::LatentPrepare)
                    && self
                        .serving(phase, groups)
                        .is_some_and(|group| in_flight(Phase::Denoise) < window(group))
            }
            Phase::VideoDecode => handed(Phase::Denoise) && in_flight(Phase::VideoDecode) == 0,
            Phase::AudioDecode => handed(Phase::Denoise),
            Phase::VideoEncode => completed(Phase::VideoDecode) > scheduled(Phase::VideoEncode),
            Phase::AudioEncode => done(Phase::AudioDecode),
            Phase::Mux => {
                let next = scheduled(Phase::Mux);
                let handed_over = if next < needed(Phase::VideoEncode) {
                    completed(Phase::VideoEncode) > next
                } else {
                    done(Phase::VideoEncode) && done(Phase::AudioEncode)
                };
                in_flight(Phase::Mux) == 0 && handed_over
            }
            _ => false,
        }
    }

    /// The media units call `call` of `phase` covers: a video decode round and the encode of what
    /// it decoded take the decoder group's width of the request's media units, the last round
    /// what remains. Empty for every other call. A media row's `unit` is the start.
    pub(super) fn round_units(
        &self,
        phase: Phase,
        call: u32,
        groups: &[WorkerGroup],
    ) -> Range<u32> {
        let Some(video) = self.video() else {
            return 0..0;
        };
        if !matches!(phase, Phase::VideoDecode | Phase::VideoEncode) {
            return 0..0;
        }
        let (width, units) = (self.width(groups), video.sampling.video_units);
        let start = call.saturating_mul(width);
        if start >= units {
            return 0..0;
        }
        start..start.saturating_add(width).min(units)
    }

    /// The call of a video decode or encode round whose first unit is `unit`, a media row's
    /// `unit`: the round's index among the request's calls of its phase ([`Request::round_units`]).
    pub(super) fn round_of(&self, unit: u32, groups: &[WorkerGroup]) -> u32 {
        unit / self.width(groups)
    }

    /// The declared results call `call` of `phase` reads, then those it writes, as positions in
    /// the request's product table, and how many it reads: a media row's `results` and `reads`
    /// (SD §2.3; `S/execution/media.rs:1043-1150`).
    ///
    /// The text encoding reads the vision pixels of the conditions' vision blocks, whose features
    /// it computes inside the call. The latent preparation reads the conditioning, the condition
    /// pixels and the condition samples, in the order the denoiser tells them apart by class, and
    /// encodes the condition latents inside the call. The video units are one block, reserved
    /// whole at admission, of which each round writes the slice of its units (from its row's
    /// `unit`). A video decode round after the first reads the splice tail of the unit before its
    /// own, which the round before it wrote, and every round writes the tail of each unit it
    /// decodes, one block per unit. An encode round writes the encoded unit of each unit of its
    /// round, one block per unit, which the process that decoded the unit encodes, so that each
    /// block is written in one process. Only the last denoising step writes the latents the
    /// decoders read; the steps before it keep the trajectory in the slot. A muxing call hands
    /// over the encoded units of the round it follows, and the final one reads nothing.
    pub(super) fn results(
        &self,
        phase: Phase,
        call: u32,
        groups: &[WorkerGroup],
    ) -> (Vec<u32>, u32) {
        let last = call.saturating_add(1) == self.video_needed(phase, groups);
        let (mut reads, mut writes) = (Vec::new(), Vec::new());
        for spec in products(groups) {
            let product = spec.product;
            match spec.extent {
                // A product of one block per request: its reading phase reads it, and its
                // writing phase writes it, every call its slice of it, except that only the last
                // denoising step writes the latents: the steps before it keep the trajectory in
                // the slot.
                Blocks::Request => {
                    let declared = self.result(product, 0, groups);
                    if spec.reader == phase && spec.writer != phase {
                        reads.extend(declared);
                    }
                    if spec.writer == phase && (phase != Phase::Denoise || last) {
                        writes.extend(declared);
                    }
                }
                // A product of one block per media unit: the writing round writes the blocks of
                // its units, the reading call of the same index reads them, and a product its own
                // phase reads is the carry the round before left at the unit before the round's.
                Blocks::Unit => {
                    let round = self.round_units(spec.writer, call, groups);
                    let blocks = |units: Range<u32>| -> Vec<u32> {
                        units
                            .filter_map(|unit| self.result(product, unit, groups))
                            .collect()
                    };
                    if spec.writer == phase {
                        if spec.reader == phase {
                            let before = round.start.checked_sub(1);
                            reads
                                .extend(before.and_then(|unit| self.result(product, unit, groups)));
                        }
                        writes.extend(blocks(round));
                    } else if spec.reader == phase && call < self.video_needed(spec.writer, groups)
                    {
                        reads.extend(blocks(round));
                    }
                }
            }
        }

        let count = u32::try_from(reads.len()).unwrap_or(u32::MAX);
        reads.append(&mut writes);
        (reads, count)
    }

    /// The captured media shape the request rides on the group that denoises it, with the bytes
    /// of a slot's trajectory and of a block of each product kind there: the smallest of the
    /// group's shapes that holds the request (`Capacity::media_bytes`).
    pub(super) fn media_shape<'a>(
        &self,
        groups: &'a [WorkerGroup],
    ) -> Option<&'a (Shape, u64, Vec<u64>)> {
        let video = self.video()?;
        captured(self.serving(Phase::Denoise, groups)?, video)
    }

    /// Whether a video request stores its trajectory before it leaves (SD §3.8, §7.3
    /// "Preemption"; `S/preemption/media.rs:150-175`): a drain or a failed crossing preempted it
    /// (its terminal is `Migratable` with the cause `Drain` or `Transfer`), the daemon gave it a
    /// trajectory identity, it completed at least one denoising step, and it is attached to the
    /// group that denoises it, whose slot holds the trajectory. A lost group's trajectory is gone
    /// with it, and step 0 is the request's noise, which any engine draws again.
    pub(super) fn stores_trajectory(&self) -> bool {
        let Some(video) = self.video() else {
            return false;
        };
        let preempted = matches!(
            self.terminal,
            Some(Event::Migratable {
                cause: MigrationCause::Drain | MigrationCause::Transfer,
                ..
            })
        );
        let attached = self.placement[Phase::Denoise.index()]
            .and_then(|group| self.slot_on(group))
            .is_some();
        preempted
            && video.trajectory.id.is_some()
            && self.completed[Phase::Denoise.index()] > 0
            && attached
    }

    /// The key a video request's trajectory at its completed step is stored under,
    /// `latent_content(seed, trajectory, steps)`, with `seed` the latent seed of what the serving
    /// epoch's denoising computes (SD §4.3; `S/trajectories.rs:487`). `None` for a token request
    /// and for a request that names no trajectory.
    pub(super) fn trajectory_key(&self, seed: u64) -> Option<Key> {
        let video = self.video()?;
        let steps = self.completed[Phase::Denoise.index()];
        let trajectory = video.trajectory.id?;
        Some(Key(latent_content(seed, trajectory.get(), steps)))
    }

    /// The key a resumed video request's admission restores its trajectory from: when it names
    /// `0 < denoised_steps < steps`, the trajectory after those steps, under the latent seed of
    /// the epoch that stored it when the request names one (`state_seed`) and `seed` otherwise
    /// (SD §7.3 "Resume"; `S/preemption/media.rs:318-325`, `S/trajectories.rs:817`). `None` for a
    /// request that starts from noise.
    pub(super) fn restore_key(&self, seed: u64, groups: &[WorkerGroup]) -> Option<Key> {
        let trajectory = &self.video()?.trajectory;
        let steps = trajectory.denoised_steps;
        if steps == 0 || steps >= self.video_needed(Phase::Denoise, groups) {
            return None;
        }
        let seed = trajectory.state_seed.unwrap_or(seed);
        Some(Key(latent_content(seed, trajectory.id?.get(), steps)))
    }

    /// Whether a video request runs to its end instead of being preempted: every denoising step
    /// is scheduled, so what remains is bounded (SD §3.8, §7.3 "Preemption"). False for a token
    /// request.
    pub(super) fn runs_to_end(&self, groups: &[WorkerGroup]) -> bool {
        self.is_video()
            && self.scheduled[Phase::Denoise.index()] >= self.video_needed(Phase::Denoise, groups)
    }

    /// The progress a video request reports once a call of `phase` was applied
    /// (`S/execution/results.rs:308-333`): `preparing` after the read of its conditions or the
    /// encoding of its prompt; after its preparation or a step, `denoising` while steps remain and
    /// `decoding` once none does; `decoding` after a decode or an encode of its media;
    /// `finalizing` after a muxing call, the final one included; with the steps its trajectory
    /// completed. `None` for a token request and for a phase a video request does not run.
    pub(super) fn progress(&self, phase: Phase, groups: &[WorkerGroup]) -> Option<Event> {
        self.video()?;
        let denoised = self.completed[Phase::Denoise.index()];
        let name = match phase {
            Phase::MediaRead | Phase::TextEncode => "preparing",
            Phase::LatentPrepare | Phase::Denoise
                if denoised >= self.video_needed(Phase::Denoise, groups) =>
            {
                "decoding"
            }
            Phase::LatentPrepare | Phase::Denoise => "denoising",
            Phase::VideoDecode | Phase::AudioDecode | Phase::VideoEncode | Phase::AudioEncode => {
                "decoding"
            }
            Phase::Mux => "finalizing",
            _ => return None,
        };
        Some(Event::MediaProgress {
            phase: name.to_owned(),
            completed_steps: denoised,
        })
    }

    /// The request's declared results, in position order: per result, the phase whose call
    /// writes it and its product kind, an index into the pipeline's products ([`products`]).
    fn declared(&self, groups: &[WorkerGroup]) -> Vec<(Phase, usize)> {
        products(groups)
            .iter()
            .enumerate()
            .flat_map(|(kind, spec)| {
                let count = usize::try_from(self.declares(spec)).unwrap_or(0);
                iter::repeat_n((spec.writer, kind), count)
            })
            .collect()
    }

    /// The position of a video request's conditioning in its product table: where its admission
    /// pins or allocates it, its text encoding writes it and its latent preparation reads it.
    pub(super) fn conditioning_position(&self, groups: &[WorkerGroup]) -> Option<u32> {
        self.result(Product::Conditioning, 0, groups)
    }

    /// Whether a video request holds its conditioning without encoding it: its admission's `Pin`
    /// found it under its key (`encodings`, its one encoder output), before any text encoding
    /// completed, which holds it from then on too.
    fn found_conditioning(&self) -> bool {
        let held = self.encodings.first().is_some_and(|(_, at)| at.is_some());
        held && self.completed[Phase::TextEncode.index()] == 0
    }

    /// The position of the `index`-th declared result of `product`, if the request declares it.
    fn result(&self, product: Product, index: u32, groups: &[WorkerGroup]) -> Option<u32> {
        let mut position: u32 = 0;
        for spec in products(groups) {
            let count = self.declares(spec);
            if spec.product == product {
                return (index < count).then(|| position.saturating_add(index));
            }
            position = position.saturating_add(count);
        }
        None
    }

    /// How many results of the product `spec` declares a video request declares
    /// (`S/graph.rs:140-242`): one per media unit of a product of units (a splice tail, commit
    /// `e69a91146`, and an encoded unit), a condition product when its conditions have what it
    /// holds, and one of every other product.
    fn declares(&self, spec: &ProductSpec) -> u32 {
        let Some(video) = self.video() else {
            return 0;
        };
        let has = |extent: fn(&ConditionExtent) -> u64| u32::from(total(video, extent) > 0);
        match (spec.extent, spec.product) {
            (Blocks::Unit, _) => video.sampling.video_units,
            (_, Product::ConditionPixels) => has(|condition| condition.pixel_bytes / 3),
            (_, Product::ConditionSamples) => has(|condition| condition.audio_samples),
            (_, Product::VisionPixels) => has(|condition| condition.vision_patches),
            _ => 1,
        }
    }

    /// The group that serves `phase` for the request: the one it is placed on, or before its
    /// placement the first group that serves the phase.
    fn serving<'a>(&self, phase: Phase, groups: &'a [WorkerGroup]) -> Option<&'a WorkerGroup> {
        match self.placement.get(phase.index()).copied().flatten() {
            Some(id) => groups.iter().find(|group| group.id() == id),
            None => groups.iter().find(|group| group.serves(phase)),
        }
    }

    /// The media units one video decode round covers on the group that decodes the request's
    /// video: the ranks of its block that serves video decodes, each taking the units the block
    /// deals a rank (`BlockConfig::rank_units`, `S/execution/media.rs:564-572`); 1 when the group
    /// places no such block. Every rank of the group decodes its share of a round in one call,
    /// the rounds' shares dealt to the ranks in order and in reverse alternately (SD §3.3).
    fn width(&self, groups: &[WorkerGroup]) -> u32 {
        let Some(block) = self
            .serving(Phase::VideoDecode, groups)
            .and_then(|group| decoder_block(groups, &group.config))
        else {
            return 1;
        };
        let units = block.ranks.len().saturating_mul(block.rank_units());
        u32::try_from(units).unwrap_or(u32::MAX).max(1)
    }

    /// Whether every crossing of a product the next call of `phase` reads into the call's group
    /// was answered `Replicated` (`Request::crossings`): the call then reads the product in each
    /// memory its group reads it in. True for a call that reads nothing another process wrote.
    fn crossed(&self, phase: Phase, groups: &[WorkerGroup]) -> bool {
        let group = self.placement[phase.index()];
        if group.is_none() || self.crossings.is_empty() {
            return true;
        }
        let call = self.scheduled[phase.index()];
        let (results, reads) = self.results(phase, call, groups);
        let read = &results[..(reads as usize).min(results.len())];
        self.crossings
            .iter()
            .filter(|(at, into, ..)| Some(*into) == group && read.contains(at))
            .all(|(.., state)| *state == Some(true))
    }

    /// Whether a call reads declared result `at`: the request makes calls of the phase that reads
    /// its kind. A request whose admission found its conditioning reads no vision pixels.
    pub(super) fn reads_result(&self, at: u32, groups: &[WorkerGroup]) -> bool {
        self.declared(groups)
            .get(at as usize)
            .and_then(|&(_, kind)| products(groups).get(kind))
            .is_some_and(|spec| self.video_needed(spec.reader, groups) > 0)
    }

    /// Whether the call that writes declared result `at` was applied: the call of the result's
    /// phase that writes it (for a product of one block per media unit, the round of the block's
    /// unit; the last call of the phase otherwise), counted in order or ahead of an earlier
    /// call. A conditioning the admission's `Pin`
    /// found was written before the request arrived.
    fn written(&self, at: u32, groups: &[WorkerGroup]) -> bool {
        let Some(&(phase, kind)) = self.declared(groups).get(at as usize) else {
            return false;
        };
        if Some(at) == self.conditioning_position(groups) && self.found_conditioning() {
            return true;
        }
        // A block of a product of units is written by the round of its unit.
        let spec = products(groups).get(kind);
        let unit = spec
            .filter(|spec| spec.extent == Blocks::Unit)
            .and_then(|spec| self.result(spec.product, 0, groups))
            .map(|first| at.saturating_sub(first));
        let call = match unit {
            Some(unit) => self.round_of(unit, groups),
            None => self.video_needed(phase, groups).saturating_sub(1),
        };
        self.completed[phase.index()] > call || self.early.contains(&(phase, call))
    }
}

/// Whether every rank of `group` that runs calls of `phase` runs calls of `before` too, by the
/// blocks the deployment places on its ranks (`WorkerGroup::positions`), so that each such rank's
/// worker
/// runs a call of `before` ahead of the call of `phase` submitted after it: a decoder rank that
/// denoises reads the latents its own last step wrote into its run. A rank that does not run
/// `before` passes its batch, and nothing orders its call of `phase` after the writes of `before`
/// on other ranks. A group whose configuration places no block runs every phase it serves on
/// every rank.
fn ordered(group: &WorkerGroup, before: Phase, phase: Phase) -> bool {
    group
        .positions
        .iter()
        .all(|phases| !phases.contains(&phase) || phases.contains(&before))
}

/// The block of `config` whose ranks serve video decodes, by the pipeline's declaration
/// ([`pipeline`]); `None` for a group that places none.
pub(super) fn decoder_block<'a>(
    groups: &[WorkerGroup],
    config: &'a proto::config::WorkerGroupConfig,
) -> Option<&'a proto::config::BlockConfig> {
    pipeline(groups)?
        .serving(Phase::VideoDecode)
        .find_map(|block| config.blocks.get(&block.name))
}

/// A video request against the deployment, and the sizes of its blocks.
impl Scheduler {
    /// Checks a video request against the deployment (SD §7.2 "Validation of a video request";
    /// `S/admission.rs:590-609`, `S/graph.rs:140-256`), in this order, refusing it with the
    /// message of the first rule it breaks:
    ///
    /// - the request's own consistency (`DiffusionRequest::validate`);
    /// - a group serves every phase that produces the artifact, `Mux` among them;
    /// - a group denoises, with a positive number of steps;
    /// - a conditioned request finds a group that reads its conditions;
    /// - each extent of its products fits the protocol's `u32` dimensions;
    /// - a group that denoises captured a shape that holds it and states the bytes of every
    ///   product kind there.
    ///
    /// # Errors
    ///
    /// The message the request is rejected with as `Invalid`.
    pub(super) fn validate_video(&self, video: &DiffusionRequest) -> Result<(), String> {
        video.validate().map_err(|error| error.to_string())?;
        let served = |phase: Phase| self.groups.iter().any(|group| group.serves(phase));

        let produces = [
            Phase::TextEncode,
            Phase::LatentPrepare,
            Phase::VideoDecode,
            Phase::AudioDecode,
            Phase::VideoEncode,
            Phase::AudioEncode,
            Phase::Mux,
        ];
        if !produces.into_iter().all(served) {
            return Err("worker does not provide the video media stages".to_owned());
        }
        let denoisers = || {
            self.groups
                .iter()
                .filter(|group| group.serves(Phase::Denoise) && group.capacity.denoise_steps > 0)
        };
        if denoisers().next().is_none() {
            return Err("worker reports no video denoiser".to_owned());
        }

        // The text encoder and the latent preparation encode the conditions inside their calls:
        // only their read is a phase of its own.
        if !video.conditions.is_empty() && !served(Phase::MediaRead) {
            return Err("the deployment's denoiser takes no conditions".to_owned());
        }

        let extents = [
            (
                "condition pixels",
                total(video, |condition| condition.pixel_bytes / 3),
            ),
            (
                "condition samples",
                total(video, |condition| condition.audio_samples),
            ),
            (
                "vision patches",
                total(video, |condition| condition.vision_patches),
            ),
        ];
        if let Some((quantity, _)) = extents
            .into_iter()
            .find(|(_, extent)| u32::try_from(*extent).is_err())
        {
            return Err(format!(
                "the request's {quantity} exceed the protocol's extents"
            ));
        }

        if !denoisers().any(|group| captured(group, video).is_some()) {
            let sampling = &video.sampling;
            return Err(format!(
                "loaded media stages cannot hold the request: no captured denoising shape holds \
                 {} frames at {}x{} with {} prompt rows and {} condition rows",
                sampling.num_frames,
                sampling.width,
                sampling.height,
                video.prompt_token_ids.len(),
                sampling.condition_rows
            ));
        }
        Ok(())
    }

    /// Per declared result of a video request, in position order: the phase whose call writes
    /// it, its product class, and the units of its block, rounded up from the bytes the group
    /// that denoises states for the request's captured shape to the unit of the class's region on
    /// the writing group (SD §4.4 "Classes", "Products"). The request's admission asks for each
    /// result by an `Allocate` at its position for its owner on the writing group, with these
    /// units. `None` when a block cannot be sized: no captured shape holds the request, or the
    /// writing group states no region of the class.
    pub(super) fn declared_results(&self, request: &Request) -> Option<Vec<(Phase, u8, u32)>> {
        let (_, _, bytes) = request.media_shape(&self.groups)?;
        request
            .declared(&self.groups)
            .into_iter()
            .map(|(phase, kind)| {
                let class = self.product_class(kind)?;
                let group = request.serving(phase, &self.groups)?;
                let region = *group.regions(class).first()?;
                let units = self.units_in(region, *bytes.get(kind)?)?;
                Some((phase, class, units))
            })
            .collect()
    }

    /// A video request's trajectory's size in units of the host latent region, the host region
    /// of the class of stored trajectories: the trajectory bytes of the request's captured shape,
    /// rounded up, or the class's fixed host block size when it states one (SD §3.8, §4.4).
    /// `None` when no captured shape holds the request or the class has no host region.
    pub(super) fn trajectory_units(&self, request: &Request) -> Option<u32> {
        let (_, bytes, _) = request.media_shape(&self.groups)?;
        let (region, units) = self.classes.get(usize::from(self.trajectory))?.host?;
        if units > 0 {
            return Some(units);
        }
        self.units_in(region, *bytes)
    }

    /// The class of a video request's conditioning, the product class of its kind.
    pub(super) fn conditioning_class(&self) -> Option<u8> {
        let kind = pipeline(&self.groups)?.kind(Product::Conditioning)?;
        self.product_class(kind)
    }

    /// The class of product kind `kind`: the `kind`-th of the classes just before the class of
    /// stored trajectories, one per product of the pipeline.
    pub(super) fn product_class(&self, kind: usize) -> Option<u8> {
        let first = usize::from(self.trajectory).checked_sub(products(&self.groups).len())?;
        u8::try_from(first.checked_add(kind)?).ok()
    }

    /// The units of the session's region `region` (an index into its regions) that hold `bytes`
    /// on one rank, a part of each unit (`Region::part_bytes`): at least one, since a block has a
    /// size.
    fn units_in(&self, region: u8, bytes: u64) -> Option<u32> {
        let (_, stated) = self.regions.get(usize::from(region))?;
        let units = bytes.div_ceil(stated.part_bytes.max(1)).max(1);
        u32::try_from(units).ok()
    }
}

/// A video request's crossings (SD §3.3, §4.1; BP §15 "A product read in another process"): a
/// product block lies in the product buffers of the ranks that back it, each main process's ranks
/// reach those of their process alone, and a group that reads a block in a process its writer did
/// not write it in receives it by a `Replicate`, which a Transfer Server runs between the two
/// processes' memories. The scheduler asks for it once the writing call's output is applied,
/// stamped by the executors of the groups that may hold the block's entry, and offers the reading
/// call only once it was answered `Replicated`.
impl Scheduler {
    /// The crossings the calls of `request`, as it is placed, wait for, none of them asked
    /// (`Request::crossings`): per declared result its writer writes in one memory
    /// ([`Scheduler::writer_region`]), the group placed for the phase that reads its kind
    /// ([`products`]), when that group reads the class in another memory too.
    pub(super) fn crossings_of(&self, request: &Request) -> Vec<(u32, GroupId, u8, Option<bool>)> {
        let mut crossings = Vec::new();
        for (at, (_, kind)) in (0u32..).zip(request.declared(&self.groups)) {
            let Some(reader) = products(&self.groups).get(kind).map(|spec| spec.reader) else {
                continue;
            };
            let reading = request.placement[reader.index()].and_then(|id| self.group(id));
            let (Some(class), Some(group)) = (self.product_class(kind), reading) else {
                continue;
            };
            let Some(from) = self.writer_region(request, at, kind) else {
                continue;
            };
            let (Some(written), Some(&(number, _))) = (
                self.region_memory(from),
                self.regions.get(usize::from(from)),
            ) else {
                continue;
            };
            let elsewhere = self.reading_memories(group, class);
            if elsewhere.iter().any(|memory| *memory != written) {
                crossings.push((at, group.id(), number, None));
            }
        }
        crossings
    }

    /// The region (an index into the session's regions) of the run the writer of `request`'s
    /// declared result `at`, of product kind `kind`, writes it in, where it writes it in one
    /// memory (SD §3.3): an encoded unit in the process of the rank that decodes its unit, whose
    /// host work encodes it there; the conditioning in its one run, its group's first text
    /// encoder rank's; the condition media, the vision pixels and the audio samples in the run of
    /// the writing group's first rank that serves the writing phase, where its call runs. `None`
    /// for the latents, the video units and the splice tails, each read in the process whose
    /// ranks wrote it (by the decoders of each denoising process, by the encode of the process
    /// that decoded each unit, inside the decode call), and where the layout does not tell the
    /// run.
    fn writer_region(&self, request: &Request, at: u32, kind: usize) -> Option<u8> {
        let spec = *products(&self.groups).get(kind)?;
        let phase = spec.writer;
        let class = self.product_class(kind)?;
        let writer = self.group(request.placement[phase.index()]?)?;
        let regions = writer.regions(class);
        match spec.product {
            Product::VideoLatents
            | Product::AudioLatents
            | Product::VideoUnits
            | Product::VideoTail => None,
            Product::EncodedUnits => {
                let unit = at.checked_sub(request.result(spec.product, 0, &self.groups)?)?;
                let decoder = self.group(request.placement[Phase::VideoDecode.index()]?)?;
                let block = decoder_block(&self.groups, &decoder.config)?;
                // The rounds' deal, by the ranks' positions in the decoder group.
                let ranks = block.ranks.iter().map(|&rank| u32::try_from(rank));
                let round = RoundConfig {
                    ranks: ranks.collect::<Result<_, _>>().ok()?,
                    units: u32::try_from(block.rank_units()).ok()?,
                };
                let rank = usize::try_from(round.rank_of(unit)?).ok()?;
                let memory = self.rank_memory(decoder, rank)?;
                regions
                    .iter()
                    .copied()
                    .find(|&region| self.region_memory(region) == Some(memory))
            }
            _ if regions.len() == 1 => regions.first().copied(),
            _ => {
                // A group's regions of a class it writes on every rank are its ranks' product
                // buffers, in rank order.
                let ranks = writer.config.ranks.len();
                let serving = (0..ranks).find(|&position| {
                    writer
                        .positions
                        .get(position)
                        .is_some_and(|phases| phases.contains(&phase))
                });
                let position = serving.filter(|_| regions.len() == ranks).unwrap_or(0);
                regions.get(position).copied()
            }
        }
    }

    /// The memory of the product buffer of the rank at `position` of `group`: its region among
    /// the group's regions of a product class its phases write on every rank, one per rank in
    /// rank order (`crate::process::ranks::layout`). `None` when it writes no such class or the
    /// region lies in every memory that maps it.
    fn rank_memory(&self, group: &WorkerGroup, position: usize) -> Option<&str> {
        let ranks = group.config.ranks.len();
        let regions = (0..products(&self.groups).len())
            .filter(|&kind| group.serves(products(&self.groups)[kind].writer))
            .filter_map(|kind| self.product_class(kind))
            .map(|class| group.regions(class))
            .find(|regions| regions.len() == ranks)?;
        self.region_memory(*regions.get(position)?)
    }

    /// The memory region `region` (an index into the session's regions) lies in, a rank's
    /// product buffer's main process; `None` for a region whose units exist in every memory that
    /// maps it.
    fn region_memory(&self, region: u8) -> Option<&str> {
        let (_, stated) = self.regions.get(usize::from(region))?;
        (!stated.memory.is_empty()).then_some(stated.memory.as_str())
    }

    /// The memories `group` reads blocks of `class` in: those of its regions and its receiving
    /// regions of the class.
    fn reading_memories(&self, group: &WorkerGroup, class: u8) -> Vec<&str> {
        let receives = group.layout.receives.get(usize::from(class));
        let receives = receives.map_or(&[][..], Vec::as_slice);
        let mut memories: Vec<&str> = group
            .regions(class)
            .iter()
            .chain(receives)
            .filter_map(|&region| self.region_memory(region))
            .collect();
        memories.sort_unstable();
        memories.dedup();
        memories
    }

    /// Per receiving region (an index into the session's regions), the units the crossing of
    /// `request`'s declared result `at` into `group`, read from the run in region `from` (by its
    /// number), takes there (BP §11 `Replicate`): in each memory the group reads the class in
    /// where the block has no run, a run of the block's bytes in the group's receiving region
    /// there. The block's runs lie in its writer group's regions of the class and in `from`.
    pub(super) fn receiving(
        &self,
        request: &Request,
        (at, group, from): (u32, GroupId, u8),
    ) -> Vec<(usize, u32)> {
        let mut taken = Vec::new();
        let Some(&(phase, kind)) = request.declared(&self.groups).get(at as usize) else {
            return taken;
        };
        let writer = request.placement[phase.index()].and_then(|id| self.group(id));
        let (Some(class), Some(reader), Some(writer)) =
            (self.product_class(kind), self.group(group), writer)
        else {
            return taken;
        };
        let from = self
            .region_index(from)
            .and_then(|index| u8::try_from(index).ok());
        let Some((from, written)) = from.and_then(|from| Some((from, self.region_memory(from)?)))
        else {
            return taken;
        };

        // The block's bytes: its units of its writer's region, as its admission sized it.
        let results = self.declared_results(request).unwrap_or_default();
        let Some(&(_, _, units)) = results.get(at as usize) else {
            return taken;
        };
        let unit_bytes = |region: u8| {
            self.regions
                .get(usize::from(region))
                .map_or(1, |(_, stated)| stated.unit_bytes.max(1))
        };
        let bytes = u64::from(units).saturating_mul(unit_bytes(from));

        let receives = reader.layout.receives.get(usize::from(class));
        let receives = receives.map_or(&[][..], Vec::as_slice);
        let mut held: Vec<&str> = writer
            .regions(class)
            .iter()
            .filter_map(|&region| self.region_memory(region))
            .collect();
        held.push(written);
        for memory in self.reading_memories(reader, class) {
            if held.contains(&memory) {
                continue;
            }
            let receive = receives
                .iter()
                .copied()
                .find(|&region| self.region_memory(region) == Some(memory));
            if let Some(receive) = receive {
                let units = bytes.div_ceil(unit_bytes(receive));
                taken.push((
                    usize::from(receive),
                    u32::try_from(units).unwrap_or(u32::MAX),
                ));
            }
        }
        taken
    }

    /// Whether `group` reads the block that `owner` holds at position `at` of `class` in a
    /// memory its writer writes it in, so that its allocation or its pin is handed to the group's
    /// executor (BP §9, obligation 1): a group a crossing brings the block to reads it only by
    /// the crossing's `Replicated` where it reads it in no such memory. True for the owner's own
    /// group, which writes the block, and for a block no crossing brings to the group.
    pub(super) fn reads_where_written(
        &self,
        owner: OwnerId,
        (class, at): (u8, u32),
        group: GroupId,
    ) -> bool {
        let Some(request) = self.request_of(owner) else {
            return true;
        };
        if request
            .attachment(owner)
            .is_some_and(|(home, _)| home == group)
        {
            return true;
        }
        let crossing = request
            .crossings
            .iter()
            .find(|(position, into, ..)| *position == at && *into == group);
        let (Some(&(_, _, from, _)), Some(reader)) = (crossing, self.group(group)) else {
            return true;
        };
        let from = self
            .region_index(from)
            .and_then(|index| u8::try_from(index).ok());
        let written = from.and_then(|from| self.region_memory(from));
        written.is_none_or(|written| self.reading_memories(reader, class).contains(&written))
    }

    /// Asks for each crossing of a running video request whose product its writer's call wrote
    /// and which is not asked yet ([`Request::written`]), once per pass of the loop: a `Replicate`
    /// for the result's owner on its writing group, read from the run its writer wrote, into the
    /// group that reads it (SD §4.1). It is stamped by the executors of the groups that may hold
    /// the result's entry (`Scheduler::submit`), after the `Commit` of a conditioning its text
    /// encoding computed, which its output's application submitted. A request waiting for units
    /// since a shortage asks once they show (`allocate_ahead`).
    pub(super) fn cross(&mut self) {
        let mut asks = Vec::new();
        for id in &self.running {
            let Some(request) = self.requests.get(id) else {
                continue;
            };
            if request.status != RequestStatus::Running || self.short.contains_key(id) {
                continue;
            }
            for (index, &(at, group, from, state)) in request.crossings.iter().enumerate() {
                let due =
                    request.reads_result(at, &self.groups) && request.written(at, &self.groups);
                if state.is_some() || !due {
                    continue;
                }
                let Some(&(phase, kind)) = request.declared(&self.groups).get(at as usize) else {
                    continue;
                };
                let (Some(owner), Some(class)) =
                    (request.owner_on(phase), self.product_class(kind))
                else {
                    continue;
                };
                let replicate = runtime::Request::Replicate {
                    owner,
                    class,
                    at,
                    from,
                    group,
                    after: Vec::new(),
                };
                asks.push((*id, index, replicate));
            }
        }

        for (id, index, replicate) in asks {
            // The runs it takes are not available to the step's later requests.
            for (region, units) in self.taken(&replicate) {
                if let Some(available) = self.available.get_mut(region) {
                    *available = available.saturating_sub(units);
                }
            }
            let crossing = self
                .requests
                .get_mut(&id)
                .and_then(|row| row.crossings.get_mut(index));
            if let Some(crossing) = crossing {
                crossing.3 = Some(false);
            }
            self.submit(replicate);
        }
    }

    /// Applies the answer to a crossing of request `id` (SD §3.5): `Replicated`, the calls that
    /// read the result in the group go on; `Full`, it is asked again once the receiving region's
    /// `available` shows the units, the request short there meanwhile (SD §3.6); `Failed`, the
    /// request ends as migratable with the cause `Transfer`, its trajectory saved where it
    /// completed a step, and the daemon resubmits it, counting the migration against the
    /// request's limit: the engine does not cross again, since a second copy over the same path
    /// has no reason to succeed (SD §3.6); `Refused` with the cause `key`, its conditioning is a
    /// cached block an epoch on trial may not copy into another memory (BP §9, R8), which it
    /// gives up to encode its own.
    pub(super) fn crossing_answered(
        &mut self,
        id: RequestId,
        request: &runtime::Request,
        completion: &Completion,
    ) {
        let runtime::Request::Replicate {
            owner,
            class,
            at,
            group,
            ..
        } = *request
        else {
            return;
        };
        let replicated = matches!(completion, Completion::Replicated { .. });
        let crossing = self.requests.get_mut(&id).and_then(|row| {
            row.crossings
                .iter_mut()
                .find(|(position, into, ..)| *position == at && *into == group)
        });
        if let Some(crossing) = crossing {
            crossing.3 = replicated.then_some(true);
        }

        match completion {
            Completion::Replicated { .. } => self.mark(id),
            Completion::Full { region } => {
                let index = self
                    .region_index(*region)
                    .and_then(|index| u8::try_from(index).ok());
                if let (true, Some(index)) = (self.is_running(id), index) {
                    self.preempt(id, index);
                }
            }
            Completion::Failed => self.finish(id, migratable(MigrationCause::Transfer)),
            Completion::Refused { cause: Cause::Key } => self.encode_again(id, owner, class, at),
            _ => {}
        }
    }

    /// Gives up request `id`'s cached conditioning, which `owner` holds at position `at` of
    /// `class`, by a `Release` through the executors, and asks for a block of its own at that
    /// position, which waits behind the `Release`: its text encoding computes it and its crossing
    /// then copies it from its text encoder rank's run (BP §9, R8).
    fn encode_again(&mut self, id: RequestId, owner: OwnerId, class: u8, at: u32) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let kind = request
            .declared(&self.groups)
            .get(at as usize)
            .map(|&(_, kind)| kind);
        let from = kind.and_then(|kind| self.writer_region(request, at, kind));
        let from = from.and_then(|from| self.regions.get(usize::from(from)));
        let results = self.declared_results(request).unwrap_or_default();
        let (Some(&(number, _)), Some(&(_, _, units)), Some(&(key, _))) =
            (from, results.get(at as usize), request.encodings.first())
        else {
            return;
        };

        if let Some(row) = self.requests.get_mut(&id) {
            if let Some(encoding) = row.encodings.first_mut() {
                encoding.1 = None;
            }
            let crossing = row
                .crossings
                .iter_mut()
                .find(|(position, ..)| *position == at);
            if let Some(crossing) = crossing {
                crossing.2 = number;
            }
        }
        self.unpin(id, &[key]);
        self.submit(runtime::Request::Release {
            owner,
            class,
            at,
            count: 1,
            after: Vec::new(),
        });
        self.submit(runtime::Request::Allocate {
            owner,
            class,
            at,
            count: 1,
            units,
        });
    }
}

/// A sum over a video request's conditions.
fn total(video: &DiffusionRequest, extent: impl Fn(&ConditionExtent) -> u64) -> u64 {
    video
        .conditions
        .iter()
        .map(extent)
        .fold(0, u64::saturating_add)
}

/// The smallest of `group`'s captured shapes that holds a video request and states the bytes of
/// every product kind, with its trajectory's and products' bytes.
fn captured<'a>(
    group: &'a WorkerGroup,
    video: &DiffusionRequest,
) -> Option<&'a (Shape, u64, Vec<u64>)> {
    group
        .capacity
        .media_bytes
        .iter()
        .filter(|(shape, _, bytes)| {
            let declared = group
                .capacity
                .pipeline
                .as_ref()
                .map(|pipeline| pipeline.products.len());
            Some(bytes.len()) == declared && holds(shape, video)
        })
        .min_by_key(|(shape, _, _)| *shape)
}

/// Whether a captured denoising shape holds a video request (`crate::runner::buckets`, the rule a
/// bucket serves a request by): the request's frames and raster when the shape states them, and
/// its prompt and condition rows within the shape's, which has condition rows exactly when the
/// request has conditions.
fn holds(shape: &Shape, video: &DiffusionRequest) -> bool {
    let sampling = &video.sampling;
    let text = u32::try_from(video.prompt_token_ids.len()).unwrap_or(u32::MAX);
    let condition = sampling.condition_rows;
    let rows = |text_rows: u32, condition_rows: u32| {
        text_rows >= text && condition_rows >= condition && (condition_rows > 0) == (condition > 0)
    };
    match shape.extent {
        Extent::Request {
            frames,
            width,
            height,
            text_rows,
            condition_rows,
        } => {
            (frames, width, height) == (sampling.num_frames, sampling.width, sampling.height)
                && rows(text_rows, condition_rows)
        }
        Extent::Latents(layout) => rows(layout.text_rows, layout.condition_rows),
        Extent::None | Extent::Clip { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use crate::batch::{Batch, BatchId, Columns, Extent, Phase, Shape, Variant};
    use proto::config::{
        BlockConfig, BlockDistribution, HostBlock, SchedulerConfig, WorkerGroupConfig,
        WorkerGroupId, WorkerGroupRole, WorkerRank,
    };
    use proto::request::{
        ConditionExtent, DiffusionRequest, DiffusionSamplingParams, EncoderInputs, Event,
        MigrationCause, RequestId, Trajectory,
    };
    use storage::key::{encoder_output, latent_content};
    use storage::runtime::{self, Class, Completion, GroupId, OwnerId, Region, Tier};
    use storage::{Key, QueuePair};

    use crossbeam_channel::Receiver;
    use nn::pipeline::{Pipeline, Product};

    use crate::scheduler::Output;
    use crate::scheduler::cache::DEVICE;
    use crate::scheduler::{Capacity, Executor, Request, RequestStatus, Scheduler, WorkerGroup};

    const A: GroupId = GroupId(0);
    const B: GroupId = GroupId(1);

    /// The fixture's denoising steps, raster and timeline.
    const STEPS: u32 = 4;
    const FRAMES: u32 = 121;
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;

    /// The pipeline the fixture's groups serve: MiniMax-H3's, as its family declares it.
    fn pipeline() -> Pipeline {
        models::minimax_h3::family::pipeline()
    }

    /// Every phase of a video request, in the order the pipeline offers them.
    fn phases() -> Vec<Phase> {
        pipeline().phases
    }

    /// The session's classes: KV blocks, one class per product kind, then stored trajectories,
    /// whose host copies are in region `LATENTS`.
    fn trajectory() -> u8 {
        1 + u8::try_from(pipeline().products.len()).unwrap()
    }
    const LATENTS: u8 = 2;

    /// A captured denoising shape of the fixture's raster and timeline.
    fn shape(text_rows: u32, condition_rows: u32) -> Shape {
        Shape {
            rows: 1,
            tokens: 0,
            variant: Variant::Denoise,
            extent: Extent::Request {
                frames: FRAMES,
                width: WIDTH,
                height: HEIGHT,
                text_rows,
                condition_rows,
            },
        }
    }

    /// What a group serving `phases` states: four steps when it denoises, and three captured
    /// shapes whose trajectories take 30 000, 10 000 and 20 000 bytes, and whose product of kind
    /// `k` takes one byte more than `k` units of the product region (256 bytes).
    fn capacity(phases: &[Phase]) -> Capacity {
        let denoises = phases.contains(&Phase::Denoise);
        let products: Vec<u64> = (0..pipeline().products.len() as u64)
            .map(|kind| 256 * kind + 1)
            .collect();
        let media_bytes = if denoises {
            vec![
                (shape(128, 0), 30_000, products.clone()),
                (shape(64, 0), 10_000, products.clone()),
                (shape(64, 4096), 20_000, products),
            ]
        } else {
            Vec::new()
        };
        Capacity {
            phases: phases
                .iter()
                .fold(0, |bits, phase| bits | 1 << phase.index()),
            slots: 4,
            kv_cache: None,
            buckets: Vec::new(),
            rows: [1; Phase::COUNT],
            max_model_len: 0,
            denoise_steps: if denoises { STEPS } else { 0 },
            bos_token_id: 0,
            eos_token_ids: Vec::new(),
            token_history: 0,
            feature_bytes: 0,
            encoder_identity: String::new(),
            latent_seed: 0,
            media_bytes,
            pipeline: Some(pipeline()),
            call_window: 2,
            cache_groups: Vec::new(),
            seeds: Vec::new(),
        }
    }

    /// A group of two ranks that both encode prompts and denoise, whose video decoder deals units
    /// to both, one unit each in a round.
    fn config(id: GroupId) -> WorkerGroupConfig {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            "video_decoder".to_owned(),
            BlockConfig {
                distribution: Some(BlockDistribution::TemporalUnits),
                ..BlockConfig::parallel(vec![0, 1])
            },
        );
        for block in ["text_encoder", "transformer"] {
            blocks.insert(block.to_owned(), BlockConfig::parallel(vec![0, 1]));
        }
        let rank = WorkerRank {
            host: "localhost".to_owned(),
            device: "cpu".parse().unwrap(),
        };
        WorkerGroupConfig {
            id: WorkerGroupId(format!("video-{}", id.0)),
            role: WorkerGroupRole::Model,
            copies: 1,
            ranks: vec![rank; 2],
            blocks,
            queue_depth: 2,
            mem_fraction_static: None,
        }
    }

    /// The session's statement of a group: every product class in the product region.
    fn layout(id: GroupId) -> runtime::Group {
        runtime::Group {
            id,
            memory: format!("memory-{}", id.0),
            slots: 1,
            regions: (0..=trajectory())
                .map(|class| {
                    if (1..trajectory()).contains(&class) {
                        vec![0]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            receives: Vec::new(),
            parts: Vec::new(),
        }
    }

    fn groups(served: &[(GroupId, &[Phase])]) -> Vec<WorkerGroup> {
        served
            .iter()
            .map(|&(id, phases)| {
                WorkerGroup::new(config(id), capacity(phases), layout(id)).unwrap()
            })
            .collect()
    }

    /// A video request of `units` media units and a prompt of `prompt` tokens.
    fn video(units: u32, prompt: usize, conditions: Vec<ConditionExtent>) -> DiffusionRequest {
        let condition_rows = conditions
            .iter()
            .map(|condition| condition.video_rows() as u32 + condition.audio_rows)
            .sum();
        DiffusionRequest {
            trace: 0,
            request_id: RequestId(1),
            prompt_token_ids: vec![1; prompt],
            conditioning: proto::video::Conditions {
                prompt_tokens: prompt as u32,
                ..proto::video::Conditions::default()
            }
            .encode(),
            media: Vec::new(),
            conditions,
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: FRAMES,
                video_units: units,
                seed: 7,
                width: WIDTH,
                height: HEIGHT,
                condition_rows,
            },
            trajectory: Trajectory::default(),
            encoder_inputs: None,
        }
    }

    /// A video reference with a soundtrack: 9 condition units, vision patches and audio rows.
    fn reference() -> Vec<ConditionExtent> {
        vec![ConditionExtent {
            pixel_bytes: 9 * u64::from(WIDTH * HEIGHT) * 3,
            audio_samples: 1_000,
            vision_patches: 12,
            vision_tokens: 6,
            latent_units: vec![100; 9],
            audio_rows: 10,
        }]
    }

    /// A running request of `input`, its phases placed as `placement` states, attached to each of
    /// those groups.
    fn request(input: DiffusionRequest, placement: &[(Phase, GroupId)]) -> Request {
        let now = Instant::now();
        let mut placed = [None; Phase::COUNT];
        let mut attachments: Vec<(GroupId, OwnerId, u32, bool)> = Vec::new();
        for &(phase, group) in placement {
            placed[phase.index()] = Some(group);
            if !attachments.iter().any(|&(attached, ..)| attached == group) {
                attachments.push((group, OwnerId(10 + u64::from(group.0)), 1, false));
            }
        }
        Request {
            id: input.request_id,
            input: proto::request::Request::Diffusion(Box::new(input)),
            arrival: now,
            status: RequestStatus::Running,
            placement: placed,
            attachments,
            scheduled: [0; Phase::COUNT],
            completed: [0; Phase::COUNT],
            early: Vec::new(),
            blocks: Vec::new(),
            queued: false,
            pending: 0,
            token_ids: Vec::new(),
            num_prompt_tokens: 0,
            num_computed_tokens: 0,
            num_cached_tokens: 0,
            num_host_tokens: 0,
            num_peer_tokens: 0,
            listing: 0,
            keys: Vec::new(),
            spec_token_ids: Vec::new(),
            num_draft_tokens: 0,
            num_encoded_inputs: 0,
            encodings: Vec::new(),
            unpinned: Vec::new(),
            crossings: Vec::new(),
            num_acknowledged_tokens: 0,
            terminal: None,
            event_credit: 64,
            event_window: 64,
            draft_limit: 0,
            last_progress: now,
            dirty: false,
            path: None,
        }
    }

    /// Every video phase on `group`.
    fn on(group: GroupId) -> Vec<(Phase, GroupId)> {
        phases().into_iter().map(|phase| (phase, group)).collect()
    }

    /// The phases the request may be given a call of now, in the order it offers them.
    fn ready(request: &Request, groups: &[WorkerGroup]) -> Vec<Phase> {
        phases()
            .into_iter()
            .filter(|&phase| request.video_ready(phase, groups))
            .collect()
    }

    fn schedule(request: &mut Request, phase: Phase, calls: u32) {
        request.scheduled[phase.index()] += calls;
    }

    fn complete(request: &mut Request, phase: Phase, calls: u32) {
        request.completed[phase.index()] += calls;
    }

    /// An executor that runs nothing and records what it is handed to stamp, by the group whose
    /// workers stamp it.
    struct Stamps(Arc<Mutex<Vec<(GroupId, runtime::Request)>>>);

    impl Executor for Stamps {
        fn submit(&self, _: Batch) {}

        fn map(
            &self,
            _: GroupId,
            _: u32,
            _: runtime::RequestId,
            _: runtime::Request,
            _: Completion,
        ) {
        }

        fn stamp(&self, group: GroupId, requests: Vec<runtime::Request>) {
            let stamped = requests.into_iter().map(|request| (group, request));
            self.0.lock().unwrap().extend(stamped);
        }
    }

    /// What a [`Stamps`] executor records.
    type Stamped = Arc<Mutex<Vec<(GroupId, runtime::Request)>>>;

    /// A scheduler over `served`, in a session whose regions are the product buffer (units of 256
    /// bytes), the slot rows and the host latent pool (units of 4 096 bytes).
    fn scheduler(served: &[(GroupId, &[Phase])]) -> Scheduler {
        stamping(served).0
    }

    /// A [`scheduler`] over `served`, with what its executor was handed to stamp.
    fn stamping(served: &[(GroupId, &[Phase])]) -> (Scheduler, Stamped) {
        let region = |name: &str, tier, unit_bytes| Region {
            name: name.to_owned(),
            tier,
            space: name.to_owned(),
            unit_bytes,
            part_bytes: unit_bytes,
            reserved: 0,
            units: 1 << 20,
            limit: 1 << 20,
            segments: Vec::new(),
            memory: String::new(),
        };
        let sized = |host| Class {
            units: 0,
            host,
            state_in_slot: false,
            tensors: [0; 32],
            encoding: [0; 32],
            source: None,
        };
        let mut classes = vec![Class {
            units: 1,
            host: None,
            state_in_slot: false,
            tensors: [0; 32],
            encoding: [0; 32],
            source: None,
        }];
        classes.extend(pipeline().products.iter().map(|_| sized(None)));
        classes.push(sized(Some((LATENTS, 0))));
        let ids: Vec<GroupId> = served.iter().map(|&(id, _)| id).collect();

        let layout = runtime::Open {
            version: runtime::VERSION,
            epoch: 1,
            trial: false,
            regions: vec![
                region("products", Tier::Device, 256),
                region("slots", Tier::Device, 1),
                region("latents", Tier::Host, 4096),
            ],
            classes,
            trajectory: trajectory(),
            peers: Vec::new(),
            groups: ids.iter().map(|&id| layout(id)).collect(),
            schedulers: vec![ids.clone()],
        };
        let opened = runtime::Opened {
            regions: vec![(4, 1 << 20), (5, 1 << 20), (6, 1 << 20)],
            segments: Vec::new(),
            limits: Vec::new(),
            registers: vec![runtime::Registers {
                completed: (0..ids.len() as u32).collect(),
                capacity: vec![(4, 4), (5, 7), (6, 10)],
            }],
            session: 0,
        };
        let configs = served
            .iter()
            .map(|&(id, phases)| (config(id), capacity(phases)))
            .collect();
        let (output, _) = crossbeam_channel::unbounded();
        let stamped = Arc::new(Mutex::new(Vec::new()));
        let scheduler = Scheduler::new(
            SchedulerConfig::default(),
            proto::request::GenerationLimits::default(),
            &layout,
            &opened,
            0,
            QueuePair::create(1 << 12, 16).unwrap(),
            configs,
            crate::scheduler::tests::idle_policy(),
            scheduling::Parameters::default(),
            Box::new(Stamps(Arc::clone(&stamped))),
            output,
            Arc::new(crate::batch::Generations::default()),
            crate::scheduler::tests::tuned(),
        )
        .unwrap();
        (scheduler, stamped)
    }

    /// The calls of each phase follow the request's conditions and the widths of the groups that
    /// serve it (SD §3.3).
    #[test]
    fn a_request_makes_the_calls_of_its_conditions_and_rounds() {
        let groups = groups(&[(A, &phases()[..])]);
        let needed = |request: &Request| {
            phases()
                .into_iter()
                .map(|phase| (phase, request.video_needed(phase, &groups)))
                .collect::<Vec<_>>()
        };

        // Three media units over a decoder of two ranks decode in two rounds, each encoded and
        // muxed, before the final muxing call.
        let text = request(video(3, 40, Vec::new()), &on(A));
        assert_eq!(
            needed(&text),
            [
                (Phase::MediaRead, 0),
                (Phase::TextEncode, 1),
                (Phase::LatentPrepare, 1),
                (Phase::Denoise, STEPS),
                (Phase::VideoEncode, 2),
                (Phase::AudioEncode, 1),
                (Phase::VideoDecode, 2),
                (Phase::AudioDecode, 1),
                (Phase::Mux, 3),
            ]
        );

        // Conditions add their read, and nothing else: the text encoding and the latent
        // preparation encode them inside their calls. One unit is one round.
        let conditioned = request(video(1, 40, reference()), &on(A));
        let counts = needed(&conditioned);
        assert_eq!(counts[0], (Phase::MediaRead, 1));
        assert_eq!(counts[1], (Phase::TextEncode, 1));
        assert_eq!(counts[8], (Phase::Mux, 2));
        for phase in [Phase::VisionEncode, Phase::LatentEncode] {
            assert_eq!(conditioned.video_needed(phase, &groups), 0, "{phase:?}");
        }
    }

    /// On one group, each phase is offered once the phase before it is scheduled; denoising
    /// keeps two steps in flight, and a decode round after the first waits until the round
    /// before it completed (SD §3.3, gap G11).
    #[test]
    fn a_text_request_offers_each_phase_once_its_predecessor_is_handed_over() {
        let groups = groups(&[(A, &phases()[..])]);
        let mut request = request(video(3, 40, Vec::new()), &on(A));
        assert_eq!(ready(&request, &groups), [Phase::TextEncode]);

        schedule(&mut request, Phase::TextEncode, 1);
        assert_eq!(ready(&request, &groups), [Phase::LatentPrepare]);
        schedule(&mut request, Phase::LatentPrepare, 1);
        assert_eq!(ready(&request, &groups), [Phase::Denoise]);

        schedule(&mut request, Phase::Denoise, 2);
        assert!(ready(&request, &groups).is_empty());
        complete(&mut request, Phase::Denoise, 1);
        assert_eq!(ready(&request, &groups), [Phase::Denoise]);
        assert!(!request.runs_to_end(&groups));

        schedule(&mut request, Phase::Denoise, 2);
        assert!(request.runs_to_end(&groups));
        assert_eq!(
            ready(&request, &groups),
            [Phase::VideoDecode, Phase::AudioDecode]
        );

        schedule(&mut request, Phase::VideoDecode, 1);
        assert_eq!(ready(&request, &groups), [Phase::AudioDecode]);
        complete(&mut request, Phase::VideoDecode, 1);
        assert_eq!(
            ready(&request, &groups),
            [Phase::VideoEncode, Phase::VideoDecode, Phase::AudioDecode]
        );
    }

    /// A conditioned request's calls run in order with no encoder call of their own: its media
    /// read first, then its text encoding once the read completed, its latent preparation once
    /// the text encoding is handed over, and its denoising after that (SD §3.3).
    #[test]
    fn a_conditioned_request_encodes_its_prompt_once_its_media_read_completed() {
        let groups = groups(&[(A, &phases()[..])]);
        let mut request = request(video(1, 40, reference()), &on(A));
        assert_eq!(ready(&request, &groups), [Phase::MediaRead]);

        schedule(&mut request, Phase::MediaRead, 1);
        assert!(ready(&request, &groups).is_empty());
        complete(&mut request, Phase::MediaRead, 1);
        assert_eq!(ready(&request, &groups), [Phase::TextEncode]);

        schedule(&mut request, Phase::TextEncode, 1);
        assert_eq!(ready(&request, &groups), [Phase::LatentPrepare]);
        schedule(&mut request, Phase::LatentPrepare, 1);
        assert_eq!(ready(&request, &groups), [Phase::Denoise]);
        for phase in [Phase::VisionEncode, Phase::LatentEncode] {
            assert!(!request.video_ready(phase, &groups), "{phase:?}");
        }
    }

    /// A request whose admission's `Pin` finds its conditioning under its key holds it at the
    /// conditioning's position, makes no text encoding, and its latent preparation reads the
    /// pinned block; one whose `Pin` finds nothing allocates the block, encodes its prompt into it
    /// and commits it under its key (SD §3.3, §4.3).
    #[test]
    fn a_conditioning_the_admission_finds_is_read_without_a_text_encoding() {
        let (mut scheduler, stamped) = stamping(&[(A, &phases()[..])]);
        let class = scheduler.conditioning_class().unwrap();
        let mut input = video(1, 40, reference());
        input.encoder_inputs = Some(Box::new(EncoderInputs {
            text: [3; 32],
            vision: None,
            video_units: Vec::new(),
            audio: None,
        }));

        // The conditioning is keyed by what the text encoder reads.
        let mut admitted = request(input, &on(A));
        admitted.encodings = scheduler.encodings_of(&admitted);
        let key = Key(encoder_output(Product::Conditioning.name(), "", &[3; 32]));
        assert_eq!(admitted.encodings, [(key, None)]);
        let at = admitted.conditioning_position(&scheduler.groups).unwrap();
        let id = admitted.id;
        scheduler.owners.insert(OwnerId(10), id);
        scheduler.requests.insert(id, admitted.clone());

        // Not cached: allocated at its position, of its kind's size, like every declared result.
        let asks = scheduler.admission_requests(id);
        let allocate = runtime::Request::Allocate {
            owner: OwnerId(10),
            class,
            at,
            count: 1,
            units: u32::from(class),
        };
        assert!(asks.contains(&allocate), "{asks:?}");

        // Cached on the device of the group that encodes it: pinned at the same position.
        let listed = (key, class, 40, DEVICE, 1, 0);
        let listing = Completion::Listing { keys: vec![listed] };
        scheduler.learn(&runtime::Request::List, &listing);
        let asks = scheduler.admission_requests(id);
        let pin = runtime::Request::Pin {
            owner: OwnerId(10),
            class,
            at,
            keys: vec![key],
        };
        assert!(asks.contains(&pin) && !asks.contains(&allocate), "{asks:?}");

        // A `Pin` that finds nothing allocates the block at once.
        let found = |blocks: u32| Completion::Pinned {
            blocks: (0..blocks).map(|unit| vec![(4, unit, 1)]).collect(),
            host: 0,
        };
        assert_eq!(scheduler.filled(id, &pin, &found(0)), [allocate]);

        // A `Pin` that finds it: no text encoding, and the preparation reads the pinned block.
        assert!(scheduler.filled(id, &pin, &found(1)).is_empty());
        let groups = &scheduler.groups;
        let pinned = &scheduler.requests[&id];
        assert_eq!(pinned.video_needed(Phase::TextEncode, groups), 0);
        let mut pinned = pinned.clone();
        schedule(&mut pinned, Phase::MediaRead, 1);
        complete(&mut pinned, Phase::MediaRead, 1);
        assert_eq!(ready(&pinned, groups), [Phase::LatentPrepare]);
        let (results, reads) = pinned.results(Phase::LatentPrepare, 0, groups);
        assert_eq!(reads, 3);
        assert!(
            results[..3].contains(&at),
            "{results:?} reads the pinned block {at}"
        );

        // Encoded instead: the text encoding's output is committed under the key, the prompt's
        // rows as its tokens, and the request still made its one text encoding.
        scheduler.requests.insert(id, admitted);
        let mut encoded = Batch {
            id: BatchId(1),
            group: A,
            phase: Phase::TextEncode,
            kernels: 0,
            policy: 0,
            revision: 0,
            copy_budget: proto::config::DEFAULT_COPY_BUDGET,
            shape: Shape::decode(1, Variant::Encode),
            requests: vec![id],
            columns: Columns::default(),
            media: Vec::new(),
            new_slots: Vec::new(),
        };
        assert!(scheduler.push_media_row(&mut encoded, id, Phase::TextEncode));
        let row = scheduler.requests.get_mut(&id).unwrap();
        schedule(row, Phase::TextEncode, 1);
        complete(row, Phase::TextEncode, 1);
        scheduler.commit_encoding(id, &encoded, 0);
        let commit = runtime::Request::Commit {
            owner: OwnerId(10),
            class,
            at,
            key,
            tokens: 40,
            after: Vec::new(),
        };
        assert_eq!(*stamped.lock().unwrap(), [(A, commit)]);
        let groups = &scheduler.groups;
        assert_eq!(
            scheduler.requests[&id].video_needed(Phase::TextEncode, groups),
            1
        );
    }

    /// A phase placed on another group than the one it reads from waits until that one's calls
    /// completed, since no worker orders the two.
    #[test]
    fn a_phase_on_another_group_waits_until_its_predecessor_completed() {
        let groups = groups(&[(A, &phases()[..]), (B, &[Phase::TextEncode])]);
        let mut placement = on(A);
        placement[1] = (Phase::TextEncode, B);
        let mut request = request(video(1, 40, Vec::new()), &placement);

        schedule(&mut request, Phase::TextEncode, 1);
        assert!(ready(&request, &groups).is_empty());
        complete(&mut request, Phase::TextEncode, 1);
        assert_eq!(ready(&request, &groups), [Phase::LatentPrepare]);
    }

    /// The muxer takes each encode round once it completed, one call at a time, and its final
    /// call waits for every round and the audio (SD §3.3).
    #[test]
    fn the_muxer_takes_each_completed_round_and_ends_after_the_audio() {
        let groups = groups(&[(A, &phases()[..])]);
        let mut request = request(video(3, 40, Vec::new()), &on(A));
        for (phase, calls) in [
            (Phase::TextEncode, 1),
            (Phase::LatentPrepare, 1),
            (Phase::Denoise, STEPS),
            (Phase::VideoDecode, 2),
            (Phase::AudioDecode, 1),
        ] {
            schedule(&mut request, phase, calls);
            complete(&mut request, phase, calls);
        }
        schedule(&mut request, Phase::VideoEncode, 2);
        let mux = |request: &Request| request.video_ready(Phase::Mux, &groups);
        assert!(!mux(&request));

        complete(&mut request, Phase::VideoEncode, 1);
        assert!(mux(&request));
        schedule(&mut request, Phase::Mux, 1);
        assert!(!mux(&request), "one muxing call at a time");
        complete(&mut request, Phase::Mux, 1);
        assert!(!mux(&request), "the second round is still encoding");

        complete(&mut request, Phase::VideoEncode, 1);
        schedule(&mut request, Phase::Mux, 1);
        complete(&mut request, Phase::Mux, 1);
        assert!(!mux(&request), "the final call waits for the audio");

        schedule(&mut request, Phase::AudioEncode, 1);
        complete(&mut request, Phase::AudioEncode, 1);
        assert!(mux(&request));
        schedule(&mut request, Phase::Mux, 1);
        assert!(!mux(&request));
    }

    /// A decode round after the first reads the splice tail of the unit before its own, which the
    /// round before it wrote, and the rounds read the latents the last denoising step wrote
    /// (commit `e69a91146`; SD §2.3 `MediaRow::results`).
    #[test]
    fn a_decode_round_reads_the_tail_the_round_before_it_wrote() {
        let groups = groups(&[(A, &phases()[..])]);
        let request = request(video(3, 40, Vec::new()), &on(A));
        let results = |phase, call| request.results(phase, call, &groups);

        // A round takes a unit for each of the decoder's two ranks; the round a row is is its
        // first unit over that width, whatever order rounds return in.
        assert_eq!(request.round_units(Phase::VideoDecode, 0, &groups), 0..2);
        assert_eq!(request.round_units(Phase::VideoDecode, 1, &groups), 2..3);
        assert_eq!(request.round_of(2, &groups), 1);
        let (first, first_reads) = results(Phase::VideoDecode, 0);
        let (second, second_reads) = results(Phase::VideoDecode, 1);
        assert_eq!(first_reads, 1, "the first round reads the latents alone");
        assert_eq!(
            second_reads, 2,
            "a later round reads the latents and a tail"
        );

        // Each round writes the video units and one tail per unit it decodes; the tail the second
        // round reads is the first round's last.
        assert_eq!(first.len(), 1 + 1 + 2);
        assert_eq!(second.len(), 2 + 1 + 1);
        assert_eq!(second[1], first[3]);
        assert_eq!(second[2], first[1], "both rounds write the video units");
        assert!(!first.contains(&second[3]), "each unit's tail is its own");

        // Only the last step writes the latents, which both rounds read.
        assert_eq!(results(Phase::Denoise, 0), (Vec::new(), 0));
        let (latents, reads) = results(Phase::Denoise, STEPS - 1);
        assert_eq!(reads, 0);
        assert!(latents.contains(&first[0]) && first[0] == second[0]);

        // An encode round reads the video units and writes an encoded unit per unit of its
        // round, a muxing call reads the encoded units of its round, and the final muxing call
        // nothing.
        let (encode, reads) = results(Phase::VideoEncode, 0);
        assert_eq!((encode.len(), reads, encode[0]), (3, 1, first[1]));
        let (last, _) = results(Phase::VideoEncode, 1);
        assert_eq!(results(Phase::Mux, 0), (encode[1..].to_vec(), 2));
        assert_eq!(results(Phase::Mux, 1), (last[1..].to_vec(), 1));
        assert_eq!(results(Phase::Mux, 2), (Vec::new(), 0));
    }

    /// Every result a call writes is a declared result of the call's phase, and each is written
    /// once, but the video units, of which each round writes its slice; each block is sized in
    /// its product class, rounded up to units of the product region (SD §4.4 "Products").
    #[test]
    fn declared_results_are_written_by_their_phase_and_sized_in_their_class() {
        let scheduler = scheduler(&[(A, &phases()[..])]);
        let groups = &scheduler.groups;
        for request in [
            request(video(3, 40, Vec::new()), &on(A)),
            request(video(2, 40, reference()), &on(A)),
        ] {
            let declared = scheduler.declared_results(&request).unwrap();
            let mut written = vec![0; declared.len()];
            for phase in phases() {
                for call in 0..request.video_needed(phase, groups) {
                    let (results, reads) = request.results(phase, call, groups);
                    for &position in &results[reads as usize..] {
                        assert_eq!(declared[position as usize].0, phase);
                        written[position as usize] += 1;
                    }
                }
            }
            // The video units are written by every decode round.
            let rounds = request.video_needed(Phase::VideoDecode, groups);
            let sliced = request.results(Phase::VideoDecode, 0, groups).0[1];
            for (position, count) in written.into_iter().enumerate() {
                let expected = if position == sliced as usize {
                    rounds
                } else {
                    1
                };
                assert_eq!(count, expected, "result {position}");
            }

            // The kind of a block is its class's place before the class of trajectories, and it
            // takes one byte more than that place's units.
            for &(_, class, units) in &declared {
                assert!((1..trajectory()).contains(&class));
                assert_eq!(units, u32::from(class));
            }
        }

        // A text request of three units declares the conditioning, both latents, the video
        // units, one tail and one encoded unit per unit, and the audio; a conditioned request of
        // two the condition pixels and samples and the vision pixels besides, and no encoded
        // condition of its own.
        let text = request(video(3, 40, Vec::new()), &on(A));
        assert_eq!(scheduler.declared_results(&text).unwrap().len(), 11);
        let conditioned = request(video(2, 40, reference()), &on(A));
        assert_eq!(scheduler.declared_results(&conditioned).unwrap().len(), 12);
    }

    /// A trajectory is sized from the smallest captured shape that holds its request, in units of
    /// the host latent region (SD §3.8, §4.4).
    #[test]
    fn a_trajectory_is_sized_in_units_of_the_host_latent_region() {
        let scheduler = scheduler(&[(A, &phases()[..])]);
        let units = |input| scheduler.trajectory_units(&request(input, &on(A)));

        // 10 000 bytes of the shape of 64 prompt rows, 30 000 of the one of 128, and 20 000 of the
        // one with condition rows, in units of 4 096 bytes.
        assert_eq!(units(video(1, 40, Vec::new())), Some(3));
        assert_eq!(units(video(1, 100, Vec::new())), Some(8));
        assert_eq!(units(video(1, 40, reference())), Some(5));
        assert_eq!(units(video(1, 200, Vec::new())), None);
    }

    /// A request the deployment cannot serve is refused with the first rule it breaks (SD §7.2).
    #[test]
    fn a_video_request_the_deployment_cannot_serve_is_refused() {
        let all = scheduler(&[(A, &phases()[..])]);
        assert_eq!(all.validate_video(&video(2, 40, Vec::new())), Ok(()));
        assert_eq!(all.validate_video(&video(2, 40, reference())), Ok(()));

        let empty = video(2, 0, Vec::new());
        assert_eq!(
            all.validate_video(&empty),
            Err("media prompt tokens must not be empty".to_owned())
        );

        let without = |missing: &[Phase]| {
            let served: Vec<Phase> = phases()
                .into_iter()
                .filter(|phase| !missing.contains(phase))
                .collect();
            scheduler(&[(A, &served)])
        };
        assert_eq!(
            without(&[Phase::Mux]).validate_video(&video(2, 40, Vec::new())),
            Err("worker does not provide the video media stages".to_owned())
        );
        assert_eq!(
            without(&[Phase::Denoise]).validate_video(&video(2, 40, Vec::new())),
            Err("worker reports no video denoiser".to_owned())
        );
        let unconditioned = without(&[Phase::MediaRead]);
        assert_eq!(
            unconditioned.validate_video(&video(2, 40, Vec::new())),
            Ok(())
        );
        assert_eq!(
            unconditioned.validate_video(&video(2, 40, reference())),
            Err("the deployment's denoiser takes no conditions".to_owned())
        );

        let mut oversized = reference();
        oversized[0].pixel_bytes = 3 << 32;
        assert_eq!(
            all.validate_video(&video(2, 40, oversized)),
            Err("the request's condition pixels exceed the protocol's extents".to_owned())
        );

        let mut wide = video(2, 40, Vec::new());
        wide.sampling.width = 2 * WIDTH;
        let refused = all.validate_video(&wide).unwrap_err();
        assert!(
            refused.starts_with("loaded media stages cannot hold the request")
                && refused.contains("128x32"),
            "{refused}"
        );
    }

    /// A request a drain or a failed crossing preempted after a completed step stores its
    /// trajectory under the key of that step, and a resumed request restores the steps it names
    /// under the seed that stored them (SD §3.8, §4.3, §7.3).
    #[test]
    fn a_drained_request_stores_its_step_and_resumes_from_it() {
        let groups = groups(&[(A, &phases()[..])]);
        let mut input = video(1, 40, Vec::new());
        input.trajectory.id = std::num::NonZeroU64::new(77);
        let drained = |completed: u32, cause: MigrationCause, placement: &[(Phase, GroupId)]| {
            let mut request = request(input.clone(), placement);
            request.completed[Phase::Denoise.index()] = completed;
            request.terminal = Some(Event::Migratable {
                cause,
                denoised_steps: 0,
                stored_layout: Vec::new(),
                lend: None,
            });
            request
        };

        let stored = drained(2, MigrationCause::Drain, &on(A));
        assert!(stored.stores_trajectory());
        assert_eq!(
            stored.trajectory_key(5),
            Some(Key(latent_content(5, 77, 2)))
        );
        assert!(drained(2, MigrationCause::Transfer, &on(A)).stores_trajectory());
        assert!(!drained(2, MigrationCause::Loss, &on(A)).stores_trajectory());
        assert!(!drained(0, MigrationCause::Drain, &on(A)).stores_trajectory());
        assert!(!drained(2, MigrationCause::Drain, &[]).stores_trajectory());
        let mut anonymous = drained(2, MigrationCause::Drain, &on(A));
        if let proto::request::Request::Diffusion(video) = &mut anonymous.input {
            video.trajectory.id = None;
        }
        assert!(!anonymous.stores_trajectory());

        let resumed = |steps: u32, state_seed: Option<u64>| {
            let mut input = input.clone();
            input.trajectory.denoised_steps = steps;
            input.trajectory.state_seed = state_seed;
            request(input, &on(A)).restore_key(5, &groups)
        };
        assert_eq!(resumed(2, None), Some(Key(latent_content(5, 77, 2))));
        assert_eq!(resumed(2, Some(9)), Some(Key(latent_content(9, 77, 2))));
        assert_eq!(resumed(0, None), None);
        assert_eq!(resumed(STEPS, None), None);
    }

    /// A decode is offered once the last denoising step is scheduled on its group only when each
    /// rank that decodes denoises too, reading the latents its own last step wrote; where a rank
    /// decodes that does not denoise, which receives its latents from another rank's step, once
    /// the last step's output is applied (SD §3.3).
    #[test]
    fn a_decode_waits_for_the_last_step_s_output_where_a_decoder_rank_does_not_denoise() {
        let decoding = |denoisers: Vec<usize>| {
            let mut placed = config(A);
            placed
                .blocks
                .insert("transformer".to_owned(), BlockConfig::parallel(denoisers));
            placed
                .blocks
                .insert("audio_decoder".to_owned(), BlockConfig::parallel(vec![1]));
            let groups = vec![WorkerGroup::new(placed, capacity(&phases()), layout(A)).unwrap()];
            let mut request = request(video(2, 40, Vec::new()), &on(A));
            for phase in [Phase::TextEncode, Phase::LatentPrepare] {
                schedule(&mut request, phase, 1);
                complete(&mut request, phase, 1);
            }
            schedule(&mut request, Phase::Denoise, STEPS);
            complete(&mut request, Phase::Denoise, STEPS - 1);
            let early = ready(&request, &groups);
            complete(&mut request, Phase::Denoise, 1);
            (early, ready(&request, &groups))
        };
        let decodes = [Phase::VideoDecode, Phase::AudioDecode];

        // Both ranks denoise: both decodes follow the scheduled last step.
        assert_eq!(decoding(vec![0, 1]), (decodes.to_vec(), decodes.to_vec()));

        // Rank 1 decodes the video's second unit and the track, and does not denoise: both wait
        // for the last step's output.
        assert_eq!(decoding(vec![0]), (Vec::new(), decodes.to_vec()));
    }

    /// The two processes of the crossing fixture, by the memory their product buffers lie in.
    const FIRST: &str = "process a";
    const SECOND: &str = "process b";

    /// The session regions of the crossing fixture, by their index, and the numbers completions
    /// name them by: each rank's product buffer, then the two groups' slot rows and the host
    /// latent pool.
    const PRODUCTS_FIRST: u8 = 0;
    const PRODUCTS_SECOND: u8 = 1;
    const NUMBERS: [u8; 5] = [4, 5, 6, 7, 8];

    /// A scheduler over a model group [`A`] of two ranks, rank 0 in the first process and rank 1
    /// in the second, which encodes prompts (rank 0 holding the conditioning), denoises and
    /// decodes video on both ranks, decodes the track on rank 0 and encodes the video; and a
    /// muxer group [`B`] of one rank in the first process. The layout is the engine's for it
    /// (`crate::process::ranks::layout`): the model group's blocks lie in both ranks' product
    /// buffers, its conditioning in rank 0's alone, which it receives in rank 1's; the muxer reads
    /// the encoded units and the track in the first process. With what its executor was handed to
    /// stamp, and what the scheduler reports.
    fn two_processes() -> (Scheduler, Stamped, Receiver<Output>) {
        let region = |name: &str, tier, unit_bytes, memory: &str| Region {
            name: name.to_owned(),
            tier,
            space: name.to_owned(),
            unit_bytes,
            part_bytes: unit_bytes,
            reserved: 0,
            units: 1 << 20,
            limit: 1 << 20,
            segments: Vec::new(),
            memory: memory.to_owned(),
        };
        let class = |units, host| Class {
            units,
            host,
            state_in_slot: false,
            tensors: [0; 32],
            encoding: [0; 32],
            source: None,
        };
        let mut classes = vec![class(1, None)];
        classes.extend(pipeline().products.iter().map(|_| class(0, None)));
        classes.push(class(0, Some((4, 0))));

        // Per class, the regions of each group and those it receives in.
        let kind = |product: Product| pipeline().kind(product).unwrap() + 1;
        let mut model = vec![Vec::new(); classes.len()];
        let mut receives = vec![Vec::new(); classes.len()];
        let mut muxer = vec![Vec::new(); classes.len()];
        model[kind(Product::Conditioning)] = vec![PRODUCTS_FIRST];
        receives[kind(Product::Conditioning)] = vec![PRODUCTS_SECOND];
        for name in [
            Product::VideoLatents,
            Product::AudioLatents,
            Product::VideoUnits,
            Product::VideoTail,
            Product::EncodedUnits,
            Product::AudioSamples,
        ] {
            model[kind(name)] = vec![PRODUCTS_FIRST, PRODUCTS_SECOND];
        }
        for name in [Product::EncodedUnits, Product::AudioSamples] {
            muxer[kind(name)] = vec![PRODUCTS_FIRST];
        }
        let group = |id, slots, regions, receives| runtime::Group {
            id,
            memory: FIRST.to_owned(),
            slots,
            regions,
            receives,
            parts: Vec::new(),
        };

        let layout = runtime::Open {
            version: runtime::VERSION,
            epoch: 1,
            trial: false,
            regions: vec![
                region("products a", Tier::Device, 256, FIRST),
                region("products b", Tier::Device, 256, SECOND),
                region("slots model", Tier::Device, 1, ""),
                region("slots muxer", Tier::Device, 1, ""),
                region("latents", Tier::Host, 4096, ""),
            ],
            classes,
            trajectory: trajectory(),
            peers: Vec::new(),
            groups: vec![
                group(A, 2, model, receives),
                group(B, 3, muxer, vec![Vec::new(); trajectory() as usize + 1]),
            ],
            schedulers: vec![vec![A, B]],
        };
        let opened = runtime::Opened {
            regions: NUMBERS.iter().map(|&number| (number, 1 << 20)).collect(),
            segments: Vec::new(),
            limits: Vec::new(),
            registers: vec![runtime::Registers {
                completed: vec![0, 1],
                capacity: NUMBERS
                    .iter()
                    .zip((2..).step_by(3))
                    .map(|(&number, word)| (number, word))
                    .collect(),
            }],
            session: 0,
        };

        let rank = |host: &str| WorkerRank {
            host: host.to_owned(),
            device: "cpu".parse().unwrap(),
        };
        let mut model = config(A);
        model.ranks = vec![rank("a"), rank("b")];
        model.blocks.insert(
            "audio_decoder".to_owned(),
            BlockConfig::parallel(vec![0, 1]),
        );
        model.blocks.insert(
            HostBlock::VideoCodec.name().to_owned(),
            BlockConfig {
                distribution: Some(BlockDistribution::TemporalUnits),
                ..BlockConfig::parallel(vec![0, 1])
            },
        );
        let mut muxer = config(B);
        muxer.ranks = vec![rank("a")];
        muxer.blocks = BTreeMap::new();
        muxer.blocks.insert(
            HostBlock::Muxer.name().to_owned(),
            BlockConfig::parallel(vec![0]),
        );
        let modelled = [
            Phase::TextEncode,
            Phase::LatentPrepare,
            Phase::Denoise,
            Phase::VideoDecode,
            Phase::AudioDecode,
            Phase::VideoEncode,
        ];
        let configs = vec![
            (model, capacity(&modelled)),
            (muxer, capacity(&[Phase::AudioEncode, Phase::Mux])),
        ];

        let (output, outputs) = crossbeam_channel::unbounded();
        let stamped = Arc::new(Mutex::new(Vec::new()));
        let scheduler = Scheduler::new(
            SchedulerConfig::default(),
            proto::request::GenerationLimits::default(),
            &layout,
            &opened,
            0,
            QueuePair::create(1 << 12, 16).unwrap(),
            configs,
            crate::scheduler::tests::idle_policy(),
            scheduling::Parameters::default(),
            Box::new(Stamps(Arc::clone(&stamped))),
            output,
            Arc::new(crate::batch::Generations::default()),
            crate::scheduler::tests::tuned(),
        )
        .unwrap();
        (scheduler, stamped, outputs)
    }

    /// A text request of two units running on [`two_processes`], whose conditioning is keyed by
    /// what its text encoder reads: its media phases on the model group but the audio encode and
    /// the muxing calls, on the muxer, with its crossings. Its owners are 10 on the model group and
    /// 11 on the muxer.
    fn crossing(scheduler: &mut Scheduler) -> RequestId {
        let mut placement = on(A);
        for entry in &mut placement {
            if matches!(entry.0, Phase::AudioEncode | Phase::Mux) {
                entry.1 = B;
            }
        }
        let mut input = video(2, 40, Vec::new());
        input.encoder_inputs = Some(Box::new(EncoderInputs {
            text: [3; 32],
            vision: None,
            video_units: Vec::new(),
            audio: None,
        }));
        let mut row = request(input, &placement);
        row.encodings = scheduler.encodings_of(&row);
        row.crossings = scheduler.crossings_of(&row);
        let id = row.id;
        scheduler.owners.insert(OwnerId(10), id);
        scheduler.owners.insert(OwnerId(11), id);
        scheduler.requests.insert(id, row);
        scheduler.running.push(id);
        id
    }

    /// The calls of `phase` request `id` was scheduled, and applied.
    fn ran(scheduler: &mut Scheduler, id: RequestId, phase: Phase, calls: u32) {
        let request = scheduler.requests.get_mut(&id).unwrap();
        schedule(request, phase, calls);
        complete(request, phase, calls);
    }

    /// A product a group reads in another process than its writer's crosses once the writing
    /// call's output is applied, by a `Replicate` from the run its writer wrote, stamped by the
    /// executors of the groups that may hold its entry; the reading call waits for the answer
    /// (SD §3.3, §4.1). The conditioning, written by the first text encoder rank, crosses into
    /// the model group's memory in the second process, where its latent preparation reads it
    /// too, after its `Commit`, which the buffer pool takes while the block has its one run; of
    /// the encoded units, one block each, the one the second process's rank decoded crosses
    /// into the muxer's process, and the muxer is handed only the allocation of the one decoded
    /// where it reads (BP §9, obligation 1).
    #[test]
    fn a_product_read_in_another_process_crosses_once_written_and_its_reader_waits_for_it() {
        let (mut scheduler, stamped, _) = two_processes();
        let id = crossing(&mut scheduler);
        let row = &scheduler.requests[&id];
        let conditioning = row.conditioning_position(&scheduler.groups).unwrap();
        let units: Vec<u32> = (0..2)
            .map(|unit| {
                row.result(Product::EncodedUnits, unit, &scheduler.groups)
                    .unwrap()
            })
            .collect();
        let [first, second] = [NUMBERS[0], NUMBERS[1]];
        assert_eq!(
            row.crossings,
            [(conditioning, A, first, None), (units[1], B, second, None)]
        );
        let ready = |scheduler: &Scheduler, phase: Phase| {
            scheduler.requests[&id].video_ready(phase, &scheduler.groups)
        };
        let asked = |stamped: &Stamped| stamped.lock().unwrap().drain(..).collect::<Vec<_>>();

        // Nothing crosses before its writer's output, and the latent preparation that follows the
        // text encoding on its group waits for the conditioning in the second process.
        let mut encoded = Batch {
            id: BatchId(1),
            group: A,
            phase: Phase::TextEncode,
            kernels: 0,
            policy: 0,
            revision: 0,
            copy_budget: proto::config::DEFAULT_COPY_BUDGET,
            shape: Shape::decode(1, Variant::Encode),
            requests: Vec::new(),
            columns: Columns::default(),
            media: Vec::new(),
            new_slots: Vec::new(),
        };
        assert!(scheduler.push_media_row(&mut encoded, id, Phase::TextEncode));
        let request = scheduler.requests.get_mut(&id).unwrap();
        schedule(request, Phase::TextEncode, 1);
        scheduler.cross();
        assert!(asked(&stamped).is_empty());
        assert!(!ready(&scheduler, Phase::LatentPrepare));

        // Its output applied, the conditioning is committed under its key and crosses from rank
        // 0's run into the model group's memories, once.
        let request = scheduler.requests.get_mut(&id).unwrap();
        complete(request, Phase::TextEncode, 1);
        scheduler.commit_encoding(id, &encoded, 0);
        scheduler.cross();
        scheduler.cross();
        let class = scheduler.conditioning_class().unwrap();
        let commit = runtime::Request::Commit {
            owner: OwnerId(10),
            class,
            at: conditioning,
            key: Key(encoder_output(Product::Conditioning.name(), "", &[3; 32])),
            tokens: 40,
            after: Vec::new(),
        };
        let replicate = runtime::Request::Replicate {
            owner: OwnerId(10),
            class,
            at: conditioning,
            from: first,
            group: A,
            after: Vec::new(),
        };
        assert_eq!(asked(&stamped), [(A, commit), (A, replicate.clone())]);
        assert!(!ready(&scheduler, Phase::LatentPrepare));
        let replicated = Completion::Replicated {
            runs: vec![(first, 0, 1), (second, 0, 1)],
            evicted: Vec::new(),
        };
        scheduler.crossing_answered(id, &replicate, &replicated);
        assert!(ready(&scheduler, Phase::LatentPrepare));

        // The round's encode writes unit 0 in the first process and unit 1 in the second: the
        // second crosses into the muxer's, stamped by the model group and the muxer, which reads
        // it; its muxing call waits for it.
        for (phase, calls) in [
            (Phase::LatentPrepare, 1),
            (Phase::Denoise, STEPS),
            (Phase::VideoDecode, 1),
            (Phase::VideoEncode, 1),
        ] {
            ran(&mut scheduler, id, phase, calls);
        }
        scheduler.cross();
        let class = pipeline()
            .kind(Product::EncodedUnits)
            .and_then(|kind| scheduler.product_class(kind))
            .unwrap();
        let replicate = runtime::Request::Replicate {
            owner: OwnerId(10),
            class,
            at: units[1],
            from: second,
            group: B,
            after: Vec::new(),
        };
        assert_eq!(
            asked(&stamped),
            [(A, replicate.clone()), (B, replicate.clone())]
        );
        assert!(!ready(&scheduler, Phase::Mux));
        let replicated = Completion::Replicated {
            runs: vec![(first, 0, 1)],
            evicted: Vec::new(),
        };
        scheduler.crossing_answered(id, &replicate, &replicated);
        assert!(ready(&scheduler, Phase::Mux));

        // The muxer is handed the allocation of the unit decoded where it reads it, and the other
        // one only by its crossing.
        let handed = |at| scheduler.reads_where_written(OwnerId(10), (class, at), B);
        assert_eq!((handed(units[0]), handed(units[1])), (true, false));
        assert!(scheduler.reads_where_written(OwnerId(10), (class, units[1]), A));
    }

    /// A crossing that fails ends its request as migratable with the cause `Transfer`, which the
    /// daemon resubmits and counts against the request's migrations, with no retry inside the
    /// engine (SD §3.6); one the receiving region had no room for is asked again once the request
    /// is no longer short there.
    #[test]
    fn a_failed_crossing_ends_its_request_as_migratable_and_a_full_one_is_asked_again() {
        let (mut scheduler, stamped, outputs) = two_processes();
        let id = crossing(&mut scheduler);
        ran(&mut scheduler, id, Phase::TextEncode, 1);
        scheduler.cross();
        let replicate = stamped.lock().unwrap().remove(0).1;

        let region = NUMBERS[usize::from(PRODUCTS_SECOND)];
        scheduler.crossing_answered(id, &replicate, &Completion::Full { region });
        scheduler.short.remove(&id);
        scheduler.cross();
        let asked: Vec<runtime::Request> = stamped
            .lock()
            .unwrap()
            .drain(..)
            .map(|(_, request)| request)
            .collect();
        assert_eq!(asked, std::slice::from_ref(&replicate));

        scheduler.crossing_answered(id, &replicate, &Completion::Failed);
        assert!(!scheduler.running.contains(&id));
        let ended = outputs
            .try_iter()
            .flat_map(|output| match output {
                Output::Events(events) => events,
                _ => Vec::new(),
            })
            .any(|(of, event)| {
                of == id
                    && matches!(
                        event,
                        Event::Migratable {
                            cause: MigrationCause::Transfer,
                            ..
                        }
                    )
            });
        assert!(ended);
    }

    /// Progress names what the request is doing and the steps its trajectory completed
    /// (SD §7.3 "Progress and artifact").
    #[test]
    fn progress_names_the_stage_of_the_request() {
        let groups = groups(&[(A, &phases()[..])]);
        let mut request = request(video(1, 40, Vec::new()), &on(A));
        request.completed[Phase::Denoise.index()] = 1;
        let progress = |request: &Request, phase| match request.progress(phase, &groups) {
            Some(Event::MediaProgress {
                phase,
                completed_steps,
            }) => Some((phase, completed_steps)),
            _ => None,
        };

        assert_eq!(
            progress(&request, Phase::TextEncode),
            Some(("preparing".to_owned(), 1))
        );
        assert_eq!(
            progress(&request, Phase::Denoise),
            Some(("denoising".to_owned(), 1))
        );
        request.completed[Phase::Denoise.index()] = STEPS;
        assert_eq!(
            progress(&request, Phase::Denoise),
            Some(("decoding".to_owned(), STEPS))
        );
        assert_eq!(
            progress(&request, Phase::AudioEncode),
            Some(("decoding".to_owned(), STEPS))
        );
        assert_eq!(
            progress(&request, Phase::Mux),
            Some(("finalizing".to_owned(), STEPS))
        );
        assert_eq!(progress(&request, Phase::Prefill), None);
    }
}
