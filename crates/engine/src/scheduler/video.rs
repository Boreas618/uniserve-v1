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
//! for the owner on the group whose phase writes it, in the class of its product kind
//! ([`PRODUCTS`]). A request's declared results list each kind's results together, in kind order,
//! and a result's position in the request's product table is its index in that list; a media row
//! names the results its call reads and writes by these positions ([`Request::results`]).
//!
//! The conditioning is the request's encoder output (SD §3.3, §4.3): keyed by what its text
//! encoder reads (its prompt and the conditions its vision blocks present) and the encoder's
//! identity, pinned at admission where cached instead of allocated, and committed once encoded.
//! A request whose admission found it makes no text encoding, and its latent preparation reads
//! the pinned block at the same position.

use std::iter;
use std::ops::Range;

use ipc::batch::{Extent, Phase, Shape};
use ipc::product;
use proto::config::VIDEO_DECODER_BLOCK;
use proto::request::{ConditionExtent, DiffusionRequest, Event, MigrationCause};
use storage::Key;
use storage::key::latent_content;

use super::{Request, Scheduler, WorkerGroup};

/// Denoising steps one request keeps in flight: the step its denoiser runs and the next, queued
/// behind it, so that the denoiser never waits for the host between steps and a preemption meets
/// the request at a step boundary within that many steps.
const DENOISING_LOOKAHEAD: u32 = 2;

/// The media phases of a video request, in the order a request offers the ones that are ready:
/// the first is its next phase, which the policy reads. After denoising, completed decodes are
/// encoded before further decodes are asked for (`S/execution/media.rs:380-410`).
pub(super) const VIDEO_PHASES: [Phase; 9] = [
    Phase::MediaRead,
    Phase::TextEncode,
    Phase::LatentPrepare,
    Phase::Denoise,
    Phase::VideoEncode,
    Phase::AudioEncode,
    Phase::VideoDecode,
    Phase::AudioDecode,
    Phase::Mux,
];

/// The product kinds, in the order of the product classes, each with the phase whose calls write
/// it: the one list of `ipc::product::KINDS`, which the ranks read too. The class of kind `k` is
/// the `k`-th of the `PRODUCTS.len()` classes just before the class of stored trajectories, and
/// `Capacity::media_bytes` states the bytes of a block of each kind in this order.
pub(super) const PRODUCTS: [(&str, Phase); 10] = {
    let mut kinds = [("", Phase::Other); product::KINDS.len()];
    let mut kind = 0;
    while kind < kinds.len() {
        let (name, writer, _) = product::KINDS[kind];
        kinds[kind] = (name, writer);
        kind += 1;
    }
    kinds
};

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
    /// phase left, and:
    ///
    /// - every call but the media read waits until the read completed: the text encoding reads
    ///   the vision pixels it wrote and the latent preparation the condition pixels and samples,
    ///   and nothing later is offered before them;
    /// - the latent preparation waits until the text encoding is handed over (at once when the
    ///   admission found the conditioning), and a denoising step until the preparation is, with
    ///   at most [`DENOISING_LOOKAHEAD`] steps in flight;
    /// - the decoders wait until every denoising step is handed over, and a video decode round
    ///   until the round before it completed: its first unit splices onto the tail of that
    ///   round's last unit;
    /// - a video encode round waits until its decode round completed, the audio encode until the
    ///   audio decode did, and a muxing call until the encode round it hands over did, one muxing
    ///   call at a time; the final one waits for every encode round and the audio encode.
    ///
    /// A predecessor is handed over once its calls are scheduled when it runs on the same group
    /// as `phase`, whose executor runs them in submission order, and once they completed
    /// otherwise, since nothing else orders another group's write before the read.
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
            let along = group.is_some() && group == self.placement[phase.index()];
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
        match phase {
            Phase::MediaRead | Phase::TextEncode => true,
            Phase::LatentPrepare => handed(Phase::TextEncode),
            Phase::Denoise => {
                handed(Phase::LatentPrepare) && in_flight(Phase::Denoise) < DENOISING_LOOKAHEAD
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
    /// encodes the condition latents inside the call. The video units and the encoded units are
    /// one block each, reserved whole at admission, of which each round writes the slice of its
    /// units (from its row's `unit`). A video decode round after the first reads the splice tail
    /// of the unit before its own, which the round before it wrote, and every round writes the
    /// tail of each unit it decodes, one block per unit. Only the last denoising step writes the
    /// latents the decoders read; the steps before it keep the trajectory in the slot. A muxing
    /// call hands over the encode round it follows, and the final one reads nothing.
    pub(super) fn results(
        &self,
        phase: Phase,
        call: u32,
        groups: &[WorkerGroup],
    ) -> (Vec<u32>, u32) {
        use ipc::product::*;

        let last = call.saturating_add(1) == self.video_needed(phase, groups);
        let rounds = self.video_needed(Phase::VideoEncode, groups);
        let (reads, writes): (&[&str], &[&str]) = match phase {
            Phase::MediaRead => (&[], &[CONDITION_PIXELS, CONDITION_SAMPLES, VISION_PIXELS]),
            Phase::TextEncode => (&[VISION_PIXELS], &[CONDITIONING]),
            Phase::LatentPrepare => (&[CONDITIONING, CONDITION_PIXELS, CONDITION_SAMPLES], &[]),
            Phase::Denoise if last => (&[], &[VIDEO_LATENTS, AUDIO_LATENTS]),
            Phase::VideoDecode => (&[VIDEO_LATENTS], &[VIDEO_UNITS]),
            Phase::AudioDecode => (&[AUDIO_LATENTS], &[AUDIO_SAMPLES]),
            Phase::VideoEncode => (&[VIDEO_UNITS], &[ENCODED_UNITS]),
            Phase::AudioEncode => (&[AUDIO_SAMPLES], &[]),
            Phase::Mux if call < rounds => (&[ENCODED_UNITS], &[]),
            _ => (&[], &[]),
        };

        // A product the request does not declare is neither read nor written.
        let declared = |names: &[&str]| -> Vec<u32> {
            names
                .iter()
                .filter_map(|name| self.result(name, 0))
                .collect()
        };
        let (mut reads, mut writes) = (declared(reads), declared(writes));
        if phase == Phase::VideoDecode {
            let units = self.round_units(phase, call, groups);
            let before = units.start.checked_sub(1);
            reads.extend(before.and_then(|unit| self.result(VIDEO_TAIL, unit)));
            writes.extend(units.filter_map(|unit| self.result(VIDEO_TAIL, unit)));
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
    /// "Preemption"; `S/preemption/media.rs:150-175`): a drain preempted it (its terminal is
    /// `Migratable { cause: Drain }`), the daemon gave it a trajectory identity, it completed at
    /// least one denoising step, and it is attached to the group that denoises it, whose slot
    /// holds the trajectory. A lost group's trajectory is gone with it, and step 0 is the
    /// request's noise, which any engine draws again.
    pub(super) fn stores_trajectory(&self) -> bool {
        let Some(video) = self.video() else {
            return false;
        };
        let drained = matches!(
            self.terminal,
            Some(Event::Migratable {
                cause: MigrationCause::Drain,
                ..
            })
        );
        let attached = self.placement[Phase::Denoise.index()]
            .and_then(|group| self.slot_on(group))
            .is_some();
        drained
            && video.trajectory.id != 0
            && self.completed[Phase::Denoise.index()] > 0
            && attached
    }

    /// The key a video request's trajectory at its completed step is stored under,
    /// `latent_content(seed, trajectory, steps)`, with `seed` the latent seed of what the serving
    /// epoch's denoising computes (SD §4.3; `S/trajectories.rs:487`). `None` for a token request.
    pub(super) fn trajectory_key(&self, seed: u64) -> Option<Key> {
        let video = self.video()?;
        let steps = self.completed[Phase::Denoise.index()];
        Some(Key(latent_content(seed, video.trajectory.id, steps)))
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
        Some(Key(latent_content(seed, trajectory.id, steps)))
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
    /// writes it and its product kind, an index into [`PRODUCTS`].
    fn declared(&self) -> Vec<(Phase, usize)> {
        PRODUCTS
            .iter()
            .enumerate()
            .flat_map(|(kind, &(name, phase))| {
                let count = usize::try_from(self.declares(name)).unwrap_or(0);
                iter::repeat_n((phase, kind), count)
            })
            .collect()
    }

    /// The position of a video request's conditioning in its product table: where its admission
    /// pins or allocates it, its text encoding writes it and its latent preparation reads it.
    pub(super) fn conditioning_position(&self) -> Option<u32> {
        self.result(product::CONDITIONING, 0)
    }

    /// Whether a video request holds its conditioning without encoding it: its admission's `Pin`
    /// found it under its key (`encodings`, its one encoder output), before any text encoding
    /// completed, which holds it from then on too.
    fn found_conditioning(&self) -> bool {
        let held = self.encodings.first().is_some_and(|(_, at)| at.is_some());
        held && self.completed[Phase::TextEncode.index()] == 0
    }

    /// The position of the `index`-th declared result of product `name`, if the request
    /// declares it.
    fn result(&self, name: &str, index: u32) -> Option<u32> {
        let mut position: u32 = 0;
        for &(kind, _) in &PRODUCTS {
            let count = self.declares(kind);
            if kind == name {
                return (index < count).then(|| position.saturating_add(index));
            }
            position = position.saturating_add(count);
        }
        None
    }

    /// How many results of product `name` a video request declares (`S/graph.rs:140-242`): a
    /// condition product when its conditions have what it holds, one splice tail per media unit
    /// (commit `e69a91146`), and one of every other product.
    fn declares(&self, name: &str) -> u32 {
        let Some(video) = self.video() else {
            return 0;
        };
        let has = |extent: fn(&ConditionExtent) -> u64| u32::from(total(video, extent) > 0);
        match name {
            product::CONDITION_PIXELS => has(|condition| condition.pixel_bytes / 3),
            product::CONDITION_SAMPLES => has(|condition| condition.audio_samples),
            product::VISION_PIXELS => has(|condition| condition.vision_patches),
            product::VIDEO_TAIL => video.sampling.video_units,
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
    /// video: the ranks of its video decoder block, each taking the units the block deals a rank
    /// (`BlockConfig::rank_units`, `S/execution/media.rs:564-572`); 1 when the group states no
    /// such block. Every rank of the group decodes its share of a round in one call, the round's
    /// units dealt to the ranks in order (SD §3.3).
    fn width(&self, groups: &[WorkerGroup]) -> u32 {
        let Some(block) = self
            .serving(Phase::VideoDecode, groups)
            .and_then(|group| group.config.blocks.get(VIDEO_DECODER_BLOCK))
        else {
            return 1;
        };
        let units = block.ranks.len().saturating_mul(block.rank_units());
        u32::try_from(units).unwrap_or(u32::MAX).max(1)
    }
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
            .declared()
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
        let kind = PRODUCTS
            .iter()
            .position(|&(name, _)| name == product::CONDITIONING)?;
        self.product_class(kind)
    }

    /// The class of product kind `kind`: the `kind`-th of the classes just before the class of
    /// stored trajectories.
    fn product_class(&self, kind: usize) -> Option<u8> {
        let first = usize::from(self.trajectory).checked_sub(PRODUCTS.len())?;
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
        .filter(|(shape, _, products)| products.len() == PRODUCTS.len() && holds(shape, video))
        .min_by_key(|(shape, _, _)| *shape)
}

/// Whether a captured denoising shape holds a video request (`worker::buckets`, the rule a
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
    use std::sync::atomic::AtomicU64;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use ipc::batch::{Batch, BatchId, Columns, Extent, Phase, Shape, Variant};
    use ipc::product;
    use proto::config::{
        BlockConfig, BlockDistribution, SchedulerConfig, VIDEO_DECODER_BLOCK, WorkerGroupConfig,
        WorkerGroupId, WorkerGroupRole, WorkerRank,
    };
    use proto::request::{
        ConditionExtent, DiffusionRequest, DiffusionSamplingParams, EncoderInputs, Event,
        MigrationCause, RequestId, Trajectory,
    };
    use storage::key::{encoder_output, latent_content};
    use storage::runtime::{self, Class, Completion, GroupId, OwnerId, Region, Tier};
    use storage::{Key, QueuePair};

    use super::{PRODUCTS, VIDEO_PHASES};
    use crate::scheduler::cache::DEVICE;
    use crate::scheduler::{Capacity, Executor, Request, RequestStatus, Scheduler, WorkerGroup};

    const A: GroupId = GroupId(0);
    const B: GroupId = GroupId(1);

    /// The fixture's denoising steps, raster and timeline.
    const STEPS: u32 = 4;
    const FRAMES: u32 = 121;
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;

    /// The session's classes: KV blocks, one class per product kind, then stored trajectories,
    /// whose host copies are in region `LATENTS`.
    const TRAJECTORY: u8 = 1 + PRODUCTS.len() as u8;
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
        let products: Vec<u64> = (0..PRODUCTS.len() as u64)
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
            feature_bytes: 0,
            encoder_identity: String::new(),
            media_bytes,
            call_window: 2,
            cache_groups: Vec::new(),
            seeds: Vec::new(),
        }
    }

    /// A group of two ranks whose video decoder deals units to both, one unit each in a round.
    fn config(id: GroupId) -> WorkerGroupConfig {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            VIDEO_DECODER_BLOCK.to_owned(),
            BlockConfig {
                distribution: Some(BlockDistribution::TemporalUnits),
                ..BlockConfig::parallel(vec![0, 1])
            },
        );
        let rank = WorkerRank {
            host: "localhost".to_owned(),
            device: "cpu".to_owned(),
        };
        WorkerGroupConfig {
            id: WorkerGroupId(format!("video-{}", id.0)),
            role: WorkerGroupRole::Model,
            replicas: 1,
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
            regions: (0..=TRAJECTORY)
                .map(|class| {
                    if (1..TRAJECTORY).contains(&class) {
                        vec![0]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
        }
    }

    fn groups(served: &[(GroupId, &[Phase])]) -> Vec<WorkerGroup> {
        served
            .iter()
            .map(|&(id, phases)| WorkerGroup::new(config(id), capacity(phases), layout(id)))
            .collect()
    }

    /// A video request of `units` media units and a prompt of `prompt` tokens.
    fn video(units: u32, prompt: usize, conditions: Vec<ConditionExtent>) -> DiffusionRequest {
        let condition_rows = conditions
            .iter()
            .map(|condition| condition.video_rows() as u32 + condition.audio_rows)
            .sum();
        DiffusionRequest {
            request_id: RequestId(1),
            prompt_token_ids: vec![1; prompt],
            conditioning: b"{}".to_vec(),
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
            input: proto::request::Request::Diffusion(input),
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
            keys: Vec::new(),
            spec_token_ids: Vec::new(),
            num_draft_tokens: 0,
            num_encoded_inputs: 0,
            encodings: Vec::new(),
            unpinned: Vec::new(),
            num_acknowledged_tokens: 0,
            artifact: None,
            terminal: None,
            event_credit: 64,
            last_progress: now,
            dirty: false,
        }
    }

    /// Every video phase on `group`.
    fn on(group: GroupId) -> Vec<(Phase, GroupId)> {
        VIDEO_PHASES.iter().map(|&phase| (phase, group)).collect()
    }

    /// The phases the request may be given a call of now, in the order it offers them.
    fn ready(request: &Request, groups: &[WorkerGroup]) -> Vec<Phase> {
        VIDEO_PHASES
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

    /// An executor that runs nothing and records what it is handed to stamp.
    struct Stamps(Arc<Mutex<Vec<runtime::Request>>>);

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

        fn stamp(&self, _: GroupId, requests: Vec<runtime::Request>) {
            self.0.lock().unwrap().extend(requests);
        }
    }

    /// A scheduler over `served`, in a session whose regions are the product buffer (units of 256
    /// bytes), the slot rows and the host latent pool (units of 4 096 bytes).
    fn scheduler(served: &[(GroupId, &[Phase])]) -> Scheduler {
        stamping(served).0
    }

    /// A [`scheduler`] over `served`, with what its executors were handed to stamp.
    fn stamping(served: &[(GroupId, &[Phase])]) -> (Scheduler, Arc<Mutex<Vec<runtime::Request>>>) {
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
        };
        let sized = |host| Class {
            units: 0,
            host,
            state_in_slot: false,
            tensors: [0; 32],
            encoding: [0; 32],
        };
        let mut classes = vec![Class {
            units: 1,
            host: None,
            state_in_slot: false,
            tensors: [0; 32],
            encoding: [0; 32],
        }];
        classes.extend(PRODUCTS.iter().map(|_| sized(None)));
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
            trajectory: TRAJECTORY,
            peers: Vec::new(),
            groups: ids.iter().map(|&id| layout(id)).collect(),
            schedulers: vec![ids.clone()],
        };
        let opened = runtime::Opened {
            regions: vec![(4, 1 << 20), (5, 1 << 20), (6, 1 << 20)],
            segments: Vec::new(),
            registers: vec![runtime::Registers {
                completed: (0..ids.len() as u32).collect(),
                capacity: vec![(4, 4), (5, 7), (6, 10)],
            }],
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
            Arc::new(AtomicU64::new(0)),
        )
        .unwrap();
        (scheduler, stamped)
    }

    /// The calls of each phase follow the request's conditions and the widths of the groups that
    /// serve it (SD §3.3).
    #[test]
    fn a_request_makes_the_calls_of_its_conditions_and_rounds() {
        let groups = groups(&[(A, &VIDEO_PHASES)]);
        let needed = |request: &Request| {
            VIDEO_PHASES.map(|phase| (phase, request.video_needed(phase, &groups)))
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
        let groups = groups(&[(A, &VIDEO_PHASES)]);
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
        let groups = groups(&[(A, &VIDEO_PHASES)]);
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
        let (mut scheduler, stamped) = stamping(&[(A, &VIDEO_PHASES)]);
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
        let key = Key(encoder_output(product::CONDITIONING, "", &[3; 32]));
        assert_eq!(admitted.encodings, [(key, None)]);
        let at = admitted.conditioning_position().unwrap();
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
        let listed = (key, class, 40, DEVICE, 1);
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
        let found = |runs: u32| Completion::Pinned {
            runs: (0..runs).map(|unit| (4, unit, 1)).collect(),
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
        assert_eq!((results[0], reads), (at, 3));

        // Encoded instead: the text encoding's output is committed under the key, the prompt's
        // rows as its tokens, and the request still made its one text encoding.
        scheduler.requests.insert(id, admitted);
        let mut encoded = Batch {
            id: BatchId(1),
            group: A,
            phase: Phase::TextEncode,
            generation: 0,
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
        assert_eq!(*stamped.lock().unwrap(), [commit]);
        let groups = &scheduler.groups;
        assert_eq!(
            scheduler.requests[&id].video_needed(Phase::TextEncode, groups),
            1
        );
    }

    /// A phase placed on another group than the one it reads from waits until that one's calls
    /// completed, since no executor orders the two.
    #[test]
    fn a_phase_on_another_group_waits_until_its_predecessor_completed() {
        let groups = groups(&[(A, &VIDEO_PHASES), (B, &[Phase::TextEncode])]);
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
        let groups = groups(&[(A, &VIDEO_PHASES)]);
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
        let groups = groups(&[(A, &VIDEO_PHASES)]);
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

        // The encode round reads the video units, a muxing call the encoded round, and the final
        // muxing call nothing.
        let (encode, _) = results(Phase::VideoEncode, 0);
        assert_eq!(encode[0], first[1]);
        assert_eq!(results(Phase::Mux, 1), (vec![encode[1]], 1));
        assert_eq!(results(Phase::Mux, 2), (Vec::new(), 0));
    }

    /// Every result a call writes is a declared result of the call's phase, and each is written
    /// once, but the video units and the encoded units, of which each round writes its slice;
    /// each block is sized in its product class, rounded up to units of the product region
    /// (SD §4.4 "Products").
    #[test]
    fn declared_results_are_written_by_their_phase_and_sized_in_their_class() {
        let scheduler = scheduler(&[(A, &VIDEO_PHASES)]);
        let groups = &scheduler.groups;
        for request in [
            request(video(3, 40, Vec::new()), &on(A)),
            request(video(2, 40, reference()), &on(A)),
        ] {
            let declared = scheduler.declared_results(&request).unwrap();
            let mut written = vec![0; declared.len()];
            for phase in VIDEO_PHASES {
                for call in 0..request.video_needed(phase, groups) {
                    let (results, reads) = request.results(phase, call, groups);
                    for &position in &results[reads as usize..] {
                        assert_eq!(declared[position as usize].0, phase);
                        written[position as usize] += 1;
                    }
                }
            }
            // The rounds of each sliced block: the decode rounds and their encodes.
            let rounds = request.video_needed(Phase::VideoDecode, groups);
            let sliced = [
                (request.results(Phase::VideoDecode, 0, groups).0[1], rounds),
                (request.results(Phase::VideoEncode, 0, groups).0[1], rounds),
            ];
            for (position, count) in written.into_iter().enumerate() {
                let expected = sliced
                    .iter()
                    .find(|&&(sliced, _)| sliced as usize == position)
                    .map_or(1, |&(_, rounds)| rounds);
                assert_eq!(count, expected, "result {position}");
            }

            // The kind of a block is its class's place before the class of trajectories, and it
            // takes one byte more than that place's units.
            for &(_, class, units) in &declared {
                assert!((1..TRAJECTORY).contains(&class));
                assert_eq!(units, u32::from(class));
            }
        }

        // A text request of three units declares the conditioning, both latents, the video
        // units, one tail per unit, the encoded units and the audio; a conditioned request of two
        // the condition pixels and samples and the vision pixels besides, and no encoded
        // condition of its own.
        let text = request(video(3, 40, Vec::new()), &on(A));
        assert_eq!(scheduler.declared_results(&text).unwrap().len(), 9);
        let conditioned = request(video(2, 40, reference()), &on(A));
        assert_eq!(scheduler.declared_results(&conditioned).unwrap().len(), 11);
    }

    /// A trajectory is sized from the smallest captured shape that holds its request, in units of
    /// the host latent region (SD §3.8, §4.4).
    #[test]
    fn a_trajectory_is_sized_in_units_of_the_host_latent_region() {
        let scheduler = scheduler(&[(A, &VIDEO_PHASES)]);
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
        let all = scheduler(&[(A, &VIDEO_PHASES)]);
        assert_eq!(all.validate_video(&video(2, 40, Vec::new())), Ok(()));
        assert_eq!(all.validate_video(&video(2, 40, reference())), Ok(()));

        let empty = video(2, 0, Vec::new());
        assert_eq!(
            all.validate_video(&empty),
            Err("media prompt tokens must not be empty".to_owned())
        );

        let without = |missing: &[Phase]| {
            let phases: Vec<Phase> = VIDEO_PHASES
                .into_iter()
                .filter(|phase| !missing.contains(phase))
                .collect();
            scheduler(&[(A, &phases)])
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

    /// A request a drain preempted after a completed step stores its trajectory under the key of
    /// that step, and a resumed request restores the steps it names under the seed that stored
    /// them (SD §3.8, §4.3, §7.3).
    #[test]
    fn a_drained_request_stores_its_step_and_resumes_from_it() {
        let groups = groups(&[(A, &VIDEO_PHASES)]);
        let mut input = video(1, 40, Vec::new());
        input.trajectory.id = 77;
        let drained = |completed: u32, cause: MigrationCause, placement: &[(Phase, GroupId)]| {
            let mut request = request(input.clone(), placement);
            request.completed[Phase::Denoise.index()] = completed;
            request.terminal = Some(Event::Migratable {
                cause,
                denoised_steps: 0,
                stored_layout: Vec::new(),
            });
            request
        };

        let stored = drained(2, MigrationCause::Drain, &on(A));
        assert!(stored.stores_trajectory());
        assert_eq!(
            stored.trajectory_key(5),
            Some(Key(latent_content(5, 77, 2)))
        );
        assert!(!drained(2, MigrationCause::Loss, &on(A)).stores_trajectory());
        assert!(!drained(0, MigrationCause::Drain, &on(A)).stores_trajectory());
        assert!(!drained(2, MigrationCause::Drain, &[]).stores_trajectory());
        let mut anonymous = drained(2, MigrationCause::Drain, &on(A));
        if let proto::request::Request::Diffusion(video) = &mut anonymous.input {
            video.trajectory.id = 0;
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

    /// Progress names what the request is doing and the steps its trajectory completed
    /// (SD §7.3 "Progress and artifact").
    #[test]
    fn progress_names_the_stage_of_the_request() {
        let groups = groups(&[(A, &VIDEO_PHASES)]);
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
