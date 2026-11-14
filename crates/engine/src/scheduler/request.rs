//! One request of a scheduler from its arrival to its end (`specs/scheduler-design.md` §2.2):
//! [`Request`], its status ([`RequestStatus`]) and its table of each class at the buffer pool as
//! the scheduler counts it ([`Blocks`]).
//!
//! A request is one row of plain fields, vLLM's `Request`: what the daemon submitted, where the
//! request runs, the calls of each phase scheduled and applied, its token and video progress, the
//! owners it holds at the buffer pool and the positions of each class's table. It holds no unit and
//! no run: a batch names its slot and positions, and the workers find its blocks through their
//! device block tables. Everything else about it is derived from these fields (SD §2.2, §3.3).
//!
//! This file also holds what a token request's fields say about its next call (SD §3.3): how
//! many calls of a phase it makes ([`Request::needed`]), whether it may be given one more now
//! ([`Request::ready`]), the chunk a prefill or a refill takes, the drafts a verification
//! carries, the stop rules and the consumer's window. A video request's media rows of the two are
//! `video.rs`'s. The validation of a request at its arrival is [`Scheduler::validate`].
//!
//! # A token request's context
//!
//! The context is what its prefill writes: the prompt's text tokens with the blocks of its input
//! images, then the tokens a resumed request generated before. Each encoder input of an image
//! writes one block of `num_cache_tokens` rows, before the prompt token at the image's position,
//! the encoders of an image in order and images at one position in input order. `token_ids` holds
//! one id per cache position (an image block's rows are id 0), so `num_prompt_tokens` is the
//! context's length and `num_computed_tokens` counts cache positions, image rows included
//! ([`Request::image_spans`]). A chunk takes a block whole or ends before it, so a block is
//! either written or wholly ahead of `num_computed_tokens`.
//!
//! Encoder input `i` (in context order) has its output at entry `i` of `encodings`. The entry's
//! position is set once the request holds that output: from `Pinned` at its admission, or once
//! the encode call that computed it was applied. An encoder input whose entry has no position
//! still needs its encode call.

use std::time::Instant;

use crate::batch::Phase;
use proto::request::{
    CacheResume, EVENT_WINDOW, Event, FinishReason, GenerationFeatures, GenerationRequest,
    ImageEncoder, ImageEncoderInput, ImageInput, RequestId, StopReason,
};
use storage::Key;
use storage::runtime::{GroupId, OwnerId};

use super::group::variants;
use super::video::VIDEO_PHASES;
use super::{Scheduler, WorkerGroup};

/// The most drafts one verification carries (SD §7.2, `S/lib.rs:611`): the bound the consumer's
/// window reserves for a verification's tokens.
pub(super) const MAX_VERIFY_DRAFTS: u32 = 16;

/// The events a request's window keeps free for its terminal events (SD §7.1).
const TERMINAL_RESERVE: u32 = 2;

/// A request of the scheduler.
///
/// Invariants: `completed[p] <= scheduled[p]` and `num_computed_tokens <= token_ids.len()`; no
/// call writes a block the request does not hold or one that is read-only; it holds one owner
/// per group it is attached to, from that group's `Attach` (or the `Transfer` that moves the
/// owner there) until that owner's `Detach`.
#[derive(Clone, Debug)]
pub(crate) struct Request {
    // What the daemon submitted. Immutable.
    pub id: RequestId,
    /// The token or video request: prompt, sampling, images, limits, priority, cache policy, and
    /// what a resumed request generated.
    pub input: proto::request::Request,
    pub arrival: Instant,

    // Where it is.
    pub status: RequestStatus,
    /// Per phase, by [`Phase::index`], the group that serves it; `None` until placed.
    pub placement: [Option<GroupId>; Phase::COUNT],
    /// Per group the request is attached to: its owner there (drawn at each admission), its slot
    /// row there (from `Attached` or `Transferred`; 0, a row no owner holds, until then), and
    /// whether that group has seen it.
    pub attachments: Vec<(GroupId, OwnerId, u32, bool)>,
    /// Per phase, the calls handed to the executor.
    pub scheduled: [u32; Phase::COUNT],
    /// Per phase, the calls whose output was applied, in index order.
    pub completed: [u32; Phase::COUNT],
    /// Outputs that returned ahead of an earlier call of the same phase: the phase and the call's
    /// index.
    pub early: Vec<(Phase, u32)>,
    /// Per class, its table at the buffer pool as the scheduler counts it.
    pub blocks: Vec<Blocks>,
    /// Whether it is in the scheduler's waiting queue.
    pub queued: bool,
    /// The answers its admission or its transfer still waits for.
    pub pending: u32,

    // Token progress.
    /// The prompt, then the output; a resumed request's generated tokens are prompt.
    pub token_ids: Vec<u32>,
    pub num_prompt_tokens: u32,
    /// Tokens written or scheduled to be.
    pub num_computed_tokens: u32,
    /// The tokens of its prefix hit, over all tiers.
    pub num_cached_tokens: u32,
    /// Of `num_cached_tokens`, the tokens its admission loaded from the host tier: the hit
    /// beyond what its `Pin`s found on the device. Set with the hit once the admission's last
    /// answer arrived, and counted as host hits at admission.
    pub num_host_tokens: u32,
    /// Of `num_host_tokens`, the tokens a `Prefetch` from the peers brought to the host (the
    /// peers' `Prefetched`), counted as peer hits at admission.
    pub num_peer_tokens: u32,
    /// The `List`s the scheduler had asked when the request arrived. A resumed request whose
    /// expected hit falls short of what its previous life computed is admitted once a `List`
    /// asked after it arrived has been answered (SD §4.4 "The hit").
    pub listing: u64,
    /// Per cache group, the key of every complete block of `token_ids`.
    pub keys: Vec<Vec<Key>>,
    /// The drafts the next verification carries.
    pub spec_token_ids: Vec<u32>,
    /// Tokens the draft group has computed.
    pub num_draft_tokens: u32,
    /// Images whose encoding is placed.
    pub num_encoded_inputs: u32,
    /// The key of each encoder output, and the position the scheduler named for it in the
    /// request's encoder-output table, once held. A video request's one encoder output is its
    /// conditioning, held at its position in the request's product table.
    pub encodings: Vec<(Key, Option<u32>)>,
    /// The keys of blocks the request holds that `Cached::pins` does not count it among the
    /// holders of: a block answered `Duplicate`, which stays private to it (BP §9, R4), and an
    /// encoder output it computed and did not commit. Giving up its pins passes over them.
    pub unpinned: Vec<Key>,
    /// The crossings a video request's calls wait for (SD §3.3, BP §15 "A product read in
    /// another process"): per declared result that a group of its placement reads in a memory
    /// its writer does not write it in, the result's position, that group, the region (by its
    /// number) of the run its writer writes it in, and where the crossing is: `None` until it is
    /// asked, `Some(false)` while it is asked and not answered, `Some(true)` once it was answered
    /// `Replicated`. Set by its admission.
    pub crossings: Vec<(u32, GroupId, u8, Option<bool>)>,
    /// Output tokens the daemon's decoder has passed.
    pub num_acknowledged_tokens: u32,

    // Output and clocks.
    /// The terminal event, held until the calls in flight return when it is `Migratable`.
    pub terminal: Option<Event>,
    /// Events the consumer can still take.
    pub event_credit: u32,
    /// Its admission or its last applied call.
    pub last_progress: Instant,
    /// Whether its view changed since the policy was told.
    pub dirty: bool,
}

/// Where a request is (vLLM's `RequestStatus`).
///
/// Only `Running` requests are in batches; `Finished` is absorbing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RequestStatus {
    Waiting,
    /// Answers of its admission or its transfer are outstanding at the buffer pool.
    WaitingForRemoteKvs,
    Running,
    Finished,
}

/// A request's table of one class at the buffer pool, as the scheduler counts it.
///
/// Invariant: `first <= shared <= end <= asked`. The counts are positions in the table of the
/// owner on the group that writes the class; in a cache group's table a position is the logical
/// index of its block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Blocks {
    /// Entries before it were given up or never asked for: for a sliding window, its window's
    /// first block.
    pub first: u32,
    /// Entries before it are read-only: pinned, loaded, or committed through the executor.
    pub shared: u32,
    /// Entries held. Raised only by completions.
    pub end: u32,
    /// Entries asked for, held or not.
    pub asked: u32,
}

impl Request {
    /// The group the owner `owner` of this request is attached to, and its slot row there.
    pub(crate) fn attachment(&self, owner: OwnerId) -> Option<(GroupId, u32)> {
        self.attachments
            .iter()
            .find(|(_, attached, _, _)| *attached == owner)
            .map(|&(group, _, slot, _)| (group, slot))
    }

    /// The slot row of this request's own owner on `group`, if it is attached there.
    pub(crate) fn slot_on(&self, group: GroupId) -> Option<u32> {
        self.attachments
            .iter()
            .find(|(attached, _, _, _)| *attached == group)
            .map(|&(_, _, slot, _)| slot)
    }

    /// The owner of this request on the group placed for `phase`, which every request for the
    /// classes that phase writes names (SD §4.2); `None` while the phase is placed nowhere or the
    /// request is not attached to its group.
    pub(crate) fn owner_on(&self, phase: Phase) -> Option<OwnerId> {
        let group = self.placement.get(phase.index()).copied().flatten()?;
        self.attachments
            .iter()
            .find(|(attached, _, _, _)| *attached == group)
            .map(|&(_, owner, _, _)| owner)
    }
}

/// A request's arrival, its progress and its next call (SD §2.2, §3.2, §3.3, §7.2).
impl Request {
    /// A request as the daemon submitted it, waiting from `arrival` (SD §3.2).
    ///
    /// A token request's context is laid out as the module documentation states: a resumed
    /// request's generated tokens continue its prompt, so they are computed as prompt and its
    /// generated count continues from them, while its usage charges its own prompt only. A token
    /// request without `cache.resume` is given `{epoch, request_id}`, `epoch` being the engine
    /// epoch's (`S/admission.rs:116-121`). The request is queued, with nothing placed, held or
    /// scheduled and the consumer's whole window as its credit; its `blocks`, `keys` and
    /// `encodings` are its admission's to fill.
    pub(crate) fn new(input: proto::request::Request, epoch: u64, arrival: Instant) -> Self {
        let mut input = input;
        let token_ids = match &mut input {
            proto::request::Request::Generation(request) => {
                if request.cache.resume.is_none() {
                    request.cache.resume = Some(CacheResume {
                        epoch,
                        request_id: request.request_id.0,
                    });
                }
                context(request)
            }
            // A video request's prompt is its text encoding's input, which reads `input`.
            proto::request::Request::Diffusion(_) => Vec::new(),
        };
        let num_prompt_tokens = len32(token_ids.len());

        Self {
            id: input.request_id(),
            input,
            arrival,
            status: RequestStatus::Waiting,
            placement: [None; Phase::COUNT],
            attachments: Vec::new(),
            scheduled: [0; Phase::COUNT],
            completed: [0; Phase::COUNT],
            early: Vec::new(),
            blocks: Vec::new(),
            queued: true,
            pending: 0,
            token_ids,
            num_prompt_tokens,
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
            event_credit: u32::try_from(EVENT_WINDOW).unwrap_or(u32::MAX),
            last_progress: arrival,
            dirty: false,
        }
    }

    /// The token request this request is; `None` for a video request.
    pub(crate) fn generation(&self) -> Option<&GenerationRequest> {
        match &self.input {
            proto::request::Request::Generation(request) => Some(request),
            proto::request::Request::Diffusion(_) => None,
        }
    }

    /// Whether the request continues what an engine began before the daemon resubmitted it: a
    /// token request that carries generated tokens, a video request with stored denoising steps.
    /// Such a request waits ahead of every other (SD §3.2).
    pub(crate) fn resumes(&self) -> bool {
        match &self.input {
            proto::request::Request::Generation(request) => request.is_migrated(),
            proto::request::Request::Diffusion(video) => video.trajectory.denoised_steps > 0,
        }
    }

    /// The tokens a token request generated, those a resumed request brought included: what its
    /// token limit, its `min_tokens` and its usage count.
    pub(crate) fn generated(&self) -> u32 {
        let resumed = self
            .generation()
            .map_or(0, |request| len32(request.generated_token_ids.len()));
        resumed.saturating_add(self.output_tokens())
    }

    /// The calls of `phase` handed to the executor whose output has not been applied.
    pub(crate) fn in_flight(&self, phase: Phase) -> u32 {
        let index = phase.index();
        match (self.scheduled.get(index), self.completed.get(index)) {
            (Some(scheduled), Some(completed)) => scheduled.saturating_sub(*completed),
            _ => 0,
        }
    }

    /// Whether no call of the request is in flight.
    pub(crate) fn idle(&self) -> bool {
        Phase::ALL
            .into_iter()
            .all(|phase| self.in_flight(phase) == 0)
    }

    /// The blocks of a token request's input images in its context (SD §2.2, "the request's
    /// image spans"), in context order: each block's first cache position, its rows, its image
    /// and its encoder input. The `i`-th block is encoder input `i`, whose output is
    /// `encodings[i]`. Empty for a video request.
    pub(crate) fn image_spans(
        &self,
    ) -> impl Iterator<Item = (u32, u32, &ImageInput, &ImageEncoderInput)> + '_ {
        let images = self
            .generation()
            .map_or(&[][..], |request| &request.multimodal_inputs.images[..]);
        images
            .iter()
            .flat_map(|image| image.encoders.iter().map(move |input| (image, input)))
            .scan(0u32, |before, (image, input)| {
                // A block starts after the prompt tokens before its image and every earlier block.
                let start = image.position.saturating_add(*before);
                let rows = rows(input);
                *before = before.saturating_add(rows);
                Some((start, rows, image, input))
            })
    }

    /// How many calls of `phase` the request makes in all (SD §3.3). A token request's
    /// prefills, decodes, verifications and refills are not counted: each runs while its state
    /// calls for it, and this is `u32::MAX`. Its vision and latent encodings are one per encoder
    /// input of that kind whose output its admission's `Pin` did not find: those applied, and
    /// those whose output it still lacks. A video request's calls are `video_needed`'s.
    pub(crate) fn needed(&self, phase: Phase, groups: &[WorkerGroup]) -> u32 {
        if self.is_video() {
            return self.video_needed(phase, groups);
        }
        match phase {
            Phase::Prefill | Phase::Decode | Phase::Verify | Phase::Refill => u32::MAX,
            Phase::VisionEncode => self.encodings_of(ImageEncoder::Vision),
            Phase::LatentEncode => self.encodings_of(ImageEncoder::Latent),
            _ => 0,
        }
    }

    /// Whether the request may be given one more call of `phase` now (SD §3.3): it runs, it has
    /// calls of the phase left, and its state allows one. Neither the consumer's window
    /// ([`Request::has_event_room`]) nor the group's room is asked here. A video request's rows
    /// are `video_ready`'s; a token request's, on the group placed for the phase, are:
    ///
    /// - a prefill chunk, while the context is not computed and nothing of the request is in
    ///   flight, unless the context continues with an image block whose output it lacks;
    /// - a vision or latent encoding, of the next encoder input whose output it lacks, with
    ///   nothing in flight: a vision block while its image lies within the widest chunk of the
    ///   text computed, a latent block once the context reached it (`S/batching/next.rs:542-568`);
    /// - a decode, once the context is computed, in a deployment that does not speculate, behind
    ///   at most its prompt's last chunk and its decodes, fewer than the group's call window, none
    ///   when it bans a multi-token word, and not so many that they could reach its token limit
    ///   (`S/execution/projection.rs:290-352`, `:426-436`);
    /// - a verification, once the context is computed, in a deployment that speculates, with
    ///   nothing in flight;
    /// - a refill of its draft cache in such a deployment, once the context is computed, with
    ///   nothing in flight, while the draft cache ends before [`Request::refill_end`];
    /// - under stop strings, no token call while the outputs of as many calls as the group's call
    ///   window are unacknowledged (`:486`).
    pub(crate) fn ready(&self, phase: Phase, groups: &[WorkerGroup]) -> bool {
        if self.status != RequestStatus::Running {
            return false;
        }
        let Some(request) = self.generation() else {
            return self.video_ready(phase, groups);
        };
        let left = self
            .scheduled
            .get(phase.index())
            .is_some_and(|&scheduled| scheduled < self.needed(phase, groups));
        let Some(group) = self.placed(phase, groups).filter(|_| left) else {
            return false;
        };

        let computed = self.num_computed_tokens >= self.num_prompt_tokens;
        let speculates = groups.iter().any(|group| group.serves(Phase::Verify));
        let window = group.capacity.call_window.max(1);
        let acknowledged = request.stop_strings.is_empty() || self.unacknowledged() < window;
        match phase {
            Phase::Prefill => acknowledged && !computed && self.idle() && !self.awaits_encoding(),
            Phase::VisionEncode => self.idle() && self.encodes_next(ImageEncoder::Vision, groups),
            Phase::LatentEncode => self.idle() && self.encodes_next(ImageEncoder::Latent, groups),
            Phase::Decode => {
                acknowledged && computed && !speculates && self.pipelines(request, window)
            }
            Phase::Verify => acknowledged && computed && speculates && self.idle(),
            Phase::Refill => {
                acknowledged
                    && computed
                    && speculates
                    && self.idle()
                    && self.num_draft_tokens < self.refill_end()
            }
            _ => false,
        }
    }

    /// The phases the request may be given a call of now, in the order its next call is named:
    /// the policy's view of it names the first (SD §5.2). A token request's encodings come before
    /// its prefill; a video request's phases follow `VIDEO_PHASES`. A refill is never named: only
    /// the policy's `refill` asks for one.
    pub(crate) fn ready_phases<'a>(
        &'a self,
        groups: &'a [WorkerGroup],
    ) -> impl Iterator<Item = Phase> + 'a {
        let order: &'static [Phase] = if self.is_video() {
            &VIDEO_PHASES
        } else {
            &Phase::ALL
        };
        order
            .iter()
            .copied()
            .filter(move |&phase| phase != Phase::Refill && self.ready(phase, groups))
    }

    /// The end the latest token of a token request gives it, if any (`S/output.rs:794-812`).
    /// The token is in `token_ids` already, so it counts as generated. In order: a stop token
    /// ends it with `Stop` unless at most `min_tokens` tokens are generated; reaching the token
    /// limit ends it with `MaxTokens` whatever the floor; an end-of-sequence token of `eos` ends
    /// it with `Eos` unless it ignores them or is under the floor.
    pub(crate) fn check_stop(
        &self,
        token: u32,
        eos: &[u32],
    ) -> Option<(FinishReason, Option<StopReason>)> {
        let request = self.generation()?;
        let generated = usize::try_from(self.generated()).unwrap_or(usize::MAX);
        let floor = generated <= request.sampling.min_tokens;

        if !floor && request.stop_token_ids.contains(&token) {
            Some((FinishReason::Stop, Some(StopReason::Token(token))))
        } else if generated >= request.max_und_tokens {
            Some((FinishReason::MaxTokens, None))
        } else if !floor && !request.sampling.ignore_eos && eos.contains(&token) {
            Some((FinishReason::Eos, None))
        } else {
            None
        }
    }

    /// Whether a token that [`Request::check_stop`] judged `stop` is sent to the daemon
    /// (`S/output.rs:862-893`): every token that continues the request; a stop token only with
    /// `include_stop_token`; any other token that ends it unless it is one of `eos`.
    pub(crate) fn publishes(
        &self,
        token: u32,
        stop: Option<&(FinishReason, Option<StopReason>)>,
        eos: &[u32],
    ) -> bool {
        match stop {
            None => true,
            Some((_, Some(StopReason::Token(_)))) => self
                .generation()
                .is_some_and(|request| request.include_stop_token),
            Some(_) => !eos.contains(&token),
        }
    }

    /// Whether the consumer's window holds the events one more call of `phase` may add (SD §3.4,
    /// §7.1): the credit covers the bound of every call in flight, the terminal's reserve and
    /// the call's own bound. A request whose consumer is slow keeps its slot and blocks nobody.
    pub(crate) fn has_event_room(&self, phase: Phase) -> bool {
        let in_flight = Phase::ALL
            .into_iter()
            .map(|phase| self.in_flight(phase).saturating_mul(output_bound(phase)))
            .fold(0, u32::saturating_add);
        let needed = in_flight
            .saturating_add(TERMINAL_RESERVE)
            .saturating_add(output_bound(phase));
        self.event_credit >= needed
    }

    /// How many drafts the request's next verification carries, `spec_token_ids[..drafts]`
    /// (SD §7.2 "Speculation", `S/speculation.rs:229-267`): none while its draft cache does not
    /// cover its computed tokens; else what its last forward proposed, at most
    /// [`MAX_VERIFY_DRAFTS`] and one fewer than the tokens it may still generate, cut before the
    /// first draft at whose position the request bans other tokens than at the verification's
    /// first (a bad word's prefix forming across the drafts, or the `min_tokens` floor crossed),
    /// since a verification's rows sample under one banned set. `eos` are the deployment's
    /// end-of-sequence tokens.
    pub(crate) fn drafts(&self, eos: &[u32]) -> u32 {
        let Some(request) = self.generation() else {
            return 0;
        };
        if self.num_draft_tokens < self.num_computed_tokens {
            return 0;
        }

        let limit = u64::try_from(request.max_und_tokens).unwrap_or(u64::MAX);
        let remaining = limit
            .saturating_sub(u64::from(self.generated()))
            .saturating_sub(1);
        let most = self
            .spec_token_ids
            .len()
            .min(MAX_VERIFY_DRAFTS as usize)
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        let proposals = self.spec_token_ids.get(..most).unwrap_or_default();

        let first = self.banned(eos, 0, &[]);
        let uniform = (1..=most)
            .take_while(|&ahead| {
                self.banned(eos, 0, proposals.get(..ahead).unwrap_or_default()) == first
            })
            .count();
        len32(uniform)
    }

    /// The tokens a token request may not sample at the position after its generated tokens,
    /// `ahead` more that its calls in flight commit, and `drafts` (SD §7.2 "Tokens banned per
    /// step", `S/batching/next.rs:641-671`), sorted and deduplicated: the end-of-sequence tokens
    /// `eos` while those tokens are fewer than `min_tokens`, and the last token of every bad word
    /// whose prefix the generated tokens and the drafts end with (vLLM's `apply_bad_words`). The
    /// tokens `ahead` are not known yet; a request that bans a multi-token word has none in
    /// flight when it decodes, so they matter only for the floor.
    pub(crate) fn banned(&self, eos: &[u32], ahead: u32, drafts: &[u32]) -> Vec<u32> {
        let Some(request) = self.generation() else {
            return Vec::new();
        };
        let sampling = &request.sampling;
        let generated = self.generated_ids();

        let mut banned = Vec::new();
        let count = usize::try_from(self.generated().saturating_add(ahead)).unwrap_or(usize::MAX);
        if count.saturating_add(drafts.len()) < sampling.min_tokens {
            banned.extend_from_slice(eos);
        }
        for word in &sampling.bad_words_ids {
            let Some((&last, prefix)) = word.split_last() else {
                continue;
            };
            // A prefix longer than the drafts ends in the generated tokens, a shorter one in the
            // drafts alone.
            let matches = if prefix.len() <= drafts.len() {
                drafts.ends_with(prefix)
            } else {
                let (head, tail) = prefix.split_at(prefix.len() - drafts.len());
                tail == drafts && generated.ends_with(head)
            };
            if matches {
                banned.push(last);
            }
        }
        banned.sort_unstable();
        banned.dedup();
        banned
    }

    /// The first token a refill of the request's draft cache evaluates (SD §7.2 "A refill
    /// chunk", `S/speculation.rs:394-412`): the end of its valid draft cache when the model keeps
    /// no state per slot or the request holds the state checkpoint that ends there, 0 otherwise.
    /// `state` is the class of the model's state checkpoints, by its index in `blocks`, with the
    /// tokens one checkpoint block spans, `None` for a model that keeps no state per slot; the
    /// checkpoint ending at token `t` is at position `t / block - 1` of that class's table.
    pub(crate) fn refill_start(&self, state: Option<(usize, u32)>) -> u32 {
        let start = self.num_draft_tokens;
        let resumes = match state {
            None => true,
            Some(_) if start == 0 => true,
            Some((class, block)) => {
                block > 0
                    && start.is_multiple_of(block)
                    && self.blocks.get(class).is_some_and(|table| {
                        let position = start / block - 1;
                        table.first <= position && position < table.end
                    })
            }
        };
        if resumes { start } else { 0 }
    }

    /// Where a refill of the request's draft cache ends (SD §3.3 `Refill`, §7.2 "Speculation"):
    /// at the tokens its cache holds, or one position before the first image block that ends
    /// after the draft cache does. A draft entry is the drafter's evaluation of a position with
    /// the token after it, and an image block's rows carry no token: no refill row reads one,
    /// and none takes one as the token after its last (`rows.rs`). The draft cache counts its
    /// entries from its start, so it never grows past the image either: such a request verifies
    /// without drafts from there on.
    pub(crate) fn refill_end(&self) -> u32 {
        let image = self
            .image_spans()
            .filter(|&(_, rows, ..)| rows > 0)
            .find(|&(block, rows, ..)| block.saturating_add(rows) > self.num_draft_tokens);
        match image {
            Some((block, ..)) => block.saturating_sub(1).min(self.num_computed_tokens),
            None => self.num_computed_tokens,
        }
    }

    /// The tokens the next chunk of the request's refill takes within `budget` tokens (SD §3.4,
    /// §7.2 "A refill chunk", `S/speculation.rs:394-441`): from [`Request::refill_start`] toward
    /// [`Request::refill_end`], ending on a checkpoint boundary of `state` when it stops short of
    /// it, so that the next chunk continues from a kept checkpoint. 0 when no chunk fits.
    pub(crate) fn refill_chunk(&self, budget: u32, state: Option<(usize, u32)>) -> u32 {
        let start = self.refill_start(state);
        let target = self.refill_end();
        let mut end = start.saturating_add(budget).min(target);
        if end < target
            && let Some((_, block)) = state
            && block > 0
        {
            end = end / block * block;
        }
        end.saturating_sub(start)
    }

    /// The cache positions a prefill chunk of at most `tokens` takes from `num_computed_tokens`
    /// (SD §3.4, §7.1 "A prefill chunk", `S/batching/next.rs:438-500`): the text up to the limit
    /// and each vision block it reaches whole, when the request holds the block's output and the
    /// limit holds the block or the block starts the chunk (the step's budget then decides alone,
    /// in `charge`); otherwise the chunk ends before the block. A latent block is a prefill of its
    /// own: a chunk that starts with it takes it whole and nothing more, any other ends before it.
    pub(crate) fn clip_to_images(&self, tokens: u32) -> u32 {
        let start = self.num_computed_tokens;
        let mut end = start.saturating_add(tokens);
        for (index, (block, rows, _, input)) in self.image_spans().enumerate() {
            if block < start || rows == 0 {
                continue;
            }
            if block >= end {
                break;
            }

            let first = block == start;
            let held = self.holds_encoding(index);
            match input.encoder {
                ImageEncoder::Latent => return if first && held { rows } else { block - start },
                ImageEncoder::Vision if !held => return block - start,
                ImageEncoder::Vision if first => end = end.max(block.saturating_add(rows)),
                ImageEncoder::Vision if block.saturating_add(rows) > end => return block - start,
                ImageEncoder::Vision => {}
            }
        }
        end - start
    }

    /// The tokens this engine generated: `token_ids` after the context.
    fn output_tokens(&self) -> u32 {
        len32(self.token_ids.len()).saturating_sub(self.num_prompt_tokens)
    }

    /// The generated tokens in order: a resumed request's, which end its context, then this
    /// engine's.
    pub(super) fn generated_ids(&self) -> &[u32] {
        let resumed = self
            .generation()
            .map_or(0, |request| request.generated_token_ids.len());
        let start = (self.num_prompt_tokens as usize).saturating_sub(resumed);
        self.token_ids.get(start..).unwrap_or_default()
    }

    /// The output tokens sent that the daemon's decoder has not passed yet.
    fn unacknowledged(&self) -> u32 {
        self.output_tokens()
            .saturating_sub(self.num_acknowledged_tokens)
    }

    /// The group placed for `phase`, if any.
    fn placed<'a>(&self, phase: Phase, groups: &'a [WorkerGroup]) -> Option<&'a WorkerGroup> {
        let id = self.placement.get(phase.index()).copied().flatten()?;
        groups.iter().find(|group| group.id() == id)
    }

    /// Whether the request holds the output of encoder input `index`.
    fn holds_encoding(&self, index: usize) -> bool {
        self.encodings
            .get(index)
            .is_some_and(|(_, position)| position.is_some())
    }

    /// The encodings of `kind` a token request makes: those applied, and one per encoder input
    /// of that kind whose output it lacks. An applied encoding's output is held, so the sum stays
    /// as the encodings complete.
    fn encodings_of(&self, kind: ImageEncoder) -> u32 {
        let lacking = self
            .image_spans()
            .enumerate()
            .filter(|(index, (_, _, _, input))| {
                input.encoder == kind && !self.holds_encoding(*index)
            })
            .count();
        let phase = match kind {
            ImageEncoder::Vision => Phase::VisionEncode,
            ImageEncoder::Latent => Phase::LatentEncode,
        };
        self.completed[phase.index()].saturating_add(len32(lacking))
    }

    /// Whether the context continues, at the computed tokens, with an image block whose output
    /// the request lacks: its prefill waits for the block's encoding.
    fn awaits_encoding(&self) -> bool {
        self.image_spans()
            .enumerate()
            .any(|(index, (start, rows, _, _))| {
                start == self.num_computed_tokens && rows > 0 && !self.holds_encoding(index)
            })
    }

    /// Whether the next encoder input whose output the request lacks is of `kind` and may be
    /// encoded now (`S/batching/next.rs:542-568`): a latent block once the context reached it,
    /// since a prefill of its own writes it; a vision block while its image lies within the
    /// widest chunk of the prefill's group from the text computed, so that a chunk can take it
    /// whole. An unwritten latent block ends the search: the blocks after it are encoded once it
    /// is written.
    fn encodes_next(&self, kind: ImageEncoder, groups: &[WorkerGroup]) -> bool {
        let computed = self.num_computed_tokens;
        let text = self.text_before(computed);
        let reach = self
            .placed(Phase::Prefill, groups)
            .map_or(u32::MAX, widest_prefill);

        for (index, (start, _, image, input)) in self.image_spans().enumerate() {
            if start < computed {
                continue;
            }
            let lacking = !self.holds_encoding(index);
            match input.encoder {
                ImageEncoder::Latent => {
                    return lacking && kind == ImageEncoder::Latent && start == computed;
                }
                ImageEncoder::Vision if lacking => {
                    let near = image.position <= text || image.position - text < reach;
                    return kind == ImageEncoder::Vision && near;
                }
                ImageEncoder::Vision => {}
            }
        }
        false
    }

    /// The text tokens of a token request's context before cache position `position`, which no
    /// block straddles.
    fn text_before(&self, position: u32) -> u32 {
        let rows = self
            .image_spans()
            .filter(|&(start, _, _, _)| start < position)
            .map(|(_, rows, _, _)| rows)
            .fold(0, u32::saturating_add);
        position.saturating_sub(rows)
    }

    /// Whether a decode may join behind the request's calls in flight (SD §7.2 "Pipelined
    /// decodes"): only its prompt's last chunk and its decodes may be ahead of it, fewer than
    /// `window`, none when it bans a multi-token word (whose next ban depends on tokens not seen
    /// yet), and not so many that they could reach its token limit.
    fn pipelines(&self, request: &GenerationRequest, window: u32) -> bool {
        let ahead = self
            .in_flight(Phase::Prefill)
            .saturating_add(self.in_flight(Phase::Decode));
        if ahead == 0 {
            return true;
        }

        let others = Phase::ALL
            .into_iter()
            .filter(|phase| !matches!(phase, Phase::Prefill | Phase::Decode))
            .any(|phase| self.in_flight(phase) > 0);
        let single = request
            .sampling
            .bad_words_ids
            .iter()
            .all(|word| word.len() <= 1);
        !others && single && ahead < window && !self.reaches_limit()
    }

    /// Whether the token calls of the request in flight may sample the last token its limit
    /// allows (`S/execution/projection.rs:290-352`): its generated tokens and one token for each
    /// prefill, decode and verification in flight reach `max_und_tokens`. Its next token call,
    /// if any, waits for their outputs. False for a video request.
    pub(crate) fn reaches_limit(&self) -> bool {
        let Some(request) = self.generation() else {
            return false;
        };
        let ahead = [Phase::Prefill, Phase::Decode, Phase::Verify]
            .into_iter()
            .map(|phase| u64::from(self.in_flight(phase)))
            .sum::<u64>();
        let limit = u64::try_from(request.max_und_tokens).unwrap_or(u64::MAX);
        u64::from(self.generated()) + ahead >= limit
    }
}

/// A request's checks at its arrival (SD §3.2, §7.2).
impl Scheduler {
    /// Checks a request against the deployment before it is queued (SD §3.2); the message
    /// rejects it as `Invalid`. A video request is `validate_video`'s. A token request is refused,
    /// with the message of the first rule it breaks (`S/admission.rs:200-291`), when:
    ///
    /// - no group holds a KV cache;
    /// - it fails its own validation (`GenerationRequest::validate`);
    /// - an encoder input of its images states no exact KV length: the prefill that writes the
    ///   block states its rows;
    /// - its prompt and its vision blocks, which one prefill writes whole, exceed the step's
    ///   token budget;
    /// - its largest vision block's features exceed what one encoding holds;
    /// - the groups do not serve a feature it needs;
    /// - it needs more than the deployment's generation limits allow
    ///   (`GenerationRequest::validate_resources`).
    pub(super) fn validate(&self, input: &proto::request::Request) -> Result<(), String> {
        let request = match input {
            proto::request::Request::Generation(request) => request,
            proto::request::Request::Diffusion(video) => return self.validate_video(video),
        };

        if !self
            .groups
            .iter()
            .any(|group| group.capacity.kv_cache.is_some())
        {
            return Err("generation request requires worker KV resources".to_owned());
        }
        if let Err(error) = request.validate() {
            return Err(format!("invalid generation request: {error:?}"));
        }
        let encoders = || {
            request
                .multimodal_inputs
                .images
                .iter()
                .flat_map(|image| &image.encoders)
        };
        if encoders().any(|input| input.num_cache_tokens.is_none()) {
            return Err(
                "an input image's encoder states no exact KV length, which the prefill that \
                 writes its block states"
                    .to_owned(),
            );
        }

        // A prefill writes every vision block it reaches whole together with the prompt around
        // it, so the whole context must fit one step's token budget.
        let vision = || {
            encoders()
                .filter(|input| input.encoder == ImageEncoder::Vision)
                .filter_map(|input| input.num_cache_tokens)
        };
        let blocks: usize = vision().map(|tokens| tokens as usize).sum();
        let context = request.prompt_token_ids.len().saturating_add(blocks);
        let budget = self.config.max_num_batched_tokens;
        if blocks > 0 && context > budget {
            return Err(format!(
                "a {context}-token context, whose {blocks} tokens of input image vision block \
                 are written whole by one context prefill, exceeds the {budget}-token step budget"
            ));
        }

        // A vision block's features are one encoder output, which one encoding holds whole.
        let feature_bytes = self
            .groups
            .iter()
            .filter(|group| group.serves(Phase::VisionEncode))
            .map(|group| u64::from(group.capacity.feature_bytes))
            .max()
            .filter(|&bytes| bytes > 0);
        if let Some(feature_bytes) = feature_bytes {
            let largest = vision().max().unwrap_or(0);
            let bytes = u64::from(largest).saturating_mul(feature_bytes);
            let held = self.limits.max_vision_feature_bytes;
            if bytes > held {
                return Err(format!(
                    "an input image's vision block of {largest} tokens takes {bytes} bytes of \
                     features, beyond the {held} bytes one encoding holds"
                ));
            }
        }

        if let Err(feature) = self.limits.covers(request.required_features()) {
            return Err(format!(
                "generation request requires worker feature `{feature}`, but the worker does \
                 not support it"
            ));
        }
        request
            .validate_resources(&self.limits)
            .map_err(|error| format!("invalid generation resource requirements: {error}"))
    }
}

/// The generation features the groups serve (SD §7.2, the features of the workers,
/// `S/run.rs:22-45`): understanding where prefill and decode are served, and each image encoder
/// whose encoding is.
pub(super) fn served_features(groups: &[WorkerGroup]) -> GenerationFeatures {
    let serves = |phase: Phase| groups.iter().any(|group| group.serves(phase));
    let mut features = GenerationFeatures::empty();
    if serves(Phase::Prefill) && serves(Phase::Decode) {
        features |= GenerationFeatures::UNDERSTANDING;
    }
    if serves(Phase::VisionEncode) {
        features |= GenerationFeatures::VISION_ENCODE;
    }
    if serves(Phase::LatentEncode) {
        features |= GenerationFeatures::LATENT_ENCODE;
    }
    features
}

/// The ids of a token request's context, one per cache position, as the module documentation
/// lays it out: the prompt with each image block's rows (id 0) before the prompt token at its
/// image's position, then the tokens a resumed request generated.
fn context(request: &GenerationRequest) -> Vec<u32> {
    let prompt = &request.prompt_token_ids;
    let blocks: usize = request
        .multimodal_inputs
        .images
        .iter()
        .flat_map(|image| &image.encoders)
        .map(|input| rows(input) as usize)
        .sum();
    let mut ids = Vec::with_capacity(prompt.len() + blocks + request.generated_token_ids.len());

    let mut text = 0;
    for image in &request.multimodal_inputs.images {
        let at = (image.position as usize).clamp(text, prompt.len());
        ids.extend_from_slice(prompt.get(text..at).unwrap_or_default());
        text = at;
        for input in &image.encoders {
            ids.resize(ids.len() + rows(input) as usize, 0);
        }
    }
    ids.extend_from_slice(prompt.get(text..).unwrap_or_default());
    ids.extend_from_slice(&request.generated_token_ids);
    ids
}

/// The cache rows of an encoder input's block: its exact KV length, which admission requires.
fn rows(input: &ImageEncoderInput) -> u32 {
    input.num_cache_tokens.unwrap_or(0)
}

/// The most events one call of `phase` adds to its request's stream (SD §7.1, `S/lib.rs:604-626`):
/// a prefill chunk, a refill or a decode 4, a verification 4 and one for each draft it may carry,
/// a denoising step 3, every other call 2.
const fn output_bound(phase: Phase) -> u32 {
    match phase {
        Phase::Prefill | Phase::Refill | Phase::Decode => 4,
        Phase::Verify => 4 + MAX_VERIFY_DRAFTS,
        Phase::Denoise => 3,
        _ => 2,
    }
}

/// The most tokens a prefill chunk on `group` holds: its widest captured prefill bucket's, which
/// bounds every chunk (SD §5.2 ceilings); unbounded when it captured none.
fn widest_prefill(group: &WorkerGroup) -> u32 {
    group
        .capacity
        .buckets
        .iter()
        .filter(|shape| variants(Phase::Prefill).contains(&shape.variant))
        .map(|shape| shape.tokens)
        .max()
        .unwrap_or(u32::MAX)
}

/// A count as the `u32` the request's counters hold, saturated.
fn len32(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::Instant;

    use crate::batch::{Phase, Shape, Variant};
    use proto::config::{KvCacheConfig, WorkerGroupConfig};
    use proto::request::{
        CacheResume, FinishReason, GenerationFeatures, GenerationLimits, GenerationRequest,
        ImageEncoder, ImageEncoderInput, ImageInput, RequestId, SamplingParams, StopReason,
    };
    use storage::Key;
    use storage::runtime::{self, GroupId};

    use super::super::tests::bench;
    use super::super::{Capacity, WorkerGroup};
    use super::{Blocks, Request, RequestStatus};

    const EOS: u32 = 2;

    /// A token request of `prompt` that may generate `max` tokens.
    fn tokens(prompt: &[u32], max: usize) -> GenerationRequest {
        GenerationRequest {
            request_id: RequestId(7),
            prompt_token_ids: prompt.to_vec(),
            multimodal_inputs: Default::default(),
            sampling: SamplingParams::default(),
            max_und_tokens: max,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            include_stop_token: false,
            generated_token_ids: Vec::new(),
            adapter: 0,
        }
    }

    /// An image at prompt position `position` whose encoder of `kind` writes `rows` rows.
    fn image(position: u32, kind: ImageEncoder, rows: u32) -> ImageInput {
        ImageInput {
            digest: [position as u8; 32],
            b64: "aW1hZ2U=".to_owned(),
            position,
            num_positions: 1,
            encoders: vec![ImageEncoderInput {
                encoder: kind,
                num_cache_tokens: Some(rows),
                max_cache_tokens: None,
                height: None,
                width: None,
            }],
        }
    }

    /// Group `id` serving `phases`, with `window` calls of a request unresolved at once and
    /// prefill buckets of up to `chunk` tokens.
    fn group(id: u8, phases: &[Phase], window: u32, chunk: u32) -> WorkerGroup {
        let configs = WorkerGroupConfig::data_parallel(
            &["localhost".to_owned()],
            "cpu",
            1,
            1,
            2,
            BTreeMap::new(),
        );
        let capacity = Capacity {
            phases: phases.iter().map(|phase| 1 << phase.index()).sum(),
            slots: 16,
            kv_cache: Some(KvCacheConfig::default()),
            buckets: vec![Shape::prefill(4, chunk, Variant::PrefillOutput)],
            rows: [16; Phase::COUNT],
            max_model_len: 4096,
            denoise_steps: 0,
            bos_token_id: 1,
            eos_token_ids: vec![EOS],
            feature_bytes: 0,
            encoder_identity: String::new(),
            latent_seed: 0,
            media_bytes: Vec::new(),
            call_window: window,
            cache_groups: Vec::new(),
            seeds: Vec::new(),
        };
        let layout = runtime::Group {
            id: GroupId(id),
            memory: format!("memory-{id}"),
            slots: 0,
            regions: Vec::new(),
            receives: Vec::new(),
            parts: Vec::new(),
        };
        WorkerGroup::new(configs[0].clone(), capacity, layout)
    }

    /// `request`, admitted with every phase on group 0.
    fn running(request: GenerationRequest) -> Request {
        let mut request = Request::new(request.into(), 3, Instant::now());
        request.status = RequestStatus::Running;
        request.placement = [Some(GroupId(0)); Phase::COUNT];
        request
    }

    /// Schedules `calls` calls of `phase` without completing them; a prefill or decode computes
    /// its tokens when scheduled.
    fn schedule(request: &mut Request, phase: Phase, calls: u32, tokens: u32) {
        request.scheduled[phase.index()] += calls;
        request.num_computed_tokens += tokens;
    }

    /// Commits `generated` output tokens.
    fn generate(request: &mut Request, generated: &[u32]) {
        request.token_ids.extend_from_slice(generated);
        request.num_computed_tokens = request.token_ids.len() as u32;
    }

    /// A resumed request's generated tokens continue its prompt and its generated count; a
    /// request is named for its continuation by the epoch and its id unless it carries a name.
    #[test]
    fn a_resumed_request_continues_its_prompt_and_its_generated_count() {
        let mut input = tokens(&[1, 2, 3], 10);
        input.generated_token_ids = vec![7, 8];
        let request = Request::new(input.into(), 3, Instant::now());

        assert_eq!(request.token_ids, [1, 2, 3, 7, 8]);
        assert_eq!(request.num_prompt_tokens, 5);
        assert_eq!(request.generated(), 2);
        assert!(request.resumes() && request.queued);
        assert_eq!(request.event_credit, 64);
        let resume = request
            .generation()
            .and_then(|request| request.cache.resume);
        assert_eq!(
            resume,
            Some(CacheResume {
                epoch: 3,
                request_id: 7
            })
        );

        let mut named = tokens(&[1], 4);
        let carried = CacheResume {
            epoch: 1,
            request_id: 5,
        };
        named.cache.resume = Some(carried);
        let request = Request::new(named.into(), 3, Instant::now());
        assert!(!request.resumes());
        let resume = request
            .generation()
            .and_then(|request| request.cache.resume);
        assert_eq!(resume, Some(carried));
    }

    /// An image's block takes its rows in the context before the prompt token at its position.
    #[test]
    fn image_blocks_take_their_rows_in_the_context() {
        let mut input = tokens(&[1, 2, 3, 4], 4);
        input.multimodal_inputs.images = vec![
            image(2, ImageEncoder::Vision, 3),
            image(2, ImageEncoder::Latent, 2),
        ];
        let request = Request::new(input.into(), 0, Instant::now());

        assert_eq!(request.token_ids, [1, 2, 0, 0, 0, 0, 0, 3, 4]);
        assert_eq!(request.num_prompt_tokens, 9);
        let spans: Vec<(u32, u32)> = request
            .image_spans()
            .map(|(start, rows, _, _)| (start, rows))
            .collect();
        assert_eq!(spans, [(2, 3), (5, 2)]);
    }

    /// The stop rules in order: a stop token over the `min_tokens` floor, the token limit
    /// whatever the floor, an end-of-sequence token over the floor unless ignored; what of them
    /// is published.
    #[test]
    fn stop_conditions_follow_their_order_and_the_floor() {
        let mut input = tokens(&[1], 4);
        input.stop_token_ids = vec![9];
        input.sampling.min_tokens = 2;
        let mut request = running(input);

        generate(&mut request, &[9]);
        assert_eq!(request.check_stop(9, &[EOS]), None, "under the floor");
        assert_eq!(request.check_stop(EOS, &[EOS]), None, "under the floor");

        generate(&mut request, &[5, 6]);
        let stop = Some((FinishReason::Stop, Some(StopReason::Token(9))));
        assert_eq!(request.check_stop(9, &[EOS]), stop);
        assert_eq!(
            request.check_stop(EOS, &[EOS]),
            Some((FinishReason::Eos, None))
        );
        assert_eq!(request.check_stop(5, &[EOS]), None);

        generate(&mut request, &[5]);
        assert_eq!(
            request.check_stop(9, &[EOS]),
            stop,
            "a stop token comes first"
        );
        let limit = Some((FinishReason::MaxTokens, None));
        assert_eq!(request.check_stop(5, &[EOS]), limit);
        assert_eq!(request.check_stop(EOS, &[EOS]), limit);

        assert!(request.publishes(5, None, &[EOS]));
        assert!(!request.publishes(9, stop.as_ref(), &[EOS]));
        assert!(request.publishes(5, limit.as_ref(), &[EOS]));
        assert!(!request.publishes(EOS, limit.as_ref(), &[EOS]));
        assert!(!request.publishes(EOS, Some(&(FinishReason::Eos, None)), &[EOS]));

        let mut input = tokens(&[1], 4);
        input.stop_token_ids = vec![9];
        input.include_stop_token = true;
        input.sampling.ignore_eos = true;
        let mut request = running(input);
        generate(&mut request, &[9]);
        assert!(request.publishes(9, stop.as_ref(), &[EOS]));
        assert_eq!(request.check_stop(EOS, &[EOS]), None, "ignored");
    }

    /// The credit covers the bound of every call in flight, the terminal's reserve and the next
    /// call's bound: 4 for a decode, 4 and one per draft for a verification.
    #[test]
    fn the_window_reserves_the_bounds_of_calls_in_flight_and_the_terminal() {
        let mut request = running(tokens(&[1], 8));
        request.event_credit = 13;
        assert!(request.has_event_room(Phase::Decode));
        assert!(!request.has_event_room(Phase::Verify), "2 + 4 + 16 > 13");

        schedule(&mut request, Phase::Decode, 1, 1);
        assert!(request.has_event_room(Phase::Decode), "4 + 2 + 4 <= 13");
        schedule(&mut request, Phase::Decode, 1, 1);
        assert!(!request.has_event_room(Phase::Decode), "8 + 2 + 4 > 13");

        request.completed[Phase::Decode.index()] = 2;
        request.event_credit = 22;
        assert!(request.has_event_room(Phase::Verify));
    }

    /// A decode pipelines behind its prompt's last chunk and its decodes within the group's call
    /// window, and not with a multi-token bad word, near its token limit, behind another call, or
    /// under stop strings with a window of unacknowledged outputs.
    #[test]
    fn a_decode_pipelines_within_the_call_window() {
        let groups = [group(0, &[Phase::Prefill, Phase::Decode], 2, 64)];
        let mut input = tokens(&[1, 2, 3], 8);
        input.stop_strings = vec!["end".to_owned()];
        let mut request = running(input);
        assert!(request.ready(Phase::Prefill, &groups));
        assert!(
            !request.ready(Phase::Decode, &groups),
            "the prompt is not computed"
        );

        schedule(&mut request, Phase::Prefill, 1, 3);
        assert!(request.ready(Phase::Decode, &groups));
        assert!(!request.ready(Phase::Prefill, &groups));
        schedule(&mut request, Phase::Decode, 1, 1);
        assert!(
            !request.ready(Phase::Decode, &groups),
            "two calls fill the window"
        );
        request.completed[Phase::Prefill.index()] = 1;
        assert!(request.ready(Phase::Decode, &groups));

        // Two outputs the daemon's decoder has not passed fill the window of a request with stop
        // strings.
        request.completed[Phase::Decode.index()] = 1;
        generate(&mut request, &[5, 6]);
        assert!(!request.ready(Phase::Decode, &groups));
        request.num_acknowledged_tokens = 1;
        assert!(request.ready(Phase::Decode, &groups));
        assert_eq!(request.ready_phases(&groups).next(), Some(Phase::Decode));

        // The calls in flight would reach the limit of 8 with the 2 generated tokens.
        schedule(&mut request, Phase::Decode, 5, 5);
        request.num_acknowledged_tokens = 2;
        assert!(!request.ready(Phase::Decode, &groups));

        let mut input = tokens(&[1], 8);
        input.sampling.bad_words_ids = vec![vec![4, 5]];
        let mut request = running(input);
        schedule(&mut request, Phase::Prefill, 1, 1);
        assert!(!request.ready(Phase::Decode, &groups), "a multi-token word");
        request.completed[Phase::Prefill.index()] = 1;
        assert!(request.ready(Phase::Decode, &groups));
        request.status = RequestStatus::WaitingForRemoteKvs;
        assert!(!request.ready(Phase::Decode, &groups));
    }

    /// A speculating deployment verifies instead of decoding, never behind another call, and
    /// refills a draft cache that does not cover the computed tokens; a refill is never the
    /// request's named phase.
    #[test]
    fn a_speculating_request_verifies_alone_and_refills_its_drafts() {
        let phases = [Phase::Prefill, Phase::Decode, Phase::Verify, Phase::Refill];
        let groups = [group(0, &phases, 2, 64)];
        let mut request = running(tokens(&[1, 2, 3], 8));
        generate(&mut request, &[]);

        assert!(!request.ready(Phase::Decode, &groups));
        assert!(request.ready(Phase::Verify, &groups));
        assert!(
            request.ready(Phase::Refill, &groups),
            "no draft entries yet"
        );
        assert_eq!(
            request.ready_phases(&groups).collect::<Vec<_>>(),
            [Phase::Verify]
        );

        request.num_draft_tokens = 3;
        assert!(!request.ready(Phase::Refill, &groups));
        schedule(&mut request, Phase::Verify, 1, 2);
        assert!(!request.ready(Phase::Verify, &groups));
    }

    /// A verification carries the proposals while the draft cache covers the computed tokens,
    /// one fewer than the tokens left, cut where the banned tokens change.
    #[test]
    fn drafts_stop_where_the_banned_tokens_change() {
        let mut input = tokens(&[1], 8);
        input.sampling.min_tokens = 3;
        input.sampling.bad_words_ids = vec![vec![6, 9]];
        let mut request = running(input);
        generate(&mut request, &[4]);
        request.spec_token_ids = vec![5, 6, 7, 8];

        assert_eq!(request.drafts(&[EOS]), 0, "the draft cache is behind");
        request.num_draft_tokens = request.num_computed_tokens;
        // The floor of 3 bans the end-of-sequence token after 1 + 1 generated, not after 1 + 2.
        assert_eq!(request.banned(&[EOS], 0, &[5]), [EOS]);
        assert_eq!(request.banned(&[EOS], 0, &[5, 6]), [9]);
        assert_eq!(request.drafts(&[EOS]), 1);

        request.spec_token_ids = vec![5, 5, 5, 5];
        generate(&mut request, &[4, 4]);
        request.num_draft_tokens = request.num_computed_tokens;
        assert_eq!(request.drafts(&[EOS]), 4);
        generate(&mut request, &[4, 4, 4]);
        request.num_draft_tokens = request.num_computed_tokens;
        assert_eq!(request.drafts(&[EOS]), 1, "6 generated of 8");
    }

    /// A refill starts where the draft cache ends when the state checkpoint there is kept, from
    /// 0 otherwise, and a chunk that stops short ends on a checkpoint boundary.
    #[test]
    fn a_refill_continues_from_a_kept_checkpoint_in_whole_blocks() {
        let mut request = running(tokens(&[1; 300], 8));
        request.num_computed_tokens = 300;
        request.num_draft_tokens = 128;
        request.blocks = vec![Blocks {
            first: 1,
            shared: 2,
            end: 2,
            asked: 2,
        }];
        let state = Some((0, 64));

        assert_eq!(request.refill_start(state), 128);
        assert_eq!(request.refill_chunk(100, state), 64);
        assert_eq!(request.refill_chunk(1000, state), 172);
        assert_eq!(request.refill_chunk(50, state), 0);
        assert_eq!(request.refill_chunk(100, None), 100);

        request.blocks[0].first = 2;
        assert_eq!(request.refill_start(state), 0);
        assert_eq!(request.refill_chunk(100, state), 64);
    }

    /// A chunk takes a vision block whole when the limit holds it or the block starts it, and
    /// ends before a block it cannot take or whose output the request lacks; a latent block is a
    /// chunk of its own.
    #[test]
    fn a_chunk_takes_vision_blocks_whole_or_ends_before_them() {
        let mut input = tokens(&[1; 10], 4);
        input.multimodal_inputs.images = vec![
            image(4, ImageEncoder::Vision, 3),
            image(8, ImageEncoder::Vision, 4),
            image(9, ImageEncoder::Latent, 2),
        ];
        let mut request = running(input);
        let held = |index: usize| (Key([index as u8; 32]), Some(index as u32));
        request.encodings = vec![held(0), held(1), held(2)];

        // Text 0..4, block 4..7, text 7..11, block 11..15, text 15..16, block 16..18, text 18..19.
        assert_eq!(request.num_prompt_tokens, 19);
        assert_eq!(request.clip_to_images(5), 4);
        assert_eq!(request.clip_to_images(8), 8);
        assert_eq!(request.clip_to_images(12), 11);
        assert_eq!(request.clip_to_images(19), 16, "before the latent block");

        request.num_computed_tokens = 4;
        assert_eq!(
            request.clip_to_images(2),
            3,
            "a block that starts the chunk"
        );
        request.num_computed_tokens = 16;
        assert_eq!(request.clip_to_images(3), 2, "a latent block alone");

        request.encodings[1].1 = None;
        request.num_computed_tokens = 7;
        assert_eq!(request.clip_to_images(10), 4);
    }

    /// An image is encoded before the chunk that reaches it, within the widest chunk of the text
    /// computed, and the prefill waits only for a block at its start; the encodings counted stay
    /// as they complete.
    #[test]
    fn an_image_is_encoded_ahead_of_the_chunk_that_reaches_it() {
        let groups = [group(
            0,
            &[Phase::Prefill, Phase::Decode, Phase::VisionEncode],
            2,
            8,
        )];
        let mut input = tokens(&[1; 20], 4);
        input.multimodal_inputs.images = vec![image(12, ImageEncoder::Vision, 3)];
        let mut request = running(input);

        assert_eq!(request.needed(Phase::VisionEncode, &groups), 1);
        assert!(
            !request.ready(Phase::VisionEncode, &groups),
            "beyond the widest chunk"
        );
        assert!(request.ready(Phase::Prefill, &groups));
        assert_eq!(request.clip_to_images(20), 12);

        schedule(&mut request, Phase::Prefill, 1, 8);
        request.completed[Phase::Prefill.index()] = 1;
        assert!(request.ready(Phase::VisionEncode, &groups));
        assert_eq!(
            request.ready_phases(&groups).next(),
            Some(Phase::VisionEncode)
        );

        schedule(&mut request, Phase::Prefill, 1, 4);
        request.completed[Phase::Prefill.index()] = 2;
        assert!(
            !request.ready(Phase::Prefill, &groups),
            "the block lacks its output"
        );

        schedule(&mut request, Phase::VisionEncode, 1, 0);
        assert!(!request.ready(Phase::VisionEncode, &groups));
        request.completed[Phase::VisionEncode.index()] = 1;
        request.encodings = vec![(Key([0; 32]), Some(0))];
        assert_eq!(request.needed(Phase::VisionEncode, &groups), 1);
        assert!(!request.ready(Phase::VisionEncode, &groups));
        assert!(request.ready(Phase::Prefill, &groups));
    }

    /// A token request is refused, with its message, when no group holds a KV cache, when one
    /// prefill cannot write its vision blocks within a step, when the groups do not serve what it
    /// needs, and beyond the generation limits.
    #[test]
    fn a_token_request_is_validated_against_the_deployment() {
        let mut bench = bench(1024);
        let scheduler = &mut bench.scheduler;
        let mut input = tokens(&[1; 6], 4);
        input.multimodal_inputs.images = vec![image(2, ImageEncoder::Vision, 4)];
        let request = proto::request::Request::from(input.clone());

        assert_eq!(
            scheduler.validate(&request),
            Err("generation request requires worker KV resources".to_owned())
        );

        scheduler.groups[0].capacity.kv_cache = Some(KvCacheConfig::default());
        scheduler.config.max_num_batched_tokens = 8;
        assert_eq!(
            scheduler.validate(&request),
            Err(
                "a 10-token context, whose 4 tokens of input image vision block are written \
                 whole by one context prefill, exceeds the 8-token step budget"
                    .to_owned()
            )
        );

        scheduler.config.max_num_batched_tokens = 10;
        assert_eq!(
            scheduler.validate(&request),
            Err(
                "generation request requires worker feature `runtime_und_execution`, but the \
                 worker does not support it"
                    .to_owned()
            )
        );

        scheduler.limits = GenerationLimits {
            features: GenerationFeatures::all(),
            max_vit_grid_tokens: 64,
            max_vision_feature_bytes: 1 << 20,
            encoder_cache_entries: 4,
            ..GenerationLimits::default()
        };
        assert_eq!(scheduler.validate(&request), Ok(()));

        input.sampling.return_logprobs = true;
        input.sampling.n_logprobs = 5;
        assert_eq!(
            scheduler.validate(&input.into()),
            Err(
                "invalid generation resource requirements: requested logprobs of 5, which is \
                 greater than max allowed: 0"
                    .to_owned()
            )
        );
    }
}
