//! A request's tables (`specs/scheduler-design.md` §4.2 to §4.4): the decision half of vLLM's
//! cache managers (`vllm/v1/core/single_type_kv_cache_manager.py` per cache group,
//! `kv_cache_coordinator.py` for the hybrid hit). The buffer pool allocates, pins, copies and
//! evicts; this file decides what a request asks of it, at which positions, and when, and
//! returns those requests for the scheduler to submit.
//!
//! # A request's classes
//!
//! A token request's first classes are the model's cache groups, class `c` being cache group `c`
//! with its kind and block size (`Capacity::cache_groups`), then the encoder-output class when
//! the session states one before the product classes. A video request holds one block per
//! declared result in the product classes and its slot's trajectory. Every request for a class
//! names the request's owner on the group that writes it (SD §4.2).
//!
//! A class's table is counted by its [`Blocks`]: read-only entries in `[first, shared)`, private
//! ones in `[shared, end)`, positions asked up to `asked`. A position in a cache group's table
//! is the logical index of its block. Two tables are not dense:
//!
//! - **A recurrent-state table** holds at most two checkpoints: its base at `first`, from which
//!   a continuation or a refill restores, and the one at `end - 1` that the next call writes
//!   ([`checkpoints`]). The checkpoint ending at boundary `b` lies at `b / block - 1` (`rows.rs`),
//!   and one is asked for at a time, beyond the table's end.
//! - **The encoder-output table** holds encoder input `i`'s output at position `i`; a product
//!   table holds declared result `i` at position `i` of its kind's class. A video request's
//!   conditioning is its one encoder output (entry 0 of `Request::encodings`), held at its
//!   declared result's position: pinned there where cached, allocated otherwise.
//!
//! # Admission rounds
//!
//! An admission's first round ([`Kv::admission_requests`]) asks for the hit `cache` expects: per
//! cache class a `Pin` of the leading blocks on the device, a `Load` of the blocks after them on
//! the host alone, a `Clone` of a partial last block, and the `Allocate`s of the first chunk and
//! the low watermark, which wait for a later round when the round asks a `Prefetch` or a
//! `Clone`. Each answer is applied by [`Kv::filled`]: the read-only prefix of a class grows only
//! where it continues, so `shared` is always the end of what the request holds read-only from
//! `first`, and an encoder output that was not found is allocated at once. When every answer
//! arrived and the entries do not hold the first call ([`settled`]), one more round
//! ([`Kv::missing_requests`]) either asks again after a prefetch, or makes the hit final: it
//! gives up the read-only entries beyond the deepest boundary the answers support and allocates
//! what no entry holds. The expected hit is `num_cached_tokens` until the hit is final.
//!
//! The rules that read the request and its groups alone are free functions; those that read
//! what the scheduler knows of the cache, the session's classes or the key seeds are [`Kv`]'s.

use std::collections::HashSet;
use std::ops::Range;
use std::time::Instant;

use crate::batch::cache::CacheGroupKind;
use crate::batch::{Batch, Phase};
use proto::config::{KvCacheConfig, SchedulerConfig};
use proto::request::{ImageEncoder, RequestId};
use scheduling::Event;
use storage::Key;
use storage::key::{Digest, encoder_output, page_key, page_root};
use storage::runtime::{self, Completion, GroupId, OwnerId, Tier};

use super::super::rows::state_interval;
use super::super::video::products;
use super::super::{Blocks, Budget, Request, RequestStatus, WorkerGroup, monotonic_nanos, span};
use super::cache::{Cached, DEVICE, HOST, LOW_WATERMARK, REMOTE};
use super::{Kv, Step};

/// The checkpoints a recurrent-state table holds: its base at `first`, and the one at `end - 1`
/// when that is another.
fn checkpoints(blocks: &Blocks) -> [Option<u32>; 2] {
    if blocks.first >= blocks.end {
        return [None, None];
    }
    let ahead = blocks.end - 1;
    [Some(blocks.first), (ahead > blocks.first).then_some(ahead)]
}

/// The blocks of a class a reader continuing from token `boundary` needs (SD §4.4 "The hit"):
/// every block before it for full attention and the draft cache, the blocks covering the window
/// before it for a sliding window, and for a recurrent state the checkpoint that ends there.
fn needed(kind: CacheGroupKind, block: u32, boundary: u32) -> Range<u32> {
    let end = boundary / block;
    match kind {
        CacheGroupKind::State => end.saturating_sub(1)..end,
        CacheGroupKind::SlidingWindow { window } => {
            (boundary.saturating_sub(window) / block).min(end)..end
        }
        CacheGroupKind::Full | CacheGroupKind::Draft => 0..end,
    }
}

/// Whether a hit of `request` may read host copies: never for a request that continues another
/// epoch's state under the seeds it states (SD §4.4 "The hit").
fn reads_host(request: &Request) -> bool {
    request
        .generation()
        .is_some_and(|input| input.cache.state_seeds.is_empty())
}

/// Over `range` of a class's expected tiers, the blocks on the device in the owner's group and
/// then those on the host alone, when the range is those two runs in that order: a `Pin` and a
/// `Load` after it hold it. `None` when a block is neither, or lies beyond the tiers known.
fn usable(tiers: &[u8], range: Range<u32>, host: bool) -> Option<(u32, u32)> {
    let slice = tiers.get(range.start as usize..range.end as usize)?;
    let device = slice.iter().take_while(|tier| **tier & DEVICE != 0).count();
    let loaded = slice[device..]
        .iter()
        .take_while(|tier| host && **tier & (DEVICE | HOST) == HOST)
        .count();
    (device + loaded == slice.len()).then_some((count(device), count(loaded)))
}

/// The block boundary a call of `phase` over tokens `start..end` keeps the recurrent state at,
/// in blocks of `block` tokens (SD §3.4, §7.4): a prompt or refill chunk the last it reaches, a
/// decode or a verification only a multiple of the state interval. The checkpoint of boundary
/// `b` lies at position `b / block - 1` of the state table, which the row's `checkpoints` and
/// `state_copies` name (`rows.rs`).
pub(crate) fn kept(phase: Phase, start: u32, end: u32, block: u32, interval: u32) -> Option<u32> {
    let boundary = end / block * block;
    let keeps =
        matches!(phase, Phase::Prefill | Phase::Refill) || boundary.is_multiple_of(interval);
    (boundary > start && keeps).then_some(boundary)
}

/// The position of the recurrent-state checkpoint the first row on a slot restores when it
/// continues a cached prefix at token `start` (SD §3.4): the block that ends there; `None` when
/// `start` ends no block, where no checkpoint holds the state.
pub(crate) fn restored(start: u32, block: u32) -> Option<u32> {
    start
        .is_multiple_of(block)
        .then(|| (start / block).checked_sub(1))
        .flatten()
}

/// The counts of a table holding read-only entries `first..shared` and private ones up to `end`,
/// with nothing more asked.
fn run(first: u32, shared: u32, end: u32) -> Blocks {
    Blocks {
        first,
        shared,
        end,
        asked: end,
    }
}

/// Records positions `at..at + count` as asked in `blocks`. A table that holds and asks for
/// nothing starts again at the first position asked (SD §4.2).
fn ask(blocks: &mut Blocks, at: u32, count: u32) {
    if blocks.first == blocks.asked {
        *blocks = run(at, at, at);
    }
    blocks.asked = blocks.asked.max(at.saturating_add(count));
}

/// The largest block size of the classes that bound a hit: all but the draft cache's.
fn step(shapes: &[(CacheGroupKind, u32)]) -> u32 {
    let sizes = shapes.iter().filter(|(kind, _)| !kind.is_draft());
    sizes.map(|&(_, block)| block).max().unwrap_or(1).max(1)
}

/// Whether block `at` of a sliding window of `window` tokens in blocks of `block` lies in the
/// window before an aligned boundary (SD §4.4), the model's cache groups being `shapes`: a
/// multiple of `boundary_interval` blocks of the largest block size, and, with the request's
/// `prompt`, the last boundary a reissue of the prompt can reuse and the last boundary of the
/// prompt.
fn before_boundary(
    shapes: &[(CacheGroupKind, u32)],
    kv: &KvCacheConfig,
    prompt: Option<u32>,
    at: u32,
    block: u32,
    window: u32,
) -> bool {
    let step = step(shapes);
    let interval = kv.boundary_interval.get();
    let aligned = interval.saturating_mul(step).max(1);
    let after = (at + 1).saturating_mul(block);
    let reissued = prompt.map(|prompt| prompt.saturating_sub(1) / step * step);
    let published = prompt.map(|prompt| prompt / step * step);
    [
        Some(after.div_ceil(aligned).saturating_mul(aligned)),
        reissued,
        published,
    ]
    .into_iter()
    .flatten()
    .any(|boundary| boundary >= after && boundary.saturating_sub(window) / block <= at)
}

/// Whether a request's blocks have keys (SD §4.3): a token request whose context holds no image,
/// whose rows have no token ids a key could cover.
fn keyed(request: &Request) -> bool {
    request
        .generation()
        .is_some_and(|input| input.multimodal_inputs.images.is_empty())
}

fn count(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// An `Allocate` of `count` blocks of `class` at `at`; `units` sizes a block of a sized class.
fn allocate(owner: OwnerId, class: u8, at: u32, count: u32, units: u32) -> runtime::Request {
    runtime::Request::Allocate {
        owner,
        class,
        at,
        count,
        units,
    }
}

/// A `Release` of positions `at..at + count` of `class`, before its stamps.
fn release(owner: OwnerId, class: u8, at: u32, count: u32) -> runtime::Request {
    let after = Vec::new();
    runtime::Request::Release {
        owner,
        class,
        at,
        count,
        after,
    }
}

/// A `Commit` of the block at `at` of `class` under `key`, holding `tokens` tokens.
pub(crate) fn commit(
    owner: OwnerId,
    class: u8,
    at: u32,
    key: Key,
    tokens: u16,
) -> runtime::Request {
    let after = Vec::new();
    runtime::Request::Commit {
        owner,
        class,
        at,
        key,
        tokens,
        after,
    }
}

/// Where a request's cache lies (SD §4.2).
///
/// The owner of a token request's cache classes: its owner on the group that computes its
/// prompt, which decodes it too.
pub(crate) fn kv_owner(request: &Request) -> Option<OwnerId> {
    request.owner_on(Phase::Prefill)
}

/// The group whose memory holds a token request's cache: its cache owner's; before it is
/// attached, the group placed for its prefill, or the first of `groups` that prefills with a
/// cache.
pub(crate) fn kv_group(request: &Request, groups: &[WorkerGroup]) -> Option<GroupId> {
    let attached = kv_owner(request).and_then(|owner| request.attachment(owner));
    attached
        .map(|(group, _)| group)
        .or(request.placement[Phase::Prefill.index()])
        .or_else(|| {
            let serving = groups.iter().find(|group| {
                group.serves(Phase::Prefill) && !group.capacity.cache_groups.is_empty()
            });
            serving.map(WorkerGroup::id)
        })
}

/// The group of `groups` the session numbers `id`.
fn group(groups: &[WorkerGroup], id: GroupId) -> Option<&WorkerGroup> {
    groups.iter().find(|group| group.id() == id)
}

/// The group whose memory holds a token request's cache, as [`kv_group`] finds it.
fn kv_worker<'a>(request: &Request, groups: &'a [WorkerGroup]) -> Option<&'a WorkerGroup> {
    kv_group(request, groups).and_then(|id| group(groups, id))
}

/// The model's cache groups as a token request's first classes, as its cache group states
/// them: per class, its kind and block size in tokens.
pub(crate) fn shapes<'a>(
    request: &Request,
    groups: &'a [WorkerGroup],
) -> &'a [(CacheGroupKind, u32)] {
    kv_worker(request, groups).map_or(&[], |group| &group.capacity.cache_groups)
}

/// The longest sequence of `request`'s cache group, in tokens: the model's length its ranks laid
/// their page tables out for (`Capacity::max_model_len`); unbounded for a request placed nowhere.
fn longest(request: &Request, groups: &[WorkerGroup]) -> u32 {
    kv_worker(request, groups).map_or(u32::MAX, |group| group.capacity.max_model_len)
}

/// The positions of an attention class's table of `block`-token blocks that a call writing the
/// tokens before `end` needs held: the blocks it writes and the low watermark after them (SD
/// §4.4), within the blocks a sequence of `longest` tokens holds. Every rank's pooled page table
/// holds that many positions per request slot (`crate::memory::cache::tables`), so a block past
/// them is one no call can write, and a request in its last block keeps no watermark.
pub(crate) fn reach(end: u32, block: u32, longest: u32) -> u32 {
    let written = end.div_ceil(block);
    let most = longest.div_ceil(block).max(written);

    written.saturating_add(LOW_WATERMARK).min(most)
}

fn kv_config<'a>(request: &Request, groups: &'a [WorkerGroup]) -> Option<&'a KvCacheConfig> {
    kv_worker(request, groups)?.capacity.kv_cache.as_ref()
}

/// The recurrent-state class of a request's model: its index, block size and interval.
pub(crate) fn state_shape(request: &Request, groups: &[WorkerGroup]) -> Option<(usize, u32, u32)> {
    let group = kv_worker(request, groups)?;
    let shapes = &group.capacity.cache_groups;
    let class = shapes.iter().position(|(kind, _)| kind.is_state())?;
    Some((class, shapes[class].1, state_interval(&group.capacity)?))
}

/// Whether the step caches prefixes for `request`'s group: its cache was laid out for them and
/// the step's values keep them on (`kv_cache_config.prefix_caching`).
fn caches_prefixes(request: &Request, groups: &[WorkerGroup], budget: &Budget) -> bool {
    budget.prefix_caching && kv_config(request, groups).is_some_and(|kv| kv.prefix_caching)
}

/// Whether an admission reads its cached prefix (SD §4.3 "Reading"): the deployment caches
/// prefixes and the request reads them, or it resumes, whose own chain it reads whatever the
/// switches say; never with context images.
fn reads(request: &Request, groups: &[WorkerGroup], budget: &Budget) -> bool {
    let Some(input) = request.generation() else {
        return false;
    };
    let caching = caches_prefixes(request, groups, budget);
    keyed(request) && (request.resumes() || (caching && input.cache.read))
}

/// Whether a request commits the blocks it fills (SD §4.3 "Writing"): the deployment caches
/// prefixes and the request writes them, it has no context image, and it does not continue
/// another epoch's state (`state_seeds`), whose blocks are neither epoch's.
fn writes(request: &Request, groups: &[WorkerGroup], budget: &Budget) -> bool {
    let Some(input) = request.generation() else {
        return false;
    };
    let caching = caches_prefixes(request, groups, budget);
    let cache = &input.cache;
    keyed(request) && caching && cache.write && cache.state_seeds.is_empty()
}

/// How a request is keyed (SD §4.3).
impl Kv {
    /// The class of encoder outputs: the one after the cache groups, when the session states one
    /// there before the product classes (one per product of the pipeline, none for a text
    /// deployment).
    fn encoder_class(&self, request: &Request, groups: &[WorkerGroup]) -> Option<u8> {
        let class = shapes(request, groups).len();
        let products = usize::from(self.trajectory).checked_sub(products(groups).len())?;
        (class < products)
            .then(|| u8::try_from(class).ok())
            .flatten()
    }

    /// The key every chain of class `class` starts from (SD §4.3): the root of the request's
    /// seed for the class, its `state_seeds` override or the scheduler's, under its isolation
    /// key.
    fn root(&self, request: &Request, class: u32) -> Key {
        let cache = request.generation().map(|input| &input.cache);
        let stated = cache.and_then(|cache| cache.state_seeds.get(class as usize));
        let seed = stated
            .map(|seed| Key(*seed))
            .or_else(|| self.seeds.get(class as usize).copied())
            .unwrap_or(Key([0; 32]));
        page_root(&seed, cache.and_then(|cache| cache.isolation_key))
    }

    /// Per cache class, the key of every complete block of a token request's `token_ids`
    /// (SD §4.3): `Request::keys`. A draft block's key covers the token after it. A request
    /// without keys has one empty chain per class.
    pub(crate) fn keys_of(&self, request: &Request, groups: &[WorkerGroup]) -> Vec<Vec<Key>> {
        let mut keys = vec![Vec::new(); shapes(request, groups).len()];
        self.chain(request, &mut keys, groups);
        keys
    }

    /// Extends each chain of `keys` over the complete blocks of the request's tokens it does not
    /// cover yet.
    fn chain(&self, request: &Request, keys: &mut [Vec<Key>], groups: &[WorkerGroup]) {
        if !keyed(request) {
            return;
        }
        let tokens = &request.token_ids;
        let shapes = shapes(request, groups).iter().zip(keys.iter_mut());
        for (class, (&(kind, block), chain)) in (0u32..).zip(shapes) {
            let (span, block) = (kind.key_tokens(block) as usize, block as usize);
            let complete = (tokens.len() + block).saturating_sub(span) / block.max(1);
            let mut parent = match chain.last() {
                Some(key) => *key,
                None => self.root(request, class),
            };
            for index in chain.len()..complete {
                let Some(covered) = tokens.get(index * block..index * block + span) else {
                    break;
                };
                parent = page_key(&parent, class, 0, covered);
                chain.push(parent);
            }
        }
    }

    /// Extends `Request::keys` of `request` over the tokens it gained.
    fn extend_keys(&self, request: &mut Request, groups: &[WorkerGroup]) {
        let mut keys = std::mem::take(&mut request.keys);
        keys.resize(shapes(request, groups).len(), Vec::new());
        self.chain(request, &mut keys, groups);
        request.keys = keys;
    }

    /// The key of a request's partial last block of class `class` when its tokens end at `end`
    /// (SD §4.3 "A tail"): the key of `token_ids[full * block..end]` after its complete blocks.
    fn tail_key(
        &self,
        request: &Request,
        (class, end): (usize, u32),
        groups: &[WorkerGroup],
    ) -> Option<Key> {
        let &(_, block) = shapes(request, groups).get(class)?;
        let full = end / block;
        let parent = match full.checked_sub(1) {
            Some(last) => *request.keys.get(class)?.get(last as usize)?,
            None => self.root(request, u32::try_from(class).ok()?),
        };
        let tokens = request
            .token_ids
            .get((full * block) as usize..end as usize)?;
        Some(page_key(&parent, u32::try_from(class).ok()?, 0, tokens))
    }

    /// The key of each encoder output of a request, with no position held yet (SD §4.3),
    /// `encoder_output(name, identity, inputs)`, `identity` being that of the weights that compute
    /// the scheduler's encoder outputs (`Kv::encoder_identity`). A token request's outputs are its
    /// images', in context order, each over the image's digest and, for an encoder that samples
    /// its posterior, the input's draw among the request's latent encodings under a seed of 0
    /// (`S/encodings.rs:35-72`); no group states an encoder that samples yet, so none has a draw.
    /// A video request's one output is its conditioning, over the digest of what its text encoder
    /// reads (`EncoderInputs::text`: its prompt and the conditions its vision blocks present, which
    /// the daemon computes from the request's facts); a request that names no digests has none, and
    /// its conditioning is neither found nor stored (`S/encoder_outputs.rs:14-21`).
    pub(crate) fn encodings_of(&self, request: &Request) -> Vec<(Key, Option<u32>)> {
        let input = match &request.input {
            proto::request::Request::Generation(input) => input,
            proto::request::Request::Diffusion(video) => {
                let text = video.encoder_inputs.as_ref().map(|inputs| inputs.text);
                let identity = &self.encoder_identity;
                let name = nn::pipeline::Product::Conditioning.name();
                let key = |text: [u8; 32]| Key(encoder_output(name, identity, &text));
                return text.into_iter().map(|text| (key(text), None)).collect();
            }
        };
        let samples = false;
        let mut draws = 0u64;
        let mut keys = Vec::new();
        for image in &input.multimodal_inputs.images {
            for encoder in &image.encoders {
                let name = encoder.encoder.output();
                let mut digest = Digest::new(b"image-encoding");
                digest.bytes(&image.digest);
                if samples && encoder.encoder == ImageEncoder::Latent {
                    digest.number(1);
                    digest.number(0);
                    digest.number(draws);
                } else {
                    digest.number(0);
                }
                draws += u64::from(encoder.encoder == ImageEncoder::Latent);
                let inputs = digest.finish();
                let identity = &self.encoder_identity;
                keys.push((Key(encoder_output(name, identity, &inputs)), None));
            }
        }
        keys
    }

    /// Charges `units`, per region of the session, with `count` blocks of `class` of `size`
    /// units each (the class's block size, or `size` for a sized class) in each of its regions on
    /// `group`.
    fn charge_units(
        &self,
        units: &mut [u32],
        (group_id, class): (GroupId, u8),
        (count, size): (u32, u32),
        groups: &[WorkerGroup],
    ) {
        let Some(group) = group(groups, group_id) else {
            return;
        };
        let each = self.block_units(class, size).unwrap_or(0);
        for &region in group.regions(class) {
            if let Some(total) = units.get_mut(usize::from(region)) {
                *total = total.saturating_add(count.saturating_mul(each));
            }
        }
    }

    /// The units of an encoder output of `rows` rows computed by `group` in the encoder class's
    /// region there: its rows' features, rounded up to whole parts of a unit on one rank (each
    /// rank of the group holds the output whole), at least one.
    fn encoding_units(
        &self,
        (group_id, class): (GroupId, u8),
        rows: u32,
        groups: &[WorkerGroup],
    ) -> u32 {
        let Some(group) = group(groups, group_id) else {
            return 1;
        };
        let bytes = u64::from(rows) * u64::from(group.capacity.feature_bytes);
        let region = group.regions(class).first().copied();
        let unit = region
            .and_then(|region| self.regions.get(usize::from(region)))
            .map_or(1, |(_, stated)| stated.part_bytes.max(1));
        u32::try_from(bytes.div_ceil(unit).max(1)).unwrap_or(u32::MAX)
    }

    /// The units of the encoder output `request` holds at position `at` of `class`, as its owner
    /// on `group` allocates it (SD §4.4 "Classes"): a video request's conditioning its declared
    /// result's size, a chat image's output its rows of features ([`Kv::encoding_units`]).
    /// `None` for any other block, and for a conditioning whose result cannot be sized.
    pub(crate) fn encoding_size(
        &self,
        request: &Request,
        (class, at): (u8, u32),
        group: GroupId,
        groups: &[WorkerGroup],
    ) -> Option<u32> {
        let (_, rows) = self.encoding_at(request, class, at, groups)?;
        if request.is_video() {
            let results = self.declared_results(request, groups)?;
            return results.get(at as usize).map(|&(_, _, units)| units);
        }
        Some(self.encoding_units((group, class), rows, groups))
    }

    /// Which of a request's encoder outputs position `at` of `class` holds, as an index into
    /// `Request::encodings`, and its rows (SD §4.2, §4.3): for a token request, encoder input
    /// `at`'s output in the encoder-output class, of its block's rows; for a video request, its
    /// conditioning at its position in the conditioning's product class, of its prompt's rows.
    /// `None` for any other block.
    fn encoding_at(
        &self,
        request: &Request,
        class: u8,
        at: u32,
        groups: &[WorkerGroup],
    ) -> Option<(usize, u32)> {
        if let Some(video) = request.video() {
            let conditioning = Some(class) == self.conditioning_class(groups)
                && Some(at) == request.conditioning_position(groups);
            return conditioning.then(|| (0, count(video.prompt_token_ids.len())));
        }
        if Some(class) != self.encoder_class(request, groups) {
            return None;
        }
        let rows = request.image_spans().nth(at as usize);
        Some((at as usize, rows.map_or(0, |(_, rows, ..)| rows)))
    }
}

/// The blocks a call needs (SD §4.4 "The blocks a call needs", "The low watermark").
///
/// The tokens the request's full-attention and sliding-window classes hold blocks for, the
/// least of them: where a call of its target model may write up to (SD §2.2).
pub(crate) fn allocated(request: &Request, groups: &[WorkerGroup]) -> u32 {
    let classes = shapes(request, groups).iter().zip(&request.blocks);
    classes
        .filter(|((kind, _), _)| kind.required_rows())
        .map(|(&(_, block), blocks)| blocks.end.saturating_mul(block))
        .min()
        .unwrap_or(u32::MAX)
}

/// The tokens the request's draft class holds blocks for (SD §2.2).
pub(crate) fn allocated_draft(request: &Request, groups: &[WorkerGroup]) -> u32 {
    draft_class(request, groups).map_or(0, |(class, block)| {
        let held = request.blocks.get(class).map_or(0, |blocks| blocks.end);
        held.saturating_mul(block)
    })
}

/// The proposals a verification of the request asks its drafter for, which it writes beyond its
/// own tokens: the deployment's draft count while the draft cache covers the request's tokens
/// (`rows.rs`).
fn proposals(request: &Request, config: &SchedulerConfig) -> u32 {
    let complete = request.num_draft_tokens >= request.num_computed_tokens;
    let drafts = u32::try_from(config.draft_tokens).unwrap_or(u32::MAX);
    if complete { drafts } else { 0 }
}

/// Whether the blocks a call of `phase` of `tokens` tokens writes are ones the request holds and
/// may write (SD §3.4, §4.4): its end lies within `allocated` (a verification's proposals
/// included), a refill's within `allocated_draft`; every block it writes lies at or after its
/// class's `shared`; the first row on a slot that continues a cached prefix finds the checkpoint
/// it restores; and a call that keeps a recurrent-state checkpoint holds that one writable. A
/// media call's blocks are held from its admission.
pub(crate) fn holds_blocks_for(
    request: &Request,
    (phase, tokens): (Phase, u32),
    groups: &[WorkerGroup],
    config: &SchedulerConfig,
) -> bool {
    let computed = request.num_computed_tokens;
    let (start, end, draft) = match phase {
        Phase::Prefill | Phase::Decode => (computed, computed.saturating_add(tokens), false),
        Phase::Verify => {
            let end = computed.saturating_add(tokens);
            (
                computed,
                end.saturating_add(proposals(request, config)),
                false,
            )
        }
        Phase::Refill => {
            let start = request.num_draft_tokens;
            (start, start.saturating_add(tokens), true)
        }
        _ => return true,
    };
    let limit = if draft {
        allocated_draft(request, groups)
    } else {
        allocated(request, groups)
    };
    if end > limit {
        return false;
    }

    let classes = shapes(request, groups).iter().zip(&request.blocks);
    let writable = classes
        .filter(|((kind, _), _)| !kind.is_state() && kind.is_draft() == draft)
        .all(|(&(_, block), blocks)| start / block >= blocks.shared);
    writable && holds_state(request, phase, (start, end), groups)
}

/// The recurrent-state part of [`holds_blocks_for`].
fn holds_state(
    request: &Request,
    phase: Phase,
    (start, end): (u32, u32),
    groups: &[WorkerGroup],
) -> bool {
    let Some((class, block, interval)) = state_shape(request, groups) else {
        return true;
    };
    let Some(blocks) = request.blocks.get(class) else {
        return false;
    };
    let held = checkpoints(blocks);

    let group = request.placement[phase.index()];
    let seen = request
        .attachments
        .iter()
        .any(|&(attached, _, _, seen)| Some(attached) == group && seen);
    if !seen
        && start > 0
        && restored(start, block).is_none_or(|position| !held.contains(&Some(position)))
    {
        return false;
    }
    kept(phase, start, end, block, interval).is_none_or(|boundary| {
        let position = boundary / block - 1;
        held.contains(&Some(position)) && position >= blocks.shared
    })
}

/// The tokens of a prefill or refill chunk of at most `tokens` tokens that a model with a
/// recurrent state can run with the checkpoints the request holds: the chunk ends where it keeps
/// the checkpoint the request holds writable, or short of the next block boundary when it holds
/// none ahead. `tokens` for every other call and model. The step applies it with the image rule
/// (`Request::clip_to_images`) after the budget cut a chunk.
pub(crate) fn fit_chunk(
    request: &Request,
    phase: Phase,
    tokens: u32,
    groups: &[WorkerGroup],
) -> u32 {
    let start = match phase {
        Phase::Prefill => request.num_computed_tokens,
        Phase::Refill => request.num_draft_tokens,
        _ => return tokens,
    };
    let Some((class, block, _)) = state_shape(request, groups) else {
        return tokens;
    };
    let end = start.saturating_add(tokens);
    if end / block * block <= start {
        return tokens;
    }

    let writable = request.blocks.get(class).and_then(|blocks| {
        let positions = checkpoints(blocks).into_iter().flatten();
        let ahead = positions.filter(|position| *position >= blocks.shared);
        let boundaries = ahead.map(|position| (position + 1) * block);
        boundaries
            .filter(|boundary| start < *boundary && *boundary <= end)
            .max()
    });
    match writable {
        Some(boundary) => end.min(boundary + block - 1) - start,
        None => ((start / block + 1) * block - 1).saturating_sub(start),
    }
}

/// The first chunk of a prompt computed from token `from` (SD §7.1): the prompt left, at most the
/// step's chunk and token budget, at least one token.
fn first_chunk(request: &Request, from: u32, budget: &Budget) -> u32 {
    let limit = budget.chunk.min(budget.tokens).max(1);
    request
        .num_prompt_tokens
        .saturating_sub(from)
        .min(limit)
        .max(1)
}

/// Where a running token request's next call starts and ends (SD §4.4): its next prefill chunk,
/// else a verification of its drafts and proposals in a deployment that speculates, else one
/// decoded token. `None` once the prompt is scheduled and its calls in flight may sample the
/// last token its limit allows: it may make no further call (`Request::reaches_limit`).
pub(crate) fn next_call(request: &Request, step: &Step) -> Option<(Phase, u32, u32)> {
    let start = request.num_computed_tokens;
    if start < request.num_prompt_tokens {
        let end = start.saturating_add(first_chunk(request, start, step.budget));
        return Some((Phase::Prefill, start, end));
    }
    if request.reaches_limit() {
        return None;
    }
    if step.config.draft_tokens == 0 {
        return Some((Phase::Decode, start, start.saturating_add(1)));
    }
    let eos =
        kv_worker(request, step.groups).map_or(&[][..], |group| &group.capacity.eos_token_ids[..]);
    let verified = 1 + request.drafts(eos) + proposals(request, step.config);
    Some((Phase::Verify, start, start.saturating_add(verified)))
}

/// The `Allocate`s a running token request needs so that the blocks asked for cover its next
/// call and [`LOW_WATERMARK`] blocks beyond it (SD §3.4 `allocate_ahead`, §4.4), an attention
/// class's within the blocks of the model's length ([`reach`]), per class: the
/// class, the position, the count, and how many of those blocks the next call itself writes (the
/// rest are the low watermark's). A full-attention or sliding-window class is grown from `asked`;
/// a recurrent-state class asks for the checkpoint its next call or refill keeps, or a later
/// decode keeps within the low watermark, one ahead at a time; the draft cache is grown over the
/// target's next call while it covers the request's tokens, or over its next refill chunk (the
/// one after the refill in flight) when the request is placed for refills, and no call needs it.
/// A request with no next call asks for nothing. The caller asks only what the regions'
/// `available` covers, and at most one `Allocate` per class outstanding; the request is short
/// only when the blocks its next call writes are not covered (SD §3.6).
pub(crate) fn ahead(request: &Request, step: &Step) -> Vec<(u8, u32, u32, u32)> {
    let mut asks = Vec::new();
    if request.status != RequestStatus::Running || request.generation().is_none() {
        return asks;
    }
    let Some((phase, start, end)) = next_call(request, step) else {
        return asks;
    };
    let groups = step.groups;
    let longest = longest(request, groups);
    let drafts = step.config.draft_tokens;
    let state = state_shape(request, groups).map(|(class, block, _)| (class, block));
    let budget = step.budget.tokens;
    // A refill is planned only for a request placed for one: no group of a model that keeps a
    // recurrent state serves a refill, and a request split across prefill and decode groups never
    // refills (SD §3.5 `place`). The checkpoint a refill that never runs keeps would hold the
    // state table's place ahead, and the checkpoint the decode's next kept boundary needs would
    // never be asked for.
    let refills = request.placement[Phase::Refill.index()].is_some();
    let refill =
        (drafts > 0 && refills && request.num_draft_tokens < request.refill_end()).then(|| {
            let from = request.refill_start(state);
            (
                from,
                from.saturating_add(request.refill_chunk(budget, state)),
            )
        });
    // A refill's draft blocks are asked for one step before it runs, as a prompt chunk's are:
    // while a refill is in flight, those of the chunk after it, which starts where the one in
    // flight ends, within the draft blocks the table holds.
    let refilled = refill.map(|(_, end)| {
        if request.in_flight(Phase::Refill) == 0 {
            return end;
        }
        let held = allocated_draft(request, groups);
        held.saturating_add(budget)
            .min(request.refill_end())
            .max(end)
    });

    for (index, (&(kind, block), blocks)) in shapes(request, groups)
        .iter()
        .zip(&request.blocks)
        .enumerate()
    {
        let Ok(class) = u8::try_from(index) else {
            continue;
        };
        if kind.is_state() {
            // The checkpoint of the next call is needed when the call keeps it; a refill's is
            // not, since a refill takes only the units the regions have free.
            let target = checkpoint_ahead(request, phase, (start, end), groups);
            let refilled = || {
                let (start, end) = refill?;
                let (position, _) = checkpoint_ahead(request, Phase::Refill, (start, end), groups)?;
                Some((position, false))
            };
            if let Some((position, kept)) = target.or_else(refilled) {
                asks.push((class, position, 1, u32::from(kept)));
            }
            continue;
        }
        let (end, required) = match (kind.is_draft(), refilled) {
            (false, _) => (end, true),
            (true, None) => (end, false),
            (true, Some(end)) => (end, false),
        };
        let written = end.div_ceil(block);
        let wanted = if kind.is_draft() {
            written.saturating_add(LOW_WATERMARK)
        } else {
            reach(end, block, longest)
        };
        if blocks.asked < wanted && (required || drafts > 0) {
            let needed = if required {
                written.saturating_sub(blocks.asked)
            } else {
                0
            };
            asks.push((class, blocks.asked, wanted - blocks.asked, needed));
        }
    }
    asks
}

/// The position of the recurrent-state checkpoint a call of `phase` over `start..end` keeps and
/// that the request does not hold or ask for yet, when it may ask for it now: the table holds at
/// most its base and asks for nothing, and the position lies beyond it; with whether the call
/// itself keeps it. A decode asks for the checkpoint of the next kept boundary once it lies
/// within the low watermark, before a call keeps it.
fn checkpoint_ahead(
    request: &Request,
    phase: Phase,
    (start, end): (u32, u32),
    groups: &[WorkerGroup],
) -> Option<(u32, bool)> {
    let (class, block, interval) = state_shape(request, groups)?;
    let blocks = request.blocks.get(class)?;
    if blocks.asked > blocks.end || checkpoints(blocks)[1].is_some() {
        return None;
    }

    let boundary = match phase {
        Phase::Prefill | Phase::Refill => kept(phase, start, end, block, interval)?,
        _ => {
            // The kept boundaries of decodes are the multiples of both sizes.
            let (mut a, mut b) = (block, interval.max(1));
            while b != 0 {
                (a, b) = (b, a % b);
            }
            let multiple = block / a.max(1) * interval.max(1);
            let next = (start / multiple + 1) * multiple;
            (next <= end.saturating_add(LOW_WATERMARK * block)).then_some(next)?
        }
    };
    let position = boundary / block - 1;
    let beyond = blocks.first == blocks.end || position >= blocks.end;
    let keeps = kept(phase, start, end, block, interval) == Some(boundary);
    beyond.then_some((position, keeps))
}

/// Records that `request` asked for positions `at..at + count` of `class`, as every `Allocate`,
/// `Pin`, `Load` and `Clone` of a scheduler is recorded: `Blocks::asked` rises, and a table that
/// held and asked for nothing starts at `at`.
pub(crate) fn count_asked(request: &mut Request, class: u8, at: u32, count: u32) {
    if let Some(blocks) = request.blocks.get_mut(usize::from(class)) {
        ask(blocks, at, count);
    }
}

/// What the regions hold for running requests (SD §3.8, §4.4).
impl Kv {
    /// Per region (an index into the session's regions), the units the `running` requests still
    /// need to reach their next call and the low watermark (SD §4.4 "The admission reserve"),
    /// which an admission leaves available: every class but the draft cache, which takes only
    /// the units the regions have free. A video request's crossings are the caller's to add.
    pub(crate) fn admission_reserve<'a>(
        &self,
        running: impl IntoIterator<Item = &'a Request>,
        step: &Step,
    ) -> Vec<u32> {
        let mut reserve: Vec<u32> = vec![0; self.regions.len()];
        for request in running {
            let Some(group) = kv_group(request, step.groups) else {
                continue;
            };
            let shapes = shapes(request, step.groups);
            for (class, _, count, _) in ahead(request, step) {
                let drafts = shapes
                    .get(usize::from(class))
                    .is_some_and(|(kind, _)| kind.is_draft());
                if !drafts {
                    self.charge_units(&mut reserve, (group, class), (count, 0), step.groups);
                }
            }
        }
        reserve
    }

    /// Per region (an index), the units the `running` requests may still take before they end,
    /// beyond the blocks they asked for (SD §3.8: a draining scheduler keeps stepping its running
    /// requests until they end). A token request's longest sequence is its context and every
    /// token its limit lets it generate, at most its group's longest sequence; per class but the
    /// draft cache, which takes only free units, it takes the blocks of that sequence and the low
    /// watermark: all of them for full attention, those of its window and its next chunk for a
    /// sliding window, two checkpoints for a recurrent state. A video request holds every block
    /// it takes from its admission on.
    pub(crate) fn units_to_end<'a>(
        &self,
        running: impl IntoIterator<Item = &'a Request>,
        groups: &[WorkerGroup],
        budget: &Budget,
    ) -> Vec<u32> {
        let mut units = vec![0; self.regions.len()];
        for request in running {
            let (Some(input), Some(group_id)) = (request.generation(), kv_group(request, groups))
            else {
                continue;
            };
            let longest =
                group(groups, group_id).map_or(u32::MAX, |group| group.capacity.max_model_len);
            let left = count(input.max_und_tokens).saturating_sub(request.generated());
            let end = count(request.token_ids.len())
                .saturating_add(left)
                .min(longest);
            let chunk = if request.num_computed_tokens < request.num_prompt_tokens {
                first_chunk(request, request.num_computed_tokens, budget)
            } else {
                1
            };

            for (index, (&(kind, block), blocks)) in shapes(request, groups)
                .iter()
                .zip(&request.blocks)
                .enumerate()
            {
                let Ok(class) = u8::try_from(index) else {
                    continue;
                };
                let block = block.max(1);
                let most = match kind {
                    CacheGroupKind::Draft => continue,
                    CacheGroupKind::State => 2,
                    CacheGroupKind::SlidingWindow { window } => end
                        .div_ceil(block)
                        .min((window + chunk).div_ceil(block) + 1),
                    CacheGroupKind::Full => end.div_ceil(block),
                };
                let asked = blocks.asked.saturating_sub(blocks.first);
                // The watermark of an attention class stays within the longest sequence's blocks.
                let ceiling = match kind {
                    CacheGroupKind::State => u32::MAX,
                    _ => longest.div_ceil(block).max(most),
                };
                let need = most
                    .saturating_add(LOW_WATERMARK)
                    .min(ceiling)
                    .saturating_sub(asked);
                self.charge_units(&mut units, (group_id, class), (need, 0), groups);
            }
        }
        units
    }

    /// Per region (an index), the units of the blocks the live token `requests`' tables hold or
    /// asked for there (SD §3.8, "the units its live requests still hold"), each block once: a
    /// read-only entry is the block of its key in the request's group, which the other requests
    /// of the group that read the key (a prefix they share) hold too; every other entry is the
    /// request's own: a private block, a block answered `Duplicate` (`Request::unpinned`), and
    /// an entry of an `Allocate` not answered yet, whose units the buffer pool gives it once it
    /// is answered. The buffer pool's shrink takes none of these units. A video request holds
    /// blocks of sized classes alone, in regions no running request takes units of
    /// ([`Kv::units_to_end`]).
    pub(crate) fn units_asked<'a>(
        &self,
        requests: impl IntoIterator<Item = &'a Request>,
        groups: &[WorkerGroup],
    ) -> Vec<u32> {
        let mut units = vec![0; self.regions.len()];
        // The read-only blocks counted, by group and the number of their key (`Cached::id`).
        let mut read: HashSet<(GroupId, u64)> = HashSet::new();
        for request in requests {
            let (Some(_), Some(group)) = (request.generation(), kv_group(request, groups)) else {
                continue;
            };
            for (index, blocks) in request.blocks.iter().enumerate() {
                let Ok(class) = u8::try_from(index) else {
                    continue;
                };
                let chain = request.keys.get(index).map_or(&[][..], Vec::as_slice);
                let shared = blocks.shared.max(blocks.first).min(blocks.asked);
                let mut held = blocks.asked.saturating_sub(shared);
                for position in blocks.first..shared {
                    let counts = match chain.get(position as usize) {
                        Some(key) if !request.unpinned.contains(key) => {
                            read.insert((group, Cached::id(key)))
                        }
                        _ => true,
                    };
                    held += u32::from(counts);
                }
                self.charge_units(&mut units, (group, class), (held, 0), groups);
            }
        }
        units
    }
}

/// The blocks an admission needs (SD §7.2 "Admission to the cache").
impl Kv {
    /// Per region, the units the admission of a request whose prefix hit holds `hit` tokens
    /// allocates (SD §7.2, `S/admission.rs:1051-1160`, `S/kv_cache/mod.rs:2101-2206`): what
    /// `fits` compares with the units available. Per cache class but the draft cache, the most
    /// of the blocks its first call and the low watermark take and the least of the blocks after
    /// the prefix and the fewest that must coexist to compute the prompt in chunks, the blocks of
    /// the hit that `cache` holds on the host alone, which its `Load` allocates on the device,
    /// and the blocks of the hit its `Pin` takes out of the eviction order, which `available`
    /// counts as evictable (`unheld`; `S/admission.rs`'s `cached_free_units`); a request with
    /// context images every block of its context. Then every encoder output and declared result
    /// but those it holds, an encoder output whose `Pin` `cache` expects to find taking only what
    /// `available` counts of it as evictable ([`Kv::charge_encoding`]; SD §3.5, §4.4). A video
    /// request's crossings are the caller's to add.
    pub(crate) fn units_needed(
        &self,
        request: &Request,
        hit: u32,
        groups: &[WorkerGroup],
        budget: &Budget,
    ) -> Vec<u32> {
        self.units(request, (Some(hit), false), groups, budget)
    }

    /// Per region, the units a request's admission holds there when no other request holds any:
    /// what `never_fits` compares with the units the region is declared to hold (SD §3.5, §7.2).
    /// Per cache class but the draft cache, the most of the fewest blocks that must coexist to
    /// compute its prompt in chunks and the blocks its first call and the low watermark take from
    /// its start, every block of a context with images; and every encoder output and declared
    /// result, pinned or allocated, since a pinned block lies in the region too. A request whose
    /// `never_fits` passes is admitted once the requests ahead of it have ended: alone, and with
    /// no hit, it takes no more than this, and its hit is one it holds alone
    /// ([`Kv::holds_alone`]). A video request's crossings are the caller's to add.
    pub(crate) fn units_alone(
        &self,
        request: &Request,
        groups: &[WorkerGroup],
        budget: &Budget,
    ) -> Vec<u32> {
        self.units(request, (None, true), groups, budget)
    }

    /// Whether the admission of a token request with a prefix hit of `hit` tokens holds no more
    /// than the units each region is declared to hold ([`Kv::declared`]) when no other request
    /// holds any (SD §4.4 "The hit"): per cache class the blocks of the hit it reads, pinned or
    /// loaded, beside what [`Kv::units_needed`] counts from the hit on, and every encoder output.
    /// From a hit the first chunk may reach further than from position 0, the low watermark
    /// beyond it, so a hit can hold more than [`Kv::units_alone`] counts. With a hit that holds
    /// no more than the declared units, `fits` holds once every other request has left the
    /// regions and a resize in progress completed, so no admission is tried with a deeper one: it
    /// would wait at the head of the queue for good.
    fn holds_alone(
        &self,
        request: &Request,
        hit: u32,
        groups: &[WorkerGroup],
        budget: &Budget,
    ) -> bool {
        let units = self.units(request, (Some(hit), true), groups, budget);
        units
            .iter()
            .enumerate()
            .all(|(region, &units)| units <= self.declared(region))
    }

    /// [`Kv::units_needed`] with a hit. With `alone`, what the admission holds when no other
    /// request holds any: [`Kv::units_alone`] without a hit; with one, what [`Kv::holds_alone`]
    /// compares with the capacity, every block of the hit the request reads and every encoder
    /// output and declared result counted.
    fn units(
        &self,
        request: &Request,
        (hit, alone): (Option<u32>, bool),
        groups: &[WorkerGroup],
        budget: &Budget,
    ) -> Vec<u32> {
        let mut units = vec![0; self.regions.len()];
        if request.is_video() {
            let results = self.declared_results(request, groups).unwrap_or_default();
            for (at, (phase, class, size)) in (0u32..).zip(results) {
                let group = request.placement[phase.index()].or_else(|| {
                    let serving = groups.iter().find(|group| group.serves(phase));
                    serving.map(WorkerGroup::id)
                });
                let Some(group) = group else {
                    continue;
                };
                // Its conditioning is its encoder output, pinned where `cache` holds it.
                if !alone && Some(at) == request.conditioning_position(groups) {
                    self.charge_encoding(&mut units, request, (0, group, class, size), groups);
                } else {
                    self.charge_units(&mut units, (group, class), (1, size), groups);
                }
            }
            return units;
        }
        let Some(group) = kv_group(request, groups) else {
            return units;
        };

        let prompt = request.num_prompt_tokens;
        let longest = longest(request, groups);
        let cached = hit.unwrap_or(0);
        let chunk = first_chunk(request, cached, budget);
        // The tiers `cache` expects of the prompt's blocks: a hit's blocks on the host alone are
        // loaded into blocks the `Load` allocates on the device, those on the device are pinned.
        // A request alone in the regions holds every block of its hit, whatever its tier.
        let tiers = (hit.is_some() && !alone).then(|| self.tiers(request, group, groups));
        let host = reads_host(request);
        for (index, &(kind, block)) in shapes(request, groups).iter().enumerate() {
            let Ok(class) = u8::try_from(index) else {
                continue;
            };
            if kind.is_draft() {
                continue;
            }
            let pages = prompt.div_ceil(block);
            let least = match kind {
                CacheGroupKind::State => pages.min(2),
                CacheGroupKind::SlidingWindow { window } => {
                    pages.min((window + chunk).div_ceil(block) + 1)
                }
                _ => pages,
            };
            let reaches = |from: u32, to: u32| u32::from(to / block * block > from);
            // The blocks the admission's first round or rounds hold before the request runs:
            // its first chunk from the hit and the low watermark, the checkpoint the chunk keeps
            // for a recurrent state (`settled`).
            let first = if kind.is_state() {
                reaches(cached, cached + chunk)
            } else {
                reach(cached + chunk, block, longest) - cached / block
            };
            let blocks = if !keyed(request) && request.image_spans().next().is_some() {
                if kind.is_state() {
                    u32::from(prompt >= block)
                } else {
                    pages
                }
            } else if hit.is_none() {
                least
            } else if kind.is_state() {
                reaches(cached, prompt).min(least)
            } else {
                pages.saturating_sub(cached / block).min(least)
            };
            let range = needed(kind, block, cached);
            let read = if alone {
                range.end.saturating_sub(range.start)
            } else {
                let (device, loaded) = tiers
                    .as_ref()
                    .and_then(|tiers| tiers.get(index))
                    .and_then(|tiers| usable(tiers, range.clone(), host))
                    .unwrap_or((0, 0));
                // The hit's blocks on the device are pinned, and those no request holds leave
                // the eviction order with the `Pin`: the units `available` counts of them are no
                // allocation's, the admission's own included.
                let pinned = request.keys.get(index).and_then(|keys| {
                    keys.get(range.start as usize..range.start.saturating_add(device) as usize)
                });
                let unheld = pinned.map_or(0, |keys| self.unheld(keys, group));
                loaded.saturating_add(unheld)
            };
            let blocks = blocks.max(first).saturating_add(read);
            self.charge_units(&mut units, (group, class), (blocks, 0), groups);
        }

        if let Some(class) = self.encoder_class(request, groups) {
            for (index, (_, rows, _, input)) in request.image_spans().enumerate() {
                let phase = match input.encoder {
                    ImageEncoder::Vision => Phase::VisionEncode,
                    ImageEncoder::Latent => Phase::LatentEncode,
                };
                let encoder = request.placement[phase.index()].unwrap_or(group);
                let size = self.encoding_units((encoder, class), rows, groups);
                if alone {
                    self.charge_units(&mut units, (encoder, class), (1, size), groups);
                } else {
                    let output = (index, encoder, class, size);
                    self.charge_encoding(&mut units, request, output, groups);
                }
            }
        }
        units
    }

    /// Charges `units` with what the admission of `request` takes for its encoder output `index`
    /// (into `Request::encodings`), a block of `size` units of `class` for its owner on `group`
    /// (SD §3.5 `fits`, §4.4): nothing when the request holds it. When the request reads cached
    /// outputs (a video request's conditioning always) and `cache` holds this one on the device in
    /// `group`, its `Pin` takes what `available` counts of it as evictable ([`Kv::evictable`]):
    /// nothing while another request holds it, and its size while it lies in the head of the
    /// order this queue pair installed, which the buffer pool's word counts. Otherwise its
    /// `Allocate`, or the `Load` of its host copy, takes a block of `size`.
    fn charge_encoding(
        &self,
        units: &mut [u32],
        request: &Request,
        (index, group_id, class, size): (usize, GroupId, u8, u32),
        groups: &[WorkerGroup],
    ) {
        let entry = request.encodings.get(index);
        if entry.is_some_and(|(_, held)| held.is_some()) {
            return;
        }
        let reads = request.generation().is_none_or(|input| input.cache.read);
        let pinned = entry
            .map(|&(key, _)| key)
            .filter(|&key| reads && self.on_device(key, group_id));
        let (Some(key), Some(group)) = (pinned, group(groups, group_id)) else {
            self.charge_units(units, (group_id, class), (1, size), groups);
            return;
        };
        for (region, taken) in self.evictable(&key, group, size) {
            if let Some(total) = units.get_mut(region) {
                *total = total.saturating_add(taken);
            }
        }
    }

    /// Whether `cache` holds the block of `key` on the device in `group`.
    fn on_device(&self, key: Key, group: GroupId) -> bool {
        self.expected_hit(&[key], group)[0] & DEVICE != 0
    }
}

/// An admission's rounds (SD §3.5, §4.1, §4.4 "The hit", "Pins, loads, clones and prefetches").
impl Kv {
    /// The hit `cache` expects for a token request, in tokens (SD §4.4): what `fits` is
    /// evaluated with before the admission's first round. Its keys are set.
    pub(crate) fn expected(
        &self,
        request: &Request,
        groups: &[WorkerGroup],
        budget: &Budget,
    ) -> u32 {
        kv_group(request, groups).map_or(0, |group| {
            self.lookup(request, (group, true), groups, budget).0
        })
    }

    /// Whether the admission of `request`, whose hit `cache` expects to be `expected` tokens,
    /// waits for `cache` to learn the blocks its previous life committed (SD §4.4 "The hit",
    /// §4.5): a resumed token request that reads the cache, of which `cache` expects less than
    /// that life computed (every token but its last), waits until a `List` asked after it
    /// arrived has been answered; and with it, whether the caller is to ask that `List` now: none
    /// asked since it arrived is on its way, and none asked before is, whose answer comes first,
    /// since one is outstanding at a time.
    pub(crate) fn awaits_listing(
        &self,
        request: &Request,
        expected: u32,
        groups: &[WorkerGroup],
        budget: &Budget,
    ) -> (bool, bool) {
        let computed = request.num_prompt_tokens.saturating_sub(1);
        let token = request.generation().is_some();
        let short =
            token && request.resumes() && reads(request, groups, budget) && expected < computed;
        if !short || self.lists_answered > request.listing {
            return (false, false);
        }
        let ask = self.lists_asked == request.listing && self.lists_answered == self.lists_asked;
        (true, ask)
    }

    /// What `cache` expects of a token request's prefix on `group` (SD §4.4 "The hit"): the
    /// deepest multiple of the step, at most the reuse limit, that every cache class but the
    /// draft cache accepts and with which the admission holds no more than the regions' capacity
    /// alone ([`Kv::holds_alone`]), a rejected boundary not ending the search; with `tails`,
    /// extended to the longest end short of the next multiple at which every such class holds a
    /// partial last block on the device under its tail key, which a recurrent state never does,
    /// and with which the admission holds no more than that alone either. Per class, the table
    /// the hit fills when the answers bear `cache` out: read-only from `first` to `shared`, the
    /// hit's blocks on the device and then on the host alone (never for a request that continues
    /// another epoch's state), and to `end` the clone of its partial last block; for the draft
    /// cache, its leading blocks on the device within the hit.
    fn lookup(
        &self,
        request: &Request,
        (group, tails): (GroupId, bool),
        groups: &[WorkerGroup],
        budget: &Budget,
    ) -> (u32, Vec<Blocks>) {
        let shapes = shapes(request, groups);
        let mut outlooks = vec![Blocks::default(); shapes.len()];
        if !reads(request, groups, budget) || shapes.is_empty() {
            return (0, outlooks);
        }
        let step = step(shapes);
        let prompt = request.num_prompt_tokens;
        let limit = prompt.saturating_sub(1) / step * step;
        let host = reads_host(request);
        let tiers = self.tiers(request, group, groups);
        let required: Vec<usize> = (0..shapes.len())
            .filter(|&class| !shapes[class].0.is_draft())
            .collect();

        let accepts = |boundary: u32| {
            let read = required.iter().all(|&class| {
                let (kind, block) = shapes[class];
                usable(&tiers[class], needed(kind, block, boundary), host).is_some()
            });
            read && self.holds_alone(request, boundary, groups, budget)
        };
        let mut hit = (1..=limit / step)
            .rev()
            .map(|boundary| boundary * step)
            .find(|&boundary| accepts(boundary))
            .unwrap_or(0);

        let top = prompt.saturating_sub(1).min(hit + step - 1);
        for end in (hit + 1..=top).rev().filter(|_| tails) {
            let held = required.iter().all(|&class| {
                let (kind, block) = shapes[class];
                let rest = end % block;
                if (kind.is_state() && rest != 0)
                    || usable(&tiers[class], needed(kind, block, end), host).is_none()
                {
                    return false;
                }
                rest == 0
                    || self
                        .tail_key(request, (class, end), groups)
                        .is_some_and(|key| self.expected_hit(&[key], group)[0] & DEVICE != 0)
            });
            if held && self.holds_alone(request, end, groups, budget) {
                hit = end;
                break;
            }
        }

        for (class, &(kind, block)) in shapes.iter().enumerate() {
            let tiers = &tiers[class];
            outlooks[class] = if kind.is_draft() {
                let range = 0..(hit / block).min(count(tiers.len()));
                let device = tiers[..range.end as usize]
                    .iter()
                    .take_while(|tier| **tier & DEVICE != 0)
                    .count();
                run(0, count(device), count(device))
            } else {
                let range = needed(kind, block, hit);
                let (device, loaded) = usable(tiers, range.clone(), host).unwrap_or((0, 0));
                let shared = range.start + device + loaded;
                // A hit past the last boundary ends in a partial last block, held on the device
                // under its tail key, which is cloned after the blocks read.
                let clone = !hit.is_multiple_of(step) && !hit.is_multiple_of(block);
                run(range.start, shared, shared + u32::from(clone))
            };
        }
        (hit, outlooks)
    }

    /// Per cache class, the tiers `cache` expects for the keys of the request's prompt blocks a
    /// hit may read: those before the prompt's last token.
    fn tiers(&self, request: &Request, group: GroupId, groups: &[WorkerGroup]) -> Vec<Vec<u8>> {
        let prompt = request.num_prompt_tokens.saturating_sub(1);
        let shapes = shapes(request, groups).iter().zip(&request.keys);
        shapes
            .map(|(&(_, block), keys)| {
                let blocks = ((prompt / block) as usize).min(keys.len());
                self.expected_hit(&keys[..blocks], group)
            })
            .collect()
    }

    /// The `Prefetch`es that deepen a token request's expected `hit` (SD §4.4): from the remote
    /// store (Mooncake Store) when the blocks `cache` says it holds deepen the hit by at least
    /// the policy's `prefetch_min_tokens`; with `forced`, the policy named the request and any
    /// deepening is asked for. It deepens it only to a boundary with which the admission holds
    /// no more than the regions' capacity alone ([`Kv::holds_alone`]). One per cache class
    /// but the draft cache, with the class's keys beyond what the device and the host hold, and a
    /// deadline of the copy timeout of the deepening from now (`SchedulerConfig::copy_timeout`).
    pub(crate) fn prefetch_requests(
        &self,
        request: &Request,
        forced: bool,
        step: &Step,
    ) -> Vec<runtime::Request> {
        let Some(group) = kv_group(request, step.groups) else {
            return Vec::new();
        };
        let (hit, outlooks) = self.lookup(request, (group, true), step.groups, step.budget);
        self.prefetches(request, (group, hit, &outlooks), forced, step)
    }

    fn prefetches(
        &self,
        request: &Request,
        (group, hit, outlooks): (GroupId, u32, &[Blocks]),
        forced: bool,
        step: &Step,
    ) -> Vec<runtime::Request> {
        let groups = step.groups;
        let shapes = shapes(request, groups);
        if !reads(request, groups, step.budget) || shapes.is_empty() {
            return Vec::new();
        }
        let block_step = self::step(shapes);
        let limit = request.num_prompt_tokens.saturating_sub(1) / block_step * block_step;
        let least = if forced {
            1
        } else {
            step.prefetch_min_tokens.max(1)
        };
        let tiers = self.tiers(request, group, groups);
        let present = |boundary: u32| {
            shapes.iter().zip(&tiers).all(|(&(kind, block), tiers)| {
                let range = needed(kind, block, boundary);
                kind.is_draft()
                    || tiers
                        .get(range.start as usize..range.end as usize)
                        .is_some_and(|tiers| {
                            tiers
                                .iter()
                                .all(|tier| *tier & (DEVICE | HOST | REMOTE) != 0)
                        })
            })
        };
        // A prefetch deepens the hit only to a boundary the admission holds alone, as `lookup`
        // reads it (SD §4.4 "The hit").
        let deepest = |accepts: &dyn Fn(u32) -> bool| {
            (1..=limit / block_step)
                .rev()
                .map(|boundary| boundary * block_step)
                .find(|&boundary| {
                    accepts(boundary) && self.holds_alone(request, boundary, groups, step.budget)
                })
                .unwrap_or(0)
        };
        let target = deepest(&present);
        if target < hit.saturating_add(least) {
            return Vec::new();
        }

        let wait = step.config.copy_timeout(target - hit).as_nanos();
        let now = monotonic_nanos(Instant::now());
        let deadline = now.saturating_add(u64::try_from(wait).unwrap_or(u64::MAX));
        let mut asks = Vec::new();
        for (class, (&(kind, block), outlook)) in shapes.iter().zip(outlooks).enumerate() {
            let range = needed(kind, block, target);
            let from = range.start.max(outlook.shared) as usize;
            let keys = request.keys.get(class).map_or(&[][..], Vec::as_slice);
            let keys = keys
                .get(from..(range.end as usize).min(keys.len()))
                .unwrap_or_default();
            if kind.is_draft() || keys.is_empty() {
                continue;
            }
            asks.push(runtime::Request::Prefetch {
                group,
                class: u8::try_from(class).unwrap_or(u8::MAX),
                units: 0,
                keys: keys.to_vec(),
                deadline,
            });
        }
        asks
    }

    /// The first round of `request`'s admission (SD §3.5 `admit`, §4.1), once its owners are
    /// attached, its keys set and its blocks sized: the requests to ask, in the order the buffer
    /// pool takes them. A video request allocates one product block per declared result, at its
    /// position for its owner on the writing group, and restores the trajectory it resumes; the
    /// crossings its calls wait for are the caller's to set. A token request asks per cache class
    /// for the hit `cache` expects ([`Kv::lookup`]), for each encoder output a `Pin` when `cache`
    /// holds it on the device, a `Load` when on the host alone and an `Allocate` otherwise, the
    /// `Prefetch`es that pay, and, unless it prefetches or clones, the blocks of its first chunk
    /// and the low watermark. Sets the expected hit as `num_cached_tokens`.
    pub(crate) fn admission_requests(
        &self,
        request: &mut Request,
        step: &Step,
    ) -> Vec<runtime::Request> {
        let groups = step.groups;
        let mut asks = Vec::new();
        let mut blocks = vec![Blocks::default(); self.classes.len()];
        let mut hit = 0;

        if request.is_video() {
            let results = self.declared_results(request, groups).unwrap_or_default();
            // The conditioning is the request's encoder output, held where `cache` holds it.
            let conditioning = request.conditioning_position(groups);
            let key = request.encodings.first().map(|&(key, _)| key);
            for (at, (phase, class, units)) in (0u32..).zip(results) {
                let (Some(owner), Some(group)) =
                    (request.owner_on(phase), request.placement[phase.index()])
                else {
                    continue;
                };
                asks.push(match key.filter(|_| Some(at) == conditioning) {
                    Some(key) => self.encoder_ask(owner, (class, at), key, Some(group), units),
                    None => allocate(owner, class, at, 1, units),
                });
                if let Some(table) = blocks.get_mut(usize::from(class)) {
                    ask(table, at, 1);
                }
            }
            let key = request.restore_key(self.latent_seed, groups);
            if let (Some(owner), Some(key)) = (request.owner_on(Phase::Denoise), key) {
                asks.push(runtime::Request::Restore { owner, key });
            }
        } else if let (Some(owner), Some(group)) = (kv_owner(request), kv_group(request, groups)) {
            let outlooks;
            (hit, outlooks) = self.lookup(request, (group, true), groups, step.budget);
            let cloned = outlooks.iter().any(|outlook| outlook.end > outlook.shared);
            self.read_asks(
                request,
                (owner, group, hit),
                &outlooks,
                (&mut blocks, &mut asks),
                groups,
            );
            self.encoder_asks(request, &mut blocks, &mut asks, groups);
            let prefetch = self.prefetches(request, (group, hit, &outlooks), false, step);
            if prefetch.is_empty() && !cloned {
                self.first_call_asks(request, (owner, hit), (&mut blocks, &mut asks), step);
            }
            asks.extend(prefetch);
        }

        request.blocks = blocks;
        request.num_cached_tokens = hit;
        request.num_host_tokens = 0;
        asks
    }

    /// The `Pin`, `Load` and `Clone` of each class's expected table `outlooks` of a hit of `hit`
    /// tokens ([`Kv::lookup`]), from positions the request does not hold yet: from `shared` of
    /// the class's counts in `blocks`, which start there. The read-only run's leading blocks
    /// that `cache` holds on the device in `group`, the owner's, are pinned and the rest loaded
    /// from the host; a partial last block is cloned under its tail key.
    fn read_asks(
        &self,
        request: &Request,
        (owner, group, hit): (OwnerId, GroupId, u32),
        outlooks: &[Blocks],
        (blocks, asks): (&mut [Blocks], &mut Vec<runtime::Request>),
        groups: &[WorkerGroup],
    ) {
        for (index, (outlook, table)) in outlooks.iter().zip(blocks).enumerate() {
            let (Ok(class), Some(keys)) = (u8::try_from(index), request.keys.get(index)) else {
                continue;
            };
            let first = outlook.first;
            if table.first == table.asked {
                *table = run(first, first, first);
            }
            let keys_of = |range: Range<u32>| keys.get(range.start as usize..range.end as usize);
            let read = keys_of(first..outlook.shared).unwrap_or_default();
            let device = self
                .expected_hit(read, group)
                .iter()
                .take_while(|tier| **tier & DEVICE != 0)
                .count();
            let from = table.shared.max(first);
            let pinned = first + count(device);
            if let Some(keys) = keys_of(from..pinned).filter(|keys| !keys.is_empty()) {
                asks.push(runtime::Request::Pin {
                    owner,
                    class,
                    at: from,
                    keys: keys.to_vec(),
                });
            }
            let from = from.max(pinned);
            if let Some(keys) = keys_of(from..outlook.shared).filter(|keys| !keys.is_empty()) {
                asks.push(runtime::Request::Load {
                    owner,
                    class,
                    at: from,
                    keys: keys.to_vec(),
                });
            }
            let tail = (outlook.end > outlook.shared)
                .then(|| self.tail_key(request, (index, hit), groups))
                .flatten();
            if let Some(key) = tail {
                asks.push(runtime::Request::Clone {
                    owner,
                    class,
                    at: outlook.shared,
                    key,
                });
            }
            table.asked = table.asked.max(outlook.end);
        }
    }

    /// The encoder outputs' requests of a token request's first round, at the position of each
    /// encoder input, for its owner on the group of its encode phase; `cache` is read only when
    /// the request reads cached outputs (SD §4.3).
    fn encoder_asks(
        &self,
        request: &Request,
        blocks: &mut [Blocks],
        asks: &mut Vec<runtime::Request>,
        groups: &[WorkerGroup],
    ) {
        let Some(class) = self.encoder_class(request, groups) else {
            return;
        };
        let reads = request.generation().is_some_and(|input| input.cache.read);
        let inputs = request.encodings.iter().zip(request.image_spans());
        for (at, ((key, _), (_, rows, _, input))) in (0u32..).zip(inputs) {
            let phase = match input.encoder {
                ImageEncoder::Vision => Phase::VisionEncode,
                ImageEncoder::Latent => Phase::LatentEncode,
            };
            let (Some(owner), Some(group)) =
                (request.owner_on(phase), request.placement[phase.index()])
            else {
                continue;
            };
            let units = self.encoding_units((group, class), rows, groups);
            let read = reads.then_some(group);
            asks.push(self.encoder_ask(owner, (class, at), *key, read, units));
            if let Some(table) = blocks.get_mut(usize::from(class)) {
                ask(table, at, 1);
            }
        }
    }

    /// The request that gives `owner` the encoder output of `key` at position `at` of `class`
    /// (SD §4.3, §4.4): with `read`, the owner's group whose copies `cache` is read for, a `Pin`
    /// when it holds the output on the device there and a `Load` when on the host alone;
    /// otherwise an `Allocate` of one block of `units`, which the output's encode fills.
    fn encoder_ask(
        &self,
        owner: OwnerId,
        (class, at): (u8, u32),
        key: Key,
        read: Option<GroupId>,
        units: u32,
    ) -> runtime::Request {
        let tiers = read.map_or(0, |group| self.expected_hit(&[key], group)[0]);
        let keys = vec![key];
        if tiers & DEVICE != 0 {
            runtime::Request::Pin {
                owner,
                class,
                at,
                keys,
            }
        } else if tiers & HOST != 0 {
            runtime::Request::Load {
                owner,
                class,
                at,
                keys,
            }
        } else {
            allocate(owner, class, at, 1, units)
        }
    }

    /// The `Allocate`s of a token request's first chunk from `hit` and the low watermark, per
    /// cache class: a full-attention or sliding-window class from the positions its read requests
    /// asked for, a recurrent-state class the checkpoint the chunk keeps, and the draft cache,
    /// where the deployment drafts, as full attention over the draft entries.
    pub(crate) fn first_call_asks(
        &self,
        request: &Request,
        (owner, hit): (OwnerId, u32),
        (blocks, asks): (&mut [Blocks], &mut Vec<runtime::Request>),
        step: &Step,
    ) {
        let groups = step.groups;
        let end = hit.saturating_add(first_chunk(request, hit, step.budget));
        let interval = state_shape(request, groups).map_or(1, |(_, _, interval)| interval);
        for (index, (&(kind, block), table)) in
            shapes(request, groups).iter().zip(blocks).enumerate()
        {
            let Ok(class) = u8::try_from(index) else {
                continue;
            };
            let (at, count) = if kind.is_draft() {
                // The draft cache over the first chunk, where the deployment drafts: the prefill
                // writes its entries, which count only in draft blocks the request holds (SD
                // §4.4). Draft blocks are outside admission: a `Full` to them leaves the draft
                // cache short (`admission_answered`). A draft cache that reads blocks of the hit
                // grows past them only once they are held (`allocate_ahead`), so that a pin that
                // falls short leaves no gap in its table, which the admission does not settle
                // on (`settled`).
                if step.config.draft_tokens == 0 || table.asked > table.first {
                    continue;
                }
                let wanted = end.div_ceil(block) + LOW_WATERMARK;
                (table.asked, wanted.saturating_sub(table.asked))
            } else if kind.is_state() {
                let Some(boundary) = kept(Phase::Prefill, hit, end, block, interval) else {
                    continue;
                };
                (boundary / block - 1, 1)
            } else {
                let wanted = reach(end, block, longest(request, groups));
                (table.asked, wanted.saturating_sub(table.asked))
            };
            if count > 0 && !(kind.is_state() && checkpoints(table).contains(&Some(at))) {
                asks.push(allocate(owner, class, at, count, 0));
                ask(table, at, count);
            }
        }
    }

    /// Applies what one answer of `request`'s memory requests states of its tables (SD §3.5
    /// `answered`, §4.2), and returns the requests the answer calls for, which the caller asks
    /// like the admission's. `Allocated` and `Cloned` hold private entries; `Pinned`, and the
    /// `Loaded` or `Duplicate` that answers a `Load`'s copy, hold read-only ones, which extend
    /// `shared` only where they continue it: entries beyond a position the hit lost are given up
    /// at once; a `Commit` answered `Duplicate` states nothing of them, its block staying private
    /// and held as it was (BP §11). An encoder output the answer did not hold is allocated at
    /// once, so that its encode computes it; one it holds has its position. A `Loaded` that
    /// copied nothing holds no entry:
    /// its `Load` comes narrowed to no key ([`super::covered`]), and the positions it named are
    /// missing like those a short pin leaves, for the admission's last round to allocate. A pin
    /// that falls short of what `cache` expected, and the prefetch it calls for, read the
    /// positions it lost from the lower tiers ([`Kv::reread`]). What the request gives up is
    /// told to the policy through `told`.
    pub(crate) fn filled(
        &mut self,
        request: &mut Request,
        (asked, completion): (&runtime::Request, &Completion),
        step: &Step,
        told: &mut Vec<Event>,
    ) -> Vec<runtime::Request> {
        // A handoff's import holds what its lend held (`handoff.rs`).
        if let runtime::Request::Import { owner, tables, .. } = asked {
            return self.imported(request, (*owner, tables), completion, step);
        }
        let groups = step.groups;
        let mut follow = Vec::new();
        let first_required = shapes(request, groups)
            .iter()
            .position(|(kind, _)| kind.required_rows());
        let block = |class: u8| {
            shapes(request, groups)
                .get(usize::from(class))
                .map_or(0, |&(_, b)| b)
        };
        let Some((owner, Some(class), positions)) = span(asked) else {
            let (
                runtime::Request::Prefetch { class, keys, .. },
                Completion::Prefetched { found, .. },
            ) = (asked, completion)
            else {
                return follow;
            };
            let block = block(*class);

            // A prefetch from the remote store of positions within the expected hit is one a
            // short pin called for ([`Kv::reread`]): what it found is loaded at those positions
            // now. One that deepens the hit is loaded by the admission's next round
            // (`missing_requests`).
            let at = request.keys.get(usize::from(*class)).and_then(|chain| {
                let first = keys.first()?;
                chain.iter().position(|key| key == first).map(count)
            });
            if let (Some(owner), Some(at)) = (kv_owner(request), at)
                && (at + 1).saturating_mul(block) <= request.num_cached_tokens
            {
                let found = keys.get(..*found as usize).unwrap_or(keys);
                if !found.is_empty() {
                    follow.push(runtime::Request::Load {
                        owner,
                        class: *class,
                        at,
                        keys: found.to_vec(),
                    });
                }
            }
            return follow;
        };
        let at = positions.start;
        let keys = match asked {
            runtime::Request::Pin { keys, .. } | runtime::Request::Load { keys, .. } => &keys[..],
            _ => &[],
        };
        let group = request
            .attachment(owner)
            .map_or(GroupId(u8::MAX), |(group, _)| group);
        let held = match (asked, completion) {
            (_, Completion::Allocated { .. } | Completion::Cloned) => {
                raise_end(request, class, positions.end);
                return follow;
            }
            // A copy holds nothing yet: the answer to its `Finish` is its result.
            (_, Completion::Copy { .. }) => return follow,
            // The answer states one block per leading key it pinned.
            (_, Completion::Pinned { blocks, .. }) => blocks.len(),
            // A load's copy that found its keys cached meanwhile holds them all the same,
            // read-only; a `Commit` answered `Duplicate` leaves its block private, held as it was,
            // and states nothing more of the tables.
            (runtime::Request::Load { .. }, Completion::Duplicate) => keys.len(),
            (_, Completion::Loaded) => self
                .expected_hit(keys, group)
                .iter()
                .take_while(|tier| **tier & DEVICE != 0)
                .count(),
            (runtime::Request::Pin { .. } | runtime::Request::Load { .. }, _) => 0,
            _ => return follow,
        };
        let held = count(held);

        // An encoder output is one block at its position. One the answer did not hold is
        // allocated, of its declared result's size for a video request's conditioning and of its
        // rows of features for a chat image's. A video request's conditioning a `Pin` found is
        // read from the cached block's one run when it crosses to another process.
        if let Some((index, _)) = self.encoding_at(request, class, at, groups) {
            let pinned = match completion {
                Completion::Pinned { blocks, .. } => blocks
                    .first()
                    .and_then(|runs| runs.first())
                    .map(|run| run.0),
                _ => None,
            };
            if held == 0 {
                let units = self
                    .encoding_size(request, (class, at), group, groups)
                    .unwrap_or(1);
                follow.push(allocate(owner, class, at, 1, units));
            } else {
                if let Some(entry) = request.encodings.get_mut(index) {
                    entry.1 = Some(at);
                }
                let crossings = request.crossings.iter_mut();
                for crossing in crossings.filter(|(position, ..)| *position == at) {
                    crossing.2 = pinned.unwrap_or(crossing.2);
                }
                raise_end(request, class, at.saturating_add(1));
            }
            return follow;
        }

        let loaded = matches!(completion, Completion::Loaded | Completion::Duplicate);
        let tokens = block(class).saturating_mul(held);
        let state = state_shape(request, groups).map(|(state, _, _)| state);
        let draft =
            draft_class(request, groups).is_some_and(|(draft, _)| draft == usize::from(class));
        let read = count(keys.len());
        let Some(blocks) = request.blocks.get_mut(usize::from(class)) else {
            return follow;
        };
        let continued = at == blocks.shared;
        if continued {
            blocks.shared = at.saturating_add(held);
            blocks.end = blocks.end.max(blocks.shared);
            // A recurrent-state base not found leaves the table its checkpoint ahead, or nothing.
            if held == 0 && state == Some(usize::from(class)) && at == blocks.first {
                let first = if blocks.asked > at + 1 {
                    blocks.asked - 1
                } else {
                    blocks.asked
                };
                (blocks.first, blocks.shared) = (first, first);
                blocks.end = blocks.end.max(first);
            }
            if loaded && first_required == Some(usize::from(class)) {
                request.num_host_tokens = request.num_host_tokens.saturating_add(tokens);
            }
            // A draft cache whose read fell short grows from what it holds: its blocks are
            // outside admission, which settles without them, so no later round asks for the
            // positions the read lost, and a block allocated past them would leave a gap.
            if draft && held < read && blocks.asked == at.saturating_add(read) {
                blocks.asked = blocks.end;
            }
        } else if held > 0 {
            follow.push(release(owner, class, at, held));
            let released = keys.get(..held as usize).unwrap_or_default().to_vec();
            self.unpin(Some(request), &released, groups, told);
        }

        if let Completion::Pinned { host, .. } = completion
            && continued
            && held < read
        {
            let lost = keys.get(held as usize..).unwrap_or_default();
            let from = at.saturating_add(held);
            follow.extend(self.reread(request, (owner, class, from), lost, *host, step));
        }
        follow
    }

    /// The requests that read again the positions from `at` on of class `class` of `request`'s
    /// admission, whose `keys` a pin did not find valid in the owner's group (SD §4.4 "Pins,
    /// loads, clones and prefetches", BP §12.1): a `Load` of the leading `host` of them, which
    /// the pin's answer found on the host; when it found none there, a `Prefetch` from the
    /// remote store of the leading ones `cache` holds there, when they are at least the policy's
    /// `prefetch_min_tokens`, whose answer loads what it found ([`Kv::filled`]). The positions
    /// they do not fill are allocated by the admission's last round, from the hit the answers
    /// support. Nothing for a recurrent state or the draft cache, or for a request that is not an
    /// admission.
    fn reread(
        &self,
        request: &Request,
        (owner, class, at): (OwnerId, u8, u32),
        keys: &[Key],
        host: u32,
        step: &Step,
    ) -> Vec<runtime::Request> {
        let admitting = request.queued && request.status == RequestStatus::WaitingForRemoteKvs;
        let (Some(&(kind, block)), Some(group)) = (
            shapes(request, step.groups).get(usize::from(class)),
            kv_group(request, step.groups),
        ) else {
            return Vec::new();
        };
        if !admitting || keys.is_empty() || kind.is_state() || kind.is_draft() {
            return Vec::new();
        }

        if host > 0 {
            let keys = keys.get(..host as usize).unwrap_or(keys);
            return vec![runtime::Request::Load {
                owner,
                class,
                at,
                keys: keys.to_vec(),
            }];
        }
        let stored = self
            .expected_hit(keys, group)
            .iter()
            .take_while(|tier| **tier & REMOTE != 0)
            .count();
        let tokens = count(stored).saturating_mul(block);
        if stored == 0 || tokens < step.prefetch_min_tokens.max(1) {
            return Vec::new();
        }
        let wait = step.config.copy_timeout(tokens).as_nanos();
        let deadline =
            monotonic_nanos(Instant::now()).saturating_add(u64::try_from(wait).unwrap_or(u64::MAX));
        vec![runtime::Request::Prefetch {
            group,
            class,
            units: 0,
            keys: keys[..stored].to_vec(),
            deadline,
        }]
    }

    /// One more round of `request`'s admission, once every answer arrived and its entries do not
    /// hold its first call ([`settled`]); the requests to ask, in order (SD §3.5
    /// `allocate_admission`, §4.4). After a round that allocated nothing because it prefetched,
    /// the round asks again for the hit `cache` now expects, beyond what the request holds, and
    /// for the blocks of its first chunk. Otherwise the hit is final: the deepest boundary, at
    /// most the expected hit, whose blocks the read-only entries hold (the cloned partial last
    /// block included where every class holds it); the read-only entries beyond it are given up,
    /// and every position of the first chunk and the low watermark that no entry holds is
    /// allocated. What the request gives up is told to the policy through `told`.
    pub(crate) fn missing_requests(
        &mut self,
        request: &mut Request,
        step: &Step,
        told: &mut Vec<Event>,
    ) -> Vec<runtime::Request> {
        let groups = step.groups;
        let mut asks = Vec::new();
        if request.generation().is_none() {
            return asks;
        }
        let (Some(owner), Some(group)) = (kv_owner(request), kv_group(request, groups)) else {
            return asks;
        };
        let shapes = shapes(request, groups).to_vec();
        let expected = request.num_cached_tokens;
        let block_step = self::step(&shapes);
        let mut blocks = request.blocks.clone();
        let mut unpinned = Vec::new();

        // Whether an earlier round asked for the first chunk's blocks: a class's positions asked
        // reach beyond what the expected hit reads.
        let allocated = shapes
            .iter()
            .zip(&blocks)
            .find(|((kind, _), _)| !kind.is_draft())
            .is_some_and(|(&(_, block), table)| table.asked > expected.div_ceil(block));
        let cloned = !expected.is_multiple_of(block_step);

        let hit = if !allocated && !cloned {
            let (hit, outlooks) = self.lookup(request, (group, false), groups, step.budget);
            for (index, (outlook, table)) in outlooks.iter().zip(blocks.iter_mut()).enumerate() {
                // What the request holds that the new hit does not continue is given up.
                let continues = outlook.first == table.first && table.shared <= outlook.shared;
                if table.shared > table.first && !continues {
                    let held = table.first..table.shared;
                    give_up(
                        request,
                        (owner, index, held, table.shared),
                        &mut asks,
                        &mut unpinned,
                    );
                    *table = run(outlook.first, outlook.first, outlook.first);
                } else {
                    // Every answer arrived: nothing beyond what it holds is held or asked.
                    table.end = table.shared;
                    table.asked = table.shared;
                }
            }
            let round = (&mut blocks[..], &mut asks);
            self.read_asks(request, (owner, group, hit), &outlooks, round, groups);
            self.first_call_asks(request, (owner, hit), (&mut blocks, &mut asks), step);
            hit
        } else {
            let hit = self.supported(
                request,
                (&blocks, expected, block_step),
                groups,
                step.budget,
            );
            let round = (&mut blocks[..], &mut asks, &mut unpinned);
            finalize(request, (owner, expected, hit), round, step);
            hit
        };

        self.unpin(Some(request), &unpinned, groups, told);
        request.blocks = blocks;
        request.num_cached_tokens = hit;
        request.num_host_tokens = request.num_host_tokens.min(hit);
        asks
    }

    /// The deepest hit of `request`, at most `expected`, whose blocks the read-only entries of
    /// `blocks` hold in every cache class but the draft cache and with which its admission holds
    /// no more than the regions' capacity alone ([`Kv::holds_alone`]): `expected` itself when it
    /// ends in a partial block every class cloned, else a multiple of `step`.
    fn supported(
        &self,
        request: &Request,
        (blocks, expected, step): (&[Blocks], u32, u32),
        groups: &[WorkerGroup],
        budget: &Budget,
    ) -> u32 {
        let shapes = shapes(request, groups);
        let holds = |boundary: u32, tail: bool| {
            let read = shapes.iter().zip(blocks).all(|(&(kind, block), table)| {
                let range = needed(kind, block, boundary);
                let read =
                    range.is_empty() || (table.first <= range.start && range.end <= table.shared);
                let clone = !tail
                    || boundary.is_multiple_of(block)
                    || (table.shared == boundary / block && table.end > boundary / block);
                kind.is_draft() || (read && clone)
            });
            read && self.holds_alone(request, boundary, groups, budget)
        };
        if !expected.is_multiple_of(step) && holds(expected, true) {
            return expected;
        }
        (1..=expected / step)
            .rev()
            .map(|boundary| boundary * step)
            .find(|&boundary| holds(boundary, false))
            .unwrap_or(0)
    }
}

/// Grows a table's `end` to `end`, as an answer that holds private entries does.
fn raise_end(request: &mut Request, class: u8, end: u32) {
    if let Some(blocks) = request.blocks.get_mut(usize::from(class)) {
        blocks.end = blocks.end.max(end);
    }
}

/// Whether the entries of `request`'s admission hold what its first call needs (SD §3.5), once
/// every answer arrived: per cache class but the draft cache, the read-only blocks its hit reads
/// with the partial last block it cloned, the private blocks of its first chunk and the low
/// watermark, and the checkpoint the chunk keeps. A video request's blocks are all or none.
/// False while a later round is due ([`Kv::missing_requests`]).
pub(crate) fn settled(request: &Request, groups: &[WorkerGroup], budget: &Budget) -> bool {
    if request.generation().is_none() {
        return true;
    }
    let hit = request.num_cached_tokens;
    let end = hit.saturating_add(first_chunk(request, hit, budget));
    let interval = state_shape(request, groups).map_or(1, |(_, _, interval)| interval);
    let shapes = shapes(request, groups).iter().zip(&request.blocks);
    shapes
        .filter(|((kind, _), _)| !kind.is_draft())
        .all(|(&(kind, block), blocks)| {
            let range = needed(kind, block, hit);
            let read =
                range.is_empty() || (blocks.first <= range.start && range.end <= blocks.shared);
            let tail = hit.is_multiple_of(block)
                || (blocks.shared == hit / block && blocks.end > hit / block);
            let private = if kind.is_state() {
                kept(Phase::Prefill, hit, end, block, interval).is_none_or(|boundary| {
                    checkpoints(blocks).contains(&Some(boundary / block - 1))
                })
            } else {
                blocks.end >= reach(end, block, longest(request, groups))
            };
            read && tail && private
        })
}

/// The tokens of a settled admission's draft cache: the leading draft blocks its hit pinned,
/// within the hit (SD §4.4, the draft cache as full attention over the draft tokens), from which
/// its `num_draft_tokens` starts.
pub(crate) fn draft_tokens(request: &Request, groups: &[WorkerGroup]) -> u32 {
    draft_class(request, groups).map_or(0, |(class, block)| {
        let shared = request.blocks.get(class).map_or(0, |table| table.shared);
        shared.saturating_mul(block).min(request.num_cached_tokens)
    })
}

/// The final round of an admission whose hit is `hit`, `expected` before (SD §4.4 "Pins, loads,
/// clones and prefetches"), per cache class: the read-only entries outside what the hit reads
/// are given up; a recurrent-state table keeps the checkpoint the hit ends at and the one its
/// first chunk keeps; and every position from the hit to the first chunk and the low watermark
/// that holds no private entry (the clone of the expected partial last block, or what an earlier
/// round allocated from the expected hit on) is allocated; the draft cache keeps its read-only
/// entries within the hit and is allocated again over the first chunk. Fills `unpinned` with the
/// keys of the shared entries given up.
fn finalize(
    request: &Request,
    (owner, expected, hit): (OwnerId, u32, u32),
    (blocks, asks, unpinned): (&mut [Blocks], &mut Vec<runtime::Request>, &mut Vec<Key>),
    step: &Step,
) {
    let groups = step.groups;
    let end = hit.saturating_add(first_chunk(request, hit, step.budget));
    let interval = state_shape(request, groups).map_or(1, |(_, _, interval)| interval);
    let shapes = shapes(request, groups).iter().zip(blocks.iter_mut());
    for (index, (&(kind, block), table)) in shapes.enumerate() {
        let Ok(class) = u8::try_from(index) else {
            continue;
        };

        if kind.is_state() {
            let [base, ahead] = checkpoints(table);
            let start = hit.checked_div(block).and_then(|end| end.checked_sub(1));
            let keep = base.filter(|base| Some(*base) == start && table.shared > *base);
            let wanted = kept(Phase::Prefill, hit, end, block, interval)
                .map(|boundary| boundary / block - 1);
            // The first chunk keeps its checkpoint into a private block: a held one at that
            // position serves only when private, and a read-only one there (a checkpoint the
            // expected hit pinned beyond the hit that holds) is given up and allocated again.
            let private = wanted.filter(|position| {
                [base, ahead].contains(&Some(*position)) && *position >= table.shared
            });
            for position in [base, ahead].into_iter().flatten() {
                if Some(position) != keep && Some(position) != private {
                    let range = position..position + 1;
                    give_up(request, (owner, index, range, table.shared), asks, unpinned);
                }
            }
            // The base the hit restores, then the checkpoint the first chunk keeps.
            *table = keep.map_or_else(Blocks::default, |base| run(base, base + 1, base + 1));
            if let Some(position) = wanted {
                if private.is_some() {
                    let first = keep.unwrap_or(position);
                    *table = run(first, table.shared.max(first), position + 1);
                } else {
                    asks.push(allocate(owner, class, position, 1, 0));
                    ask(table, position, 1);
                }
            }
            continue;
        }

        // The read-only entries the hit reads, and those outside it given up; the draft cache
        // gives up its private entries too, and holds the first chunk again from the end of its
        // read-only ones.
        let keep = if kind.is_draft() {
            0..table.shared.min(hit / block)
        } else {
            needed(kind, block, hit)
        };
        let read = table.first..table.shared;
        let held = if kind.is_draft() {
            table.end.max(read.end)
        } else {
            read.end
        };
        let (before, beyond) = (
            read.start..keep.start.min(read.end),
            keep.end.max(read.start)..held,
        );
        give_up(request, (owner, index, before, read.end), asks, unpinned);
        give_up(request, (owner, index, beyond, read.end), asks, unpinned);
        if kind.is_draft() {
            *table = run(0, keep.end, keep.end);
            let reach = end.div_ceil(block) + LOW_WATERMARK;
            if step.config.draft_tokens > 0 && reach > keep.end {
                asks.push(allocate(owner, class, keep.end, reach - keep.end, 0));
                table.asked = reach;
            }
            continue;
        }

        // The private entries held from the expected hit on.
        let clone = expected / block;
        let held = if !expected.is_multiple_of(block) {
            if table.end > clone {
                clone..clone + 1
            } else {
                clone..clone
            }
        } else {
            let from = expected.div_ceil(block);
            from..table.end.max(from)
        };
        // Up to the first chunk and the low watermark, and up to the blocks held beyond them, so
        // that the private entries are one run.
        let reach = reach(end, block, longest(request, groups));
        let wanted = hit / block..if held.is_empty() {
            reach
        } else {
            reach.max(held.start)
        };
        table.first = keep.start;
        table.shared = keep.end;
        table.end = if held.is_empty() {
            keep.end
        } else {
            keep.end.max(held.end)
        };
        for missing in [
            wanted.start..held.start.min(wanted.end),
            held.end.max(wanted.start)..wanted.end,
        ] {
            if missing.start < missing.end {
                let count = missing.end - missing.start;
                asks.push(allocate(owner, class, missing.start, count, 0));
                table.asked = table.asked.max(missing.end);
            }
        }
    }
}

/// Gives up positions `range` of a request's class `class` by one `Release`; the keys of the
/// read-only entries among them, those before `shared`, join `unpinned`, to leave the request's
/// holds in `cache`.
fn give_up(
    request: &Request,
    (owner, class, range, shared): (OwnerId, usize, Range<u32>, u32),
    asks: &mut Vec<runtime::Request>,
    unpinned: &mut Vec<Key>,
) {
    let Ok(number) = u8::try_from(class) else {
        return;
    };
    if range.is_empty() {
        return;
    }
    let keys = request.keys.get(class).map_or(&[][..], Vec::as_slice);
    let read_only = range.start as usize..(range.end.min(shared) as usize).min(keys.len());
    unpinned.extend_from_slice(keys.get(read_only).unwrap_or_default());
    asks.push(release(owner, number, range.start, range.end - range.start));
}

/// The checkpoint position the call of `batch`'s row `row` kept the recurrent state at, from the
/// row's `checkpoints` column (`rows.rs`), when the tokens the call committed reach its boundary.
/// A verification keeps the state after its tokens up to the boundary, its drafts among them,
/// before it learns which drafts are accepted: when it rejects a draft before the boundary, the
/// checkpoint holds the state of tokens the request does not have, so it is neither committed
/// nor counted as written, and the entry stays writable for the call that next reaches the
/// boundary, which keeps it again.
fn kept_by(
    request: &Request,
    (batch, row): (&Batch, usize),
    groups: &[WorkerGroup],
) -> Option<u32> {
    let (_, block, _) = state_shape(request, groups)?;
    let boundary = u32::try_from(*batch.columns.checkpoints.get(row)?).ok()?;
    let committed = request
        .num_computed_tokens
        .saturating_sub(request.in_flight(Phase::Decode));
    if boundary > committed {
        return None;
    }
    (boundary / block).checked_sub(1)
}

/// What a call's output leaves a request's tables to do (SD §3.7, §4.4 "Giving up behind",
/// "Commit", "The write policy", "Drafts first").
impl Kv {
    /// The `Commit`s of the blocks the output of `request`'s call in `batch`'s row `row` showed
    /// full, when the request's keys are given (SD §3.7, §4.4 "Commit"): per class, in index
    /// order from `shared`, every complete block of the tokens written (of the draft cache's, for
    /// the draft class) and the recurrent-state checkpoint the call kept, each under its key with
    /// the class's block size. The caller submits each, whose workers mark the entry read-only
    /// and stamp it; `shared` rises now.
    pub(crate) fn commit_filled(
        &mut self,
        request: &mut Request,
        (batch, row): (&Batch, usize),
        groups: &[WorkerGroup],
        budget: &Budget,
    ) -> Vec<runtime::Request> {
        self.extend_keys(request, groups);
        if !writes(request, groups, budget) {
            return Vec::new();
        }
        let checkpoint = kept_by(request, (batch, row), groups);
        let commits = self.commits(request, checkpoint, false, groups);
        self.hand_commits(request, commits, groups)
    }

    /// The `Commit`s of what a request that leaves as migratable holds and its continuation
    /// reads, once no call of it is in flight and before its owners' `Detach` (SD §3.7 `free`,
    /// §4.3): every complete block not committed yet and each full-attention and sliding-window
    /// class's partial last block under its tail key, whatever the write switches say; and its
    /// recurrent-state base, when the target wrote it and no refill lags behind it. A request
    /// whose blocks have no keys commits nothing.
    pub(crate) fn commit_tails(
        &mut self,
        request: &mut Request,
        groups: &[WorkerGroup],
        config: &SchedulerConfig,
    ) -> Vec<runtime::Request> {
        self.extend_keys(request, groups);
        if !keyed(request) {
            return Vec::new();
        }
        let base = state_shape(request, groups).and_then(|(class, block, _)| {
            let [base, _] = checkpoints(request.blocks.get(class)?);
            let boundary = (base? + 1) * block;
            let lagging =
                config.draft_tokens > 0 && request.num_draft_tokens < request.num_computed_tokens;
            let refilled = !lagging || boundary <= request.num_draft_tokens;
            (boundary <= request.num_computed_tokens && refilled).then_some(base?)
        });
        let commits = self.commits(request, base, true, groups);
        self.hand_commits(request, commits, groups)
    }

    /// The commits of [`Kv::commit_filled`] and, with `tails`, of [`Kv::commit_tails`]: per
    /// class the class, the position, the key and the tokens.
    fn commits(
        &self,
        request: &Request,
        checkpoint: Option<u32>,
        tails: bool,
        groups: &[WorkerGroup],
    ) -> Vec<(u8, u32, Key, u16)> {
        let visible = request
            .num_computed_tokens
            .saturating_sub(request.in_flight(Phase::Decode));
        let mut commits = Vec::new();
        let classes = shapes(request, groups).iter().zip(&request.blocks);
        for (index, (&(kind, block), table)) in classes.enumerate() {
            let (Ok(class), Some(keys)) = (u8::try_from(index), request.keys.get(index)) else {
                continue;
            };
            let tokens = u16::try_from(block).unwrap_or(u16::MAX);
            if kind.is_state() {
                let held = checkpoint.filter(|position| {
                    *position >= table.shared && checkpoints(table).contains(&Some(*position))
                });
                if let Some((position, key)) = held.and_then(|p| Some((p, *keys.get(p as usize)?)))
                {
                    commits.push((class, position, key, tokens));
                }
                continue;
            }

            // A request that leaves as migratable comes back with its tokens as its prompt and
            // computes the last of them again to sample, so it looks its blocks up to the token
            // before its last (SD §4.4 "The hit": a partial last block ends at most at
            // `prompt_len - 1`); a decode in flight as it left wrote that last token's entry,
            // which no lookup reaches. Its blocks are committed as it looks them up.
            let written = if kind.is_draft() {
                request.num_draft_tokens
            } else if tails {
                visible.min(count(request.token_ids.len()).saturating_sub(1))
            } else {
                visible
            };
            let from = table.shared.max(table.first);
            let complete = (written / block).min(table.end).min(count(keys.len()));
            for position in from..complete {
                commits.push((class, position, keys[position as usize], tokens));
            }
            let (full, rest) = (written / block, written % block);
            if tails
                && kind.required_rows()
                && rest > 0
                && full == complete.max(from)
                && full < table.end
                && let Some(key) = self.tail_key(request, (index, written), groups)
            {
                commits.push((class, full, key, u16::try_from(rest).unwrap_or(u16::MAX)));
            }
        }
        commits
    }

    /// The `Commit`s of `request`'s `commits`, raising `shared` over each. The full pages of the
    /// request's first full-attention class are offered to its KV events with their tokens
    /// (`super::super::events`).
    fn hand_commits(
        &mut self,
        request: &mut Request,
        commits: Vec<(u8, u32, Key, u16)>,
        groups: &[WorkerGroup],
    ) -> Vec<runtime::Request> {
        let Some(owner) = kv_owner(request) else {
            return Vec::new();
        };
        if self.events.publishing() {
            let shapes = shapes(request, groups);
            let published = shapes
                .iter()
                .position(|(kind, _)| *kind == CacheGroupKind::Full);
            let isolation = request
                .generation()
                .and_then(|input| input.cache.isolation_key);
            for &(class, at, key, tokens) in &commits {
                let Some(published) = published.filter(|&index| index == usize::from(class)) else {
                    continue;
                };
                let block = shapes[published].1;
                if u32::from(tokens) != block {
                    continue;
                }
                let keys = request.keys.get(published);
                let previous = at
                    .checked_sub(1)
                    .and_then(|previous| keys?.get(previous as usize));
                self.events.offer(
                    key,
                    previous,
                    at as usize,
                    block as usize,
                    &request.token_ids,
                    isolation,
                );
            }
        }
        let mut asks = Vec::with_capacity(commits.len());
        for (class, at, key, tokens) in commits {
            asks.push(commit(owner, class, at, key, tokens));
            if let Some(table) = request.blocks.get_mut(usize::from(class)) {
                table.shared = table.shared.max(at + 1);
            }
        }
        asks
    }

    /// The `Commit` of the encoder output the call in `batch`'s row `row` computed, when the
    /// request writes cached outputs (SD §3.7, §4.3), under its key with its rows as tokens;
    /// either way the request holds it at its position from now on. The output is the one result
    /// the row writes after those it reads: a chat image's encoding writes its encoder input's
    /// output, and a video request's text encoding its conditioning, which a video request that
    /// names no digests has no key for and keeps private.
    pub(crate) fn commit_encoding(
        &self,
        request: &mut Request,
        (batch, row): (&Batch, usize),
        groups: &[WorkerGroup],
    ) -> Option<runtime::Request> {
        let written = batch
            .media
            .get(row)
            .and_then(|media| media.results.get(media.reads as usize));
        let (Some(&at), Some(owner)) = (written, request.owner_on(batch.phase)) else {
            return None;
        };
        let class = if request.is_video() {
            self.conditioning_class(groups)
        } else {
            self.encoder_class(request, groups)
        }?;
        let (index, rows) = self.encoding_at(request, class, at, groups)?;
        let &(key, _) = request.encodings.get(index)?;
        // A video request names the digests that key its conditioning to have it found and kept.
        let writes = request.generation().is_none_or(|input| input.cache.write);
        if let Some(table) = request.blocks.get_mut(usize::from(class))
            && writes
        {
            table.shared = table.shared.max(at + 1);
        }
        // An output not committed is the request's own: `cache` counts no hold of it.
        if !writes {
            request.unpinned.push(key);
        }
        if let Some(entry) = request.encodings.get_mut(index) {
            entry.1 = Some(at);
        }
        let tokens = u16::try_from(rows).unwrap_or(u16::MAX);
        writes.then(|| commit(owner, class, at, key, tokens))
    }

    /// The `Release`s of what no later call of `request` reads, after the output of its call in
    /// `batch`'s row `row` (SD §4.4 "Giving up behind", `S/kv_cache/mod.rs:2427-2445`): a sliding
    /// window the blocks entirely before `visible - window`; a recurrent-state table, once the
    /// call wrote the checkpoint ahead of its base, its base, unless a refill continues from the
    /// base, which then stays while a checkpoint the target wrote is given up. A full-attention
    /// table keeps everything. What the request gives up is told to the policy through `told`.
    pub(crate) fn give_up_behind(
        &mut self,
        request: &mut Request,
        (batch, row): (&Batch, usize),
        (groups, config): (&[WorkerGroup], &SchedulerConfig),
        told: &mut Vec<Event>,
    ) -> Vec<runtime::Request> {
        let Some(owner) = kv_owner(request) else {
            return Vec::new();
        };
        let visible = request
            .num_computed_tokens
            .saturating_sub(request.in_flight(Phase::Decode));
        let written = kept_by(request, (batch, row), groups);
        let lagging =
            config.draft_tokens > 0 && request.num_draft_tokens < request.num_computed_tokens;

        let (mut asks, mut unpinned) = (Vec::new(), Vec::new());
        let mut tables = request.blocks.clone();
        for (index, (&(kind, block), table)) in shapes(request, groups)
            .iter()
            .zip(tables.iter_mut())
            .enumerate()
        {
            match kind {
                CacheGroupKind::SlidingWindow { window } => {
                    let bound = (visible.saturating_sub(window) / block).min(table.end);
                    if bound > table.first {
                        let behind = table.first..bound;
                        let given = (owner, index, behind, table.shared);
                        give_up(request, given, &mut asks, &mut unpinned);
                        table.first = bound;
                        table.shared = table.shared.max(bound);
                    }
                }
                CacheGroupKind::State => {
                    let [Some(base), Some(ahead)] = checkpoints(table) else {
                        continue;
                    };
                    if written != Some(ahead) {
                        continue;
                    }
                    let continues = batch.phase != Phase::Refill
                        && lagging
                        && request.num_draft_tokens == (base + 1) * block;
                    if continues {
                        let given = (owner, index, ahead..ahead + 1, table.shared);
                        give_up(request, given, &mut asks, &mut unpinned);
                        table.end = base + 1;
                        table.asked = table.end;
                        table.shared = table.shared.min(table.end);
                    } else {
                        let given = (owner, index, base..base + 1, table.shared);
                        give_up(request, given, &mut asks, &mut unpinned);
                        table.first = ahead;
                        table.shared = table.shared.max(ahead);
                    }
                }
                CacheGroupKind::Full | CacheGroupKind::Draft => {}
            }
        }

        request.blocks = tables;
        self.unpin(Some(request), &unpinned, groups, told);
        asks
    }

    /// What the write policy stores once request `asked` was answered `completion` (SD §4.4 "The
    /// write policy", `S/kv_cache/offload.rs:57-80`, `:880-904`): a `Committed` block of a
    /// full-attention class, or of a sliding window that lies in the window before an aligned
    /// boundary, is stored to the host by a group whose memory holds its device copy (the first
    /// that stamped the `Commit`, whose table held the entry), when that group's deployment has a
    /// host tier, the step's values keep it on and the class has a host region; a host `Stored`
    /// is stored on to the remote store when the deployment has one. A block is offered once: a
    /// `Full` or a `Duplicate` stores nothing.
    ///
    /// `holder` is the request of the owner `asked` acts for: a block whose `Committed` arrives
    /// after its owner's `Detach` (a migratable request's tails and last blocks, committed as it
    /// leaves) is stored as any other, its request gone: a sliding window's block is then judged
    /// by the aligned boundaries alone, its prompt's being unknown.
    pub(crate) fn write_policy(
        &self,
        (asked, completion): (&runtime::Request, &Completion),
        holder: Option<&Request>,
        groups: &[WorkerGroup],
        budget: &Budget,
    ) -> Option<runtime::Request> {
        match (asked, completion) {
            (
                runtime::Request::Commit {
                    class,
                    at,
                    key,
                    after,
                    ..
                },
                Completion::Committed,
            ) => {
                let &(stamped, _) = after.first()?;
                let capacity = &group(groups, stamped)?.capacity;
                let kv = capacity.kv_cache.as_ref()?;
                self.classes.get(usize::from(*class))?.host?;
                let shapes = &capacity.cache_groups;
                let &(kind, block) = shapes.get(usize::from(*class))?;
                let prompt = holder.map(|request| request.num_prompt_tokens);
                // The step's values switch the host tier off where the epoch opened it.
                let tier = kv.host_tier && budget.host_tier;
                let stores = match kind {
                    CacheGroupKind::Full => tier,
                    CacheGroupKind::SlidingWindow { window } => {
                        tier && before_boundary(shapes, kv, prompt, *at, block, window)
                    }
                    CacheGroupKind::State | CacheGroupKind::Draft => false,
                };
                stores.then(|| runtime::Request::Store {
                    group: stamped,
                    keys: vec![*key],
                    tier: Tier::Host,
                })
            }
            (
                runtime::Request::Store {
                    group: storing,
                    keys,
                    tier: Tier::Host,
                },
                Completion::Stored { .. },
            ) => {
                let remote = group(groups, *storing)
                    .and_then(|group| group.capacity.kv_cache.as_ref())
                    .is_some_and(|kv| kv.mooncake.is_some());
                (remote && !keys.is_empty()).then(|| runtime::Request::Store {
                    group: *storing,
                    keys: keys.clone(),
                    tier: Tier::Remote,
                })
            }
            _ => None,
        }
    }

    /// The `Release` of `request`'s whole draft table: its draft cache is empty from now on, and
    /// its table starts again at the first block its next call needs (SD §4.2). Returns the
    /// release, per region the units it frees once the buffer pool reclaims them, and the blocks
    /// given up; `None` for a request with no draft table. What the request gives up is told to
    /// the policy through `told`.
    pub(crate) fn give_up_draft(
        &mut self,
        request: &mut Request,
        groups: &[WorkerGroup],
        told: &mut Vec<Event>,
    ) -> Option<(Vec<runtime::Request>, Vec<u32>, u32)> {
        let (owner, group) = (kv_owner(request)?, kv_group(request, groups)?);
        let (index, _) = draft_class(request, groups)?;
        let table = *request.blocks.get(index)?;
        let mut freed = vec![0; self.regions.len()];
        let (mut asks, mut unpinned) = (Vec::new(), Vec::new());
        let held = table.first..table.end;
        give_up(
            request,
            (owner, index, held, table.shared),
            &mut asks,
            &mut unpinned,
        );
        if let Ok(class) = u8::try_from(index) {
            let blocks = table.end - table.first;
            self.charge_units(&mut freed, (group, class), (blocks, 0), groups);
        }

        request.blocks[index] = Blocks::default();
        request.num_draft_tokens = 0;
        self.unpin(Some(request), &unpinned, groups, told);
        Some((asks, freed, table.end.saturating_sub(table.first)))
    }
}

/// The class of a token request's draft cache and the tokens of one of its blocks; `None` for a
/// model whose drafter keeps no cache of its own.
pub(crate) fn draft_class(request: &Request, groups: &[WorkerGroup]) -> Option<(usize, u32)> {
    let shapes = shapes(request, groups);
    let class = shapes.iter().position(|(kind, _)| kind.is_draft())?;
    Some((class, shapes[class].1))
}

/// The `running` requests whose draft blocks a shortage of `region` (an index) gives up first,
/// in the order it gives them up (SD §4.4 "Drafts first", `S/speculation.rs:474-525`): those
/// running with no call in flight whose draft table holds blocks in the region and asks for none
/// it does not hold, the fewest tokens left to generate first.
pub(crate) fn draft_order<'a>(
    running: impl IntoIterator<Item = &'a Request>,
    region: usize,
    groups: &[WorkerGroup],
) -> Vec<RequestId> {
    let mut order: Vec<(u64, RequestId)> = running
        .into_iter()
        .filter(|request| request.status == RequestStatus::Running && request.idle())
        .filter_map(|request| {
            let (class, _) = draft_class(request, groups)?;
            let group = kv_worker(request, groups)?;
            let class_regions = group.regions(u8::try_from(class).ok()?);
            if !class_regions.iter().any(|&r| usize::from(r) == region) {
                return None;
            }
            let table = request.blocks.get(class)?;
            let input = request.generation()?;
            let left = u64::try_from(input.max_und_tokens)
                .unwrap_or(u64::MAX)
                .saturating_sub(u64::from(request.generated()));
            (table.first < table.end && table.asked == table.end).then_some((left, request.id))
        })
        .collect();
    order.sort_unstable();
    order.into_iter().map(|(_, id)| id).collect()
}

#[cfg(test)]
mod tests {
    //! A request's tables as a scheduler's admissions, calls and answers change them, over a
    //! session of one worker group whose cache classes are the model's cache groups, each a
    //! one-unit block in one device region with a host region beside it, answered by hand on a
    //! real queue pair.

    use std::collections::BTreeMap;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use crate::batch::cache::CacheGroupKind;
    use crate::batch::{Batch, BatchId, Columns, Phase, Shape, Variant};
    use proto::config::{KvCacheConfig, SchedulerConfig, WorkerGroupConfig};
    use proto::request::{Event, FinishReason, GenerationLimits, GenerationRequest, RequestId};
    use storage::runtime::{self, Class, Completion, GroupId, OwnerId, Region, Tier};
    use storage::{Key, QueuePair};

    use storage::key::{Digest, encoder_output};

    /// The event window a test's consumer states in its welcome.
    const WINDOW: std::num::NonZeroU32 = std::num::NonZeroU32::new(64).unwrap();

    use super::super::super::tests::{capacity, idle_policy};
    use super::super::super::{Blocks, Executor, Output, Request, RequestStatus, Scheduler};
    use super::super::{DEVICE, HOST, REMOTE};
    use super::{
        ahead, commit, count_asked, draft_order, draft_tokens, fit_chunk, holds_blocks_for, run,
        settled,
    };

    const A: GroupId = GroupId(0);

    /// What the executor was handed to stamp.
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

    struct Session {
        scheduler: Scheduler,
        pool: QueuePair,
        stamped: Arc<Mutex<Vec<runtime::Request>>>,
    }

    /// A session whose group A serves prefills and decodes of a model of `shapes`: regions 4
    /// (the device's KV units), 5 (the host's) and 6 (slot rows), one class per cache group and
    /// the class of trajectories after them.
    fn serving(shapes: &[(CacheGroupKind, u32)], kv: KvCacheConfig) -> Session {
        layout(shapes, kv, 0)
    }

    /// A session as [`serving`] states it, with `sized` sized classes before the trajectories':
    /// with 14, the encoder-output class and the product classes.
    fn layout(shapes: &[(CacheGroupKind, u32)], kv: KvCacheConfig, sized: usize) -> Session {
        let region = |name: &str, tier| Region {
            name: name.to_owned(),
            tier,
            space: name.to_owned(),
            unit_bytes: 1 << 20,
            part_bytes: 1 << 20,
            reserved: 1,
            units: 256,
            limit: 256,
            segments: Vec::new(),
            memory: String::new(),
        };
        let class = |units| Class {
            units,
            host: Some((1, 0)),
            state_in_slot: false,
            tensors: [0; 32],
            encoding: [0; 32],
        };
        let mut classes: Vec<Class> = shapes.iter().map(|_| class(1)).collect();
        classes.extend((0..=sized).map(|_| class(0)));
        let layout = runtime::Open {
            oldest: runtime::VERSIONS.oldest(),
            newest: Some(runtime::VERSIONS.newest()),
            epoch: 1,
            trial: false,
            regions: vec![
                region("kv", Tier::Device),
                region("host", Tier::Host),
                region("slots", Tier::Device),
            ],
            trajectory: u8::try_from(shapes.len() + sized).unwrap(),
            groups: vec![runtime::Group {
                id: A,
                memory: "memory".to_owned(),
                slots: 2,
                regions: classes.iter().map(|_| vec![0]).collect(),
                receives: Vec::new(),
                parts: Vec::new(),
            }],
            classes,
            schedulers: vec![vec![A]],
        };
        let opened = runtime::Opened {
            regions: vec![(4, 256), (5, 256), (6, 256)],
            segments: Vec::new(),
            limits: Vec::new(),
            registers: vec![runtime::Registers {
                completed: vec![0],
                capacity: vec![(4, 1), (5, 4), (6, 7)],
            }],
            session: 0,
            version: None,
        };

        let pool = QueuePair::create(1 << 16, 10).unwrap();
        let engine = QueuePair::open(pool.duplicate().unwrap(), true).unwrap();
        let groups = WorkerGroupConfig::data_parallel(
            &["localhost".to_owned()],
            proto::config::DeviceKind::Cpu,
            1,
            1,
            2,
            BTreeMap::new(),
        );
        let mut stated = capacity();
        stated.phases = [Phase::Prefill, Phase::Decode, Phase::VisionEncode]
            .iter()
            .fold(0, |phases, phase| phases | (1 << phase.index()));
        stated.feature_bytes = 1 << 16;
        stated.kv_cache = Some(kv);
        stated.cache_groups = shapes.to_vec();
        stated.eos_token_ids = vec![2];

        let stamped = Arc::new(Mutex::new(Vec::new()));
        let (output, _) = crossbeam_channel::unbounded();
        // As many running requests as the group holds slots, which the startup checks require
        // of a deployment with a KV cache.
        let config = SchedulerConfig {
            max_running_requests: stated.slots as usize,
            ..SchedulerConfig::default()
        };
        let mut scheduler = Scheduler::new(
            config,
            GenerationLimits::default(),
            &layout,
            &opened,
            0,
            engine,
            vec![(groups[0].clone(), stated)],
            idle_policy(),
            scheduling::Parameters::default(),
            Box::new(Stamps(Arc::clone(&stamped))),
            output,
            Arc::new(crate::batch::Generations::default()),
            crate::scheduler::tests::tuned(),
        )
        .unwrap();
        scheduler.tune(|tuning| tuning.chunked_prefill_size = std::num::NonZeroUsize::new(8));
        let mut session = Session {
            scheduler,
            pool,
            stamped,
        };

        // The scheduler's first request is its startup `List` (SD §4.5): answered with an empty
        // `Listing`, every case starts with an empty cache and nothing outstanding.
        let listing = Completion::Listing { keys: Vec::new() };
        session.answer(&[], &[listing]);
        session
    }

    fn full(block: u32) -> (CacheGroupKind, u32) {
        (CacheGroupKind::Full, block)
    }

    fn window(window: u32, block: u32) -> (CacheGroupKind, u32) {
        (CacheGroupKind::SlidingWindow { window }, block)
    }

    impl Session {
        /// Token request `id` of `prompt`, placed and attached on group A as owner `id`, with its
        /// keys set and its tables sized to the session's classes.
        fn request(&mut self, id: u64, prompt: Vec<u32>) -> RequestId {
            let input = GenerationRequest {
                trace: 0,
                request_id: RequestId(id),
                prompt_token_ids: prompt,
                multimodal_inputs: Default::default(),
                sampling: Default::default(),
                max_und_tokens: 16,
                stop_strings: Vec::new(),
                stop_token_ids: Vec::new(),
                priority: 0,
                cache: Default::default(),
                include_stop_token: false,
                generated_token_ids: Vec::new(),
                adapter: 0,
                hand_off: false,
                lend: None,
            };
            let mut request = Request::new(input.into(), 1, Instant::now(), WINDOW, 0);
            request.placement[Phase::Prefill.index()] = Some(A);
            request.placement[Phase::Decode.index()] = Some(A);
            request.placement[Phase::VisionEncode.index()] = Some(A);
            request.attachments = vec![(A, OwnerId(id), 1, true)];
            request.blocks = vec![Blocks::default(); self.scheduler.kv.classes().len()];
            request.keys = self.scheduler.kv.keys_of(&request, &self.scheduler.groups);
            self.scheduler.owners.insert(OwnerId(id), RequestId(id));
            self.scheduler.requests.insert(RequestId(id), request);
            RequestId(id)
        }

        fn row(&self, id: RequestId) -> &Request {
            &self.scheduler.requests[&id]
        }

        fn row_mut(&mut self, id: RequestId) -> &mut Request {
            self.scheduler.requests.get_mut(&id).unwrap()
        }

        /// The buffer pool lists `keys`, each of `class` with its tiers, on the device of group A.
        fn list(&mut self, keys: &[(Key, u8, u8)]) {
            let keys = keys
                .iter()
                .map(|&(key, class, tiers)| (key, class, 4, tiers, 1, 0))
                .collect();
            self.answer(&[runtime::Request::List], &[Completion::Listing { keys }]);
        }

        /// Writes `requests`, answers each with its completion in order, and applies every
        /// completion read as an admission does; returns the requests the answers call for.
        fn answer(
            &mut self,
            requests: &[runtime::Request],
            completions: &[Completion],
        ) -> Vec<runtime::Request> {
            for request in requests {
                self.scheduler.submit(request.clone());
            }
            self.scheduler.post().unwrap();
            let mut record = Vec::new();
            let mut numbers = Vec::new();
            while self.pool.next_request(&mut record).unwrap() {
                numbers.push(runtime::Request::decode(&record).unwrap().0);
            }
            assert_eq!(
                numbers.len(),
                completions.len(),
                "one answer per request written"
            );
            for (number, completion) in numbers.into_iter().zip(completions) {
                record.clear();
                completion.encode(number, &mut record);
                assert!(self.pool.complete(&record).unwrap());
            }

            let mut follow = Vec::new();
            while let Some((_, owner, request, completion)) =
                self.scheduler.next_completion().unwrap()
            {
                if let Some(id) = owner.and_then(|owner| self.scheduler.owners.get(&owner).copied())
                {
                    follow.extend(self.scheduler.filled(id, &request, &completion));
                }
            }
            follow
        }

        fn stamped(&self) -> Vec<runtime::Request> {
            std::mem::take(&mut self.stamped.lock().unwrap())
        }
    }

    fn pin(owner: u64, class: u8, at: u32, keys: &[Key]) -> runtime::Request {
        runtime::Request::Pin {
            owner: OwnerId(owner),
            class,
            at,
            keys: keys.to_vec(),
        }
    }

    fn allocate(owner: u64, class: u8, at: u32, count: u32) -> runtime::Request {
        runtime::Request::Allocate {
            owner: OwnerId(owner),
            class,
            at,
            count,
            units: 0,
        }
    }

    /// An `Allocate` of one block of a sized class, of `units` units.
    fn sized(owner: u64, class: u8, at: u32, units: u32) -> runtime::Request {
        runtime::Request::Allocate {
            owner: OwnerId(owner),
            class,
            at,
            count: 1,
            units,
        }
    }

    fn release(owner: u64, class: u8, at: u32, count: u32) -> runtime::Request {
        runtime::Request::Release {
            owner: OwnerId(owner),
            class,
            at,
            count,
            after: Vec::new(),
        }
    }

    fn pinned(blocks: u32) -> Completion {
        Completion::Pinned {
            blocks: (0..blocks).map(|unit| vec![(4, unit + 1, 1)]).collect(),
            host: 0,
        }
    }

    fn allocated(count: u32) -> Completion {
        Completion::Allocated {
            runs: (0..count).map(|unit| (4, unit + 100, 1)).collect(),
            evicted: Vec::new(),
        }
    }

    /// A batch of `phase` whose one row kept the recurrent state at `checkpoint` (0: none).
    fn batch(phase: Phase, checkpoint: i32) -> Batch {
        Batch {
            id: BatchId(1),
            group: A,
            phase,
            kernels: 0,
            policy: 0,
            revision: 0,
            copy_budget: proto::config::DEFAULT_COPY_BUDGET,
            shape: Shape::decode(1, Variant::DecodeGreedy),
            requests: Vec::new(),
            columns: Columns {
                checkpoints: vec![checkpoint],
                ..Columns::default()
            },
            media: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    fn hex(key: &Key) -> String {
        key.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn keys_chain_the_prompt_under_the_seed_and_the_isolation_key() {
        // SD §4.3: zero seed, group 0, blocks of 4 tokens, prompt 1..=8.
        let mut session = serving(&[full(4)], KvCacheConfig::default());
        let id = session.request(1, (1..=8).collect());
        let keys: Vec<String> = session.row(id).keys[0].iter().map(hex).collect();
        assert_eq!(
            keys,
            [
                "d82404dd8bfea249aec17a3856952089f9a004f8c56e533b9f9cbae86f47b421",
                "ba86d1709ff8ac538e9aabba75a7f8838f16d3c622ce9b286bd63301cb520c37"
            ]
        );

        let mut isolated = session.row(id).clone();
        if let proto::request::Request::Generation(input) = &mut isolated.input {
            input.cache.isolation_key = Some(7);
        }
        let keys: Vec<String> = session
            .scheduler
            .kv
            .keys_of(&isolated, &session.scheduler.groups)[0]
            .iter()
            .map(hex)
            .collect();
        assert_eq!(
            keys,
            [
                "e96a1c1f75c37f1ee0012bb59a32daabf95e1086c1e0585b2df75f575abd99d9",
                "2fa8b480f6a0e7db59c2d7d139b2eb3ed0e146ea90f851648a545756422560d5"
            ]
        );

        // A draft block's key covers the token after it: seven tokens complete one block.
        let mut session = serving(
            &[full(4), (CacheGroupKind::Draft, 4)],
            KvCacheConfig::default(),
        );
        let id = session.request(1, (1..=8).collect());
        assert_eq!(session.row(id).keys[1].len(), 1);

        // A hit of 4 tokens whose draft block is on the device starts the draft cache there.
        let keys = session.row(id).keys.clone();
        session.list(&[(keys[0][0], 0, DEVICE), (keys[1][0], 1, DEVICE)]);
        let asks = session.scheduler.admission_requests(id);
        assert_eq!(
            asks[..2],
            [pin(1, 0, 0, &keys[0][..1]), pin(1, 1, 0, &keys[1][..1])]
        );
        session.answer(&asks, &[pinned(1), pinned(1), allocated(3)]);
        assert!(settled(
            session.row(id),
            &session.scheduler.groups,
            &session.scheduler.budget
        ));
        assert_eq!(draft_tokens(session.row(id), &session.scheduler.groups), 4);
    }

    #[test]
    fn the_hit_is_the_deepest_boundary_every_class_accepts_read_by_pins_then_loads() {
        // A full-attention class and a window of one block, blocks of 4 tokens; a 25-token
        // prompt may reuse up to 24 tokens.
        let mut session = serving(&[full(4), window(4, 4)], KvCacheConfig::default());
        let id = session.request(1, (1..=25).collect());
        let (full_keys, window_keys) = (
            session.row(id).keys[0].clone(),
            session.row(id).keys[1].clone(),
        );

        // Full blocks 0 to 4 on the device; the window holds block 3 but not block 4: 20 is
        // rejected and the search goes on to 16.
        let mut listed: Vec<(Key, u8, u8)> =
            full_keys[..5].iter().map(|key| (*key, 0, DEVICE)).collect();
        listed.push((window_keys[3], 1, DEVICE));
        session.list(&listed);
        assert_eq!(
            session.scheduler.kv.expected(
                session.row(id),
                &session.scheduler.groups,
                &session.scheduler.budget
            ),
            16
        );

        let asks = session.scheduler.admission_requests(id);
        // From the hit, the first chunk of the prompt left (8 tokens) and one block beyond.
        assert_eq!(
            asks,
            [
                pin(1, 0, 0, &full_keys[..4]),
                pin(1, 1, 3, &window_keys[3..4]),
                allocate(1, 0, 4, 3),
                allocate(1, 1, 4, 3),
            ]
        );
        assert_eq!(session.row(id).num_cached_tokens, 16);

        // With the window's block 4 on the host, 20 is accepted: what the device lacks is loaded.
        listed.push((window_keys[4], 1, HOST));
        listed[4].2 = HOST;
        session.list(&listed);
        let asks = session.scheduler.admission_requests(id);
        let load = |class, at, keys: &[Key]| runtime::Request::Load {
            owner: OwnerId(1),
            class,
            at,
            keys: keys.to_vec(),
        };
        assert_eq!(
            asks,
            [
                pin(1, 0, 0, &full_keys[..4]),
                load(0, 4, &full_keys[4..5]),
                load(1, 4, &window_keys[4..5]),
                allocate(1, 0, 5, 3),
                allocate(1, 1, 5, 3),
            ]
        );
        assert_eq!(session.row(id).num_cached_tokens, 20);
    }

    #[test]
    fn a_partial_last_block_is_cloned_and_its_allocations_wait_for_the_answers() {
        let mut session = serving(&[full(4)], KvCacheConfig::default());
        let id = session.request(1, (1..=11).collect());
        let keys = session.row(id).keys[0].clone();
        let tail = session
            .scheduler
            .kv
            .tail_key(session.row(id), (0, 10), &session.scheduler.groups)
            .unwrap();
        session.list(&[
            (keys[0], 0, DEVICE),
            (keys[1], 0, DEVICE),
            (tail, 0, DEVICE),
        ]);

        // The hit reaches token 10, past the reuse limit of 8, in a clone of the tail.
        let asks = session.scheduler.admission_requests(id);
        let clone = runtime::Request::Clone {
            owner: OwnerId(1),
            class: 0,
            at: 2,
            key: tail,
        };
        assert_eq!(asks, [pin(1, 0, 0, &keys[..2]), clone.clone()]);
        assert_eq!(session.row(id).num_cached_tokens, 10);

        // The pins and the clone arrive; the request holds no block for its first chunk yet.
        session.answer(&asks[..1], &[pinned(2)]);
        session.scheduler.filled(id, &clone, &Completion::Cloned);
        assert!(!settled(
            session.row(id),
            &session.scheduler.groups,
            &session.scheduler.budget
        ));

        // The hit is final at 10; the blocks after the clone are allocated.
        assert_eq!(
            session.scheduler.missing_requests(id),
            [allocate(1, 0, 3, 1)]
        );
        session.answer(&[allocate(1, 0, 3, 1)], &[allocated(1)]);
        assert!(settled(
            session.row(id),
            &session.scheduler.groups,
            &session.scheduler.budget
        ));
        assert_eq!(session.row(id).num_cached_tokens, 10);
    }

    #[test]
    fn answers_that_fall_short_make_the_hit_final_at_what_they_hold() {
        let mut session = serving(&[full(4)], KvCacheConfig::default());
        let id = session.request(1, (1..=17).collect());
        let keys = session.row(id).keys[0].clone();
        let listed: Vec<(Key, u8, u8)> = keys[..4].iter().map(|key| (*key, 0, DEVICE)).collect();
        session.list(&listed);

        let asks = session.scheduler.admission_requests(id);
        assert_eq!(asks, [pin(1, 0, 0, &keys[..4]), allocate(1, 0, 4, 2)]);

        // Two of the four blocks were evicted since `cache` saw them.
        session.answer(&asks, &[pinned(2), allocated(2)]);
        assert!(!settled(
            session.row(id),
            &session.scheduler.groups,
            &session.scheduler.budget
        ));

        // The hit falls to 8; the positions between it and the blocks allocated are allocated.
        assert_eq!(
            session.scheduler.missing_requests(id),
            [allocate(1, 0, 2, 2)]
        );
        assert_eq!(session.row(id).num_cached_tokens, 8);
        session.answer(&[allocate(1, 0, 2, 2)], &[allocated(2)]);
        assert!(settled(
            session.row(id),
            &session.scheduler.groups,
            &session.scheduler.budget
        ));
        assert_eq!(
            session.row(id).blocks[0],
            Blocks {
                first: 0,
                shared: 2,
                end: 6,
                asked: 6
            }
        );
    }

    #[test]
    fn a_pin_that_falls_short_reads_what_it_lost_from_the_host_or_the_remote_store() {
        let load = |at, keys: &[Key]| runtime::Request::Load {
            owner: OwnerId(1),
            class: 0,
            at,
            keys: keys.to_vec(),
        };
        // `cache` believes the first ten blocks are on the device, and stored in the remote store;
        // block 12 is on the host.
        let admission = |session: &mut Session| {
            let id = session.request(1, (1..=1000).collect());
            let keys = session.row(id).keys[0].clone();
            let mut listed: Vec<(Key, u8, u8)> = keys[..10]
                .iter()
                .map(|key| (*key, 0, DEVICE | REMOTE))
                .collect();
            listed.push((keys[12], 0, HOST));
            session.list(&listed);
            let asks = session.scheduler.admission_requests(id);
            assert_eq!(asks, [pin(1, 0, 0, &keys[..10]), allocate(1, 0, 10, 2)]);
            session.row_mut(id).status = RequestStatus::WaitingForRemoteKvs;
            (id, keys, asks)
        };
        let held = |blocks: u32, host: u32| Completion::Pinned {
            blocks: (0..blocks).map(|unit| vec![(4, unit + 1, 1)]).collect(),
            host,
        };

        // Eight of the blocks left the device, and the host holds three of them: those three are
        // loaded at the positions the pin lost.
        let mut session = serving(&[full(64)], KvCacheConfig::default());
        let (_, keys, asks) = admission(&mut session);
        let follow = session.answer(&asks, &[held(2, 3), allocated(2)]);
        assert_eq!(follow, [load(2, &keys[2..5])]);

        // None of them is on the host: the remote store, which holds all eight, is asked for
        // them, and what it found is loaded at the positions the pin lost.
        let mut session = serving(&[full(64)], KvCacheConfig::default());
        let (id, keys, asks) = admission(&mut session);
        let follow = session.answer(&asks, &[held(2, 0), allocated(2)]);
        let [
            runtime::Request::Prefetch {
                class: 0,
                keys: wanted,
                ..
            },
        ] = &follow[..]
        else {
            panic!("one prefetch from the remote store, not {follow:?}");
        };
        assert_eq!(wanted, &keys[2..10]);
        // It found six, on the host now, in blocks for which it evicted block 12's host copy.
        let found = Completion::Prefetched {
            found: 6,
            evicted: vec![keys[12]],
        };
        session.answer(&follow, std::slice::from_ref(&found));
        assert_eq!(
            session.scheduler.kv.expected_hit(&keys[2..13], A),
            [&[HOST | REMOTE; 6][..], &[REMOTE, REMOTE, 0, 0, 0]].concat()
        );
        assert_eq!(
            session.scheduler.filled(id, &follow[0], &found),
            [load(2, &keys[2..8])]
        );

        // Three blocks lost, 192 tokens, are short of the policy's 256: they are computed.
        let mut session = serving(&[full(64)], KvCacheConfig::default());
        let (_, _, asks) = admission(&mut session);
        assert!(
            session
                .answer(&asks, &[held(7, 0), allocated(2)])
                .is_empty()
        );
    }

    #[test]
    fn a_window_that_falls_short_gives_up_the_blocks_it_no_longer_reads() {
        let mut session = serving(&[full(4), window(4, 4)], KvCacheConfig::default());
        let id = session.request(1, (1..=17).collect());
        let (full_keys, window_keys) = (
            session.row(id).keys[0].clone(),
            session.row(id).keys[1].clone(),
        );
        let mut listed: Vec<(Key, u8, u8)> =
            full_keys[..4].iter().map(|key| (*key, 0, DEVICE)).collect();
        listed.push((window_keys[3], 1, DEVICE));
        session.list(&listed);

        let asks = session.scheduler.admission_requests(id);
        assert_eq!(session.row(id).num_cached_tokens, 16);
        // The full class's pin falls short at 8, where the window's block 1 was never pinned:
        // the hit ends at 0, and every read-only block goes.
        session.answer(&asks, &[pinned(2), pinned(1), allocated(2), allocated(2)]);
        let missing = session.scheduler.missing_requests(id);
        assert_eq!(session.row(id).num_cached_tokens, 0);
        assert_eq!(
            missing,
            [
                release(1, 0, 0, 2),
                allocate(1, 0, 0, 4),
                release(1, 1, 3, 1),
                allocate(1, 1, 0, 4),
            ]
        );
    }

    #[test]
    fn a_remote_hit_long_enough_is_prefetched_before_the_chunk_is_allocated() {
        let mut session = serving(&[full(64)], KvCacheConfig::default());
        let id = session.request(1, (1..=1000).collect());
        let keys = session.row(id).keys[0].clone();
        let tiers = |remote: usize| -> Vec<(Key, u8, u8)> {
            let device = keys[..2].iter().map(|key| (*key, 0, DEVICE));
            let stored = keys[2..remote].iter().map(|key| (*key, 0, REMOTE));
            device.chain(stored).collect()
        };

        // The remote store deepens the hit of 128 tokens to 640: a prefetch, and no allocation.
        session.list(&tiers(10));
        let asks = session.scheduler.admission_requests(id);
        assert_eq!(asks.len(), 2);
        assert_eq!(asks[0], pin(1, 0, 0, &keys[..2]));
        let runtime::Request::Prefetch {
            class,
            keys: fetched,
            ..
        } = &asks[1]
        else {
            panic!("a prefetch, not {:?}", asks[1]);
        };
        assert_eq!((*class, &fetched[..]), (0, &keys[2..10]));

        // 192 tokens more is short of the policy's 256: the chunk is allocated instead.
        session.list(&tiers(5));
        let asks = session.scheduler.admission_requests(id);
        assert_eq!(asks, [pin(1, 0, 0, &keys[..2]), allocate(1, 0, 2, 1 + 1)]);
    }

    #[test]
    fn a_running_request_asks_ahead_and_writes_only_blocks_it_holds_writable() {
        let mut session = serving(&[full(4)], KvCacheConfig::default());
        let id = session.request(1, (1..=10).collect());
        let request = session.row_mut(id);
        request.status = RequestStatus::Running;
        request.num_computed_tokens = 10;
        request.blocks[0] = Blocks {
            first: 0,
            shared: 0,
            end: 3,
            asked: 3,
        };

        // The next decode writes token 10 into block 2, which it holds; one more block is the
        // low watermark, which the call does not write.
        assert_eq!(
            ahead(session.row(id), &session.scheduler.kv_step()),
            [(0, 3, 1, 0)]
        );
        assert!(holds_blocks_for(
            session.row(id),
            (Phase::Decode, 1),
            &session.scheduler.groups,
            &session.scheduler.config
        ));
        assert!(!holds_blocks_for(
            session.row(id),
            (Phase::Prefill, 3),
            &session.scheduler.groups,
            &session.scheduler.config
        ));
        count_asked(session.row_mut(id), 0, 3, 1);
        assert!(ahead(session.row(id), &session.scheduler.kv_step()).is_empty());

        // A block marked read-only by its commit is not written again.
        session.row_mut(id).blocks[0].shared = 3;
        assert!(!holds_blocks_for(
            session.row(id),
            (Phase::Decode, 1),
            &session.scheduler.groups,
            &session.scheduler.config
        ));
    }

    /// A request whose calls in flight may sample the last token its limit allows makes no
    /// further call, and asks for no block ahead of one: a prompt that fills its blocks and may
    /// generate one token is not short of the block a decode would write (SD §3.3 `Decode`, §3.6).
    #[test]
    fn a_request_whose_calls_in_flight_reach_its_limit_asks_for_nothing_ahead() {
        let mut session = serving(&[full(4)], KvCacheConfig::default());
        let id = session.request(1, (1..=8).collect());
        let limit = |session: &mut Session, tokens: usize| {
            if let proto::request::Request::Generation(input) = &mut session.row_mut(id).input {
                input.max_und_tokens = tokens;
            }
        };
        // The prompt's last chunk, which samples the first token, is in flight; the blocks hold
        // the prompt and its low watermark.
        let request = session.row_mut(id);
        request.status = RequestStatus::Running;
        request.num_computed_tokens = 8;
        request.scheduled[Phase::Prefill.index()] = 1;
        request.blocks[0] = Blocks {
            first: 0,
            shared: 0,
            end: 3,
            asked: 3,
        };
        limit(&mut session, 1);
        assert!(ahead(session.row(id), &session.scheduler.kv_step()).is_empty());

        // Allowed a second token, its decode writes token 8 into block 2, which it holds; the
        // block after it is the low watermark.
        limit(&mut session, 2);
        assert_eq!(
            ahead(session.row(id), &session.scheduler.kv_step()),
            [(0, 3, 1, 0)]
        );
        session.row_mut(id).blocks[0].end = 2;
        session.row_mut(id).blocks[0].asked = 2;
        assert_eq!(
            ahead(session.row(id), &session.scheduler.kv_step()),
            [(0, 2, 2, 1)]
        );
    }

    /// The low watermark of an attention class stays within the blocks a sequence of the group's
    /// `max_model_len` holds, the positions every rank's page table holds per request slot
    /// (SD §4.4): a request whose first chunk or next decode writes its last such block asks for
    /// none beyond it. A model length of one block is a warm-up request's on a deployment whose
    /// length is its block size.
    #[test]
    fn the_low_watermark_stays_within_the_blocks_of_the_model_s_length() {
        let mut session = serving(&[full(4)], KvCacheConfig::default());
        session.scheduler.groups[0].capacity.max_model_len = 4;
        let one = session.request(1, vec![1]);
        assert_eq!(
            session.scheduler.admission_requests(one),
            [allocate(1, 0, 0, 1)]
        );

        // A model length of two blocks: a five-token prompt's chunk writes both.
        session.scheduler.groups[0].capacity.max_model_len = 8;
        let id = session.request(2, (1..=5).collect());
        assert_eq!(
            session.scheduler.admission_requests(id),
            [allocate(2, 0, 0, 2)]
        );

        // Running at six tokens, its next decode writes token 6 into block 1, its last: nothing
        // is asked ahead, and its units to its end are the two blocks it holds.
        let request = session.row_mut(id);
        request.status = RequestStatus::Running;
        request.num_computed_tokens = 6;
        request.token_ids = (1..=6).collect();
        request.blocks[0] = Blocks {
            first: 0,
            shared: 0,
            end: 2,
            asked: 2,
        };
        assert!(ahead(session.row(id), &session.scheduler.kv_step()).is_empty());
        let running = [session.row(id).clone()];
        let left = session.scheduler.kv.units_to_end(
            running.iter(),
            &session.scheduler.groups,
            &session.scheduler.budget,
        );
        assert!(left.iter().all(|&units| units == 0), "{left:?}");
    }

    /// A request the whole cache cannot hold ends with `CacheCapacity`, at the length the cache
    /// holds, once its calls in flight have returned: they write within its blocks and may end
    /// it themselves, and their outputs are applied (SD §3.6, §3.7).
    #[test]
    fn a_request_the_cache_cannot_hold_ends_once_its_calls_in_flight_return() {
        let mut session = serving(&[full(4)], KvCacheConfig::default());
        let (output, outputs) = crossbeam_channel::unbounded();
        session.scheduler.output = output;
        // The device region holds four blocks, every one the request's: its prompt.
        session.scheduler.kv.resized(0, 4);
        let id = session.request(1, (1..=16).collect());
        session.scheduler.running.push(id);
        let request = session.row_mut(id);
        request.status = RequestStatus::Running;
        request.queued = false;
        request.num_computed_tokens = 16;
        request.scheduled[Phase::Prefill.index()] = 1;
        request.blocks[0] = Blocks {
            first: 0,
            shared: 0,
            end: 4,
            asked: 4,
        };

        // Its next decode writes block 4, which no victim could free; its prefill is in flight.
        session.scheduler.preempt(id, 0);
        assert_eq!(session.row(id).status, RequestStatus::Running);
        assert!(outputs.try_recv().is_err());

        session.row_mut(id).completed[Phase::Prefill.index()] = 1;
        session.scheduler.preempt(id, 0);
        let ended: Vec<Event> = outputs
            .try_iter()
            .flat_map(|output| match output {
                Output::Events(events) => events,
                _ => Vec::new(),
            })
            .filter_map(|(of, event)| (of == id).then_some(event))
            .collect();
        let [Event::Finished { reason, .. }] = &ended[..] else {
            panic!("{ended:?}");
        };
        assert_eq!(*reason, FinishReason::CacheCapacity);
        assert!(!session.scheduler.requests.contains_key(&id));
    }

    /// A step installs the eviction order, whose head its `available` counts, ahead of the
    /// allocations it asks for: the buffer pool takes the `Victims` first, so those allocations
    /// may evict the blocks it names (SD §4.5).
    #[test]
    fn a_step_installs_the_eviction_order_ahead_of_its_allocations() {
        let mut session = serving(&[full(4)], KvCacheConfig::default());
        // One cached block no request holds; the region has room for the request's blocks.
        let cached = Key([7; 32]);
        session.list(&[(cached, 0, DEVICE)]);
        session.pool.register(1).store(8, Ordering::Release);
        let id = session.request(1, (1..=10).collect());
        session.scheduler.running.push(id);
        let request = session.row_mut(id);
        request.status = RequestStatus::Running;
        request.queued = false;
        request.num_computed_tokens = 10;
        request.blocks[0] = Blocks {
            first: 0,
            shared: 0,
            end: 3,
            asked: 3,
        };

        session.scheduler.step();
        let mut record = Vec::new();
        let mut written = Vec::new();
        while session.pool.next_request(&mut record).unwrap() {
            written.push(runtime::Request::decode(&record).unwrap().1);
        }
        let victims = runtime::Request::Victims {
            region: 4,
            keys: vec![cached],
        };
        assert_eq!(written, [victims, allocate(1, 0, 3, 1)]);
    }

    /// A port no socket of the host listens on now.
    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    /// A ZMQ payload read as SGLang's consumers read it (`sgl-router`'s `kv_events/wire.rs`):
    /// a MessagePack array `[ts, events, attn_dp_rank]` whose events are maps tagged by `type`.
    fn decoded(payload: &[u8]) -> (f64, Vec<serde_json::Value>, serde_json::Value) {
        let batch: serde_json::Value = rmp_serde::from_slice(payload).unwrap();
        let [time, events, rank] = batch.as_array().unwrap().as_slice() else {
            panic!("a batch is an array of three: {batch}");
        };
        (
            time.as_f64().unwrap(),
            events.as_array().unwrap().clone(),
            rank.clone(),
        )
    }

    /// SGLang's block hashes of `tokens` in pages of `block` (`mem_cache/utils.py`,
    /// `get_hash_str` and `hash_str_to_int64`): each page's SHA-256 over the previous digest and
    /// its tokens as little-endian `u32`s, its leading eight bytes as a big-endian `i64`.
    fn sglang_hashes(tokens: &[u32], block: usize) -> Vec<i64> {
        use sha2::Digest as _;
        let mut previous: Option<Vec<u8>> = None;
        let mut hashes = Vec::new();
        for page in tokens.chunks_exact(block) {
            let mut digest = sha2::Sha256::new();
            if let Some(previous) = &previous {
                digest.update(previous);
            }
            for token in page {
                digest.update(token.to_le_bytes());
            }
            let digest = digest.finalize().to_vec();
            hashes.push(i64::from_be_bytes(digest[..8].try_into().unwrap()));
            previous = Some(digest);
        }
        hashes
    }

    /// With `kv_events_config` stated, the scheduler publishes the pages its queue pair commits
    /// and learns evicted in SGLang's ZMQ format: a subscriber reads each step's batch as
    /// three frames, the stores of a request's full pages with their tokens, SGLang's block
    /// hashes and the page size, and the removal of the page the cache no longer lists; the
    /// replay socket answers every batch from a sequence number on, then its end marker.
    #[test]
    fn the_pages_it_commits_and_loses_are_published_as_sglang_kv_events() {
        use zeromq::{Socket as _, SocketRecv as _, SocketSend as _};

        let mut session = serving(&[full(4)], KvCacheConfig::default());
        let (live, replay) = (free_port(), free_port());
        session.scheduler.tune(|tuning| {
            tuning.kv_events.endpoint = Some(format!("tcp://127.0.0.1:{live}"));
            tuning.kv_events.replay_endpoint = Some(format!("tcp://127.0.0.1:{replay}"));
            tuning.kv_events.topic = "kv".to_owned();
            // No page is offloaded, so the cache's requests are the case's own.
            tuning.host_tier = false;
        });
        // The step takes the settings: the publisher binds and states that it holds nothing.
        session.scheduler.step();

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut subscriber = zeromq::SubSocket::new();
        let mut dealer = zeromq::DealerSocket::new();
        runtime.block_on(async {
            subscriber
                .connect(&format!("tcp://127.0.0.1:{live}"))
                .await
                .unwrap();
            subscriber.subscribe("kv").await.unwrap();
            dealer
                .connect(&format!("tcp://127.0.0.1:{replay}"))
                .await
                .unwrap();
            // The subscription reaches the publisher before the next batch is sent.
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        });

        // A request of ten tokens fills two pages of four, whose commits are answered.
        let id = session.request(1, (1..=10).collect());
        let keys = session.row(id).keys[0].clone();
        let request = session.row_mut(id);
        request.status = RequestStatus::Running;
        request.num_computed_tokens = 10;
        request.blocks[0] = Blocks {
            first: 0,
            shared: 0,
            end: 3,
            asked: 3,
        };
        session
            .scheduler
            .commit_filled(id, &batch(Phase::Prefill, 0), 0);
        let commits: Vec<runtime::Request> = session
            .stamped()
            .into_iter()
            .map(|mut commit| {
                if let runtime::Request::Commit { after, .. } = &mut commit {
                    after.push((A, 1));
                }
                commit
            })
            .collect();
        assert_eq!(commits.len(), 2);
        session.scheduler.stamped(commits);
        session.scheduler.post().unwrap();
        let mut record = Vec::new();
        while session.pool.next_request(&mut record).unwrap() {
            let (number, _) = runtime::Request::decode(&record).unwrap();
            record.clear();
            Completion::Committed.encode(number, &mut record);
            assert!(session.pool.complete(&record).unwrap());
            record.clear();
        }
        session.scheduler.step();

        // The buffer pool lists only the first page: the second left the device.
        session.list(&[(keys[0], 0, DEVICE)]);
        session.scheduler.step();

        let hashes = sglang_hashes(&(1..=8).collect::<Vec<u32>>(), 4);
        let received = runtime.block_on(async {
            let mut received = Vec::new();
            for _ in 0..2 {
                let message =
                    tokio::time::timeout(std::time::Duration::from_secs(10), subscriber.recv())
                        .await
                        .expect("a batch arrives")
                        .unwrap();
                received.push(message.into_vec());
            }
            received
        });

        // Batch 1: the two pages, stored on the device as one chain from no parent.
        let [topic, sequence, payload] = &received[0][..] else {
            panic!("a batch is three frames");
        };
        assert_eq!(&topic[..], b"kv");
        assert_eq!(&sequence[..], &1u64.to_be_bytes());
        let (time, events, rank) = decoded(payload);
        assert!(time > 1.0e9, "the publisher's wall clock in seconds");
        assert_eq!(rank, serde_json::json!(0));
        assert_eq!(
            events,
            [serde_json::json!({
                "type": "BlockStored",
                "block_hashes": hashes,
                "parent_block_hash": null,
                "token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                "block_size": 4,
                "lora_id": null,
                "medium": "GPU",
            })]
        );

        // Batch 2: the second page left the device.
        let [_, sequence, payload] = &received[1][..] else {
            panic!("a batch is three frames");
        };
        assert_eq!(&sequence[..], &2u64.to_be_bytes());
        assert_eq!(
            decoded(payload).1,
            [serde_json::json!({
                "type": "BlockRemoved",
                "block_hashes": [hashes[1]],
                "medium": "GPU",
            })]
        );

        // The replay answers from batch 0, the publisher's start, which states it holds nothing.
        let replayed = runtime.block_on(async {
            let mut request = zeromq::ZmqMessage::from(Vec::new());
            request.push_back(0u64.to_be_bytes().to_vec().into());
            dealer.send(request).await.unwrap();
            let mut replayed = Vec::new();
            loop {
                let message =
                    tokio::time::timeout(std::time::Duration::from_secs(10), dealer.recv())
                        .await
                        .expect("the replay answers")
                        .unwrap()
                        .into_vec();
                let end = message[2][..] == (-1i64).to_be_bytes();
                replayed.push(message);
                if end {
                    break replayed;
                }
            }
        });
        let sequences: Vec<&[u8]> = replayed.iter().map(|message| &message[2][..]).collect();
        assert_eq!(
            sequences,
            [
                &0u64.to_be_bytes()[..],
                &1u64.to_be_bytes()[..],
                &2u64.to_be_bytes()[..],
                &(-1i64).to_be_bytes()[..],
            ]
        );
        assert_eq!(
            decoded(&replayed[0][3]).1,
            [serde_json::json!({"type": "AllBlocksCleared"})]
        );
        assert_eq!(&replayed[1][3], &received[0][2]);
    }

    #[test]
    fn filled_blocks_are_committed_in_order_and_a_migrating_tail_under_its_tail_key() {
        let mut session = serving(&[full(4)], KvCacheConfig::default());
        let id = session.request(1, (1..=10).collect());
        let keys = session.row(id).keys[0].clone();
        let request = session.row_mut(id);
        request.status = RequestStatus::Running;
        // Its prompt is computed, and its prefill sampled its first token.
        request.num_computed_tokens = 10;
        request.token_ids.push(11);
        request.blocks[0] = Blocks {
            first: 0,
            shared: 0,
            end: 3,
            asked: 3,
        };

        let commit = |at: u32, key: Key, tokens: u16| runtime::Request::Commit {
            owner: OwnerId(1),
            class: 0,
            at,
            key,
            tokens,
            after: Vec::new(),
        };
        session
            .scheduler
            .commit_filled(id, &batch(Phase::Prefill, 0), 0);
        assert_eq!(
            session.stamped(),
            [commit(0, keys[0], 4), commit(1, keys[1], 4)]
        );
        assert_eq!(session.row(id).blocks[0].shared, 2);

        // Leaving as migratable, its partial last block is committed under its tail key.
        let tail = session
            .scheduler
            .kv
            .tail_key(session.row(id), (0, 10), &session.scheduler.groups)
            .unwrap();
        session.scheduler.commit_tails(id);
        assert_eq!(session.stamped(), [commit(2, tail, 2)]);

        // A request that does not write commits nothing as it fills.
        let other = session.request(2, (1..=10).collect());
        if let proto::request::Request::Generation(input) = &mut session.row_mut(other).input {
            input.cache.write = false;
        }
        session.row_mut(other).num_computed_tokens = 10;
        session.row_mut(other).blocks[0] = Blocks {
            first: 0,
            shared: 0,
            end: 3,
            asked: 3,
        };
        session
            .scheduler
            .commit_filled(other, &batch(Phase::Prefill, 0), 0);
        assert!(session.stamped().is_empty());
    }

    /// A request that leaves as migratable with a decode in flight commits its tail as its
    /// resubmission looks it up: up to the token before the last it was sent, whose entry the
    /// decode in flight wrote and which the resubmission computes again (SD §4.4 "The hit").
    #[test]
    fn a_tail_left_with_a_decode_in_flight_ends_where_its_resubmission_looks_it_up() {
        let mut session = serving(&[full(4)], KvCacheConfig::default());
        let id = session.request(1, (1..=10).collect());
        let request = session.row_mut(id);
        request.status = RequestStatus::Running;
        // Two tokens were sent; the decode that read the second wrote position 11 and its
        // output was dropped as the request left.
        request.token_ids.extend([11, 12]);
        request.num_computed_tokens = 12;
        request.blocks[0] = Blocks {
            first: 0,
            shared: 2,
            end: 4,
            asked: 4,
        };

        let tail = session
            .scheduler
            .kv
            .tail_key(session.row(id), (0, 11), &session.scheduler.groups)
            .unwrap();
        session.scheduler.commit_tails(id);
        let commit = runtime::Request::Commit {
            owner: OwnerId(1),
            class: 0,
            at: 2,
            key: tail,
            tokens: 3,
            after: Vec::new(),
        };
        assert_eq!(session.stamped(), [commit]);
    }

    /// A migratable request's tail is committed as it leaves, and its answer arrives after the
    /// owner's `Detach`: the commit outlives the `Detach` (BP §9, R6), so the answer is applied
    /// and the write policy stores the block to the host from the group that stamped it.
    #[test]
    fn a_tail_committed_as_its_request_leaves_is_stored_after_the_detach() {
        let kv = KvCacheConfig {
            host_tier: true,
            ..KvCacheConfig::default()
        };
        let mut session = serving(&[full(4)], kv);
        let id = session.request(1, (1..=10).collect());
        let request = session.row_mut(id);
        request.status = RequestStatus::Running;
        request.num_computed_tokens = 10;
        request.token_ids.push(11);
        request.blocks[0] = Blocks {
            first: 0,
            shared: 2,
            end: 3,
            asked: 3,
        };
        let tail = session
            .scheduler
            .kv
            .tail_key(session.row(id), (0, 10), &session.scheduler.groups)
            .unwrap();

        session.scheduler.commit_tails(id);
        session.scheduler.submit(runtime::Request::Detach {
            owner: OwnerId(1),
            after: Vec::new(),
        });
        let [commit @ runtime::Request::Commit { .. }, _] = &session.stamped()[..] else {
            panic!("the tail's `Commit` and the `Detach` are handed to group A");
        };
        let mut commit = commit.clone();
        if let runtime::Request::Commit { after, .. } = &mut commit {
            after.push((A, 1));
        }
        session.scheduler.stamped(vec![commit]);

        session.scheduler.post().unwrap();
        let mut record = Vec::new();
        assert!(session.pool.next_request(&mut record).unwrap());
        let (number, _) = runtime::Request::decode(&record).unwrap();
        record.clear();
        Completion::Committed.encode(number, &mut record);
        assert!(session.pool.complete(&record).unwrap());
        while let Some((number, owner, request, completion)) =
            session.scheduler.next_completion().unwrap()
        {
            session
                .scheduler
                .answered(number, owner, request, completion);
        }

        let store = runtime::Request::Store {
            group: A,
            keys: vec![tail],
            tier: Tier::Host,
        };
        session.scheduler.post().unwrap();
        record.clear();
        assert!(session.pool.next_request(&mut record).unwrap());
        assert_eq!(runtime::Request::decode(&record).unwrap().1, store);
    }

    #[test]
    fn a_sliding_window_gives_up_what_lies_behind_it() {
        let mut session = serving(&[full(4), window(8, 4)], KvCacheConfig::default());
        let id = session.request(1, (1..=24).collect());
        let request = session.row_mut(id);
        request.num_computed_tokens = 20;
        request.blocks[0] = Blocks {
            first: 0,
            shared: 0,
            end: 6,
            asked: 6,
        };
        request.blocks[1] = Blocks {
            first: 0,
            shared: 1,
            end: 6,
            asked: 6,
        };

        // Later calls read from token 12 on: blocks 0 to 2 go, the full-attention class keeps all.
        session
            .scheduler
            .give_up_behind(id, &batch(Phase::Decode, 0), 0);
        assert_eq!(session.stamped(), [release(1, 1, 0, 3)]);
        assert_eq!(
            session.row(id).blocks[1],
            Blocks {
                first: 3,
                shared: 3,
                end: 6,
                asked: 6
            }
        );
        assert_eq!(session.row(id).blocks[0].first, 0);
    }

    #[test]
    fn a_recurrent_state_table_keeps_its_base_and_one_checkpoint_ahead() {
        let state = (CacheGroupKind::State, 4);
        let mut session = serving(&[full(4), state], KvCacheConfig::default());
        let id = session.request(1, (1..=20).collect());

        // No hit: the first chunk of 8 tokens keeps the checkpoint ending at token 8.
        let asks = session.scheduler.admission_requests(id);
        assert_eq!(asks, [allocate(1, 0, 0, 3), allocate(1, 1, 1, 1)]);
        session.answer(&asks, &[allocated(3), allocated(1)]);
        assert!(settled(
            session.row(id),
            &session.scheduler.groups,
            &session.scheduler.budget
        ));
        session.row_mut(id).status = RequestStatus::Running;

        // The chunk ran: the next one keeps the checkpoint ending at 16, asked ahead.
        session.row_mut(id).num_computed_tokens = 8;
        session
            .scheduler
            .give_up_behind(id, &batch(Phase::Prefill, 8), 0);
        assert!(session.stamped().is_empty());
        assert_eq!(
            ahead(session.row(id), &session.scheduler.kv_step())[1],
            (1, 3, 1, 1)
        );
        count_asked(session.row_mut(id), 1, 3, 1);
        session.answer(&[allocate(1, 1, 3, 1)], &[allocated(1)]);
        session.row_mut(id).blocks[0].end = 5;

        // A chunk ends where the table holds its checkpoint, or short of the next boundary.
        assert!(holds_blocks_for(
            session.row(id),
            (Phase::Prefill, 8),
            &session.scheduler.groups,
            &session.scheduler.config
        ));
        assert!(!holds_blocks_for(
            session.row(id),
            (Phase::Prefill, 5),
            &session.scheduler.groups,
            &session.scheduler.config
        ));
        assert_eq!(
            fit_chunk(
                session.row(id),
                Phase::Prefill,
                12,
                &session.scheduler.groups
            ),
            11
        );
        assert_eq!(
            fit_chunk(
                session.row(id),
                Phase::Prefill,
                5,
                &session.scheduler.groups
            ),
            3
        );

        // Once the call wrote the checkpoint ahead, the base is given up.
        session.row_mut(id).num_computed_tokens = 16;
        session
            .scheduler
            .give_up_behind(id, &batch(Phase::Prefill, 16), 0);
        assert_eq!(session.stamped(), [release(1, 1, 1, 1)]);
        assert_eq!(
            session.row(id).blocks[1],
            Blocks {
                first: 3,
                shared: 3,
                end: 4,
                asked: 4
            }
        );
        // Leaving as migratable, the base the target wrote is committed with the blocks.
        session.scheduler.commit_tails(id);
        let state = session.row(id).keys[1][3];
        let committed = session.stamped();
        assert_eq!(
            committed.len(),
            5,
            "four full blocks and the checkpoint: {committed:?}"
        );
        assert_eq!(committed[4], commit(OwnerId(1), 1, 3, state, 4));
    }

    /// A verification keeps its checkpoint before it learns which drafts are accepted. When it
    /// rejects a draft before the boundary, the checkpoint holds the state of tokens the request
    /// does not have: it is neither committed nor taken for written, so the base stays and the
    /// next call that reaches the boundary finds the entry writable and keeps it again, which is
    /// committed once the accepted tokens reach the boundary.
    #[test]
    fn a_verification_commits_its_checkpoint_once_its_accepted_tokens_reach_the_boundary() {
        let kv = KvCacheConfig {
            state_interval: std::num::NonZeroU32::new(8).unwrap(),
            ..KvCacheConfig::default()
        };
        let mut session = serving(&[full(4), (CacheGroupKind::State, 4)], kv);
        let id = session.request(1, (1..=6).collect());
        let request = session.row_mut(id);
        request.status = RequestStatus::Running;
        request.placement[Phase::Verify.index()] = Some(A);
        // The verification of token 7 and its draft at position 7 ran up to 8 and kept the state
        // there; the draft was rejected, so 7 tokens are computed and the corrected token 9 is
        // sampled. The full-attention table holds three blocks, the first committed; the state
        // table its committed base at boundary 4 and the checkpoint at boundary 8 ahead of it.
        request.token_ids.extend([7, 9]);
        request.num_computed_tokens = 7;
        let table = |shared, end| Blocks {
            first: 0,
            shared,
            end,
            asked: end,
        };
        request.blocks = vec![table(1, 3), table(1, 2)];

        let verified = batch(Phase::Verify, 8);
        session.scheduler.commit_filled(id, &verified, 0);
        session.scheduler.give_up_behind(id, &verified, 0);
        assert!(session.stamped().is_empty());
        assert_eq!(session.row(id).blocks[1], table(1, 2));
        assert!(holds_blocks_for(
            session.row(id),
            (Phase::Verify, 1),
            &session.scheduler.groups,
            &session.scheduler.config
        ));

        // The next verification reached 8 again and its draft was accepted: 9 tokens are
        // computed. Its checkpoint is committed with the full block it filled, and the base is
        // given up.
        let request = session.row_mut(id);
        request.token_ids.extend([10, 11]);
        request.num_computed_tokens = 9;
        session.scheduler.commit_filled(id, &verified, 0);
        session.scheduler.give_up_behind(id, &verified, 0);
        let keys = session.row(id).keys.clone();
        assert_eq!(
            session.stamped(),
            [
                commit(OwnerId(1), 0, 1, keys[0][1], 4),
                commit(OwnerId(1), 1, 1, keys[1][1], 4),
                release(1, 1, 0, 1),
            ]
        );
        assert_eq!(
            session.row(id).blocks[1],
            Blocks {
                first: 1,
                shared: 2,
                end: 2,
                asked: 2
            }
        );
    }

    #[test]
    fn drafts_are_given_up_from_the_request_with_the_fewest_tokens_left() {
        let draft = (CacheGroupKind::Draft, 4);
        let mut session = serving(&[full(4), draft], KvCacheConfig::default());
        for (id, generated) in [(1, 2), (2, 10), (3, 0)] {
            let id = session.request(id, (1..=8).collect());
            let request = session.row_mut(id);
            request.status = RequestStatus::Running;
            request.token_ids.extend(std::iter::repeat_n(5, generated));
            request.num_draft_tokens = 8;
            request.blocks[1] = Blocks {
                first: 0,
                shared: 1,
                end: 3,
                asked: 3,
            };
            session.scheduler.running.push(id);
        }
        // Request 3 has a draft block outstanding.
        session.row_mut(RequestId(3)).blocks[1].asked = 4;

        assert_eq!(
            draft_order(
                session.scheduler.running.iter().map(|id| session.row(*id)),
                0,
                &session.scheduler.groups
            ),
            [RequestId(2), RequestId(1)]
        );
        let freed = session.scheduler.give_up_draft(RequestId(2));
        assert_eq!(freed[0], 3);
        assert_eq!(session.stamped(), [release(2, 1, 0, 3)]);
        assert_eq!(session.row(RequestId(2)).blocks[1], Blocks::default());
        assert_eq!(session.row(RequestId(2)).num_draft_tokens, 0);
    }

    #[test]
    fn the_write_policy_stores_full_blocks_and_windows_before_aligned_boundaries() {
        let kv = KvCacheConfig {
            boundary_interval: std::num::NonZeroU32::new(4).unwrap(),
            ..KvCacheConfig::default()
        };
        let mut session = serving(&[full(4), window(4, 4)], kv);
        let id = session.request(1, (1..=100).collect());
        let key = session.row(id).keys[0][5];
        // A `Commit` is written with the stamp of group A, whose table held the entry.
        let commit = |class, at| runtime::Request::Commit {
            owner: OwnerId(1),
            class,
            at,
            key,
            tokens: 4,
            after: vec![(A, 1)],
        };
        let store = |tier| runtime::Request::Store {
            group: A,
            keys: vec![key],
            tier,
        };
        let policy = |session: &Session, asked: &runtime::Request, answer: &Completion| {
            let scheduler = &session.scheduler;
            let holder = scheduler.requests.get(&id);
            let answer = (asked, answer);
            let budget = &scheduler.budget;
            scheduler
                .kv
                .write_policy(answer, holder, &scheduler.groups, budget)
        };

        assert_eq!(
            policy(&session, &commit(0, 5), &Completion::Committed),
            Some(store(Tier::Host))
        );
        assert_eq!(
            policy(&session, &commit(0, 5), &Completion::Duplicate),
            None
        );
        // A window block is stored only before a boundary of 16 blocks... of 4 tokens: block 15
        // ends at 64; block 14 lies outside the window of 4 tokens before it; the last boundary
        // a reissue reuses, 96, ends block 23.
        assert_eq!(
            policy(&session, &commit(1, 15), &Completion::Committed),
            Some(store(Tier::Host))
        );
        assert_eq!(
            policy(&session, &commit(1, 14), &Completion::Committed),
            None
        );
        assert_eq!(
            policy(&session, &commit(1, 23), &Completion::Committed),
            Some(store(Tier::Host))
        );

        // A host copy is stored on to the remote store only when the deployment has one.
        let stored = Completion::Stored { keys: Vec::new() };
        assert_eq!(policy(&session, &store(Tier::Host), &stored), None);
        let kv = KvCacheConfig {
            mooncake: Some(storage::connector::Settings {
                master: "master:1".parse().unwrap(),
                metadata: storage::connector::Metadata::P2pHandshake,
                transport: storage::connector::Transport::Tcp,
                device: String::new(),
                segment_gib: 0,
                events: None,
            }),
            ..KvCacheConfig::default()
        };
        session.scheduler.groups[0].capacity.kv_cache = Some(kv);
        assert_eq!(
            policy(&session, &store(Tier::Host), &stored),
            Some(store(Tier::Remote))
        );
    }

    #[test]
    fn an_admission_needs_its_first_chunk_or_what_its_prompt_needs_in_chunks() {
        // SD §7.2: blocks of 64 tokens, a 1000-token prompt and chunks of 512.
        let mut session = serving(&[full(64), window(128, 64)], KvCacheConfig::default());
        session
            .scheduler
            .tune(|tuning| tuning.chunked_prefill_size = std::num::NonZeroUsize::new(512));
        let id = session.request(1, (1..=1000).collect());

        // Without a hit: every block of the full class, and of the window what a chunk's window
        // and query span: 16 + min(16, ceil(640 / 64) + 1).
        assert_eq!(session.scheduler.units_alone(session.row(id))[0], 16 + 11);
        assert_eq!(
            session.scheduler.units_needed(session.row(id), 0)[0],
            16 + 11
        );

        // A hit of 896 tokens leaves 2 blocks after the prefix, the first chunk's and one more.
        assert_eq!(
            session.scheduler.units_needed(session.row(id), 896)[0],
            3 + 3
        );
    }

    #[test]
    fn an_encoder_output_is_pinned_where_cached_and_else_allocated_and_committed_once_encoded() {
        let mut session = layout(&[full(4)], KvCacheConfig::default(), 14);
        let id = session.request(1, vec![1, 2, 3]);
        let image = proto::request::ImageInput {
            digest: [9; 32],
            encoded: Vec::new(),
            position: 1,
            num_positions: 1,
            encoders: vec![proto::request::ImageEncoderInput {
                encoder: proto::request::ImageEncoder::Vision,
                num_cache_tokens: Some(32),
                max_cache_tokens: None,
                height: None,
                width: None,
            }],
        };
        if let proto::request::Request::Generation(input) = &mut session.row_mut(id).input {
            input.multimodal_inputs.images = vec![image];
        }
        // SD §4.3: the output's key covers the image's digest, and no draw.
        let mut inputs = Digest::new(b"image-encoding");
        inputs.bytes(&[9; 32]);
        inputs.number(0);
        let key = Key(encoder_output("vision_features", "", &inputs.finish()));
        let encodings = session.scheduler.kv.encodings_of(session.row(id));
        assert_eq!(encodings, [(key, None)]);
        session.row_mut(id).encodings = encodings;

        // Not cached: allocated at its input's position, 32 rows of 64 KiB in 1 MiB units.
        let encoder = |asks: Vec<runtime::Request>| -> Vec<runtime::Request> {
            let class = |ask: &runtime::Request| {
                super::super::super::span(ask).and_then(|(_, class, _)| class)
            };
            asks.into_iter()
                .filter(|ask| class(ask) == Some(1))
                .collect()
        };
        let asks = session.scheduler.admission_requests(id);
        assert_eq!(encoder(asks), [sized(1, 1, 0, 2)]);

        // Cached on the device: pinned; a pin that finds nothing allocates it at once.
        session.list(&[(key, 1, DEVICE)]);
        let asks = encoder(session.scheduler.admission_requests(id));
        assert_eq!(asks, [pin(1, 1, 0, &[key])]);
        assert_eq!(session.answer(&asks, &[pinned(0)]), [sized(1, 1, 0, 2)]);
        assert_eq!(session.row(id).encodings[0].1, None);

        // The encode computed it: it is committed under its key, its rows as tokens, and held.
        let mut encoded = batch(Phase::VisionEncode, 0);
        encoded.media.push(crate::batch::MediaRow {
            request: id,
            slot: 1,
            conditioning: Vec::new(),
            media: Vec::new(),
            step: 0,
            steps: 0,
            seed: 0,
            width: 0,
            height: 0,
            frames: 0,
            rounds: 0,
            unit: 0,
            rank: None,
            image: None,
            results: vec![0],
            reads: 0,
        });
        session.scheduler.commit_encoding(id, &encoded, 0);
        assert_eq!(session.stamped(), [commit(OwnerId(1), 1, 0, key, 32)]);
        assert_eq!(session.row(id).encodings[0].1, Some(0));
    }

    #[test]
    fn the_admission_reserve_is_what_running_requests_need_to_their_low_watermark() {
        let mut session = serving(&[full(4)], KvCacheConfig::default());
        for (id, computed) in [(1, 10), (2, 12)] {
            let id = session.request(id, (1..=computed).collect());
            let request = session.row_mut(id);
            request.status = RequestStatus::Running;
            request.num_computed_tokens = computed;
            request.blocks[0] = run(0, 0, 3);
            request.blocks[0].asked = 3;
            session.scheduler.running.push(id);
        }
        // Request 1 decodes token 10 and lacks one block beyond it; request 2 decodes token 12
        // into block 3 and lacks it and one more.
        assert_eq!(session.scheduler.admission_reserve()[0], 1 + 2);
    }

    #[test]
    fn a_waiting_request_the_policy_names_is_prefetched_for_any_deepening() {
        let mut session = serving(&[full(64)], KvCacheConfig::default());
        let id = session.request(1, (1..=1000).collect());
        let keys = session.row(id).keys[0].clone();
        session.list(&[(keys[0], 0, DEVICE), (keys[1], 0, REMOTE)]);

        // 64 tokens more is short of the admission's threshold, not of the policy's naming.
        let fetched = |forced| {
            let step = session.scheduler.kv_step();
            session
                .scheduler
                .kv
                .prefetch_requests(session.row(id), forced, &step)
        };
        assert!(fetched(false).is_empty());
        let asks = fetched(true);
        let [runtime::Request::Prefetch { keys: wanted, .. }] = &asks[..] else {
            panic!("one prefetch, not {asks:?}");
        };
        assert_eq!(&wanted[..], &keys[1..2]);
    }

    /// The KV decisions built alone answer an admission from the keys its request needs and the
    /// group it runs on: the leading blocks `cache` holds on the device in that group are pinned
    /// as the hit, and the blocks of the first chunk after it and the low watermark are
    /// allocated, for the request's owner there.
    #[test]
    fn the_kv_decisions_alone_pin_the_hit_and_allocate_the_first_chunk() {
        use proto::config::DEFAULT_POLICY_MIN_PREFETCH_TOKENS;

        use super::super::super::Budget;
        use super::super::super::tests::{KV, OWNER_A, attached};
        use super::super::tests::kv;

        // Group A keeps one full-attention cache group of 2-token blocks, which prefixes are
        // cached in; the request prefills and decodes there.
        let (mut kv, mut groups) = kv(4096);
        groups[0].capacity.cache_groups = vec![full(2)];
        groups[0].capacity.kv_cache = Some(KvCacheConfig {
            prefix_caching: true,
            ..KvCacheConfig::default()
        });
        let mut request = attached(&[(A, OWNER_A, 3)]);
        request.placement[Phase::Prefill.index()] = Some(A);
        request.token_ids = (1..=9).collect();
        request.num_prompt_tokens = 9;
        request.keys = kv.keys_of(&request, &groups);
        let keys = request.keys[0].clone();
        assert_eq!(keys.len(), 4, "the complete blocks of nine tokens");

        // The buffer pool lists the first three blocks on group A's device.
        let listing = keys[..3]
            .iter()
            .map(|&key| (key, KV, 2, DEVICE, 1, 0))
            .collect();
        kv.submitted(&runtime::Request::List);
        let mut told = Vec::new();
        let listed = (&runtime::Request::List, None);
        let completion = Completion::Listing { keys: listing };
        kv.learn(listed, None, &completion, &groups, &mut told);

        let config = SchedulerConfig::default();
        let budget = Budget {
            revision: 0,
            tokens: 64,
            calls: 8,
            chunk: 64,
            queued: None,
            prefix_caching: true,
            host_tier: true,
            copy_budget: 0,
            synthetic_acceptance_length: None,
        };
        let step = super::Step {
            groups: &groups,
            config: &config,
            budget: &budget,
            prefetch_min_tokens: DEFAULT_POLICY_MIN_PREFETCH_TOKENS,
        };
        assert_eq!(kv.expected(&request, &groups, &budget), 6);
        let asks = kv.admission_requests(&mut request, &step);
        assert_eq!(
            asks,
            [
                pin(OWNER_A.0, KV, 0, &keys[..3]),
                allocate(OWNER_A.0, KV, 3, 3)
            ]
        );
        // The table holds nothing until the answers arrive, and asked for six positions.
        assert_eq!(request.num_cached_tokens, 6);
        let asked = Blocks {
            first: 0,
            shared: 0,
            end: 0,
            asked: 6,
        };
        assert_eq!(request.blocks[0], asked);
    }
}
