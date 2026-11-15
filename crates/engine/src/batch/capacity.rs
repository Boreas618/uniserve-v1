//! What a worker group's workers serve, stated once by them when they are ready
//! (`specs/scheduler-design.md` §2.1 `Capacity`).
//!
//! A rank builds its group's statement from its runners, its buckets and its cache; the
//! scheduler that drives the group schedules within it. The statement crosses processes when a
//! group has ranks on another host: the peer's answer to the session the scheduler node relays
//! carries what its ranks state (`MainProcessResponse::Placed`).

use proto::config::KvCacheConfig;
use serde::{Deserialize, Serialize};

use super::cache::CacheGroupKind;
use super::{Batch, Phase, Shape, Variant};

/// What a worker group's workers serve, stated once by them when they are ready.
///
/// Immutable after construction, and every rank of a group states the same value. The sizes are
/// the workers' for the model, from which the scheduler sizes every sized block
/// (`specs/buffer-pool.md` §11 "Sizes").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Capacity {
    /// The phases the group serves: bit [`Phase::index`] per phase.
    pub phases: u32,
    /// Request slots the group holds at once.
    pub slots: u32,
    /// The cache groups with their kind, block size, window and seeds, and the state and boundary
    /// intervals, from which keys and the block rules are computed; `None` for a group that holds
    /// no cache.
    pub kv_cache: Option<KvCacheConfig>,
    /// The model's cache groups, in class order: class `c` of the session is cache group `c`,
    /// with its retention kind (a sliding window's window) and its block size in tokens. Empty
    /// for a group that holds no cache.
    pub cache_groups: Vec<(CacheGroupKind, u32)>,
    /// The page seed of each cache group, in class order (`storage::key::page_seed` of the
    /// group's page identity: the weights that compute its pages and their layout), which roots
    /// the key chains of the group's blocks. Empty for a group that holds no cache.
    pub seeds: Vec<storage::Key>,
    /// The shapes the group's workers captured; a batch must fit one of them.
    pub buckets: Vec<Shape>,
    /// Per phase, by [`Phase::index`], the requests the stage serving it holds at once: one for
    /// an exclusive stage, the units of a stage distributed by units, the tasks of a host task
    /// pool, the slots of a token phase. A phase the group serves states at least one; a latent
    /// preparation counts in the denoiser's.
    pub rows: [u32; Phase::COUNT],
    /// The longest sequence a request may reach on the group, in tokens.
    pub max_model_len: u32,
    /// The denoising steps of a video request.
    pub denoise_steps: u32,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
    /// The bytes of one token of an encoder output the group computes; 0 when it encodes nothing.
    pub feature_bytes: u32,
    /// The identity of the weights that compute the group's encoder outputs (a chat image's
    /// features, a video's conditioning): `storage::key::encoder_output`'s `identity`, from the
    /// weights of the group's ranks that encode, the encoder's and its vision tower's. It keys
    /// those outputs as `seeds` root the group's KV blocks, so an output other weights computed
    /// is another block. Empty for a group that encodes nothing.
    pub encoder_identity: String,
    /// The seed of the keys of the trajectories the group stores (`storage::key::latent_content`'s
    /// `seed`): `storage::key::latent_seed` of the identities of the weights of the process's
    /// ranks, whose denoiser and encoders compute a trajectory and its conditioning. It keys the
    /// stored trajectories as `seeds` root the group's KV blocks, so a trajectory other weights
    /// computed is another block. 0 for a group that denoises nothing.
    pub latent_seed: u64,
    /// Per captured media shape of a group that denoises: the bytes of a slot's trajectory, and
    /// of each product kind's block in the order of the product classes.
    pub media_bytes: Vec<(Shape, u64, Vec<u64>)>,
    /// The media pipeline the group's ranks serve a part of, as the model family declares it
    /// (`nn::pipeline`): the phases a video request offers, its products in the order of their
    /// classes and the blocks a deployment places. `None` for a group of a text deployment.
    /// Every group of a media deployment states the same.
    pub pipeline: Option<nn::pipeline::Pipeline>,
    /// The calls of one request the group holds unresolved at once: the bound on a request's
    /// pipelined decodes, `min(queue_depth * calls, max(queue_depth, 2))` for batches of up to
    /// `calls` calls (`crate::memory::division::call_window`).
    pub call_window: u32,
}

impl Capacity {
    /// Whether the workers stated `phase`.
    pub fn serves(&self, phase: Phase) -> bool {
        self.phases & (1 << phase.index()) != 0
    }

    /// Whether `batch`, a batch of `phase` being formed, still fits a bucket the workers
    /// captured with one more call (SD §7.1, "Rows of a batch"): its rows stay within
    /// `max(1, min(largest rows, slots))`, the largest rows over the buckets of the phase's
    /// variants. A phase no bucket names runs one call per batch. A host stage's bucket bounds
    /// nothing, since the stage serves however many calls arrive one after another. The batch's
    /// tokens are bounded by the step's token budget, which is within the widest bucket.
    pub fn holds(&self, batch: &Batch, phase: Phase) -> bool {
        let serving = || {
            self.buckets
                .iter()
                .filter(move |shape| variants(phase).contains(&shape.variant))
        };
        if serving().any(|shape| shape.variant.host()) {
            return true;
        }

        let largest = serving().map(|shape| shape.rows).max().unwrap_or(0);
        let rows = u32::try_from(batch.requests.len()).unwrap_or(u32::MAX);
        rows < largest.min(self.slots).max(1)
    }
}

/// The variants a batch of `phase` may be launched as: the prefill, decode and verification
/// variants a token batch's rows choose among, and the one variant of each media phase. A video
/// request's condition encodes are one variant, whose work a later call computes.
pub fn variants(phase: Phase) -> &'static [Variant] {
    match phase {
        Phase::Prefill | Phase::Refill => &[
            Variant::PrefillWrite,
            Variant::PrefillOutput,
            Variant::PrefillGreedy,
            Variant::PrefillScored,
        ],
        Phase::Decode => &[Variant::DecodeGreedy, Variant::DecodeSampled],
        Phase::Verify => &[Variant::VerifyGreedy, Variant::VerifySampled],
        Phase::TextEncode => &[Variant::Encode],
        Phase::VisionEncode | Phase::LatentEncode => &[Variant::ConditionEncode],
        Phase::MediaRead => &[Variant::MediaRead],
        Phase::Denoise => &[Variant::Denoise],
        Phase::VideoDecode => &[Variant::Decode],
        Phase::AudioDecode => &[Variant::AudioDecode],
        Phase::VideoEncode => &[Variant::VideoEncode],
        Phase::AudioEncode => &[Variant::AudioEncode],
        Phase::Mux => &[Variant::Mux],
        Phase::LatentPrepare => &[Variant::LatentPrepare],
    }
}
