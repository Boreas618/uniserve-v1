//! The one test double at the worker's batch interface.
//!
//! [`FakeBatches`] implements [`Batches`], the door a serving scheduler executes through, and
//! answers every batch from what the mechanism wrote into it: the calls it names, the rows each
//! call owns ([`ipc::execution::BatchRow::rows`]), the keys those rows see (`seq_lens`) and the
//! Philox address of each sampling row (`draws`). It reads nothing private to the scheduler, so a
//! case written against it observes the mechanism through the same completion path a device
//! worker's answer takes.
//!
//! What a case can vary, and nothing else:
//!
//! - the pools the fake states ([`FakeBatches::new`]) and how many devices run them
//!   ([`FakeBatches::with_devices`]);
//! - the token each call samples ([`Tokens`]) and the extent it leaves ([`Extent`]), or a whole
//!   answer of its own ([`FakeBatches::answering`]);
//! - the media computations it serves ([`FakeBatches::with_media`]);
//! - when a batch completes: at once, or held until the case releases it
//!   ([`FakeBatches::hold`], [`FakeBatches::release`]), which is how a case paces the mechanism
//!   or leaves a batch stalled on the device;
//! - how a batch ends: refused at submission ([`FakeBatches::refuse`]), or failed after it was
//!   accepted, which is a lost device ([`FakeBatches::fail_held`]).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ipc::TimingCounters;
use ipc::execution::{
    Batch, BatchId, BatchOutput, Batches, CallOutput, Computation, EpochOperation,
    Extent as MediaExtent, Fault, Inventory, Outcome, RowState, Submission, SubmissionId,
    SubmissionState,
};
use ipc::philox::DrawAddress;

use crate::text::EndOfText;

/// The first token id a fake samples: every token it samples lies in `SAMPLED..SAMPLED + 5000`, a
/// window of legal ids that no prompt of the tests uses.
pub const SAMPLED: u32 = 1000;

/// Width of the window of sampled token ids above [`SAMPLED`].
const VOCABULARY: u32 = 5_000;

/// The token a fake samples at `draw`: a function of the request's Philox stream and the logical
/// position of the token alone, as a device sampler's draw is.
///
/// A case computes what a request samples from the request's sampling key
/// (`ipc::philox::request_sampling_key`) and the position, so a request that continues on another
/// engine, in another slot or another batch, samples what it would have sampled uninterrupted.
pub fn draw_token(draw: DrawAddress) -> u32 {
    // SplitMix64's finalizer over the two halves of the address: a fixed bijective mix, so nearby
    // keys and positions give unrelated tokens.
    let mut mixed = draw.key ^ draw.position.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^= mixed >> 31;

    SAMPLED + (mixed % u64::from(VOCABULARY)) as u32
}

/// Which token a call samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tokens {
    /// [`draw_token`] of the call's last row's draw: what a device sampler's choice depends on.
    #[default]
    Draw,
    /// [`SAMPLED`] plus the call's place in the batch: every batch looks alike.
    Positional,
    /// [`SAMPLED`] plus the keys the call's last row sees, modulo the window: a function of the
    /// request's history as the mechanism states it, independent of the sampling key.
    Extent,
}

/// What a call's row state reports once the call completed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Extent {
    /// The keys the call's last row sees (`seq_lens`): cached prefix plus its own query, which is
    /// exactly the cache the call leaves behind.
    #[default]
    Stated,
    /// One past [`Extent::Stated`], for every field including the step index.
    OnePast,
}

/// A case's own answer to a batch, in place of the [`Tokens`] and [`Extent`] rules.
pub type Answer = Arc<dyn Fn(&Batch) -> Outcome<BatchOutput> + Send + Sync>;

/// The media capability a fake states: the bucket it serves a computation in, or `None`.
pub type Media = Arc<dyn Fn(&Computation) -> Option<MediaExtent> + Send + Sync>;

/// The storage operations a fake performs ([`Batches::operate`]): the operation in, each rank's
/// JSON reply out.
pub type Operate = Arc<dyn Fn(&EpochOperation) -> Result<Vec<String>, Box<Fault>> + Send + Sync>;
/// One accepted serving batch and the batches still on the device when it was accepted.
#[derive(Clone, Debug)]
pub struct Accepted {
    /// The batch as the mechanism submitted it.
    pub batch: Batch,
    /// The batches accepted earlier and not yet completed.
    pub in_flight: Vec<BatchId>,
}

/// A batch accepted and not yet completed, with the outcome it completes with.
struct Held {
    batch: BatchId,
    state: Arc<SubmissionState>,
    outcome: Outcome<BatchOutput>,
}

/// The fake worker: a [`Batches`] over the pools a case states.
pub struct FakeBatches {
    inventory: Vec<Inventory>,
    devices: usize,
    tokens: Tokens,
    extent: Extent,
    answer: Option<Answer>,
    media: Option<Media>,
    end: Option<EndOfText>,
    operate: Option<Operate>,
    /// Every batch submitted, in submission order: what the mechanism asked for is half of what a
    /// case observes.
    submitted: Mutex<Vec<Batch>>,
    /// Every shadow batch, in submission order.
    shadows: Mutex<Vec<Batch>>,
    /// Every accepted serving batch with the batches still on the device when it was accepted.
    accepted: Mutex<Vec<Accepted>>,
    /// Every trajectory copy to a host latent pool, in submission order.
    latent_offloads: Mutex<Vec<ipc::LatentOffload>>,
    /// While set, every submission is refused with this message.
    refusal: Mutex<Option<String>>,
    /// While true, accepted batches wait in `held` for the case to release them.
    holding: Mutex<bool>,
    held: Mutex<VecDeque<Held>>,
    next: AtomicU64,
}

/// Locks `mutex`, reading through a poisoned lock: a case that panicked holding it has already
/// failed, and the fake's state stays consistent under every lock it takes.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl FakeBatches {
    /// A fake that serves `inventory` on one device, completes every batch at once, samples by
    /// [`Tokens::Draw`] and reports [`Extent::Stated`].
    pub fn new(inventory: Vec<Inventory>) -> Self {
        Self {
            inventory,
            devices: 1,
            tokens: Tokens::default(),
            extent: Extent::default(),
            answer: None,
            media: None,
            end: None,
            operate: None,
            submitted: Mutex::new(Vec::new()),
            shadows: Mutex::new(Vec::new()),
            accepted: Mutex::new(Vec::new()),
            latent_offloads: Mutex::new(Vec::new()),
            refusal: Mutex::new(None),
            holding: Mutex::new(false),
            held: Mutex::new(VecDeque::new()),
            next: AtomicU64::new(1),
        }
    }

    /// Samples each call's token by `tokens`.
    #[must_use]
    pub fn with_tokens(mut self, tokens: Tokens) -> Self {
        self.tokens = tokens;
        self
    }

    /// Reports each call's row state by `extent`.
    #[must_use]
    pub fn with_extent(mut self, extent: Extent) -> Self {
        self.extent = extent;
        self
    }

    /// Runs the pools' batches on `devices` devices, each batch on all of them.
    #[must_use]
    pub fn with_devices(mut self, devices: usize) -> Self {
        self.devices = devices.max(1);
        self
    }

    /// Samples `end.token` for every row whose draw lies at or past `end.position`: a model whose
    /// text ends there (`crate::text`).
    #[must_use]
    pub fn with_end(mut self, end: EndOfText) -> Self {
        self.end = Some(end);
        self
    }

    /// Answers every batch with `answer` instead of the token and extent rules.
    #[must_use]
    pub fn answering(mut self, answer: Answer) -> Self {
        self.answer = Some(answer);
        self
    }

    /// States the media computations this fake serves, and the bucket of each.
    #[must_use]
    pub fn with_media(mut self, media: Media) -> Self {
        self.media = Some(media);
        self
    }

    /// Performs the storage operations the mechanism asks for with `operate`, as the ranks of a
    /// worker with storage of its own answer them; without one, every operation answers nothing.
    pub fn with_operate(mut self, operate: Operate) -> Self {
        self.operate = Some(operate);
        self
    }

    /// Every batch submitted so far, in submission order.
    pub fn submitted(&self) -> Vec<Batch> {
        lock(&self.submitted).clone()
    }

    /// Every serving batch accepted so far, in acceptance order, each with the batches that were
    /// held on the device when it was accepted: what ran at the same time as it.
    pub fn accepted(&self) -> Vec<Accepted> {
        lock(&self.accepted).clone()
    }

    /// Every trajectory copy to a host latent pool submitted so far, in submission order.
    pub fn latent_offloads(&self) -> Vec<ipc::LatentOffload> {
        lock(&self.latent_offloads).clone()
    }

    /// Every shadow batch submitted so far, in submission order.
    pub fn shadows(&self) -> Vec<Batch> {
        lock(&self.shadows).clone()
    }

    /// Refuses every later submission with `message`, or accepts them again with `None`.
    pub fn refuse(&self, message: Option<&str>) {
        *lock(&self.refusal) = message.map(str::to_owned);
    }

    /// Holds every later accepted batch until [`Self::release`] completes it: a batch still on the
    /// device, which is what a paced or stalled case needs.
    pub fn hold(&self) {
        *lock(&self.holding) = true;
    }

    /// Completes every held batch and completes later batches at once again.
    pub fn resume(&self) {
        *lock(&self.holding) = false;
        self.release(usize::MAX);
    }

    /// How many accepted batches have not completed.
    pub fn held(&self) -> usize {
        lock(&self.held).len()
    }

    /// Completes up to `count` held batches, oldest first, with their answers; answers how many it
    /// completed.
    pub fn release(&self, count: usize) -> usize {
        let released: Vec<Held> = {
            let mut held = lock(&self.held);
            let count = count.min(held.len());
            held.drain(..count).collect()
        };
        let completed = released.len();
        for held in released {
            held.state.finish(held.outcome);
        }
        completed
    }

    /// Fails every held batch with `message`: the device that ran them was lost after it accepted
    /// them. Answers how many it failed.
    pub fn fail_held(&self, message: &str) -> usize {
        let failed: Vec<Held> = lock(&self.held).drain(..).collect();
        let count = failed.len();
        for held in failed {
            let fault: Arc<Fault> = Arc::from(Box::<Fault>::from(message.to_owned()));
            held.state.finish(Outcome::Failed(fault));
        }
        count
    }

    /// The outcome a batch completes with: the case's own answer, or each call completed by the
    /// token and extent rules.
    fn answer(&self, batch: &Batch) -> Outcome<BatchOutput> {
        if let Some(answer) = &self.answer {
            return answer(batch);
        }

        Outcome::Success(completed_until(batch, self.tokens, self.extent, self.end))
    }

    /// Accepts `batch` and answers its submission: completed now, or held.
    fn serve(&self, batch: &Batch) -> Result<Submission, Box<Fault>> {
        if let Some(message) = lock(&self.refusal).clone() {
            return Err(message.into());
        }

        let id = SubmissionId(self.next.fetch_add(1, Ordering::Relaxed));
        let outcome = self.answer(batch);
        if !*lock(&self.holding) {
            return Ok(Submission::completed(id, outcome));
        }

        let (submission, state) = Submission::pending(id);
        lock(&self.held).push_back(Held {
            batch: batch.id,
            state,
            outcome,
        });
        Ok(submission)
    }
}

/// `batch` completed as the fake completes it by `tokens` and `extent`: each call samples one
/// token and leaves the extent its last row states. A case's own [`Answer`] starts from this to
/// change some calls and keep the rest.
pub fn completed(batch: &Batch, tokens: Tokens, extent: Extent) -> BatchOutput {
    completed_until(batch, tokens, extent, None)
}

/// [`completed`], with every row at or past `end`'s logical position sampling `end`'s token.
fn completed_until(
    batch: &Batch,
    tokens: Tokens,
    extent: Extent,
    end: Option<EndOfText>,
) -> BatchOutput {
    let calls = (0..batch.calls.len())
        .map(|index| complete(batch, index, tokens, extent, end))
        .collect();
    BatchOutput {
        calls,
        timing: TimingCounters::default(),
        forward_stats: None,
    }
}

/// Call `index` of `batch`, completed with one sampled token.
///
/// The call's last row is the one that samples: a call's rows span `BatchRow::rows`, and a
/// batch whose columns state no rows for the call (a media batch) reads row `index`.
fn complete(
    batch: &Batch,
    index: usize,
    tokens: Tokens,
    extent: Extent,
    end: Option<EndOfText>,
) -> CallOutput {
    let rows = &batch.columns.rows;
    let row = rows
        .calls
        .get(index)
        .and_then(|call| call.rows.end.checked_sub(1))
        .map_or(index, |row| row as usize);
    let seen = rows.seq_lens.get(row).copied().unwrap_or(0).max(0) as u32;

    let token = match tokens {
        Tokens::Draw => batch
            .columns
            .draws
            .get(row)
            .copied()
            .map_or(SAMPLED + index as u32, draw_token),
        Tokens::Positional => SAMPLED + index as u32,
        Tokens::Extent => SAMPLED + seen % VOCABULARY,
    };
    let ended = end.filter(|end| {
        batch
            .columns
            .draws
            .get(row)
            .is_some_and(|draw| draw.position >= end.position)
    });
    let token = ended.map_or(token, |end| end.token);
    let state = match extent {
        // A text call carries no denoising trajectory: its step index stays at zero.
        Extent::Stated => RowState {
            step_index: 0,
            cache_computed_len: seen,
            cache_visible_len: seen,
        },
        Extent::OnePast => RowState {
            step_index: seen + 1,
            cache_computed_len: seen + 1,
            cache_visible_len: seen + 1,
        },
    };

    // A causal-LM step reports the raw log-probability of every token it samples, asked for or
    // not (`CallOutput::token_logprobs`); the scheduler keeps them for the requests that ask. A
    // row whose logits the mechanism did not ask for (`logits_rows` states -1 for it: a chunk that
    // ends no prompt) samples nothing, as a device's projection skips it.
    let samples = rows.logits_rows.get(row).is_none_or(|&logits| logits >= 0);
    if !samples {
        return CallOutput::completed(batch.calls[index], Vec::new(), state);
    }
    let mut output = CallOutput::completed(batch.calls[index], vec![token], state);
    output.token_logprobs = vec![crate::text::LOGPROB];
    output
}

impl Batches for FakeBatches {
    fn submit(&self, batch: Batch) -> Result<Submission, Box<Fault>> {
        lock(&self.submitted).push(batch.clone());
        let in_flight = lock(&self.held).iter().map(|held| held.batch).collect();
        let served = self.serve(&batch);
        if served.is_ok() {
            lock(&self.accepted).push(Accepted { batch, in_flight });
        }
        served
    }

    fn shadow(&self, batch: Batch) -> Result<Submission, Box<Fault>> {
        lock(&self.shadows).push(batch.clone());
        self.serve(&batch)
    }

    fn media(
        &self,
        _phase: ipc::PhaseId,
        computation: &Computation,
    ) -> Result<Option<MediaExtent>, Box<Fault>> {
        Ok(self.media.as_ref().and_then(|media| media(computation)))
    }

    fn devices(&self, _batch: &Batch) -> Result<Vec<usize>, Box<Fault>> {
        Ok((0..self.devices).collect())
    }

    fn device_count(&self) -> usize {
        self.devices
    }

    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }

    fn operate(&self, operation: &EpochOperation) -> Result<Vec<String>, Box<Fault>> {
        match &self.operate {
            Some(operate) => operate(operation),
            None => Ok(Vec::new()),
        }
    }

    /// Records the copies and completes them at once, when a rank of the fake's pools maps a host
    /// latent pool; refuses them otherwise, as a worker that maps none does. The fake copies no
    /// bytes: what a case reads is what the mechanism asked to have stored.
    fn offload_latents(&self, offloads: Vec<ipc::LatentOffload>) -> Result<Submission, Box<Fault>> {
        let maps_pool = self
            .inventory
            .iter()
            .flat_map(|inventory| &inventory.ranks)
            .any(|rank| rank.latent_pool.is_some());
        if !maps_pool {
            return Err("this worker maps no host latent pool to offload a trajectory to".into());
        }

        lock(&self.latent_offloads).extend(offloads);
        let id = SubmissionId(self.next.fetch_add(1, Ordering::Relaxed));
        Ok(Submission::completed(
            id,
            Outcome::Success(BatchOutput::default()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_draw_token_depends_on_the_stream_and_the_position_alone() {
        let at = |key, position| draw_token(DrawAddress { key, position });

        assert_eq!(
            at(7, 3),
            at(7, 3),
            "the same address samples the same token"
        );
        assert_ne!(at(7, 3), at(7, 4), "another position samples another token");
        assert_ne!(at(7, 3), at(8, 3), "another stream samples another token");
        for position in 0..64 {
            let token = at(11, position);
            assert!(
                (SAMPLED..SAMPLED + VOCABULARY).contains(&token),
                "{token} lies in the sampled window"
            );
        }
    }
}
