//! The fake executor: a scheduler's executor at its public boundary
//! ([`engine::executor::Executor`]), over device block tables kept on the host and batches
//! answered by a fixed rule (`specs/scheduler-design.md` §8, SD).
//!
//! A [`FakeExecutor`] stands for the executor of one scheduler's worker groups and keeps one
//! record per group, as the engine's workers of a group keep their device block tables:
//!
//! - `submit` numbers the batch among its group's batches (its sequence), starts the copies
//!   queued on the group, and answers the batch with [`Message::Output`]: at once, or when the
//!   case releases it ([`FakeExecutor::hold`]). Answering ends the batch: the group's `completed`
//!   sequence passes it once every earlier batch has ended, and the `Finish` of each copy it ran
//!   returns, stamped, in a [`Message::Submit`] ahead of the output.
//! - `map` writes the entries a completion gives into the group's tables at the positions its
//!   request named, under the slot it is handed, or queues the copies of a `Copy`. The entries
//!   of a `Load` or a `Clone` appear with the answer to its `Finish`.
//! - `stamp` applies a `Release`, `Detach`, `Commit`, `Transfer` or give-up `Finish` to the
//!   group's tables, stamps it with the sequence of the group's last batch, and returns it in a
//!   [`Message::Submit`].
//!
//! A batch's output follows one rule: a call samples [`crate::draw_token`] of its last row's
//! draw, and nothing when that row's logits are not wanted ([`output_of`]). A case varies where text ends, the one token every call samples, the
//! whole output, which calls fail, which batches are held, and whether copies fail.
//!
//! What the fake records answers the scheduler's side of A1 to A4 (`specs/buffer-pool.md` §9,
//! §17; SD §10, L5). Every breach found is a [`FakeExecutor::violations`] entry:
//!
//! - a completion handed to `map` is the buffer pool's answer to the request handed with it
//!   ([`crate::InProcessPool`] tells the fake each request it takes and each completion it
//!   writes); an entry is written only at a free position, and never from the completion of a
//!   request the buffer pool took before a `Release`, `Detach`, `Commit` or `Transfer` that names
//!   the entry and that the group has applied;
//! - a `Release`, `Detach` or `Transfer` reaches the submission queue only once no group holds
//!   its entries, a `Commit` once no group holds its entry writable, and each with the stamp of
//!   every group that applied it; a `Finish` with the stamp of the group that ran or gave up its
//!   copy;
//! - a token row writes cache positions, and a checkpoint row keeps state, only in blocks its
//!   slot holds writable.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::Write as _;
use std::sync::{Arc, Mutex, MutexGuard};

use crossbeam_channel::Sender;
use engine::Error;
use engine::batch::{
    Batch, BatchOutput, CallOutput, CallStatus, ErrorCode, Phase, Rows, StateDirection,
    TimingCounters,
};
use engine::executor::Executor;
use engine::scheduler::Message;
use proto::request::RequestId;
use service::lock;
use storage::runtime::{self, Completion, GroupId, OwnerId, Request, Run, Stamps};

use crate::text::{EOS, EndOfText, SAMPLED, draw_token, scored};

/// A case's own response to a batch, in place of the fake's rule. It runs under the fake's lock
/// and must not call the fake.
pub type OutputFn = Arc<dyn Fn(&Batch) -> Result<BatchOutput, Error> + Send + Sync>;

/// The key of an entry of a group's tables: the slot it is kept under, its class and its
/// position in the owner's table of that class.
pub type Place = (u32, u8, u32);

/// One entry of a group's device block tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The runs of the entry's block: one, or one per backing rank of a product.
    pub runs: Vec<Run>,
    /// Whether a batch may write it: it came from `Allocated`, `Cloned` or a private block of
    /// `Transferred`, and no `Commit` has marked it.
    pub writable: bool,
    /// The owner the completion's request named.
    pub owner: OwnerId,
    /// The number of the request that named the entry's position: the request answered, or for
    /// the destination of a `Load` or a `Clone`, that request rather than its `Finish`.
    pub from: runtime::RequestId,
}

/// The executor of one scheduler's worker groups. A clone is another handle on the same
/// executor: the scheduler holds one, the case and the [`crate::InProcessPool`] others.
#[derive(Clone)]
pub struct FakeExecutor(Arc<Mutex<State>>);

/// A copy `map` queued: the request answered `Copy` and its number, the destination runs, the
/// sequence of the batch that started it, and whether it ended or was given up.
struct Copying {
    id: runtime::RequestId,
    request: Request,
    to: Vec<Run>,
    started: Option<u64>,
    ended: bool,
    given_up: bool,
}

/// One group's workers, as the fake keeps them: their tables, their copies and their batch
/// sequences.
#[derive(Default)]
struct Tables {
    entries: BTreeMap<Place, Entry>,
    copies: Vec<Copying>,
    /// The requests the group applied, without stamps, each with its number once the buffer pool
    /// took it.
    applied: Vec<(Request, Option<u64>)>,
    /// The sequence of the last batch launched; batches are numbered from 1.
    launched: u64,
    /// Batches ended beyond `completed`.
    ended: BTreeSet<u64>,
    /// The last batch that ended with every earlier one ended.
    completed: u64,
}

struct State {
    inbox: Sender<Message>,
    groups: BTreeMap<GroupId, Tables>,
    /// The classes a prefill, decode or verification row writes, each with its block's tokens.
    cache: Vec<(u8, u32)>,
    /// The class a refill row writes, with its block's tokens.
    draft: Option<(u8, u32)>,
    /// The class of recurrent-state checkpoints.
    checkpoints: Option<u8>,

    end: Option<EndOfText>,
    /// The one token every call samples in place of its draw's.
    token: Option<u32>,
    /// The tokens each request's text holds, and the position of the first token sampled on each
    /// Philox stream.
    text_len: Option<u64>,
    text_starts: HashMap<u64, u64>,
    output: Option<OutputFn>,
    failing: HashMap<RequestId, (ErrorCode, Option<String>)>,
    holding: bool,
    fail_copies: bool,
    held: VecDeque<(u64, Batch)>,

    submitted: Vec<Batch>,
    mapped: Vec<(GroupId, u32, runtime::RequestId, Request, Completion)>,
    stamps: Vec<(GroupId, Request)>,
    /// Stamps not yet seen in the submission queue: the group and the request without stamps.
    unchecked: Vec<(GroupId, Request)>,
    /// What the buffer pool took and wrote, by request number.
    requests: HashMap<u64, Request>,
    answers: HashMap<u64, Completion>,
    violations: Vec<String>,
}

impl FakeExecutor {
    /// The executor of `groups`, which sends its messages to the scheduler's `inbox`. A
    /// prefill, decode or verification row writes every class of `cache`, each with its block's
    /// tokens, at the block of each cache position.
    pub fn new(groups: &[GroupId], cache: Vec<(u8, u32)>, inbox: Sender<Message>) -> Self {
        Self(Arc::new(Mutex::new(State {
            inbox,
            groups: groups
                .iter()
                .map(|group| (*group, Tables::default()))
                .collect(),
            cache,
            draft: None,
            checkpoints: None,
            end: None,
            token: None,
            text_len: None,
            text_starts: HashMap::new(),
            output: None,
            failing: HashMap::new(),
            holding: false,
            fail_copies: false,
            held: VecDeque::new(),
            submitted: Vec::new(),
            mapped: Vec::new(),
            stamps: Vec::new(),
            unchecked: Vec::new(),
            requests: HashMap::new(),
            answers: HashMap::new(),
            violations: Vec::new(),
        })))
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        lock(&self.0)
    }

    /// A refill row writes `class`, whose blocks hold `tokens` tokens.
    #[must_use]
    pub fn with_draft(self, class: u8, tokens: u32) -> Self {
        self.lock().draft = Some((class, tokens));
        self
    }

    /// A checkpoint a row keeps (`StateDirection::Keep`) is written in `class`.
    #[must_use]
    pub fn with_checkpoints(self, class: u8) -> Self {
        self.lock().checkpoints = Some(class);
        self
    }

    /// Every row whose draw lies at or past `end.position` samples `end.token`.
    #[must_use]
    pub fn with_end(self, end: EndOfText) -> Self {
        self.lock().end = Some(end);
        self
    }

    /// Every call samples `token` in place of its draw's, short of the end of text: a model whose
    /// sampling a request restricted to one token (`allowed_token_ids`), which a case reads back
    /// as text.
    #[must_use]
    pub fn with_token(self, token: u32) -> Self {
        self.lock().token = Some(token);
        self
    }

    /// Every request's text ends with [`EOS`] after `tokens` sampled tokens, counted from the
    /// first token sampled on its Philox stream.
    #[must_use]
    pub fn with_text_len(self, tokens: u64) -> Self {
        self.lock().text_len = Some(tokens);
        self
    }

    /// Answers every batch with `output` instead of the fake's rule.
    #[must_use]
    pub fn with_output(self, output: OutputFn) -> Self {
        self.lock().output = Some(output);
        self
    }

    /// Fails every later call of request `id` with an error code and, for an `InputError`, the
    /// reason the rank states; `None` stops failing it.
    pub fn fail_calls(&self, id: RequestId, failure: Option<(ErrorCode, Option<String>)>) {
        let mut state = self.lock();
        match failure {
            Some(failure) => state.failing.insert(id, failure),
            None => state.failing.remove(&id),
        };
    }

    /// Ends every later copy with `ok: false`, or with `ok: true` again.
    pub fn fail_copies(&self, fail: bool) {
        self.lock().fail_copies = fail;
    }

    /// Holds every later batch until [`Self::release`] answers it: a batch still on the device.
    pub fn hold(&self) {
        self.lock().holding = true;
    }

    /// Answers every held batch and every later batch at once again.
    pub fn resume(&self) {
        self.lock().holding = false;
        self.release(usize::MAX);
    }

    /// How many batches are held.
    pub fn held(&self) -> usize {
        self.lock().held.len()
    }

    /// Answers up to `count` held batches, oldest first; returns how many.
    pub fn release(&self, count: usize) -> usize {
        let mut state = self.lock();
        let count = count.min(state.held.len());
        for _ in 0..count {
            if let Some((sequence, batch)) = state.held.pop_front() {
                let outcome = state.outcome(&batch);
                state.end(sequence, batch, outcome);
            }
        }
        count
    }

    /// Ends every held batch with `Error::Executor(message)`: its device was lost after it
    /// launched them. Returns how many.
    pub fn fail_held(&self, message: &str) -> usize {
        let mut state = self.lock();
        let held: Vec<(u64, Batch)> = state.held.drain(..).collect();
        let count = held.len();
        for (sequence, batch) in held {
            state.end(sequence, batch, Err(Error::Executor(message.to_owned())));
        }
        count
    }

    /// Every batch submitted, in submission order.
    pub fn submitted(&self) -> Vec<Batch> {
        self.lock().submitted.clone()
    }

    /// Every completion handed to `map`, in order: the group, the slot, the request's number,
    /// the request and the completion.
    pub fn mapped(&self) -> Vec<(GroupId, u32, runtime::RequestId, Request, Completion)> {
        self.lock().mapped.clone()
    }

    /// Every request a group returned stamped, in order: the scheduler's and the `Finish` of
    /// each copy a group ran.
    pub fn stamps(&self) -> Vec<(GroupId, Request)> {
        self.lock().stamps.clone()
    }

    /// The entries of `group`'s tables.
    pub fn entries(&self, group: GroupId) -> BTreeMap<Place, Entry> {
        self.lock()
            .groups
            .get(&group)
            .map(|tables| tables.entries.clone())
            .unwrap_or_default()
    }

    /// `group`'s `completed` sequence: the last batch that ended with every earlier one ended.
    pub fn completed(&self, group: GroupId) -> u64 {
        self.lock()
            .groups
            .get(&group)
            .map_or(0, |tables| tables.completed)
    }

    /// Every breach of A1 to A4 found so far.
    pub fn violations(&self) -> Vec<String> {
        self.lock().violations.clone()
    }

    /// The buffer pool took `request` under `id`: the request a completion of `id` answers. A
    /// request the workers stamp is checked against A2 here, as it enters the submission queue.
    pub(crate) fn taken(&self, id: runtime::RequestId, request: &Request) {
        let mut state = self.lock();
        state.requests.insert(id.0, request.clone());

        let plain = unstamped(request);
        for tables in state.groups.values_mut() {
            let record = tables
                .applied
                .iter_mut()
                .find(|(applied, number)| number.is_none() && *applied == plain);
            if let Some((_, number)) = record {
                *number = Some(id.0);
            }
        }
        let mut stamped = request.clone();
        let Some(after) = stamps(&mut stamped).map(std::mem::take) else {
            return;
        };

        // Its entries left every table (a commit's became read-only) before it was submitted.
        let mut breaches = Vec::new();
        for (group, tables) in &state.groups {
            let holds = tables.entries.iter().any(|(place, entry)| {
                names(request, entry.owner, place.1, place.2)
                    && (entry.writable || !matches!(request, Request::Commit { .. }))
            });
            if holds {
                breaches.push(format!(
                    "{request:?} reached the submission queue while group {} holds its entries",
                    group.0
                ));
            }
        }

        // It carries the stamp of every group that applied it.
        state.unchecked.retain(|(group, applied)| {
            if *applied != plain {
                return true;
            }
            if !after.iter().any(|(stamped, _)| stamped == group) {
                breaches.push(format!(
                    "{request:?} reached the submission queue without the stamp of group {}",
                    group.0
                ));
            }
            false
        });
        state.violations.extend(breaches);
    }

    /// The buffer pool answered request `id` with `completion`.
    pub(crate) fn answered(&self, id: runtime::RequestId, completion: &Completion) {
        self.lock().answers.insert(id.0, completion.clone());
    }
}

impl State {
    fn violate(&mut self, breach: String) {
        self.violations.push(breach);
    }

    fn send(&self, message: Message) {
        // A scheduler that is gone takes nothing more.
        let _ = self.inbox.send(message);
    }

    /// The output of `batch`: the case's own, or the fake's rule with its knobs.
    fn outcome(&mut self, batch: &Batch) -> Result<BatchOutput, Error> {
        if let Some(output) = &self.output {
            return output(batch);
        }
        let mut output = output_of(batch, self.end);

        for (index, call) in output.calls.iter_mut().enumerate() {
            let draw = batch.columns.draws.get(sampling_row(batch, index));
            // The fixed token takes the place of what the draw chose, short of the end of text.
            let ended = draw
                .zip(self.end)
                .is_some_and(|(draw, end)| draw.position >= end.position);
            if let Some(token) = self.token
                && !call.tokens.is_empty()
                && !ended
            {
                call.tokens = vec![token];
                call.token_logprobs = vec![scored(token)];
            }
            if let Some(tokens) = self.text_len
                && !call.tokens.is_empty()
                && let Some(draw) = draw
            {
                let start = *self.text_starts.entry(draw.key).or_insert(draw.position);
                if draw.position >= start.saturating_add(tokens) {
                    call.tokens = vec![EOS];
                    call.token_logprobs = vec![scored(EOS)];
                }
            }
            if let Some((code, message)) = batch
                .requests
                .get(index)
                .and_then(|id| self.failing.get(id))
            {
                *call = failed(*code, message.clone());
            }
        }
        Ok(output)
    }

    /// Ends the batch numbered `sequence` with `outcome`: `completed` passes it once every
    /// earlier batch has ended, the copies it ran return their stamped `Finish` (failed with a
    /// failed batch), then its output.
    fn end(&mut self, sequence: u64, batch: Batch, outcome: Result<BatchOutput, Error>) {
        let group = batch.group;
        let ok = !self.fail_copies && outcome.is_ok();
        let mut finished = Vec::new();
        if let Some(tables) = self.groups.get_mut(&group) {
            tables.ended.insert(sequence);
            while tables.ended.remove(&(tables.completed + 1)) {
                tables.completed += 1;
            }

            // A copy given up after it started is covered by the give-up's stamp and returns no
            // `Finish` of its own; a load or a clone that ended well awaits its answer's entries.
            for copy in &mut tables.copies {
                if copy.started == Some(sequence) && !copy.ended {
                    copy.ended = true;
                    if !copy.given_up {
                        finished.push(Request::Finish {
                            copy: copy.id,
                            ok,
                            after: vec![(group, tables.launched)],
                        });
                    }
                }
            }
            tables.copies.retain(|copy| {
                let fills = matches!(copy.request, Request::Load { .. } | Request::Clone { .. });
                !copy.ended || (fills && ok && !copy.given_up)
            });
        }

        for finish in &finished {
            self.stamps.push((group, finish.clone()));
            self.unchecked.push((group, unstamped(finish)));
        }
        if !finished.is_empty() {
            self.send(Message::Submit(finished));
        }
        self.send(Message::Output(batch, outcome));
    }

    /// The breaches of A4 in `batch`: a token row writing a cache position, or keeping a
    /// checkpoint, in a block its slot does not hold writable.
    fn unwritable(&self, batch: &Batch) -> Vec<String> {
        let Some(tables) = self.groups.get(&batch.group) else {
            return Vec::new();
        };
        let writable = |slot: u32, class: u8, position: u32| {
            tables
                .entries
                .get(&(slot, class, position))
                .is_some_and(|entry| entry.writable)
        };
        let classes = if batch.phase == Phase::Refill {
            self.draft.as_slice()
        } else {
            self.cache.as_slice()
        };

        let mut breaches = Vec::new();
        let rows = &batch.columns.rows;
        for call in &rows.calls {
            for token in offset(rows, call.rows.start)..offset(rows, call.rows.end) {
                let Some(position) = rows
                    .cache_positions
                    .get(token)
                    .and_then(|position| u32::try_from(*position).ok())
                else {
                    continue;
                };
                for &(class, tokens) in classes {
                    let block = position / tokens.max(1);
                    if !writable(call.slot, class, block) {
                        breaches.push(format!(
                            "batch {} writes position {position} of slot {} in block {block} of \
                             class {class}, which the slot does not hold writable",
                            batch.id.0, call.slot
                        ));
                    }
                }
            }
        }
        if let Some(class) = self.checkpoints {
            let kept = batch.columns.state_copies.iter();
            for copy in kept.filter(|copy| copy.direction == StateDirection::Keep) {
                if !writable(copy.slot, class, copy.position) {
                    breaches.push(format!(
                        "batch {} keeps a checkpoint at position {} of slot {}, which the slot \
                         does not hold writable",
                        batch.id.0, copy.position, copy.slot
                    ));
                }
            }
        }
        breaches
    }

    /// The entries `request`'s `completion` gives, with the owner and the number of the request
    /// that named their positions: `(class, position, runs, writable)` each. `None` for a
    /// completion that gives none; the copies of a `Copy` are queued here.
    #[expect(
        clippy::type_complexity,
        reason = "an entry is stated in the storage interface's own terms"
    )]
    fn given(
        &mut self,
        group: GroupId,
        id: runtime::RequestId,
        request: &Request,
        completion: &Completion,
    ) -> Option<(OwnerId, runtime::RequestId, Vec<(u8, u32, Vec<Run>, bool)>)> {
        let tables = self.groups.get_mut(&group)?;
        match (request, completion) {
            (
                Request::Allocate {
                    owner,
                    class,
                    at,
                    count,
                    ..
                },
                Completion::Allocated { runs, .. },
            ) => {
                // Each block has the same number of runs, one per backing rank.
                let each = (runs.len() / (*count).max(1) as usize).max(1);
                let blocks = runs.chunks(each).zip(*at..);
                let entries = blocks.map(|(runs, at)| (*class, at, runs.to_vec(), true));
                Some((*owner, id, entries.collect()))
            }
            (
                Request::Pin {
                    owner, class, at, ..
                },
                Completion::Pinned { runs, .. },
            ) => {
                let entries = runs.iter().zip(*at..);
                let entries = entries.map(|(run, at)| (*class, at, vec![*run], false));
                Some((*owner, id, entries.collect()))
            }
            (Request::Transfer { owner, .. }, Completion::Transferred { runs, .. }) => {
                let mut blocks: BTreeMap<(u8, u32), (Vec<Run>, bool)> = BTreeMap::new();
                for &(class, at, run, private) in runs {
                    let block = blocks.entry((class, at)).or_insert((Vec::new(), private));
                    block.0.push(run);
                }
                let entries = blocks.into_iter();
                let entries =
                    entries.map(|((class, at), (runs, private))| (class, at, runs, private));
                Some((*owner, id, entries.collect()))
            }
            (_, Completion::Copy { to, .. }) => {
                tables.copies.push(Copying {
                    id,
                    request: request.clone(),
                    to: to.clone(),
                    started: None,
                    ended: false,
                    given_up: false,
                });
                None
            }
            (
                Request::Finish { copy, .. },
                Completion::Loaded | Completion::Cloned | Completion::Duplicate,
            ) => {
                let at = tables
                    .copies
                    .iter()
                    .position(|copying| copying.id == *copy)?;
                let copying = tables.copies.remove(at);
                match copying.request {
                    // A loaded block is read-only, a duplicate too; a clone is the owner's to write.
                    Request::Load {
                        owner, class, at, ..
                    } => {
                        let entries = copying.to.iter().zip(at..);
                        let entries = entries.map(|(run, at)| (class, at, vec![*run], false));
                        Some((owner, copying.id, entries.collect()))
                    }
                    Request::Clone {
                        owner, class, at, ..
                    } => {
                        let writable = *completion == Completion::Cloned;
                        Some((owner, copying.id, vec![(class, at, copying.to, writable)]))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

impl Executor for FakeExecutor {
    fn submit(&self, batch: Batch) {
        let mut state = self.lock();
        let breaches = state.unwritable(&batch);
        state.violations.extend(breaches);
        state.submitted.push(batch.clone());

        let Some(tables) = state.groups.get_mut(&batch.group) else {
            let message = format!(
                "batch {} names group {}, which this executor does not serve",
                batch.id.0, batch.group.0
            );
            state.violate(message.clone());
            state.send(Message::Output(batch, Err(Error::Executor(message))));
            return;
        };
        tables.launched += 1;
        let sequence = tables.launched;
        for copy in tables
            .copies
            .iter_mut()
            .filter(|copy| copy.started.is_none())
        {
            copy.started = Some(sequence);
        }

        if state.holding {
            state.held.push_back((sequence, batch));
            return;
        }
        let outcome = state.outcome(&batch);
        state.end(sequence, batch, outcome);
    }

    fn map(
        &self,
        group: GroupId,
        slot: u32,
        id: runtime::RequestId,
        request: Request,
        completion: Completion,
    ) {
        let mut state = self.lock();
        state
            .mapped
            .push((group, slot, id, request.clone(), completion.clone()));
        let answers = state.requests.get(&id.0) == Some(&request)
            && state.answers.get(&id.0) == Some(&completion);
        if !answers {
            state.violate(format!(
                "group {} is handed {completion:?} for {request:?} as the answer to request {}, \
                 which the buffer pool did not answer so",
                group.0, id.0
            ));
        }
        if matches!(completion, Completion::Attached { .. }) {
            return;
        }

        let Some((owner, from, entries)) = state.given(group, id, &request, &completion) else {
            if !matches!(completion, Completion::Copy { .. }) {
                state.violate(format!(
                    "group {} is handed {completion:?} for {request:?}, which gives it no entry",
                    group.0
                ));
            }
            return;
        };

        let mut breaches = Vec::new();
        if let Some(tables) = state.groups.get_mut(&group) {
            for (class, position, runs, writable) in entries {
                // A1: a stamped request that names the entry and was taken after this completion's
                // request, or is not taken yet, makes the completion stale.
                let stale = tables.applied.iter().any(|(applied, number)| {
                    names(applied, owner, class, position) && number.is_none_or(|n| n > from.0)
                });
                if stale {
                    breaches.push(format!(
                        "group {} is handed the completion of request {} after a request that \
                         names position {position} of class {class} of owner {}",
                        group.0, from.0, owner.0
                    ));
                }
                let place = (slot, class, position);
                if tables.entries.contains_key(&place) {
                    breaches.push(format!(
                        "group {} is given position {position} of class {class} in slot {slot}, \
                         which holds an entry",
                        group.0
                    ));
                }
                let entry = Entry {
                    runs,
                    writable,
                    owner,
                    from,
                };
                tables.entries.insert(place, entry);
            }
        }
        state.violations.extend(breaches);
    }

    fn stamp(&self, group: GroupId, requests: Vec<Request>) {
        let mut state = self.lock();
        let mut stamped = Vec::with_capacity(requests.len());
        for mut request in requests {
            let Some(tables) = state.groups.get_mut(&group) else {
                state.violate(format!(
                    "group {} is not this executor's, which cannot stamp {request:?}",
                    group.0
                ));
                continue;
            };
            match &request {
                Request::Release { .. } | Request::Detach { .. } | Request::Transfer { .. } => {
                    let named = |place: &Place, entry: &Entry| {
                        names(&request, entry.owner, place.1, place.2)
                    };
                    tables.entries.retain(|place, entry| !named(place, entry));
                }
                Request::Commit { .. } => {
                    for (place, entry) in &mut tables.entries {
                        if names(&request, entry.owner, place.1, place.2) {
                            entry.writable = false;
                        }
                    }
                }
                // A give-up: a copy no batch started never runs, one that started runs on under
                // this stamp, and one whose own `Finish` returned is ended already.
                Request::Finish {
                    copy, ok: false, ..
                } => {
                    if let Some(at) = tables.copies.iter().position(|queued| queued.id == *copy) {
                        if tables.copies[at].started.is_none() {
                            tables.copies.remove(at);
                        } else if !tables.copies[at].ended {
                            tables.copies[at].given_up = true;
                        }
                    }
                }
                _ => {
                    state.violate(format!(
                        "group {} is handed {request:?}, which a worker does not stamp",
                        group.0
                    ));
                    continue;
                }
            }

            tables.applied.push((unstamped(&request), None));
            let launched = tables.launched;
            if let Some(after) = stamps(&mut request) {
                after.push((group, launched));
            }
            state.stamps.push((group, request.clone()));
            state.unchecked.push((group, unstamped(&request)));
            stamped.push(request);
        }
        if !stamped.is_empty() {
            state.send(Message::Submit(stamped));
        }
    }
}

/// `batch` answered by the fake's rule: a token call samples [`draw_token`] of its last row's
/// draw ([`SAMPLED`] plus its place when the row has none), `end.token` from `end.position` on,
/// and nothing when that row's logits are not wanted; it reports the keys that row sees as its
/// extent and scores every prompt target its tokens name. A media call completes its denoising
/// step, opens its trajectory at the step its row states, and on the final muxing call (the one
/// that hands no round) publishes the artifact into a POSIX shared-memory object. A case's own
/// [`OutputFn`] starts from this to change some calls and keep the rest.
pub fn output_of(batch: &Batch, end: Option<EndOfText>) -> BatchOutput {
    let token = matches!(
        batch.phase,
        Phase::Prefill | Phase::Decode | Phase::Verify | Phase::Refill
    );
    let calls = (0..batch.requests.len()).map(|index| {
        if token {
            sampled(batch, index, end)
        } else {
            media(batch, index)
        }
    });
    BatchOutput {
        calls: calls.collect(),
        timing: TimingCounters::default(),
        forward_stats: None,
    }
}

/// Call `index` of a token batch.
fn sampled(batch: &Batch, index: usize, end: Option<EndOfText>) -> CallOutput {
    let columns = &batch.columns;
    let rows = &columns.rows;
    let row = sampling_row(batch, index);
    let seen = rows
        .seq_lens
        .get(row)
        .map_or(0, |seen| u32::try_from(*seen).unwrap_or(0));
    let mut output = completed(Vec::new(), seen);

    // The prompt tokens the call's tokens score, in input order.
    if let Some(call) = rows.calls.get(index) {
        let tokens = offset(rows, call.rows.start)..offset(rows, call.rows.end);
        let targets = tokens.filter_map(|token| columns.targets.get(token).copied());
        let targets = targets.filter_map(|target| u32::try_from(target).ok());
        output.prompt_logprobs = targets.map(scored).collect();
    }

    if rows.logits_rows.get(row).is_some_and(|logits| *logits < 0) {
        return output;
    }
    let token = match (columns.draws.get(row), end) {
        (Some(draw), Some(end)) if draw.position >= end.position => end.token,
        (Some(draw), _) => draw_token(*draw),
        (None, _) => SAMPLED + u32::try_from(index).unwrap_or(0),
    };
    output.tokens = vec![token];
    output.token_logprobs = vec![scored(token)];
    output
}

/// Call `index` of a media batch, from its row.
fn media(batch: &Batch, index: usize) -> CallOutput {
    let mut output = completed(Vec::new(), 0);
    let Some(row) = batch.media.get(index) else {
        return output;
    };
    match batch.phase {
        Phase::Denoise => output.step_index = row.step + 1,
        Phase::LatentPrepare => output.step_index = row.step,
        Phase::Mux if row.rounds == 0 => {
            let payload = format!("uniserve-fake-video request={}", row.request.0);
            match publish(payload.as_bytes()) {
                Ok(media) => output.media_output = Some(media),
                Err(_) => return failed(ErrorCode::ResourceError, None),
            }
        }
        _ => {}
    }
    output
}

/// A call that completed with `tokens` and the extent `seen`.
fn completed(tokens: Vec<u32>, seen: u32) -> CallOutput {
    CallOutput {
        status: CallStatus::Ok,
        error_code: None,
        error_message: None,
        tokens,
        step_index: 0,
        cache_computed_len: seen,
        cache_visible_len: seen,
        token_logprobs: Vec::new(),
        prompt_logprobs: Vec::new(),
        media_output: None,
        draft_token_ids: Vec::new(),
    }
}

/// Publishes `bytes` as a POSIX shared-memory object, as a worker's media host relinquishes an
/// artifact for the scheduler to take.
fn publish(bytes: &[u8]) -> Result<engine::batch::MediaOutput, String> {
    let storage = || -> std::io::Result<String> {
        let mut storage = tempfile::Builder::new()
            .prefix("uniserve-fake-video-")
            .tempfile_in("/dev/shm")?;
        storage.write_all(bytes)?;
        let (_file, path) = storage.keep().map_err(|error| error.error)?;
        path.file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned)
            .ok_or_else(|| std::io::Error::other("the shared-memory object has no printable name"))
    };
    let name = storage().map_err(|error| format!("publishing the artifact: {error}"))?;

    Ok(engine::batch::MediaOutput {
        handle: engine::batch::ArtifactHandle::PosixShm { name },
        bytes: bytes.len() as u64,
    })
}

/// A call that failed with `code`; only an input error states its reason.
fn failed(code: ErrorCode, message: Option<String>) -> CallOutput {
    CallOutput {
        status: CallStatus::Error,
        error_code: Some(code),
        error_message: message.filter(|_| code == ErrorCode::InputError),
        ..completed(Vec::new(), 0)
    }
}

/// The row of call `index` that samples: the last of the rows the call owns, or row `index`
/// when the columns state no rows for it.
fn sampling_row(batch: &Batch, index: usize) -> usize {
    let call = batch.columns.rows.calls.get(index);
    let last = call.and_then(|call| call.rows.end.checked_sub(1));
    last.map_or(index, |row| row as usize)
}

/// The first token of row `row`, from the packed query offsets.
fn offset(rows: &Rows, row: u32) -> usize {
    let offset = rows.query_offsets.get(row as usize).copied().unwrap_or(0);
    usize::try_from(offset).unwrap_or(0)
}

/// Whether `request` names the entry of `owner` at `position` of `class`: a `Release` its
/// positions, a `Commit` its one, a `Detach` and a `Transfer` every entry of the owner.
fn names(request: &Request, owner: OwnerId, class: u8, position: u32) -> bool {
    match *request {
        Request::Release {
            owner: named,
            class: of,
            at,
            count,
            ..
        } => named == owner && of == class && position.checked_sub(at).is_some_and(|n| n < count),
        Request::Commit {
            owner: named,
            class: of,
            at,
            ..
        } => named == owner && of == class && at == position,
        Request::Detach { owner: named, .. } | Request::Transfer { owner: named, .. } => {
            named == owner
        }
        _ => false,
    }
}

/// The stamps of a request a worker stamps; `None` for every other request.
fn stamps(request: &mut Request) -> Option<&mut Stamps> {
    match request {
        Request::Detach { after, .. }
        | Request::Release { after, .. }
        | Request::Commit { after, .. }
        | Request::Transfer { after, .. }
        | Request::Finish { after, .. } => Some(after),
        _ => None,
    }
}

/// `request` without its stamps: what every group is handed and returns, stamps apart.
fn unstamped(request: &Request) -> Request {
    let mut request = request.clone();
    if let Some(after) = stamps(&mut request) {
        after.clear();
    }
    request
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::{Receiver, unbounded};
    use engine::batch::philox::DrawAddress;
    use engine::batch::{BatchId, Columns, Row, Shape, Variant};
    use storage::Key;

    use super::*;

    const A: GroupId = GroupId(0);
    /// The KV class, four tokens a block.
    const KV: u8 = 0;
    const OWNER: OwnerId = OwnerId(1);
    const SLOT: u32 = 3;

    fn fake() -> (FakeExecutor, Receiver<Message>) {
        let (inbox, messages) = unbounded();
        (FakeExecutor::new(&[A], vec![(KV, 4)], inbox), messages)
    }

    /// A token batch of one call per `(request, slot, cache positions, draw position)`: one row
    /// each, sampling on the request's own stream at the draw position.
    fn batch(id: u64, phase: Phase, calls: &[(u64, u32, Vec<i32>, u64)]) -> Batch {
        let mut rows = Rows {
            query_offsets: vec![0],
            ..Rows::default()
        };
        let mut draws = Vec::new();
        for (index, (request, slot, positions, draw)) in (0..).zip(calls) {
            rows.calls.push(Row {
                slot: *slot,
                kind: phase,
                cache_group: 0,
                rows: index..index + 1,
            });
            rows.positions.extend(positions);
            rows.cache_positions.extend(positions);
            rows.seq_lens
                .push(positions.last().map_or(0, |last| last + 1));
            rows.query_offsets.push(rows.cache_positions.len() as i32);
            rows.logits_rows.push(index as i32);
            draws.push(DrawAddress {
                key: *request,
                position: *draw,
            });
        }
        let tokens = rows.cache_positions.len() as u32;
        Batch {
            id: BatchId(id),
            group: A,
            phase,
            generation: 0,
            shape: Shape::prefill(calls.len() as u32, tokens, Variant::DecodeSampled),
            requests: calls.iter().map(|call| RequestId(call.0)).collect(),
            columns: Columns {
                rows,
                draws,
                ..Columns::default()
            },
            media: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    fn output(messages: &Receiver<Message>) -> (Batch, Result<BatchOutput, Error>) {
        match messages.try_recv() {
            Ok(Message::Output(batch, output)) => (batch, output),
            other => panic!("expected an output, got {other:?}"),
        }
    }

    fn submitted(messages: &Receiver<Message>) -> Vec<Request> {
        match messages.try_recv() {
            Ok(Message::Submit(requests)) => requests,
            other => panic!("expected stamped requests, got {other:?}"),
        }
    }

    /// Plays the buffer pool: request `id` was taken and answered `completion`.
    fn answer(fake: &FakeExecutor, id: u64, request: &Request, completion: &Completion) {
        fake.taken(runtime::RequestId(id), request);
        fake.answered(runtime::RequestId(id), completion);
    }

    /// Plays the scheduler: the pool answered request `id`, and the answer is mapped on `A`.
    fn mapped(fake: &FakeExecutor, id: u64, request: Request, completion: Completion) {
        answer(fake, id, &request, &completion);
        fake.map(A, SLOT, runtime::RequestId(id), request, completion);
    }

    /// The owner attached in `SLOT`, holding writable blocks at positions 0 and 1 (units 5 and 6)
    /// and a pinned one at position 2 (unit 7).
    fn attached(fake: &FakeExecutor) {
        let attach = Request::Attach {
            owner: OWNER,
            group: A,
        };
        mapped(fake, 1, attach, Completion::Attached { slot: SLOT });
        let allocate = Request::Allocate {
            owner: OWNER,
            class: KV,
            at: 0,
            count: 2,
            units: 0,
        };
        let allocated = Completion::Allocated {
            runs: vec![(0, 5, 1), (0, 6, 1)],
            evicted: Vec::new(),
        };
        mapped(fake, 2, allocate, allocated);
        let pin = Request::Pin {
            owner: OWNER,
            class: KV,
            at: 2,
            keys: vec![Key([1; 32])],
        };
        let pinned = Completion::Pinned {
            runs: vec![(0, 7, 1)],
            host: 0,
        };
        mapped(fake, 3, pin, pinned);
    }

    /// The stamps a request carries.
    fn after(request: &Request) -> Stamps {
        stamps(&mut request.clone()).cloned().unwrap_or_default()
    }

    fn release(at: u32) -> Request {
        Request::Release {
            owner: OWNER,
            class: KV,
            at,
            count: 1,
            after: Vec::new(),
        }
    }

    #[test]
    fn a_call_samples_its_last_rows_draw_and_text_ends_where_it_is_told() {
        let (fake, messages) = fake();
        let fake = fake.with_end(EndOfText::at(20));
        let mut decode = batch(
            1,
            Phase::Decode,
            &[(1, 1, vec![9], 10), (2, 2, vec![19], 20)],
        );
        fake.submit(decode.clone());

        let (_, answered) = output(&messages);
        let answered = answered.unwrap();
        answered.check(&decode).unwrap();
        let drawn = draw_token(DrawAddress {
            key: 1,
            position: 10,
        });
        assert_eq!(answered.calls[0].tokens, vec![drawn]);
        assert_eq!(answered.calls[0].cache_visible_len, 10);
        assert_eq!(answered.calls[1].tokens, vec![EOS], "the text ends at 20");

        // A chunk whose logits are not wanted samples nothing.
        decode.id = BatchId(2);
        decode.columns.rows.logits_rows[0] = -1;
        fake.submit(decode);
        let (_, answered) = output(&messages);
        assert!(answered.unwrap().calls[0].tokens.is_empty());
    }

    #[test]
    fn a_held_batch_is_answered_on_release_and_the_group_completes_it_then() {
        let (fake, messages) = fake();
        fake.hold();
        fake.submit(batch(1, Phase::Decode, &[(1, 1, Vec::new(), 1)]));
        fake.submit(batch(2, Phase::Decode, &[(1, 1, Vec::new(), 2)]));
        assert!(messages.try_recv().is_err());
        assert_eq!((fake.held(), fake.completed(A)), (2, 0));

        assert_eq!(fake.release(1), 1);
        let (answered, outcome) = output(&messages);
        assert_eq!((answered.id, outcome.is_ok()), (BatchId(1), true));
        assert_eq!(fake.completed(A), 1);

        assert_eq!(fake.fail_held("the device was lost"), 1);
        let (answered, outcome) = output(&messages);
        assert_eq!(answered.id, BatchId(2));
        assert!(
            matches!(outcome, Err(Error::Executor(message)) if message == "the device was lost")
        );
        assert_eq!(fake.completed(A), 2);
    }

    #[test]
    fn a_failing_request_fails_its_calls_with_their_code() {
        let (fake, messages) = fake();
        let refusal = Some((ErrorCode::InputError, Some("bad image".to_owned())));
        fake.fail_calls(RequestId(2), refusal);
        let decode = batch(
            1,
            Phase::Decode,
            &[(1, 1, Vec::new(), 1), (2, 2, Vec::new(), 1)],
        );
        fake.submit(decode.clone());

        let answered = output(&messages).1.unwrap();
        answered.check(&decode).unwrap();
        assert_eq!(answered.calls[0].status, CallStatus::Ok);
        assert_eq!(answered.calls[1].status, CallStatus::Error);
        assert_eq!(answered.calls[1].error_code, Some(ErrorCode::InputError));
        assert_eq!(
            answered.calls[1].error_message.as_deref(),
            Some("bad image")
        );
    }

    #[test]
    fn entries_go_where_their_request_named_and_leave_with_a_stamped_request() {
        let (fake, messages) = fake();
        attached(&fake);
        let entries = fake.entries(A);
        let expected = |unit, writable, from| Entry {
            runs: vec![(0, unit, 1)],
            writable,
            owner: OWNER,
            from: runtime::RequestId(from),
        };
        assert_eq!(entries[&(SLOT, KV, 0)], expected(5, true, 2));
        assert_eq!(entries[&(SLOT, KV, 1)], expected(6, true, 2));
        assert_eq!(entries[&(SLOT, KV, 2)], expected(7, false, 3));

        // One batch launched: what is stamped now carries its sequence.
        fake.submit(batch(1, Phase::Decode, &[(1, SLOT, vec![4], 5)]));
        output(&messages).1.unwrap();
        let commit = Request::Commit {
            owner: OWNER,
            class: KV,
            at: 0,
            key: Key([2; 32]),
            tokens: 4,
            after: Vec::new(),
        };
        fake.stamp(A, vec![release(1), commit]);

        let stamped = submitted(&messages);
        assert!(stamped.iter().all(|request| after(request) == vec![(A, 1)]));
        let entries = fake.entries(A);
        assert!(
            !entries[&(SLOT, KV, 0)].writable,
            "the commit marked it read-only"
        );
        assert!(
            !entries.contains_key(&(SLOT, KV, 1)),
            "the release removed it"
        );
        assert!(fake.violations().is_empty(), "{:?}", fake.violations());
    }

    #[test]
    fn a_copy_runs_with_the_next_batch_and_its_load_gives_read_only_entries() {
        let (fake, messages) = fake();
        let load = Request::Load {
            owner: OWNER,
            class: KV,
            at: 4,
            keys: vec![Key([1; 32]), Key([2; 32])],
        };
        let copy = Completion::Copy {
            group: A,
            keys: vec![Key([1; 32]), Key([2; 32])],
            from: vec![(3, 0, 1), (3, 1, 1)],
            to: vec![(0, 8, 1), (0, 9, 1)],
            evicted: Vec::new(),
            encodings: None,
        };
        mapped(&fake, 10, load, copy);
        assert!(
            fake.entries(A).is_empty(),
            "a load's entries wait for its answer"
        );

        fake.submit(batch(1, Phase::Decode, &[(2, 4, Vec::new(), 1)]));
        let finish = Request::Finish {
            copy: runtime::RequestId(10),
            ok: true,
            after: vec![(A, 1)],
        };
        assert_eq!(submitted(&messages), vec![finish.clone()]);
        output(&messages).1.unwrap();

        mapped(&fake, 11, finish, Completion::Loaded);
        let entries = fake.entries(A);
        assert_eq!(entries[&(SLOT, KV, 4)].runs, vec![(0, 8, 1)]);
        assert_eq!(entries[&(SLOT, KV, 5)].runs, vec![(0, 9, 1)]);
        assert!(entries.values().all(|entry| !entry.writable));
        assert!(fake.violations().is_empty(), "{:?}", fake.violations());
    }

    #[test]
    fn a_copy_given_up_before_a_batch_starts_it_never_runs() {
        let (fake, messages) = fake();
        fake.fail_copies(true);
        let clone = Request::Clone {
            owner: OWNER,
            class: KV,
            at: 4,
            key: Key([1; 32]),
        };
        let copy = Completion::Copy {
            group: A,
            keys: vec![Key([1; 32])],
            from: vec![(0, 2, 1)],
            to: vec![(0, 8, 1)],
            evicted: Vec::new(),
            encodings: None,
        };
        mapped(&fake, 10, clone, copy);

        let give_up = Request::Finish {
            copy: runtime::RequestId(10),
            ok: false,
            after: Vec::new(),
        };
        fake.stamp(A, vec![give_up]);
        let stamped = submitted(&messages);
        assert_eq!(after(&stamped[0]), vec![(A, 0)]);

        fake.submit(batch(1, Phase::Decode, &[(2, 4, Vec::new(), 1)]));
        output(&messages).1.unwrap();
        assert!(
            messages.try_recv().is_err(),
            "the copy returns no `Finish` of its own"
        );
    }

    #[test]
    fn a_row_writing_a_block_its_slot_does_not_hold_writable_is_a_violation() {
        let (fake, _messages) = fake();
        attached(&fake);

        // Positions 0 to 7 lie in the two allocated blocks.
        fake.submit(batch(1, Phase::Prefill, &[(1, SLOT, (0..8).collect(), 8)]));
        assert!(fake.violations().is_empty(), "{:?}", fake.violations());

        // Position 8 lies in the pinned block, position 12 in none.
        fake.submit(batch(2, Phase::Decode, &[(1, SLOT, vec![8], 9)]));
        fake.submit(batch(3, Phase::Decode, &[(1, SLOT, vec![12], 13)]));
        let violations = fake.violations();
        assert_eq!(violations.len(), 2, "{violations:?}");
        assert!(violations[0].contains("block 2"));
        assert!(violations[1].contains("block 3"));
    }

    #[test]
    fn a_stamped_request_is_submitted_once_it_left_every_table_with_every_stamp() {
        let (fake, messages) = fake();
        attached(&fake);

        // Submitted without passing through the group, which still holds the entry.
        fake.taken(runtime::RequestId(20), &release(2));
        assert_eq!(fake.violations().len(), 1);

        // Stamped by the group, then submitted without the stamp.
        fake.stamp(A, vec![release(2)]);
        submitted(&messages);
        fake.taken(runtime::RequestId(21), &release(2));
        assert_eq!(fake.violations().len(), 2);
        assert!(fake.violations()[1].contains("without the stamp of group 0"));

        // Stamped and submitted with the stamp.
        fake.stamp(A, vec![release(0)]);
        let stamped = submitted(&messages);
        fake.taken(runtime::RequestId(22), &stamped[0]);
        assert_eq!(fake.violations().len(), 2, "{:?}", fake.violations());
    }

    #[test]
    fn a_completion_is_mapped_only_as_the_pool_answered_it_and_never_stale() {
        let (fake, messages) = fake();
        attached(&fake);
        let allocate = |at| Request::Allocate {
            owner: OWNER,
            class: KV,
            at,
            count: 1,
            units: 0,
        };
        let allocated = |unit| Completion::Allocated {
            runs: vec![(0, unit, 1)],
            evicted: Vec::new(),
        };

        // A completion the buffer pool did not write.
        fake.map(A, SLOT, runtime::RequestId(30), allocate(9), allocated(12));
        assert_eq!(fake.violations().len(), 1);

        // An answer mapped after a release of its position was handed to the group.
        answer(&fake, 31, &allocate(7), &allocated(10));
        fake.stamp(A, vec![release(7)]);
        submitted(&messages);
        fake.map(A, SLOT, runtime::RequestId(31), allocate(7), allocated(10));
        assert_eq!(fake.violations().len(), 2);
        assert!(fake.violations()[1].contains("after a request that names position 7"));

        // A release taken before the request that allocates its position again.
        fake.stamp(A, vec![release(6)]);
        let stamped = submitted(&messages);
        fake.taken(runtime::RequestId(32), &stamped[0]);
        mapped(&fake, 33, allocate(6), allocated(11));
        assert_eq!(fake.violations().len(), 2, "{:?}", fake.violations());
    }

    #[test]
    fn text_ends_after_its_length_on_each_stream_and_a_case_output_replaces_the_rule() {
        let (fake, messages) = fake();
        let fake = fake.with_text_len(2);
        let mut tokens = Vec::new();
        for position in 10..13 {
            fake.submit(batch(
                position,
                Phase::Decode,
                &[(1, 1, Vec::new(), position)],
            ));
            tokens.extend(output(&messages).1.unwrap().calls[0].tokens.clone());
        }
        let drawn = |position| draw_token(DrawAddress { key: 1, position });
        assert_eq!(tokens, vec![drawn(10), drawn(11), EOS]);

        let empty = Arc::new(|_: &Batch| Ok(BatchOutput::default()));
        let fake = fake.with_output(empty);
        fake.submit(batch(13, Phase::Decode, &[(1, 1, Vec::new(), 13)]));
        assert_eq!(output(&messages).1.unwrap(), BatchOutput::default());
    }

    #[test]
    fn a_fixed_token_takes_the_draws_place_and_text_still_ends() {
        let (fake, messages) = fake();
        let fake = fake.with_token(97).with_text_len(2);
        let mut tokens = Vec::new();
        for position in 10..13 {
            fake.submit(batch(
                position,
                Phase::Decode,
                &[(1, 1, Vec::new(), position)],
            ));
            tokens.extend(output(&messages).1.unwrap().calls[0].tokens.clone());
        }
        assert_eq!(tokens, vec![97, 97, EOS]);
    }

    #[test]
    fn a_copy_fails_with_the_knob_and_with_the_batch_that_ran_it() {
        let (fake, messages) = fake();
        let load = |id: u64| {
            let request = Request::Load {
                owner: OWNER,
                class: KV,
                at: 4,
                keys: vec![Key([1; 32])],
            };
            let copy = Completion::Copy {
                group: A,
                keys: vec![Key([1; 32])],
                from: vec![(3, 0, 1)],
                to: vec![(0, 8, 1)],
                evicted: Vec::new(),
                encodings: None,
            };
            mapped(&fake, id, request, copy);
        };
        let failed = |id: u64, sequence| {
            vec![Request::Finish {
                copy: runtime::RequestId(id),
                ok: false,
                after: vec![(A, sequence)],
            }]
        };

        fake.fail_copies(true);
        load(10);
        fake.submit(batch(1, Phase::Decode, &[(2, 4, Vec::new(), 1)]));
        assert_eq!(submitted(&messages), failed(10, 1));
        output(&messages).1.unwrap();

        fake.fail_copies(false);
        load(11);
        fake.hold();
        fake.submit(batch(2, Phase::Decode, &[(2, 4, Vec::new(), 2)]));
        fake.fail_held("the device was lost");
        assert_eq!(submitted(&messages), failed(11, 2));
    }

    #[test]
    fn a_refill_writes_the_draft_class_and_a_kept_checkpoint_the_state_class() {
        const DRAFT: u8 = 1;
        const STATE: u8 = 2;
        let (fake, _messages) = fake();
        let fake = fake.with_draft(DRAFT, 4).with_checkpoints(STATE);
        let attach = Request::Attach {
            owner: OWNER,
            group: A,
        };
        mapped(&fake, 1, attach, Completion::Attached { slot: SLOT });
        let allocate = Request::Allocate {
            owner: OWNER,
            class: DRAFT,
            at: 0,
            count: 1,
            units: 0,
        };
        let allocated = Completion::Allocated {
            runs: vec![(0, 5, 1)],
            evicted: Vec::new(),
        };
        mapped(&fake, 2, allocate, allocated);

        // The refill writes the draft block it holds, and no class of the target.
        fake.submit(batch(1, Phase::Refill, &[(1, SLOT, (0..4).collect(), 4)]));
        assert!(fake.violations().is_empty(), "{:?}", fake.violations());

        // A checkpoint kept at a state position the slot does not hold.
        let mut decode = batch(2, Phase::Decode, &[(1, SLOT, Vec::new(), 4)]);
        decode.columns.state_copies.push(engine::batch::StateCopy {
            slot: SLOT,
            position: 0,
            direction: StateDirection::Keep,
        });
        fake.submit(decode);
        let violations = fake.violations();
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("keeps a checkpoint at position 0"));
    }
}
