//! The in-process buffer pool: the Storage Server's [`BufferPool`] behind real queue pairs,
//! driven on the test's thread (`specs/scheduler-design.md` §8; §10, L6).
//!
//! [`InProcessPool::open`] opens an engine epoch's session as the Storage Server does: the pool
//! admits the layout, and each scheduler gets a queue pair whose registers hold the `completed`
//! word of each of its groups, in the order the scheduler lists them, then the three capacity
//! words of every region its groups and the classes name, by the region's number. The engine's
//! end of each queue pair is returned for its scheduler. Each [`InProcessPool::turn`] is one turn
//! of the server's thread:
//!
//! 1. each group's `completed` word is written from its [`FakeExecutor`]'s ended batches, as the
//!    workers write it, and read back into the pool;
//! 2. the requests of each submission queue are taken in order;
//! 3. the tasks the pool started run to their end: a transfer copies the owner's blocks, a
//!    prefetch brings the leading keys the remote store holds ([`InProcessPool::store_remote`]), a put to
//!    the remote store is acknowledged key by key;
//! 4. the pool is told the host's monotonic clock ([`runtime::monotonic_nanos`]), which a
//!    task's deadline is stated in;
//! 5. the completions that are due are written, in order, as far as each completion queue has
//!    room, the rest staying in order for the next turn;
//! 6. each region's capacity words are written as the pool states them.
//!
//! The fake executor of each queue pair is told each request the pool takes and each completion
//! it writes, which is how it checks what the scheduler hands it ([`crate::executor`]).
//!
//! The knobs: completions delayed by a number of turns, reordered, withheld until released; the
//! pool stopped, so that it takes nothing, writes nothing and a submission queue fills; capacity
//! words set by the case; tasks that fail; requests refused for a cause the case names. The
//! records: every request taken and every completion written, in order.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::atomic::Ordering;

use storage::runtime::{self, Cause, Completion, GroupId, Open, Opened, Registers, Request};
use storage::{Key, QueuePair, ServerSettings};
use storage_server::buffer_pool::{self, BufferPool, ClientId, QueueId};

use crate::executor::FakeExecutor;

/// Which completions to withhold, by the request each answers.
type Withhold = Box<dyn Fn(&Request, &Completion) -> bool>;

/// Which requests to refuse, and the cause each is refused for.
type Refuse = Box<dyn Fn(&Request) -> Option<Cause>>;

/// A completion produced and not yet written: due at a turn, or withheld until released.
struct Delayed {
    queue: QueueId,
    id: runtime::RequestId,
    completion: Completion,
    due: u64,
    withheld: bool,
}

/// The buffer pool's end of one scheduler's queue pair.
struct Served {
    queue: QueueId,
    pair: QueuePair,
    executor: FakeExecutor,
    /// Per group of the scheduler, its id and the register of its `completed` word.
    completed: Vec<(GroupId, usize)>,
    /// Per region the scheduler uses, its number and the register of its first capacity word.
    capacity: Vec<(u8, usize)>,
    /// Completion records that found no room, in order.
    pending: VecDeque<Vec<u8>>,
}

/// A deployment's buffer pool, served on the test's thread.
pub struct InProcessPool {
    pool: BufferPool,
    ring_bytes: usize,
    queues: Vec<Served>,
    next_client: ClientId,
    turns: u64,

    delay: u64,
    reorder: bool,
    withhold: Option<Withhold>,
    refuse: Option<Refuse>,
    stopped: bool,
    capacity: HashMap<u8, (u32, u32, u32)>,
    fail_tasks: bool,
    /// The keys the remote store holds, with the tokens of each.
    remote: HashMap<Key, u16>,

    requests: HashMap<(QueueId, u64), Request>,
    taken: Vec<(QueueId, runtime::RequestId, Request)>,
    written: Vec<(QueueId, runtime::RequestId, Completion)>,
    delayed: Vec<Delayed>,
}

impl InProcessPool {
    /// A buffer pool with no session, whose queue pairs' rings each hold `ring_bytes` (a power of
    /// two of at least 64; the Storage Server's are 4 MiB). Its transfers and puts are bounded as
    /// a Storage Server of the default settings bounds them; their tasks end within the turn that
    /// starts them, long before.
    pub fn new(ring_bytes: usize) -> Self {
        Self {
            pool: BufferPool::new(buffer_pool::limits(&ServerSettings::default())),
            ring_bytes,
            queues: Vec::new(),
            next_client: 1,
            turns: 0,
            delay: 0,
            reorder: false,
            withhold: None,
            refuse: None,
            stopped: false,
            capacity: HashMap::new(),
            fail_tasks: false,
            remote: HashMap::new(),
            requests: HashMap::new(),
            taken: Vec::new(),
            written: Vec::new(),
            delayed: Vec::new(),
        }
    }

    /// Opens the session of an engine epoch's `layout`, with the fake executor of each of its
    /// schedulers, in the order of `layout.schedulers`. Returns the answer to `Open` and the
    /// engine's end of each scheduler's queue pair.
    ///
    /// # Errors
    ///
    /// The pool's refusal of the layout, an executor missing or extra, or a queue pair that
    /// cannot be made; nothing is opened then.
    pub fn open(
        &mut self,
        layout: &Open,
        executors: Vec<FakeExecutor>,
    ) -> Result<(Opened, Vec<QueuePair>), String> {
        if executors.len() != layout.schedulers.len() {
            return Err(format!(
                "{} executors for {} schedulers",
                executors.len(),
                layout.schedulers.len()
            ));
        }
        let client = self.next_client;
        let regions = self.pool.admit(client, layout)?;
        self.next_client += 1;

        let made = self.queue_pairs(client, layout, &regions, executors);
        let (served, ends, registers) = match made {
            Ok(made) => made,
            Err(error) => {
                self.pool.end(client);
                return Err(format!("a queue pair cannot be made: {error}"));
            }
        };
        self.queues.extend(served);
        // Each region's segment table and numbering, as the Storage Server answers them.
        let segments = regions
            .iter()
            .map(|(number, _)| self.pool.segments_of(*number))
            .collect();
        let limits = regions
            .iter()
            .map(|(number, _)| self.pool.limit_of(*number))
            .collect();
        Ok((
            Opened {
                regions,
                segments,
                limits,
                registers,
                session: client,
            },
            ends,
        ))
    }

    /// One queue pair per scheduler of `layout`, laid out as the Storage Server lays them out.
    fn queue_pairs(
        &self,
        client: ClientId,
        layout: &Open,
        regions: &[(u8, u32)],
        executors: Vec<FakeExecutor>,
    ) -> storage::Result<(Vec<Served>, Vec<QueuePair>, Vec<Registers>)> {
        let number = |index: u8| regions.get(usize::from(index)).map(|(number, _)| *number);
        let hosts: BTreeSet<u8> = layout
            .classes
            .iter()
            .filter_map(|class| class.host.and_then(|(region, _)| number(region)))
            .collect();

        let (mut served, mut ends, mut registers) = (Vec::new(), Vec::new(), Vec::new());
        for (index, (groups, executor)) in layout.schedulers.iter().zip(executors).enumerate() {
            // The regions the scheduler uses: its groups' and every class's host region.
            let mut used = hosts.clone();
            for group in layout
                .groups
                .iter()
                .filter(|group| groups.contains(&group.id))
            {
                let named = group.regions.iter().flatten().chain([&group.slots]);
                used.extend(named.filter_map(|region| number(*region)));
            }
            let completed: Vec<u32> = (0..groups.len() as u32).collect();
            let capacity: Vec<(u8, u32)> = used
                .iter()
                .enumerate()
                .map(|(order, region)| (*region, groups.len() as u32 + 3 * order as u32))
                .collect();

            let pair = QueuePair::create(self.ring_bytes, groups.len() + 3 * used.len())?;
            ends.push(QueuePair::open(pair.duplicate()?, true)?);
            served.push(Served {
                queue: (client, index as u16),
                pair,
                executor,
                completed: groups
                    .iter()
                    .zip(0..)
                    .map(|(group, at)| (*group, at))
                    .collect(),
                capacity: capacity
                    .iter()
                    .map(|(region, at)| (*region, *at as usize))
                    .collect(),
                pending: VecDeque::new(),
            });
            registers.push(Registers {
                completed,
                capacity,
            });
        }
        Ok((served, ends, registers))
    }

    /// Ends the session of `epoch` and drops its queue pairs, as the Storage Server does at its
    /// `Close` or when its engine is gone. Returns whether it was open.
    pub fn close(&mut self, epoch: u64) -> bool {
        let Some(client) = self.pool.session(epoch) else {
            return false;
        };
        self.pool.end(client);
        self.queues.retain(|served| served.queue.0 != client);
        self.delayed.retain(|delayed| delayed.queue.0 != client);
        true
    }

    /// One turn of the buffer pool's thread; nothing while the pool is stopped.
    ///
    /// # Errors
    ///
    /// A queue pair whose indices the engine broke.
    pub fn turn(&mut self) -> storage::Result<()> {
        if self.stopped {
            return Ok(());
        }
        self.turns += 1;

        // 1. The groups' `completed` words, as the fakes' ended batches state them.
        for served in &self.queues {
            for &(group, register) in &served.completed {
                let word = served.pair.register(register);
                word.store(served.executor.completed(group), Ordering::Release);
                self.pool
                    .advance(served.queue.0, group, word.load(Ordering::Acquire));
            }
        }

        // 2. The requests, in the order each submission queue holds them.
        let mut record = Vec::new();
        for index in 0..self.queues.len() {
            let queue = self.queues[index].queue;
            while self.queues[index].pair.next_request(&mut record)? {
                match Request::decode(&record) {
                    Ok((id, request)) => self.take(index, id, request),
                    // A record of no request is refused where its number can be read.
                    Err(_) => {
                        if let Some(id) = runtime::number(&record) {
                            let refused = Completion::Refused {
                                cause: Cause::Request,
                            };
                            self.produced(queue, id, refused);
                        }
                    }
                }
            }
        }

        // 3. The tasks, each to its end; a task's end may let a waiting transfer start.
        loop {
            let started = self.pool.started();
            if started.is_empty() {
                break;
            }
            for task in started {
                self.run(task);
            }
        }
        for task in self.pool.stopped() {
            self.pool.ended(task, false);
        }

        // 4. The clock, and 5. the completions that are due.
        self.pool.tick(runtime::monotonic_nanos());
        for (queue, id, completion) in self.pool.completions() {
            self.produced(queue, id, completion);
        }
        self.write();

        // 6. The capacity words, and the scheduler's doorbell when they changed, as the server's
        // thread rings it.
        for served in &self.queues {
            let mut changed = false;
            for &(region, register) in &served.capacity {
                let words = self
                    .capacity
                    .get(&region)
                    .copied()
                    .unwrap_or_else(|| self.pool.capacity(served.queue, region));
                let (available, retiring, evictable) = words;
                for (offset, word) in [available, retiring, evictable].into_iter().enumerate() {
                    let register = served.pair.register(register + offset);
                    changed |= register.swap(u64::from(word), Ordering::AcqRel) != u64::from(word);
                }
            }
            if changed {
                served.pair.notify_registers();
            }
        }
        Ok(())
    }

    /// Takes one request of the queue pair at `index`.
    fn take(&mut self, index: usize, id: runtime::RequestId, request: Request) {
        let served = &self.queues[index];
        served.executor.taken(id, &request);
        self.requests.insert((served.queue, id.0), request.clone());
        self.taken.push((served.queue, id, request.clone()));
        // A refused request changes nothing (`specs/buffer-pool.md` §9, R1).
        let queue = served.queue;
        if let Some(cause) = self.refuse.as_ref().and_then(|refuse| refuse(&request)) {
            return self.produced(queue, id, Completion::Refused { cause });
        }
        self.pool.take(queue, id, request);
    }

    /// Runs a task the pool started to its end: a crossing or an import copies everything, a
    /// prefetch the leading keys the remote store holds, a put every key; with failing tasks, none of
    /// it. An import's lend of this pool ends with it, as the server ends it.
    fn run(&mut self, task: (QueueId, runtime::RequestId)) {
        let ok = !self.fail_tasks;
        if let Some(work) = self.pool.import_work(task)
            && work.lender.endpoint.is_none()
        {
            self.pool.give_back(work.lender.session, work.lender.owner);
        }
        if ok {
            // A transfer's copies name no key; a prefetch's read nothing; a put's read its block.
            for (key, reads, ..) in self.pool.work(task) {
                let Some(key) = key else {
                    break;
                };
                let tokens = if reads.is_empty() {
                    match self.remote.get(&key) {
                        Some(tokens) => *tokens,
                        None => break,
                    }
                } else {
                    0
                };
                self.pool.done(task, tokens);
            }
        }
        self.pool.ended(task, ok);
    }

    /// A completion the pool produced: due after the turns of delay, or withheld.
    fn produced(&mut self, queue: QueueId, id: runtime::RequestId, completion: Completion) {
        let request = self.requests.get(&(queue, id.0));
        let withheld = self
            .withhold
            .as_ref()
            .zip(request)
            .is_some_and(|(withhold, request)| withhold(request, &completion));
        self.delayed.push(Delayed {
            queue,
            id,
            completion,
            due: self.turns + self.delay,
            withheld,
        });
    }

    /// Writes the completions that are due, in the order they were produced (reversed when
    /// reordering), as far as each completion queue has room.
    fn write(&mut self) {
        let turns = self.turns;
        let mut due: Vec<Delayed> = self
            .delayed
            .extract_if(.., |delayed| !delayed.withheld && delayed.due <= turns)
            .collect();
        if self.reorder {
            due.reverse();
        }

        for delayed in due {
            let Some(served) = self
                .queues
                .iter_mut()
                .find(|served| served.queue == delayed.queue)
            else {
                continue;
            };
            // The fake learns the answer before the scheduler can read it.
            served.executor.answered(delayed.id, &delayed.completion);
            let mut record = Vec::new();
            delayed.completion.encode(delayed.id, &mut record);
            served.pending.push_back(record);
            self.written
                .push((delayed.queue, delayed.id, delayed.completion));
        }

        for served in &mut self.queues {
            while let Some(record) = served.pending.front() {
                match served.pair.complete(record) {
                    Ok(true) => {
                        served.pending.pop_front();
                    }
                    Ok(false) | Err(_) => break,
                }
            }
        }
    }

    /// Writes each completion produced from now on `turns` turns after the one that produced it.
    pub fn delay(&mut self, turns: u64) {
        self.delay = turns;
    }

    /// Writes the completions that fall due in one turn in the reverse of the order they were
    /// produced in, or in that order again.
    pub fn reorder(&mut self, reorder: bool) {
        self.reorder = reorder;
    }

    /// Withholds every completion produced from now on whose request and completion `matches`,
    /// until [`Self::release`].
    pub fn withhold(&mut self, matches: impl Fn(&Request, &Completion) -> bool + 'static) {
        self.withhold = Some(Box::new(matches));
    }

    /// Withholds nothing more, and lets every withheld completion fall due at the next turn.
    pub fn release(&mut self) {
        self.withhold = None;
        for delayed in &mut self.delayed {
            delayed.withheld = false;
        }
    }

    /// How many completions are withheld.
    pub fn withheld(&self) -> usize {
        self.delayed
            .iter()
            .filter(|delayed| delayed.withheld)
            .count()
    }

    /// Refuses, from now on, every request for which `cause` names a cause: the pool answers it
    /// `Refused` with that cause and changes nothing, as it does a request that does not fit
    /// (`specs/buffer-pool.md` §9, R1). A `cause` that names none refuses nothing more.
    pub fn refuse(&mut self, cause: impl Fn(&Request) -> Option<Cause> + 'static) {
        self.refuse = Some(Box::new(cause));
    }

    /// Stops the pool: a turn takes no request, reads no `completed` word and writes nothing, so
    /// every submission queue fills; or serves again.
    pub fn stop(&mut self, stopped: bool) {
        self.stopped = stopped;
    }

    /// Writes `words` (`available`, `retiring`, `evictable`) for the region numbered `region` in
    /// every queue pair, or the pool's own again with `None`.
    pub fn set_capacity(&mut self, region: u8, words: Option<(u32, u32, u32)>) {
        match words {
            Some(words) => self.capacity.insert(region, words),
            None => self.capacity.remove(&region),
        };
    }

    /// Fails every later task: a transfer, a prefetch or a put copies nothing.
    pub fn fail_tasks(&mut self, fail: bool) {
        self.fail_tasks = fail;
    }

    /// The remote store holds `key`, whose block holds `tokens` tokens: a prefetch brings it.
    pub fn store_remote(&mut self, key: Key, tokens: u16) {
        self.remote.insert(key, tokens);
    }

    /// Every request taken, in the order taken.
    pub fn requests(&self) -> &[(QueueId, runtime::RequestId, Request)] {
        &self.taken
    }

    /// Every completion written into a completion queue, in the order written.
    pub fn completions(&self) -> &[(QueueId, runtime::RequestId, Completion)] {
        &self.written
    }

    /// The buffer pool itself: its capacity words, its listing, its consistency check.
    pub fn pool(&mut self) -> &mut BufferPool {
        &mut self.pool
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::{Receiver, unbounded};
    use engine::batch::{Batch, BatchId, Columns, Phase, Row, Rows, Shape, Variant};
    use engine::executor::Executor;
    use engine::scheduler::Message;
    use storage::runtime::{Class, Group, OwnerId, Region, Tier, VERSION};

    use super::*;

    const A: GroupId = GroupId(0);
    const B: GroupId = GroupId(1);
    /// The KV class: one unit a block, four tokens, host copies of one unit.
    const KV: u8 = 0;
    /// The regions, numbered as the pool numbers them: the KV units both groups take blocks from
    /// (each in its own memory), the slot rows of each group, the host pool.
    const UNITS: u8 = 0;

    fn key(byte: u8) -> Key {
        Key([byte; 32])
    }

    /// Eight KV units, four slot rows per group, four host units; one scheduler of both groups.
    fn layout() -> Open {
        let region = |name: &str, tier, unit_bytes, reserved, units| Region {
            name: name.to_owned(),
            tier,
            space: name.to_owned(),
            unit_bytes,
            part_bytes: unit_bytes,
            reserved,
            units,
            limit: reserved + units,
            segments: Vec::new(),
            memory: String::new(),
        };
        let group = |id, memory: &str, slots| Group {
            id,
            memory: memory.to_owned(),
            slots,
            regions: vec![vec![UNITS]],
            receives: Vec::new(),
            parts: Vec::new(),
        };
        Open {
            version: VERSION,
            epoch: 1,
            trial: false,
            regions: vec![
                region("units", Tier::Device, 64, 1, 8),
                region("rows a", Tier::Device, 8, 1, 4),
                region("rows b", Tier::Device, 8, 1, 4),
                region("host", Tier::Host, 64, 0, 4),
            ],
            classes: vec![Class {
                units: 1,
                host: Some((3, 1)),
                state_in_slot: false,
                tensors: [0; 32],
                encoding: [0; 32],
            }],
            trajectory: KV,
            groups: vec![group(A, "gpu 0", 1), group(B, "gpu 1", 2)],
            schedulers: vec![vec![A, B]],
        }
    }

    /// The pool, the fake executor of its one scheduler, and the engine's end of the queue pair,
    /// which the tests drive as the scheduler would.
    struct Bench {
        pool: InProcessPool,
        fake: FakeExecutor,
        engine: QueuePair,
        opened: Opened,
        messages: Receiver<Message>,
        number: u64,
    }

    fn bench(ring_bytes: usize) -> Bench {
        let (inbox, messages) = unbounded();
        let fake = FakeExecutor::new(&[A, B], vec![(KV, 4)], inbox);
        let mut pool = InProcessPool::new(ring_bytes);
        let (opened, mut ends) = pool.open(&layout(), vec![fake.clone()]).unwrap();
        Bench {
            pool,
            fake,
            engine: ends.remove(0),
            opened,
            messages,
            number: 0,
        }
    }

    impl Bench {
        /// Writes `request` under the next number; false when the submission queue is full.
        fn submit(&mut self, request: &Request) -> bool {
            let mut record = Vec::new();
            request.encode(runtime::RequestId(self.number + 1), &mut record);
            let written = self.engine.submit(&record).unwrap();
            self.number += u64::from(written);
            written
        }

        /// Every completion the completion queue holds, in order.
        fn read(&mut self) -> Vec<(u64, Completion)> {
            let (mut record, mut read) = (Vec::new(), Vec::new());
            while self.engine.next_completion(&mut record).unwrap() {
                let (id, completion) = Completion::decode(&record).unwrap();
                read.push((id.0, completion));
            }
            read
        }

        /// Submits `request`, turns once, and maps its one completion on `group` under `slot`.
        fn ask(&mut self, group: GroupId, slot: u32, request: Request) -> Completion {
            assert!(self.submit(&request));
            self.pool.turn().unwrap();
            let mut read = self.read();
            assert_eq!(read.len(), 1, "{read:?}");
            let (id, completion) = read.remove(0);
            if !matches!(completion, Completion::Done | Completion::Full { .. }) {
                let id = runtime::RequestId(id);
                self.fake.map(group, slot, id, request, completion.clone());
            }
            completion
        }

        /// The capacity words of region `region`.
        fn words(&self, region: u8) -> [u64; 3] {
            let registers = &self.opened.registers[0];
            let (_, first) = registers
                .capacity
                .iter()
                .find(|(of, _)| *of == region)
                .unwrap();
            let first = *first as usize;
            [0, 1, 2].map(|offset| self.engine.register(first + offset).load(Ordering::Acquire))
        }

        /// The requests the fake returned in its next message, which must be stamped requests.
        fn stamped(&self) -> Vec<Request> {
            match self.messages.try_recv() {
                Ok(Message::Submit(requests)) => requests,
                other => panic!("expected stamped requests, got {other:?}"),
            }
        }
    }

    fn attach(owner: u64, group: GroupId) -> Request {
        Request::Attach {
            owner: OwnerId(owner),
            group,
        }
    }

    fn allocate(owner: u64, count: u32) -> Request {
        Request::Allocate {
            owner: OwnerId(owner),
            class: KV,
            at: 0,
            count,
            units: 0,
        }
    }

    /// A prefill on group `A` of one call in `slot` writing `positions`.
    fn prefill(slot: u32, positions: Vec<i32>) -> Batch {
        let tokens = positions.len() as i32;
        Batch {
            id: BatchId(1),
            group: A,
            phase: Phase::Prefill,
            kernels: 0,
            policy: 0,
            revision: 0,
            copy_budget: proto::config::DEFAULT_COPY_BUDGET,
            shape: Shape::prefill(1, tokens as u32, Variant::PrefillOutput),
            requests: vec![proto::request::RequestId(1)],
            columns: Columns {
                rows: Rows {
                    calls: vec![Row {
                        slot,
                        kind: Phase::Prefill,
                        replay: 0..0,
                        rows: 0..1,
                    }],
                    positions: positions.clone(),
                    cache_positions: positions,
                    seq_lens: vec![tokens],
                    query_offsets: vec![0, tokens],
                    logits_rows: vec![0],
                },
                ..Columns::default()
            },
            media: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    #[test]
    fn a_request_is_answered_at_the_next_turn_and_the_words_state_the_pool() {
        let mut bench = bench(4096);
        assert_eq!(bench.words(UNITS), [0, 0, 0], "no turn wrote them yet");
        assert_eq!(
            bench.ask(A, 0, attach(1, A)),
            Completion::Attached { slot: 1 }
        );
        assert_eq!(bench.words(UNITS), [8, 0, 0]);

        let allocated = bench.ask(A, 1, allocate(1, 2));
        assert!(matches!(allocated, Completion::Allocated { ref runs, .. } if runs.len() == 2));
        assert_eq!(bench.words(UNITS), [6, 0, 0]);
        assert_eq!(bench.pool.requests().len(), 2);
        assert_eq!(bench.pool.completions().len(), 2);
        assert!(
            bench.fake.violations().is_empty(),
            "{:?}",
            bench.fake.violations()
        );
    }

    #[test]
    fn a_delayed_completion_is_written_after_its_turns() {
        let mut bench = bench(4096);
        bench.pool.delay(2);
        assert!(bench.submit(&attach(1, A)));
        bench.pool.turn().unwrap();
        bench.pool.turn().unwrap();
        assert!(bench.read().is_empty());
        bench.pool.turn().unwrap();
        assert_eq!(bench.read(), vec![(1, Completion::Attached { slot: 1 })]);
    }

    #[test]
    fn completions_due_in_one_turn_are_written_in_reverse_when_reordered() {
        let mut bench = bench(4096);
        bench.pool.reorder(true);
        assert!(bench.submit(&attach(1, A)));
        assert!(bench.submit(&attach(2, A)));
        bench.pool.turn().unwrap();
        let read = bench.read();
        let ids: Vec<u64> = read.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![2, 1]);
    }

    #[test]
    fn a_withheld_completion_waits_for_its_release_and_later_ones_pass_it() {
        let mut bench = bench(4096);
        bench.pool.withhold(|request, _| *request == attach(1, A));
        assert!(bench.submit(&attach(1, A)));
        assert!(bench.submit(&attach(2, A)));
        bench.pool.turn().unwrap();
        assert_eq!(bench.read(), vec![(2, Completion::Attached { slot: 2 })]);
        assert_eq!(bench.pool.withheld(), 1);

        bench.pool.release();
        bench.pool.turn().unwrap();
        assert_eq!(bench.read(), vec![(1, Completion::Attached { slot: 1 })]);
        assert_eq!(bench.pool.withheld(), 0);
    }

    #[test]
    fn a_stopped_pool_takes_nothing_and_its_submission_queue_fills() {
        // A ring of 128 bytes holds four attach records of 32 bytes with their length words.
        let mut bench = bench(128);
        bench.pool.stop(true);
        let written = (1..=8)
            .take_while(|owner| bench.submit(&attach(*owner, A)))
            .count();
        assert_eq!(written, 4);
        bench.pool.turn().unwrap();
        assert!(bench.read().is_empty());
        assert!(bench.pool.requests().is_empty());

        bench.pool.stop(false);
        bench.pool.turn().unwrap();
        let ids: Vec<u64> = bench.read().into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn capacity_words_a_case_sets_replace_the_pools_until_it_unsets_them() {
        let mut bench = bench(4096);
        bench.pool.set_capacity(UNITS, Some((1, 2, 3)));
        bench.pool.turn().unwrap();
        assert_eq!(bench.words(UNITS), [1, 2, 3]);
        bench.pool.set_capacity(UNITS, None);
        bench.pool.turn().unwrap();
        assert_eq!(bench.words(UNITS), [8, 0, 0]);
    }

    #[test]
    fn a_commit_takes_effect_once_the_groups_completed_word_passes_its_stamp() {
        let mut bench = bench(4096);
        bench.ask(A, 0, attach(1, A));
        bench.ask(A, 1, allocate(1, 1));

        // A batch writes the block and is still on the device when the commit is stamped.
        bench.fake.hold();
        bench.fake.submit(prefill(1, vec![0, 1, 2, 3]));
        let commit = Request::Commit {
            owner: OwnerId(1),
            class: KV,
            at: 0,
            key: key(1),
            tokens: 4,
            after: Vec::new(),
        };
        bench.fake.stamp(A, vec![commit]);
        let stamped = bench.stamped();
        assert!(matches!(&stamped[0], Request::Commit { after, .. } if *after == vec![(A, 1)]));
        assert!(bench.submit(&stamped[0]));
        bench.pool.turn().unwrap();
        assert!(bench.read().is_empty(), "the commit waits for the batch");

        bench.fake.release(1);
        bench.pool.turn().unwrap();
        let completed = bench.opened.registers[0].completed[0] as usize;
        assert_eq!(bench.engine.register(completed).load(Ordering::Acquire), 1);
        assert_eq!(bench.read(), vec![(3, Completion::Committed)]);
        assert!(
            bench.fake.violations().is_empty(),
            "{:?}",
            bench.fake.violations()
        );
    }

    #[test]
    fn an_import_runs_within_the_turn_ends_its_lend_and_a_failed_task_answers_failed() {
        let mut bench = bench(4096);
        bench.ask(A, 0, attach(1, A));
        bench.ask(A, 1, allocate(1, 2));
        let lend = Request::Lend {
            owner: OwnerId(1),
            lease: 60_000_000_000,
            after: Vec::new(),
        };
        bench.fake.stamp(A, vec![lend]);
        let stamped = bench.stamped().remove(0);
        assert!(
            bench.fake.entries(A).is_empty(),
            "the lender's workers removed the entries"
        );
        assert_eq!(bench.ask(A, 2, stamped), Completion::Done);

        // The import fills two blocks of its own from the lend, which ends with it.
        bench.ask(B, 0, attach(2, B));
        let session = bench.opened.session;
        let import = |owner| Request::Import {
            owner: OwnerId(owner),
            tables: vec![(0, 0, 2)],
            lender: runtime::Lender {
                endpoint: None,
                session,
                owner: OwnerId(1),
            },
            push: false,
            transport: scheduling::kv::Transport::Nixl,
            trace: 0,
        };
        let imported = bench.ask(B, 1, import(2));
        let Completion::Imported { runs, .. } = imported else {
            panic!("the import answered {imported:?}");
        };
        assert_eq!(runs.len(), 2);
        assert!(bench.fake.entries(B).values().all(|entry| entry.writable));
        assert!(
            bench.fake.violations().is_empty(),
            "{:?}",
            bench.fake.violations()
        );

        // A lend of another host's pool whose task fails takes nothing.
        bench.ask(B, 3, attach(3, B));
        bench.pool.fail_tasks(true);
        let remote = Request::Import {
            owner: OwnerId(3),
            tables: vec![(0, 0, 1)],
            lender: runtime::Lender {
                endpoint: Some("lender:1".into()),
                session: 1,
                owner: OwnerId(1),
            },
            push: true,
            transport: scheduling::kv::Transport::Nixl,
            trace: 0,
        };
        assert_eq!(bench.ask(B, 4, remote), Completion::Failed);
    }

    #[test]
    fn a_prefetch_brings_the_leading_keys_the_remote_store_holds() {
        let mut bench = bench(4096);
        bench.pool.store_remote(key(1), 4);
        bench.pool.store_remote(key(2), 4);
        let prefetch = Request::Prefetch {
            group: A,
            class: KV,
            units: 0,
            keys: vec![key(1), key(2), key(3)],
            deadline: u64::MAX,
        };
        assert!(bench.submit(&prefetch));
        bench.pool.turn().unwrap();
        assert_eq!(
            bench.read(),
            vec![(
                1,
                Completion::Prefetched {
                    found: 2,
                    evicted: Vec::new()
                }
            )]
        );
    }

    #[test]
    fn a_closed_epoch_is_served_no_more() {
        let mut bench = bench(4096);
        bench.ask(A, 0, attach(1, A));
        assert!(bench.pool.close(1));
        assert!(!bench.pool.close(1));
        assert_eq!(bench.pool.pool().session(1), None);

        assert!(bench.submit(&attach(2, A)));
        bench.pool.turn().unwrap();
        assert!(bench.read().is_empty());
    }
}
