//! Every request sequence of small configurations, enumerated: no sequence
//! breaks an invariant, and every state reached is written in the terms of
//! the design model so that it can be compared with the states of the
//! model's buffer pool (`verification/buffer-pool/agreement.sh`).
//!
//! The configurations are those of `MC_Pool` there, one per [`Shape`]: one
//! key and one position, one or two owners, a prefill group and for
//! transfers a decode group with a memory each. The bounds are the model's:
//! `completed` goes up to 1, a sequence makes as many copies as the shape
//! states, and one prefetch or put is in flight at a time.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::Write as _;
use std::rc::Rc;

use storage::protocol::Key;
use storage::runtime::{Completion, GroupId, OwnerId, Request, RequestId, Source, Stamps, Tier};

use super::{Block, BufferPool, Class, Copy, Deferred, Entry, Group, Held, Owner, Target};

const KV: u8 = 0;
const TRAJECTORY: u8 = 1;
const UNITS: u8 = 0;
const SESSION: u32 = 1;
const QUEUE: (u32, u16) = (SESSION, 0);

/// The model's bound `MaxBatch`.
const MAX_COMPLETED: u64 = 1;

/// A configuration of `MC_Pool`: what exists, and which requests arrive.
#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    owners: u64,
    groups: u8,
    /// The slot rows of each group.
    rows: u32,
    units: u32,
    host_units: u32,
    /// The model's bound `MaxCopies`: the copies a sequence makes.
    copies: u32,
    transfers: bool,
    /// Whether prefetches and puts to the remote store arrive.
    tiers: bool,
}

const SHAPES: [Shape; 4] = [
    Shape {
        name: "core",
        owners: 2,
        groups: 1,
        rows: 2,
        units: 2,
        host_units: 0,
        copies: 0,
        transfers: false,
        tiers: false,
    },
    Shape {
        name: "copies",
        owners: 2,
        groups: 1,
        rows: 2,
        units: 2,
        host_units: 1,
        copies: 2,
        transfers: false,
        tiers: false,
    },
    Shape {
        name: "transfer",
        owners: 2,
        groups: 2,
        rows: 2,
        units: 1,
        host_units: 0,
        copies: 0,
        transfers: true,
        tiers: false,
    },
    Shape {
        name: "tiers",
        owners: 1,
        groups: 1,
        rows: 1,
        units: 1,
        host_units: 1,
        copies: 2,
        transfers: false,
        tiers: true,
    },
];

/// One thing that happens to the buffer pool. A stamped request carries its
/// stamps as a mask: bit `g` set is the stamp 1 of group `g`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Attach(u64, u8),
    Allocate(u64),
    Pin(u64),
    Clone(u64),
    Load(u64),
    Save(u64),
    Restore(u64),
    Store(u8),
    Evict,
    Release(u64, u8),
    Detach(u64, u8),
    Commit(u64, u8),
    Transfer(u64, u8, u8),
    /// The `Finish` of the n-th copy an executor runs, with its result.
    Finish(usize, bool, u8),
    Advance(u8),
    Prefetch,
    Put,
    Deadline,
    /// The n-th task's next key is done, or the task has ended with a result.
    Done(usize),
    Ended(usize, bool),
}

fn key() -> Key {
    Key([1; 32])
}

fn stamps(mask: u8) -> Stamps {
    (0..2)
        .filter(|group| mask & (1 << group) != 0)
        .map(|group| (GroupId(group), 1))
        .collect()
}

struct Driver {
    shape: Shape,
    pool: BufferPool,
    number: u64,
    /// The copies made so far, as the model counts them.
    copies: u32,
}

impl Driver {
    /// The regions are the KV units, then each group's slot rows, then the
    /// host units; units and rows are numbered from 1, host units from 0.
    fn new(shape: Shape) -> Self {
        let host = shape.groups + 1;
        let classes = vec![
            Class {
                units: 1,
                host: Some((host, 1)),
                state_in_slot: false,
            },
            Class {
                units: 0,
                host: Some((host, 0)),
                state_in_slot: false,
            },
        ];
        let mut pool = BufferPool::new((1_000, 1_000));
        for space in 0..=host {
            pool.grant(space, 0, 1 << 16);
        }
        pool.open_region(Tier::Device, 0, 64, 1, shape.units, shape.units + 1);
        for group in 0..shape.groups {
            pool.open_region(Tier::Device, group + 1, 8, 1, shape.rows, shape.rows + 1);
        }
        pool.open_region(Tier::Host, host, 64, 0, shape.host_units, shape.host_units);
        let groups = (0..shape.groups).map(|group| {
            (
                GroupId(group),
                group,
                group + 1,
                vec![vec![UNITS], Vec::new()],
            )
        });
        pool.open(SESSION, 1, false, (classes, TRAJECTORY), groups.collect());

        // The key may be evicted for an allocation, as the model evicts a
        // shared block without a pin when its region has no free unit.
        let mut driver = Self {
            shape,
            pool,
            number: 0,
            copies: 0,
        };
        driver.submit(Request::Victims {
            region: UNITS,
            keys: vec![key()],
        });
        driver.submit(Request::Victims {
            region: host,
            keys: vec![key()],
        });
        driver
    }

    /// A second driver in the same state: the enumeration branches from a
    /// state without replaying the steps that led to it.
    fn duplicate(&self) -> Self {
        let pool = &self.pool;
        let mut blocks: HashMap<*const Block, Rc<Block>> = HashMap::new();
        let mut block = |block: &Rc<Block>| {
            let copy = || {
                let runs = block.runs.iter().map(super::Run::duplicate).collect();
                Rc::new(Block {
                    class: block.class,
                    runs,
                    valid: Cell::new(block.valid.get()),
                })
            };
            Rc::clone(blocks.entry(Rc::as_ptr(block)).or_insert_with(copy))
        };

        let mut copy = BufferPool::new(pool.limits);
        for (client, (classes, trajectory)) in &pool.classes {
            let classes = classes.iter().map(|class| Class {
                units: class.units,
                host: class.host,
                state_in_slot: class.state_in_slot,
            });
            copy.classes
                .insert(*client, (classes.collect(), *trajectory));
        }
        copy.regions = pool.regions.iter().map(super::Region::duplicate).collect();
        copy.quota = pool.quota.clone();
        for (key, entry) in &pool.cache {
            let device = entry
                .device
                .iter()
                .map(|(held, maker)| (block(held), *maker))
                .collect();
            let host = entry
                .host
                .as_ref()
                .map(|(held, maker)| (block(held), *maker));
            copy.cache.insert(
                *key,
                Entry {
                    class: entry.class,
                    device,
                    host,
                    tokens: entry.tokens,
                    remote: entry.remote,
                },
            );
        }
        for (id, owner) in &pool.owners {
            let mut table = |table: &BTreeMap<u32, Held>| {
                let mut entry = |held: &Held| match held {
                    Held::Private(held) => Held::Private(block(held)),
                    Held::Shared(held) => Held::Shared(block(held)),
                };
                table
                    .iter()
                    .map(|(position, held)| (*position, entry(held)))
                    .collect()
            };
            let tables = owner.tables.iter().map(&mut table).collect();
            copy.owners.insert(
                *id,
                Owner {
                    group: owner.group,
                    slot: block(&owner.slot),
                    tables,
                    moving: owner.moving,
                },
            );
        }
        for (id, group) in &pool.groups {
            let mut work = |work: &Deferred| match work {
                Deferred::Release { blocks } => Deferred::Release {
                    blocks: blocks.iter().map(&mut block).collect(),
                },
                Deferred::Commit {
                    queue,
                    request,
                    owner,
                    class,
                    at,
                    block: held,
                    key,
                    tokens,
                    rest,
                } => Deferred::Commit {
                    queue: *queue,
                    request: *request,
                    owner: *owner,
                    class: *class,
                    at: *at,
                    block: block(held),
                    key: *key,
                    tokens: *tokens,
                    rest: rest.clone(),
                },
                Deferred::Transfer {
                    queue,
                    request,
                    owner,
                    to,
                    rest,
                } => Deferred::Transfer {
                    queue: *queue,
                    request: *request,
                    owner: *owner,
                    to: *to,
                    rest: rest.clone(),
                },
                Deferred::Finish {
                    queue,
                    request,
                    copy,
                    ok,
                    rest,
                } => Deferred::Finish {
                    queue: *queue,
                    request: *request,
                    copy: *copy,
                    ok: *ok,
                    rest: rest.clone(),
                },
            };
            let deferred = group
                .deferred
                .iter()
                .map(|(stamp, deferred)| (*stamp, work(deferred)))
                .collect();
            let regions = group.regions.clone();
            copy.groups.insert(
                *id,
                Group {
                    memory: group.memory,
                    slots: group.slots,
                    regions,
                    completed: group.completed,
                    deferred,
                },
            );
        }
        for (id, copies) in &pool.copies {
            let mut one = |held: &Copy| Copy {
                memory: held.memory,
                reads: held.reads.iter().map(&mut block).collect(),
                write: block(&held.write),
                to: held.to,
                fill: held.fill,
                deadline: held.deadline,
            };
            copy.copies
                .insert(*id, copies.iter().map(&mut one).collect());
        }
        copy.answers = pool.answers.clone();
        copy.sessions = pool.sessions.clone();
        copy.trial = pool.trial.clone();
        copy.numbers = pool.numbers.clone();
        copy.now = pool.now;
        Self {
            shape: self.shape,
            pool: copy,
            number: self.number,
            copies: self.copies,
        }
    }

    fn submit(&mut self, request: Request) {
        self.number += 1;
        self.pool.take(QUEUE, RequestId(self.number), request);
        // Every request has exactly one completion, now or later.
        for (_, _, completion) in self.pool.completions() {
            if let Completion::Copy { .. } = completion {
                self.copies += 1;
            }
        }
    }

    /// The copies in flight by the request that started each: those an
    /// executor runs, or the tasks.
    fn in_flight(&self, tasks: bool) -> Vec<((u32, u16), RequestId)> {
        let of_kind = |copies: &Vec<Copy>| {
            copies
                .first()
                .is_some_and(|copy| copy.deadline.is_some() == tasks)
        };
        let mut found: Vec<_> = self
            .pool
            .copies
            .iter()
            .filter(|(_, copies)| of_kind(copies))
            .map(|(id, _)| *id)
            .collect();
        found.sort_unstable();
        found
    }

    /// Whether a prefetch or a put is in flight.
    fn tier_task(&self) -> bool {
        let tier = |copies: &Vec<Copy>| {
            matches!(
                copies.first(),
                Some(Copy {
                    to: Target::Entry { .. },
                    deadline: Some(_),
                    ..
                })
            )
        };
        self.pool.copies.values().any(tier)
    }

    /// The steps the model's environment can take in this state.
    fn steps(&self) -> Vec<Step> {
        let shape = self.shape;
        // The prefill group's executor stamps: the model's `Stamping`.
        let masks = 2;
        let mut steps = vec![Step::Evict];
        let room = self.copies < shape.copies;
        for owner in 1..=shape.owners {
            steps.extend((0..shape.groups).map(|group| Step::Attach(owner, group)));
            steps.extend([Step::Allocate(owner), Step::Pin(owner)]);
            if room {
                steps.extend([
                    Step::Clone(owner),
                    Step::Load(owner),
                    Step::Save(owner),
                    Step::Restore(owner),
                ]);
            }
            for mask in 0..masks {
                steps.extend([
                    Step::Release(owner, mask),
                    Step::Detach(owner, mask),
                    Step::Commit(owner, mask),
                ]);
                if shape.transfers {
                    steps.extend((0..shape.groups).map(|group| Step::Transfer(owner, group, mask)));
                }
            }
        }
        for group in 0..shape.groups {
            if room {
                steps.push(Step::Store(group));
            }
            if self.pool.groups[&(SESSION, GroupId(group))].completed < MAX_COMPLETED {
                steps.push(Step::Advance(group));
            }
        }
        for copy in 0..self.in_flight(false).len() {
            for mask in 0..masks {
                steps.extend([
                    Step::Finish(copy, true, mask),
                    Step::Finish(copy, false, mask),
                ]);
            }
        }
        if shape.tiers {
            steps.push(Step::Deadline);
            if !self.tier_task() {
                steps.extend([Step::Prefetch, Step::Put]);
            }
        }
        for task in 0..self.in_flight(true).len() {
            steps.extend([
                Step::Done(task),
                Step::Ended(task, true),
                Step::Ended(task, false),
            ]);
        }
        steps
    }

    fn apply(&mut self, step: Step) {
        let owner = OwnerId;
        match step {
            Step::Attach(o, group) => self.submit(Request::Attach {
                owner: owner(o),
                group: GroupId(group),
            }),
            Step::Allocate(o) => self.submit(Request::Allocate {
                owner: owner(o),
                class: KV,
                at: 0,
                count: 1,
                units: 0,
            }),
            Step::Pin(o) => self.submit(Request::Pin {
                owner: owner(o),
                class: KV,
                at: 0,
                keys: vec![key()],
            }),
            Step::Clone(o) => self.submit(Request::Clone {
                owner: owner(o),
                class: KV,
                at: 0,
                key: key(),
            }),
            Step::Load(o) => self.submit(Request::Load {
                owner: owner(o),
                class: KV,
                at: 0,
                keys: vec![key()],
            }),
            Step::Save(o) => self.submit(Request::Save {
                owner: owner(o),
                key: key(),
                units: 1,
            }),
            Step::Restore(o) => self.submit(Request::Restore {
                owner: owner(o),
                key: key(),
            }),
            Step::Store(group) => self.submit(Request::Store {
                group: GroupId(group),
                keys: vec![key()],
                tier: Tier::Host,
            }),
            Step::Evict => self.submit(Request::Evict { keys: vec![key()] }),
            Step::Release(o, mask) => {
                self.submit(Request::Release {
                    owner: owner(o),
                    class: KV,
                    at: 0,
                    count: 1,
                    after: stamps(mask),
                });
            }
            Step::Detach(o, mask) => self.submit(Request::Detach {
                owner: owner(o),
                after: stamps(mask),
            }),
            Step::Commit(o, mask) => {
                let commit = Request::Commit {
                    owner: owner(o),
                    class: KV,
                    at: 0,
                    key: key(),
                    tokens: 64,
                    after: stamps(mask),
                };
                self.submit(commit);
            }
            Step::Transfer(o, group, mask) => {
                self.submit(Request::Transfer {
                    owner: owner(o),
                    group: GroupId(group),
                    after: stamps(mask),
                });
            }
            Step::Finish(copy, ok, mask) => {
                let copy = self.in_flight(false)[copy].1;
                self.submit(Request::Finish {
                    copy,
                    ok,
                    after: stamps(mask),
                });
            }
            Step::Advance(group) => {
                let completed = self.pool.groups[&(SESSION, GroupId(group))].completed;
                self.pool.advance(SESSION, GroupId(group), completed + 1);
            }
            Step::Prefetch => {
                let prefetch = Request::Prefetch {
                    class: KV,
                    units: 0,
                    keys: vec![key()],
                    source: Source::Remote,
                    deadline: u64::MAX - 1,
                };
                self.submit(prefetch);
            }
            Step::Put => self.submit(Request::Store {
                group: GroupId(0),
                keys: vec![key()],
                tier: Tier::Remote,
            }),
            Step::Deadline => self.pool.tick(u64::MAX - 1),
            Step::Done(task) => self.pool.done(self.in_flight(true)[task], 64),
            Step::Ended(task, ok) => self.pool.ended(self.in_flight(true)[task], ok),
        }
        self.pool.completions();
        self.pool.started();
        self.pool.stopped();
    }

    /// The state in the terms of the design model, one line per fact.
    fn state(&self) -> String {
        let pool = &self.pool;
        let unit = |block: &Rc<Block>| {
            block
                .stated()
                .map(|run| format!("{}:{}", run.0, run.1))
                .collect::<Vec<_>>()
                .join("+")
        };
        let units = |blocks: BTreeSet<String>| blocks.into_iter().collect::<Vec<_>>().join(" ");
        let mut lines = vec![format!("ncopy {}", self.copies)];

        let completed: BTreeMap<u8, u64> = pool
            .groups
            .iter()
            .map(|((_, group), found)| (group.0, found.completed))
            .collect();
        lines.push(
            completed
                .iter()
                .fold("completed".to_owned(), |line, (group, at)| {
                    format!("{line} g{group}={at}")
                }),
        );

        let mut free = BTreeSet::new();
        for (id, region) in pool.regions.iter().enumerate() {
            for (first, count) in region.free_runs() {
                free.extend((first..first + count).map(|unit| format!("{id}:{unit}")));
            }
        }
        lines.push(format!("free {}", units(free)));

        // An owner's number is taken once: a number below the highest that
        // names no owner is of an owner that is gone.
        let taken = pool.numbers.get(&QUEUE).map_or(0, |numbers| numbers.0);
        for number in 1..=self.shape.owners {
            let line = match pool.owners.get(&(SESSION, OwnerId(number))) {
                Some(owner) => {
                    let mut line = format!(
                        "owner {number} live g{} slot={}",
                        owner.group.0,
                        unit(&owner.slot)
                    );
                    for (position, held) in &owner.tables[usize::from(KV)] {
                        let kind = if matches!(held, Held::Private(_)) {
                            "private"
                        } else {
                            "shared"
                        };
                        let _ = write!(line, " p{position}={kind}:{}", unit(held.block()));
                    }
                    line
                }
                None if number <= taken => format!("owner {number} gone"),
                None => format!("owner {number} new"),
            };
            lines.push(line);
        }

        let mut valid = BTreeMap::new();
        let mut note = |block: &Rc<Block>| {
            let memories: Vec<String> = (0..8)
                .filter(|bit| block.valid.get() & (1 << bit) != 0)
                .map(|bit| format!("m{bit}"))
                .collect();
            valid.insert(unit(block), memories.join(" "));
        };
        for owner in pool.owners.values() {
            owner.tables[usize::from(KV)]
                .values()
                .for_each(|held| note(held.block()));
        }
        for (key, entry) in &pool.cache {
            entry.device.iter().for_each(|(block, _)| note(block));
            let device = entry
                .device
                .first()
                .map_or("-".to_owned(), |(block, _)| unit(block));
            let host = entry
                .host
                .as_ref()
                .map_or("-".to_owned(), |(block, _)| unit(block));
            lines.push(format!(
                "key {:02x} d={device} h={host} remote={}",
                key.0[0],
                u8::from(entry.remote)
            ));
        }
        lines.extend(
            valid
                .into_iter()
                .map(|(unit, memories)| format!("valid {unit} {memories}").trim_end().to_owned()),
        );

        let copy_line = |copy: &Copy| {
            let (kind, owner, position, memory) = match copy.to {
                Target::Owner {
                    owner,
                    at,
                    key: None,
                    ..
                } => ("clone", owner.0.to_string(), at.to_string(), copy.memory),
                Target::Owner {
                    owner,
                    at,
                    key: Some(_),
                    ..
                } => ("load", owner.0.to_string(), at.to_string(), copy.memory),
                Target::Slot { owner } => ("restore", owner.0.to_string(), "-".to_owned(), 0),
                Target::Entry { .. }
                    if copy
                        .reads
                        .first()
                        .is_some_and(|block| block.class == super::SLOT) =>
                {
                    let owner = pool
                        .owners
                        .iter()
                        .find(|(_, owner)| Rc::ptr_eq(&owner.slot, &copy.reads[0]));
                    (
                        "save",
                        owner.map_or("-".to_owned(), |((_, owner), _)| owner.0.to_string()),
                        "-".to_owned(),
                        0,
                    )
                }
                _ => ("store", "-".to_owned(), "-".to_owned(), 0),
            };
            let memory = if memory == 0 {
                "-".to_owned()
            } else {
                format!("m{}", memory.trailing_zeros())
            };
            let fill = u8::from(copy.fill);
            format!(
                "kind={kind} o={owner} p={position} read={} write={} fill={fill} mem={memory}",
                unit(&copy.reads[0]),
                unit(&copy.write)
            )
        };

        let mut released = BTreeSet::new();
        for (_, work) in pool.groups.values().flat_map(|group| &group.deferred) {
            match work {
                Deferred::Release { blocks } => released.extend(blocks.iter().map(&unit)),
                Deferred::Commit {
                    owner, at, block, ..
                } => {
                    lines.push(format!("commit o={} p={at} u={}", owner.0, unit(block)));
                }
                Deferred::Finish {
                    queue, copy, ok, ..
                } => {
                    let copies = pool
                        .copies
                        .get(&(*queue, *copy))
                        .map_or(&[][..], Vec::as_slice);
                    lines.extend(
                        copies
                            .iter()
                            .map(|copy| format!("finish ok={} {}", u8::from(*ok), copy_line(copy))),
                    );
                }
                Deferred::Transfer { owner, to, .. } => {
                    lines.push(format!("transfer o={} to=g{}", owner.0, to.0))
                }
            }
        }
        if !released.is_empty() {
            lines.push(format!("release {}", units(released)));
        }

        for (task, copies) in &pool.copies {
            let expired = u8::from(!pool.answers.contains_key(task));
            match copies.first().map(|copy| &copy.to) {
                Some(Target::Group { owner, to }) => {
                    let held = copies
                        .iter()
                        .flat_map(|copy| &copy.reads)
                        .map(&unit)
                        .collect();
                    let writes = copies
                        .iter()
                        .filter(|copy| copy.fill)
                        .map(|copy| unit(&copy.write))
                        .collect();
                    let slot = copies
                        .iter()
                        .find(|copy| !copy.fill)
                        .map_or("-".to_owned(), |copy| unit(&copy.write));
                    let line = format!(
                        "task transfer o={} to=g{} slot={slot} units={} writes={}",
                        owner.0,
                        to.0,
                        units(held),
                        units(writes)
                    );
                    lines.push(line.trim_end().to_owned());
                }
                Some(Target::Entry { remote, .. }) if copies[0].deadline.is_some() => {
                    let kind = if *remote { "put" } else { "prefetch" };
                    lines.push(format!(
                        "task {kind} h={} expired={expired}",
                        unit(&copies[0].write)
                    ));
                }
                _ => lines.extend(
                    copies
                        .iter()
                        .map(|copy| format!("copy {}", copy_line(copy))),
                ),
            }
        }
        lines.sort_unstable();
        lines.join("\n")
    }
}

impl Driver {
    /// What tells two states of the enumeration apart: the state in the
    /// model's terms, and what the model does not state and a later step
    /// depends on, in lines that start with `#` and are not written out.
    /// The model has one kind of block and frees any row; the buffer pool
    /// keeps a key's class and the order in which rows were given back.
    fn identity(&self) -> String {
        let mut identity = self.state();
        for (key, entry) in &self.pool.cache {
            let _ = write!(identity, "\n# class {:02x} {}", key.0[0], entry.class);
        }
        for (id, region) in self.pool.regions.iter().enumerate() {
            let _ = write!(identity, "\n# rows {id} {:?}", region.returned());
        }
        identity
    }
}

/// L1: every sequence of steps keeps the invariants. With
/// `UNISERVE_POOL_STATES` set to a directory, the states each shape reaches
/// and the transitions between them are written there for the comparison
/// with the model.
#[test]
#[ignore = "enumerates every request sequence of four small configurations; run by verification/buffer-pool/agreement.sh"]
fn every_request_sequence_keeps_the_invariants() {
    for shape in SHAPES {
        let start = Driver::new(shape);
        let mut index = BTreeMap::from([(start.identity(), 0usize)]);
        let mut frontier = VecDeque::from([(0usize, start)]);
        let mut transitions = BTreeSet::new();

        while let Some((source, from)) = frontier.pop_front() {
            for step in from.steps() {
                let mut driver = from.duplicate();
                driver.apply(step);
                if let Err(inconsistency) = driver.pool.check() {
                    panic!(
                        "{}: {inconsistency} after {step:?} from\n{}",
                        shape.name,
                        from.state()
                    );
                }

                let known = index.len();
                let target = *index.entry(driver.identity()).or_insert(known);
                if target == known {
                    frontier.push_back((target, driver));
                }
                if target != source {
                    transitions.insert((source, target));
                }
            }
        }

        if let Ok(directory) = std::env::var("UNISERVE_POOL_STATES") {
            let mut states: Vec<(&String, &usize)> = index.iter().collect();
            states.sort_unstable_by_key(|(_, number)| **number);
            let mut text = String::new();
            for (state, number) in states {
                let stated: Vec<&str> = state
                    .lines()
                    .filter(|line| !line.starts_with('#'))
                    .collect();
                let _ = writeln!(text, "state {number}\n{}\n", stated.join("\n"));
            }
            for (source, target) in &transitions {
                let _ = writeln!(text, "step {source} {target}");
            }
            std::fs::write(format!("{directory}/{}.txt", shape.name), text).unwrap();
        }
        println!(
            "{}: {} states, {} transitions",
            shape.name,
            index.len(),
            transitions.len()
        );
        assert!(
            index.len() > 100,
            "{}: only {} states were reached",
            shape.name,
            index.len()
        );
    }
}
