//! Every request sequence of small configurations, enumerated: no sequence
//! breaks an invariant, and every state reached is written in the terms of
//! the design model so that it can be compared with the states of the
//! model's buffer pool (`tools/verification/buffer-pool/scripts/agreement.sh`), and every
//! transition with the input that made it, so that it can be compared with
//! the model's steps (`agreement_steps.sh`), both through
//! `buffer-pool-verify enumerate`. The pool is driven by the inputs the
//! buffer pool's thread records ([`super::trace`]).
//!
//! The configurations are those of `MC_Pool` there, one per [`Shape`]: one
//! key and one position, one or two owners, a prefill group and for
//! transfers, crossings and host parts a decode group, with a memory each
//! and the host parts the shape states; or, for an update, two sessions of
//! two layouts, the second on trial, each with a group and a region of its
//! own over one device memory. The bounds are the model's: `completed` goes
//! up to 1, a sequence makes as many copies as the shape states, and one
//! prefetch or put is in flight at a time.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::path::Path;

use storage::protocol::Key;
use storage::runtime::{GroupId, OwnerId, Request, RequestId, Source, Stamps, Tier};

use super::replay::{SessionNames, apply, classes, copies_asked, layout, stated, stated_input};
use super::trace::Input;
use super::{BufferPool, Class, Copy, Group, Memories, Target};

const KV: u8 = 0;
const TRAJECTORY: u8 = 1;
const UNITS: u8 = 0;
const SESSION: u32 = 1;
const QUEUE: (u32, u16) = (SESSION, 0);
/// The second session of the shape `trial`: the target of an update, of
/// another layout, on trial, with its region over the first's memory.
const TARGET: u32 = 2;
const TARGET_QUEUE: (u32, u16) = (TARGET, 0);
/// The regions of the shape `trial`: the serving session's KV units and
/// slot rows, the target's KV units over the same memory space and with
/// units of the same size (unit `i` of either occupies the same bytes) and
/// its slot rows, and the host units.
const TARGET_UNITS: u8 = 2;

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
    /// Whether the decode group reads, in its own memory, the class an
    /// owner of the prefill group holds: crossings into it arrive (the
    /// model's `Readers`).
    crossings: bool,
    /// The host parts each group copies with, as bits of a block's validity
    /// after the groups' memories (the model's `GroupParts`).
    parts: [&'static [u8]; 2],
    /// Whether the shape is an update's: owner 1 of the serving session's
    /// group and owner 1 of a target session of another layout, on trial,
    /// whose update is committed, or whose engine is lost and whose update
    /// is rolled back, and memory that moves between the two layouts'
    /// regions by `Resize` (the model's `MC_PoolTrial`).
    trial: bool,
}

const SHAPES: [Shape; 7] = [
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
        crossings: false,
        parts: [&[1], &[]],
        trial: false,
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
        crossings: false,
        parts: [&[1], &[]],
        trial: false,
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
        crossings: false,
        parts: [&[2], &[2]],
        trial: false,
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
        crossings: false,
        parts: [&[1], &[]],
        trial: false,
    },
    // Two memories, the decode group's reading what owners of the prefill
    // group hold: crossings, with the transfers, commits and detaches they
    // must agree with. One owner and one unit, so that every step is
    // checked too.
    Shape {
        name: "cross",
        owners: 1,
        groups: 2,
        rows: 1,
        units: 1,
        host_units: 0,
        copies: 0,
        transfers: true,
        tiers: false,
        crossings: true,
        parts: [&[2], &[2]],
        trial: false,
    },
    // Two host parts: the prefill group's ranks on one host, the decode
    // group's on that host and another. One owner and one unit, so that
    // every step is checked too.
    Shape {
        name: "parts",
        owners: 1,
        groups: 2,
        rows: 1,
        units: 1,
        host_units: 1,
        copies: 2,
        transfers: false,
        tiers: false,
        crossings: false,
        parts: [&[2], &[2, 3]],
        trial: false,
    },
    // An update: the serving session's group and a target session's, of
    // another layout and on trial, each with an owner and a region of its
    // own over one device memory: one unit of memory, in the serving
    // session's region at first, that either region can hold. The target's
    // update is committed, or its engine is lost and its update rolled back;
    // the memory moves between the regions by `Resize`.
    Shape {
        name: "trial",
        owners: 2,
        groups: 1,
        rows: 1,
        units: 1,
        host_units: 0,
        copies: 0,
        transfers: false,
        tiers: false,
        crossings: false,
        parts: [&[2], &[2]],
        trial: true,
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
    /// A crossing of the owner's block into the decode group's memory.
    Replicate(u64, u8),
    /// The `Finish` of the n-th copy a worker runs, with its result.
    Finish(usize, bool, u8),
    /// The `completed` word of a session's group.
    Advance(u32, u8),
    /// A region resized to so many units.
    Resize(u8, u32),
    /// The target's update committed; its session ended (its engine lost);
    /// its update rolled back.
    Publish,
    Close,
    RollBack,
    Prefetch,
    Put,
    Deadline,
    /// The n-th task's next key is done, or the task has ended with a result.
    Done(usize),
    Ended(usize, bool),
}

/// Numbers the memories of a shape as admission numbers a layout's: the device memories of its
/// `memories` groups first, then its host parts, so that each is the number the shape states.
fn name_memories(pool: &mut BufferPool, shape: &Shape, memories: u8) {
    let mut reserved = Memories::NONE;
    for memory in 0..memories {
        assert_eq!(
            pool.number_memory((format!("g{memory}"), None), &mut reserved),
            memory
        );
    }
    let parts: BTreeSet<u8> = shape.parts.iter().copied().flatten().copied().collect();
    for part in parts {
        assert_eq!(
            pool.number_memory(("host".to_owned(), Some(u32::from(part))), &mut reserved),
            part
        );
    }
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
    fn new(shape: Shape) -> Result<Self, String> {
        if shape.trial {
            return Self::update(shape);
        }
        let host = shape.groups + 1;
        let classes = vec![
            Class {
                units: 1,
                host: Some((host, 1)),
                state_in_slot: false,
                tensors: [0; 32],
                encoding: [0; 32],
                source: None,
            },
            Class {
                units: 0,
                host: Some((host, 0)),
                state_in_slot: false,
                tensors: [1; 32],
                encoding: [0; 32],
                source: None,
            },
        ];
        let mut pool = BufferPool::new((1_000, 1_000));
        for space in 0..=host {
            pool.grant(space, 0, 1 << 16);
        }
        pool.open_region(
            "units",
            Tier::Device,
            0,
            (64, 64),
            1,
            shape.units,
            shape.units + 1,
        );
        for group in 0..shape.groups {
            let rows = format!("rows {group}");
            pool.open_region(
                &rows,
                Tier::Device,
                group + 1,
                (8, 8),
                1,
                shape.rows,
                shape.rows + 1,
            );
        }
        pool.open_region(
            "host",
            Tier::Host,
            host,
            (64, 64),
            0,
            shape.host_units,
            shape.host_units,
        );
        name_memories(&mut pool, &shape, shape.groups);
        // Each group copies both classes with the host parts the shape states.
        let groups = (0..shape.groups).map(|group| {
            let parts: Memories = shape.parts[usize::from(group)]
                .iter()
                .map(|part| Memories::one(*part))
                .collect();
            let found = Group {
                parts: vec![parts, parts],
                ..Group::new(group, group + 1, vec![vec![UNITS], Vec::new()])
            };
            (GroupId(group), found)
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
        for region in [UNITS, host] {
            let victims = driver.request(Request::Victims {
                region,
                keys: vec![key()],
            });
            driver.act(&victims)?;
        }
        Ok(driver)
    }

    /// The shape `trial`: the serving session (epoch 1) with its group in
    /// region 0 of the shape's units and slot rows in region 1, and the
    /// target (epoch 2, on trial) of another layout, its group addressing
    /// memory of its own, with region 2 over region 0's memory space and
    /// units of the same size, its one unit over region 0's last unit and
    /// empty at first, and slot rows in region 3; host region 4.
    fn update(shape: Shape) -> Result<Self, String> {
        let host = 4;
        let classes = || {
            vec![
                Class {
                    units: 1,
                    host: Some((host, 1)),
                    state_in_slot: false,
                    tensors: [0; 32],
                    encoding: [0; 32],
                    source: None,
                },
                Class {
                    units: 0,
                    host: Some((host, 0)),
                    state_in_slot: false,
                    tensors: [1; 32],
                    encoding: [0; 32],
                    source: None,
                },
            ]
        };
        let mut pool = BufferPool::new((1_000, 1_000));
        for space in 0..=3 {
            pool.grant(space, 0, 1 << 16);
        }
        pool.open_region(
            "units a",
            Tier::Device,
            0,
            (64, 64),
            1,
            shape.units,
            shape.units + 1,
        );
        pool.open_region(
            "rows a",
            Tier::Device,
            1,
            (8, 8),
            1,
            shape.rows,
            shape.rows + 1,
        );
        // The target's one unit, numbered as the serving region's last and
        // over its bytes.
        pool.open_region(
            "units b",
            Tier::Device,
            0,
            (64, 64),
            shape.units,
            0,
            shape.units + 1,
        );
        pool.open_region(
            "rows b",
            Tier::Device,
            2,
            (8, 8),
            1,
            shape.rows,
            shape.rows + 1,
        );
        pool.open_region("host", Tier::Host, 3, (64, 64), 0, 0, 0);
        name_memories(&mut pool, &shape, 2);
        let group = |memory: u8, slots: u8, units: u8| {
            let parts: Memories = shape.parts[usize::from(memory)]
                .iter()
                .map(|part| Memories::one(*part))
                .collect();
            Group {
                parts: vec![parts, parts],
                ..Group::new(memory, slots, vec![vec![units], Vec::new()])
            }
        };
        pool.open(
            SESSION,
            1,
            false,
            (classes(), TRAJECTORY),
            vec![(GroupId(0), group(0, 1, UNITS))],
        );
        pool.open(
            TARGET,
            2,
            true,
            (classes(), TRAJECTORY),
            vec![(GroupId(0), group(1, 3, TARGET_UNITS))],
        );

        let mut driver = Self {
            shape,
            pool,
            number: 0,
            copies: 0,
        };
        for (queue, region) in [(QUEUE, UNITS), (TARGET_QUEUE, TARGET_UNITS)] {
            let victims = driver.request_on(
                queue,
                Request::Victims {
                    region,
                    keys: vec![key()],
                },
            );
            driver.act(&victims)?;
        }
        Ok(driver)
    }

    /// Whether the target's session is open.
    fn target_open(&self) -> bool {
        self.shape.trial && self.pool.sessions.contains_key(&TARGET)
    }

    /// The queue pair an owner of the shape's numbering takes requests on,
    /// and its number there: the target's owner is its session's owner 1.
    fn queue_of(&self, owner: u64) -> ((u32, u16), OwnerId) {
        if self.shape.trial && owner == 2 {
            (TARGET_QUEUE, OwnerId(1))
        } else {
            (QUEUE, OwnerId(owner))
        }
    }

    /// A second driver in the same state: the enumeration branches from a
    /// state without replaying the steps that led to it, by making the
    /// pool again from its image.
    fn duplicate(&self) -> Result<Self, String> {
        let image = rmp_serde::to_vec(&self.pool).map_err(|error| error.to_string())?;
        Ok(Self {
            shape: self.shape,
            pool: rmp_serde::from_slice(&image).map_err(|error| error.to_string())?,
            number: self.number,
            copies: self.copies,
        })
    }

    /// A request on the queue pair, under the next number, as the buffer
    /// pool's thread takes it.
    fn request(&mut self, request: Request) -> Input {
        self.request_on(QUEUE, request)
    }

    /// A request on `queue`, under the next number.
    fn request_on(&mut self, queue: (u32, u16), request: Request) -> Input {
        self.number += 1;
        let mut record = Vec::new();
        request.encode(RequestId(self.number), &mut record);
        Input::Request(queue, record)
    }

    /// Acts on an input as the buffer pool's thread does, and counts the
    /// copies its completions ask for.
    fn act(&mut self, input: &Input) -> Result<(), String> {
        apply(&mut self.pool, input)?;
        for (_, _, completion) in self.pool.completions() {
            self.copies += copies_asked(&completion);
        }
        self.pool.started();
        self.pool.stopped();
        Ok(())
    }

    /// The copies in flight by the request that started each: those an
    /// worker runs, or the tasks.
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
        // The prefill group's workers stamp: the model's `Stamping`.
        let masks = 2;
        let mut steps = vec![Step::Evict];
        let room = self.copies < shape.copies;
        for owner in 1..=shape.owners {
            if shape.trial && owner == 2 && !self.target_open() {
                continue;
            }
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
                if shape.crossings {
                    steps.push(Step::Replicate(owner, mask));
                }
            }
        }
        for group in 0..shape.groups {
            if room {
                steps.push(Step::Store(group));
            }
            for client in [SESSION, TARGET] {
                let found = self.pool.groups.get(&(client, GroupId(group)));
                if found.is_some_and(|found| found.completed < MAX_COMPLETED) {
                    steps.push(Step::Advance(client, group));
                }
            }
        }
        if shape.trial {
            // Memory moves by a unit at a time between the two regions.
            for region in [UNITS, TARGET_UNITS] {
                if region == TARGET_UNITS && !self.target_open() {
                    continue;
                }
                let now = self.pool.regions[usize::from(region)].units();
                if now > 0 {
                    steps.push(Step::Resize(region, now - 1));
                }
                let room = if region == TARGET_UNITS {
                    1
                } else {
                    shape.units
                };
                if now < room {
                    steps.push(Step::Resize(region, now + 1));
                }
            }
            match self.pool.trial.get(&2) {
                Some(true) if self.target_open() => steps.extend([Step::Publish, Step::Close]),
                Some(true) => steps.push(Step::RollBack),
                _ if self.target_open() => steps.push(Step::Close),
                _ => {}
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

    /// Takes a step: the input it is in this state, acted on as the buffer
    /// pool's thread acts on it. Returns the input.
    fn apply(&mut self, step: Step) -> Result<Input, String> {
        // An owner's request goes on its session's queue pair, under its
        // number there.
        let named = match step {
            Step::Attach(o, _)
            | Step::Allocate(o)
            | Step::Pin(o)
            | Step::Clone(o)
            | Step::Load(o)
            | Step::Save(o)
            | Step::Restore(o)
            | Step::Release(o, _)
            | Step::Detach(o, _)
            | Step::Commit(o, _)
            | Step::Transfer(o, _, _)
            | Step::Replicate(o, _) => Some(self.queue_of(o)),
            _ => None,
        };
        let queue = match step {
            Step::Resize(TARGET_UNITS, _) => TARGET_QUEUE,
            Step::Finish(copy, ..) => self.in_flight(false)[copy].0,
            _ => named.map_or(QUEUE, |(queue, _)| queue),
        };
        let owner = |_: u64| named.map_or(OwnerId(0), |(_, owner)| owner);
        let request = match step {
            Step::Attach(o, group) => Request::Attach {
                owner: owner(o),
                group: GroupId(group),
            },
            Step::Allocate(o) => Request::Allocate {
                owner: owner(o),
                class: KV,
                at: 0,
                count: 1,
                units: 0,
            },
            Step::Pin(o) => Request::Pin {
                owner: owner(o),
                class: KV,
                at: 0,
                keys: vec![key()],
            },
            Step::Clone(o) => Request::Clone {
                owner: owner(o),
                class: KV,
                at: 0,
                key: key(),
            },
            Step::Load(o) => Request::Load {
                owner: owner(o),
                class: KV,
                at: 0,
                keys: vec![key()],
            },
            Step::Save(o) => Request::Save {
                owner: owner(o),
                key: key(),
                units: 1,
            },
            Step::Restore(o) => Request::Restore {
                owner: owner(o),
                key: key(),
            },
            Step::Store(group) => Request::Store {
                group: GroupId(group),
                keys: vec![key()],
                tier: Tier::Host,
            },
            Step::Evict => Request::Evict { keys: vec![key()] },
            Step::Release(o, mask) => Request::Release {
                owner: owner(o),
                class: KV,
                at: 0,
                count: 1,
                after: stamps(mask),
            },
            Step::Detach(o, mask) => Request::Detach {
                owner: owner(o),
                after: stamps(mask),
            },
            Step::Commit(o, mask) => Request::Commit {
                owner: owner(o),
                class: KV,
                at: 0,
                key: key(),
                tokens: 64,
                after: stamps(mask),
            },
            Step::Transfer(o, group, mask) => Request::Transfer {
                trace: 0,
                owner: owner(o),
                group: GroupId(group),
                after: stamps(mask),
            },
            Step::Replicate(o, mask) => Request::Replicate {
                owner: owner(o),
                class: KV,
                at: 0,
                from: UNITS,
                group: GroupId(1),
                after: stamps(mask),
            },
            Step::Finish(copy, ok, mask) => Request::Finish {
                copy: self.in_flight(false)[copy].1,
                ok,
                after: stamps(mask),
            },
            Step::Prefetch => Request::Prefetch {
                group: GroupId(0),
                class: KV,
                units: 0,
                keys: vec![key()],
                source: Source::Remote,
                deadline: u64::MAX - 1,
            },
            Step::Put => Request::Store {
                group: GroupId(0),
                keys: vec![key()],
                tier: Tier::Remote,
            },
            Step::Advance(client, group) => {
                let completed = self.pool.groups[&(client, GroupId(group))].completed;
                return self.acted(Input::Completed(client, GroupId(group), completed + 1));
            }
            Step::Resize(region, units) => Request::Resize { region, units },
            Step::Publish => return self.acted(Input::Publish(2)),
            Step::Close => return self.acted(Input::Close(TARGET)),
            Step::RollBack => return self.acted(Input::RollBack(2)),
            Step::Deadline => return self.acted(Input::Tick(u64::MAX - 1)),
            Step::Done(task) => return self.acted(Input::Done(self.in_flight(true)[task], 64)),
            Step::Ended(task, ok) => {
                return self.acted(Input::Ended(self.in_flight(true)[task], ok));
            }
        };
        let input = self.request_on(queue, request);
        self.acted(input)
    }

    fn acted(&mut self, input: Input) -> Result<Input, String> {
        self.act(&input)?;
        Ok(input)
    }

    /// The state in the terms of the design model, one line per fact.
    fn state(&self) -> String {
        let owners: Vec<_> = (1..=self.shape.owners)
            .map(|number| self.queue_of(number))
            .collect();
        stated(&self.pool, &SessionNames::default(), &owners, self.copies)
    }
}

impl Driver {
    /// What tells two states of the enumeration apart: the state in the
    /// model's terms, and what the model does not state and a later step
    /// depends on, in lines that start with `#`, which the states for the
    /// model's states leave out. The model has one kind of block and frees
    /// any row; the buffer pool keeps a key's class and the order in which
    /// rows were given back.
    fn identity(&self) -> String {
        let mut identity = format!(
            "{}\n{}",
            self.state(),
            classes(&self.pool, &SessionNames::default())
        );
        for (id, region) in self.pool.regions.iter().enumerate() {
            let _ = write!(identity, "\n# rows {id} {:?}", region.returned());
        }
        identity
    }
}

/// A transition the model does not allow, made by hand: a commit takes
/// effect while the stamp it waits for is not reached (R3). The step check
/// of `tools/verification/buffer-pool/scripts/agreement_steps.sh` is to report it.
fn forged() -> Result<String, String> {
    let mut driver = Driver::new(SHAPES[0])?;
    driver.apply(Step::Attach(1, 0))?;
    driver.apply(Step::Allocate(1))?;
    let before = driver.state();

    // The commit waits for the stamp 1 of the group; it is made due with
    // `completed` at 1 and left with `completed` at 0.
    let commit = driver.apply(Step::Commit(1, 1))?;
    let group = (SESSION, GroupId(0));
    let completed = |driver: &mut Driver, value| match driver.pool.groups.get_mut(&group) {
        Some(found) => {
            found.completed = value;
            Ok(())
        }
        None => Err("the forged transition's group is not open".to_owned()),
    };
    completed(&mut driver, MAX_COMPLETED)?;
    driver.pool.settle();
    completed(&mut driver, 0)?;
    driver.pool.settle_counts();
    driver.pool.completions();
    Ok(format!(
        "{}state 0\n{before}\n\nstate 1\n{}\n\nstep 0 1 {}\n",
        layout(&driver.pool, &SessionNames::default()),
        driver.state(),
        stated_input(&driver.pool, &SessionNames::default(), &commit)
    ))
}

/// What the enumeration of one configuration reached.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reached {
    /// The configuration, by the name of its `MC_Pool` configuration (`core` is `MC_PoolCore`).
    pub shape: &'static str,
    /// The distinct states reached.
    pub states: usize,
    /// The distinct transitions between two states.
    pub transitions: usize,
}

/// The names of the configurations, in the order [`enumerate`] takes them.
pub fn shapes() -> impl Iterator<Item = &'static str> {
    SHAPES.iter().map(|shape| shape.name)
}

/// L1 (`specs/buffer-pool.md` §19): every request sequence of the configurations `names` (every
/// configuration when empty) keeps the pool's invariants, and reaches the model's bounds: a
/// configuration whose sequences make copies reaches a state that made as many as its bound
/// (`MaxCopies`). With `states` a directory, each configuration's states and the transitions
/// between them are written there for the comparison with the model's states (`<shape>.txt`),
/// and with the input of each transition and the layout for the comparison with the model's
/// steps (`<shape>.steps`, and a transition the model does not allow as `forged.steps`).
///
/// # Errors
///
/// The first inconsistency, with the step and the state it arose from; a configuration that
/// does not reach its bound; a name of no configuration; or a file that cannot be written.
pub fn enumerate(names: &[String], states: Option<&Path>) -> Result<Vec<Reached>, String> {
    if let Some(unknown) = names
        .iter()
        .find(|name| !shapes().any(|shape| shape == *name))
    {
        return Err(format!("no configuration is named {unknown}"));
    }
    let chosen = SHAPES
        .iter()
        .filter(|shape| names.is_empty() || names.iter().any(|name| name == shape.name));
    let mut reached = Vec::new();
    for shape in chosen {
        reached.push(enumerate_shape(*shape, states)?);
    }
    if let Some(directory) = states {
        let path = directory.join("forged.steps");
        std::fs::write(&path, forged()?).map_err(|error| format!("{}: {error}", path.display()))?;
    }
    Ok(reached)
}

/// Every request sequence of one configuration ([`enumerate`]).
fn enumerate_shape(shape: Shape, directory: Option<&Path>) -> Result<Reached, String> {
    let start = Driver::new(shape)?;
    let layout = layout(&start.pool, &SessionNames::default());
    let mut index = BTreeMap::from([(start.identity(), 0usize)]);
    let mut frontier = VecDeque::from([(0usize, start)]);
    let mut transitions = BTreeSet::new();
    let mut steps = BTreeSet::new();
    let mut most_copies = 0;

    while let Some((source, from)) = frontier.pop_front() {
        for step in from.steps() {
            let mut driver = from.duplicate()?;
            let input = driver.apply(step)?;
            if let Err(inconsistency) = driver.pool.check() {
                return Err(format!(
                    "{}: {inconsistency} after {step:?} from\n{}",
                    shape.name,
                    from.state()
                ));
            }
            most_copies = most_copies.max(driver.copies);

            let known = index.len();
            let target = *index.entry(driver.identity()).or_insert(known);
            if target == known {
                frontier.push_back((target, driver));
            }
            if target != source {
                transitions.insert((source, target));
                steps.insert((
                    source,
                    target,
                    stated_input(&from.pool, &SessionNames::default(), &input),
                ));
            }
        }
    }
    if most_copies < shape.copies {
        return Err(format!(
            "{}: the sequences made at most {most_copies} copies, short of the bound {}",
            shape.name, shape.copies
        ));
    }

    if let Some(directory) = directory {
        let mut states: Vec<(&String, &usize)> = index.iter().collect();
        states.sort_unstable_by_key(|(_, number)| **number);
        let (mut text, mut with_inputs) = (String::new(), layout);
        for (state, number) in states {
            let stated: Vec<&str> = state
                .lines()
                .filter(|line| !line.starts_with('#'))
                .collect();
            let _ = writeln!(text, "state {number}\n{}\n", stated.join("\n"));
            let lines: Vec<&str> = state.lines().filter(|line| !line.is_empty()).collect();
            let _ = writeln!(with_inputs, "state {number}\n{}\n", lines.join("\n"));
        }
        for (source, target) in &transitions {
            let _ = writeln!(text, "step {source} {target}");
        }
        for (source, target, input) in &steps {
            let _ = writeln!(with_inputs, "step {source} {target} {input}");
        }
        for (extension, contents) in [("txt", text), ("steps", with_inputs)] {
            let path = directory.join(format!("{}.{extension}", shape.name));
            std::fs::write(&path, contents)
                .map_err(|error| format!("{}: {error}", path.display()))?;
        }
    }
    Ok(Reached {
        shape: shape.name,
        states: index.len(),
        transitions: transitions.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::enumerate;

    /// L1 on the smallest configuration, on every build of the tests: every request sequence of
    /// `core` keeps the invariants. `buffer-pool-verify enumerate` takes every configuration.
    #[test]
    fn every_request_sequence_of_the_core_configuration_keeps_the_invariants() {
        let reached = enumerate(&["core".to_owned()], None).unwrap();
        assert_eq!(reached.len(), 1);
        assert!(reached[0].transitions > 0, "{reached:?}");
    }
}
