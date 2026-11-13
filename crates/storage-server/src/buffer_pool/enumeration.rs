//! Every request sequence of small configurations, enumerated: no sequence
//! breaks an invariant, and every state reached is written in the terms of
//! the design model so that it can be compared with the states of the
//! model's buffer pool (`verification/buffer-pool/agreement.sh`), and every
//! transition with the input that made it, so that it can be compared with
//! the model's steps (`agreement_steps.sh`). The pool is driven by the
//! inputs the buffer pool's thread records ([`super::trace`]).
//!
//! The configurations are those of `MC_Pool` there, one per [`Shape`]: one
//! key and one position, one or two owners, a prefill group and for
//! transfers, crossings and host parts a decode group, with a memory each
//! and the host parts the shape states. The bounds are the model's:
//! `completed` goes up to 1, a sequence makes as many copies as the shape
//! states, and one prefetch or put is in flight at a time.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;

use storage::protocol::Key;
use storage::runtime::{GroupId, OwnerId, Request, RequestId, Source, Stamps, Tier};

use super::trace::{Input, apply, classes, copies_asked, layout, stated, stated_input};
use super::{BufferPool, Class, Copy, Group, Target};

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
    /// Whether the decode group reads, in its own memory, the class an
    /// owner of the prefill group holds: crossings into it arrive (the
    /// model's `Readers`).
    crossings: bool,
    /// The host parts each group copies with, as bits of a block's validity
    /// after the groups' memories (the model's `GroupParts`).
    parts: [u64; 2],
}

const SHAPES: [Shape; 6] = [
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
        parts: [1 << 1, 0],
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
        parts: [1 << 1, 0],
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
        parts: [1 << 2, 1 << 2],
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
        parts: [1 << 1, 0],
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
        parts: [1 << 2, 1 << 2],
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
        parts: [1 << 2, 1 << 2 | 1 << 3],
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
                tensors: [0; 32],
                encoding: [0; 32],
            },
            Class {
                units: 0,
                host: Some((host, 0)),
                state_in_slot: false,
                tensors: [1; 32],
                encoding: [0; 32],
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
        // Each group copies both classes with the host parts the shape states.
        let groups = (0..shape.groups).map(|group| {
            let parts = shape.parts[usize::from(group)];
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
            driver.act(&victims);
        }
        driver
    }

    /// A second driver in the same state: the enumeration branches from a
    /// state without replaying the steps that led to it, by making the
    /// pool again from its image.
    fn duplicate(&self) -> Self {
        let image = rmp_serde::to_vec(&self.pool).unwrap();
        Self {
            shape: self.shape,
            pool: rmp_serde::from_slice(&image).unwrap(),
            number: self.number,
            copies: self.copies,
        }
    }

    /// A request on the queue pair, under the next number, as the buffer
    /// pool's thread takes it.
    fn request(&mut self, request: Request) -> Input {
        self.number += 1;
        let mut record = Vec::new();
        request.encode(RequestId(self.number), &mut record);
        Input::Request(QUEUE, record)
    }

    /// Acts on an input as the buffer pool's thread does, and counts the
    /// copies its completions ask for.
    fn act(&mut self, input: &Input) {
        apply(&mut self.pool, input).unwrap();
        for (_, _, completion) in self.pool.completions() {
            self.copies += copies_asked(&completion);
        }
        self.pool.started();
        self.pool.stopped();
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

    /// Takes a step: the input it is in this state, acted on as the buffer
    /// pool's thread acts on it. Returns the input.
    fn apply(&mut self, step: Step) -> Input {
        let owner = OwnerId;
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
            Step::Advance(group) => {
                let completed = self.pool.groups[&(SESSION, GroupId(group))].completed;
                return self.acted(Input::Completed(SESSION, GroupId(group), completed + 1));
            }
            Step::Deadline => return self.acted(Input::Tick(u64::MAX - 1)),
            Step::Done(task) => return self.acted(Input::Done(self.in_flight(true)[task], 64)),
            Step::Ended(task, ok) => {
                return self.acted(Input::Ended(self.in_flight(true)[task], ok));
            }
        };
        let input = self.request(request);
        self.acted(input)
    }

    fn acted(&mut self, input: Input) -> Input {
        self.act(&input);
        input
    }

    /// The state in the terms of the design model, one line per fact.
    fn state(&self) -> String {
        let owners: Vec<_> = (1..=self.shape.owners)
            .map(|number| (QUEUE, OwnerId(number)))
            .collect();
        stated(&self.pool, &owners, self.copies)
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
        let mut identity = format!("{}\n{}", self.state(), classes(&self.pool));
        for (id, region) in self.pool.regions.iter().enumerate() {
            let _ = write!(identity, "\n# rows {id} {:?}", region.returned());
        }
        identity
    }
}

/// A transition the model does not allow, made by hand: a commit takes
/// effect while the stamp it waits for is not reached (R3). The step check
/// of `verification/buffer-pool/agreement_steps.sh` is to report it.
fn forged() -> String {
    let mut driver = Driver::new(SHAPES[0]);
    driver.apply(Step::Attach(1, 0));
    driver.apply(Step::Allocate(1));
    let before = driver.state();

    // The commit waits for the stamp 1 of the group; it is made due with
    // `completed` at 1 and left with `completed` at 0.
    let commit = driver.apply(Step::Commit(1, 1));
    let group = (SESSION, GroupId(0));
    driver.pool.groups.get_mut(&group).unwrap().completed = MAX_COMPLETED;
    driver.pool.settle();
    driver.pool.groups.get_mut(&group).unwrap().completed = 0;
    driver.pool.settle_counts();
    driver.pool.completions();
    format!(
        "{}state 0\n{before}\n\nstate 1\n{}\n\nstep 0 1 {}\n",
        layout(&driver.pool),
        driver.state(),
        stated_input(&commit)
    )
}

/// L1: every sequence of steps keeps the invariants. With
/// `UNISERVE_POOL_STATES` set to a directory, each shape's states and the
/// transitions between them are written there for the comparison with the
/// model's states (`<shape>.txt`), and with the input of each transition
/// and the layout for the comparison with the model's steps
/// (`<shape>.steps`, and the transition of [`forged`] as `forged.steps`).
#[test]
#[ignore = "enumerates every request sequence of six small configurations; run by verification/buffer-pool/agreement.sh"]
fn every_request_sequence_keeps_the_invariants() {
    let directory = std::env::var("UNISERVE_POOL_STATES").ok();
    for shape in SHAPES {
        let start = Driver::new(shape);
        let layout = layout(&start.pool);
        let mut index = BTreeMap::from([(start.identity(), 0usize)]);
        let mut frontier = VecDeque::from([(0usize, start)]);
        let mut transitions = BTreeSet::new();
        let mut steps = BTreeSet::new();

        while let Some((source, from)) = frontier.pop_front() {
            for step in from.steps() {
                let mut driver = from.duplicate();
                let input = driver.apply(step);
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
                    steps.insert((source, target, stated_input(&input)));
                }
            }
        }

        if let Some(directory) = &directory {
            let mut states: Vec<(&String, &usize)> = index.iter().collect();
            states.sort_unstable_by_key(|(_, number)| **number);
            let (mut text, mut with_inputs) = (String::new(), layout.clone());
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
            std::fs::write(format!("{directory}/{}.txt", shape.name), text).unwrap();
            std::fs::write(format!("{directory}/{}.steps", shape.name), with_inputs).unwrap();
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
    if let Some(directory) = &directory {
        std::fs::write(format!("{directory}/forged.steps"), forged()).unwrap();
    }
}
