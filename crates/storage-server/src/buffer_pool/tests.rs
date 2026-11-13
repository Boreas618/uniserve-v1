//! The buffer pool's behaviour as its queue pairs see it: each request's
//! completions, and what the regions hold afterwards. Every step ends with
//! [`BufferPool::check`].

use storage::protocol::Key;
use storage::runtime::{Cause, Completion, GroupId, OwnerId, Request, RequestId, Source, Tier};

use super::{BufferPool, Class};

/// The KV class: one unit per block, host copies of one unit.
const KV: u8 = 0;
/// The class of stored trajectories: sized.
const TRAJECTORY: u8 = 1;
/// A media product: sized, one run on each of two ranks.
const PRODUCT: u8 = 2;

/// The regions: the KV units, the slot rows of the two groups, the host KV
/// pool, the host latent pool, and the product buffers of two ranks.
const UNITS: u8 = 0;
const HOST: u8 = 3;
const LATENTS: u8 = 4;

const PREFILL: GroupId = GroupId(0);
const DECODE: GroupId = GroupId(1);

/// The serving engine's session and its queue pair.
const SERVING: u32 = 1;
const QUEUE: (u32, u16) = (SERVING, 0);

fn key(byte: u8) -> Key {
    Key([byte; 32])
}

/// The classes of a session: KV blocks, stored trajectories, media products.
fn classes() -> (Vec<Class>, u8) {
    let classes = vec![
        Class {
            units: 1,
            host: Some((HOST, 1)),
            state_in_slot: false,
        },
        Class {
            units: 0,
            host: Some((LATENTS, 0)),
            state_in_slot: false,
        },
        Class {
            units: 0,
            host: None,
            state_in_slot: false,
        },
    ];
    (classes, TRAJECTORY)
}

struct Fixture {
    pool: BufferPool,
    number: u64,
}

impl Fixture {
    /// A pool with four KV units (1 to 4), two slot rows per group, two host
    /// KV units and eight host latent units, and one session of two groups
    /// over two memories.
    fn new() -> Self {
        let mut pool = BufferPool::new((1_000, 1_000));
        for space in 0..7 {
            pool.grant(space, 0, 1 << 20);
        }
        assert_eq!(
            pool.open_region("units", Tier::Device, 0, 64, 1, 4, 5),
            UNITS
        );
        pool.open_region("prefill rows", Tier::Device, 1, 8, 1, 2, 3);
        pool.open_region("decode rows", Tier::Device, 2, 8, 1, 2, 3);
        assert_eq!(pool.open_region("host", Tier::Host, 3, 64, 0, 2, 2), HOST);
        assert_eq!(
            pool.open_region("latents", Tier::Host, 4, 16, 0, 8, 8),
            LATENTS
        );
        pool.open_region("products 0", Tier::Device, 5, 256, 0, 4, 4);
        pool.open_region("products 1", Tier::Device, 6, 256, 0, 4, 4);

        let regions = || vec![vec![UNITS], Vec::new(), vec![5, 6]];
        pool.open(
            SERVING,
            1,
            false,
            classes(),
            vec![(PREFILL, 0, 1, regions()), (DECODE, 1, 2, regions())],
        );
        let mut fixture = Self { pool, number: 0 };
        fixture.checked();
        fixture
    }

    fn checked(&mut self) {
        if let Err(inconsistency) = self.pool.check() {
            panic!("{inconsistency}");
        }
    }

    /// Submits a request on a queue pair and returns its number.
    fn submit(&mut self, queue: (u32, u16), request: Request) -> RequestId {
        self.number += 1;
        let id = RequestId(self.number);
        self.pool.take(queue, id, request);
        self.checked();
        id
    }

    /// Submits a request on the serving queue pair and returns the
    /// completion it has now, if any; other completions stay queued.
    fn ask(&mut self, request: Request) -> Option<Completion> {
        let id = self.submit(QUEUE, request);
        self.answer(id)
    }

    /// The completion of a request, if it was produced.
    fn answer(&mut self, id: RequestId) -> Option<Completion> {
        let mut found = None;
        for (queue, request, completion) in self.pool.completions() {
            if request == id {
                found = Some(completion);
            } else {
                self.pool.completions.push((queue, request, completion));
            }
        }
        found
    }

    fn attach(&mut self, owner: u64, group: GroupId) -> u32 {
        match self.ask(Request::Attach {
            owner: OwnerId(owner),
            group,
        }) {
            Some(Completion::Attached { slot }) => slot,
            other => panic!("attach answered {other:?}"),
        }
    }

    /// Allocates `count` KV blocks at `at` and returns their units.
    fn allocate(&mut self, owner: u64, at: u32, count: u32) -> Vec<u32> {
        let request = Request::Allocate {
            owner: OwnerId(owner),
            class: KV,
            at,
            count,
            units: 0,
        };
        match self.ask(request) {
            Some(Completion::Allocated { runs, .. }) => runs.into_iter().map(|run| run.1).collect(),
            other => panic!("allocate answered {other:?}"),
        }
    }

    fn commit(
        &mut self,
        owner: u64,
        at: u32,
        name: u8,
        after: Vec<(GroupId, u64)>,
    ) -> Option<Completion> {
        self.ask(Request::Commit {
            owner: OwnerId(owner),
            class: KV,
            at,
            key: key(name),
            tokens: 64,
            after,
        })
    }

    fn release(
        &mut self,
        owner: u64,
        at: u32,
        count: u32,
        after: Vec<(GroupId, u64)>,
    ) -> Option<Completion> {
        self.ask(Request::Release {
            owner: OwnerId(owner),
            class: KV,
            at,
            count,
            after,
        })
    }

    fn pin(&mut self, owner: u64, at: u32, names: &[u8]) -> Option<Completion> {
        let keys = names.iter().map(|name| key(*name)).collect();
        self.ask(Request::Pin {
            owner: OwnerId(owner),
            class: KV,
            at,
            keys,
        })
    }

    fn detach(&mut self, owner: u64, after: Vec<(GroupId, u64)>) -> Option<Completion> {
        self.ask(Request::Detach {
            owner: OwnerId(owner),
            after,
        })
    }

    /// An owner on the prefill group that committed one block under `name`
    /// and detached: the block is shared without a pin. Returns its unit.
    fn cached(&mut self, owner: u64, name: u8) -> u32 {
        self.attach(owner, PREFILL);
        let unit = self.allocate(owner, 0, 1)[0];
        assert_eq!(
            self.commit(owner, 0, name, Vec::new()),
            Some(Completion::Committed)
        );
        assert_eq!(self.detach(owner, Vec::new()), Some(Completion::Done));
        unit
    }

    /// Runs a copy the pool answered with and finishes it.
    fn finish(&mut self, copy: RequestId, ok: bool) -> Option<Completion> {
        self.ask(Request::Finish {
            copy,
            ok,
            after: Vec::new(),
        })
    }

    fn free(&self, region: u8) -> u32 {
        self.pool.regions[usize::from(region)].free_units()
    }

    fn listing(&mut self) -> Vec<(Key, u8, u16, u8, u8)> {
        match self.ask(Request::List) {
            Some(Completion::Listing { keys }) => keys,
            other => panic!("list answered {other:?}"),
        }
    }
}

#[test]
fn an_owner_gets_the_lowest_slot_row_and_then_the_one_given_up_last() {
    let mut fixture = Fixture::new();
    assert_eq!(fixture.attach(1, PREFILL), 1);
    assert_eq!(fixture.attach(2, PREFILL), 2);
    let full = fixture.ask(Request::Attach {
        owner: OwnerId(3),
        group: PREFILL,
    });
    assert_eq!(full, Some(Completion::Full { region: 1 }));

    assert_eq!(fixture.detach(1, Vec::new()), Some(Completion::Done));
    assert_eq!(fixture.attach(4, PREFILL), 1);
    assert_eq!(fixture.attach(5, DECODE), 1);
}

#[test]
fn a_request_that_names_what_does_not_fit_is_refused_and_changes_nothing() {
    let mut fixture = Fixture::new();
    fixture.attach(5, PREFILL);
    let refused = |cause| Some(Completion::Refused { cause });

    // R1: an owner number is taken once, and names an owner of the session.
    assert_eq!(
        fixture.ask(Request::Attach {
            owner: OwnerId(5),
            group: PREFILL
        }),
        refused(Cause::Owner)
    );
    assert_eq!(
        fixture.ask(Request::Attach {
            owner: OwnerId(4),
            group: PREFILL
        }),
        refused(Cause::Owner)
    );
    assert_eq!(
        fixture.ask(Request::Attach {
            owner: OwnerId(6),
            group: GroupId(9)
        }),
        refused(Cause::Group)
    );
    assert_eq!(fixture.detach(9, Vec::new()), refused(Cause::Owner));

    // A position is occupied from the request that fills it.
    fixture.allocate(5, 0, 2);
    let again = Request::Allocate {
        owner: OwnerId(5),
        class: KV,
        at: 1,
        count: 1,
        units: 0,
    };
    assert_eq!(fixture.ask(again), refused(Cause::Position));
    assert_eq!(
        fixture.release(5, 1, 2, Vec::new()),
        refused(Cause::Position)
    );
    assert_eq!(
        fixture.commit(5, 3, 1, Vec::new()),
        refused(Cause::Position)
    );

    // A class, a size, a group of a stamp and a copy that do not fit.
    let class = Request::Allocate {
        owner: OwnerId(5),
        class: 9,
        at: 5,
        count: 1,
        units: 0,
    };
    assert_eq!(fixture.ask(class), refused(Cause::Class));
    let sized = Request::Allocate {
        owner: OwnerId(5),
        class: KV,
        at: 5,
        count: 1,
        units: 3,
    };
    assert_eq!(fixture.ask(sized), refused(Cause::Size));
    assert_eq!(
        fixture.release(5, 0, 1, vec![(GroupId(9), 1)]),
        refused(Cause::Group)
    );
    assert_eq!(fixture.finish(RequestId(99), true), refused(Cause::Copy));

    // A request number is taken once.
    fixture.pool.take(QUEUE, RequestId(1), Request::List);
    assert_eq!(fixture.answer(RequestId(1)), refused(Cause::Request));
    assert_eq!(fixture.free(UNITS), 2);
}

#[test]
fn an_allocation_is_all_or_none() {
    let mut fixture = Fixture::new();
    fixture.attach(1, PREFILL);
    assert_eq!(fixture.allocate(1, 0, 3), [1, 2, 3]);

    let request = Request::Allocate {
        owner: OwnerId(1),
        class: KV,
        at: 3,
        count: 2,
        units: 0,
    };
    assert_eq!(
        fixture.ask(request),
        Some(Completion::Full { region: UNITS })
    );
    assert_eq!(fixture.free(UNITS), 1);
    assert_eq!(fixture.allocate(1, 3, 1), [4]);
}

#[test]
fn a_released_block_is_reclaimed_when_completed_reaches_every_stamp() {
    let mut fixture = Fixture::new();
    fixture.attach(1, PREFILL);
    fixture.allocate(1, 0, 2);

    // R3: both groups' executors stamped the release.
    let after = vec![(PREFILL, 3), (DECODE, 2)];
    assert_eq!(fixture.release(1, 0, 2, after), Some(Completion::Done));
    assert_eq!(fixture.free(UNITS), 2);
    assert_eq!(fixture.pool.capacity(QUEUE, UNITS), (2, 2, 0));

    fixture.pool.advance(SERVING, PREFILL, 3);
    fixture.checked();
    assert_eq!(fixture.free(UNITS), 2);
    fixture.pool.advance(SERVING, DECODE, 1);
    assert_eq!(fixture.free(UNITS), 2);
    fixture.pool.advance(SERVING, DECODE, 2);
    fixture.checked();
    assert_eq!(fixture.free(UNITS), 4);
    assert_eq!(fixture.pool.capacity(QUEUE, UNITS), (4, 0, 0));

    // The positions were free from the moment the release was taken.
    assert_eq!(fixture.allocate(1, 0, 1), [1]);
}

#[test]
fn a_commit_takes_effect_at_its_stamp_and_shares_the_block() {
    let mut fixture = Fixture::new();
    fixture.attach(1, PREFILL);
    fixture.attach(2, PREFILL);
    let unit = fixture.allocate(1, 0, 1)[0];

    // R3: until the stamp is reached the block is private and no key names it.
    let id = fixture.submit(
        QUEUE,
        Request::Commit {
            owner: OwnerId(1),
            class: KV,
            at: 0,
            key: key(7),
            tokens: 40,
            after: vec![(PREFILL, 5)],
        },
    );
    assert_eq!(fixture.answer(id), None);
    assert_eq!(
        fixture.pin(2, 0, &[7]),
        Some(Completion::Pinned {
            runs: Vec::new(),
            host: 0
        })
    );
    assert_eq!(
        fixture.commit(1, 0, 8, Vec::new()),
        Some(Completion::Refused {
            cause: Cause::Position
        })
    );

    fixture.pool.advance(SERVING, PREFILL, 5);
    fixture.checked();
    assert_eq!(fixture.answer(id), Some(Completion::Committed));
    assert_eq!(
        fixture.pin(2, 0, &[7]),
        Some(Completion::Pinned {
            runs: vec![(UNITS, unit, 1)],
            host: 0
        })
    );
    assert_eq!(fixture.listing(), [(key(7), KV, 40, 0b001, 0b01)]);

    // R5: a shared block is not committed again.
    assert_eq!(
        fixture.commit(1, 0, 9, Vec::new()),
        Some(Completion::Refused {
            cause: Cause::Position
        })
    );
}

#[test]
fn a_key_that_is_cached_keeps_its_block() {
    let mut fixture = Fixture::new();
    let first = fixture.cached(1, 7);
    fixture.attach(2, PREFILL);
    let second = fixture.allocate(2, 0, 1)[0];

    // R4: the second block stays private to its owner.
    assert_eq!(
        fixture.commit(2, 0, 7, Vec::new()),
        Some(Completion::Duplicate)
    );
    fixture.attach(3, PREFILL);
    assert_eq!(
        fixture.pin(3, 0, &[7]),
        Some(Completion::Pinned {
            runs: vec![(UNITS, first, 1)],
            host: 0
        })
    );
    assert_eq!(fixture.release(2, 0, 1, Vec::new()), Some(Completion::Done));
    assert_eq!(fixture.allocate(2, 0, 1), [second]);
}

#[test]
fn a_commit_outlives_the_release_of_its_block() {
    let mut fixture = Fixture::new();
    fixture.attach(1, PREFILL);
    let unit = fixture.allocate(1, 0, 1)[0];
    let id = fixture.submit(
        QUEUE,
        Request::Commit {
            owner: OwnerId(1),
            class: KV,
            at: 0,
            key: key(7),
            tokens: 64,
            after: vec![(PREFILL, 2)],
        },
    );

    // R6: the detach does not cancel the commit; the block becomes shared without a pin.
    assert_eq!(
        fixture.detach(1, vec![(PREFILL, 2)]),
        Some(Completion::Done)
    );
    fixture.pool.advance(SERVING, PREFILL, 2);
    fixture.checked();
    assert_eq!(fixture.answer(id), Some(Completion::Committed));
    assert_eq!(fixture.pool.capacity(QUEUE, UNITS), (3, 0, 1));

    fixture.attach(2, PREFILL);
    assert_eq!(
        fixture.pin(2, 0, &[7]),
        Some(Completion::Pinned {
            runs: vec![(UNITS, unit, 1)],
            host: 0
        })
    );
}

#[test]
fn an_allocation_evicts_unpinned_blocks_in_the_installed_order() {
    let mut fixture = Fixture::new();
    let units: Vec<u32> = (1..=4)
        .map(|name| fixture.cached(u64::from(name), name))
        .collect();
    fixture.attach(5, PREFILL);
    assert_eq!(
        fixture.pin(5, 0, &[2]),
        Some(Completion::Pinned {
            runs: vec![(UNITS, units[1], 1)],
            host: 0
        })
    );

    // Without an order nothing is evicted.
    let request = Request::Allocate {
        owner: OwnerId(5),
        class: KV,
        at: 1,
        count: 2,
        units: 0,
    };
    assert_eq!(
        fixture.ask(request.clone()),
        Some(Completion::Full { region: UNITS })
    );

    // R7: the pinned block is skipped; the order decides among the others.
    let order = vec![key(2), key(4), key(9), key(1), key(3)];
    assert_eq!(
        fixture.ask(Request::Victims {
            region: UNITS,
            keys: order
        }),
        Some(Completion::Done)
    );
    assert_eq!(fixture.pool.capacity(QUEUE, UNITS), (3, 0, 3));
    let Some(Completion::Allocated { runs, evicted }) = fixture.ask(request) else {
        panic!("not allocated")
    };
    assert_eq!(evicted, [key(4), key(1)]);
    assert_eq!(runs, [(UNITS, units[0], 1), (UNITS, units[3], 1)]);
    assert_eq!(
        fixture
            .listing()
            .iter()
            .map(|listed| listed.0)
            .collect::<Vec<_>>(),
        [key(2), key(3)]
    );
}

#[test]
fn evict_reclaims_only_shared_blocks_that_nothing_else_holds() {
    let mut fixture = Fixture::new();
    fixture.cached(1, 1);
    fixture.cached(2, 2);
    fixture.attach(3, PREFILL);
    fixture.pin(3, 0, &[2]);

    // R7: the pinned block stays; the completion lists what left.
    let keys = vec![key(1), key(2), key(9)];
    assert_eq!(
        fixture.ask(Request::Evict { keys: keys.clone() }),
        Some(Completion::Evicted { keys: vec![key(1)] })
    );
    assert_eq!(fixture.free(UNITS), 3);

    // R2: a release that waits is a holder too.
    assert_eq!(
        fixture.release(3, 0, 1, vec![(PREFILL, 4)]),
        Some(Completion::Done)
    );
    assert_eq!(
        fixture.ask(Request::Evict { keys: keys.clone() }),
        Some(Completion::Evicted { keys: Vec::new() })
    );
    fixture.pool.advance(SERVING, PREFILL, 4);
    assert_eq!(
        fixture.ask(Request::Evict { keys }),
        Some(Completion::Evicted { keys: vec![key(2)] })
    );
    assert_eq!(fixture.free(UNITS), 4);
}

#[test]
fn a_clone_is_a_new_private_block_filled_by_the_executor() {
    let mut fixture = Fixture::new();
    let cached = fixture.cached(1, 7);
    fixture.attach(2, PREFILL);

    let id = fixture.submit(
        QUEUE,
        Request::Clone {
            owner: OwnerId(2),
            class: KV,
            at: 0,
            key: key(7),
        },
    );
    let Some(Completion::Copy {
        group,
        keys,
        from,
        to,
        evicted,
    }) = fixture.answer(id)
    else {
        panic!("no copy")
    };
    assert_eq!(
        (group, keys, from, evicted),
        (PREFILL, vec![key(7)], vec![(UNITS, cached, 1)], Vec::new())
    );
    // R5: the destination is another block than the cached one.
    assert_ne!(to, [(UNITS, cached, 1)]);

    // The position is occupied while the copy writes for it, and the source is held.
    let request = Request::Allocate {
        owner: OwnerId(2),
        class: KV,
        at: 0,
        count: 1,
        units: 0,
    };
    assert_eq!(
        fixture.ask(request),
        Some(Completion::Refused {
            cause: Cause::Position
        })
    );
    assert_eq!(
        fixture.ask(Request::Evict { keys: vec![key(7)] }),
        Some(Completion::Evicted { keys: Vec::new() })
    );

    assert_eq!(fixture.finish(id, true), Some(Completion::Cloned));
    assert_eq!(
        fixture.finish(id, true),
        Some(Completion::Refused { cause: Cause::Copy })
    );
    assert_eq!(
        fixture.commit(2, 0, 8, Vec::new()),
        Some(Completion::Committed)
    );
    assert_eq!(
        fixture.ask(Request::Clone {
            owner: OwnerId(2),
            class: KV,
            at: 1,
            key: key(3)
        }),
        Some(Completion::Refused { cause: Cause::Key })
    );
}

#[test]
fn a_failed_copy_retires_its_destination_and_frees_its_position() {
    let mut fixture = Fixture::new();
    fixture.cached(1, 7);
    fixture.attach(2, PREFILL);
    let id = fixture.submit(
        QUEUE,
        Request::Clone {
            owner: OwnerId(2),
            class: KV,
            at: 0,
            key: key(7),
        },
    );
    assert_eq!(fixture.free(UNITS), 2);

    // The give-up is stamped like any finish: the destination is released
    // only when the copy has ended.
    let finish = fixture.submit(
        QUEUE,
        Request::Finish {
            copy: id,
            ok: false,
            after: vec![(PREFILL, 1)],
        },
    );
    assert_eq!(fixture.answer(finish), None);
    assert_eq!(fixture.free(UNITS), 2);
    fixture.pool.advance(SERVING, PREFILL, 1);
    fixture.checked();
    assert_eq!(fixture.answer(finish), Some(Completion::Done));
    assert_eq!(fixture.free(UNITS), 3);
    assert_eq!(fixture.allocate(2, 0, 1).len(), 1);
}

#[test]
fn a_store_and_a_load_move_a_key_between_the_device_and_the_host() {
    let mut fixture = Fixture::new();
    let unit = fixture.cached(1, 7);

    // Store: the group's executor copies the device block into a new host block.
    let store = fixture.submit(
        QUEUE,
        Request::Store {
            group: PREFILL,
            keys: vec![key(7), key(9)],
            tier: Tier::Host,
        },
    );
    let Some(Completion::Copy { keys, from, to, .. }) = fixture.answer(store) else {
        panic!("no copy")
    };
    assert_eq!(
        (keys, from, to),
        (vec![key(7)], vec![(UNITS, unit, 1)], vec![(HOST, 0, 1)])
    );
    assert_eq!(fixture.listing(), [(key(7), KV, 64, 0b001, 0b01)]);
    assert_eq!(
        fixture.finish(store, true),
        Some(Completion::Stored { keys: Vec::new() })
    );
    assert_eq!(fixture.listing(), [(key(7), KV, 64, 0b011, 0b01)]);

    // The device copy leaves; a pin now states the host copy.
    assert_eq!(
        fixture.ask(Request::Evict { keys: vec![key(7)] }),
        Some(Completion::Evicted { keys: vec![key(7)] })
    );
    fixture.attach(2, PREFILL);
    assert_eq!(
        fixture.pin(2, 0, &[7]),
        Some(Completion::Pinned {
            runs: Vec::new(),
            host: 1
        })
    );

    // Load: a new device block, shared and pinned for the owner when its finish takes effect.
    let load = fixture.submit(
        QUEUE,
        Request::Load {
            owner: OwnerId(2),
            class: KV,
            at: 0,
            keys: vec![key(7), key(9)],
        },
    );
    let Some(Completion::Copy { keys, from, to, .. }) = fixture.answer(load) else {
        panic!("no copy")
    };
    assert_eq!((keys, from), (vec![key(7)], vec![(HOST, 0, 1)]));
    assert_eq!(fixture.finish(load, true), Some(Completion::Loaded));
    assert_eq!(
        fixture.pin(2, 1, &[7]),
        Some(Completion::Pinned { runs: to, host: 0 })
    );
    assert_eq!(
        fixture.ask(Request::Evict { keys: vec![key(7)] }),
        Some(Completion::Evicted { keys: Vec::new() })
    );

    // Nothing to load: the request itself is answered.
    let none = Request::Load {
        owner: OwnerId(2),
        class: KV,
        at: 2,
        keys: vec![key(9)],
    };
    assert_eq!(fixture.ask(none), Some(Completion::Loaded));
}

#[test]
fn a_key_that_gained_a_copy_before_the_finish_keeps_it() {
    let mut fixture = Fixture::new();
    fixture.cached(1, 7);
    let first = fixture.submit(
        QUEUE,
        Request::Store {
            group: PREFILL,
            keys: vec![key(7)],
            tier: Tier::Host,
        },
    );
    let second = fixture.submit(
        QUEUE,
        Request::Store {
            group: PREFILL,
            keys: vec![key(7)],
            tier: Tier::Host,
        },
    );
    assert_eq!(fixture.free(HOST), 0);

    // R4: the second host block is released.
    assert_eq!(
        fixture.finish(first, true),
        Some(Completion::Stored { keys: Vec::new() })
    );
    assert_eq!(
        fixture.finish(second, true),
        Some(Completion::Stored { keys: Vec::new() })
    );
    assert_eq!(fixture.free(HOST), 1);

    // R4 for a load: two owners load the key at once; the second block stays private.
    assert_eq!(
        fixture.ask(Request::Evict { keys: vec![key(7)] }),
        Some(Completion::Evicted { keys: vec![key(7)] })
    );
    fixture.attach(2, PREFILL);
    fixture.attach(3, PREFILL);
    let load = |owner| Request::Load {
        owner: OwnerId(owner),
        class: KV,
        at: 0,
        keys: vec![key(7)],
    };
    let (one, two) = (
        fixture.submit(QUEUE, load(2)),
        fixture.submit(QUEUE, load(3)),
    );
    assert_eq!(fixture.finish(one, true), Some(Completion::Loaded));
    assert_eq!(fixture.finish(two, true), Some(Completion::Duplicate));
    assert_eq!(fixture.pool.capacity(QUEUE, UNITS), (2, 0, 0));
    assert_eq!(fixture.detach(3, Vec::new()), Some(Completion::Done));
    assert_eq!(fixture.free(UNITS), 3);
}

#[test]
fn a_trajectory_is_saved_from_the_slot_and_restored_into_it() {
    let mut fixture = Fixture::new();
    let slot = fixture.attach(1, PREFILL);
    let save = fixture.submit(
        QUEUE,
        Request::Save {
            owner: OwnerId(1),
            key: key(5),
            units: 3,
        },
    );
    let Some(Completion::Copy { from, to, .. }) = fixture.answer(save) else {
        panic!("no copy")
    };
    assert_eq!(
        (from, to.clone()),
        (vec![(1, slot, 1)], vec![(LATENTS, 0, 3)])
    );

    // R1: one save or restore of a slot at a time. The owner may leave meanwhile.
    assert_eq!(
        fixture.ask(Request::Restore {
            owner: OwnerId(1),
            key: key(5)
        }),
        Some(Completion::Refused { cause: Cause::Copy })
    );
    assert_eq!(fixture.detach(1, Vec::new()), Some(Completion::Done));
    assert_eq!(fixture.free(1), 1);
    assert_eq!(fixture.finish(save, true), Some(Completion::Saved));
    assert_eq!(fixture.free(1), 2);

    let slot = fixture.attach(2, DECODE);
    let restore = fixture.submit(
        QUEUE,
        Request::Restore {
            owner: OwnerId(2),
            key: key(5),
        },
    );
    let Some(Completion::Copy { from, to: into, .. }) = fixture.answer(restore) else {
        panic!("no copy")
    };
    assert_eq!((from, into), (to, vec![(2, slot, 1)]));
    assert_eq!(fixture.finish(restore, true), Some(Completion::Restored));
    assert_eq!(
        fixture.ask(Request::Restore {
            owner: OwnerId(2),
            key: key(6)
        }),
        Some(Completion::Refused { cause: Cause::Key })
    );
}

#[test]
fn a_product_is_one_run_on_every_backing_rank() {
    let mut fixture = Fixture::new();
    fixture.attach(1, PREFILL);
    let request = Request::Allocate {
        owner: OwnerId(1),
        class: PRODUCT,
        at: 0,
        count: 1,
        units: 3,
    };
    let allocated = Completion::Allocated {
        runs: vec![(5, 0, 3), (6, 0, 3)],
        evicted: Vec::new(),
    };
    assert_eq!(fixture.ask(request), Some(allocated));

    // All or none across the ranks.
    let more = Request::Allocate {
        owner: OwnerId(1),
        class: PRODUCT,
        at: 1,
        count: 1,
        units: 2,
    };
    assert_eq!(fixture.ask(more), Some(Completion::Full { region: 5 }));
    assert_eq!((fixture.free(5), fixture.free(6)), (1, 1));
    assert_eq!(fixture.detach(1, Vec::new()), Some(Completion::Done));
    assert_eq!((fixture.free(5), fixture.free(6)), (4, 4));
}

#[test]
fn a_transfer_moves_an_owner_and_copies_what_the_target_group_lacks() {
    let mut fixture = Fixture::new();
    let shared = fixture.cached(1, 7);
    fixture.attach(2, PREFILL);
    fixture.pin(2, 0, &[7]);
    let private = fixture.allocate(2, 1, 1)[0];

    // R3: the task starts when the source group has ended its batches.
    let id = fixture.submit(
        QUEUE,
        Request::Transfer {
            owner: OwnerId(2),
            group: DECODE,
            after: vec![(PREFILL, 2)],
        },
    );
    assert_eq!(fixture.pool.started(), Vec::new());
    // An owner in transfer takes only a detach.
    assert_eq!(
        fixture.release(2, 1, 1, Vec::new()),
        Some(Completion::Refused {
            cause: Cause::Owner
        })
    );
    fixture.pool.advance(SERVING, PREFILL, 2);
    fixture.checked();
    assert_eq!(fixture.pool.started(), [(QUEUE, id)]);

    // The task copies both blocks into the decode group's memory and the slot into a row there.
    let work = fixture.pool.work((QUEUE, id));
    let copied: Vec<_> = work
        .iter()
        .map(|(_, from, to, _, _)| (from.clone(), to.clone()))
        .collect();
    assert_eq!(
        copied[..2],
        [
            (vec![(UNITS, shared, 1)], vec![(UNITS, shared, 1)]),
            (vec![(UNITS, private, 1)], vec![(UNITS, private, 1)])
        ]
    );
    assert_eq!(copied[2], (vec![(1, 1, 1)], vec![(2, 1, 1)]));

    // R9: what the task holds is not reclaimed before its end.
    assert_eq!(
        fixture.ask(Request::Evict { keys: vec![key(7)] }),
        Some(Completion::Evicted { keys: Vec::new() })
    );
    fixture.pool.ended((QUEUE, id), true);
    fixture.checked();
    let runs = vec![
        (KV, 0, (UNITS, shared, 1), false),
        (KV, 1, (UNITS, private, 1), true),
    ];
    assert_eq!(
        fixture.answer(id),
        Some(Completion::Transferred { slot: 1, runs })
    );
    assert_eq!(fixture.free(1), 2);

    // The shared block is valid in both groups; the private one in the decode group alone.
    assert_eq!(fixture.listing(), [(key(7), KV, 64, 0b001, 0b11)]);
    assert_eq!(
        fixture.commit(2, 1, 8, Vec::new()),
        Some(Completion::Committed)
    );
    assert_eq!(fixture.listing()[1], (key(8), KV, 64, 0b001, 0b10));
}

#[test]
fn a_failed_transfer_leaves_the_owner_in_its_group() {
    let mut fixture = Fixture::new();
    fixture.attach(1, PREFILL);
    fixture.allocate(1, 0, 1);
    let id = fixture.submit(
        QUEUE,
        Request::Transfer {
            owner: OwnerId(1),
            group: DECODE,
            after: Vec::new(),
        },
    );
    assert_eq!(fixture.free(2), 1);

    fixture.pool.ended((QUEUE, id), false);
    fixture.checked();
    assert_eq!(fixture.answer(id), Some(Completion::Failed));
    assert_eq!(fixture.free(2), 2);
    assert_eq!(fixture.release(1, 0, 1, Vec::new()), Some(Completion::Done));

    // Without a free row in the target group the transfer fails at once.
    fixture.attach(2, DECODE);
    fixture.attach(3, DECODE);
    let request = Request::Transfer {
        owner: OwnerId(1),
        group: DECODE,
        after: Vec::new(),
    };
    assert_eq!(fixture.ask(request), Some(Completion::Failed));
    assert_eq!(fixture.detach(1, Vec::new()), Some(Completion::Done));
}

#[test]
fn a_transfer_holds_its_blocks_until_its_task_ends_whatever_its_owner_does() {
    let mut fixture = Fixture::new();
    fixture.attach(1, PREFILL);
    fixture.allocate(1, 0, 2);
    let id = fixture.submit(
        QUEUE,
        Request::Transfer {
            owner: OwnerId(1),
            group: DECODE,
            after: Vec::new(),
        },
    );

    // R9: the detach releases the owner's blocks, and the task still holds them.
    assert_eq!(fixture.detach(1, Vec::new()), Some(Completion::Done));
    assert_eq!(
        (fixture.free(UNITS), fixture.free(1), fixture.free(2)),
        (2, 1, 1)
    );
    assert_eq!(fixture.pool.capacity(QUEUE, UNITS), (2, 2, 0));

    fixture.pool.ended((QUEUE, id), true);
    fixture.checked();
    assert_eq!(fixture.answer(id), Some(Completion::Failed));
    assert_eq!(
        (fixture.free(UNITS), fixture.free(1), fixture.free(2)),
        (4, 2, 2)
    );
}

#[test]
fn a_transfer_waits_while_another_fills_the_target_memory_of_a_block_it_holds() {
    let mut fixture = Fixture::new();
    fixture.cached(1, 7);
    fixture.attach(2, PREFILL);
    fixture.attach(3, PREFILL);
    fixture.pin(2, 0, &[7]);
    fixture.pin(3, 0, &[7]);

    let transfer = |owner| Request::Transfer {
        owner: OwnerId(owner),
        group: DECODE,
        after: Vec::new(),
    };
    let (first, second) = (
        fixture.submit(QUEUE, transfer(2)),
        fixture.submit(QUEUE, transfer(3)),
    );
    assert_eq!(fixture.pool.started(), [(QUEUE, first)]);

    // When the first has ended the block is valid in the target group, and
    // the second copies only its slot.
    fixture.pool.ended((QUEUE, first), true);
    fixture.checked();
    assert_eq!(fixture.pool.started(), [(QUEUE, second)]);
    assert_eq!(fixture.pool.work((QUEUE, second)).len(), 1);
    fixture.pool.ended((QUEUE, second), true);
    fixture.checked();
    assert!(matches!(
        fixture.answer(second),
        Some(Completion::Transferred { slot: 2, .. })
    ));
}

#[test]
fn a_prefetch_brings_the_leading_keys_into_the_host_tier_by_its_deadline() {
    let mut fixture = Fixture::new();
    let keys = vec![key(1), key(2), key(3)];
    let request = Request::Prefetch {
        class: KV,
        units: 0,
        keys,
        source: Source::Remote,
        deadline: 50,
    };
    let id = fixture.submit(QUEUE, request);

    // The host pool has two units: the task is for the first two keys.
    assert_eq!(fixture.pool.started(), [(QUEUE, id)]);
    assert_eq!(fixture.pool.work((QUEUE, id)).len(), 2);
    fixture.pool.done((QUEUE, id), 64);
    fixture.checked();
    assert_eq!(fixture.listing(), [(key(1), KV, 64, 0b010, 0)]);

    // At the deadline the request is answered with what was found; R9: the
    // block of the second key stays held until the task has ended.
    fixture.pool.tick(50);
    assert_eq!(
        fixture.answer(id),
        Some(Completion::Prefetched { found: 1 })
    );
    assert_eq!(fixture.pool.stopped(), [(QUEUE, id)]);
    assert_eq!(fixture.free(HOST), 0);
    fixture.pool.done((QUEUE, id), 64);
    fixture.checked();
    assert_eq!((fixture.free(HOST), fixture.listing().len()), (1, 1));

    // A key that is on the host counts as found without a task.
    let again = Request::Prefetch {
        class: KV,
        units: 0,
        keys: vec![key(1)],
        source: Source::Peers,
        deadline: 90,
    };
    assert_eq!(
        fixture.ask(again),
        Some(Completion::Prefetched { found: 1 })
    );
}

#[test]
fn a_store_to_the_remote_store_is_answered_when_the_connector_has_put_the_keys() {
    let mut fixture = Fixture::new();
    fixture.cached(1, 7);
    let store = fixture.submit(
        QUEUE,
        Request::Store {
            group: PREFILL,
            keys: vec![key(7)],
            tier: Tier::Host,
        },
    );
    fixture.finish(store, true);

    let put = fixture.submit(
        QUEUE,
        Request::Store {
            group: PREFILL,
            keys: vec![key(7), key(9)],
            tier: Tier::Remote,
        },
    );
    assert_eq!(fixture.answer(put), None);
    assert_eq!(
        fixture.pool.work((QUEUE, put)),
        [(
            Some(key(7)),
            vec![(HOST, 0, 1)],
            vec![(HOST, 0, 1)],
            u8::MAX,
            0
        )]
    );
    fixture.pool.done((QUEUE, put), 0);
    fixture.checked();
    assert_eq!(
        fixture.answer(put),
        Some(Completion::Stored { keys: vec![key(7)] })
    );
    assert_eq!(fixture.listing(), [(key(7), KV, 64, 0b111, 0b01)]);

    // Nothing left to put: answered at once.
    let again = Request::Store {
        group: PREFILL,
        keys: vec![key(7)],
        tier: Tier::Remote,
    };
    assert_eq!(
        fixture.ask(again),
        Some(Completion::Stored { keys: Vec::new() })
    );
}

#[test]
fn a_copy_made_on_trial_is_visible_to_its_epoch_alone_until_the_update_is_committed() {
    let mut fixture = Fixture::new();
    let regions = vec![vec![UNITS], Vec::new(), vec![5, 6]];
    fixture
        .pool
        .open(2, 2, true, classes(), vec![(GroupId(0), 0, 1, regions)]);
    let trial = (2, 0);

    // The epoch on trial commits a block.
    fixture.submit(
        trial,
        Request::Attach {
            owner: OwnerId(1),
            group: GroupId(0),
        },
    );
    fixture.submit(
        trial,
        Request::Allocate {
            owner: OwnerId(1),
            class: KV,
            at: 0,
            count: 1,
            units: 0,
        },
    );
    let commit = Request::Commit {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        key: key(7),
        tokens: 64,
        after: Vec::new(),
    };
    let id = fixture.submit(trial, commit);
    assert_eq!(fixture.answer(id), Some(Completion::Committed));

    // R8: the serving epoch neither lists nor pins it.
    fixture.attach(1, PREFILL);
    assert_eq!(fixture.listing(), []);
    assert_eq!(
        fixture.pin(1, 0, &[7]),
        Some(Completion::Pinned {
            runs: Vec::new(),
            host: 0
        })
    );

    fixture.pool.commit_update(2);
    assert_eq!(fixture.listing().len(), 1);
    assert!(
        matches!(fixture.pin(1, 0, &[7]), Some(Completion::Pinned { runs, .. }) if runs.len() == 1)
    );
}

#[test]
fn what_a_rolled_back_epoch_made_is_evicted() {
    let mut fixture = Fixture::new();
    let regions = vec![vec![UNITS], Vec::new(), vec![5, 6]];
    fixture
        .pool
        .open(2, 2, true, classes(), vec![(GroupId(0), 0, 1, regions)]);
    let trial = (2, 0);
    fixture.submit(
        trial,
        Request::Attach {
            owner: OwnerId(1),
            group: GroupId(0),
        },
    );
    fixture.submit(
        trial,
        Request::Allocate {
            owner: OwnerId(1),
            class: KV,
            at: 0,
            count: 2,
            units: 0,
        },
    );
    let commit = Request::Commit {
        owner: OwnerId(1),
        class: KV,
        at: 0,
        key: key(7),
        tokens: 64,
        after: Vec::new(),
    };
    fixture.submit(trial, commit);
    assert_eq!(fixture.free(UNITS), 2);

    // The engine is gone: its owners are detached, what it committed stays.
    fixture.pool.close(2);
    fixture.checked();
    assert_eq!((fixture.free(UNITS), fixture.free(1)), (3, 2));
    fixture.pool.roll_back(2);
    fixture.checked();
    assert_eq!(fixture.free(UNITS), 4);
}

#[test]
fn a_lost_engine_s_stamps_count_as_reached_and_its_waiting_commits_are_dropped() {
    let mut fixture = Fixture::new();
    fixture.cached(1, 1);
    fixture.attach(2, PREFILL);
    fixture.allocate(2, 0, 2);
    fixture.commit(2, 0, 7, vec![(PREFILL, 9)]);
    fixture.release(2, 1, 1, vec![(PREFILL, 9)]);
    let clone = fixture.submit(
        QUEUE,
        Request::Clone {
            owner: OwnerId(2),
            class: KV,
            at: 2,
            key: key(1),
        },
    );
    assert_eq!(fixture.free(UNITS), 0);

    fixture.pool.close(SERVING);
    fixture.checked();
    // Only what was committed before stays; a block whose last batch may
    // not have ended is never shared.
    assert_eq!(fixture.free(UNITS), 3);
    assert_eq!(fixture.pool.completions(), Vec::new());
    fixture
        .pool
        .take(QUEUE, RequestId(clone.0 + 1), Request::List);
    assert_eq!(fixture.pool.completions().len(), 1);
}

#[test]
fn a_resize_moves_free_units_between_a_region_and_the_quota() {
    let mut fixture = Fixture::new();
    fixture.cached(1, 1);
    fixture.attach(2, PREFILL);
    fixture.allocate(2, 0, 1);

    // A shrink never takes a private block, and a shared one only in the order.
    let shrink = Request::Resize {
        region: UNITS,
        units: 0,
    };
    assert_eq!(
        fixture.ask(shrink.clone()),
        Some(Completion::Resized {
            units: 2,
            evicted: Vec::new()
        })
    );
    fixture.ask(Request::Victims {
        region: UNITS,
        keys: vec![key(1)],
    });
    assert_eq!(
        fixture.ask(shrink),
        Some(Completion::Resized {
            units: 1,
            evicted: vec![key(1)]
        })
    );

    // Memory is released only where no region holds it: the shrunk units'
    // bytes can be withdrawn, the private block's cannot.
    assert!(!fixture.pool.withdraw(0, 128, 64));
    assert!(fixture.pool.withdraw(0, 192, 128));
    fixture.pool.grant(0, 192, 128);

    // R10: a region of another layout over the same memory grows only by
    // bytes that left this one.
    let other = fixture
        .pool
        .open_region("other layout", Tier::Device, 0, 128, 0, 0, 2);
    let grow = Request::Resize {
        region: other,
        units: 2,
    };
    assert_eq!(
        fixture.ask(grow.clone()),
        Some(Completion::Resized {
            units: 1,
            evicted: Vec::new()
        })
    );
    // The first region gets back the two units whose bytes the other does not hold.
    let back = Request::Resize {
        region: UNITS,
        units: 4,
    };
    assert_eq!(
        fixture.ask(back),
        Some(Completion::Resized {
            units: 3,
            evicted: Vec::new()
        })
    );
    assert_eq!(
        fixture.ask(grow),
        Some(Completion::Resized {
            units: 1,
            evicted: Vec::new()
        })
    );
}

#[test]
fn a_peer_s_hold_keeps_a_host_copy_until_it_is_released() {
    let mut fixture = Fixture::new();
    // Both host units hold a copy: of key 7, then of key 8.
    for (owner, name) in [(1, 7), (2, 8)] {
        fixture.cached(owner, name);
        let store = fixture.submit(
            QUEUE,
            Request::Store {
                group: PREFILL,
                keys: vec![key(name)],
                tier: Tier::Host,
            },
        );
        fixture.finish(store, true);
    }
    assert_eq!(fixture.pool.capacity(QUEUE, HOST), (0, 0, 2));

    // A key without a host copy is not held; one with a copy is, and a
    // hold is a holder: the copy is no longer evictable.
    let held = fixture.pool.hold(9, &[key(7), key(9)]);
    fixture.checked();
    assert_eq!(held, [(key(7), (HOST, 0, 1))]);
    assert_eq!(fixture.pool.capacity(QUEUE, HOST), (0, 0, 1));

    // A store that needs room evicts in the installed order and skips it.
    fixture.ask(Request::Victims {
        region: HOST,
        keys: vec![key(7), key(8)],
    });
    fixture.cached(3, 9);
    let store = Request::Store {
        group: PREFILL,
        keys: vec![key(9)],
        tier: Tier::Host,
    };
    let Some(Completion::Copy { evicted, .. }) = fixture.ask(store) else {
        panic!("no copy")
    };
    assert_eq!(evicted, [key(8)]);

    // The release ends every hold of the peer; a second finds none.
    assert!(fixture.pool.release_peer(9));
    fixture.checked();
    assert!(!fixture.pool.release_peer(9));
    assert_eq!(fixture.pool.capacity(QUEUE, HOST), (1, 0, 1));
}
