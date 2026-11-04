//! The host side of the policy protocol: toy policies written in `wat` are
//! loaded, instantiated and called the way the scheduler thread calls a policy
//! module, and each failure of §4.4 is shown to be reported as the failure it
//! is.
//!
//! The toys speak the call convention of `scheduling_guest::wire`: `sched_alloc`
//! and `sched_free` for the buffers, one export per call with `(i32 ptr, i32
//! len)`, a reply packed as `ptr << 32 | len`, the chosen index itself for
//! `route`/`place`, and JSON messages. What they test is therefore the transport
//! the mechanism relies on, with the interface's real types on both ends.

// Clippy counts an integration test crate's helpers as tests only when the
// crate says so; a failed unwrap here is the test's failure report.
#![cfg(test)]

use scheduling::guest::wire::{self, Call};
use scheduling::guest::{
    CallView, ExpertLoad, ExpertOrder, Limits, Nanos, PageId, Parameters, PhaseId, Plan, RankView,
    RequestId, RequestView, Room, Shortage, Snapshot, Storage, Victim, Work,
};
use scheduling::{Engine, PolicyError, PolicyInstance, PolicyModule};

/// Where the toys put their reply buffers, and where the bump allocator starts.
const REPLY_BASE: u32 = 2048;
const REPLY_STRIDE: u32 = 256;
const HEAP_BASE: u32 = 65536 / 2;

/// A toy policy module.
///
/// Each call of the protocol gets one of: a reply buffer holding the encoding of
/// a value, raw bytes that need not decode, the index itself (for `route` and
/// `place`), or a `wat` body written directly (for a call that traps, loops or
/// answers with an address of its own). A call not given anything answers the
/// empty reply, except `limits`, which answers the default limits so that every
/// toy instantiates.
#[derive(Default)]
struct Toy {
    omitted: Option<&'static str>,
    /// Whether the module's allocator refuses every buffer, as a module does
    /// when it cannot grow its memory.
    refuses_allocation: bool,
    replies: Vec<(&'static str, Vec<u8>)>,
    indexes: Vec<(&'static str, u32)>,
    bodies: Vec<(&'static str, String)>,
}

impl Toy {
    fn new() -> Self {
        Self::default()
    }

    /// Answers `export` with the encoding of `value`.
    fn reply<T: serde::Serialize>(mut self, export: &'static str, value: &T) -> Self {
        self.replies.push((export, wire::encode(value).unwrap()));
        self
    }

    /// Answers `export` with these bytes, which need not decode.
    fn raw_reply(mut self, export: &'static str, bytes: &[u8]) -> Self {
        self.replies.push((export, bytes.to_vec()));
        self
    }

    /// Answers `export` with the index itself.
    fn index(mut self, export: &'static str, index: u32) -> Self {
        self.indexes.push((export, index));
        self
    }

    /// Answers `export` with a body written in `wat`, whose value is the
    /// export's result.
    fn body(mut self, export: &'static str, body: impl Into<String>) -> Self {
        self.bodies.push((export, body.into()));
        self
    }

    /// Makes the module's allocator refuse every buffer.
    fn refusing_allocation(mut self) -> Self {
        self.refuses_allocation = true;
        self
    }

    /// Leaves `export` out of the module.
    fn omit(mut self, export: &'static str) -> Self {
        self.omitted = Some(export);
        self
    }

    /// The module's text.
    fn wat(&self) -> String {
        // Explicit replies first, then the default limits reply when the toy
        // does not state its own, all in the module's data.
        let mut data: Vec<(u32, &[u8])> = Vec::new();
        let mut reply_offsets = Vec::new();
        for (_, bytes) in &self.replies {
            assert!(
                bytes.len() < REPLY_STRIDE as usize,
                "a toy reply fits one data segment"
            );
            let offset = REPLY_BASE + REPLY_STRIDE * data.len() as u32;
            reply_offsets.push(offset);
            data.push((offset, bytes.as_slice()));
        }
        let states_limits = self
            .replies
            .iter()
            .any(|(export, _)| *export == Call::Limits.export())
            || self
                .bodies
                .iter()
                .any(|(export, _)| *export == Call::Limits.export());
        let default_limits = wire::encode(&Limits::default()).unwrap();
        let default_limits_offset = if states_limits {
            None
        } else {
            let offset = REPLY_BASE + REPLY_STRIDE * data.len() as u32;
            data.push((offset, default_limits.as_slice()));
            Some(offset)
        };

        let allocator = if self.refuses_allocation {
            r#"  (func (export "sched_alloc") (param $len i32) (result i32)
    (i32.const 0))
"#
        } else {
            r#"  (func (export "sched_alloc") (param $len i32) (result i32)
    (local $ptr i32)
    (local.set $ptr (global.get $heap))
    (global.set $heap
      (i32.add (global.get $heap) (i32.add (local.get $len) (i32.const 8))))
    (local.get $ptr))
"#
        };
        let mut text = format!(
            r#"(module
  (memory (export "memory") 1)
{allocator}  (func (export "sched_free") (param $ptr i32) (param $len i32))
"#
        );
        text.push_str(&format!(
            "  (global $heap (mut i32) (i32.const {HEAP_BASE}))\n"
        ));
        for (offset, bytes) in &data {
            text.push_str(&format!(
                "  (data (i32.const {offset}) \"{}\")\n",
                escape(bytes)
            ));
        }

        for call in Call::ALL {
            let export = call.export();
            if self.omitted == Some(export) {
                continue;
            }
            let is_index = call.replies_with_index();
            let body = if let Some((_, body)) = self.bodies.iter().find(|(name, _)| *name == export)
            {
                body.clone()
            } else if let Some(index) = self.replies.iter().position(|(name, _)| *name == export) {
                let (offset, bytes) = (reply_offsets[index], &self.replies[index].1);
                packed(offset, bytes.len())
            } else if let Some((_, index)) = self.indexes.iter().find(|(name, _)| *name == export) {
                format!("(i32.const {index})")
            } else if is_index {
                "(i32.const 0)".into()
            } else if call == Call::Limits {
                match default_limits_offset {
                    Some(offset) => packed(offset, default_limits.len()),
                    None => "i64.const 0".into(),
                }
            } else {
                "i64.const 0".into()
            };
            match call {
                Call::Limits => text.push_str(&format!(
                    "  (func (export \"{export}\") (result i64)\n    {body})\n"
                )),
                _ if is_index => text.push_str(&format!(
                    "  (func (export \"{export}\") (param i32 i32) (result i32)\n    {body})\n"
                )),
                _ => text.push_str(&format!(
                    "  (func (export \"{export}\") (param i32 i32) (result i64)\n    {body})\n"
                )),
            }
        }
        text.push_str(")\n");
        text
    }
}

/// The `wat` expression answering with the buffer at `offset` of `len` bytes.
fn packed(offset: u32, len: usize) -> String {
    if len == 0 {
        return "i64.const 0".into();
    }
    format!("(i64.or (i64.shl (i64.const {offset}) (i64.const 32)) (i64.const {len}))")
}

/// Escapes bytes as a `wat` data string literal.
fn escape(bytes: &[u8]) -> String {
    let mut escaped = String::new();
    for byte in bytes {
        match byte {
            b'"' => escaped.push_str("\\\""),
            b'\\' => escaped.push_str("\\\\"),
            0x20..=0x7e => escaped.push(*byte as char),
            other => escaped.push_str(&format!("\\{other:02x}")),
        }
    }
    escaped
}

/// The engine of one test.
fn engine() -> Engine {
    Engine::new().unwrap()
}

/// A queued token request, as the mechanism views one.
///
/// Built from the encoding rather than from a struct literal: a view is an
/// append-only list, so the fields this test does not name are the defaults the
/// interface gives them, and a field added to `RequestView` later does not
/// touch this test.
fn request() -> RequestView {
    wire::decode(br#"{"id":1,"state":{"state":"queued"},"class":{"class":"token"}}"#).unwrap()
}

/// A request's first call on a block.
fn call() -> CallView {
    wire::decode(br#"{"request":1,"phase":1,"kind":{"kind":"prefill"}}"#).unwrap()
}

/// Loads and instantiates `toy` against the default limits.
fn policy(engine: &Engine, toy: Toy) -> PolicyInstance {
    let module = PolicyModule::load(engine, toy.wat().as_bytes()).unwrap();
    module
        .instantiate(
            &Snapshot::default(),
            &Parameters::default(),
            &Limits::default(),
        )
        .unwrap()
}

#[test]
fn every_decision_reaches_its_export_and_answers() {
    let engine = engine();
    let plan = Plan {
        work: vec![Work {
            request: RequestId(7),
            phase: PhaseId(1),
            tokens: 32,
        }],
        ..Plan::default()
    };
    let victims = vec![Victim::Request(RequestId(3)), Victim::Page(PageId(4))];
    let toy = Toy::new()
        .reply(Call::Dispatch.export(), &plan)
        .reply(Call::Reclaim.export(), &victims)
        .reply(Call::Warm.export(), &vec![PageId(9)])
        .reply(
            Call::Experts.export(),
            &ExpertOrder {
                placement: vec![1, 0],
            },
        )
        .index(Call::Route.export(), 2)
        .index(Call::Place.export(), 1);
    let mut policy = policy(&engine, toy);

    assert_eq!(policy.dispatch(&Room::default()).unwrap().work, plan.work);
    assert_eq!(
        policy
            .reclaim(&Shortage {
                kind: Storage::KvUnits,
                need: 4,
            })
            .unwrap(),
        victims
    );
    assert_eq!(policy.warm(&Room::default()).unwrap(), vec![PageId(9)]);
    assert_eq!(
        policy.experts(&ExpertLoad::default()).unwrap().placement,
        vec![1, 0]
    );
    assert_eq!(policy.route(&request(), &[]).unwrap(), 2);

    assert_eq!(policy.place(&call(), &[]).unwrap(), 1);
    policy.observe(&[]).unwrap();
    assert_eq!(policy.failure(), None);
}

#[test]
fn reclaim_and_warm_may_answer_nothing() {
    let engine = engine();
    // A toy whose reclaim and warm answer the empty reply frees and warms
    // nothing; that is a decision, not a failure.
    let mut policy = policy(&engine, Toy::new());

    assert!(policy
        .reclaim(&Shortage {
            kind: Storage::HostPages,
            need: 1,
        })
        .unwrap()
        .is_empty());
    assert!(policy.warm(&Room::default()).unwrap().is_empty());
}

#[test]
fn an_answer_with_a_field_the_host_does_not_know_still_decodes() {
    let engine = engine();
    // The views and decisions are append-only lists: a policy built against a
    // later interface writes fields this host does not know, and its answer
    // still means what the host knows of it.
    let toy = Toy::new().raw_reply(
        Call::Dispatch.export(),
        br#"{"work":[{"request":7,"phase":1,"tokens":32,"future":true}],"future_entry":[]}"#,
    );
    let mut policy = policy(&engine, toy);

    let plan = policy.dispatch(&Room::default()).unwrap();
    assert_eq!(
        plan.work,
        vec![Work {
            request: RequestId(7),
            phase: PhaseId(1),
            tokens: 32,
        }]
    );
}

#[test]
fn the_message_is_written_into_the_module() {
    let engine = engine();
    // The toy answers with the first byte of the message it was handed, so the
    // host must have written that message at the pointer the module returned.
    // The route message is the interface's `(RequestView, &[RankView])` tuple
    // (the guest decodes a tuple too), and a tuple encodes as a JSON sequence:
    // its first byte is `[`, not the `{` of one object.
    let toy = Toy::new().body(Call::Route.export(), "(i32.load8_u (local.get 0))");
    let mut policy = policy(&engine, toy);
    let ranks: Vec<RankView> = Vec::new();
    let message = wire::encode(&(request(), &ranks)).unwrap();

    assert_eq!(
        policy.route(&request(), &[]).unwrap(),
        usize::from(message[0])
    );
    assert_eq!(message[0], b'[');
}

/// An artifact outlives the engine that made it: the Program Server
/// precompiles a policy with one engine, a scheduler thread loads it with
/// another built by `Engine::new()`, and the target `new` pins is what makes
/// the two agree (a module from a differently targeted engine does not load).
#[test]
fn a_precompiled_module_loads_and_runs_in_another_engine() {
    let artifact = engine().precompile(Toy::new().wat().as_bytes()).unwrap();
    // A second engine of the same process: the store's artifact is not tied to
    // the compiling engine, only to the target both are built for.
    let second = engine();
    let module = PolicyModule::load(&second, &artifact).unwrap();
    let policy = module
        .instantiate(
            &Snapshot::default(),
            &Parameters::default(),
            &Limits::default(),
        )
        .unwrap();

    assert!(policy.failure().is_none());
    // The module answers the limits it was initialized with, so the call runs.
    let limits = policy.limits();
    assert_eq!(limits.max_batch, Limits::default().max_batch);
    assert_eq!(
        limits.max_num_batched_tokens,
        Limits::default().max_num_batched_tokens
    );
}

#[test]
fn fuel_is_renewed_for_every_call() {
    let engine = engine();
    // A dispatch that costs a measurable amount, so the budget can be set
    // between one call's cost and two calls' cost: three calls then prove the
    // budget is renewed per call rather than shared by the instance.
    // Dispatch is one of the calls whose answer must carry something, so the
    // body runs the loop and then answers with an empty plan: every call of the
    // loop below is a complete dispatch, not one the host rejects for silence.
    let reply = wire::encode(&Plan::default()).unwrap();
    let toy = Toy::new()
        .reply(Call::Dispatch.export(), &Plan::default())
        .body(
            Call::Dispatch.export(),
            format!(
                r#"
    (local $n i32)
    (local.set $n (i32.const 1000))
    (block $done
      (loop $l
        (br_if $done (i32.eqz (local.get $n)))
        (local.set $n (i32.sub (local.get $n) (i32.const 1)))
        (br $l)))
    {}"#,
                packed(REPLY_BASE, reply.len()),
            ),
        );
    let mut policy = policy(&engine, toy);

    assert!(policy.dispatch(&Room::default()).unwrap().work.is_empty());
    let per_call = policy.fuel_consumed();
    assert!(per_call > 0, "a call spends fuel");
    policy.set_call_fuel(per_call + per_call / 2);

    for _ in 0..3 {
        assert!(policy.dispatch(&Room::default()).unwrap().work.is_empty());
        assert!(policy.failure().is_none());
    }
}

#[test]
fn a_decision_that_never_returns_runs_out_of_fuel() {
    let engine = engine();
    let toy = Toy::new().body(
        Call::Dispatch.export(),
        r#"
    (loop $forever (br $forever))
    (i64.const 0)"#,
    );
    let mut policy = policy(&engine, toy);
    policy.set_call_fuel(10_000);

    match policy.dispatch(&Room::default()) {
        Err(PolicyError::FuelExhausted { call }) => assert_eq!(call, Call::Dispatch),
        other => panic!("expected a fuel failure, got {other:?}"),
    }
}

#[test]
fn a_decision_that_traps_fails_and_ends_the_instance() {
    let engine = engine();
    let toy = Toy::new().body(Call::Observe.export(), "unreachable");
    let mut policy = policy(&engine, toy);

    match policy.observe(&[]) {
        Err(PolicyError::Trap { call, .. }) => assert_eq!(call, Call::Observe),
        other => panic!("expected a trap, got {other:?}"),
    }
    assert_eq!(policy.failure(), Some(Call::Observe));
    // The instance is not called again; the mechanism makes a new one from the
    // module it keeps.
    match policy.dispatch(&Room::default()) {
        Err(PolicyError::Poisoned { call }) => assert_eq!(call, Call::Observe),
        other => panic!("expected the instance to be unusable, got {other:?}"),
    }
}

#[test]
fn an_answer_that_does_not_decode_is_malformed() {
    let engine = engine();
    let toy = Toy::new().raw_reply(Call::Dispatch.export(), b"not a plan");
    let mut policy = policy(&engine, toy);

    match policy.dispatch(&Room::default()) {
        Err(PolicyError::Malformed { call, .. }) => assert_eq!(call, Call::Dispatch),
        other => panic!("expected a malformed answer, got {other:?}"),
    }
}

#[test]
fn a_module_that_cannot_reserve_the_message_fails_the_call() {
    let engine = engine();
    // The allocator refuses, so the host has nowhere to write the message: a
    // protocol failure of this call, not a trap and not a decision.
    let toy = Toy::new().refusing_allocation();
    let module = PolicyModule::load(&engine, toy.wat().as_bytes()).unwrap();
    // `limits()` is read at instantiation and needs a buffer too, so the
    // module refuses to instantiate before any decision is made.
    match module.instantiate(
        &Snapshot::default(),
        &Parameters::default(),
        &Limits::default(),
    ) {
        Err(PolicyError::Malformed { call, .. }) => assert_eq!(call, Call::Init),
        Err(other) => panic!("expected a malformed call, got {other:?}"),
        Ok(_) => panic!("expected a malformed call, got an instance"),
    }
}

#[test]
fn a_reply_outside_the_module_memory_is_malformed() {
    let engine = engine();
    // Address 0xffff_0000 is past the module's one page: the pointer is the
    // high half of the reply and 16 its low half.
    let toy = Toy::new().body(
        Call::Dispatch.export(),
        "(i64.or (i64.shl (i64.const 65535) (i64.const 48)) (i64.const 16))",
    );
    let mut policy = policy(&engine, toy);

    match policy.dispatch(&Room::default()) {
        Err(PolicyError::Malformed { call, .. }) => assert_eq!(call, Call::Dispatch),
        other => panic!("expected a malformed answer, got {other:?}"),
    }
}

#[test]
fn a_module_without_a_decision_does_not_instantiate() {
    let engine = engine();
    let toy = Toy::new().omit(Call::Warm.export());
    let module = PolicyModule::load(&engine, toy.wat().as_bytes()).unwrap();

    match module.instantiate(
        &Snapshot::default(),
        &Parameters::default(),
        &Limits::default(),
    ) {
        Err(PolicyError::Instantiate { message }) => {
            assert!(message.contains("sched_warm"), "{message}");
        }
        Err(other) => panic!("expected a missing export, got {other:?}"),
        Ok(_) => panic!("expected a missing export, got an instance"),
    }
}

#[test]
fn limits_outside_the_build_config_are_refused() {
    let engine = engine();
    let wider = Limits {
        max_batch: Limits::default().max_batch + 1,
        ..Limits::default()
    };
    let toy = Toy::new().reply(Call::Limits.export(), &wider);
    let module = PolicyModule::load(&engine, toy.wat().as_bytes()).unwrap();

    match module.instantiate(
        &Snapshot::default(),
        &Parameters::default(),
        &Limits::default(),
    ) {
        Err(PolicyError::BadLimits { message }) => {
            assert!(message.contains("max_batch"), "{message}");
        }
        Err(other) => panic!("expected a limits failure, got {other:?}"),
        Ok(_) => panic!("expected a limits failure, got an instance"),
    }
}

#[test]
fn a_starvation_timeout_above_thirty_seconds_is_refused() {
    let engine = engine();
    let late = Limits {
        timeout: Nanos(scheduling::MAX_STARVATION_TIMEOUT.0 + 1),
        ..Limits::default()
    };
    let toy = Toy::new().reply(Call::Limits.export(), &late);
    // The build config's own ceiling is above the design's 30 seconds, so only
    // the design's cap can refuse this.
    let ceilings = Limits {
        timeout: Nanos(scheduling::MAX_STARVATION_TIMEOUT.0 * 2),
        ..Limits::default()
    };
    let module = PolicyModule::load(&engine, toy.wat().as_bytes()).unwrap();

    match module.instantiate(&Snapshot::default(), &Parameters::default(), &ceilings) {
        Err(PolicyError::BadLimits { message }) => {
            assert!(message.contains("timeout"), "{message}");
        }
        Err(other) => panic!("expected a limits failure, got {other:?}"),
        Ok(_) => panic!("expected a limits failure, got an instance"),
    }
}

#[test]
fn a_prefetch_floor_below_the_build_config_s_floor_is_refused() {
    let engine = engine();
    // The mechanism deepens a prefix hit only by at least the policy's floor;
    // a policy that asks to deepen by less than the epoch guarantees would
    // prefetch more sparingly than the build config allows.
    let sparing = Limits {
        prefetch_min_tokens: Limits::default().prefetch_min_tokens - 1,
        ..Limits::default()
    };
    let toy = Toy::new().reply(Call::Limits.export(), &sparing);
    let module = PolicyModule::load(&engine, toy.wat().as_bytes()).unwrap();

    match module.instantiate(
        &Snapshot::default(),
        &Parameters::default(),
        &Limits::default(),
    ) {
        Err(PolicyError::BadLimits { message }) => {
            assert!(message.contains("prefetch_min_tokens"), "{message}");
        }
        Err(other) => panic!("expected a limits failure, got {other:?}"),
        Ok(_) => panic!("expected a limits failure, got an instance"),
    }
}

#[test]
fn a_prefetch_floor_above_the_build_config_s_floor_is_a_decision() {
    let engine = engine();
    // Prefetching more eagerly than the build config's floor is the policy's
    // decision to make, not a limit outside the epoch.
    let eager = Limits {
        prefetch_min_tokens: Limits::default().prefetch_min_tokens * 2,
        ..Limits::default()
    };
    let toy = Toy::new().reply(Call::Limits.export(), &eager);
    let policy = policy(&engine, toy);

    assert_eq!(policy.limits().prefetch_min_tokens, 512);
}

#[test]
fn the_policy_s_limits_are_read_at_instantiation() {
    let engine = engine();
    let tighter = Limits {
        chunked_prefill_size: 2048,
        max_batch: 16,
        ..Limits::default()
    };
    let toy = Toy::new().reply(Call::Limits.export(), &tighter);
    let policy = policy(&engine, toy);

    assert_eq!(policy.limits().chunked_prefill_size, 2048);
    assert_eq!(policy.limits().max_batch, 16);
}

#[test]
fn a_precompiled_module_round_trips() {
    let engine = engine();
    let plan = Plan {
        admit: vec![RequestId(5)],
        ..Plan::default()
    };
    let toy = Toy::new().reply(Call::Dispatch.export(), &plan);
    let text = toy.wat();

    // The Program Server compiles and precompiles for this host and serves the
    // artifact; the main process loads it (§5.3, §6.2).
    let artifact = engine.precompile(text.as_bytes()).unwrap();
    let module = engine.module(&artifact).unwrap();
    let mut policy = module
        .instantiate(
            &Snapshot::default(),
            &Parameters::default(),
            &Limits::default(),
        )
        .unwrap();

    assert_eq!(
        policy.dispatch(&Room::default()).unwrap().admit,
        vec![RequestId(5)]
    );
}

#[test]
fn bytes_that_are_not_a_module_are_rejected() {
    let engine = engine();
    match engine.module(b"not a wasm module") {
        Err(PolicyError::Load { .. }) => {}
        Err(other) => panic!("expected a load failure, got {other:?}"),
        Ok(_) => panic!("expected a load failure, got a module"),
    }
}

#[test]
fn two_instances_of_one_module_are_independent() {
    let engine = engine();
    // The serving instance, the control thread's route instance and a trial
    // instance during precopy are all made from one module (§4.4, §7.3): a
    // failure of one must leave the others serving.
    let plan = Plan {
        admit: vec![RequestId(1)],
        ..Plan::default()
    };
    let toy = Toy::new()
        .body(Call::Observe.export(), "unreachable")
        .reply(Call::Dispatch.export(), &plan);
    let module = PolicyModule::load(&engine, toy.wat().as_bytes()).unwrap();
    let mut failing = module
        .instantiate(
            &Snapshot::default(),
            &Parameters::default(),
            &Limits::default(),
        )
        .unwrap();
    let mut serving = module
        .instantiate(
            &Snapshot::default(),
            &Parameters::default(),
            &Limits::default(),
        )
        .unwrap();

    assert!(matches!(
        failing.observe(&[]),
        Err(PolicyError::Trap {
            call: Call::Observe,
            ..
        })
    ));
    assert_eq!(
        serving.dispatch(&Room::default()).unwrap().admit,
        vec![RequestId(1)]
    );
    assert!(serving.failure().is_none());
}

#[test]
fn a_module_is_shared_by_the_scheduler_threads_and_an_instance_moves_to_one() {
    // Every DP rank's scheduler thread instantiates the same policy from the
    // module it holds (§4.4: one instance per scheduler thread), and the route
    // instance on the control thread is another; so a module is shared across
    // threads and an instance can be built on one and called on another.
    fn shared<T: Send + Sync + Clone>() {}
    fn movable<T: Send>() {}
    shared::<PolicyModule>();
    shared::<Engine>();
    movable::<PolicyInstance>();
}

#[test]
fn the_default_budget_is_a_call_budget() {
    let engine = engine();
    let policy = policy(&engine, Toy::new());
    assert_eq!(policy.call_fuel(), scheduling::DEFAULT_CALL_FUEL);
}
