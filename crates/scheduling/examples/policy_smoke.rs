//! Drives one compiled policy module through every call of the interface.
//!
//! This is the check that a policy crate compiled to `wasm32` and this host
//! agree on the call convention (§4.4) — the two sides are written from the
//! same types (`scheduling_guest`) but the exports, the buffers and the JSON
//! only meet when a real module runs. It instantiates the module from a small
//! snapshot, drives `observe` and `dispatch` once and then each other call
//! once, and prints what the policy decided. It exits non-zero on the first
//! failure, with the failure named as the mechanism reports it.
//!
//! ```text
//! cargo run -p scheduling --example policy_smoke -- <module.wasm> [--precompile]
//! ```
//!
//! `<module.wasm>` is a `scheduling-policy-*` crate built for
//! `wasm32-unknown-unknown`. With `--precompile` the bytes are first
//! precompiled the way the Program Server does it (§5.3), and the artifact is
//! what loads, which is the path a deployment takes.
//!
//! Each decision is reported with the calls it took and the fuel it burned, and
//! `--steps <n>` repeats `observe` + `dispatch` to give the per-step figure of
//! §12. The figure a deployment sets `PolicyInstance::set_call_fuel` from is a
//! step of a real workload, which this fixture only bounds from below: a step
//! here observes one event and dispatches two requests.

use std::error::Error;
use std::process::ExitCode;

use scheduling::guest::wire;
use scheduling::guest::{
    CallView, Duration, Event, ExpertLoad, Limits, Parameters, Plan, RankView, ReplicaView,
    RequestId, RequestView, Room, Shortage, Snapshot, Storage,
};
use scheduling::{Engine, PolicyInstance};
use serde::de::DeserializeOwned;

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(path) = arguments.next() else {
        eprintln!("usage: policy_smoke <module.wasm> [--precompile] [--steps <n>]");
        return ExitCode::from(2);
    };
    let mut precompile = false;
    let mut steps = 1u32;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--precompile" => precompile = true,
            "--steps" => match arguments.next().and_then(|n| n.parse().ok()) {
                Some(n) if n > 0 => steps = n,
                _ => {
                    eprintln!("usage: policy_smoke <module.wasm> [--precompile] [--steps <n>]");
                    return ExitCode::from(2);
                }
            },
            other => {
                eprintln!("unknown argument `{other}`");
                return ExitCode::from(2);
            }
        }
    }

    match run(&path, precompile, steps) {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            // A `PolicyError`'s message names the call that failed, which is
            // what the daemon reports as the update's first error (§7.3).
            eprintln!("{failure}");
            ExitCode::FAILURE
        }
    }
}

/// Loads the module, instantiates it, drives every call once.
fn run(path: &str, precompile: bool, steps: u32) -> Result<(), Box<dyn Error>> {
    let bytes = std::fs::read(path).map_err(|error| scheduling::PolicyError::Load {
        message: format!("cannot read {path}: {error}"),
    })?;
    let engine = Engine::new()?;
    let module = if precompile {
        let artifact = engine.precompile(&bytes)?;
        eprintln!(
            "precompiled {} bytes into {} bytes",
            bytes.len(),
            artifact.len()
        );
        engine.module(&artifact)?
    } else {
        engine.module(&bytes)?
    };

    // The program's arguments construct the policy; the state is its `init`.
    let parameters = Parameters::default();
    let ceilings = Limits::default();
    let mut policy = module.instantiate(&snapshot()?, &parameters, &ceilings)?;
    let limits = policy.limits();
    println!(
        "limits: chunked_prefill_size {} max_batch {} max_num_batched_tokens {} \
         prefetch_min_tokens {} timeout {} ns",
        limits.chunked_prefill_size,
        limits.max_batch,
        limits.max_num_batched_tokens,
        limits.prefetch_min_tokens,
        limits.timeout.0
    );

    // An arrival each, then a resident request's view changing: a policy that
    // sizes a chunk from the prefilled prefix, as `fcfs` does, reads it from
    // `Updated` (§4.4: a resident request's changing view reaches the policy
    // as a state change, not as a per-step dump of every resident).
    let events = [
        Event::Arrived {
            request: request(1)?,
        },
        Event::Arrived {
            request: request(2)?,
        },
        Event::Admitted {
            request: RequestId(1),
        },
        Event::Updated {
            request: updated(1, 128, 64)?,
        },
        Event::Updated {
            request: updated(1, 128, 128)?,
        },
        Event::Stepped {
            step: 0,
            duration: Duration(0),
        },
    ];
    let before = Meter::of(&policy);
    policy.observe(&events)?;
    before.report(&policy, "observe (6 events)");

    let room = Room {
        step: 1,
        kv_units: 4096,
        slots: 8,
        token_budget: limits.max_num_batched_tokens,
        call_budget: limits.max_batch,
        page_tokens: 64,
        ..Room::default()
    };
    let step_room = room.clone();
    let mut step_fuel = 0;
    for step in 1..=steps {
        let before = Meter::of(&policy);
        if step > 1 {
            policy.observe(&[Event::Stepped {
                step: u64::from(step - 1),
                duration: Duration(0),
            }])?;
        }
        let mut room = step_room.clone();
        room.step = u64::from(step);
        let plan = policy.dispatch(&room)?;
        if step == 1 {
            print_plan(&plan);
        }
        // Step 1's `observe` ran above, outside the loop; later steps observe
        // the one event a quiet step carries.
        let label = if step == 1 {
            format!("step {step} (dispatch; the observe is above)")
        } else {
            format!("step {step} (observe + dispatch)")
        };
        step_fuel += before.report(&policy, &label);
    }
    if steps > 1 {
        println!(
            "per step over {steps} steps on this fixture: {} fuel (a floor: a real step \
             observes what changed, not one event)",
            step_fuel / u64::from(steps)
        );
    }

    let before = Meter::of(&policy);
    let victims = policy.reclaim(&Shortage {
        kind: Storage::KvUnits,
        need: 1,
    })?;
    println!("reclaim: {} victims", victims.len());
    before.report(&policy, "reclaim");

    let ranks: Vec<RankView> = decode(br#"[{"index":0,"in_flight":1},{"index":1,"in_flight":0}]"#)?;
    let before = Meter::of(&policy);
    let rank = policy.route(&request(1)?, &ranks)?;
    println!("route: rank {rank}");
    before.report(&policy, "route");

    let call: CallView = decode(br#"{"request":1,"phase":1,"kind":{"kind":"prefill"}}"#)?;
    let replicas: Vec<ReplicaView> = decode(
        br#"[{"index":0,"block":0,"kinds":[{"kind":"prefill"},{"kind":"decode"}],
              "resident_requests":1,"capacity":7}]"#,
    )?;
    let before = Meter::of(&policy);
    let replica = policy.place(&call, &replicas)?;
    println!("place: replica {replica}");
    before.report(&policy, "place");

    let before = Meter::of(&policy);
    let warm = policy.warm(&room)?.len();
    println!("warm: {warm} pages");
    before.report(&policy, "warm");

    let load = ExpertLoad {
        ranks: 2,
        experts: 4,
        tokens: vec![10, 0, 4, 4],
    };
    let before = Meter::of(&policy);
    let placements = policy.experts(&load)?.placement.len();
    println!("experts: {placements} placements");
    before.report(&policy, "experts");

    println!(
        "total: fuel {}, calls {:?}",
        policy.fuel_consumed(),
        policy
            .cost()
            .each()
            .filter(|(_, calls)| *calls > 0)
            .collect::<Vec<_>>()
    );
    println!("ok: every call answered");
    Ok(())
}

/// A small state at instantiation: two queued token requests, one prefill
/// phase, two ranks.
///
/// Written as the encoding rather than as a struct literal, so a view that
/// gains a field does not change this harness.
fn snapshot() -> Result<Snapshot, Box<dyn Error>> {
    decode(
        br#"{
            "step": 0,
            "requests": [
                {"id":1,"state":{"state":"queued"},"class":{"class":"token"},
                 "prompt_tokens":128,"phase":1},
                {"id":2,"state":{"state":"queued"},"class":{"class":"token"},
                 "prompt_tokens":64,"phase":1}
            ],
            "phases": [{"id":1,"kind":{"kind":"prefill"},"block":0}],
            "ranks": [{"index":0,"in_flight":0},{"index":1,"in_flight":0}]
        }"#,
    )
}

/// One resident token request whose prompt is `prompt` tokens long and whose
/// prefix cache holds `prefilled` of them.
fn updated(id: u64, prompt: u32, prefilled: u32) -> Result<RequestView, Box<dyn Error>> {
    decode(
        format!(
            r#"{{"id":{id},"state":{{"state":"resident"}},"class":{{"class":"token"}},
                 "phase":1,"prompt_tokens":{prompt},"cached_prefix_tokens":{prefilled}}}"#
        )
        .as_bytes(),
    )
}

/// One queued token request.
fn request(id: u64) -> Result<RequestView, Box<dyn Error>> {
    decode(
        format!(
            r#"{{"id":{id},"state":{{"state":"queued"}},"class":{{"class":"token"}},"phase":1}}"#
        )
        .as_bytes(),
    )
}

/// Decodes a view or message of the interface from JSON.
fn decode<T: DeserializeOwned>(json: &[u8]) -> Result<T, Box<dyn Error>> {
    Ok(wire::decode(json)?)
}

/// What one decision cost the module: the fuel it burned and the calls it made.
///
/// §4.4 gives every call a fuel budget; §12's per-step figure is the difference
/// `fuel_consumed` takes around a step's `observe` and `dispatch`. The harness
/// reports both per call so a policy that is expensive in one decision can be
/// told from one that is expensive in many.
struct Meter {
    fuel: u64,
    cost: scheduling::PolicyCost,
}

impl Meter {
    fn of(policy: &PolicyInstance) -> Self {
        Self {
            fuel: policy.fuel_consumed(),
            cost: policy.cost(),
        }
    }

    /// Prints what the module did since `self`, and answers the fuel it burned.
    fn report(&self, policy: &PolicyInstance, call: &str) -> u64 {
        let fuel = policy.fuel_consumed().saturating_sub(self.fuel);
        let calls: u64 = policy
            .cost()
            .each()
            .map(|(each, after)| after.saturating_sub(self.cost.calls(each)))
            .sum();
        println!("  {call}: fuel {fuel}, calls {calls}");
        fuel
    }
}

/// Prints what a step's plan asks for.
fn print_plan(plan: &Plan) {
    println!(
        "dispatch: admit {} reject {} work {} offload {} prefetch {} refill {}",
        plan.admit.len(),
        plan.reject.len(),
        plan.work.len(),
        plan.offload.len(),
        plan.prefetch.len(),
        plan.refill.len()
    );
    for work in &plan.work {
        println!(
            "  work request {} phase {} tokens {}",
            work.request.0, work.phase.0, work.tokens
        );
    }
}
