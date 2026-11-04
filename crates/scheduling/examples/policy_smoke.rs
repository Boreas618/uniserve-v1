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

use std::error::Error;
use std::process::ExitCode;

use scheduling::guest::wire;
use scheduling::guest::{
    CallView, Duration, Event, ExpertLoad, Limits, Parameters, Plan, RankView, ReplicaView,
    RequestId, RequestView, Room, Shortage, Snapshot, Storage,
};
use scheduling::Engine;
use serde::de::DeserializeOwned;

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(path) = arguments.next() else {
        eprintln!("usage: policy_smoke <module.wasm> [--precompile]");
        return ExitCode::from(2);
    };
    let precompile = arguments.next().as_deref() == Some("--precompile");

    match run(&path, precompile) {
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
fn run(path: &str, precompile: bool) -> Result<(), Box<dyn Error>> {
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
    policy.observe(&[
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
    ])?;
    println!("observe: 6 events");

    let room = Room {
        step: 1,
        kv_units: 4096,
        slots: 8,
        token_budget: limits.max_num_batched_tokens,
        call_budget: limits.max_batch,
        page_tokens: 64,
        ..Room::default()
    };
    let plan = policy.dispatch(&room)?;
    print_plan(&plan);

    let victims = policy.reclaim(&Shortage {
        kind: Storage::KvUnits,
        need: 1,
    })?;
    println!("reclaim: {} victims", victims.len());

    let ranks: Vec<RankView> = decode(br#"[{"index":0,"in_flight":1},{"index":1,"in_flight":0}]"#)?;
    println!("route: rank {}", policy.route(&request(1)?, &ranks)?);

    let call: CallView = decode(br#"{"request":1,"phase":1,"kind":{"kind":"prefill"}}"#)?;
    let replicas: Vec<ReplicaView> = decode(
        br#"[{"index":0,"block":0,"kinds":[{"kind":"prefill"},{"kind":"decode"}],
              "resident_requests":1,"capacity":7}]"#,
    )?;
    println!("place: replica {}", policy.place(&call, &replicas)?);

    println!("warm: {} pages", policy.warm(&room)?.len());

    let load = ExpertLoad {
        ranks: 2,
        experts: 4,
        tokens: vec![10, 0, 4, 4],
    };
    println!(
        "experts: {} placements",
        policy.experts(&load)?.placement.len()
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
