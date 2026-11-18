//! The `chunk` program's module, built for wasm32, stepped in the program host over stated
//! per-replica p99s.
#![cfg(test)]

use program::{Budget, Config, Error, Function, Instance, Level, Module, Report, State};
use program_testing::{Deployment, Stat, module, revision};
use scheduling::time::{Instant, hours, mins, secs};
use serde_json::json;

const NOW: Instant = Instant(1_000_000_000_000_000);

fn args() -> serde_json::Value {
    json!({ "tpot": "50ms", "ttft": "1s", "low": 512, "high": 8192 })
}

/// A deployment whose replicas finished requests at these p99 TPOTs and TTFTs, in seconds, over
/// the last ten seconds.
fn deployment(replicas: &[(f64, f64)]) -> Deployment {
    let mut deployment = Deployment::default();
    for (replica, &(tpot, ttft)) in replicas.iter().enumerate() {
        let replica = replica.to_string();
        for (name, value) in [
            ("uniserve_tpot_seconds", tpot),
            ("uniserve_ttft_seconds", ttft),
        ] {
            deployment = deployment.stat(
                Stat::new(name, &[("replica", &replica)], value)
                    .window(secs(10))
                    .function(Function::Quantile(0.99)),
            );
        }
    }
    deployment
}

fn chunk(deployment: &Deployment) -> Instance<Deployment> {
    let module = Module::<Deployment>::control(&module!("policy-chunk")).unwrap();
    Instance::new(&module, deployment.clone(), &Budget::CONTROL, &args()).unwrap()
}

fn state(config: serde_json::Value) -> State {
    State::new("chunk", NOW, Config(config))
}

fn with_chunk(chunk: u32) -> State {
    state(json!({ "scheduler_config": { "chunked_prefill_size": chunk } }))
}

fn chunk_of(output: Option<Config>) -> Option<u32> {
    output?.get("scheduler_config.chunked_prefill_size")
}

/// One replica's TPOT over its target halves the chunk, whatever the others' TTFT; TTFT over
/// its target while every TPOT has room doubles it, up to `high`; both within their targets
/// leave the configuration as it is, without validating.
#[test]
fn the_chunk_follows_the_worst_replica() {
    let tpot_over = deployment(&[(0.020, 0.3), (0.080, 2.0)]);
    let mut program = chunk(&tpot_over);
    assert_eq!(
        chunk_of(program.step(&with_chunk(2048)).unwrap()),
        Some(1024)
    );
    assert_eq!(tpot_over.seen().validated.len(), 1);

    *program.host_mut() = deployment(&[(0.020, 0.3), (0.030, 1.5)]);
    assert_eq!(
        chunk_of(program.step(&with_chunk(2048)).unwrap()),
        Some(4096)
    );
    assert_eq!(
        chunk_of(program.step(&with_chunk(6000)).unwrap()),
        Some(8192)
    );

    let within = deployment(&[(0.020, 0.3), (0.030, 0.5)]);
    *program.host_mut() = within.clone();
    assert_eq!(
        program
            .step::<_, Option<Config>>(&with_chunk(2048))
            .unwrap(),
        None
    );
    assert!(within.seen().validated.is_empty());
}

/// A deployment with no finished request in the window, and a chunk at the bound the change
/// would cross, change nothing.
#[test]
fn nothing_finished_or_a_chunk_at_its_bound_changes_nothing() {
    let mut program = chunk(&Deployment::default());
    assert_eq!(
        program
            .step::<_, Option<Config>>(&with_chunk(2048))
            .unwrap(),
        None
    );

    *program.host_mut() = deployment(&[(0.080, 0.3)]);
    assert_eq!(
        program.step::<_, Option<Config>>(&with_chunk(512)).unwrap(),
        None
    );
    // A chunk below `low` is not raised by halving.
    assert_eq!(
        program.step::<_, Option<Config>>(&with_chunk(256)).unwrap(),
        None
    );

    *program.host_mut() = deployment(&[(0.020, 3.0)]);
    assert_eq!(
        program
            .step::<_, Option<Config>>(&with_chunk(8192))
            .unwrap(),
        None
    );
}

/// An unset chunk is the step's whole token budget: the stated `max_num_batched_tokens`, and
/// `high` when no budget is stated either.
#[test]
fn an_unset_chunk_is_the_token_budget() {
    let mut program = chunk(&deployment(&[(0.080, 0.3)]));
    let budget = state(json!({ "max_num_batched_tokens": 4096, "scheduler_config": {} }));
    assert_eq!(chunk_of(program.step(&budget).unwrap()), Some(2048));

    let unstated = state(json!({ "scheduler_config": {} }));
    assert_eq!(chunk_of(program.step(&unstated).unwrap()), Some(4096));
}

/// A change of the program's own rolled back holds it back for an hour; a change another writer
/// made, or one of its own that converged, does not.
#[test]
fn a_rolled_back_change_pauses_the_program() {
    let tpot_over = deployment(&[(0.080, 0.3)]);

    let paused = tpot_over
        .clone()
        .record(revision("chunk", "rolled_back", NOW - mins(10)));
    let mut program = chunk(&paused);
    assert_eq!(
        program
            .step::<_, Option<Config>>(&with_chunk(2048))
            .unwrap(),
        None
    );
    assert!(paused.seen().validated.is_empty());

    *program.host_mut() =
        tpot_over
            .clone()
            .record(revision("chunk", "rolled_back", NOW - hours(2)));
    assert_eq!(
        chunk_of(program.step(&with_chunk(2048)).unwrap()),
        Some(1024)
    );

    *program.host_mut() = tpot_over
        .clone()
        .record(revision("chunk", "rolled_back", NOW - mins(10)))
        .record(revision("chunk", "converged", NOW - mins(1)));
    assert_eq!(
        chunk_of(program.step(&with_chunk(2048)).unwrap()),
        Some(1024)
    );

    *program.host_mut() = tpot_over.record(revision("operator", "rolled_back", NOW - mins(1)));
    assert_eq!(
        chunk_of(program.step(&with_chunk(2048)).unwrap()),
        Some(1024)
    );
}

/// A change validation refuses, or reports as stopping serving, is not output; a refusal is
/// logged with its reason.
#[test]
fn a_change_is_output_only_when_it_is_valid_and_keeps_serving() {
    let interrupting = deployment(&[(0.080, 0.3)]).report(Ok(Report {
        effect: Some("epoch".into()),
        interrupts: true,
        rooms: Vec::new(),
    }));
    let mut program = chunk(&interrupting);
    assert_eq!(
        program
            .step::<_, Option<Config>>(&with_chunk(2048))
            .unwrap(),
        None
    );
    assert_eq!(interrupting.seen().validated.len(), 1);

    let refused = deployment(&[(0.020, 3.0)]).report(Err("above max_num_batched_tokens".into()));
    *program.host_mut() = refused.clone();
    assert_eq!(
        program
            .step::<_, Option<Config>>(&with_chunk(4096))
            .unwrap(),
        None
    );
    assert_eq!(
        refused.seen().logs.last(),
        Some(&(
            Level::Warn,
            "the change is refused: above max_num_batched_tokens".to_owned()
        ))
    );
}

/// The program's state, its arguments with their defaults, survives a restart; arguments that
/// are not its `Args` are refused.
#[test]
fn the_state_is_restored_and_wrong_arguments_are_refused() {
    let module = Module::<Deployment>::control(&module!("policy-chunk")).unwrap();
    let mut program = chunk(&deployment(&[(0.080, 0.3)]));
    let saved = program.save().unwrap();
    let state: serde_json::Value = serde_json::from_slice(&saved).unwrap();
    assert_eq!(state["args"]["window"], json!(10_000_000_000u64));
    assert_eq!(state["args"]["pause"], json!(3_600_000_000_000u64));

    let mut restored = Instance::restore(
        &module,
        deployment(&[(0.080, 0.3)]),
        &Budget::CONTROL,
        &saved,
    )
    .unwrap();
    assert_eq!(
        chunk_of(restored.step(&with_chunk(2048)).unwrap()),
        Some(1024)
    );

    let error = Instance::new(
        &module,
        Deployment::default(),
        &Budget::CONTROL,
        &json!({ "tpot": "50ms", "ttft": "1s", "low": 512, "high": 8192, "cooldown": "1m" }),
    )
    .err();
    assert!(matches!(error, Some(Error::Args(_))), "{error:?}");
}
