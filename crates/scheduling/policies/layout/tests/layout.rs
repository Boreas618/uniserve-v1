//! The `layout` program's module, built for wasm32, stepped in the program host over stated
//! running requests and throughput.
#![cfg(test)]

use program::{Budget, Config, Function, Instance, Module, Report, State};
use program_testing::{Deployment, Stat, module, revision};
use scheduling::time::{Instant, mins};
use serde_json::{Value, json};

const T: Instant = Instant(1_000_000_000_000_000);

fn tp8() -> Value {
    json!([{ "id": "model", "ranks": ["cuda:0-7"], "blocks": { "model": { "tensor_parallel_size": 8 } } }])
}

fn tp4x2() -> Value {
    json!([
        { "id": "a", "ranks": ["cuda:0-3"], "blocks": { "model": { "tensor_parallel_size": 4 } } },
        { "id": "b", "ranks": ["cuda:4-7"], "blocks": { "model": { "tensor_parallel_size": 4 } } },
    ])
}

fn args() -> Value {
    json!({
        "layouts": { "tp8": { "worker_groups": tp8() }, "tp4x2": { "worker_groups": tp4x2() } },
        "bounds": [32],
    })
}

/// The configuration as the leader states it: every option at its full value, so the layout's
/// groups carry more than the layout states.
fn serving(groups: Value) -> Value {
    let mut groups = groups;
    for group in groups.as_array_mut().unwrap() {
        group["queue_depth"] = json!(2);
        group["role"] = json!("serve");
    }
    json!({ "model": "Qwen/Qwen3-30B-A3B", "worker_groups": groups })
}

fn state(time: Instant, config: Value) -> State {
    State::new("layout", time, Config(config))
}

/// A deployment whose running requests averaged `running` over the last fifteen minutes and
/// that generated `rate` tokens per second over the last minute.
fn deployment(running: f64, rate: f64) -> Deployment {
    Deployment::default()
        .stat(
            Stat::new("uniserve_scheduler_running_requests", &[], running)
                .window(mins(15))
                .function(Function::Mean),
        )
        .stat(
            Stat::new("uniserve_generation_tokens_total", &[], rate)
                .window(mins(1))
                .function(Function::Rate),
        )
}

fn module() -> Module<Deployment> {
    Module::<Deployment>::control(&module!("policy-layout")).unwrap()
}

/// Steps `program` at each of `minutes` after `T` over `deployment`, asserting it outputs
/// nothing.
fn quiet(
    program: &mut Instance<Deployment>,
    deployment: Deployment,
    config: &Value,
    minutes: &[u64],
) {
    *program.host_mut() = deployment;
    for &minute in minutes {
        let output: Option<Config> = program
            .step(&state(T + mins(minute), config.clone()))
            .unwrap();
        assert_eq!(output, None, "minute {minute}");
    }
}

/// The program records each layout's throughput per range of running requests once the layout
/// served a whole window, and switches to the layout whose record in the current range is
/// higher: the narrow replicas measured more throughput at high concurrency, and TP8 more at low
/// concurrency, which the program returns to when the requests fall.
#[test]
fn the_layout_follows_the_throughput_measured_at_the_concurrency() {
    let mut program =
        Instance::new(&module(), Deployment::default(), &Budget::CONTROL, &args()).unwrap();
    let wide = serving(tp8());
    let narrow = serving(tp4x2());

    // TP8 serves from T: nothing counts in its first fifteen minutes, then its low and high
    // concurrency ranges are measured.
    quiet(
        &mut program,
        deployment(8.0, 900.0),
        &wide,
        &[0, 14, 15, 16, 17],
    );
    quiet(&mut program, deployment(64.0, 1000.0), &wide, &[18, 19, 20]);

    // The operator moves to two TP4 replicas at minute 21; they measure more at high
    // concurrency, so the program keeps them.
    quiet(
        &mut program,
        deployment(64.0, 1500.0),
        &narrow,
        &[21, 36, 37, 38],
    );

    // The requests fall: at low concurrency the narrow replicas measure less than TP8 did.
    quiet(&mut program, deployment(8.0, 600.0), &narrow, &[39, 40]);
    let output: Option<Config> = program.step(&state(T + mins(41), narrow.clone())).unwrap();
    let output = output.expect("a switch back to TP8");
    assert_eq!(output.get::<Value>("worker_groups"), Some(tp8()));
    assert_eq!(
        output.get::<String>("model").as_deref(),
        Some("Qwen/Qwen3-30B-A3B")
    );

    // The records survive a restart with the program's state.
    let saved: Value = serde_json::from_slice(&program.save().unwrap()).unwrap();
    assert_eq!(saved["throughput"]["tp8"]["0"]["samples"], json!(3));
    assert_eq!(saved["throughput"]["tp4x2"]["1"]["mean"], json!(1500.0));
}

/// A program restored with records whose other layout is higher in the current range, which
/// switches; unless its last switch was rolled back within the hour, the switch would stop
/// serving, or the other layout's record is within the margin.
#[test]
fn a_restored_record_switches_unless_paused_or_within_the_margin() {
    let restored = |tp8: f64| {
        let state = json!({
            "args": args(),
            "throughput": {
                "tp8": { "0": { "mean": tp8, "samples": 3 } },
                "tp4x2": { "0": { "mean": 600.0, "samples": 3 } },
            },
            "serving": { "layout": "tp4x2", "since": T.0 },
        });
        let bytes = serde_json::to_vec(&state).unwrap();
        Instance::restore(&module(), deployment(8.0, 600.0), &Budget::CONTROL, &bytes).unwrap()
    };
    let narrow = serving(tp4x2());
    let at = |minutes| state(T + mins(minutes), narrow.clone());

    let mut program = restored(900.0);
    let output: Option<Config> = program.step(&at(20)).unwrap();
    assert_eq!(
        output.and_then(|config| config.get::<Value>("worker_groups")),
        Some(tp8())
    );

    let mut program = restored(900.0);
    *program.host_mut() =
        deployment(8.0, 600.0).record(revision("layout", "rolled_back", T + mins(10)));
    assert_eq!(program.step::<_, Option<Config>>(&at(20)).unwrap(), None);

    let mut program = restored(900.0);
    *program.host_mut() = deployment(8.0, 600.0).report(Ok(Report {
        effect: Some("epoch".into()),
        interrupts: true,
        rooms: Vec::new(),
    }));
    assert_eq!(program.step::<_, Option<Config>>(&at(20)).unwrap(), None);

    let mut program = restored(625.0);
    assert_eq!(program.step::<_, Option<Config>>(&at(20)).unwrap(), None);
}

/// A configuration that is none of the named layouts is the operator's: the program records
/// nothing for it and writes nothing.
#[test]
fn a_layout_of_no_name_is_left_alone() {
    let deployment = deployment(64.0, 1000.0);
    let mut program =
        Instance::new(&module(), deployment.clone(), &Budget::CONTROL, &args()).unwrap();
    let other = serving(
        json!([{ "id": "model", "ranks": ["cuda:0-1"], "blocks": { "model": { "tensor_parallel_size": 2 } } }]),
    );
    for minute in [0, 15, 16, 17, 18] {
        let output: Option<Config> = program
            .step(&state(T + mins(minute), other.clone()))
            .unwrap();
        assert_eq!(output, None);
    }
    let saved: Value = serde_json::from_slice(&program.save().unwrap()).unwrap();
    assert_eq!(saved["throughput"], json!({}));
    assert!(deployment.seen().validated.is_empty());
}
