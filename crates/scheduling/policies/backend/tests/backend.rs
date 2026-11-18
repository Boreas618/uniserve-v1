//! The `backend` program's module, built for wasm32, stepped in the program host over stated
//! step times, module times and kernel launches.
#![cfg(test)]

use program::{Budget, Config, Function, Instance, Module, Report, State};
use program_testing::{Deployment, Stat, module, revision};
use scheduling::time::{Instant, hours, mins, secs};
use serde_json::{Value, json};

const T: Instant = Instant(1_000_000_000_000_000);

const ATTENTION: &str = "models.qwen3.transformer.Attention";
const MOE: &str = "models.qwen3.transformer.MoE";

/// A deployment whose decode step's median device time was `recent` over the last ten seconds
/// and `usual` over the last hour; whose module classes' medians over the last thirty seconds
/// were `probed`, and which launched the paged attention keys `keys` in them.
fn deployment(recent: f64, usual: f64, probed: &[(&str, f64)], keys: &[&str]) -> Deployment {
    let median = |module: &str, span, value| {
        Stat::new(
            "uniserve_step_device_seconds",
            &[("phase", "decode"), ("module", module)],
            value,
        )
        .window(span)
        .function(Function::Quantile(0.5))
    };
    let mut deployment = Deployment::default()
        .stat(median("step", secs(10), recent))
        .stat(median("step", hours(1), usual));
    for &(module, value) in probed {
        deployment = deployment.stat(median(module, secs(30), value));
    }
    for key in keys {
        deployment = deployment.stat(
            Stat::new(
                "uniserve_kernel_launches_total",
                &[("op", "paged_attention"), ("descriptor", key)],
                100.0,
            )
            .window(secs(30)),
        );
    }
    deployment
}

/// A slowed decode step whose probe measures attention at `attention` seconds per step.
fn slowed(attention: f64) -> Deployment {
    deployment(
        0.012,
        0.010,
        &[("step", 0.012), (ATTENTION, attention), (MOE, 0.006)],
        &["decode_a", "decode_b"],
    )
}

fn serving() -> Value {
    json!({
        "profiling_config": { "fence_probes": true, "fence_probes_off": ["models.*.MoE"] },
        "kernels": {},
    })
}

fn probing(config: &Value) -> Value {
    let mut config = config.clone();
    config["profiling_config"] = json!({
        "fence_modules": ["*"],
        "fence_probes": true,
        "fence_probes_off": [],
    });
    config
}

fn state(time: Instant, config: &Value) -> State {
    State::new("backend", time, Config(config.clone()))
}

fn module() -> Module<Deployment> {
    Module::<Deployment>::control(&module!("policy-backend")).unwrap()
}

fn program(deployment: Deployment) -> Instance<Deployment> {
    Instance::new(&module(), deployment, &Budget::CONTROL, &json!({})).unwrap()
}

fn step(program: &mut Instance<Deployment>, time: Instant, config: &Value) -> Option<Value> {
    let output: Option<Config> = program.step(&state(time, config)).unwrap();
    output.map(|config| config.0)
}

/// A slowed decode step opens a probe of every module, which runs for thirty seconds from the
/// step that reads it in the configuration and then writes the probe's options back. The first
/// probe records the attention time on the backend the Build Server selects; a second, after the
/// interval, finds attention slowed and moves the keys to the other backend, whose own probe
/// later measures it slower than the record, so the keys move back.
#[test]
fn a_slowed_step_is_probed_and_the_attention_kernels_move_to_the_faster_backend() {
    let config = serving();
    let mut program = program(slowed(0.004));

    let opened = step(&mut program, T, &config).expect("a probe");
    assert_eq!(opened, probing(&config));
    assert_eq!(step(&mut program, T + secs(10), &opened), None);
    assert_eq!(step(&mut program, T + secs(30), &opened), None);

    // The probe ends: its options as they were, with no move on the first reading.
    let closed = step(&mut program, T + secs(40), &opened).expect("the probe's end");
    assert_eq!(closed["profiling_config"]["fence_modules"], Value::Null);
    assert_eq!(closed["profiling_config"]["fence_probes"], json!(true));
    assert_eq!(
        closed["profiling_config"]["fence_probes_off"],
        json!(["models.*.MoE"])
    );
    assert_eq!(closed["kernels"], json!({}));

    // No probe again within the interval.
    assert_eq!(step(&mut program, T + mins(5), &config), None);

    // The next probe finds attention more than 10% slower than its record: with no record of
    // Triton on these keys, they move to it.
    *program.host_mut() = slowed(0.005);
    let opened = step(&mut program, T + mins(11), &config).expect("a second probe");
    assert_eq!(step(&mut program, T + mins(11) + secs(10), &opened), None);
    let moved = step(&mut program, T + mins(11) + secs(40), &opened).expect("a move");
    assert_eq!(
        moved["kernels"],
        json!({ "paged_attention": { "decode_a": "triton@latest", "decode_b": "triton@latest" } })
    );
    assert_eq!(moved["profiling_config"]["fence_modules"], Value::Null);

    // On Triton attention measures slower than TRT-LLM's record: the keys move back, at
    // `latest`, the table pinning no TRT-LLM revision.
    *program.host_mut() = slowed(0.006);
    let opened = step(&mut program, T + mins(22), &moved).expect("a third probe");
    assert_eq!(step(&mut program, T + mins(22) + secs(10), &opened), None);
    let back = step(&mut program, T + mins(22) + secs(40), &opened).expect("a move back");
    assert_eq!(
        back["kernels"],
        json!({ "paged_attention": { "decode_a": "trtllm@latest", "decode_b": "trtllm@latest" } })
    );
}

/// A step within its baseline opens no probe and validates nothing.
#[test]
fn a_step_within_its_baseline_is_not_probed() {
    let steady = deployment(0.0105, 0.010, &[], &[]);
    let mut program = program(steady.clone());
    assert_eq!(step(&mut program, T, &serving()), None);
    assert!(steady.seen().validated.is_empty());
}

/// A probe the deployment did not take ends unread and the interval still holds; a rolled-back
/// change of the program's own holds back the next probe, and an open probe still closes then,
/// without a move.
#[test]
fn a_probe_not_taken_ends_and_a_rollback_pauses_the_program() {
    let config = serving();
    let mut program = program(slowed(0.004));
    assert!(step(&mut program, T, &config).is_some());
    assert_eq!(step(&mut program, T + secs(10), &config), None);
    assert_eq!(step(&mut program, T + secs(20), &config), None);

    let paused = slowed(0.004).record(revision("backend", "rolled_back", T + mins(5)));
    *program.host_mut() = paused.clone();
    assert_eq!(step(&mut program, T + mins(11), &config), None);
    assert!(paused.seen().validated.is_empty());

    // A probe opened before the rollback closes, and moves nothing.
    let mut program = self::program(slowed(0.004));
    let opened = step(&mut program, T, &config).unwrap();
    assert_eq!(step(&mut program, T + secs(10), &opened), None);
    *program.host_mut() = slowed(0.005);
    let first = step(&mut program, T + secs(40), &opened).unwrap();
    assert_eq!(first["kernels"], json!({}));
    let opened = step(&mut program, T + mins(11), &config).unwrap();
    assert_eq!(step(&mut program, T + mins(11) + secs(10), &opened), None);
    *program.host_mut() =
        slowed(0.005).record(revision("backend", "rolled_back", T + mins(11) + secs(20)));
    let closed = step(&mut program, T + mins(11) + secs(40), &opened).unwrap();
    assert_eq!(closed["kernels"], json!({}));
    assert_eq!(closed["profiling_config"]["fence_modules"], Value::Null);
}

/// A probe validation reports as stopping serving is not opened; an open probe survives a
/// restart with the program's state and closes after it.
#[test]
fn an_interrupting_probe_is_not_opened_and_an_open_one_survives_a_restart() {
    let config = serving();
    let interrupting = slowed(0.004).report(Ok(Report {
        effect: Some("epoch".into()),
        interrupts: true,
        rooms: Vec::new(),
    }));
    let mut program = program(interrupting);
    assert_eq!(step(&mut program, T, &config), None);

    let mut program = self::program(slowed(0.004));
    let opened = step(&mut program, T, &config).unwrap();
    assert_eq!(step(&mut program, T + secs(10), &opened), None);
    let saved = program.save().unwrap();

    let mut restored =
        Instance::restore(&module(), slowed(0.004), &Budget::CONTROL, &saved).unwrap();
    assert_eq!(step(&mut restored, T + secs(30), &opened), None);
    let closed = step(&mut restored, T + secs(40), &opened).unwrap();
    assert_eq!(
        closed["profiling_config"]["fence_probes_off"],
        json!(["models.*.MoE"])
    );
}
