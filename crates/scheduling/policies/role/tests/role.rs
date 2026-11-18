//! The `role` program's module, built for wasm32, stepped in the program host over stated p99
//! latencies and running requests.

#![cfg(test)]

use program::{Budget, Config, Function, Instance, Module, Report, State};
use program_testing::{Deployment, Stat, module, revision};
use scheduling::time::{Instant, hours, mins};
use serde_json::{Value, json};

const T: Instant = Instant(1_000_000_000_000_000);

fn args() -> Value {
    json!({ "ttft": "1s", "tpot": "50ms" })
}

/// A deployment whose p99 TTFT and TPOT over the last minute were `ttft` and `tpot` seconds,
/// and whose replicas ran `running` requests.
fn deployment(ttft: f64, tpot: f64, running: &[(&str, f64)]) -> Deployment {
    let p99 = |name: &str, value| {
        Stat::new(name, &[], value)
            .window(mins(1))
            .function(Function::Quantile(0.99))
    };
    let mut deployment = Deployment::default()
        .stat(p99("uniserve_ttft_seconds", ttft))
        .stat(p99("uniserve_tpot_seconds", tpot));
    for &(replica, value) in running {
        deployment = deployment.stat(
            Stat::new(
                "uniserve_scheduler_running_requests",
                &[("replica", replica)],
                value,
            )
            .window(mins(1))
            .function(Function::Last),
        );
    }
    deployment
}

/// Two prefill replicas and three decode replicas.
fn replicas() -> Value {
    json!({
        "replicas": {
            "p0": { "role": "prefill" },
            "p1": { "role": "prefill" },
            "d0": { "role": "decode" },
            "d1": { "role": "decode" },
            "d2": { "role": "decode" },
            "m0": { "role": "mixed" },
        }
    })
}

fn program(deployment: &Deployment) -> Instance<Deployment> {
    let module = Module::<Deployment>::control(&module!("policy-role")).unwrap();
    Instance::new(&module, deployment.clone(), &Budget::CONTROL, &args()).unwrap()
}

fn step(program: &mut Instance<Deployment>, time: Instant, config: &Value) -> Option<Value> {
    let output: Option<Config> = program
        .step(&State::new("role", time, Config(config.clone())))
        .unwrap();
    output.map(|config| config.0)
}

fn roles(config: &Value) -> Vec<(String, String)> {
    config["replicas"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(id, replica)| (id.clone(), replica["role"].as_str().unwrap().to_owned()))
        .collect()
}

/// TTFT over its target while TPOT has room turns the least loaded decode replica into a
/// prefill replica; TPOT over its target while TTFT has room turns the least loaded prefill
/// replica into a decode replica, once the interval after the first change passed.
#[test]
fn the_short_side_takes_a_replica_from_the_side_with_room() {
    let running = [
        ("d0", 9.0),
        ("d1", 2.0),
        ("d2", 5.0),
        ("p0", 4.0),
        ("p1", 1.0),
    ];
    let mut program = program(&deployment(1.5, 0.020, &running));

    let changed = step(&mut program, T, &replicas()).expect("a decode replica to prefill");
    assert_eq!(changed["replicas"]["d1"]["role"], json!("prefill"));
    assert_eq!(
        roles(&changed)
            .iter()
            .filter(|(_, role)| role == "decode")
            .count(),
        2
    );
    assert_eq!(changed["replicas"]["m0"]["role"], json!("mixed"));

    // Within the interval, nothing changes, whatever the latencies.
    *program.host_mut() = deployment(0.3, 0.080, &running);
    assert_eq!(step(&mut program, T + mins(4), &changed), None);

    let back = step(&mut program, T + mins(5), &changed).expect("a prefill replica to decode");
    assert_eq!(back["replicas"]["p1"]["role"], json!("decode"));
}

/// Both sides within their targets, both over them, a side without room, or a nothing-finished
/// window change nothing and validate nothing.
#[test]
fn no_change_without_a_short_side_and_a_side_with_room() {
    for (ttft, tpot) in [(0.5, 0.030), (1.5, 0.080), (1.5, 0.045), (f64::NAN, 0.020)] {
        let deployment = deployment(ttft, tpot, &[]);
        let mut program = program(&deployment);
        assert_eq!(step(&mut program, T, &replicas()), None, "{ttft} {tpot}");
        assert!(deployment.seen().validated.is_empty());
    }
}

/// A role at `keep` replicas gives none, and a replica that reports no running requests is
/// idle.
#[test]
fn a_role_keeps_its_last_replica() {
    let config = json!({
        "replicas": { "p0": { "role": "prefill" }, "d0": { "role": "decode" } }
    });
    let mut program = program(&deployment(1.5, 0.020, &[]));
    assert_eq!(step(&mut program, T, &config), None);

    let config = json!({
        "replicas": {
            "p0": { "role": "prefill" },
            "d0": { "role": "decode" },
            "d1": { "role": "decode" },
        }
    });
    *program.host_mut() = deployment(1.5, 0.020, &[("d0", 3.0)]);
    let changed = step(&mut program, T, &config).unwrap();
    assert_eq!(changed["replicas"]["d1"]["role"], json!("prefill"));
}

/// A rolled-back change of the program's own holds it back for an hour; a change validation
/// reports as stopping serving is not output; the time of its last change survives a restart.
#[test]
fn a_rollback_pauses_the_program_and_its_last_change_is_kept() {
    let short = || deployment(1.5, 0.020, &[]);

    let mut program = program(&short().record(revision("role", "rolled_back", T - mins(20))));
    assert_eq!(step(&mut program, T, &replicas()), None);
    *program.host_mut() = short().record(revision("role", "rolled_back", T - hours(2)));
    assert!(step(&mut program, T, &replicas()).is_some());

    let mut interrupted = self::program(&short().report(Ok(Report {
        effect: Some("epoch".into()),
        interrupts: true,
        rooms: Vec::new(),
    })));
    assert_eq!(step(&mut interrupted, T, &replicas()), None);

    let saved = program.save().unwrap();
    let module = Module::<Deployment>::control(&module!("policy-role")).unwrap();
    let mut restored = Instance::restore(&module, short(), &Budget::CONTROL, &saved).unwrap();
    assert_eq!(step(&mut restored, T + mins(1), &replicas()), None);
    assert!(step(&mut restored, T + mins(5), &replicas()).is_some());
}
