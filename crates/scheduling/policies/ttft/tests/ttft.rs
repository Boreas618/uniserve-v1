//! The `ttft` program's module, built for wasm32, routing requests in the program host within
//! the leader's routing budget.

#![cfg(test)]

use program::{Budget, Error, Instance, Level, Module};
use program_testing::{Deployment, module};
use scheduling::kv::{Key, Tier};
use scheduling::routing::{Hit, Replica, Request, Role, Route, Routing};
use scheduling::time::{Duration, millis};
use serde_json::json;

fn args() -> serde_json::Value {
    json!({ "rate": 10000.0, "page": 64 })
}

fn router(deployment: &Deployment) -> Instance<Deployment> {
    let module = Module::<Deployment>::new(&module!("policy-ttft")).unwrap();
    Instance::new(&module, deployment.clone(), &Budget::ROUTING, &args()).unwrap()
}

/// A replica of `role` with `queued` prefill tokens waiting, its batch done in `remaining`,
/// `running` requests, `reservable` pages and `held` tokens of the prompt on the device.
fn replica(
    role: Role,
    queued: u64,
    remaining: Duration,
    running: u32,
    reservable: u64,
    held: u32,
) -> Replica {
    Replica {
        id: format!("{role:?}-{queued}-{running}"),
        role,
        queued,
        remaining,
        running,
        usage: Vec::new(),
        reservable,
        hits: vec![Hit {
            tier: Tier::Device,
            tokens: held,
        }],
    }
}

/// A request of a 4096-token prompt over `replicas`.
fn routing(replicas: Vec<Replica>) -> Routing {
    Routing {
        request: Request {
            prompt: 4096,
            prefix: vec![Key([7; 32])],
            session: None,
        },
        replicas,
    }
}

fn route(router: &mut Instance<Deployment>, replicas: Vec<Replica>) -> Route {
    router.step(&routing(replicas)).unwrap()
}

/// The prefill replica is the one predicted to give the first token soonest: its running batch,
/// then its queue and the prompt's tokens it does not hold at the prefill rate. A replica with a
/// longer queue that holds most of the prompt wins over an idle one that holds none.
#[test]
fn the_prefill_replica_is_the_one_predicted_first() {
    let mut router = router(&Deployment::default());

    // Replica 0: 100 ms + 4096 tokens = 0.51 s; replica 1: 20 ms + (512 + 1024) tokens = 0.17 s;
    // replica 2: 400 ms + 0 + 1024 tokens = 0.50 s.
    let prefill = vec![
        replica(Role::Prefill, 0, millis(100), 0, 0, 0),
        replica(Role::Prefill, 512, millis(20), 3, 0, 3072),
        replica(Role::Prefill, 0, millis(400), 0, 0, 3072),
        replica(Role::Decode, 0, millis(5), 4, 1000, 0),
    ];
    assert_eq!(route(&mut router, prefill).prefill, 1);

    // Equal predictions: the replica with fewer running requests, then the first.
    let tied = vec![
        replica(Role::Mixed, 0, millis(10), 5, 1000, 0),
        replica(Role::Mixed, 0, millis(10), 2, 1000, 0),
        replica(Role::Mixed, 0, millis(10), 2, 1000, 0),
    ];
    assert_eq!(
        route(&mut router, tied),
        Route {
            prefill: 1,
            decode: 1
        }
    );
}

/// A mixed replica decodes what it computed; otherwise the decode replica is the one with the
/// fewest running requests that can reserve the prompt's 64 pages, or with none that can, the
/// one with the fewest running requests.
#[test]
fn the_decode_replica_can_reserve_the_prompts_pages() {
    let mut router = router(&Deployment::default());

    let replicas = vec![
        replica(Role::Prefill, 0, millis(10), 0, 0, 0),
        replica(Role::Decode, 0, millis(5), 1, 63, 0),
        replica(Role::Decode, 0, millis(5), 6, 64, 0),
        replica(Role::Decode, 0, millis(5), 3, 500, 0),
    ];
    assert_eq!(
        route(&mut router, replicas),
        Route {
            prefill: 0,
            decode: 3
        }
    );

    let full = vec![
        replica(Role::Decode, 0, millis(5), 4, 0, 0),
        replica(Role::Prefill, 0, millis(10), 0, 0, 0),
        replica(Role::Decode, 0, millis(5), 2, 10, 0),
    ];
    assert_eq!(
        route(&mut router, full),
        Route {
            prefill: 1,
            decode: 2
        }
    );

    let mixed = vec![
        replica(Role::Decode, 0, millis(1), 0, 1000, 0),
        replica(Role::Mixed, 0, millis(10), 9, 0, 0),
    ];
    assert_eq!(
        route(&mut router, mixed),
        Route {
            prefill: 1,
            decode: 1
        }
    );
}

/// A deployment without a replica that computes prompts, or without one that decodes, gets a
/// route the leader refuses, and the program says why and goes on routing.
#[test]
fn a_missing_role_is_logged_and_the_program_goes_on() {
    let deployment = Deployment::default();
    let mut router = router(&deployment);

    let decoders = vec![replica(Role::Decode, 0, millis(5), 0, 1000, 0)];
    assert_eq!(
        route(&mut router, decoders),
        Route {
            prefill: 0,
            decode: 0
        }
    );
    let prefillers = vec![
        replica(Role::Prefill, 0, millis(50), 0, 0, 0),
        replica(Role::Prefill, 0, millis(5), 0, 0, 0),
    ];
    assert_eq!(
        route(&mut router, prefillers),
        Route {
            prefill: 1,
            decode: 1
        }
    );
    assert_eq!(
        deployment.seen().logs,
        [
            (Level::Warn, "no replica computes prompts".to_owned()),
            (Level::Warn, "no replica decodes".to_owned()),
        ]
    );
}

/// Routing over a large deployment fits the leader's routing budget; the program's state, its
/// arguments, survives a restart, and arguments that are not its `Args` are refused.
#[test]
fn routing_fits_the_budget_and_the_state_is_restored() {
    let module = Module::<Deployment>::new(&module!("policy-ttft")).unwrap();
    let mut router = router(&Deployment::default());
    let replicas: Vec<Replica> = (0..256)
        .map(|index| {
            let role = if index % 2 == 0 {
                Role::Prefill
            } else {
                Role::Decode
            };
            replica(role, 4096 - index, millis(10), 1, 1000, 0)
        })
        .collect();
    assert_eq!(
        route(&mut router, replicas),
        Route {
            prefill: 254,
            decode: 1
        }
    );

    let saved = router.save().unwrap();
    let mut restored =
        Instance::restore(&module, Deployment::default(), &Budget::ROUTING, &saved).unwrap();
    let mixed = vec![replica(Role::Mixed, 0, millis(1), 0, 0, 0)];
    assert_eq!(
        route(&mut restored, mixed),
        Route {
            prefill: 0,
            decode: 0
        }
    );

    let error = Instance::new(
        &module,
        Deployment::default(),
        &Budget::ROUTING,
        &json!({ "rate": 10000.0 }),
    )
    .err();
    assert!(matches!(error, Some(Error::Args(_))), "{error:?}");
}
