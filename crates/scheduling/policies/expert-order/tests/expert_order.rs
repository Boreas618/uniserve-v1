//! The `expert-order` program's module, built for wasm32, stepped in the program host over
//! stated per-expert token counts.
#![cfg(test)]

use std::collections::BTreeMap;

use program::{Budget, Config, Function, Instance, Level, Module, Report, State};
use program_testing::{Deployment, Stat, module, revision};
use scheduling::time::{Instant, hours, mins};
use serde_json::json;

const NOW: Instant = Instant(1_000_000_000_000_000);

const SKEWED: &str = "model.layers.0.mlp.experts";
const EVEN: &str = "model.layers.1.mlp.experts";

/// A deployment whose layers' experts took these tokens over the last ten minutes, by logical
/// expert id.
fn deployment(layers: &[(&str, &[f64])]) -> Deployment {
    let mut deployment = Deployment::default();
    for (layer, tokens) in layers {
        deployment = deployment.stat(
            Stat::new("uniserve_expert_tokens_total", &[("layer", layer)], 1.0).window(mins(10)),
        );
        for (expert, &count) in tokens.iter().enumerate() {
            let expert = expert.to_string();
            deployment = deployment.stat(
                Stat::new(
                    "uniserve_expert_tokens_total",
                    &[("layer", layer), ("expert", &expert)],
                    count,
                )
                .window(mins(10))
                .function(Function::Sum),
            );
        }
    }
    deployment
}

/// Rank 0's slots hold the four heaviest experts of the skewed layer; the even layer's ranks
/// take the same tokens.
fn skewed() -> Deployment {
    deployment(&[
        (SKEWED, &[40.0, 30.0, 20.0, 10.0, 4.0, 3.0, 2.0, 1.0]),
        (EVEN, &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
    ])
}

fn program(deployment: &Deployment) -> Instance<Deployment> {
    let module = Module::<Deployment>::control(&module!("policy-expert-order")).unwrap();
    Instance::new(
        &module,
        deployment.clone(),
        &Budget::CONTROL,
        &json!({ "ranks": 2 }),
    )
    .unwrap()
}

fn state(orders: serde_json::Value) -> State {
    State::new(
        "expert-order",
        NOW,
        Config(json!({ "moe_config": { "expert_parallel": true, "expert_order": orders } })),
    )
}

fn orders(output: Option<Config>) -> Option<BTreeMap<String, Vec<u32>>> {
    output?.get("moe_config.expert_order")
}

/// The skewed layer gets the order that balances its two ranks, moving only the experts that
/// change rank; the even layer keeps the checkpoint's order, and a layer the configuration
/// stated keeps its stated order.
#[test]
fn a_skewed_layer_is_balanced_and_the_others_keep_their_order() {
    let mut program = program(&skewed());
    let stated = json!({ "model.layers.2.mlp.experts": [7, 6, 5, 4, 3, 2, 1, 0] });

    let next = orders(program.step(&state(stated)).unwrap()).unwrap();
    assert_eq!(
        next,
        BTreeMap::from([
            (SKEWED.to_owned(), vec![0, 4, 7, 3, 1, 5, 6, 2]),
            (
                "model.layers.2.mlp.experts".to_owned(),
                vec![7, 6, 5, 4, 3, 2, 1, 0]
            ),
        ])
    );
}

/// A layer whose stated order already balances its ranks, and a deployment whose layers took no
/// tokens, change nothing and validate nothing.
#[test]
fn a_balanced_layer_changes_nothing() {
    let deployment = skewed();
    let mut program = program(&deployment);
    let balanced = json!({ SKEWED: [0, 4, 7, 3, 1, 5, 6, 2] });
    assert_eq!(
        program.step::<_, Option<Config>>(&state(balanced)).unwrap(),
        None
    );

    *program.host_mut() = Deployment::default();
    assert_eq!(
        program
            .step::<_, Option<Config>>(&state(json!({})))
            .unwrap(),
        None
    );
    assert!(deployment.seen().validated.is_empty());
}

/// A layer whose experts do not divide among the ranks is left as it is, with a warning.
#[test]
fn experts_that_do_not_divide_among_the_ranks_are_left_alone() {
    let odd = deployment(&[(SKEWED, &[40.0, 30.0, 20.0, 10.0, 4.0, 3.0, 2.0])]);
    let mut program = program(&odd);
    assert_eq!(
        program
            .step::<_, Option<Config>>(&state(json!({})))
            .unwrap(),
        None
    );
    assert_eq!(
        odd.seen().logs,
        [(
            Level::Warn,
            format!("layer {SKEWED}: 7 experts do not divide among 2 ranks")
        )]
    );
}

/// A rolled-back order of the program's own holds it back for an hour; an order validation
/// reports as stopping serving is not output.
#[test]
fn a_rolled_back_order_pauses_the_program() {
    let paused = skewed().record(revision("expert-order", "rolled_back", NOW - mins(30)));
    let mut program = program(&paused);
    assert_eq!(
        program
            .step::<_, Option<Config>>(&state(json!({})))
            .unwrap(),
        None
    );

    *program.host_mut() = skewed().record(revision("expert-order", "rolled_back", NOW - hours(2)));
    assert!(orders(program.step(&state(json!({}))).unwrap()).is_some());

    *program.host_mut() = skewed().report(Ok(Report {
        effect: Some("epoch".into()),
        interrupts: true,
        rooms: Vec::new(),
    }));
    assert_eq!(
        program
            .step::<_, Option<Config>>(&state(json!({})))
            .unwrap(),
        None
    );
}
