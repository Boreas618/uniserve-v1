//! The engine's runners' options and its bounds under a generation's graph options, for the
//! deployment of `testkit::serving::launch`: one group of eight slots, decodes of one and two
//! rows, prefills of 16 and 32 tokens, a longest sequence of 64.

use std::num::NonZeroU32;

use proto::config::{CudaGraphConfig, EngineConfig, GraphOptions, SchedulerConfig};
use proto::request::GenerationLimits;
use storage::runtime::GroupId;

use super::{Bounds, checked_options, runner_options};
use crate::batch::capacity::variants;
use crate::batch::{Capacity, Phase, Shape, Variant};
use crate::scheduler::policy::ceilings;

fn config() -> EngineConfig {
    testkit::serving::launch().engine_config
}

fn sizes(sizes: &[u32]) -> Option<Vec<NonZeroU32>> {
    Some(
        sizes
            .iter()
            .map(|size| NonZeroU32::new(*size).unwrap())
            .collect(),
    )
}

/// `config`'s options with decodes of `decode` rows and prefills of `prefill` tokens.
fn catalog(config: &EngineConfig, decode: &[u32], prefill: &[u32]) -> GraphOptions {
    GraphOptions {
        cuda_graph: CudaGraphConfig {
            decode_batch_sizes: sizes(decode),
            prefill_token_sizes: sizes(prefill),
        },
        ..config.graph_options()
    }
}

/// The bounds of `config`'s engine as it starts, its one group stating its catalog's shapes and
/// an image encoding beside them.
fn serving(config: &EngineConfig) -> Bounds {
    let options = runner_options(config, &config.graph_options()).unwrap();
    let mut capacity = crate::scheduler::tests::capacity();
    capacity.buckets = options
        .buckets
        .iter()
        .map(|bucket| bucket.shape())
        .chain([Shape::prefill(1, 64, Variant::ConditionEncode)])
        .collect();
    let capacities = vec![(GroupId(0), capacity)];
    let budget = config.scheduler.policy_budget;
    let ceilings = ceilings(capacities.iter().map(|(_, capacity)| capacity), &budget);
    let limits = GenerationLimits {
        max_logprobs: config.max_logprobs,
        ..GenerationLimits::default()
    };
    Bounds::of(
        capacities,
        ceilings,
        limits,
        SchedulerConfig::of(config, 2, 32, Vec::new()),
    )
}

/// The token shapes `capacity` states, by rows and tokens.
fn token_shapes(capacity: &Capacity) -> Vec<(u32, u32)> {
    let mut shapes: Vec<(u32, u32)> = capacity
        .buckets
        .iter()
        .filter(|shape| {
            [Phase::Prefill, Phase::Decode, Phase::Verify]
                .iter()
                .any(|phase| variants(*phase).contains(&shape.variant))
        })
        .map(|shape| (shape.rows, shape.tokens))
        .collect();
    shapes.sort_unstable();
    shapes.dedup();
    shapes
}

#[test]
fn a_generation_s_catalog_bounds_the_groups_steps_and_requests() {
    let config = config();
    let bounds = serving(&config);

    // Decodes of up to four rows and prefills of up to 64 tokens, with three candidates and
    // verifications of two drafts.
    let mut options = catalog(&config, &[1, 4], &[64]);
    options.max_logprobs = 3;
    options.num_draft_tokens = 2;
    let runners = runner_options(&config, &options).unwrap();
    let under = bounds.under(&config, &runners, options.max_logprobs);

    let (_, capacity) = &under.capacities[0];
    let shapes = token_shapes(capacity);
    // A verification of a row and two drafts holds three tokens; one of four rows twelve.
    assert!(
        shapes.contains(&(1, 3)) && shapes.contains(&(4, 12)),
        "{shapes:?}"
    );
    assert!(shapes.iter().all(|(_, tokens)| *tokens <= 64), "{shapes:?}");
    assert!(
        capacity
            .buckets
            .contains(&Shape::prefill(1, 64, Variant::ConditionEncode)),
        "a group's other shapes are its own"
    );
    assert_eq!(under.ceilings.tokens, 64);
    assert_eq!(under.limits.max_logprobs, 3);
    assert_eq!(under.scheduling.max_logprobs, 3);
    assert_eq!(under.scheduling.config.draft_tokens, 2);
    assert_eq!(under.scheduling.config.max_num_batched_tokens, 64);
    assert_eq!(under.scheduling.buckets[0].1, capacity.buckets);

    // The serving bounds are what a rollback restores: they state what they stated.
    assert_eq!(
        token_shapes(&bounds.capacities[0].1),
        [(1, 1), (2, 2), (2, 16), (2, 32)]
    );
    assert_eq!(bounds.ceilings.tokens, 32);
}

#[test]
fn options_the_engine_does_not_serve_in_place_are_refused_by_what_they_change() {
    let config = config();
    let serving = config.graph_options();

    // A token tier above the longest sequence is captured and never filled.
    let long = catalog(&config, &[1, 2], &[128]);
    let refused = format!(
        "{:#}",
        checked_options(&config, &serving, &long).unwrap_err()
    );
    assert!(refused.contains("max_model_len"), "{refused}");

    // Drafts where none were make the decode buckets verifications, a phase the workers'
    // queues do not route.
    let drafting = GraphOptions {
        num_draft_tokens: 2,
        ..serving.clone()
    };
    let refused = format!(
        "{:#}",
        checked_options(&config, &serving, &drafting).unwrap_err()
    );
    assert!(refused.contains("Verify"), "{refused}");

    // Other sizes within the bounds are served in place, and so is CUPTI's other mode, whose
    // graphs are captured again.
    let other = catalog(&config, &[1, 2, 4], &[8, 32]);
    assert!(checked_options(&config, &serving, &other).is_ok());
    let sampling = GraphOptions {
        graph_pc_sampling: true,
        ..serving.clone()
    };
    assert!(checked_options(&config, &serving, &sampling).is_ok());
}
