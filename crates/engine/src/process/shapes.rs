//! The engine under a kernel generation's graph options (`proto::config::GraphOptions`): the
//! options its runners take ([`runner_options`]), and what its schedulers and the requests it
//! admits are held to under them ([`Bounds`]): each worker group's captured shapes, the ceilings
//! they bound a step to, the schedulers' limits and the generation limits every request is
//! checked against.
//!
//! Every rank of a deployment captures one catalog (`crate::process::ranks::rank::catalog`), so
//! what a group's workers capture under other options follows from the options alone, on the
//! scheduler node as on every peer: a group that serves token phases states the catalog's
//! shapes, with a speculating generation's verifications, and the token phases they serve, beside
//! the shapes of whatever else it serves (its vision encodings), and a group of no token phase
//! states what it stated. A
//! generation of other options switches every scheduler to its bounds and a rollback back to
//! the serving ones (`crate::process::generation`); the control thread's copy follows each.

use std::sync::Arc;

use anyhow::Result;
use proto::config::{EngineConfig, SchedulerConfig};
use proto::request::GenerationLimits;
use storage::runtime::GroupId;

use super::ranks::rank::catalog;
use super::start::{lowered, step_budget};
use crate::batch::capacity::variants;
use crate::batch::{Capacity, Phase, Shape};
use crate::runner::GraphOptions;
use crate::scheduler::GraphLimits;
use crate::scheduler::policy::{Ceilings, ceilings};

/// The options the runners of a deployment of `config` take under `options`, a generation's
/// graph options: the catalog of their sizes within the deployment's bounds, their candidates
/// and their drafts.
///
/// # Errors
///
/// A catalog the deployment's bounds refuse (`catalog`).
pub(crate) fn runner_options(
    config: &EngineConfig,
    options: &proto::config::GraphOptions,
) -> Result<GraphOptions> {
    Ok(GraphOptions {
        buckets: catalog(config, &options.cuda_graph)?,
        candidates: options.max_logprobs,
        drafts: options.num_draft_tokens,
    })
}

/// What the engine holds its steps and its requests to under one kernel generation's graph
/// options: the capacity each worker group states, the ceilings their shapes bound a step to,
/// the generation limits a request is checked against, and every scheduler's limits.
#[derive(Clone, Debug)]
pub(crate) struct Bounds {
    pub(crate) capacities: Vec<(GroupId, Capacity)>,
    pub(crate) ceilings: Ceilings,
    pub(crate) limits: GenerationLimits,
    pub(crate) scheduling: Arc<GraphLimits>,
}

impl Bounds {
    /// The bounds of an engine whose groups state `capacities`, whose shapes bound a step to
    /// `ceilings`, and whose schedulers were configured with `scheduling`, as it starts: `limits`
    /// are the generation limits it checks requests against.
    pub(crate) fn of(
        capacities: Vec<(GroupId, Capacity)>,
        ceilings: Ceilings,
        limits: GenerationLimits,
        scheduling: SchedulerConfig,
    ) -> Self {
        let buckets = capacities
            .iter()
            .map(|(group, capacity)| (*group, capacity.buckets.clone()))
            .collect();
        let phases = capacities
            .iter()
            .map(|(group, capacity)| (*group, capacity.phases))
            .collect();
        let max_logprobs = limits.max_logprobs;
        Self {
            capacities,
            ceilings,
            limits,
            scheduling: Arc::new(GraphLimits {
                config: scheduling,
                buckets,
                phases,
                max_logprobs,
            }),
        }
    }

    /// These bounds, of an engine of `config`, under `options`, the runners' options of a
    /// generation of other graph options: every group that serves token phases states the
    /// shapes of the options' catalog with their drafts, the ceilings and the generation limits
    /// follow from them, `max_logprobs` candidates are what a request may ask for, and the
    /// schedulers are configured with the options' drafts, the batched tokens the new ceilings
    /// leave and the new graph tiers.
    pub(crate) fn under(
        &self,
        config: &EngineConfig,
        options: &GraphOptions,
        max_logprobs: u32,
    ) -> Self {
        let drafted = options.buckets.with_drafts(options.drafts);
        let phases = drafted.phases();
        let shapes: Vec<Shape> = drafted
            .iter()
            .filter(|bucket| batched(bucket))
            .map(|bucket| bucket.shape())
            .collect();
        let capacities: Vec<(GroupId, Capacity)> = self
            .capacities
            .iter()
            .map(|(group, capacity)| {
                let mut capacity = capacity.clone();
                if capacity.buckets.iter().any(token_shape) {
                    capacity.phases = token_phases(capacity.phases, &phases);
                    let mut buckets = shapes.clone();
                    buckets.extend(
                        capacity
                            .buckets
                            .iter()
                            .filter(|shape| !token_shape(shape))
                            .copied(),
                    );
                    capacity.buckets = buckets;
                }
                (*group, capacity)
            })
            .collect();

        let budget = &config.scheduler.policy_budget;
        let ceilings = ceilings(capacities.iter().map(|(_, capacity)| capacity), budget);
        let mut limits = lowered(
            GenerationLimits {
                max_logprobs,
                ..config.model.generation_limits.clone()
            },
            &capacities,
            &ceilings,
        );
        limits.features = self.limits.features;

        let mut tiers: Vec<u32> = capacities
            .iter()
            .flat_map(|(_, capacity)| capacity.buckets.iter().map(|shape| shape.tokens))
            .collect();
        tiers.sort_unstable();
        tiers.dedup();
        let mut speculating = config.clone();
        speculating.speculative.num_draft_tokens = options.drafts;
        let scheduling = SchedulerConfig::of(
            &speculating,
            self.scheduling.config.max_batch.get(),
            step_budget(config, &ceilings),
            tiers,
        );

        Self {
            scheduling: Arc::new(GraphLimits {
                config: scheduling,
                buckets: capacities
                    .iter()
                    .map(|(group, capacity)| (*group, capacity.buckets.clone()))
                    .collect(),
                phases: capacities
                    .iter()
                    .map(|(group, capacity)| (*group, capacity.phases))
                    .collect(),
                max_logprobs,
            }),
            capacities,
            ceilings,
            limits,
        }
    }
}

/// The phases a group that served `served` (bit [`Phase::index`] per phase) serves once its
/// token phases are those its workers' buckets under a generation's options serve, `phases`
/// (`Buckets::phases`): drafts make its decodes verifications and none make them decodes again,
/// a group whose runners refill serves refills under any drafts, and one whose model keeps a
/// recurrent state serves none (`CausalLmRunner::served_phases`), and every other phase stays.
fn token_phases(served: u32, phases: &[Phase]) -> u32 {
    let bit = |phase: Phase| 1_u32 << phase.index();
    let tokens = [Phase::Prefill, Phase::Decode, Phase::Verify, Phase::Refill];
    let refills = served & bit(Phase::Refill) != 0;

    let mut stated = tokens
        .iter()
        .fold(served, |stated, phase| stated & !bit(*phase));
    for phase in phases {
        if tokens.contains(phase) && (*phase != Phase::Refill || refills) {
            stated |= bit(*phase);
        }
    }
    stated
}

/// Whether a batch of a token phase is formed against `shape`: a prefill's, a decode's or a
/// verification's.
fn token_shape(shape: &Shape) -> bool {
    [Phase::Prefill, Phase::Decode, Phase::Verify]
        .iter()
        .any(|phase| variants(*phase).contains(&shape.variant))
}

/// Whether a batch is formed against `bucket`: a drafter's bucket and an expert join serve no
/// phase, and no batch states them.
fn batched(bucket: &crate::runner::buckets::Bucket) -> bool {
    Phase::ALL
        .iter()
        .any(|phase| variants(*phase).contains(&bucket.variant))
}

#[cfg(test)]
mod tests;
