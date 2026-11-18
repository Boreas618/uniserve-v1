//! The engine under a kernel generation's graph options (`proto::config::GraphOptions`): the
//! options its runners take ([`runner_options`]), and what its schedulers and the requests it
//! admits are held to under them ([`Bounds`]): each worker group's captured shapes, the ceilings
//! they bound a step to, the schedulers' limits and the generation limits every request is
//! checked against.
//!
//! Every rank of a deployment captures one catalog (`crate::process::ranks::rank::catalog`), so
//! what a group's workers capture under other options follows from the options alone, on the
//! scheduler node as on every peer: a group that serves token phases states the catalog's
//! shapes, with a speculating deployment's verifications, beside the shapes of whatever else it
//! serves (its vision encodings), and a group of no token phase states what it stated. A
//! generation of other options switches every scheduler to its bounds and a rollback back to
//! the serving ones (`crate::process::generation`); the control thread's copy follows each.

use std::sync::Arc;

use anyhow::{Result, bail};
use proto::config::{EngineConfig, SchedulerConfig};
use proto::request::GenerationLimits;
use storage::runtime::GroupId;

use super::ranks::rank::buckets::served_phases;
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

/// [`runner_options`] of `options`, a generation's graph options other than `serving`, the ones
/// the engine of `config` serves, once they are checked against what the engine serves in
/// place.
///
/// # Errors
///
/// A catalog the deployment's bounds refuse, or one whose buckets serve other phases than the
/// serving catalog's, since every worker's queue routes the phases it started with.
pub(crate) fn checked_options(
    config: &EngineConfig,
    serving: &proto::config::GraphOptions,
    options: &proto::config::GraphOptions,
) -> Result<GraphOptions> {
    let runners = runner_options(config, options)?;
    let phases = served_phases(&runners.buckets.with_drafts(runners.drafts));
    let served =
        served_phases(&catalog(config, &serving.cuda_graph)?.with_drafts(serving.num_draft_tokens));
    if phases != served {
        bail!(
            "a catalog whose buckets serve {phases:?} for workers that serve {served:?}: a \
             worker routes the phases it started with"
        );
    }
    Ok(runners)
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
        let max_logprobs = limits.max_logprobs;
        Self {
            capacities,
            ceilings,
            limits,
            scheduling: Arc::new(GraphLimits {
                config: scheduling,
                buckets,
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
        let shapes: Vec<Shape> = options
            .buckets
            .with_drafts(options.drafts)
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
                max_logprobs,
            }),
            capacities,
            ceilings,
            limits,
        }
    }
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
