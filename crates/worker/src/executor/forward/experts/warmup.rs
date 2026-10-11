//! Expert warmup: plan comparison, announced warmup steps and join capture.
//!
//! Every member of the union reaches the same collectives in the same
//! order:
//!
//! 1. [`WarmupHooks::prepare`]: every rank compares its plan (capacities,
//!    microbatches, layers and, for sources, the warmup and capture plan),
//!    before any collective step.
//! 2. [`WarmupHooks::capture`], after the text runtime captured its graphs:
//!    each capture's eager warmup was an expert step the sources announced.
//!    Dedicated expert ranks follow the announcements until the sources
//!    announce zero; then every rank captures its joins, warming each
//!    capacity, largest first, as a collective step. Steps are agreed from
//!    then on.

use super::super::super::backend::WorkerBackend;
use crate::error::WorkerError;
use crate::warmup::WarmupHooks;
use crate::{Error, Result};

/// The expert runtime's warmup work.
pub(crate) struct ExpertWarmup;

fn failed(phase: &'static str) -> impl Fn(Error) -> WorkerError {
    move |error| WorkerError::classify(error, Some(phase)).with_phase("warmup")
}

impl WarmupHooks for ExpertWarmup {
    fn prepare(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let text = backend.text_mut().map(|text| text.capture_plan());
        let diffusion_lm = backend
            .diffusion_lm_mut()
            .map(|runner| runner.capture_plan());
        let calls = (text.is_some() || diffusion_lm.is_some())
            .then(|| serde_json::json!({"text": text, "diffusion_lm": diffusion_lm}));
        let Some(experts) = backend.experts() else {
            return Ok(());
        };
        experts
            .compare_plans(calls)
            .map_err(failed("expert plan comparison"))
    }

    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let Some(experts) = backend.experts_mut() else {
            return Ok(());
        };
        follow(experts).map_err(failed("expert warmup"))?;
        experts.serve();
        experts
            .capture_joins()
            .map_err(failed("expert join capture"))
    }
}

/// Dedicated expert ranks execute every announced warmup step; sources end
/// their announcements with zero.
fn follow(experts: &mut super::ExpertRuntime) -> Result<()> {
    let parallel = std::sync::Arc::clone(experts.parallel());
    if parallel.attention_ranks() == 0 {
        return Ok(());
    }
    if !experts.dedicated() {
        parallel.warmup(0)?;
        return Ok(());
    }
    let mut steps = 0;
    loop {
        let capacity = parallel.warmup(0)?;
        if capacity == 0 {
            break;
        }
        experts.join_eager(capacity)?;
        steps += 1;
    }
    tracing::info!(
        target: "uniserve_worker.warmup",
        "joined {steps} announced expert warmup steps"
    );
    Ok(())
}
