//! The runner of a text rank: one causal language model over the rank's cache and recurrent state
//! (`specs/engine-worker-port.md` §1.1, steps 7 and 21-22).
//!
//! A rank builds its runner twice from the same config. First over owners that hold no device
//! (`KvCacheMemory::meta`, `StateMemory::meta`) and graphs that hold none (`Graphs::meta`): its
//! profile run states the region, shadow region and input bytes the division sizes the rank's pools
//! by, and every kernel the rank launches. Then over the device's owners, in the region of the
//! graph pool the division mapped, as the runner the rank's executor serves with. Both runners
//! are wired alike: the cache issues each batch's page work through the hook of
//! `worker::runners::causal_lm::wiring::page_work`, the tensor group's split is set on the graphs,
//! and a member of an expert union joins it before its buckets are profiled.

use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use nn::family::TextModel;
use ops::Generation;
use platform::Stream;
use worker::graphs::Graphs;
use worker::memory::cache::{CacheLayout, KvCacheMemory};
use worker::memory::state::StateMemory;
use worker::runners::causal_lm::experts::Member;
use worker::runners::causal_lm::speculation::DraftPool;
use worker::runners::causal_lm::wiring::page_work;
use worker::runners::causal_lm::{CausalLmConfig, CausalLmRunner};

use super::ModelRunner;
use super::buckets::served_phases;
use super::memory::PAGE_TOKENS;

/// The kernels a rank's runner and owners launch, as `(operator, key)`: what the engine resolves
/// the rank's serving generation for.
pub(crate) type Keys = std::collections::BTreeSet<(&'static str, String)>;

impl ModelRunner {
    /// The config of the rank's runner of `model` over a cache of `layout` with `units` units
    /// (step 7): the phases its buckets serve, the deployment's request slots, pages and longest
    /// sequence, the candidates a scored row reports, the scratch pages of a paired forward, and a
    /// speculating deployment's drafts and draft pool.
    ///
    /// A paired forward's row addresses the pages of the largest prefill bucket's tokens; a model
    /// that keeps a recurrent state has no scratch state to advance and computes no paired
    /// forward.
    ///
    /// # Errors
    ///
    /// More request slots than the protocol addresses, or a speculating deployment whose cache
    /// keeps no draft group.
    pub(crate) fn causal_config(
        &self,
        model: &dyn TextModel,
        layout: &CacheLayout,
        units: u32,
    ) -> Result<CausalLmConfig> {
        let drafts = self.engine_config().speculative.num_draft_tokens;
        let (_, tokens) = self.buckets.bounds();
        let scratch_pages = if model.state_layers().is_empty() {
            tokens.div_ceil(PAGE_TOKENS)
        } else {
            0
        };
        let draft_pool = if drafts > 0 {
            Some(DraftPool::of(layout, units)?)
        } else {
            None
        };

        Ok(CausalLmConfig {
            phases: served_phases(&self.buckets),
            buckets: self.buckets.clone(),
            slots: self.slots()?,
            page_size: PAGE_TOKENS,
            max_kv_len: self.engine_config().model.max_model_len,
            candidates: self.engine_config().max_logprobs,
            scratch_pages,
            drafts,
            draft_pool,
        })
    }

    /// The runner of `model` under `config`, over the rank's `cache` and `state` and computing
    /// in `graphs`, which replay on `stream` (`None` for graphs that hold no device), profiled
    /// (steps 7 and 22).
    ///
    /// The runner issues its page work through the cache, its forwards split over the tensor
    /// group `group` at this rank's place, and `member` makes it the rank's member of the node's
    /// expert union: on a device its transport is created first, with `kernels`, on the stream
    /// the runner's executor issues on.
    ///
    /// # Errors
    ///
    /// What the runner, the graphs, the union or a forward of the profile run refuse.
    #[expect(
        clippy::too_many_arguments,
        reason = "the runner is joined to six owners: the model, its graphs and stream, the cache, \
                  the state, the tensor group and the expert union"
    )]
    pub(crate) fn runner(
        &self,
        model: Arc<dyn TextModel>,
        config: CausalLmConfig,
        graphs: Graphs,
        stream: Option<Stream>,
        cache: &Arc<Mutex<KvCacheMemory>>,
        state: Option<&Arc<Mutex<StateMemory>>>,
        group: Option<Arc<ops::Group>>,
        member: Option<Member>,
        kernels: &Generation,
    ) -> Result<CausalLmRunner> {
        let scratch_pages = config.scratch_pages;

        let mut runner = {
            let cache = worker::lock(cache);
            let state = state.map(|state| worker::lock(state));
            CausalLmRunner::over(
                model,
                config,
                graphs,
                stream.clone(),
                &cache,
                state.as_deref(),
            )
            .context("building the rank's causal runner")?
        };
        runner.set_page_work(page_work(
            Arc::clone(cache),
            state.map(Arc::clone),
            stream.clone(),
            scratch_pages,
        ));

        // The meta contexts take the split alone: a profile run issues no collective.
        let (rank, ranks) = super::split(
            self.mapping.tensor_rank as usize,
            self.mapping.tensor_parallel as usize,
        );
        worker::Runner::graphs(&runner).set_group(group, rank, ranks)?;

        if let Some(mut member) = member {
            if let Some(stream) = &stream {
                member
                    .create_transport(kernels, stream)
                    .context("creating the expert union's transport")?;
            }
            runner
                .join_union(member)
                .context("joining the node's expert union")?;
        }
        Ok(runner)
    }
}

/// Runs `runner`'s profile run: the forward of every bucket it states under the meta context, which
/// states its region, shadow and input bytes and the kernels its graphs launch.
///
/// # Errors
///
/// What a forward refuses.
pub(crate) fn profile(runner: &CausalLmRunner) -> Result<()> {
    let graphs = worker::Runner::graphs(runner);
    graphs
        .profile_run(runner)
        .context("profiling the rank's causal runner")
}

/// Every kernel `runner` and the owners it reads launch: its graphs' profiled launches, the
/// cache's copies and conversions, and the recurrent state's copies.
///
/// # Errors
///
/// When an owner's descriptors refuse its tensors.
pub(crate) fn keys(
    runner: &CausalLmRunner,
    cache: &Mutex<KvCacheMemory>,
    state: Option<&Mutex<StateMemory>>,
) -> Result<Keys> {
    let mut keys = worker::Runner::graphs(runner).keys();
    keys.extend(worker::lock(cache).keys()?);
    if let Some(state) = state {
        keys.extend(worker::lock(state).keys()?);
    }
    Ok(keys)
}
