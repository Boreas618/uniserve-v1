//! What a rank states of its worker group once its workers are ready: the group's [`Capacity`]
//! (`specs/scheduler-design.md` §2.1), which the scheduler that drives the group plans against
//! (`specs/engine-worker-port.md` §1.2).

use crate::batch::Capacity;
use crate::batch::Phase;
use crate::batch::capacity::variants;
use crate::memory::cache::CacheLayout;
use crate::memory::division::call_window;
use crate::runner::causal_lm::CausalLmConfig;
use anyhow::{Context as _, Result};

use super::ModelRunner;

impl ModelRunner {
    /// The capacity of a text rank whose runner is of `config` over a cache of `layout`: the
    /// phases its buckets serve, each held by every request slot at once (a token phase's
    /// stage is the slots), the shapes it captured that a batch states, the cache groups in
    /// class order with their page `seeds` and the deployment's cache configuration, and the
    /// model's limits and control tokens. A text rank encodes nothing itself and denoises
    /// nothing.
    ///
    /// Every rank of a group states the same: its config and layout are the deployment's.
    ///
    /// # Errors
    ///
    /// A queue depth or a bucket of no rows, which bounds no window of calls.
    pub(crate) fn capacity(
        &self,
        config: &CausalLmConfig,
        layout: &CacheLayout,
        seeds: Vec<storage::Key>,
    ) -> Result<Capacity> {
        let mut phases = 0;
        let mut rows = [0; Phase::COUNT];
        for phase in &config.phases {
            phases |= 1 << phase.index();
            rows[phase.index()] = config.slots.max(1);
        }

        // The shapes a batch is formed against: a drafter's bucket is issued behind the batch it
        // follows, and no batch states it.
        let buckets = config
            .buckets
            .iter()
            .filter(|bucket| {
                Phase::ALL
                    .iter()
                    .any(|phase| variants(*phase).contains(&bucket.variant))
            })
            .map(|bucket| bucket.shape())
            .collect();

        let (calls, _) = config.buckets.bounds();
        let window = call_window(self.queue_depth(), calls as usize)
            .context("bounding the calls of a request the group holds unresolved")?;
        let model = &self.engine_config().model;

        Ok(Capacity {
            phases,
            slots: config.slots,
            kv_cache: Some(self.engine_config().kv_cache.clone()),
            cache_groups: layout
                .groups
                .iter()
                .map(|group| (group.kind, group.block_size))
                .collect(),
            seeds,
            buckets,
            rows,
            max_model_len: model.max_model_len,
            denoise_steps: 0,
            bos_token_id: model.bos,
            eos_token_ids: model.eos.clone(),
            feature_bytes: 0,
            // The group's encoder identity is stated from its encoding ranks' weights by the
            // rank start (`crate::process::ranks::capacities`).
            encoder_identity: String::new(),
            media_bytes: Vec::new(),
            call_window: u32::try_from(window).unwrap_or(u32::MAX),
        })
    }
}
