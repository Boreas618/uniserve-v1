//! What a rank states of its worker group once its workers are ready: the group's [`Capacity`]
//! (`specs/scheduler-design.md` §2.1), which the scheduler that drives the group sizes its
//! batches by (`specs/engine-worker-port.md` §1.2).

use crate::batch::Capacity;
use crate::batch::Phase;
use crate::batch::capacity::variants;
use crate::memory::cache::CacheLayout;
use crate::memory::division::call_window;
use crate::runner::causal_lm::CausalLmConfig;
use crate::runner::vision::VisionRunner;
use anyhow::{Context as _, Result};
use nn::family::TextModel;

use super::ModelRunner;

impl ModelRunner {
    /// The capacity of a text rank whose runner is of `config` over a cache of `layout`: the
    /// phases its buckets serve, each held by every request slot at once (a token phase's
    /// stage is the slots), the shapes it captured that a batch states, the cache groups in
    /// class order with their page `seeds` and the deployment's cache configuration, and the
    /// model's limits, control tokens and the token history `text_model` reads. A text rank
    /// denoises nothing.
    ///
    /// A rank that encodes images of at most `rows` rows with the vision tower of `text_model`,
    /// given as `images: Some(rows)` (`None` on a rank that encodes none,
    /// `ModelRunner::encodes_images`), also serves their encodings, held by every request slot,
    /// in the shapes its vision runner captures, and states the bytes of a row of their encoder
    /// outputs (`Capacity::feature_bytes`).
    ///
    /// Every rank of a group states the same: its config and layout are the deployment's.
    ///
    /// # Errors
    ///
    /// A queue depth or a bucket of no rows, which bounds no window of calls; a tower that sizes
    /// no image.
    pub(crate) fn capacity(
        &self,
        config: &CausalLmConfig,
        text_model: &dyn TextModel,
        layout: &CacheLayout,
        seeds: Vec<storage::Key>,
        images: Option<u32>,
    ) -> Result<Capacity> {
        let mut phases = 0;
        let mut rows = [0; Phase::COUNT];
        for phase in &config.phases {
            phases |= 1 << phase.index();
            rows[phase.index()] = config.slots.max(1);
        }

        // The shapes a batch is formed against: a drafter's bucket is issued behind the batch it
        // follows, and no batch states it.
        let mut buckets: Vec<_> = config
            .buckets
            .iter()
            .filter(|bucket| {
                Phase::ALL
                    .iter()
                    .any(|phase| variants(*phase).contains(&bucket.variant))
            })
            .map(|bucket| bucket.shape())
            .collect();

        // The image encodings, one image a batch: each bucket the vision runner captures holds
        // one.
        let mut feature_bytes = 0;
        if let Some(images) = images {
            let encode = Phase::VisionEncode.index();
            phases |= 1 << encode;
            rows[encode] = config.slots.max(1);
            let captured = VisionRunner::captured(text_model, images)?;
            buckets.extend(captured.iter().map(|bucket| bucket.shape()));
            feature_bytes = VisionRunner::feature_bytes(text_model);
        }

        let (calls, _) = config.buckets.bounds();
        let window = call_window(self.queue_depth()?, calls as usize)
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
            token_history: text_model
                .token_history()
                .map_or(0, |history| history.width),
            feature_bytes,
            // The group's encoder identity and latent seed are stated from its ranks' weights by
            // the rank start (`crate::process::ranks::capacities`).
            encoder_identity: String::new(),
            latent_seed: 0,
            media_bytes: Vec::new(),
            pipeline: None,
            call_window: u32::try_from(window).unwrap_or(u32::MAX),
        })
    }
}
