//! The runner of a text rank: one causal language model over its paged cache.

use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use ops::{DType, Generation};
use platform::DeviceId;
use worker::memory::Memory;
use worker::plan::Variant;
use worker::runner::{Binding, NamedRunner, RunnerId};
use worker::runners::causal_lm::{CausalLmRunner, Geometry, KvProvider, Shared};
use worker::runners::resident::ResidentCache;

use super::buckets::served_phases;
use super::{Loaded, ModelRunner, RankGroups};

impl ModelRunner {
    /// The runners `device` serves: one per block of the loaded model.
    ///
    /// The model's class was resolved once, when it was loaded and its resident memory opened
    /// ([`ModelRunner::load_and_open`]), so this builds the runner that class needs from the model it
    /// is handed. Everything downstream - the buckets, the arena, the sampler, the graphs - is
    /// written against `nn`'s capability traits.
    ///
    /// A media rank also maps the host latent pool its denoiser writes preempted trajectories to,
    /// at its Storage Server through `client`, keyed by `identity`, the weights it loaded; `report`
    /// is the report that states the pool for the worker group, given to the rank that states it.
    #[expect(
        clippy::too_many_arguments,
        reason = "the device, the model and its memory, the generation, the lanes, the groups, and the host latent pool's storage"
    )]
    pub(crate) fn runners(
        &self,
        device: DeviceId,
        loaded: &Loaded,
        memory: &mut Memory,
        generation: &Arc<Generation>,
        blocking: &worker::blocking::ThreadPool,
        groups: &RankGroups,
        client: &Arc<worker::memory::retained::Retained>,
        identity: &str,
        report: Option<&mut ipc::MainProcessInfo>,
    ) -> Result<Vec<NamedRunner>> {
        let model = match loaded {
            Loaded::Text(model) => model,
            Loaded::Video(pipeline) => {
                return self.media_runners(
                    device, pipeline, generation, blocking, groups, client, identity, report,
                );
            }
        };
        let mut runners: Vec<NamedRunner> = Vec::new();
        // One model per rank is loaded and mapped on the rank's first device, and a kernel on a
        // second device would read the first device's addresses: a node whose rank serves two
        // devices needs the model loaded per device. Refusing says so; serving would emit wrong
        // tokens and look like a kernel defect.
        if Some(&device) != self.devices.first() {
            bail!(
                "this build maps one model per rank, on device {}; a runner on device {} would \
                 forward from another device's addresses",
                self.devices.first().map_or(-1, |first| first.0),
                device.0
            );
        }
        {
            {
                let geometry = Geometry {
                    hidden: model.hidden(),
                    hidden_dtype: model.dtype(),
                    // A logit row is written in f32: the sampler and the read-back both read it,
                    // and the head's own precision would decide a token from the last bit.
                    logits_dtype: DType::F32,
                    vocabulary: model.vocabulary(),
                    slots: u32::try_from(self.engine_config().max_running_requests.get())
                        .context("more request slots than the protocol addresses")?,
                    // A request may ask for any top-k it likes, so the candidate width is the
                    // vocabulary: a narrower buffer would silently sample a different set.
                    width: model.vocabulary(),
                    // The model's page, which the resident cache was planned with.
                    page_size: model.cache().page_tokens,
                    max_kv_len: self.engine_config().model.max_model_len,
                };
                let stream = platform::Stream::new(device)
                    .with_context(|| format!("creating device {}'s stream", device.0))?;
                let rows = self
                    .buckets
                    .largest(Variant::DecodeGreedy)
                    .map_or(1, |bucket| bucket.rows);
                let cache = memory.shared_cache().context(
                    "a causal-language rank reads a paged cache, and this rank opened none",
                )?;
                let provider: Box<dyn KvProvider> =
                    Box::new(ResidentCache::new(cache, device, rows)?);
                let member = self.union_member(device, model.as_ref())?;
                let runner = CausalLmRunner::new(
                    Binding {
                        id: RunnerId(0),
                        phases: served_phases(&self.buckets),
                        device,
                        stream,
                    },
                    Shared::new(Arc::clone(model)),
                    Arc::clone(generation),
                    self.buckets.clone(),
                    geometry,
                    provider,
                    // A seed that changes with the process, so two runs of one deployment are not
                    // the same run.
                    seed(),
                )?;
                let runner = match member {
                    Some(member) => runner.in_union(member),
                    None => runner,
                };
                let runner = match &groups.tensor {
                    Some(group) => runner.in_tensor_group(Arc::clone(group)),
                    None => runner,
                };
                runners.push(NamedRunner::new(RunnerId(0), Box::new(runner)));
            }
        }
        Ok(runners)
    }
}

/// A seed for the runner's draws: the process's clock, so two runs differ.
///
/// One value for the whole process: the ranks of a tensor group each sample from the same gathered
/// logits, and a rank that drew another token would feed its next step another input than its
/// peers, whose sums it then joins.
pub(super) fn seed() -> u64 {
    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SEED.get_or_init(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos() as u64);
        (nanos ^ u64::from(std::process::id())) | 1
    })
}
