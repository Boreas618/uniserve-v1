//! The runner of a text rank: one causal language model over its paged cache.

use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use ipc::execution::Variant;
use ops::Generation;
use platform::DeviceId;
use worker::memory::Memory;
use worker::runner::{Binding, NamedRunner, RunnerId};
use worker::runners::causal_lm::{CausalLmRunner, KvProvider, Shared};
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
    /// is the report that states the pool for the worker group, given to the rank that states it. A
    /// recorded start has no Storage Server (`client` is `None`) and maps no host pool: a host page
    /// is read and written by a request's copies, which no recorded walk makes.
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
        client: Option<&Arc<worker::memory::retained::Retained>>,
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
                // The sampler's penalty table holds a row per request slot the deployment serves.
                let slots = u32::try_from(self.engine_config().max_running_requests.get())
                    .context("more request slots than the protocol addresses")?;
                let stream = platform::Stream::new(device)
                    .with_context(|| format!("creating device {}'s stream", device.0))?;
                let drafts = self.engine_config().speculative.num_draft_tokens;
                let rows = self
                    .buckets
                    .largest(Variant::DecodeGreedy)
                    .map_or(1, |bucket| bucket.rows);
                // A verification attends as one row per token.
                let rows = if drafts > 0 {
                    CausalLmRunner::<Shared<dyn nn::family::TextModel>>::speculative_rows(
                        slots, drafts, rows,
                    )
                } else {
                    rows
                };
                let cache = memory.shared_cache().context(
                    "a causal-language rank reads a paged cache, and this rank opened none",
                )?;
                let provider: Box<dyn KvProvider> =
                    Box::new(ResidentCache::new(cache, device, rows)?);
                let member = self.union_member(device, model.as_ref())?;
                // A model that takes images also serves their vision encodings.
                let images = model.vision().is_some();
                let mut phases = served_phases(&self.buckets, drafts > 0);
                if images {
                    phases.push(super::buckets::phase(scheduler::PhaseKind::VisionEncoding));
                    phases.sort_by_key(|phase| phase.0);
                    phases.dedup();
                }
                let runner = CausalLmRunner::new(
                    Binding {
                        id: RunnerId(0),
                        phases,
                        device,
                        stream,
                    },
                    Shared::new(Arc::clone(model)),
                    Arc::clone(generation),
                    self.buckets.clone(),
                    provider,
                    slots,
                    // The attention descriptor's static key bound: the deployment's longest
                    // sequence, which the resident cache's block-table rows were sized for.
                    self.engine_config().model.max_model_len,
                )?;
                let runner = match member {
                    Some(member) => runner.in_union(member),
                    None => runner,
                };
                let runner = if drafts > 0 {
                    runner.speculating(drafts)?
                } else {
                    runner
                };
                let runner = if images {
                    // The images' reads run on the process's media lane: a text process builds no
                    // other runner that takes it.
                    let lane = blocking
                        .lane::<worker::runners::vision::ImageRead>(
                            worker::blocking::BlockingKind::Media,
                        )
                        .context("creating the lane that reads input images")?;
                    let runner = runner.with_images(lane)?;
                    // The host encoder pool an image's features are stored in and found again by
                    // another request reading the same image: one per host, shared by its ranks
                    // and epochs, keyed by the identity of the weights that compute them. A
                    // recorded start, which has no Storage Server, maps none.
                    match (client, runner.image_features()) {
                        (Some(client), Some((row_bytes, rows))) => {
                            let (pool, info) = worker::memory::encoders::EncoderHostPool::start(
                                client,
                                device,
                                worker::memory::encoders::pool_pages(
                                    &[u64::from(rows) * row_bytes],
                                    self.engine_config().max_running_requests.get(),
                                ),
                                std::collections::BTreeMap::from([(
                                    ipc::product::VISION_FEATURES.to_owned(),
                                    identity.to_owned(),
                                )]),
                                None,
                            )
                            .context("mapping the host encoder pool")?;
                            let writes = report.is_some();
                            if let Some(report) = report {
                                report.encoder_pool = Some(Box::new(info));
                            }
                            runner.with_encoder_pool(std::sync::Arc::new(pool), writes)
                        }
                        _ => runner,
                    }
                } else {
                    runner
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
