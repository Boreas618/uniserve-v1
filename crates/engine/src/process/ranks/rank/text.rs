//! The runner of a text rank: one causal language model over the rank's cache and recurrent state
//! (`specs/engine-worker-port.md` §1.1, steps 7 and 21-22).
//!
//! A rank builds its runner twice from the same config. First over owners that hold no device
//! (`KvCacheMemory::meta`, `StateMemory::meta`) and graphs that hold none (`Graphs::meta`): its
//! profile run states the region, shadow region and input bytes the division sizes the rank's pools
//! by, and every kernel the rank launches. Then over the device's owners, in the region of the
//! graph pool the division mapped, as the runner the rank's worker serves with. Both runners
//! are wired alike: the cache issues each batch's page work through the hook of
//! `crate::runner::causal_lm::wiring::page_work`, the tensor group's split is set on the graphs,
//! and a member of an expert union joins it before its buckets are profiled.
//!
//! A rank whose model has a vision tower and whose group prefills ([`encodes`]) also encodes the
//! images of the requests it prefills: its vision runner (`RunnerId(1)`) writes each image's
//! encoder output into the rank's region of encoder outputs, and its causal runner, lent a
//! feature buffer (`CausalLmRunner::set_features`), copies the output of every image block of a
//! prefill into it before the replay (`specs/engine-worker-port.md` §1.1, §2.2). Both read and
//! write that region through one media memory of the rank (`BufferMemory`), and both are
//! profiled with it, the meta runners over a media memory that holds no device. The tower is
//! whole on every rank of a tensor group, so each rank encodes the whole output into its own
//! region, at the units the session gives every rank of the group alike.

use std::sync::{Arc, Mutex};

use crate::batch::Phase;
use crate::batch::product::VISION_FEATURES;
use crate::memory::buffers::BufferMemory;
use crate::memory::cache::{CacheLayout, KvCacheMemory};
use crate::memory::state::StateMemory;
use crate::runner::causal_lm::experts::Member;
use crate::runner::causal_lm::speculation::DraftPool;
use crate::runner::causal_lm::wiring::page_work;
use crate::runner::causal_lm::{CausalLmConfig, CausalLmRunner};
use crate::runner::graphs::Graphs;
use crate::runner::vision::VisionRunner;
use anyhow::{Context as _, Result};
use nn::family::TextModel;
use ops::Generation;
use platform::Stream;
use proto::config::WorkerGroupRole;

use super::ModelRunner;
use super::buckets::served_phases;
use super::memory::PAGE_TOKENS;

/// The kernels a rank's runner and owners launch, as `(operator, key)`: what the engine resolves
/// the rank's serving generation for.
pub(crate) type Keys = std::collections::BTreeSet<(&'static str, String)>;

impl ModelRunner {
    /// The config of the rank's runner of `model` over a cache of `layout` with `units` units
    /// (step 7): the phases its buckets serve (no refill for a model whose layers or drafter keep
    /// a recurrent state), the rank's request slots, pages and longest sequence, the
    /// candidates a scored row reports, the scratch page set of its paired forwards, and a
    /// speculating deployment's drafts and draft pool.
    ///
    /// The scratch page set is one set the rows of a paired forward share,
    /// `ceil(T / page) + R` pages for the `T` tokens and `R` rows of the largest bucket
    /// (`CausalLmRunner::scratch_pages`); a batch whose rows' contexts need more is not
    /// compared. A model that keeps a recurrent state has no scratch state to advance and
    /// computes no paired forward.
    ///
    /// # Errors
    ///
    /// A speculating deployment whose cache keeps no draft group.
    pub(crate) fn causal_config(
        &self,
        model: &dyn TextModel,
        layout: &CacheLayout,
        units: u32,
    ) -> Result<CausalLmConfig> {
        let drafts = self.engine_config().speculative.num_draft_tokens;
        let scratch_pages = if model.state_layers().is_empty() {
            CausalLmRunner::scratch_pages(&self.buckets, PAGE_TOKENS)
        } else {
            0
        };
        let draft_pool = if drafts > 0 {
            Some(DraftPool::of(layout, units)?)
        } else {
            None
        };

        // A refill evaluates positions behind its slot's own from a state of their own, which
        // the runner has none of for a model whose layers or drafter keep a recurrent state, and
        // refuses: such a rank serves no refill, and a request whose draft cache lags verifies
        // without drafts.
        let stateless = model.state_layers().is_empty()
            && model
                .drafter()
                .is_none_or(|drafter| drafter.state().is_empty());
        let phases = served_phases(&self.buckets)
            .into_iter()
            .filter(|phase| *phase != Phase::Refill || stateless)
            .collect();

        Ok(CausalLmConfig {
            phases,
            buckets: self.buckets.clone(),
            slots: self.slots(),
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
    /// the runner's worker issues on.
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
        let mut runner = {
            let cache = service::lock(cache);
            let state = state.map(|state| service::lock(state));
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
        ));

        // The meta contexts take the split alone: a profile run issues no collective.
        let (rank, ranks) = super::split(
            self.mapping.tensor_rank as usize,
            self.mapping.tensor_parallel as usize,
        );
        crate::runner::Runner::graphs(&runner).set_group(group, rank, ranks)?;

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

impl ModelRunner {
    /// Whether this rank encodes the images of the requests it prefills (module documentation):
    /// `model` has a vision tower whose images the engine admits ([`ModelRunner::image_rows`])
    /// and the rank's group prefills ([`encodes`]).
    ///
    /// # Errors
    ///
    /// A rank whose group the deployment does not state.
    pub(crate) fn encodes_images(&self, model: &dyn TextModel) -> Result<bool> {
        let groups = &self.engine_config().parallel.worker_groups;
        let group = groups
            .iter()
            .find(|group| group.id.0 == self.group())
            .with_context(|| format!("the deployment states no group {}", self.group()))?;

        Ok(self.image_rows(model) > 0 && encodes(group.role))
    }

    /// The most rows of an image `model`'s vision tower encodes on this rank: the tower's
    /// largest image, within the rows of a block the engine admits, which are the generation
    /// limits the launch states lowered to what the groups capture (`crate::process::start`'s
    /// `lowered`): an image's grid within the widest prefill bucket, which writes its block
    /// whole, and its features within the feature bytes. Every text group captures this rank's
    /// buckets. Zero for a model without a vision tower or a launch that admits no image.
    pub(crate) fn image_rows(&self, model: &dyn TextModel) -> u32 {
        let Some(vision) = model.vision() else {
            return 0;
        };
        let limits = &self.engine_config().model.generation_limits;
        let widest = self
            .buckets()
            .iter()
            .filter(|bucket| bucket.variant.stage() == "prefill")
            .map(|bucket| bucket.tokens)
            .max()
            .unwrap_or(0);
        let row = u64::from(VisionRunner::feature_bytes(model)).max(1);
        let features = u32::try_from(limits.max_vision_feature_bytes / row).unwrap_or(u32::MAX);

        vision
            .max_image_rows()
            .min(limits.max_vit_grid_tokens)
            .min(widest)
            .min(features)
    }
}

/// Whether a text group of `role` encodes images, given a model with a vision tower: a group that
/// prefills does, since a token request's image encodings are placed on the group of its prefill,
/// which reads them (`crate::scheduler`'s placement). A decode group and an expert group encode
/// none.
pub(crate) fn encodes(role: WorkerGroupRole) -> bool {
    matches!(role, WorkerGroupRole::Model | WorkerGroupRole::Prefill)
}

/// The class of encoder outputs in the session of a text deployment whose cache is laid out as
/// `layout`: the one after the cache groups, before the product classes, where the scheduler
/// finds it.
///
/// # Errors
///
/// More cache groups than a session numbers classes.
pub(crate) fn encoder_class(layout: &CacheLayout) -> Result<u8> {
    u8::try_from(layout.groups.len()).context("more cache groups than a session numbers classes")
}

/// The digest of the tensors an encoder output of `model`'s vision tower holds, which its class
/// states (`storage::runtime::Class::tensors`): rows of features of the model's width in its
/// activation type, then their rotary coordinates (`crate::runner::vision`).
pub(crate) fn feature_tensors(model: &dyn TextModel) -> [u8; 32] {
    let axes = model.vision().map_or(0, |vision| vision.position_axes());

    let mut tensors = storage::key::Digest::new(b"uniserve.vision-features");
    tensors.text(VISION_FEATURES);
    tensors.text(ops::abi::name(model.dtype()));
    tensors.number(u64::from(model.hidden()));
    tensors.number(u64::from(axes));
    tensors.finish()
}

/// The vision runner of `model` encoding images of at most `rows` rows
/// ([`ModelRunner::image_rows`]) over `graphs`, which replay on `stream` (`None` for graphs that
/// hold no device), writing the encoder outputs of class `class` into `memory`, the rank's media
/// memory, profiled (steps 7 and 22): the forward of each grid it captures, which states its
/// region and input bytes and the kernels every grid of at most `rows` rows launches.
///
/// # Errors
///
/// What the runner refuses of the model or its graphs, or a forward of the profile run.
pub(crate) fn vision_runner(
    model: Arc<dyn TextModel>,
    graphs: Graphs,
    stream: Option<Stream>,
    memory: Arc<BufferMemory>,
    (class, rows): (u8, u32),
) -> Result<VisionRunner> {
    let runner = VisionRunner::new(model, graphs, stream, memory, (class, rows))
        .context("building the rank's vision runner")?;
    crate::runner::Runner::graphs(&runner)
        .profile_run(&runner)
        .context("profiling the rank's vision runner")?;
    Ok(runner)
}

/// Runs `runner`'s profile run: the forward of every bucket it states under the meta context, which
/// states its region, shadow and input bytes and the kernels its graphs launch.
///
/// # Errors
///
/// What a forward refuses.
pub(crate) fn profile(runner: &CausalLmRunner) -> Result<()> {
    let graphs = crate::runner::Runner::graphs(runner);
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
    let mut keys = crate::runner::Runner::graphs(runner).keys();
    keys.extend(service::lock(cache).keys()?);
    if let Some(state) = state {
        keys.extend(service::lock(state).keys()?);
    }
    Ok(keys)
}
