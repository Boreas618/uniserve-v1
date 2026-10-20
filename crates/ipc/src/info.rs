//! Worker identity, supported calls, and startup capacity handshake.
//!
//! A rank answers the engine's `Info` request with a [`WorkerInfo`]: what only
//! the loaded rank knows (its endpoint, supported calls, capacities and
//! component outputs). The engine launched the rank and keeps what it launched
//! it with (world, devices, transfer mechanisms, queue depth and component
//! placement), so the report does not echo those back. The engine requires
//! every rank of a group to describe the group alike and a replacement rank to
//! report an identical description apart from its endpoint; the executor then
//! derives the capacity view the scheduler plans against from the
//! descriptions of all workers (`ExecutorInfo::runtime_info`).

use super::*;

/// Physical cache unit pool exposed by a worker that executes AR work.
///
/// The pool is `num_units` allocation units, unit zero being the padding
/// sentinel. Each group's logical pages draw `units_per_page` units from the
/// one pool, so the scheduler accounts every group's demand in units. Each
/// group also places this rank's share of the model's logical cache: its
/// layers by global id and its KV heads by offset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrefixCacheInfo {
    /// Allocation units in the pool, including the unit-zero sentinel.
    pub num_units: u32,
    /// Element data type of cache tensors.
    pub dtype: PrefixCacheDtype,
    /// Cache groups in table order.
    pub groups: Vec<CacheGroup>,
}

impl PrefixCacheInfo {
    /// Returns the units the scheduler may allocate: all but the sentinel.
    pub fn usable_units(&self) -> u32 {
        self.num_units.saturating_sub(1)
    }

    /// Returns the largest block size of any group, in tokens.
    ///
    /// Every block size divides it (`validate`), so a length aligned to it is
    /// a whole number of pages in every group.
    pub fn max_block_size(&self) -> u32 {
        self.groups
            .iter()
            .map(|group| group.block_size)
            .max()
            .unwrap_or(0)
    }

    /// Bounds the logical export of a `tokens`-long visible cache extent,
    /// independently of the producing TP size.
    ///
    /// An export carries each group's keys and values over the group's
    /// layers (`layer_ids`) and `total_kv_heads`; a sliding-window group
    /// carries at most `window` tokens, the history its readers need. With
    /// FP8 the bound also reserves one FP32 scale for each key and value
    /// head, layer and token, because a partial-page suffix may carry a
    /// complete scale for every head group.
    pub fn transfer_bytes(&self, tokens: u32) -> u64 {
        let width = match self.dtype {
            PrefixCacheDtype::Float16 | PrefixCacheDtype::BFloat16 => 2,
            PrefixCacheDtype::Float32 => 4,
            PrefixCacheDtype::Float8E4m3Fn => 1,
        };

        // FP8 scales are FP32.
        let scale_bytes = if self.dtype == PrefixCacheDtype::Float8E4m3Fn {
            4
        } else {
            0
        };

        self.groups
            .iter()
            .map(|group| {
                let published = group
                    .kind
                    .window()
                    .map_or(tokens, |window| tokens.min(window));
                let head_bytes = u64::from(group.head_dim) * width + scale_bytes;

                // Key and value tensors over every layer and head of the group.
                (2 * group.layer_ids.len() as u64)
                    .saturating_mul(u64::from(group.total_kv_heads))
                    .saturating_mul(head_bytes)
                    .saturating_mul(u64::from(published))
            })
            .fold(0u64, u64::saturating_add)
    }

    /// Validates the pool size, page shapes and each group's layer and head
    /// placement.
    ///
    /// Every block size must be a power of two that divides the largest one,
    /// so page-aligned prefix lengths agree across groups. A layer belongs to
    /// at most one group.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.num_units > 1 && !self.groups.is_empty(),
            "worker info declare an incomplete cache unit pool"
        );

        let largest = self.max_block_size();
        let mut layers = HashSet::new();
        for group in &self.groups {
            ensure_valid!(
                group.block_size.is_power_of_two()
                    && largest.is_multiple_of(group.block_size)
                    && group.units_per_page > 0
                    && group.units_per_page < self.num_units,
                "worker cache group page shape is invalid"
            );
            ensure_valid!(
                !group.layer_ids.is_empty()
                    && group.layer_ids.iter().all(|layer| layers.insert(*layer)),
                "worker cache groups omit or repeat a layer"
            );
            ensure_valid!(
                group.num_kv_heads > 0
                    && u64::from(group.kv_head_offset) + u64::from(group.num_kv_heads)
                        <= u64::from(group.total_kv_heads)
                    && group.head_dim > 0,
                "worker cache group head interval exceeds its logical bounds"
            );
        }
        Ok(())
    }

    /// Combines the pools of two cache stages that one request may traverse.
    ///
    /// Stages must agree on dtype and on every group's retention policy and
    /// page shape; their rank-local layer ids and head placement may differ.
    /// The combined pool has the smaller unit count, because the scheduler
    /// assigns one set of unit ids valid in both, and per group the union of both stages' layers, which bounds exports.
    pub fn merge(&self, other: &Self) -> ValidationResult<Self> {
        ensure_valid!(
            self.dtype == other.dtype
                && self.groups.len() == other.groups.len()
                && self
                    .groups
                    .iter()
                    .zip(&other.groups)
                    .all(|(left, right)| left.same_page_shape(right)),
            "cache stages expose incompatible cache layouts"
        );

        let groups = self
            .groups
            .iter()
            .zip(&other.groups)
            .map(|(left, right)| {
                let mut layer_ids = left.layer_ids.clone();
                layer_ids.extend(
                    right
                        .layer_ids
                        .iter()
                        .filter(|layer| !left.layer_ids.contains(layer)),
                );
                layer_ids.sort_unstable();
                CacheGroup {
                    layer_ids,
                    ..left.clone()
                }
            })
            .collect();
        let merged = Self {
            num_units: self.num_units.min(other.num_units),
            dtype: self.dtype,
            groups,
        };
        merged.validate()?;
        Ok(merged)
    }
}

/// The tensor results one placed component declares.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentInfo {
    /// Component name; `WorkerInfo::validate` requires it to be non-empty and
    /// unique within one worker.
    pub name: String,
    /// Named tensor results declared by the loaded computation. Their order
    /// defines product output indices independently of Worker grouping.
    pub outputs: Vec<OutputInfo>,
}

/// A computation's bounded tensor result, before request and storage binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputInfo {
    /// Result name; `OutputInfo::validate` requires it to be non-empty and
    /// `WorkerInfo::validate` unique within its component.
    pub name: String,
    /// Element type of the result tensor.
    pub dtype: DType,
    /// Upper bound on the result's shape, used to size storage before the
    /// actual shape is known.
    pub shape_bound: ShapeBound,
    /// The axes holding a video request's raster, for a result laid out at
    /// the request's canvas. Their extents in `shape_bound` are the largest
    /// height and width the worker admits; a request binds them to its own.
    pub raster_axes: Option<RasterAxes>,
}

/// The two axes of a result that hold a video request's raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RasterAxes {
    /// Axis holding the canvas height in pixels.
    pub height: u32,
    /// Axis holding the canvas width in pixels.
    pub width: u32,
}

impl OutputInfo {
    /// Validate the logical representation advertised to allocation and routing.
    ///
    /// Raster axes must be two distinct static axes of the bound.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(!self.name.is_empty(), "tensor result must have a name");
        self.shape_bound.validate()?;
        if let Some(raster) = self.raster_axes {
            let is_static = |axis: u32| {
                matches!(
                    self.shape_bound.dims.get(axis as usize),
                    Some(DimBound::Static(_))
                )
            };
            ensure_valid!(
                raster.height != raster.width
                    && is_static(raster.height)
                    && is_static(raster.width),
                "a tensor result's raster axes must be two distinct static axes"
            );
        }
        Ok(())
    }
}

/// A loaded rank incarnation in an explicitly identified host address space.
/// Backend addresses and storage generations are carried by each export.
///
/// The endpoint names the incarnation alone: the engine launched the
/// process as one worker's rank and resolves an incarnation to that worker
/// and rank itself, so a rank does not report them.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerEndpoint {
    /// Host identity, independent of process-local device ordinals.
    pub node: String,
    /// Process lifetime identity; multiple Workers may share it.
    pub address_space: String,
    /// Lifetime identity of this loaded Worker rank.
    pub incarnation: String,
}

impl WorkerEndpoint {
    /// Checks identities before accepting startup metadata or an export.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            !self.node.is_empty() && !self.address_space.is_empty() && !self.incarnation.is_empty(),
            "worker endpoint identity is incomplete"
        );
        Ok(())
    }
}

/// What the video denoiser a deployment places serves.
///
/// The denoiser fixes its schedule, and with it every request's step count;
/// the tasks, canvases and reference limits bound what a request may ask
/// for.
/// The worker that denoises a request admits its conditions into the
/// condition capacity it provisioned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoDenoiserInfo {
    /// Task names the denoiser serves, in canonical order.
    pub tasks: Vec<String>,
    /// Sigma points of the fixed schedule, the clean endpoint included: one
    /// more than the network evaluations ([`Self::denoise_steps`]).
    pub schedule_points: u32,
    /// Shift of the video schedule.
    pub video_shift: f64,
    /// Shift of the audio schedule.
    pub audio_shift: f64,
    /// Frame rate, in frames per second, of every video the deployment
    /// generates: the output clock durations and reference videos are
    /// counted on.
    pub frame_rate: u32,
    /// Sample rate, in Hz, of the audio track generated with each video.
    pub audio_sample_rate: u32,
    /// How the VAE divides a video's frames into clips, which counts the
    /// frames the deployment generates and their decode units.
    pub frames: uniserve_core::FrameClips,
    /// The only canvases the deployment serves: those it prepares, each
    /// with the `(short_edge, aspect_ratio)` its model names it by, in the
    /// order the deployment names them.
    pub canvases: Vec<uniserve_core::ServedCanvas>,
    /// The references one request may carry and the largest short edge an
    /// image reference is encoded at; zeros for a denoiser that takes none.
    pub references: uniserve_core::ReferenceLimits,
}

impl VideoDenoiserInfo {
    /// The network evaluations of the fixed schedule: one fewer than its
    /// sigma points, and the step count of every request.
    pub fn denoise_steps(&self) -> u32 {
        self.schedule_points.saturating_sub(1)
    }

    /// Checks that tasks are named once each, the schedule has an endpoint
    /// and positive finite shifts, the frame and audio sample rates and the
    /// frame rule are declared, at least one canvas is prepared and each is a
    /// positive raster named by its own `(short_edge, aspect_ratio)` pair,
    /// and the reference limits
    /// bound each kind within the total, with an image edge where images
    /// are taken.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            !self.tasks.is_empty()
                && self.tasks.iter().all(|task| !task.is_empty())
                && self.tasks.iter().collect::<HashSet<_>>().len() == self.tasks.len(),
            "a video denoiser names each of its tasks once"
        );
        ensure_valid!(
            self.schedule_points >= 2
                && [self.video_shift, self.audio_shift]
                    .iter()
                    .all(|shift| shift.is_finite() && *shift > 0.0),
            "a video denoiser schedule needs points and positive shifts"
        );
        ensure_valid!(
            self.frame_rate > 0,
            "a video denoiser declares no frame rate"
        );
        ensure_valid!(
            self.audio_sample_rate > 0,
            "a video denoiser declares no audio sample rate"
        );
        ensure_valid!(
            self.frames.is_valid(),
            "a video denoiser declares an empty frame rule"
        );
        let references = &self.references;
        ensure_valid!(
            references
                .images
                .max(references.videos)
                .max(references.audios)
                <= references.total
                && (references.images == 0 || references.image_short_edge > 0),
            "a video denoiser declares reference limits beyond their total or images without an edge"
        );
        ensure_valid!(
            !self.canvases.is_empty(),
            "a video denoiser prepared no canvas"
        );
        ensure_valid!(
            self.canvases.iter().all(|served| {
                served.short_edge > 0
                    && !served.aspect_ratio.is_empty()
                    && served.canvas.width > 0
                    && served.canvas.height > 0
            }),
            "a video denoiser declares an empty canvas"
        );
        ensure_valid!(
            self.canvases
                .iter()
                .map(|served| (served.short_edge, served.aspect_ratio.as_str()))
                .collect::<HashSet<_>>()
                .len()
                == self.canvases.len(),
            "a video denoiser declares a canvas name twice"
        );
        Ok(())
    }
}

/// An image shape a rank's image denoiser was prepared for: a raster and
/// the guidance branches its captured step evaluates.
///
/// The engine runs one image request of this shape through its scheduler
/// before it serves (`warmup`), at guidance scales that realize the branch
/// count (`ImageParams::guidance_scales`), so the serving path of every
/// captured branch count has run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageWarmup {
    /// Output height in pixels.
    pub height: u32,
    /// Output width in pixels.
    pub width: u32,
    /// Guidance branches one image evaluates per step, one to three.
    pub branches: u8,
    /// The model's framing of an empty negative prompt, which the warmup
    /// request carries as its negative conditioning tokens, as the server
    /// renders a client's; empty for one branch, which has no alternative
    /// prefix, and for a model without prompt framing.
    pub negative_token_ids: Vec<u32>,
}

impl ImageWarmup {
    /// Checks that the shape names a raster and one to three branches, and
    /// carries an alternative prefix only for guided generation.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.height > 0 && self.width > 0,
            "an image warmup shape is empty"
        );
        ensure_valid!(
            (1..=3).contains(&self.branches)
                && (self.branches > 1 || self.negative_token_ids.is_empty()),
            "an image warmup shape has no valid guidance branches"
        );
        Ok(())
    }
}

/// Post-load worker geometry, limits and supported work.
///
/// [`WorkerInfo::validate`] checks only internal consistency; the rank runs it
/// on the report it builds, and the engine takes the report as sent. Launch
/// values
/// (world size, devices, transfer mechanisms, queue depth and component
/// placement) are the engine's and are not reported.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Model-declared finite terminal-media computation, when supported.
    pub media_components: std::collections::BTreeMap<MediaCall, String>,
    /// What the deployment's video denoiser serves, when the model has one.
    pub video_denoiser: Option<VideoDenoiserInfo>,
    /// Image shapes this rank's image denoiser was prepared for, one per
    /// captured guidance-branch count, each of which the engine warms with a
    /// real request before serving.
    pub image_warmups: Vec<ImageWarmup>,
    /// Blocks one sampler chunk of this rank's diffusion LM denoising step
    /// covers, every row count of which the engine warms; zero when the rank
    /// generates no blocks.
    pub diffusion_lm_step_rows: u32,
    /// The tokens the image-unconditional guidance branch of this rank's
    /// image denoiser attends to, the model's framing of that branch's
    /// source, the same for every request; the engine reserves and prefills
    /// a guidance prefix of them. `None` when that branch attends to the
    /// request's own context, and on a rank without an image denoiser.
    pub image_unconditional_token_ids: Option<Vec<u32>>,
    /// Identity of the loaded rank and its host address space.
    pub endpoint: WorkerEndpoint,
    /// Whether this rank's device exports a fabric allocation handle.
    ///
    /// A fabric handle is importable from another host inside the fabric
    /// domain; a descriptor handle reaches only the host that created it, so
    /// this is what decides whether a transfer edge may cross hosts.
    pub fabric_handles: bool,
    /// The tensor results of each component the rank places.
    pub components: Vec<ComponentInfo>,
    /// Call families accepted by the worker.
    pub supported_calls: Vec<CallKind>,
    /// Maximum calls in one batch. A prefill or decode batch that no captured
    /// graph holds runs eagerly, so the captured graphs bound no batch.
    pub max_batch_calls: u32,
    /// Maximum text tokens represented in one batch.
    pub max_batch_tokens: u32,
    /// Number of resident request slots.
    pub request_slots: u32,
    /// Paged cache geometry when autoregressive work is supported.
    pub prefix_cache: Option<PrefixCacheInfo>,
    /// Model-defined units stored in one latent page.
    pub latent_page_units: u32,
    /// Physical latent pages including the reserved sentinel page, which
    /// [`WorkerInfo::latent_capacity_units`] excludes.
    pub latent_pages: u32,
    /// Bytes of the persistent buffer pool this rank backs, which the engine
    /// places this rank's buffers in: on a media rank the products of the
    /// components it executes and of the components that read them.
    pub buffer_pool_bytes: u64,
    /// Maximum simultaneously retained encoder feature products.
    pub encoder_cache_entries: u32,
    /// Maximum bytes in one encoder feature product, independent of pool size.
    pub encoder_entry_bytes: u64,
    /// Maximum unresolved calls per request.
    pub max_unresolved_calls: u32,
    /// Concurrent host-lane tasks this rank's bounded host executor admits.
    /// The scheduler treats zero as one.
    pub host_lane_capacity: u32,
    /// Transfer entries this rank's transport holds in flight at once. The
    /// scheduler keeps no more transfer calls reserved than the narrowest
    /// rank holds.
    pub transfer_tickets: u32,
}

impl WorkerInfo {
    /// Returns total latent capacity in model-defined units.
    pub fn latent_capacity_units(&self) -> u64 {
        u64::from(self.latent_pages.saturating_sub(1))
            .saturating_mul(u64::from(self.latent_page_units))
    }

    /// Returns whether the worker advertises paged cache capacity.
    pub fn uses_cache(&self) -> bool {
        self.prefix_cache.is_some()
    }

    /// Validates worker identity, capacity and call invariants.
    pub fn validate(&self) -> ValidationResult<()> {
        self.endpoint.validate()?;

        // Components: unique names, each with unique valid outputs.
        let mut names = HashSet::new();
        for component in &self.components {
            ensure_valid!(
                !component.name.is_empty() && names.insert(&component.name),
                "worker repeats or omits a component name"
            );
            let mut outputs = HashSet::new();
            for output in &component.outputs {
                output.validate()?;
                ensure_valid!(
                    outputs.insert(&output.name),
                    "component repeats a tensor result name"
                );
            }
        }

        // A collective-only participant owns no request components or prefix cache.
        // It remains addressable for progress, failure and shutdown.
        ensure_valid!(
            !self.supported_calls.is_empty()
                || (self.components.is_empty()
                    && self.prefix_cache.is_none()
                    && self.latent_pages == 0
                    && self.media_components.is_empty()),
            "a worker without call kinds must own no request call"
        );
        ensure_valid!(
            self.supported_calls
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                == self.supported_calls.len(),
            "worker info repeats a call kind"
        );
        // A worker names the component serving each media call it implements.
        // Here every named call must be one it advertises, with a non-empty
        // component name. The video graph may span workers, a model worker
        // decoding and a host worker encoding and muxing, so its completeness
        // and its diffusion step count are the executor's checks over every
        // worker.
        ensure_valid!(
            self.media_components.iter().all(|(call, component)| {
                !component.is_empty() && self.supported_calls.contains(&CallKind::Media(*call))
            }),
            "a media call names no component this worker serves"
        );
        for shape in &self.image_warmups {
            shape.validate()?;
        }
        if let Some(video) = &self.video_denoiser {
            video.validate()?;
        }
        ensure_valid!(
            self.max_batch_calls > 0
                && self.max_batch_tokens > 0
                && self.request_slots > 0
                && self.max_unresolved_calls > 0
                && self.transfer_tickets > 0,
            "worker info declare a zero scheduling bound"
        );

        // Advertised call families require their corresponding pools; every
        // token-model forward reads or extends a request's prefix cache.
        let requires_cache = self
            .supported_calls
            .iter()
            .any(|variant| matches!(variant, CallKind::Forward(_)));
        ensure_valid!(
            !requires_cache || self.prefix_cache.is_some(),
            "worker advertises token work without a prefix cache"
        );
        if let Some(prefix_cache) = &self.prefix_cache {
            prefix_cache.validate()?;
        }
        // Latent capacity is either absent or at least one usable page beyond
        // the sentinel.
        let has_latent_capacity = self.latent_page_units > 0 || self.latent_pages > 0;
        if has_latent_capacity {
            ensure_valid!(
                self.latent_page_units > 0 && self.latent_pages > 1,
                "worker info declare incomplete latent pool capacity"
            );
        }
        Ok(())
    }
}

impl Default for WorkerInfo {
    /// Returns a valid single-rank autoregressive worker description.
    fn default() -> Self {
        Self {
            media_components: Default::default(),
            video_denoiser: None,
            image_warmups: Vec::new(),
            diffusion_lm_step_rows: 0,
            image_unconditional_token_ids: None,
            fabric_handles: false,
            endpoint: WorkerEndpoint {
                node: "simulation".into(),
                address_space: "simulation".into(),
                incarnation: "simulation".into(),
            },
            components: Vec::new(),
            supported_calls: vec![
                CallKind::Forward(ForwardMode::Prefill),
                CallKind::Forward(ForwardMode::Decode),
            ],
            max_batch_calls: 1,
            max_batch_tokens: 8192,
            request_slots: 128,
            // One full-attention group of 28 layers with 8 BF16 heads of 128,
            // one unit per 64-token page.
            prefix_cache: Some(PrefixCacheInfo {
                num_units: 4096,
                dtype: PrefixCacheDtype::BFloat16,
                groups: vec![CacheGroup {
                    kind: Default::default(),
                    block_size: 64,
                    units_per_page: 1,
                    layer_ids: (0..28).collect(),
                    num_kv_heads: 8,
                    total_kv_heads: 8,
                    kv_head_offset: 0,
                    head_dim: 128,
                }],
            }),
            latent_page_units: 0,
            latent_pages: 0,
            buffer_pool_bytes: 0,
            encoder_cache_entries: 0,
            encoder_entry_bytes: 0,
            max_unresolved_calls: 1,
            host_lane_capacity: 1,
            transfer_tickets: 1,
        }
    }
}
