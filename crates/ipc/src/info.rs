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
    /// The host page pools this rank copies its cache groups' pages
    /// through, one per group; empty without a host KV cache.
    pub page_pools: Vec<PagePoolInfo>,
    /// Units `1..=scratch_units` the rank's warmup lends to its startup
    /// forwards. A standby worker warms up on them while its predecessor
    /// serves, so the engine never allocates them.
    pub scratch_units: u32,
    /// The segments the rank's value planes are made of, in unit order, each
    /// of whole units that give storage back and are created again whole;
    /// empty for planes not split.
    #[serde(default)]
    pub segments: Vec<CacheSegment>,
    /// The segments the planes hold, `[0, held_segments)`: the engine
    /// allocates no unit of the segments after, whose storage was given back.
    #[serde(default)]
    pub held_segments: u32,
    /// The units the planes map read-only from the predecessor world's
    /// planes, `[inherited_start, inherited_start + inherited_units)`: they
    /// hold the predecessor's pages, which the engine reads once its index
    /// is imported and allocates only once the predecessor is gone and the
    /// rank took its storage; zero for planes of their own.
    #[serde(default)]
    pub inherited_start: u32,
    #[serde(default)]
    pub inherited_units: u32,
    /// Upstream numerical component identities of each cache group, in
    /// group order. Each string names its resident layers and their loading
    /// inputs. Pipeline stages union these components; an unrelated group
    /// or a downstream output head changes none of a group's keys.
    #[serde(default)]
    pub identities: Vec<String>,
    /// The bytes one token of the rank's pages takes in the host KV cache,
    /// over every cache group (a group's page bytes over its block size,
    /// rounded up); zero for a rank that copies no page through it. The
    /// engine sizes the world's host KV cache in tokens from them
    /// (`uniserve.host_cache`).
    #[serde(default)]
    pub host_token_bytes: u64,
    /// Where an engine on another host reaches the rank's host's Storage
    /// Server (`StorageLaunch::tcp`); `None` on the engine's host.
    #[serde(default)]
    pub storage_endpoint: Option<String>,
}

impl PrefixCacheInfo {
    /// One stable prefix-chain root per cache group. Empty declarations
    /// denote unkeyed pools of protocol consumers without models.
    pub fn seeds(&self) -> Vec<u64> {
        use sha2::{Digest, Sha256};
        (0..self.groups.len())
            .map(|group| {
                self.identities
                    .get(group)
                    .filter(|value| !value.is_empty())
                    .map_or(0, |identity| {
                        let digest = Sha256::digest(identity.as_bytes());
                        let mut seed = [0; 8];
                        seed.copy_from_slice(&digest[..8]);
                        u64::from_le_bytes(seed)
                    })
            })
            .collect()
    }

    /// The units below the first segment given back: every unit of planes
    /// not split.
    pub fn allocatable_units(&self) -> u32 {
        if self.segments.is_empty() {
            return self.num_units;
        }
        self.segments
            .iter()
            .take(self.held_segments as usize)
            .map(|segment| segment.units)
            .sum::<u32>()
            .min(self.num_units)
    }

    /// The units the planes map read-only from the predecessor's planes.
    pub fn inherited(&self) -> std::ops::Range<u32> {
        self.inherited_start..self.inherited_start + self.inherited_units
    }
}

/// One segment of a rank's KV cache planes: an allocation of whole units on
/// every plane, which a world created and the Storage Server retains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheSegment {
    /// The world that created the segment's storage.
    pub world: u64,
    /// The segment's ordinal among those its world created.
    pub ordinal: u32,
    /// The units the segment holds.
    pub units: u32,
}

/// The exact segments a rank's KV cache planes are made of, and how many of
/// the leading ones exist: a relaunched rank of the same world, or a
/// standby rank group's, lays its planes out alike and takes them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheTable {
    pub segments: Vec<CacheSegment>,
    /// The segments `[0, held)` that exist; the others were given back.
    pub held: u32,
}

/// What a rank of a world that succeeds another maps of the predecessor's
/// KV cache: the predecessor's world, and the segments its rank of the same
/// scope holds, in unit order, which the rank maps read-only after a head
/// segment of its own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredecessorCache {
    pub world: u64,
    /// The segments the predecessor's planes hold.
    pub segments: Vec<CacheSegment>,
    /// The units of the predecessor's planes, those of the segments it gave
    /// back included: its flags and scales are laid out over them.
    pub num_units: u32,
    /// The units `1..=scratch_units` the predecessor's warmup lent, which a
    /// successor of the same configuration needs as many of.
    pub scratch_units: u32,
}

/// The host page pool of one cache group on one rank: the Storage Server
/// pool its pages are copied to and from, and what the pages hold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagePoolInfo {
    /// The cache group whose pages the pool holds.
    pub group_id: u32,
    /// The pool's identity at the rank's host's Storage Server.
    pub pool: u64,
    /// The digest of what determines the group's page encoding (the model
    /// fingerprint, the page encoding version, the cache dtype, the block
    /// size and the cache group list), which seeds the group's page keys;
    /// equal on every rank of a group.
    pub content: [u8; 32],
    /// Where an engine on another host reaches the pool's Storage Server
    /// (`StorageLaunch::tcp`).
    pub endpoint: Option<String>,
}

/// The host latent pool of a denoising rank (a video model's, or an image
/// model's): the Storage Server page pool its preempted trajectories are
/// written to, one pool page per page of its latent banks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentPoolInfo {
    /// The pool, as any rank of the deployment on the host maps it.
    pub pool: crate::LatentPoolRef,
    /// The most pages one trajectory occupies in the rank's latent banks,
    /// each a pool page: a video request slot's pages, or the pages of an
    /// image model's longest image.
    pub trajectory_pages: u32,
    /// The digest of what determines the rank's share of every request's
    /// samples: for a video denoiser, the video plan that selects a
    /// request's layout, the slot's pages, the rank's coordinate in the
    /// denoiser's mesh and its rows of every layout startup prepares; for
    /// an image denoiser, whose every rank holds whole latents, the page
    /// encoding alone. A rank of another world with the same share holds
    /// what this rank held of any trajectory.
    pub region: [u8; 32],
    /// The identity of what the rank's trajectories are computed with: the
    /// weights' values and the declared function and format identities of
    /// the model it loaded (`uniserve.numerics`), as its KV cache pages are
    /// keyed.
    pub identity: String,
    /// Where an engine on another host reaches the pool's Storage Server
    /// (`StorageLaunch::tcp`).
    pub endpoint: Option<String>,
}

/// The host encoder pool of a rank that executes encoder calls (a video
/// model's encoders, an image model's vision and latent encoders): the
/// Storage Server page pool, shared by the deployment's ranks on its host in
/// every world of the deployment, that holds encoder outputs keyed by what
/// the encoder read and what computes it (`Call::host_output`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderPoolInfo {
    /// The pool's identity at the rank's host's Storage Server.
    pub pool: u64,
    /// The bytes of one pool page; an output takes as many pages as its
    /// bytes fill.
    pub page_bytes: u64,
    /// The identity of what computes each encoder output the rank writes,
    /// by the product's name: the weights' values and the declared
    /// function and format identities of the modules computing it
    /// (`uniserve.numerics`).
    pub identities: std::collections::BTreeMap<String, String>,
    /// Where an engine on another host reaches the pool's Storage Server
    /// (`StorageLaunch::tcp`).
    pub endpoint: Option<String>,
}

/// What an image encoder writes for one image (`WorkerInfo::image_encoders`):
/// one feature row per cache token of the block the features write, at
/// most.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageEncoderOutput {
    /// The bytes of one feature row.
    pub row_bytes: u64,
    /// Whether the features follow the request's image seed and the
    /// encoding's posterior draw (`Call::latent_draw`) besides the image:
    /// a latent encoder that samples its posterior rather than taking its
    /// mean.
    pub draws: bool,
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

        // Exports carry the K/V pages a request needs alone: a worker
        // refuses to export the cache of a model that keeps state per
        // request slot, and the draft cache stays with its rank.
        self.groups
            .iter()
            .filter(|group| !group.kind.is_state() && !group.kind.is_draft())
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
        ensure_valid!(
            self.identities.is_empty() || self.identities.len() == self.groups.len(),
            "worker cache identities must cover every group"
        );
        ensure_valid!(
            self.scratch_units < self.num_units - 1,
            "worker scratch units leave no cache unit to allocate"
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
            // Each stage holds other layers of a page: its pages are its own
            // content, which the merged view does not name.
            page_pools: Vec::new(),
            // A unit the warmup of any stage lends stays out of the pool.
            scratch_units: self.scratch_units.max(other.scratch_units),
            // The stages compute the pages together: every stage's identity
            // roots them.
            identities: (0..self.groups.len())
                .map(|group| {
                    let mut components = std::collections::BTreeSet::new();
                    for pool in [self, other] {
                        if let Some(identity) = pool.identities.get(group) {
                            components.extend(identity.split('|'));
                        }
                    }
                    components.into_iter().collect::<Vec<_>>().join("|")
                })
                .collect(),
            // Stages whose planes split alike give storage back together;
            // planes split otherwise hold every unit and give back none.
            segments: if self.segments == other.segments {
                self.segments.clone()
            } else {
                Vec::new()
            },
            held_segments: if self.segments == other.segments {
                self.held_segments.min(other.held_segments)
            } else {
                0
            },
            // The stages of one world inherit the same predecessor units.
            inherited_start: self.inherited_start,
            inherited_units: self.inherited_units,
            // Each stage's pages are its own (`page_pools`); the merged view
            // copies none.
            host_token_bytes: 0,
            storage_endpoint: None,
        };
        ensure_valid!(
            self.inherited() == other.inherited(),
            "the stages of one worker inherit different units of their predecessor"
        );
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
    pub host: String,
    /// Process lifetime identity; multiple Workers may share it.
    pub address_space: String,
    /// Lifetime identity of this loaded Worker rank.
    pub incarnation: String,
}

impl WorkerEndpoint {
    /// Checks identities before accepting startup metadata or an export.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            !self.host.is_empty() && !self.address_space.is_empty() && !self.incarnation.is_empty(),
            "worker endpoint identity is incomplete"
        );
        Ok(())
    }
}

/// What the video denoiser a deployment places serves, as far as the engine
/// schedules it.
///
/// The denoiser fixes its schedule, and with it every request's step count.
/// What else bounds a request (its canvases, frame counts and reference
/// limits) is the model family's, and reaches the frontend in `description`
/// for it to plan requests with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoDenoiserInfo {
    /// The tasks the denoiser serves, in canonical order.
    pub tasks: Vec<VideoTaskInfo>,
    /// The network evaluations of the fixed schedule: the step count of
    /// every request.
    pub denoise_steps: u32,
    /// The denoiser as its model family describes it to the frontend
    /// (`models::video::DenoiserInfo`, JSON); the engine reads none of it.
    pub description: Vec<u8>,
}

/// One task a video denoiser serves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoTaskInfo {
    /// The task's request name.
    pub name: String,
    /// Whether the task's requests carry conditions, which a host component
    /// reads and the condition encoders encode before denoising.
    pub conditioned: bool,
}

impl VideoDenoiserInfo {
    /// The task named `name`, when the denoiser serves it.
    pub fn task(&self, name: &str) -> Option<&VideoTaskInfo> {
        self.tasks.iter().find(|task| task.name == name)
    }

    /// Checks that tasks are named once each, the schedule takes a step and
    /// the model family described the denoiser.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            !self.tasks.is_empty()
                && self.tasks.iter().all(|task| !task.name.is_empty())
                && self
                    .tasks
                    .iter()
                    .map(|task| &task.name)
                    .collect::<HashSet<_>>()
                    .len()
                    == self.tasks.len(),
            "a video denoiser names each of its tasks once"
        );
        ensure_valid!(
            self.denoise_steps > 0,
            "a video denoiser schedule takes no step"
        );
        ensure_valid!(
            !self.description.is_empty(),
            "a video denoiser carries no description"
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

/// The extents an image model's forwards serve on a rank: its captured
/// buckets' largest live rows and tokens, the same under graphs and in eager
/// execution, which bound what the engine admits. A forward the rank does
/// not serve has zero tokens and rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageCapacity {
    /// Tokens one denoising step evaluates at most: every guidance branch's
    /// framed latent sequence of every stepping image.
    pub max_step_tokens: u64,
    /// Guidance branch rows one denoising step evaluates at most.
    pub max_step_rows: u32,
    /// Tokens of latent images one text batch writes as conditioning at
    /// most.
    pub max_condition_tokens: u64,
    /// Latent images one text batch writes as conditioning at most.
    pub max_condition_rows: u32,
    /// Patch rows one vision encoding packs at most.
    pub max_vision_patch_rows: u64,
    /// Images one vision encoding packs at most.
    pub max_vision_images: u32,
}

impl ImageCapacity {
    /// Checks that each forward's tokens and rows are both zero, a forward
    /// the rank does not serve, or both positive.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            (self.max_step_tokens == 0) == (self.max_step_rows == 0)
                && (self.max_condition_tokens == 0) == (self.max_condition_rows == 0)
                && (self.max_vision_patch_rows == 0) == (self.max_vision_images == 0),
            "an image capacity states tokens without rows or rows without tokens"
        );
        Ok(())
    }
}

/// An image denoising step a rank captured: one image of `height` x
/// `width` pixels evaluating `branches` guidance branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CapturedImage {
    /// Output height in pixels.
    pub height: u32,
    /// Output width in pixels.
    pub width: u32,
    /// Guidance branches the step evaluates, one to three.
    pub branches: u8,
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
    /// The image denoising steps this rank captured, which alone it
    /// denoises under CUDA graphs: the engine admits an image request only
    /// when its size is captured at every guidance branch count up to its
    /// own. `None` for a rank that denoises eagerly (graph policy `off`, a
    /// CPU rank) or holds no image denoiser.
    #[serde(default)]
    pub captured_images: Option<Vec<CapturedImage>>,
    /// The `[height, width]` image sizes this rank's latent encoder
    /// captured, which alone it encodes under CUDA graphs: the engine admits
    /// an input image's latent block only at one of them. `None` for a rank
    /// that encodes eagerly or holds no latent encoder.
    #[serde(default)]
    pub captured_latents: Option<Vec<[u32; 2]>>,
    /// The extents this rank's image forwards serve; the engine admits an
    /// image request only within them. `None` on a rank without an image
    /// model's forwards.
    #[serde(default)]
    pub image_capacity: Option<ImageCapacity>,
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
    /// Maximum calls in one batch. Under CUDA graphs the rank captured a text
    /// graph for every prefill and decode batch within these bounds.
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
    /// The host latent pool of a video denoising rank whose host runs a
    /// Storage Server; `None` otherwise, and in the group's aggregate.
    /// Boxed, as it is larger than most of the report.
    pub latent_pool: Option<Box<LatentPoolInfo>>,
    /// The host encoder pool of a rank that executes encoder calls and
    /// whose host runs a Storage Server; `None` otherwise, and in the
    /// group's aggregate. Boxed, as it is larger than most of the report.
    pub encoder_pool: Option<Box<EncoderPoolInfo>>,
    /// What each image encoder of this rank writes for one image, by the
    /// name of its output (`vision_features`, `latent_features`); empty on a
    /// rank without one. Boxed, as the report is larger than the other
    /// responses a worker sends.
    #[serde(default)]
    pub image_encoders: Box<std::collections::BTreeMap<String, ImageEncoderOutput>>,
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
    /// Concurrent tasks this rank's bounded host task pool admits. The
    /// scheduler treats zero as one.
    pub host_task_capacity: u32,
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
        if let Some(capacity) = &self.image_capacity {
            capacity.validate()?;
        }
        ensure_valid!(
            self.captured_images.iter().flatten().all(|shape| {
                shape.height > 0 && shape.width > 0 && (1..=3).contains(&shape.branches)
            }),
            "a captured image step is empty or has no valid guidance branches"
        );
        ensure_valid!(
            self.captured_latents
                .iter()
                .flatten()
                .all(|&[height, width]| height > 0 && width > 0),
            "a captured latent image size is empty"
        );
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
            captured_images: None,
            captured_latents: None,
            image_capacity: None,
            diffusion_lm_step_rows: 0,
            image_unconditional_token_ids: None,
            fabric_handles: false,
            endpoint: WorkerEndpoint {
                host: "simulation".into(),
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
                page_pools: Vec::new(),
                scratch_units: 0,
                segments: Vec::new(),
                held_segments: 0,
                inherited_start: 0,
                inherited_units: 0,
                identities: Vec::new(),
                host_token_bytes: 0,
                storage_endpoint: None,
            }),
            latent_page_units: 0,
            latent_pages: 0,
            latent_pool: None,
            encoder_pool: None,
            image_encoders: Box::default(),
            buffer_pool_bytes: 0,
            encoder_cache_entries: 0,
            encoder_entry_bytes: 0,
            max_unresolved_calls: 1,
            host_task_capacity: 1,
            transfer_tickets: 1,
        }
    }
}
