//! Immutable rank settings resolved from the launch descriptor.
//!
//! [`resolve`] turns the descriptor into a [`WorkerLaunch`]; its
//! [`WorkerConfig`] holds the execution settings shared by capacity planning,
//! storage and the runners.

mod device;
mod launch;

pub use device::{Device, generation_device};
pub use launch::{
    AttentionOptions, DataPlane, ExpertUnion, IpcConfig, LoadOptions, ModelSource,
    RendezvousConfig, WorkerLaunch, resolve,
};

use uniserve_ipc::{CanvasSampling, LaneConfig};

use crate::{Error, Result};

/// A launch descriptor or argument the rank cannot run with. The process
/// reports it as a usage error (exit status 2).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct DescriptorError(String);

impl DescriptorError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl From<Error> for DescriptorError {
    fn from(error: Error) -> Self {
        Self(error.to_string())
    }
}

/// Settings resolved before model loading and physical capacity fitting.
/// Numerical backend options remain with their backend, outside this value.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerConfig {
    /// The rank's primary device, where every binding computes.
    pub device: Device,
    pub rank: usize,
    pub world_size: usize,
    pub role: String,
    pub expert_exchange: String,
    pub expert_microbatches: usize,

    /// Tokens per cache page in the group with the widest token rows; unset until binding.
    pub block_size: Option<usize>,
    pub cache_token_capacity: Option<usize>,
    pub attention_backend: Option<String>,
    pub max_batch_calls: usize,
    pub max_batch_tokens: usize,
    pub max_sequence_tokens: usize,

    /// Maximum admitted video duration, in seconds.
    pub max_video_seconds: f64,
    /// Maximum condition rows reserved for a video request; zero omits them.
    pub max_condition_rows: usize,
    /// Caller-selected executable for decoding reference media.
    pub ffmpeg: String,
    pub min_video_seconds: Option<f64>,
    /// Prompt-token capacities admitted by standalone denoiser layouts.
    pub video_text_capacities: Vec<usize>,
    /// Canvases a video deployment prepares and admits, in the order the
    /// deployment names them; empty prepares every canvas the denoiser
    /// offers.
    pub video_canvases: Vec<uniserve_core::Canvas>,
    pub deployment_components: Vec<String>,

    /// Sampling shared by all generating canvases of this deployment.
    pub canvas_sampling: Option<CanvasSampling>,
    pub max_request_pool_size: usize,
    /// Image features the encoder cache keeps resident.
    pub encoder_cache_entries: usize,
    /// Device of the generation tower when it is placed apart from `device`.
    pub generation_device: Option<Device>,
    pub min_request_pool_size: usize,
    /// Device-storage grant after loaded weights and fixed overhead, in bytes.
    pub pool_storage_bytes: Option<usize>,
    pub model_dtype: String,
    pub prefix_cache_dtype: Option<String>,
    /// Fraction of total device storage available to this process.
    pub cache_storage_fraction: f64,
    pub lanes: Vec<LaneConfig>,

    pub graph_policy: String,
    pub decode_graph_batch_sizes: Vec<usize>,
    pub prefill_cuda_graph: bool,
    pub prefill_outputs: bool,
    pub prefill_graph_token_sizes: Vec<usize>,
    pub flow_cuda_graph: bool,
    pub flow_graph_batch_sizes: Vec<usize>,
    /// Image capture sizes in (height, width) pixels.
    pub flow_graph_shapes: Vec<(usize, usize)>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            rank: 0,
            world_size: 1,
            role: "model".into(),
            expert_exchange: "alltoall".into(),
            expert_microbatches: 1,
            block_size: None,
            cache_token_capacity: None,
            attention_backend: None,
            max_batch_calls: 1024,
            max_batch_tokens: 8192,
            max_sequence_tokens: 16384,
            max_video_seconds: 15.0,
            max_condition_rows: 0,
            ffmpeg: "ffmpeg".into(),
            min_video_seconds: None,
            video_text_capacities: vec![],
            video_canvases: vec![],
            deployment_components: vec![],
            canvas_sampling: None,
            max_request_pool_size: 128,
            encoder_cache_entries: 256,
            generation_device: None,
            min_request_pool_size: 1,
            pool_storage_bytes: None,
            model_dtype: "bfloat16".into(),
            prefix_cache_dtype: None,
            cache_storage_fraction: 0.7,
            lanes: vec![],
            graph_policy: "auto".into(),
            decode_graph_batch_sizes: default_decode_sizes(),
            prefill_cuda_graph: true,
            prefill_outputs: true,
            prefill_graph_token_sizes: default_prefill_sizes(),
            flow_cuda_graph: true,
            flow_graph_batch_sizes: vec![1, 2, 3, 4],
            flow_graph_shapes: vec![(1152, 2048), (2048, 1152)],
        }
    }
}

/// Default decode buckets: dense small batches followed by strides of eight.
pub fn default_decode_sizes() -> Vec<usize> {
    (1..=32).chain((40..=128).step_by(8)).collect()
}

/// Default prefill buckets keep 64-token spacing through 4096, then 512.
pub fn default_prefill_sizes() -> Vec<usize> {
    (4..=32)
        .step_by(4)
        .chain((48..=256).step_by(16))
        .chain((288..=512).step_by(32))
        .chain((576..=4096).step_by(64))
        .chain((4608..=16384).step_by(512))
        .collect()
}

/// A prefill bucket reserves at least one inert padding sequence.
pub const PREFILL_ROW_BUCKETS: &[usize] = &[8, 16, 32];

/// Additional cache pages that the widest default graph padding may occupy.
pub fn graph_padding_block_count(block_size: usize) -> usize {
    let decode = default_decode_sizes().into_iter().max().unwrap_or(1) - 1;
    let prefill = default_prefill_sizes()
        .into_iter()
        .scan(0, |previous, size| {
            let gap = size - *previous;
            *previous = size;
            Some(gap)
        })
        .max()
        .unwrap_or(0);
    decode.max(prefill).div_ceil(block_size.max(1))
}

impl WorkerConfig {
    /// Bound request concurrency for a model that decodes resident media.
    ///
    /// Each resident media request keeps two unresolved outputs and one
    /// retiring position, so it needs three queue positions; the rank serves
    /// as many concurrent requests as the queue holds slots, at most
    /// `max_batch_calls`, and at least two. Request storage is then sized
    /// by those slots instead of the token pool.
    pub fn for_media_slots(&self, queue_depth: usize) -> Result<Self> {
        let slots = self.max_batch_calls.min(queue_depth / 3);
        if slots < 2 {
            return Err(Error::Unsupported(format!(
                "resident media execution requires two slots of three queue positions each; queue depth {queue_depth} holds {}",
                queue_depth / 3
            )));
        }
        Ok(Self {
            cache_token_capacity: None,
            attention_backend: None,
            generation_device: None,
            max_batch_calls: slots,
            max_batch_tokens: slots,
            max_request_pool_size: slots,
            min_request_pool_size: 2,
            ..self.clone()
        })
    }

    /// Distinct execution devices whose resident storage belongs to this rank.
    pub fn devices(&self) -> impl Iterator<Item = Device> {
        std::iter::once(self.device).chain(
            self.generation_device
                .filter(|device| *device != self.device),
        )
    }

    /// Check process placement and physical resource bounds before allocation.
    pub fn validate(&self) -> Result<()> {
        let invalid = |message: &str| Error::Invalid(message.into());
        if !matches!(self.graph_policy.as_str(), "off" | "auto" | "full") {
            return Err(invalid("graph policy must be off, auto, or full"));
        }

        if !matches!(self.role.as_str(), "model" | "experts") {
            return Err(invalid("worker role must be model or experts"));
        }

        if !(1..=4).contains(&self.expert_microbatches) {
            return Err(invalid("expert microbatches must be from one to four"));
        }

        if self.world_size == 0 || self.rank >= self.world_size {
            return Err(invalid("worker configuration process rank is invalid"));
        }

        if self.min_request_pool_size == 0
            || self.min_request_pool_size > self.max_request_pool_size
        {
            return Err(invalid("worker request slot bounds are invalid"));
        }

        if self.block_size == Some(0)
            || [
                self.max_batch_calls,
                self.max_batch_tokens,
                self.max_sequence_tokens,
                self.max_request_pool_size,
                self.encoder_cache_entries,
            ]
            .contains(&0)
        {
            return Err(invalid("worker configuration capacities must be positive"));
        }

        if !self.max_video_seconds.is_finite() || self.max_video_seconds <= 0.0 {
            return Err(invalid(
                "video duration capacity must be finite and positive",
            ));
        }

        if self.ffmpeg.is_empty() {
            return Err(invalid("the media reader needs an ffmpeg path"));
        }

        if self.min_video_seconds.is_some_and(|seconds| {
            !seconds.is_finite() || seconds <= 0.0 || seconds > self.max_video_seconds
        }) {
            return Err(invalid(
                "shortest video duration must lie within the capacity",
            ));
        }

        if self.video_text_capacities.contains(&0) {
            return Err(invalid("video text capacities must be positive"));
        }

        for (index, canvas) in self.video_canvases.iter().enumerate() {
            if canvas.width == 0 || canvas.height == 0 {
                return Err(invalid("video canvases must have positive sides"));
            }
            if self.video_canvases[..index].contains(canvas) {
                return Err(invalid("video canvases must be distinct"));
            }
        }

        let mut lane_ids = std::collections::HashSet::new();
        let mut operations = std::collections::HashSet::new();
        for lane in &self.lanes {
            if !lane_ids.insert(&lane.lane_id) {
                return Err(invalid("lane ids must be unique"));
            }
            if lane.call_kinds.iter().any(|kind| !operations.insert(*kind)) {
                return Err(invalid("call kinds must have one execution lane binding"));
            }
        }

        if !(self.cache_storage_fraction > 0.0 && self.cache_storage_fraction <= 1.0) {
            return Err(invalid(
                "worker configuration KV storage fraction must be in (0, 1]",
            ));
        }

        if !matches!(
            self.model_dtype.as_str(),
            "float16" | "bfloat16" | "float32"
        ) {
            return Err(invalid("worker model dtype is unsupported"));
        }

        if self.prefix_cache_dtype.as_deref().is_some_and(|dtype| {
            !matches!(dtype, "float16" | "bfloat16" | "float32" | "float8_e4m3fn")
        }) {
            return Err(invalid("worker KV dtype is unsupported"));
        }

        Ok(())
    }
}
