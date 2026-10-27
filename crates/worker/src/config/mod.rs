//! Immutable rank settings resolved from the launch descriptor.
//!
//! [`resolve`] turns the descriptor into a [`WorkerLaunch`]; its
//! [`WorkerConfig`] holds the execution settings shared by capacity planning,
//! storage and the executor's settings.

mod device;
mod launch;
pub(crate) mod placement;

pub use device::Device;
// The resolved launch configuration the launch tests read.
pub use launch::{
    AttentionOptions, DataPlane, ExpertUnion, IpcConfig, LoadOptions, ModelSource,
    RendezvousConfig, WorkerLaunch, resolve,
};

use ipc::CanvasSampling;

use crate::{Error, Result};

/// A launch descriptor or argument the rank cannot run with. The process
/// reports it as a usage error (exit status 2).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct DescriptorError(String);

impl DescriptorError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
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
    pub max_total_tokens: Option<usize>,
    /// The world the worker belongs to (`StorageLaunch::world`).
    pub world: u64,
    /// The KV cache of the predecessor world's worker of the same scope,
    /// whose segments the planes map read-only after the world's own
    /// (`LaunchDescriptor::predecessor`).
    pub predecessor: Option<ipc::PredecessorCache>,
    /// The exact KV cache segments the rank's planes are made of
    /// (`LaunchDescriptor::segments`).
    pub segments: Option<ipc::CacheTable>,
    /// Attention provider name the model is loaded with, resolved by the
    /// attention registry (`uniserve.backends.attention.resolve`).
    pub attention_backend: String,
    pub max_batch_calls: usize,
    pub max_batch_tokens: usize,
    /// Longest sequence a request may reach, in tokens (`--max-model-len`).
    pub max_model_len: usize,

    /// Maximum admitted video duration, in seconds.
    pub max_video_seconds: f64,
    /// Condition rows a video request may occupy (`video_config.max_condition_rows`);
    /// `None` provisions the largest condition set the denoiser admits.
    pub max_condition_rows: Option<u32>,
    /// Caller-selected executable for decoding reference media.
    pub ffmpeg: String,
    pub min_video_seconds: Option<f64>,
    /// Prompt-token capacities admitted by standalone denoiser layouts.
    pub video_text_capacities: Vec<usize>,
    /// Resolution classes (`768p`) a video deployment prepares, crossed with
    /// `video_aspect_ratios` resolution-major in the order the deployment
    /// names them; empty takes every class the denoiser offers.
    pub video_resolutions: Vec<String>,
    /// Display aspect ratios (`16:9`) a video deployment prepares; empty
    /// takes every ratio the denoiser offers.
    pub video_aspect_ratios: Vec<String>,
    pub deployment_components: Vec<String>,

    /// Sampling shared by all generating canvases of this deployment.
    pub canvas_sampling: Option<CanvasSampling>,
    pub max_request_pool_size: usize,
    /// Image features the encoder cache keeps resident.
    pub encoder_cache_entries: usize,
    /// Device of the generation tower when it is placed apart from `device`.
    pub generation_device: Option<Device>,
    /// Device-storage grant after loaded weights and fixed overhead, in bytes.
    pub pool_storage_bytes: Option<usize>,
    /// The bytes the rank's fraction of the device grants it once nothing
    /// else holds the device: the grant before the device's free memory
    /// bounds it, which a successor world's KV pool grows into once its
    /// predecessor is gone.
    pub pool_storage_full_bytes: Option<usize>,
    pub model_dtype: String,
    pub prefix_cache_dtype: Option<String>,
    /// Fraction of the device's total memory this process may hold for
    /// static allocations, weights and KV cache (`--mem-fraction-static`).
    pub mem_fraction_static: f64,

    pub graph_policy: String,
    pub decode_graph_batch_sizes: Vec<usize>,
    pub prefill_outputs: bool,
    pub prefill_graph_token_sizes: Vec<usize>,
    /// Image capture sizes in (height, width) pixels.
    pub flow_graph_shapes: Vec<(usize, usize)>,
}

/// Image capture sizes, in (height, width) pixels, when the launch names none.
pub(crate) const DEFAULT_FLOW_GRAPH_SHAPES: [(usize, usize); 2] = [(1152, 2048), (2048, 1152)];

/// Default decode buckets: dense small batches followed by strides of eight.
pub(crate) fn default_decode_sizes() -> Vec<usize> {
    (1..=32).chain((40..=128).step_by(8)).collect()
}

/// Default prefill buckets keep 64-token spacing through 4096, then 512.
pub(crate) fn default_prefill_sizes() -> Vec<usize> {
    (4..=32)
        .step_by(4)
        .chain((48..=256).step_by(16))
        .chain((288..=512).step_by(32))
        .chain((576..=4096).step_by(64))
        .chain((4608..=16384).step_by(512))
        .collect()
}

impl WorkerConfig {
    /// Bound request concurrency for a model that decodes resident media.
    ///
    /// Each resident media request keeps two unresolved outputs and one
    /// retiring position, so it needs three queue positions; the rank serves
    /// as many concurrent requests as the queue holds slots, at most
    /// `max_batch_calls`. One slot pipelines a request in its three
    /// positions, and requests serialize on it. Request storage is then
    /// sized by those slots instead of the token pool.
    pub(crate) fn for_media_slots(&self, queue_depth: usize) -> Result<Self> {
        let slots = self.max_batch_calls.min(queue_depth / 3);
        if slots == 0 {
            return Err(Error::Unsupported(format!(
                "resident media execution requires three queue positions per request slot; queue depth {queue_depth} holds no slot"
            )));
        }
        Ok(Self {
            max_total_tokens: None,
            world: 0,
            predecessor: None,
            segments: None,
            generation_device: None,
            max_batch_calls: slots,
            max_batch_tokens: slots,
            max_request_pool_size: slots,
            ..self.clone()
        })
    }

    /// Distinct execution devices whose resident storage belongs to this rank.
    pub(crate) fn devices(&self) -> impl Iterator<Item = Device> {
        std::iter::once(self.device).chain(
            self.generation_device
                .filter(|device| *device != self.device),
        )
    }

    /// Check what the rank alone knows or derives: its rank within its
    /// world, its request slot bounds and positive capacities. The command
    /// line and the engine check the launch values they own before a rank
    /// starts.
    pub(crate) fn validate(&self) -> Result<()> {
        let invalid = |message: &str| Error::Invalid(message.into());
        if self.world_size == 0 || self.rank >= self.world_size {
            return Err(invalid("worker configuration process rank is invalid"));
        }

        if self.block_size == Some(0)
            || [
                self.max_batch_calls,
                self.max_batch_tokens,
                self.max_model_len,
                self.max_request_pool_size,
                self.encoder_cache_entries,
            ]
            .contains(&0)
        {
            return Err(invalid("worker configuration capacities must be positive"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A media rank keeps one request slot per three queue positions and
    /// refuses a queue that holds none.
    #[test]
    fn media_slots_take_three_queue_positions_each() -> Result<()> {
        let config = crate::capacity::plan::tests::cpu_config();
        for (queue_depth, slots) in [(3, 1), (6, 2), (8, 2), (24, 8)] {
            let media = config.for_media_slots(queue_depth)?;
            assert_eq!(
                media.max_request_pool_size, slots,
                "queue depth {queue_depth}"
            );
            assert_eq!(media.max_batch_calls, slots);
            media.validate()?;
        }
        assert!(config.for_media_slots(2).is_err());

        // The batch call bound caps the slots.
        let narrow = WorkerConfig {
            max_batch_calls: 1,
            ..config.clone()
        };
        assert_eq!(narrow.for_media_slots(24)?.max_request_pool_size, 1);
        Ok(())
    }
}
