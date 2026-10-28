//! Capacity of a video model's ranks.
//!
//! A video rank keeps every resident request's media state whole: the
//! denoiser's slot state and latent pages, the decoder's retained overlap
//! and the products the rank publishes or reads. Its request slots are
//! therefore sized by storage (each candidate count's state, products,
//! product store and latent pool must fit the device grant on every rank of
//! the group), and the engine schedules against those slots instead of a
//! cache pool. The per-request quantities are the executor's
//! ([`VideoManifest`]); this module plans from them and from the products
//! every component declares.

use std::collections::{BTreeMap, BTreeSet};

use ipc::{MediaCall, OutputInfo, VideoDenoiserInfo, VideoTaskInfo};
use serde::Deserialize;

use super::plan::{MediaFacts, product_bytes};
use crate::memory::LatentPlan;
use crate::{Error, Result};

/// A video denoiser's latent pages of one request slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub(crate) struct SamplePages {
    /// Elements of one page.
    pub(crate) page_units: u64,
    /// Pages a request slot owns.
    pub(crate) pages: u64,
}

impl SamplePages {
    /// Pages of a pool of `slots` slots, the padding page included.
    pub(crate) fn pool_pages(&self, slots: u64) -> u64 {
        slots * self.pages + 1
    }
}

/// What the executor states of a video model's rank
/// (`uniserve.executor.Executor.manifest`).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VideoManifest {
    /// The products every component of the deployment publishes, by
    /// component: the model's own, the video codec's encoded unit rows and,
    /// for conditioned tasks, the media reader's condition media. Every rank
    /// plans them all, whichever components it holds.
    pub(crate) outputs: BTreeMap<String, Vec<OutputInfo>>,
    /// The denoiser as the frontend reads it.
    pub(crate) denoiser: models::video::DenoiserInfo,
    /// Each request slot's latent pages.
    pub(crate) pages: SamplePages,
    /// Device bytes of one request slot's state on this rank: the
    /// denoiser's slot state and the decoder's retained overlap.
    pub(crate) state_bytes: u64,
    /// Whether this rank holds the denoiser and so stores the latent pool.
    pub(crate) stores_latent: bool,
    /// On a denoising rank, the digest of what its latent pages mean
    /// besides the denoiser's numerics: the video plan that selects a
    /// request's layout and the rank's coordinates in the denoiser's mesh,
    /// which keys the trajectories it stores in its host latent pool.
    #[serde(default)]
    pub(crate) latent_region: Option<String>,
    /// What computes each encoder output this rank writes, by the
    /// product's name: the modules' declared function and format
    /// identities, which key the output in the host encoder pool with the
    /// weights' values.
    #[serde(default)]
    pub(crate) encoder_identities: BTreeMap<String, String>,
    /// Media units of the longest admitted video.
    pub(crate) units: usize,
}

/// What a rank's video components need beside its products.
pub(crate) struct VideoCapacity {
    pub(crate) manifest: VideoManifest,
    /// Whether the worker holds the denoiser on any of its ranks, and so
    /// reports the sample pool the engine schedules pages by.
    pub(crate) sampling: bool,
}

/// Product bytes this rank backs per request: every product of a component
/// it executes, or that a call it serves reads through the video graph's
/// persistent product edges (`MediaCall::readers`). The engine places a
/// request's products in this rank's pool by the same rule
/// (`MediaStorage::new`).
pub(crate) fn local_product_bytes(
    outputs: &BTreeMap<String, Vec<OutputInfo>>,
    routes: &BTreeMap<MediaCall, String>,
    held: &BTreeSet<String>,
) -> u64 {
    let mut consumers: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for producer in MediaCall::ALL {
        let readers = producer
            .readers()
            .iter()
            .filter(|reader| !reader.reads_resident_state());
        for consumer in readers {
            if let (Some(producer), Some(consumer)) = (routes.get(&producer), routes.get(consumer))
            {
                consumers
                    .entry(producer.as_str())
                    .or_default()
                    .insert(consumer.as_str());
            }
        }
    }
    outputs
        .iter()
        .filter(|(component, _)| {
            held.contains(component.as_str())
                || consumers
                    .get(component.as_str())
                    .is_some_and(|readers| readers.iter().any(|reader| held.contains(*reader)))
        })
        .map(|(_, outputs)| product_bytes(outputs))
        .sum()
}

impl VideoCapacity {
    /// The capacity of a rank whose executor stated `manifest`; `sampling`
    /// says whether the rank's worker holds the denoiser.
    pub(crate) fn new(manifest: VideoManifest, sampling: bool) -> Result<Self> {
        manifest
            .denoiser
            .validate()
            .map_err(|message| Error::Invalid(message.into()))?;
        Ok(Self { manifest, sampling })
    }

    /// The products every component publishes.
    pub(crate) fn outputs(&self) -> &BTreeMap<String, Vec<OutputInfo>> {
        &self.manifest.outputs
    }

    /// The sample pool of `slots` request slots: one float32 value per unit
    /// in the denoiser's pages, with no step workspace. Capacity planning
    /// sizes it and the denoising rank allocates it from this one plan.
    pub(crate) fn latent(&self, slots: usize) -> LatentPlan {
        let pages = self.manifest.pages;
        LatentPlan {
            request_pool_size: slots,
            num_pages: pages.pool_pages(slots as u64) as usize,
            page_units: pages.page_units as usize,
            latent_width: 1,
            dtype: crate::ffi::DType::F32,
        }
    }

    /// The denoiser as the handshake reports it.
    pub(crate) fn denoiser_info(&self) -> Result<VideoDenoiserInfo> {
        let description = &self.manifest.denoiser;
        let tasks = description
            .tasks
            .iter()
            .map(|name| {
                let task = models::video::VideoTask::from_name(name).ok_or_else(|| {
                    Error::Invalid(format!("the video denoiser serves unknown task {name:?}"))
                })?;
                Ok(VideoTaskInfo {
                    name: name.clone(),
                    conditioned: task.conditioned(),
                })
            })
            .collect::<Result<_>>()?;
        Ok(VideoDenoiserInfo {
            tasks,
            denoise_steps: description.denoise_steps(),
            description: description
                .encode()
                .map_err(|error| Error::Invalid(format!("encoding the denoiser: {error}")))?,
        })
    }

    /// The rank's per-request facts at `slots` request slots.
    pub(crate) fn facts(
        &self,
        slots: usize,
        local_product_bytes: u64,
        codec_ranks: Option<usize>,
    ) -> Result<MediaFacts> {
        let artifact_import_regions = match codec_ranks {
            Some(ranks) if ranks > 0 => {
                self.manifest.units.div_ceil(ranks) * ranks.saturating_sub(1)
            }
            _ => 0,
        };
        Ok(MediaFacts {
            video_denoiser: Some(self.denoiser_info()?),
            state_bytes: self.manifest.state_bytes,
            local_product_bytes,
            latent: self.sampling.then(|| self.latent(slots)),
            stores_latent: self.manifest.stores_latent,
            artifact_import_regions,
        })
    }
}
