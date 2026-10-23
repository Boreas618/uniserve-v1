//! Capacity of a video model's ranks.
//!
//! A video rank keeps every resident request's media state whole: the
//! denoiser's slot state and latent pages, the decoder's retained overlap
//! and the products the rank publishes or reads. Its request slots are
//! therefore sized by storage (each candidate count's state, products,
//! product store and latent pool must fit the device grant on every rank of
//! the group), and the engine schedules against those slots instead of a
//! cache pool. This module derives the per-request quantities from the model
//! ([`VideoModel`]) and the deployment, and the products every component
//! declares.

use std::collections::{BTreeMap, BTreeSet};

use ipc::{DType, DimBound, MediaCall, OutputInfo, ShapeBound};

use super::plan::{MediaFacts, product_bytes};
use crate::config::WorkerConfig;
use crate::host::mux::encoded_unit_bytes;
use crate::memory::LatentPlan;
use crate::request::video::{MediaPlan, admitted_canvases};
use crate::runner::denoiser::VideoModel;
use crate::runner::denoiser::{SamplePages, StateEnvelope};
use crate::{Error, Result};

/// What a rank's video components need beside its products.
pub(crate) struct VideoCapacity {
    pub(crate) media: MediaPlan,
    pub(crate) pages: SamplePages,
    /// Whether the worker holds the denoiser on any of its ranks, and so
    /// reports the sample pool the engine schedules pages by.
    pub(crate) sampling: bool,
    /// The denoiser's slot state on a denoising rank.
    pub(crate) envelope: Option<StateEnvelope>,
    /// Device bytes of one slot's retained decoder overlap.
    pub(crate) overlap_bytes: u64,
    /// Products by component, in output index order.
    pub(crate) outputs: BTreeMap<String, Vec<OutputInfo>>,
}

/// The products every component of the deployment publishes: the model's
/// own (the media timeline's and the text conditioning), the video codec's
/// encoded unit rows and, for conditioned tasks, the media reader's
/// condition media. Every rank plans them all, whichever components it
/// holds, and backs those of the components it executes or reads
/// ([`local_product_bytes`]).
pub(crate) fn outputs(
    model: &VideoModel,
    media: &MediaPlan,
    dtype: &str,
) -> Result<BTreeMap<String, Vec<OutputInfo>>> {
    let frames = media.max_frames();
    let mut result = model.outputs(
        frames,
        media.max_text_tokens,
        media.condition_rows,
        dtype,
        &media.canvases,
    )?;
    result
        .entry("video_codec".into())
        .or_default()
        .push(encoded_units(model, media)?);
    if media.condition_rows > 0 {
        let outputs = model.reader_outputs(frames, media.max_text_tokens, media.condition_rows)?;
        result
            .entry("media_reader".into())
            .or_default()
            .extend(outputs);
    }
    Ok(result)
}

/// The video codec's encoded unit rows: one row per media unit, each bounded
/// by the longest unit's encoding at the largest admitted canvas.
fn encoded_units(model: &VideoModel, media: &MediaPlan) -> Result<OutputInfo> {
    let units = model.unit_frames(media.max_frames())?;
    let longest = units.iter().copied().max().unwrap_or(0);
    let row = media
        .canvases
        .iter()
        .map(|canvas| {
            encoded_unit_bytes(
                u64::from(longest),
                u64::from(canvas.height),
                u64::from(canvas.width),
            )
        })
        .max()
        .unwrap_or(0);
    let narrow = |value: u64| {
        u32::try_from(value)
            .map_err(|_| Error::Unsupported("encoded unit rows exceed the protocol extents".into()))
    };
    Ok(OutputInfo {
        name: "encoded_units".into(),
        dtype: DType::U8,
        shape_bound: ShapeBound {
            dims: vec![
                DimBound::Device {
                    max: narrow(units.len() as u64)?,
                },
                DimBound::Static(narrow(row)?),
            ],
        },
        raster_axes: None,
    })
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

/// Plan a video rank's admitted sizes, sample pages, slot state and
/// products. `sampling` says whether the rank's worker holds the denoiser;
/// `denoising` and `decoding` say whether the rank itself holds the denoiser
/// and the video decoder.
pub(crate) fn plan(
    model: &VideoModel,
    config: &WorkerConfig,
    sampling: bool,
    denoising: bool,
    decoding: bool,
) -> Result<VideoCapacity> {
    let tokens = u32::try_from(config.max_model_len)
        .map_err(|_| Error::Invalid("prompt capacity exceeds its range".into()))?;
    let requested: Vec<u32> = config
        .video_text_capacities
        .iter()
        .map(|&capacity| u32::try_from(capacity).unwrap_or(u32::MAX))
        .collect();
    let admitted = admitted_canvases(
        &model.facts.canvases,
        &config.video_resolutions,
        &config.video_aspect_ratios,
    )?;
    // Under graphs every forward replays one captured at startup; a
    // reference brings media of any shape, which no startup capture holds.
    let graphs = config.graph_policy != "off" && config.device.is_cuda();
    if graphs && model.facts.tasks.iter().any(|task| task == "ref2va") {
        return Err(Error::Unsupported(
            "the video denoiser serves ref2va, whose references bring media of any size, \
             which no captured CUDA graph holds; serve it with cuda_graph_config.policy = \
             \"off\""
                .into(),
        ));
    }
    let media = MediaPlan::new(
        model,
        config.max_video_seconds,
        config.min_video_seconds,
        tokens,
        &requested,
        config.max_condition_rows,
        admitted,
        graphs,
    )?;
    let pages = SamplePages::plan(model, &media.largest)?;
    let envelope = if denoising {
        // A rank's share of each field moves with the prompt's length under
        // sequence parallelism (a long prompt moves media rows to other
        // ranks), so the envelope covers every layout startup prepares.
        let fields = std::iter::once(&media.maximum)
            .chain(&media.layouts)
            .chain(&media.conditioned)
            .chain(&media.largest)
            .map(|layout| model.layout_fields(layout))
            .collect::<Result<Vec<_>>>()?;
        Some(StateEnvelope::new(&fields, pages)?)
    } else {
        None
    };
    let overlap_bytes = if decoding {
        let mut largest = 0;
        for &canvas in &media.canvases {
            let (elements, dtype) = model.overlap(canvas)?;
            largest = largest.max(elements * dtype.bytes() as u64);
        }
        largest
    } else {
        0
    };
    let outputs = outputs(model, &media, &config.model_dtype)?;
    Ok(VideoCapacity {
        media,
        pages,
        sampling,
        envelope,
        overlap_bytes,
        outputs,
    })
}

impl VideoCapacity {
    /// The sample pool of `slots` request slots: one float32 value per unit
    /// in the denoiser's pages, with no step workspace. Capacity planning
    /// sizes it and the denoising rank allocates it from this one plan.
    pub(crate) fn latent(&self, slots: usize) -> LatentPlan {
        LatentPlan {
            request_pool_size: slots,
            num_pages: self.pages.pool_pages(slots as u64) as usize,
            page_units: self.pages.page_units as usize,
            latent_width: 1,
            dtype: crate::ffi::DType::F32,
            with_workspace: false,
        }
    }

    /// The rank's per-request facts at `slots` request slots.
    pub(crate) fn facts(
        &self,
        model: &VideoModel,
        slots: usize,
        local_product_bytes: u64,
        codec_ranks: Option<usize>,
    ) -> Result<MediaFacts> {
        let artifact_import_regions = match codec_ranks {
            Some(ranks) if ranks > 0 => {
                let units = model.unit_frames(self.media.max_frames())?.len();
                units.div_ceil(ranks) * ranks.saturating_sub(1)
            }
            _ => 0,
        };
        Ok(MediaFacts {
            video_denoiser: Some(model.denoiser_info(&self.media.admitted)?),
            state_bytes: self.envelope.as_ref().map_or(0, StateEnvelope::slot_bytes)
                + self.overlap_bytes,
            local_product_bytes,
            latent: self.sampling.then(|| self.latent(slots)),
            stores_latent: self.envelope.is_some(),
            artifact_import_regions,
        })
    }
}
