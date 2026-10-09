//! Capacity of a video model's ranks.
//!
//! A video rank keeps every resident request's media state whole: the
//! denoiser's slot state and latent pages, the decoder's retained overlap
//! and the products the rank publishes or reads. Its request slots are
//! therefore sized by storage (each candidate count's state, products,
//! product store and latent pool must fit the device grant on every rank of
//! the group), and the engine schedules against those slots instead of a
//! KV pool. This module derives the per-request quantities from the model
//! ([`VideoModel`]) and the deployment, and the products every component
//! declares.

use std::collections::{BTreeMap, BTreeSet};

use uniserve_ipc::{DType, DimBound, MediaCall, OutputInfo, ShapeBound};

use super::plan::{LatentPlan, MediaFacts, product_bytes};
use crate::config::WorkerConfig;
use crate::host::mux::encoded_unit_bytes;
use crate::model::video::VideoModel;
use crate::request::video::MediaPlan;
use crate::runner::diffusion::video::{SamplePages, StateEnvelope};
use crate::{Error, Result};

/// What a rank's video components need beside its products.
pub(crate) struct VideoCapacity {
    pub(crate) media: MediaPlan,
    pub(crate) pages: SamplePages,
    /// The denoiser's slot state on a denoising rank.
    pub(crate) envelope: Option<StateEnvelope>,
    /// Device bytes of one slot's retained decoder overlap.
    pub(crate) overlap_bytes: u64,
    /// Products by component, in output index order.
    pub(crate) outputs: BTreeMap<String, Vec<OutputInfo>>,
}

/// The producer and consumer calls of video products.
const EDGES: [(MediaCall, MediaCall); 10] = [
    (MediaCall::MediaReading, MediaCall::VisionEncoding),
    (MediaCall::MediaReading, MediaCall::LatentEncoding),
    (MediaCall::VisionEncoding, MediaCall::TextEncoding),
    (MediaCall::LatentEncoding, MediaCall::LatentPreparation),
    (MediaCall::TextEncoding, MediaCall::LatentPreparation),
    (MediaCall::Denoising, MediaCall::VideoDecoding),
    (MediaCall::Denoising, MediaCall::AudioDecoding),
    (MediaCall::VideoDecoding, MediaCall::VideoEncoding),
    (MediaCall::AudioDecoding, MediaCall::AudioEncoding),
    (MediaCall::VideoEncoding, MediaCall::Muxing),
];

/// The products every component of the deployment publishes: the model's
/// own (the media timeline's and the text conditioning, beside the `facts`
/// outputs of the components this rank loads), the video codec's
/// encoded unit rows and, for conditioned tasks, the media reader's
/// condition media. Every rank plans them all, whichever components it
/// holds, because the engine places a request's products in one logical
/// buffer pool that every rank sizes alike.
pub(crate) fn outputs(
    model: &VideoModel,
    media: &MediaPlan,
    facts: &BTreeMap<String, Vec<OutputInfo>>,
    dtype: &str,
) -> Result<BTreeMap<String, Vec<OutputInfo>>> {
    let frames = media.max_frames();
    let mut result = facts.clone();
    for (component, outputs) in model.outputs(
        frames,
        media.max_text_tokens,
        media.condition_rows,
        dtype,
        &media.canvases,
    )? {
        let declared = result.entry(component).or_default();
        for output in outputs {
            // A loaded component's facts already declare its products.
            if !declared.iter().any(|known| known.name == output.name) {
                declared.push(output);
            }
        }
    }
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

/// Product bytes this rank retains per request: every product a component
/// it holds publishes, or that a call it serves reads.
pub(crate) fn local_product_bytes(
    outputs: &BTreeMap<String, Vec<OutputInfo>>,
    routes: &BTreeMap<MediaCall, String>,
    held: &BTreeSet<String>,
) -> u64 {
    let mut consumers: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (producer, consumer) in EDGES {
        if let (Some(producer), Some(consumer)) = (routes.get(&producer), routes.get(&consumer)) {
            consumers
                .entry(producer.as_str())
                .or_default()
                .insert(consumer.as_str());
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
/// products. `denoising` and `decoding` say whether the rank holds the
/// denoiser and the video decoder.
pub(crate) fn plan(
    model: &VideoModel,
    config: &WorkerConfig,
    facts: &BTreeMap<String, Vec<OutputInfo>>,
    denoising: bool,
    decoding: bool,
) -> Result<VideoCapacity> {
    let rows = if model.conditions() {
        u32::try_from(config.max_condition_rows)
            .map_err(|_| Error::Invalid("condition capacity exceeds its range".into()))?
    } else {
        0
    };
    let tokens = u32::try_from(config.max_sequence_tokens)
        .map_err(|_| Error::Invalid("prompt capacity exceeds its range".into()))?;
    let requested: Vec<u32> = config
        .video_text_capacities
        .iter()
        .map(|&capacity| u32::try_from(capacity).unwrap_or(u32::MAX))
        .collect();
    let media = MediaPlan::new(
        model,
        config.max_video_seconds,
        config.min_video_seconds,
        tokens,
        &requested,
        rows,
        &config.video_canvases,
    )?;
    let pages = SamplePages::plan(model, &media.largest)?;
    let envelope = if denoising {
        // A rank's share of each field moves with the prompt's length under
        // sequence parallelism (a long prompt moves media rows to other
        // ranks), so the envelope covers every layout startup prepares.
        let fields = std::iter::once(&media.maximum)
            .chain(&media.layouts)
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
    let outputs = outputs(model, &media, facts, &config.model_dtype)?;
    Ok(VideoCapacity {
        media,
        pages,
        envelope,
        overlap_bytes,
        outputs,
    })
}

impl VideoCapacity {
    /// The rank's per-request facts at `slots` request slots.
    pub(crate) fn facts(
        &self,
        model: &VideoModel,
        slots: usize,
        local_product_bytes: u64,
        codec_ranks: Option<usize>,
    ) -> Result<MediaFacts> {
        let product_bytes = self
            .outputs
            .values()
            .map(|outputs| product_bytes(outputs))
            .sum();
        let artifact_import_regions = match codec_ranks {
            Some(ranks) if ranks > 0 => {
                let units = model.unit_frames(self.media.max_frames())?.len();
                units.div_ceil(ranks) * ranks.saturating_sub(1)
            }
            _ => 0,
        };
        Ok(MediaFacts {
            num_steps: model.facts.num_steps,
            video_denoiser: Some(model.denoiser_info(&self.media.admitted)?),
            state_bytes: self.envelope.as_ref().map_or(0, StateEnvelope::slot_bytes)
                + self.overlap_bytes,
            local_product_bytes,
            product_bytes,
            latent: self.envelope.as_ref().map(|_| LatentPlan {
                request_pool_size: slots,
                num_pages: self.pages.pool_pages(slots as u64) as usize,
                page_units: self.pages.page_units as usize,
                latent_width: 1,
                element_bytes: 4,
                with_workspace: false,
            }),
            artifact_import_regions,
        })
    }
}
