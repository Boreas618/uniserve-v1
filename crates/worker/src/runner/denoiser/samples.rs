//! Where a video request's samples and state live: the latent pages of a
//! request slot ([`SamplePages`]), the place of a layout's local samples in
//! them ([`SampleLayout`]), and the capacity of every state field across the
//! layouts a rank admits ([`StateEnvelope`]).

use std::ops::Range;

use super::{LayoutFields, VideoModel};
use crate::ffi::DType;
use crate::request::video::VideoLayout;
use crate::storage::SlotField;
use crate::{Error, Result};

/// A request's samples occupy at most this many pages, which bounds the
/// copies a step's gather and scatter make.
const REQUEST_PAGES: u64 = 8;

/// Each modality's sample starts on this element boundary.
const SAMPLE_ALIGNMENT: u64 = 256;

fn aligned(elements: u64) -> u64 {
    elements.div_ceil(SAMPLE_ALIGNMENT) * SAMPLE_ALIGNMENT
}

/// The latent pages of one request slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SamplePages {
    /// Elements of one page.
    pub(crate) page_units: u64,
    /// Pages a request slot owns.
    pub(crate) pages: u64,
}

impl SamplePages {
    /// Pages holding the complete samples of every `largest` layout: the
    /// global extents bound every sequence-parallel shard, even when a
    /// shorter prompt moves more media rows onto one rank.
    pub(crate) fn plan(model: &VideoModel, largest: &[VideoLayout]) -> Result<Self> {
        let mut capacity = 0;
        for layout in largest {
            let mut total = 0;
            for modality in &model.facts.modalities {
                total += aligned(model.latent_elements(modality, layout)?);
            }
            capacity = capacity.max(total);
        }
        if capacity == 0 {
            return Err(Error::Interface(
                "the video denoiser describes empty samples".into(),
            ));
        }
        let page_units = aligned(capacity.div_ceil(REQUEST_PAGES));
        Ok(Self {
            page_units,
            pages: capacity.div_ceil(page_units),
        })
    }

    /// Page zero is padding; every slot owns a consecutive run after it.
    pub(crate) fn for_slot(&self, slot: u64) -> Range<u64> {
        1 + (slot - 1) * self.pages..1 + slot * self.pages
    }

    /// Pages of a pool of `slots` slots, the padding page included.
    pub(crate) fn pool_pages(&self, slots: u64) -> u64 {
        slots * self.pages + 1
    }
}

/// Where a layout's local samples sit in a slot's pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SampleLayout {
    /// Element offset of each modality's local sample, in modality order.
    pub(crate) offsets: Vec<u64>,
    /// Elements of each modality's local sample.
    pub(crate) elements: Vec<u64>,
    /// Pages the samples occupy.
    pub(crate) pages: u64,
}

impl SampleLayout {
    pub(crate) fn new(fields: &LayoutFields, page_units: u64) -> Result<Self> {
        let mut offsets = Vec::new();
        let mut elements = Vec::new();
        let mut cursor = 0;
        for (name, _) in &fields.samples {
            let field = fields
                .state
                .iter()
                .find(|field| field.name == *name && field.sample)
                .ok_or_else(|| Error::Interface(format!("layout state lacks sample {name}")))?;
            offsets.push(cursor);
            elements.push(field.elements);
            cursor += aligned(field.elements);
        }
        Ok(Self {
            offsets,
            elements,
            pages: cursor.div_ceil(page_units),
        })
    }
}

/// The capacity of every field across the layouts a rank admits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StateEnvelope {
    /// Device fields of a request slot: every state table and the retained
    /// conditioning, as flat elements.
    pub(crate) device: Vec<SlotField>,
    /// Pinned host fields of a request slot: every state field's source,
    /// each modality's native draw and the condition draws.
    pub(crate) host: Vec<SlotField>,
    /// Sample elements the scratch holds: a slot's pages, which hold the
    /// complete samples and so any rank's share of them.
    pub(crate) samples: u64,
}

fn grow(
    fields: &mut Vec<SlotField>,
    name: String,
    dtype: DType,
    elements: u64,
    host: bool,
) -> Result<()> {
    let elements =
        i64::try_from(elements).map_err(|_| Error::Interface("state field overflows".into()))?;
    match fields.iter_mut().find(|field| field.name == name) {
        Some(field) if field.dtype != dtype => Err(Error::Interface(format!(
            "state field {name} changes its dtype across layouts"
        ))),
        Some(field) => {
            field.shape[0] = field.shape[0].max(elements);
            Ok(())
        }
        None => {
            fields.push(SlotField {
                name,
                dtype,
                shape: vec![elements],
                host,
            });
            Ok(())
        }
    }
}

impl StateEnvelope {
    /// The envelope of `layouts`' fields; the text conditioning is BF16.
    /// Every layout's local samples must fit a slot's `pages`.
    pub(crate) fn new(layouts: &[LayoutFields], pages: SamplePages) -> Result<Self> {
        let mut device = Vec::new();
        let mut host = Vec::new();
        for fields in layouts {
            for field in &fields.state {
                let elements = if field.sample {
                    // A rank's rows of a sample move with the rows before
                    // them in the sequence: the prompt's and conditions' of
                    // the request's own layout, which no startup layout
                    // need match. The complete sample bounds every share.
                    fields
                        .samples
                        .iter()
                        .find(|(name, _)| *name == field.name)
                        .map(|(_, sample)| sample.rows * sample.width)
                        .ok_or_else(|| {
                            Error::Interface(format!(
                                "sample field {} has no sample rows",
                                field.name
                            ))
                        })?
                } else {
                    grow(
                        &mut device,
                        field.name.clone(),
                        field.dtype,
                        field.elements,
                        false,
                    )?;
                    field.elements
                };
                grow(
                    &mut host,
                    format!("{}_source", field.name),
                    field.dtype,
                    elements,
                    true,
                )?;
            }
            for (name, elements) in &fields.noise {
                grow(
                    &mut host,
                    format!("{name}_noise"),
                    DType::F32,
                    *elements,
                    true,
                )?;
            }
            grow(
                &mut host,
                "condition_noise".into(),
                DType::F32,
                fields.condition_noise.max(1),
                true,
            )?;
            let (rows, width) = fields.text_condition;
            grow(
                &mut device,
                "text_condition".into(),
                DType::BF16,
                rows * width,
                false,
            )?;
            if SampleLayout::new(fields, pages.page_units)?.pages > pages.pages {
                return Err(Error::Invariant(
                    "a video layout's samples exceed the request's latent pages".into(),
                ));
            }
        }
        Ok(Self {
            device,
            host,
            samples: pages.pages * pages.page_units,
        })
    }

    /// Device bytes of one request slot's state.
    pub(crate) fn slot_bytes(&self) -> u64 {
        self.device
            .iter()
            .map(|field| {
                field.shape.iter().product::<i64>().max(0) as u64 * field.dtype.bytes() as u64
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::denoiser::video_model::{SampleRows, StateField};

    fn field(name: &str, dtype: DType, shape: &[i64], sample: bool) -> StateField {
        StateField {
            name: name.into(),
            dtype,
            elements: shape.iter().product::<i64>() as u64,
            sample,
        }
    }

    /// One layout's fields on a rank that holds every sample row but the
    /// video's last `shared` rows: a video and an audio sample, a timestep
    /// table and `rows` prompt rows of width 4.
    fn shared_layout(video: i64, audio: i64, table: i64, rows: u64, shared: i64) -> LayoutFields {
        let sample = |rows: i64, local: i64| SampleRows {
            rows: rows as u64,
            width: 8,
            local: (0, local as u64),
        };
        LayoutFields {
            state: vec![
                field("video", DType::F32, &[video - shared, 8], true),
                field("audio", DType::F32, &[audio, 8], true),
                field("tables", DType::I64, &[table], false),
            ],
            samples: vec![
                ("video".into(), sample(video, video - shared)),
                ("audio".into(), sample(audio, audio)),
            ],
            noise: vec![
                ("video".into(), (video * 8) as u64),
                ("audio".into(), (audio * 8) as u64),
            ],
            condition_noise: 0,
            text_condition: (rows, 4),
            workspace: 0,
        }
    }

    /// One layout's fields on a rank that holds every sample row.
    fn layout(video: i64, audio: i64, table: i64, rows: u64) -> LayoutFields {
        shared_layout(video, audio, table, rows, 0)
    }

    #[test]
    fn samples_start_on_aligned_offsets_within_their_pages() {
        // 40 video rows of 8 elements take 320 elements, aligned to 512.
        let samples = SampleLayout::new(&layout(40, 3, 5, 16), 256).unwrap();
        assert_eq!(samples.offsets, vec![0, 512]);
        assert_eq!(samples.elements, vec![320, 24]);
        assert_eq!(samples.pages, 3);
        assert!(
            SampleLayout::new(
                &LayoutFields {
                    samples: vec![(
                        "depth".into(),
                        SampleRows {
                            rows: 1,
                            width: 1,
                            local: (0, 1)
                        }
                    )],
                    ..layout(1, 1, 1, 1)
                },
                256,
            )
            .is_err()
        );
    }

    #[test]
    fn the_envelope_holds_every_layout_field_at_its_largest() {
        let pages = |pages| SamplePages {
            page_units: 256,
            pages,
        };
        let envelope =
            StateEnvelope::new(&[layout(40, 3, 5, 16), layout(8, 30, 9, 4)], pages(4)).unwrap();
        let size = |fields: &[SlotField], name: &str| {
            fields
                .iter()
                .find(|field| field.name == name)
                .map(|field| field.shape[0])
        };
        // Samples live in latent pages; only tables and the conditioning
        // are slot state on the device.
        assert_eq!(size(&envelope.device, "video"), None);
        assert_eq!(size(&envelope.device, "tables"), Some(9));
        assert_eq!(size(&envelope.device, "text_condition"), Some(64));
        assert_eq!(envelope.slot_bytes(), 9 * 8 + 64 * 2);
        // Host sources cover every field; noise and condition draws are FP32.
        assert_eq!(size(&envelope.host, "video_source"), Some(320));
        assert_eq!(size(&envelope.host, "audio_noise"), Some(240));
        assert_eq!(size(&envelope.host, "condition_noise"), Some(1));
        // The scratch holds a slot's pages, any rank's share of a sample.
        assert_eq!(envelope.samples, 4 * 256);
        // A layout whose samples exceed a slot's pages is refused.
        assert!(StateEnvelope::new(&[layout(40, 3, 5, 16)], pages(2)).is_err());

        // A rank's share of a sample grows and shrinks with the request's
        // prompt and conditions, so its source holds the complete sample
        // wherever the prepared layouts leave the rank fewer rows.
        let envelope = StateEnvelope::new(&[shared_layout(40, 3, 5, 16, 30)], pages(4)).unwrap();
        assert_eq!(size(&envelope.host, "video_source"), Some(320));

        let mut changed = layout(8, 30, 9, 4);
        changed.state[2].dtype = DType::I32;
        assert!(StateEnvelope::new(&[layout(40, 3, 5, 16), changed], pages(4)).is_err());
    }

    #[test]
    fn slots_own_consecutive_pages_after_the_padding_page() {
        let pages = SamplePages {
            page_units: 256,
            pages: 3,
        };
        assert_eq!(pages.for_slot(1), 1..4);
        assert_eq!(pages.for_slot(2), 4..7);
        assert_eq!(pages.pool_pages(2), 7);
    }
}
