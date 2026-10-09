//! A video model's facts and sizing rules, as `video.model` reports them.
//!
//! [`VideoModel`] wraps the model functions over integer descriptors that
//! admission, capacity planning and the media runtimes need: the denoiser's
//! size and layout rules, the decoders' unit divisions and the products the
//! model's components publish. Every function reads configuration only, so
//! a host rank holding the model's weight-less description builds one too.

use std::collections::BTreeMap;
use std::ops::Range;

use tvm_ffi::{Any, AnyView};
use uniserve_core::Canvas;
use uniserve_ipc::{ConditionTiles, OutputInfo, VideoDenoiserInfo};

use crate::ffi::{DType, Function, Handle, Record, integer_list, integers, list};
use crate::model::components::{Capability, Declarations};
use crate::model::interface::Interface;
use crate::request::video::{VideoLayout, VideoSize};
use crate::storage::memory::parse_dtype;
use crate::{Error, Result};

/// What the model reports about its video timeline.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VideoFacts {
    /// Sample modalities in the denoiser's order (`video`, `audio`).
    pub(crate) modalities: Vec<String>,
    pub(crate) num_steps: u32,
    pub(crate) tasks: Vec<String>,
    pub(crate) video_shift: f64,
    pub(crate) audio_shift: f64,
    /// Canvases the denoiser offers: a deployment prepares a selection of
    /// them, or all of them without one, and admits only those.
    pub(crate) canvases: Vec<Canvas>,
    pub(crate) max_sequence_rows: Option<u32>,
    pub(crate) condition_tiles: Option<ConditionTiles>,
    pub(crate) sample_dtype: DType,
    pub(crate) text_condition_width: u32,
    /// Output video frames per second.
    pub(crate) frame_rate: u32,
    /// Output audio samples per second.
    pub(crate) sample_rate: u32,
}

/// One state field of a layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StateField {
    pub(crate) name: String,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<i64>,
    /// A sample modality, stored in latent pages rather than slot banks.
    pub(crate) sample: bool,
}

impl StateField {
    pub(crate) fn elements(&self) -> u64 {
        self.shape
            .iter()
            .map(|&extent| extent.max(0) as u64)
            .product()
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.elements() * self.dtype.bytes() as u64
    }
}

/// A modality's sample in one layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SampleRows {
    /// Rows of the complete sample across all ranks.
    pub(crate) rows: u64,
    pub(crate) width: u64,
    /// This rank's rows.
    pub(crate) local: (u64, u64),
}

/// The request state of one layout on this rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayoutFields {
    pub(crate) state: Vec<StateField>,
    /// Sample rows by modality, in modality order.
    pub(crate) samples: Vec<(String, SampleRows)>,
    /// Native draw shapes by modality, in modality order.
    pub(crate) noise: Vec<(String, Vec<i64>)>,
    /// FP32 elements of the condition draws.
    pub(crate) condition_noise: u64,
    /// `[rows, width]` of the retained conditioning.
    pub(crate) text_condition: (u64, u64),
    /// Elements of the layout's workspace buffers.
    pub(crate) workspace: u64,
}

impl LayoutFields {
    /// Elements of the state and workspace the denoiser holds for the
    /// layout on this rank: what the layout takes of the storage every
    /// layout shares.
    pub(crate) fn storage(&self) -> u64 {
        self.state.iter().map(StateField::elements).sum::<u64>() + self.workspace
    }
}

struct Functions {
    legal_frame_count: Function,
    request_size: Function,
    layout_size: Function,
    holds: Function,
    condition_layout: Function,
    layout_fields: Function,
    latent_shape: Function,
    segment: Function,
    window: Function,
    overlap: Function,
    unit_frames: Function,
    track_samples: Function,
    audio_units: Function,
    audio_latent_frames: Function,
    outputs: Function,
    reader_outputs: Function,
}

/// A video model's facts and sizing rules.
pub(crate) struct VideoModel {
    /// The component of the deployment's video denoiser.
    pub(crate) denoiser: String,
    pub(crate) facts: VideoFacts,
    functions: Functions,
}

/// The video denoiser component a deployment places: the unique declared
/// video denoiser among `deployment` components (or the only one declared).
pub(crate) fn placed_denoiser(
    declarations: &Declarations,
    deployment: &[String],
) -> Result<String> {
    let declared: Vec<&str> = declarations
        .iter()
        .filter(|(_, calls)| calls.iter().any(|call| call.is(Capability::VideoDenoiser)))
        .map(|(name, _)| name)
        .collect();
    if let [only] = declared[..] {
        return Ok(only.to_owned());
    }
    let placed: Vec<&str> = declared
        .iter()
        .copied()
        .filter(|name| deployment.iter().any(|component| component == name))
        .collect();
    match placed[..] {
        [one] => Ok(one.to_owned()),
        _ => Err(Error::Unsupported(format!(
            "a deployment places exactly one of the model's video denoisers {declared:?}; it places {}",
            placed.len()
        ))),
    }
}

fn narrow(value: i64, what: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::Interface(format!("video.model reports an invalid {what}")))
}

impl VideoModel {
    /// Resolve the model functions of `loaded` for the `denoiser` component.
    pub(crate) fn new(interface: &Interface, loaded: &Handle, denoiser: String) -> Result<Self> {
        let value = interface.function("video.model")?.call(&[
            loaded.view(),
            AnyView::from(&Any::from(tvm_ffi::String::from(denoiser.as_str()))),
        ])?;
        let map = crate::ffi::Functions::new("video.model", value)?;
        let functions = Functions {
            legal_frame_count: map.get("legal_frame_count")?,
            request_size: map.get("request_size")?,
            layout_size: map.get("layout_size")?,
            holds: map.get("holds")?,
            condition_layout: map.get("condition_layout")?,
            layout_fields: map.get("layout_fields")?,
            latent_shape: map.get("latent_shape")?,
            segment: map.get("segment")?,
            window: map.get("window")?,
            overlap: map.get("overlap")?,
            unit_frames: map.get("unit_frames")?,
            track_samples: map.get("track_samples")?,
            audio_units: map.get("audio_units")?,
            audio_latent_frames: map.get("audio_latent_frames")?,
            outputs: map.get("outputs")?,
            reader_outputs: map.get("reader_outputs")?,
        };
        let record = Record::new("video.model.describe", map.get("describe")?.call(&[])?)?;
        let canvases = record
            .list("canvases")?
            .into_iter()
            .map(|value| {
                let values =
                    integers(value).map_err(|error| Error::numerical("video.model", &error))?;
                let [width, height] = values[..] else {
                    return Err(Error::Interface("a canvas is [width, height]".into()));
                };
                Ok(Canvas {
                    width: narrow(width, "canvas")?,
                    height: narrow(height, "canvas")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let tiles = record.shape("condition_tiles")?;
        let condition_tiles = match tiles[..] {
            [] => None,
            [rows, frames, height, width] => Some(ConditionTiles {
                rows: narrow(rows, "tile")?,
                video: [
                    narrow(frames, "tile")?,
                    narrow(height, "tile")?,
                    narrow(width, "tile")?,
                ],
            }),
            _ => {
                return Err(Error::Interface(
                    "condition tiles are [rows, t, h, w]".into(),
                ));
            }
        };
        let rows = record.count("max_sequence_rows")?;
        let facts = VideoFacts {
            modalities: record.strings("modalities")?,
            num_steps: narrow(record.get::<i64>("num_steps")?, "step count")?,
            tasks: record.strings("tasks")?,
            video_shift: record.get::<f64>("video_shift")?,
            audio_shift: record.get::<f64>("audio_shift")?,
            canvases,
            max_sequence_rows: (rows > 0).then_some(rows as u32),
            condition_tiles,
            sample_dtype: parse_dtype(&record.string("sample_dtype")?)?,
            text_condition_width: narrow(record.get::<i64>("text_condition_width")?, "width")?,
            frame_rate: narrow(record.get::<i64>("frame_rate")?, "frame rate")?,
            sample_rate: narrow(record.get::<i64>("sample_rate")?, "sample rate")?,
        };
        if facts.modalities.is_empty() || facts.num_steps == 0 || facts.frame_rate == 0 {
            return Err(Error::Interface(
                "video.model describes no sample, step or frame rate".into(),
            ));
        }
        Ok(Self {
            denoiser,
            facts,
            functions,
        })
    }

    /// The tasks of the denoiser this worker executes, in its order.
    pub(crate) fn executed_tasks(&self) -> Vec<String> {
        self.facts
            .tasks
            .iter()
            .filter(|task| matches!(task.as_str(), "t2va" | "fl2va" | "ref2va"))
            .cloned()
            .collect()
    }

    /// Whether any executed task takes conditions.
    pub(crate) fn conditions(&self) -> bool {
        self.executed_tasks().iter().any(|task| task != "t2va")
    }

    /// The handshake's description of the denoiser on a deployment that
    /// prepares and admits `canvases`.
    pub(crate) fn denoiser_info(&self, canvases: &[Canvas]) -> Result<VideoDenoiserInfo> {
        let tasks = self.executed_tasks();
        if tasks.is_empty() {
            return Err(Error::Unsupported(format!(
                "the placed video denoiser serves {}, none of which this worker executes",
                self.facts.tasks.join(", ")
            )));
        }
        Ok(VideoDenoiserInfo {
            tasks,
            schedule_points: self.facts.num_steps + 1,
            video_shift: self.facts.video_shift,
            audio_shift: self.facts.audio_shift,
            canvases: canvases.to_vec(),
            max_sequence_rows: self.facts.max_sequence_rows,
            condition_tiles: self.facts.condition_tiles,
        })
    }

    pub(crate) fn legal_frame_count(&self, frames: u32) -> Result<u32> {
        let value: i64 = self
            .functions
            .legal_frame_count
            .call_as(&[AnyView::from(&Any::from(i64::from(frames)))])?;
        narrow(value, "frame count")
    }

    /// A request's condition rows; a size the model rejects is an invalid
    /// descriptor.
    pub(crate) fn request_size(&self, size: &VideoSize) -> Result<u32> {
        let value: i64 = self
            .functions
            .request_size
            .call_as(
                &size_arguments(size)
                    .iter()
                    .map(AnyView::from)
                    .collect::<Vec<_>>(),
            )
            .map_err(invalid_size)?;
        narrow(value, "condition row count")
    }

    /// The smallest layout holding `size`.
    pub(crate) fn layout_size(&self, size: &VideoSize) -> Result<VideoLayout> {
        let value = self
            .functions
            .layout_size
            .call_values(&size_arguments(size))
            .map_err(invalid_size)?;
        layout_of(value)
    }

    pub(crate) fn holds(&self, layout: &VideoLayout, size: &VideoSize) -> Result<bool> {
        let mut arguments = vec![layout.descriptor()];
        arguments.extend(size_arguments(size));
        let value = self
            .functions
            .holds
            .call_values(&arguments)
            .map_err(invalid_size)?;
        bool::try_from(value).map_err(|error| Error::numerical("video.model.holds", &error))
    }

    /// `layout` widened so its condition region holds `rows`.
    pub(crate) fn condition_layout(&self, layout: &VideoLayout, rows: u32) -> Result<VideoLayout> {
        layout_of(
            self.functions
                .condition_layout
                .call_values(&[layout.descriptor(), Any::from(i64::from(rows))])?,
        )
    }

    /// A modality's complete sample shape in `layout`.
    pub(crate) fn latent_shape(&self, modality: &str, layout: &VideoLayout) -> Result<Vec<u64>> {
        let value = self.functions.latent_shape.call_values(&[
            Any::from(tvm_ffi::String::from(modality)),
            layout.descriptor(),
        ])?;
        Ok(integers(value)
            .map_err(|error| Error::numerical("video.model.latent_shape", &error))?
            .into_iter()
            .map(|extent| extent.max(0) as u64)
            .collect())
    }

    /// Elements of a modality's complete sample in `layout`.
    pub(crate) fn latent_elements(&self, modality: &str, layout: &VideoLayout) -> Result<u64> {
        let value = self.functions.latent_shape.call_values(&[
            Any::from(tvm_ffi::String::from(modality)),
            layout.descriptor(),
        ])?;
        Ok(integers(value)
            .map_err(|error| Error::numerical("video.model.latent_shape", &error))?
            .iter()
            .map(|&extent| extent.max(0) as u64)
            .product())
    }

    /// The request state of `layout` on this rank.
    pub(crate) fn layout_fields(&self, layout: &VideoLayout) -> Result<LayoutFields> {
        let record = Record::new(
            "video.model.layout_fields",
            self.functions
                .layout_fields
                .call_values(&[layout.descriptor()])?,
        )?;
        let state = record
            .records("state")?
            .iter()
            .map(|field| {
                Ok(StateField {
                    name: field.string("name")?,
                    dtype: parse_dtype(&field.string("dtype")?)?,
                    shape: field.shape("shape")?,
                    sample: field.flag("sample")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let samples_record = record.record("samples")?;
        let noise_record = record.record("noise")?;
        let mut samples = Vec::new();
        let mut noise = Vec::new();
        for name in &self.facts.modalities {
            let values = samples_record.shape(name)?;
            let [rows, width, start, stop] = values[..] else {
                return Err(Error::Interface(
                    "a layout's sample is [rows, width, start, stop]".into(),
                ));
            };
            samples.push((
                name.clone(),
                SampleRows {
                    rows: rows.max(0) as u64,
                    width: width.max(0) as u64,
                    local: (start.max(0) as u64, stop.max(0) as u64),
                },
            ));
            noise.push((name.clone(), noise_record.shape(name)?));
        }
        let text = record.shape("text_condition")?;
        let [rows, width] = text[..] else {
            return Err(Error::Interface(
                "a layout's text condition is [rows, width]".into(),
            ));
        };
        Ok(LayoutFields {
            state,
            samples,
            noise,
            condition_noise: record.count("condition_noise")? as u64,
            text_condition: (rows.max(0) as u64, width.max(0) as u64),
            workspace: record.count("workspace")? as u64,
        })
    }

    /// The `(frames, canvas)` segment unit `unit` of a video decodes to.
    pub(crate) fn segment(&self, frames: u32, canvas: Canvas, unit: u32) -> Result<(u32, Canvas)> {
        let value = self.functions.segment.call_values(&[
            Any::from(i64::from(frames)),
            Any::from(i64::from(canvas.width)),
            Any::from(i64::from(canvas.height)),
            Any::from(i64::from(unit)),
        ])?;
        let values =
            integers(value).map_err(|error| Error::numerical("video.model.segment", &error))?;
        let [frames, width, height] = values[..] else {
            return Err(Error::Interface(
                "a segment is [frames, width, height]".into(),
            ));
        };
        Ok((
            narrow(frames, "segment")?,
            Canvas {
                width: narrow(width, "segment")?,
                height: narrow(height, "segment")?,
            },
        ))
    }

    /// A segment's decoder input as `(elements, dtype)`.
    pub(crate) fn window(&self, segment: (u32, Canvas)) -> Result<(u64, DType)> {
        crate::runner::video::buffer_size(self.functions.window.call_values(&[
            Any::from(i64::from(segment.0)),
            Any::from(i64::from(segment.1.width)),
            Any::from(i64::from(segment.1.height)),
        ])?)
    }

    /// The overlap a unit of `canvas` carries, as `(elements, dtype)`.
    pub(crate) fn overlap(&self, canvas: Canvas) -> Result<(u64, DType)> {
        crate::runner::video::buffer_size(self.functions.overlap.call_values(&[
            Any::from(i64::from(canvas.width)),
            Any::from(i64::from(canvas.height)),
        ])?)
    }

    /// Frames of each media unit of a video of `frames` frames.
    pub(crate) fn unit_frames(&self, frames: u32) -> Result<Vec<u32>> {
        let value = self
            .functions
            .unit_frames
            .call_values(&[Any::from(i64::from(frames))])?;
        integers(value)
            .map_err(|error| Error::numerical("video.model.unit_frames", &error))?
            .into_iter()
            .map(|count| narrow(count, "unit frame count"))
            .collect()
    }

    /// Samples of the audio track generated with a video of `frames`.
    pub(crate) fn track_samples(&self, frames: u32) -> Result<u64> {
        let value: i64 = self
            .functions
            .track_samples
            .call_as(&[AnyView::from(&Any::from(i64::from(frames)))])?;
        u64::try_from(value).map_err(|_| Error::Interface("invalid track length".into()))
    }

    /// Each of `units` audio units' samples of a `samples`-sample track.
    pub(crate) fn audio_units(&self, samples: u64, units: u32) -> Result<Vec<Range<u64>>> {
        let value = self
            .functions
            .audio_units
            .call_values(&[Any::from(samples as i64), Any::from(i64::from(units))])?;
        let flat =
            integers(value).map_err(|error| Error::numerical("video.model.audio_units", &error))?;
        Ok(flat
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[start, stop]| start.max(0) as u64..stop.max(0) as u64)
            .collect())
    }

    /// Latent frames of a `samples`-sample track.
    pub(crate) fn audio_latent_frames(&self, samples: u64) -> Result<u32> {
        let value: i64 = self
            .functions
            .audio_latent_frames
            .call_as(&[AnyView::from(&Any::from(samples as i64))])?;
        narrow(value, "latent frame count")
    }

    /// The products each model component publishes, bounded for any request
    /// of at most `frames` frames, `tokens` prompt tokens and `rows`
    /// condition rows at any of `canvases`.
    pub(crate) fn outputs(
        &self,
        frames: u32,
        tokens: u32,
        rows: u32,
        dtype: &str,
        canvases: &[Canvas],
    ) -> Result<BTreeMap<String, Vec<OutputInfo>>> {
        let record = Record::new(
            "video.model.outputs",
            self.functions.outputs.call_values(&[
                Any::from(i64::from(frames)),
                Any::from(i64::from(tokens)),
                Any::from(i64::from(rows)),
                Any::from(tvm_ffi::String::from(dtype)),
                Any::from(list(canvases.iter().map(|canvas| {
                    Any::from(integer_list([
                        i64::from(canvas.width),
                        i64::from(canvas.height),
                    ]))
                }))),
            ])?,
        )?;
        let mut result = BTreeMap::new();
        for component in record.keys() {
            let outputs = record
                .records(&component)?
                .iter()
                .map(|output| crate::model::output_info(&component, output))
                .collect::<Result<Vec<_>>>()?;
            result.insert(component, outputs);
        }
        Ok(result)
    }

    /// The media reader's condition products.
    pub(crate) fn reader_outputs(
        &self,
        frames: u32,
        tokens: u32,
        rows: u32,
    ) -> Result<Vec<OutputInfo>> {
        let value = self.functions.reader_outputs.call_values(&[
            Any::from(i64::from(frames)),
            Any::from(i64::from(tokens)),
            Any::from(i64::from(rows)),
        ])?;
        let records: tvm_ffi::Array<Any> = value
            .try_into()
            .map_err(|error| Error::numerical("video.model.reader_outputs", &error))?;
        records
            .iter()
            .map(|record| {
                crate::model::output_info(
                    "media_reader",
                    &Record::new("video.model.reader_outputs", record)?,
                )
            })
            .collect()
    }
}

fn size_arguments(size: &VideoSize) -> Vec<Any> {
    vec![
        size.descriptor(),
        size.conditions_descriptor(),
        size.spans_descriptor(),
    ]
}

fn layout_of(value: Any) -> Result<VideoLayout> {
    let values = integers(value).map_err(|error| Error::numerical("video.model", &error))?;
    VideoLayout::from_values(&values)
}

/// A size the model rejects is the request's error, not the worker's.
fn invalid_size(error: Error) -> Error {
    match error {
        Error::Numerical { kind, message, .. } if kind == "InputError" || kind == "ValueError" => {
            Error::Invalid(message)
        }
        error => error,
    }
}
