//! A video model's facts and sizing rules ([`VideoModel`]): what the sizing
//! methods of the Python runners answer about the model's timeline and about
//! each layout's state fields and samples, which admission, capacity planning
//! and the runners read.

use std::collections::BTreeMap;
use std::ops::Range;

use tvm_ffi::Any;
use uniserve_core::{Canvas, FrameClips};
use uniserve_ipc::{DimBound, OutputInfo, RasterAxes, ShapeBound, VideoDenoiserInfo};

use super::{DECODER, DENOISER, OUTPUTS};
use crate::config::placement::{Capability, Declarations};
use crate::ffi::{DType, Function, Handle, Interpreter, integer_list, integers};
use crate::lane::{Completion, Job};
use crate::request::video::{VideoLayout, VideoSize};
use crate::runner::methods::{Argument, MethodCall};
use crate::{Error, Result};

/// What the model reports about its video timeline.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VideoFacts {
    /// Sample modalities in the denoiser's order (`video`, `audio`).
    pub(crate) modalities: Vec<String>,
    pub(crate) num_steps: u32,
    pub(crate) tasks: Vec<String>,
    /// Whether the denoiser serves requests without conditions; without
    /// it every request evaluates in a layout of its own.
    pub(crate) text_only: bool,
    pub(crate) video_shift: f64,
    pub(crate) audio_shift: f64,
    /// Canvases the denoiser offers: a deployment prepares a selection of
    /// them, or all of them without one, and admits only those.
    pub(crate) canvases: Vec<Canvas>,
    pub(crate) max_sequence_rows: Option<u32>,
    pub(crate) sample_dtype: DType,
    pub(crate) text_condition_width: u32,
    /// Output video frames per second.
    pub(crate) frame_rate: u32,
    /// Output audio samples per second.
    pub(crate) sample_rate: u32,
    /// How the video VAE divides the generated frames into clips.
    pub(crate) frames: FrameClips,
}

/// One state field of a layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StateField {
    pub(crate) name: String,
    pub(crate) dtype: DType,
    /// Elements of the field on this rank.
    pub(crate) elements: u64,
    /// A sample modality, stored in latent pages rather than slot banks.
    pub(crate) sample: bool,
}

impl StateField {
    pub(crate) fn bytes(&self) -> u64 {
        self.elements * self.dtype.bytes() as u64
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
    /// Elements of each modality's native draw, in modality order.
    pub(crate) noise: Vec<(String, u64)>,
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
        self.state.iter().map(|field| field.elements).sum::<u64>() + self.workspace
    }
}

/// What the model's layout functions answer about one layout, as
/// [`VideoModel::fields`] reads them.
struct LayoutAnswers {
    /// The state fields, in state order, and each one's dtype and elements.
    names: Vec<String>,
    dtypes: Vec<String>,
    elements: Vec<i64>,
    /// Each modality's `[rows, width, start, stop]` sample rows.
    rows: Vec<i64>,
    /// Elements of each modality's native draw.
    draws: Vec<i64>,
    condition_noise: i64,
    text_rows: i64,
    workspace: i64,
}

/// The sizing methods, each called with the loaded model first.
struct SizingMethods {
    legal_frame_count: Function,
    max_condition_rows: Function,
    refusal: Function,
    condition_rows: Function,
    layout_size: Function,
    holds: Function,
    condition_layout: Function,
    latent_shape: Function,
    state_names: Function,
    state_dtypes: Function,
    state_elements: Function,
    sample_rows: Function,
    noise_elements: Function,
    condition_noise_elements: Function,
    text_condition_rows: Function,
    workspace_elements: Function,
    products: Function,
    reader_products: Function,
    /// The column methods of the `Outputs` the two product methods return:
    /// owners, names, dtypes, ranks, extents, variable and raster axes.
    output_columns: [Function; 7],
    unit_frames: Function,
    clip_output: Function,
    clip_rows: Function,
    overlap_elements: Function,
    overlap_dtype: Function,
    track_samples: Function,
    audio_units: Function,
    audio_latent_frames: Function,
}

/// A video model's facts and sizing rules.
///
/// Admission and capacity planning size a request before any runner of its
/// layout exists, so the facts and the sizing functions live in this record
/// rather than in a `DenoiserRunner`. Planning, admission and the runners
/// read them through its read-only methods.
pub(crate) struct VideoModel {
    /// The component of the deployment's video denoiser.
    pub(crate) denoiser: String,
    pub(crate) facts: VideoFacts,
    loaded: Handle,
    methods: SizingMethods,
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
        .map_err(|_| Error::Interface(format!("the video model reports an invalid {what}")))
}

fn text(value: &str) -> Any {
    Any::from(tvm_ffi::String::from(value))
}

fn strings(function: &Function, value: Any) -> Result<Vec<String>> {
    let values: tvm_ffi::Array<Any> = value
        .try_into()
        .map_err(|error| Error::numerical(function.name(), &error))?;
    values
        .iter()
        .map(|value| {
            tvm_ffi::String::try_from(value)
                .map(|value| value.as_str().to_owned())
                .map_err(|error| Error::numerical(function.name(), &error))
        })
        .collect()
}

fn layout_arguments(layout: &VideoLayout) -> Any {
    layout.descriptor()
}

impl VideoModel {
    /// Resolve the sizing methods and read the facts of `loaded` for the
    /// `denoiser` component.
    pub(crate) fn new(loaded: &Handle, denoiser: String) -> Result<Self> {
        let of_denoiser = |name: &str| Interpreter::import(&format!("{DENOISER}.{name}"));
        let of_decoder = |name: &str| Interpreter::import(&format!("{DECODER}.{name}"));
        let methods = SizingMethods {
            legal_frame_count: of_denoiser("legal_frame_count")?,
            max_condition_rows: of_denoiser("max_condition_rows")?,
            refusal: of_denoiser("refusal")?,
            condition_rows: of_denoiser("condition_rows")?,
            layout_size: of_denoiser("layout_size")?,
            holds: of_denoiser("holds")?,
            condition_layout: of_denoiser("condition_layout")?,
            latent_shape: of_denoiser("latent_shape")?,
            state_names: of_denoiser("state_names")?,
            state_dtypes: of_denoiser("state_dtypes")?,
            state_elements: of_denoiser("state_elements")?,
            sample_rows: of_denoiser("sample_rows")?,
            noise_elements: of_denoiser("noise_elements")?,
            condition_noise_elements: of_denoiser("condition_noise_elements")?,
            text_condition_rows: of_denoiser("text_condition_rows")?,
            workspace_elements: of_denoiser("workspace_elements")?,
            products: of_denoiser("products")?,
            reader_products: of_denoiser("reader_products")?,
            output_columns: [
                "owners",
                "names",
                "dtypes",
                "ranks",
                "extents",
                "variable_axes",
                "raster_axes",
            ]
            .map(|column| Interpreter::import(&format!("{OUTPUTS}.{column}")))
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::Invariant("the outputs have seven columns".into()))?,
            unit_frames: of_decoder("unit_frames")?,
            clip_output: of_decoder("clip_output")?,
            clip_rows: of_decoder("clip_rows")?,
            overlap_elements: of_decoder("overlap_elements")?,
            overlap_dtype: of_decoder("overlap_dtype")?,
            track_samples: of_decoder("track_samples")?,
            audio_units: of_decoder("audio_units")?,
            audio_latent_frames: of_decoder("audio_latent_frames")?,
        };

        // A fact of the denoiser component, read once.
        let fact = |name: &str| -> Result<(Function, Any)> {
            let function = of_denoiser(name)?;
            let value = function.call_values(&[loaded.to_any(), text(&denoiser)])?;
            Ok((function, value))
        };
        let integer = |name: &str| -> Result<i64> {
            let (function, value) = fact(name)?;
            i64::try_from(value).map_err(|error| Error::numerical(function.name(), &error))
        };
        let list = |name: &str| -> Result<Vec<i64>> {
            let (function, value) = fact(name)?;
            integers(value).map_err(|error| Error::numerical(function.name(), &error))
        };
        let names = |name: &str| -> Result<Vec<String>> {
            let (function, value) = fact(name)?;
            strings(&function, value)
        };
        let flag = |name: &str| -> Result<bool> {
            let (function, value) = fact(name)?;
            bool::try_from(value).map_err(|error| Error::numerical(function.name(), &error))
        };
        // A fact of the decoders, which the model holds one of each.
        let rate = |name: &str| -> Result<u32> {
            let function = of_decoder(name)?;
            let value: i64 = function.call_as(&[loaded.view()])?;
            narrow(value, name)
        };

        let canvases = list("canvases")?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[width, height]| {
                Ok(Canvas {
                    width: narrow(width, "canvas")?,
                    height: narrow(height, "canvas")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let shifts = {
            let (function, value) = fact("schedule_shifts")?;
            let values: tvm_ffi::Array<Any> = value
                .try_into()
                .map_err(|error| Error::numerical(function.name(), &error))?;
            values
                .iter()
                .map(|value| {
                    f64::try_from(value).map_err(|error| Error::numerical(function.name(), &error))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let [video_shift, audio_shift] = shifts[..] else {
            return Err(Error::Interface(
                "schedule shifts are [video, audio]".into(),
            ));
        };
        let sample_dtype = {
            let (function, value) = fact("sample_dtype")?;
            let name = tvm_ffi::String::try_from(value)
                .map_err(|error| Error::numerical(function.name(), &error))?;
            DType::parse(name.as_str())?
        };
        let rows = integer("max_sequence_rows")?;
        let frames = match list("frame_clips")?[..] {
            [
                min_frames,
                clip_frames,
                clip_latent_frames,
                tail_latent_frames,
            ] => FrameClips {
                min_frames: narrow(min_frames, "frame rule")?,
                clip_frames: narrow(clip_frames, "frame rule")?,
                clip_latent_frames: narrow(clip_latent_frames, "frame rule")?,
                tail_latent_frames: narrow(tail_latent_frames, "frame rule")?,
            },
            _ => {
                return Err(Error::Interface(
                    "the frame rule is [min_frames, clip_frames, clip_latent_frames, \
                     tail_latent_frames]"
                        .into(),
                ));
            }
        };
        let facts = VideoFacts {
            modalities: names("modalities")?,
            num_steps: narrow(integer("num_steps")?, "step count")?,
            tasks: names("tasks")?,
            text_only: flag("text_only")?,
            video_shift,
            audio_shift,
            canvases,
            max_sequence_rows: (rows > 0).then_some(rows as u32),
            sample_dtype,
            text_condition_width: narrow(integer("text_condition_width")?, "width")?,
            frame_rate: rate("frame_rate")?,
            sample_rate: rate("sample_rate")?,
            frames,
        };
        Ok(Self {
            denoiser,
            facts,
            loaded: loaded.clone(),
            methods,
        })
    }

    /// Call a sizing method of the denoiser component.
    fn denoiser(&self, method: &Function, arguments: Vec<Any>) -> Result<Any> {
        let mut all = vec![self.loaded.to_any(), text(&self.denoiser)];
        all.extend(arguments);
        method.call_values(&all)
    }

    /// Call a sizing method of the decoders.
    fn decoder(&self, method: &Function, arguments: Vec<Any>) -> Result<Any> {
        let mut all = vec![self.loaded.to_any()];
        all.extend(arguments);
        method.call_values(&all)
    }

    fn integers(&self, method: &Function, value: Any) -> Result<Vec<i64>> {
        integers(value).map_err(|error| Error::numerical(method.name(), &error))
    }

    fn integer(&self, method: &Function, value: Any) -> Result<i64> {
        i64::try_from(value).map_err(|error| Error::numerical(method.name(), &error))
    }

    /// The handshake's description of the denoiser on a deployment that
    /// prepares and admits `canvases`: its tasks as it states them.
    pub(crate) fn denoiser_info(&self, canvases: &[Canvas]) -> VideoDenoiserInfo {
        VideoDenoiserInfo {
            tasks: self.facts.tasks.clone(),
            schedule_points: self.facts.num_steps + 1,
            video_shift: self.facts.video_shift,
            audio_shift: self.facts.audio_shift,
            frame_rate: self.facts.frame_rate,
            audio_sample_rate: self.facts.sample_rate,
            frames: self.facts.frames,
            canvases: canvases.to_vec(),
            max_sequence_rows: self.facts.max_sequence_rows,
        }
    }

    /// Packed rows of the largest condition set a request generating
    /// `frames` frames on `canvas` may bring, as the denoiser packs them
    /// (`VideoDenoiser.max_conditions`); zero when it admits none.
    pub(crate) fn max_condition_rows(&self, frames: u32, canvas: Canvas) -> Result<u32> {
        let method = &self.methods.max_condition_rows;
        let value = self.denoiser(
            method,
            [frames, canvas.width, canvas.height]
                .map(|value| Any::from(i64::from(value)))
                .to_vec(),
        )?;
        narrow(self.integer(method, value)?, "condition capacity")
    }

    pub(crate) fn legal_frame_count(&self, frames: u32) -> Result<u32> {
        let method = &self.methods.legal_frame_count;
        let value = self.denoiser(method, vec![Any::from(i64::from(frames))])?;
        narrow(self.integer(method, value)?, "frame count")
    }

    /// The smallest layout holding `size`.
    pub(crate) fn layout_size(&self, size: &VideoSize) -> Result<VideoLayout> {
        let value = self
            .denoiser(&self.methods.layout_size, size_arguments(size))
            .map_err(invalid_size)?;
        layout_of(value)
    }

    /// A sizing method of the denoiser component as a call for a job to
    /// make.
    fn asked(&self, method: &Function, arguments: Vec<Argument>) -> MethodCall {
        let mut all = vec![
            Argument::Handle(self.loaded.clone()),
            Argument::Text(self.denoiser.clone()),
        ];
        all.extend(arguments);
        MethodCall::new(method, all)
    }

    /// Record the call asking why the denoiser does not serve `size`: the
    /// job's completion carries the denoiser's reason, empty for a size it
    /// serves. A size it does not serve is the request's refusal, so the
    /// call answers rather than fails the job.
    pub(crate) fn ask_refusal(&self, size: &VideoSize, job: &Job) {
        let arguments = size_values(size).into();
        job.call_texts(self.asked(&self.methods.refusal, arguments));
    }

    /// Record the call asking the denoiser rows the conditions of `size`
    /// take, as the denoiser packs them: the job's completion carries the
    /// count. Asked only for a size the denoiser serves.
    pub(crate) fn ask_condition_rows(&self, size: &VideoSize, job: &Job) {
        let arguments = size_values(size).into();
        job.call_returning(self.asked(&self.methods.condition_rows, arguments));
    }

    /// Record the call asking the smallest layout holding `size`
    /// ([`Self::layout_size`] as a call of `job`): its completion carries
    /// the layout's extents.
    pub(crate) fn ask_layout_size(&self, size: &VideoSize, job: &Job) {
        let arguments = size_values(size).into();
        job.call_returning(self.asked(&self.methods.layout_size, arguments));
    }

    /// Record the call asking whether `layout` holds `size`: the job's
    /// completion carries one for a layout that does, zero otherwise. A size
    /// the model rejects fails the job with the model's error
    /// ([`invalid_size`]).
    pub(crate) fn ask_holds(&self, layout: &VideoLayout, size: &VideoSize, job: &Job) {
        let mut arguments = vec![Argument::Integers(layout.values().to_vec())];
        arguments.extend(size_values(size));
        job.call_returning(self.asked(&self.methods.holds, arguments));
    }

    /// `layout` widened so its condition region holds `rows`.
    pub(crate) fn condition_layout(&self, layout: &VideoLayout, rows: u32) -> Result<VideoLayout> {
        layout_of(self.denoiser(
            &self.methods.condition_layout,
            vec![layout_arguments(layout), Any::from(i64::from(rows))],
        )?)
    }

    /// A modality's complete sample shape in `layout`.
    pub(crate) fn latent_shape(&self, modality: &str, layout: &VideoLayout) -> Result<Vec<u64>> {
        let method = &self.methods.latent_shape;
        let value = self.denoiser(method, vec![text(modality), layout_arguments(layout)])?;
        Ok(self
            .integers(method, value)?
            .into_iter()
            .map(|extent| extent.max(0) as u64)
            .collect())
    }

    /// Elements of a modality's complete sample in `layout`.
    pub(crate) fn latent_elements(&self, modality: &str, layout: &VideoLayout) -> Result<u64> {
        Ok(self.latent_shape(modality, layout)?.iter().product())
    }

    /// The request state of `layout` on this rank, asked on the calling
    /// thread (startup).
    pub(crate) fn layout_fields(&self, layout: &VideoLayout) -> Result<LayoutFields> {
        let methods = &self.methods;
        let of_layout = |method: &Function| self.denoiser(method, vec![layout_arguments(layout)]);
        let counts =
            |method: &Function| -> Result<Vec<i64>> { self.integers(method, of_layout(method)?) };
        let count = |method: &Function| -> Result<i64> { self.integer(method, of_layout(method)?) };

        self.fields(LayoutAnswers {
            names: strings(&methods.state_names, of_layout(&methods.state_names)?)?,
            dtypes: strings(&methods.state_dtypes, of_layout(&methods.state_dtypes)?)?,
            elements: counts(&methods.state_elements)?,
            rows: counts(&methods.sample_rows)?,
            draws: counts(&methods.noise_elements)?,
            condition_noise: count(&methods.condition_noise_elements)?,
            text_rows: count(&methods.text_condition_rows)?,
            workspace: count(&methods.workspace_elements)?,
        })
    }

    /// Record into `job` the calls that ask the request state of `layout`
    /// on this rank; [`Self::read_layout_fields`] reads the job's
    /// completion.
    pub(crate) fn ask_layout_fields(&self, layout: &VideoLayout, job: &Job) {
        let methods = &self.methods;
        let of_layout = |method: &Function| {
            self.asked(method, vec![Argument::Integers(layout.values().to_vec())])
        };
        job.call_texts(of_layout(&methods.state_names));
        job.call_texts(of_layout(&methods.state_dtypes));
        for method in [
            &methods.state_elements,
            &methods.sample_rows,
            &methods.noise_elements,
            &methods.condition_noise_elements,
            &methods.text_condition_rows,
            &methods.workspace_elements,
        ] {
            job.call_returning(of_layout(method));
        }
    }

    /// The request state of a layout from the completion of the job that
    /// asked it ([`Self::ask_layout_fields`]).
    pub(crate) fn read_layout_fields(&self, completion: &mut Completion) -> Result<LayoutFields> {
        let count = |values: Vec<i64>| {
            values.first().copied().ok_or_else(|| {
                Error::Interface("a layout sizing function returned no value".into())
            })
        };
        self.fields(LayoutAnswers {
            names: completion.texts()?,
            dtypes: completion.texts()?,
            elements: completion.returned()?,
            rows: completion.returned()?,
            draws: completion.returned()?,
            condition_noise: count(completion.returned()?)?,
            text_rows: count(completion.returned()?)?,
            workspace: count(completion.returned()?)?,
        })
    }

    /// A layout's request state from the model's answers.
    fn fields(&self, answers: LayoutAnswers) -> Result<LayoutFields> {
        let extent = |value: i64| value.max(0) as u64;
        let LayoutAnswers {
            names,
            dtypes,
            elements,
            rows,
            draws,
            condition_noise,
            text_rows,
            workspace,
        } = answers;
        let state = names
            .into_iter()
            .zip(dtypes)
            .zip(elements)
            .map(|((name, dtype), elements)| {
                Ok(StateField {
                    sample: self.facts.modalities.contains(&name),
                    name,
                    dtype: DType::parse(&dtype)?,
                    elements: extent(elements),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let rows: Vec<u64> = rows.into_iter().map(extent).collect();
        let samples = self
            .facts
            .modalities
            .iter()
            .zip(rows.as_chunks::<4>().0)
            .map(|(name, &[rows, width, start, stop])| {
                (
                    name.clone(),
                    SampleRows {
                        rows,
                        width,
                        local: (start, stop),
                    },
                )
            })
            .collect();
        let noise = self
            .facts
            .modalities
            .iter()
            .cloned()
            .zip(draws.into_iter().map(extent))
            .collect();
        Ok(LayoutFields {
            state,
            samples,
            noise,
            condition_noise: extent(condition_noise),
            text_condition: (
                extent(text_rows),
                u64::from(self.facts.text_condition_width),
            ),
            workspace: extent(workspace),
        })
    }

    /// The `(frames, canvas)` clip output unit `unit` of a video decodes
    /// to: the native output of the video's clip of that index.
    pub(crate) fn clip_output(
        &self,
        frames: u32,
        canvas: Canvas,
        unit: u32,
    ) -> Result<(u32, Canvas)> {
        let method = &self.methods.clip_output;
        let value = self.decoder(
            method,
            vec![
                Any::from(i64::from(frames)),
                Any::from(i64::from(canvas.width)),
                Any::from(i64::from(canvas.height)),
                Any::from(i64::from(unit)),
            ],
        )?;
        let values = self.integers(method, value)?;
        let [frames, width, height] = values[..] else {
            return Err(Error::Interface(
                "a clip output is [frames, width, height]".into(),
            ));
        };
        Ok((
            narrow(frames, "clip output")?,
            Canvas {
                width: narrow(width, "clip output")?,
                height: narrow(height, "clip output")?,
            },
        ))
    }

    /// Latent rows the clip input of unit `unit` of a `frames`-frame video
    /// at `canvas` reads.
    pub(crate) fn clip_rows(&self, frames: u32, canvas: Canvas, unit: u32) -> Result<u64> {
        let method = &self.methods.clip_rows;
        let value = self.decoder(
            method,
            vec![
                Any::from(i64::from(frames)),
                Any::from(i64::from(canvas.width)),
                Any::from(i64::from(canvas.height)),
                Any::from(i64::from(unit)),
            ],
        )?;
        Ok(self.integers(method, value)?.len() as u64)
    }

    /// The overlap a unit of `canvas` carries, as `(elements, dtype)`.
    pub(crate) fn overlap(&self, canvas: Canvas) -> Result<(u64, DType)> {
        let arguments = || {
            vec![
                Any::from(i64::from(canvas.width)),
                Any::from(i64::from(canvas.height)),
            ]
        };
        let elements = &self.methods.overlap_elements;
        let count = self.integer(elements, self.decoder(elements, arguments())?)?;
        let dtype = &self.methods.overlap_dtype;
        let name = tvm_ffi::String::try_from(self.decoder(dtype, arguments())?)
            .map_err(|error| Error::numerical(dtype.name(), &error))?;
        Ok((count.max(0) as u64, DType::parse(name.as_str())?))
    }

    /// Frames of each media unit of a video of `frames` frames.
    pub(crate) fn unit_frames(&self, frames: u32) -> Result<Vec<u32>> {
        let method = &self.methods.unit_frames;
        let value = self.decoder(method, vec![Any::from(i64::from(frames))])?;
        self.integers(method, value)?
            .into_iter()
            .map(|count| narrow(count, "unit frame count"))
            .collect()
    }

    /// Samples of the audio track generated with a video of `frames`.
    pub(crate) fn track_samples(&self, frames: u32) -> Result<u64> {
        let method = &self.methods.track_samples;
        let value = self.decoder(method, vec![Any::from(i64::from(frames))])?;
        Ok(self.integer(method, value)?.max(0) as u64)
    }

    /// Each of `units` audio units' samples of a `samples`-sample track.
    pub(crate) fn audio_units(&self, samples: u64, units: u32) -> Result<Vec<Range<u64>>> {
        let method = &self.methods.audio_units;
        let value = self.decoder(
            method,
            vec![Any::from(samples as i64), Any::from(i64::from(units))],
        )?;
        Ok(self
            .integers(method, value)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[start, stop]| start.max(0) as u64..stop.max(0) as u64)
            .collect())
    }

    /// Latent frames of a `samples`-sample track.
    pub(crate) fn audio_latent_frames(&self, samples: u64) -> Result<u32> {
        let method = &self.methods.audio_latent_frames;
        let value = self.decoder(method, vec![Any::from(samples as i64)])?;
        narrow(self.integer(method, value)?, "latent frame count")
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
        let method = &self.methods.products;
        let outputs = self.denoiser(
            method,
            vec![
                Any::from(i64::from(frames)),
                Any::from(i64::from(tokens)),
                Any::from(i64::from(rows)),
                text(dtype),
                Any::from(integer_list(canvases.iter().flat_map(|canvas| {
                    [i64::from(canvas.width), i64::from(canvas.height)]
                }))),
            ],
        )?;
        let mut result: BTreeMap<String, Vec<OutputInfo>> = BTreeMap::new();
        for (component, output) in self.output_infos(outputs)? {
            result.entry(component).or_default().push(output);
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
        let method = &self.methods.reader_products;
        let value = self.denoiser(
            method,
            vec![
                Any::from(i64::from(frames)),
                Any::from(i64::from(tokens)),
                Any::from(i64::from(rows)),
            ],
        )?;
        Ok(self
            .output_infos(value)?
            .into_iter()
            .map(|(_, output)| output)
            .collect())
    }

    /// Read the outputs a product method returned (`Outputs`), column by
    /// column, as each output's component and bound.
    fn output_infos(&self, outputs: Any) -> Result<Vec<(String, OutputInfo)>> {
        let [owners, names, dtypes, ranks, extents, variable, raster] =
            &self.methods.output_columns;
        let strings = |column: &Function| -> Result<Vec<String>> {
            let values: tvm_ffi::Array<tvm_ffi::String> = column.call_as(&[(&outputs).into()])?;
            Ok(values
                .iter()
                .map(|value| value.as_str().to_owned())
                .collect())
        };
        let numbers = |column: &Function| -> Result<Vec<i64>> {
            self.integers(column, column.call_values(std::slice::from_ref(&outputs))?)
        };
        let (owners, names, dtypes) = (strings(owners)?, strings(names)?, strings(dtypes)?);
        let (ranks, extents) = (numbers(ranks)?, numbers(extents)?);
        let (variable, raster) = (numbers(variable)?, numbers(raster)?);
        let mut extents = extents.into_iter();
        let mut result = Vec::with_capacity(names.len());
        for (index, name) in names.into_iter().enumerate() {
            let component = &owners[index];
            let extent = |value: i64| {
                u32::try_from(value).map_err(|_| {
                    Error::Unsupported(format!("result {component}.{name} has an invalid extent"))
                })
            };
            let dtype = DType::from_name(&dtypes[index])
                .and_then(|dtype| uniserve_ipc::DType::try_from(dtype).ok())
                .ok_or_else(|| {
                    Error::Unsupported(format!("result {component}.{name} has no protocol dtype"))
                })?;
            let dims = extents
                .by_ref()
                .take(ranks[index] as usize)
                .enumerate()
                .map(|(axis, size)| {
                    Ok(if variable[index] == axis as i64 {
                        DimBound::Device { max: extent(size)? }
                    } else {
                        DimBound::Static(extent(size)?)
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            // A negative axis stands for an output that is not laid out at
            // the canvas.
            let raster_axes = match (raster[2 * index], raster[2 * index + 1]) {
                (height, width) if height >= 0 && width >= 0 => Some(RasterAxes {
                    height: extent(height)?,
                    width: extent(width)?,
                }),
                _ => None,
            };
            result.push((
                component.clone(),
                OutputInfo {
                    name,
                    dtype,
                    shape_bound: ShapeBound { dims },
                    raster_axes,
                },
            ));
        }
        Ok(result)
    }
}

pub(crate) fn size_arguments(size: &VideoSize) -> Vec<Any> {
    vec![
        size.descriptor(),
        size.conditions_descriptor(),
        size.spans_descriptor(),
    ]
}

/// A size as the arguments of a recorded call: what [`size_arguments`]
/// passes, each list as its integers.
pub(super) fn size_values(size: &VideoSize) -> [Argument; 3] {
    [
        Argument::Integers(size.values()),
        Argument::Integers(size.condition_values()),
        Argument::Integers(size.span_values()),
    ]
}

fn layout_of(value: Any) -> Result<VideoLayout> {
    let values = integers(value).map_err(|error| Error::numerical(DENOISER, &error))?;
    VideoLayout::from_values(&values)
}

/// A size the model rejects is the request's error, not the worker's.
pub(crate) fn invalid_size(error: Error) -> Error {
    match error {
        Error::Numerical { kind, message, .. } if kind == "InputError" || kind == "ValueError" => {
            Error::Invalid(message)
        }
        error => error,
    }
}
