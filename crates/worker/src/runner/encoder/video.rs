//! A video model's encoders: the runners of its text encoder component (the
//! prompt encoder and, for conditioned tasks, the vision tower bound with it),
//! its condition latent encoders ([`Conditions`]), and the encoding of a
//! request's prompt, vision blocks and conditions into denoiser rows.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::video::plan::CANVAS_MULTIPLE;
use uniserve_core::{Canvas, VideoCondition};
use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{Batch as BatchPlan, CallKind, MediaCall};

use super::{
    ENCODER, EncoderConfig, EncoderInfo, EncoderMethods, EncoderRunner, OwnedContext, PromptVision,
    TextSize,
};
use crate::config::Device;
use crate::cuda_graph::{GraphPool, GraphTable, Lookup};
use crate::executor::batch::CallState;
use crate::executor::products::{OutputShape, Placement, bind_outputs};
use crate::executor::video::{
    MediaBatch, Parts, VideoCall, bind_rows, decode_range, product_name, publish, publish_bound,
    row_width, stream_of,
};
use crate::ffi::{Backing, DType, Function, Handle, Interpreter, integer_list, integers};
use crate::lane::{Capture, Completion, Job};
use crate::runner::RunnerSetup;
use crate::runner::methods::{Argument, MethodCall, Methods, RunnerClass};
use crate::storage::PinnedPool;
use crate::storage::memory::{Storage, StorageView};
use crate::{Error, Result};

/// A condition encoding was recorded before the job that constructs its
/// encoder completed.
const MISSING_ENCODER: Error =
    Error::State("a condition encoding was recorded before its encoder was constructed");

/// A condition encoding was recorded before its posterior draw was filled.
const MISSING_POSTERIOR: Error =
    Error::State("a condition encoding was recorded before its posterior draw was filled");

/// The posterior draws [`Conditions`] holds at most: one per condition size,
/// the oldest released first. Sixteen draws of five-second 16:9 references
/// take about 230 MB of page-locked memory.
const POSTERIORS: usize = 16;

/// A condition size's posterior draw.
enum Posterior {
    /// The video encoder takes no draw: its latent is the posterior's mean.
    None,
    /// Page-locked storage of the draw, not yet filled.
    Undrawn(StorageView),
    /// The filled draw, which every encoding of the size reads.
    Drawn(StorageView),
}

/// The condition latent encoders of a video model: the Python runners of
/// its video encoder and of its audio encoder, one per input shapes.
///
/// A video encoder whose latent encoder tiles encodes every still frame as
/// tiles of one shape, each many small kernels, so its eager encoding is
/// bound by the host launching them. With graph pools, startup captures one
/// tile's encoding ([`Self::capture_tile`]), and every video runner
/// constructed afterwards replays it for each still-frame tile of that
/// shape, inside its eager forward; video clips' tiles stay eager, since
/// their encoding already keeps the device busy and a graph would keep a
/// clip tile's activations resident.
pub(super) struct Conditions {
    video: RunnerClass,
    audio: RunnerClass,
    loaded: Handle,
    /// Frames of each visual unit of a condition of some frame count.
    unit_frames: Function,
    /// What `unit_frames` answered, by frame count.
    frames: HashMap<i64, Vec<i64>>,
    /// The runner of each call's input shapes, constructed by the job
    /// before the first call's ([`Self::setup`]).
    video_runners: HashMap<Vec<i64>, Methods>,
    audio_runners: HashMap<Vec<i64>, Methods>,
    /// Elements of the posterior draw of a condition of some size, 0 when
    /// the video encoder takes none.
    posterior_elements: Function,
    /// Fills a condition size's posterior draw.
    draw_posterior: Function,
    /// The device whose copies read the draws' page-locked storage.
    device: DeviceId,
    /// The posterior draw of each condition size `[frames, width, height]`.
    /// The draw depends on the size alone (the reference samples every
    /// condition under one fixed seed), so one draw serves every unit,
    /// round and request of the size.
    posteriors: HashMap<[i64; 3], Posterior>,
    /// The sizes of `posteriors`, oldest first; at most [`POSTERIORS`].
    posterior_order: VecDeque<[i64; 3]>,
    /// `[1, height, width, 3]` of a still frame one tile in size; empty when
    /// the video encoder does not tile still frames.
    still_tile: Vec<i64>,
    /// The constructor of a tile runner (`EncoderRunner.tile`) with the
    /// video runners' fixed arguments, and the forward a tile's graph
    /// captures (`EncoderRunner.encode_tile`).
    tile_runner: MethodCall,
    encode_tile: Function,
    /// The captured still-frame tile encoding, once startup captured it.
    tile: Option<StillTile>,
}

/// A captured still-frame tile encoding: the Python tile runner, whose
/// input and output the graph addresses, and the graph.
struct StillTile {
    runner: Methods,
    graphs: GraphTable<Vec<i64>>,
}

/// The encoder of a condition's medium.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConditionMedium {
    Video,
    Audio,
}

/// One visual unit as the encoder takes it: `[pixel_offset, frames, width,
/// height, window, first_row, stop_row]`: its condition's first pixel row,
/// frame count and raster, the index of the encoder window the unit's frames
/// are, and the run of the frames' rows of 32-pixel latent patches it
/// encodes. A reference image's band reads its whole frame (window 0) and
/// encodes the band's patch rows; every other unit encodes all of them.
pub(crate) type ConditionUnit = [i64; 7];

/// What a condition encoding call encodes on this rank.
pub(crate) enum ConditionInput {
    /// Visual units ([`ConditionUnit`]), written to rows `region` of the
    /// round's `rows`.
    Units {
        units: Vec<ConditionUnit>,
        rows: u64,
        region: std::ops::Range<u64>,
    },
    /// Audio tracks of these sample counts, written to `rows` rows.
    Tracks { lengths: Vec<u64>, rows: u64 },
}

/// What the job on the lane asks the library for before a batch's
/// condition encodings are recorded.
pub(crate) enum ConditionSetup {
    /// The frames of each visual unit of conditions of these frame counts.
    Frames(Vec<i64>),
    /// The encoders of these input shapes.
    Runners(Vec<(ConditionMedium, Vec<i64>)>),
    /// The posterior draw sizes of conditions of these sizes, `[frames,
    /// width, height]`.
    Posteriors(Vec<[i64; 3]>),
    /// The posterior draws of conditions of these sizes, filled.
    Draws(Vec<[i64; 3]>),
}

impl Conditions {
    fn new(video: RunnerSetup<'_>, audio: RunnerSetup<'_>) -> Result<Self> {
        let method = |name: &str| Interpreter::import(&format!("{ENCODER}.{name}"));
        // Condition encoders are constructed per input shape while serving;
        // the natively built kernels they launch load now, so no encoding
        // compiles one.
        method("load_condition_kernels")?.call(&[video.loaded.view()])?;
        let still_tile = integers(method("still_tile")?.call_values(&[
            video.loaded.to_any(),
            Any::from(tvm_ffi::String::from(video.path)),
        ])?)
        .map_err(|error| Error::numerical(&format!("{ENCODER}.still_tile"), &error))?;
        Ok(Self {
            unit_frames: method("unit_frames")?,
            loaded: video.loaded.clone(),
            video: RunnerClass::new(ENCODER, "encode", video.arguments())?,
            audio: RunnerClass::new(ENCODER, "encode", audio.arguments())?,
            frames: HashMap::new(),
            video_runners: HashMap::new(),
            audio_runners: HashMap::new(),
            posterior_elements: method("posterior_elements")?,
            draw_posterior: method("draw_posterior")?,
            device: video.device,
            posteriors: HashMap::new(),
            posterior_order: VecDeque::new(),
            still_tile,
            tile_runner: MethodCall::new(&method("tile")?, video.arguments()),
            encode_tile: method("encode_tile")?,
            tile: None,
        })
    }

    /// The condition sizes `[frames, width, height]` of the visual units
    /// of `inputs`, each once.
    fn unit_sizes(inputs: &[ConditionInput]) -> Vec<[i64; 3]> {
        let mut sizes: Vec<[i64; 3]> = inputs
            .iter()
            .filter_map(|input| match input {
                ConditionInput::Units { units, .. } => Some(units),
                ConditionInput::Tracks { .. } => None,
            })
            .flatten()
            .map(|&[_, frames, width, height, ..]| [frames, width, height])
            .collect();
        sizes.sort_unstable();
        sizes.dedup();
        sizes
    }

    /// Hold `posterior` for `size`, releasing the oldest draws past
    /// [`POSTERIORS`] that no size of `current` (the sizes the batch being
    /// set up encodes) reads. A job that already recorded a released draw
    /// keeps its storage alive.
    fn hold_posterior(&mut self, size: [i64; 3], posterior: Posterior, current: &[[i64; 3]]) {
        if self.posteriors.insert(size, posterior).is_none() {
            self.posterior_order.push_back(size);
        }
        while self.posterior_order.len() > POSTERIORS {
            let Some(index) = self
                .posterior_order
                .iter()
                .position(|held| !current.contains(held))
            else {
                break;
            };
            if let Some(released) = self.posterior_order.remove(index) {
                self.posteriors.remove(&released);
            }
        }
    }

    /// Capture the encoding of one still-frame tile into `pool`, the
    /// encoder's graph pool on `device`, on `stream`, the stream every
    /// video runner computes on. Each step is a job `perform` performs on
    /// the encoder's lane: the construction of a tile runner, then its eager
    /// encoding, which compiles the tile's kernels outside the capture, and
    /// the capture. Nothing is captured for a video encoder that does not
    /// tile still frames.
    fn capture_tile(
        &mut self,
        pool: &GraphPool,
        stream: &Stream,
        device: DeviceId,
        perform: &mut dyn FnMut(Job) -> Result<Completion>,
    ) -> Result<()> {
        if self.still_tile.is_empty() || self.tile.is_some() {
            return Ok(());
        }
        let job = Job::new();
        job.call_handle(self.tile_runner.clone());
        let runner = self.video.constructed(perform(job)?.handle()?);

        let encode = MethodCall::new(
            &self.encode_tile,
            vec![Argument::Handle(runner.runner().clone())],
        );
        let job = Job::new();
        job.call(encode.clone());
        job.capture(Capture {
            functions: runner.graphs.clone(),
            pool: pool.value.clone(),
            peers: Vec::new(),
            stream: stream.raw(),
            device,
            method: encode,
            breaks: None,
        });
        let graph = perform(job)?.captured()?;
        let mut graphs = GraphTable::new(
            format!("{ENCODER}.encode_tile"),
            runner.graphs.release.clone(),
        );
        graphs.insert(self.still_tile.clone(), None, graph)?;
        graphs.seal();
        self.tile = Some(StillTile { runner, graphs });
        Ok(())
    }

    /// The tile arguments a video runner is constructed with: the tile
    /// runner and the graph of its captured tile encoding, or none before
    /// startup captured them.
    fn tile_arguments(&self) -> Result<[Argument; 2]> {
        let Some(tile) = &self.tile else {
            return Ok([Argument::None, Argument::None]);
        };
        match tile.graphs.lookup(&self.still_tile, None)? {
            Lookup::Replay(graph) => Ok([
                Argument::Handle(tile.runner.runner().clone()),
                Argument::Handle(graph.handle().clone()),
            ]),
            Lookup::Capture => Err(Error::Invariant(
                "a captured still-frame tile encoding is resident".into(),
            )),
        }
    }

    /// The input shapes of a call encoding `units`: each unit's frames at
    /// its raster. `None` while the unit frames of one of the conditions
    /// were not asked for.
    fn unit_shapes(&self, units: &[ConditionUnit]) -> Result<Option<Vec<i64>>> {
        let mut shapes = Vec::with_capacity(units.len() * 4);
        for &[_, frames, width, height, unit, _, _] in units {
            let Some(lengths) = self.frames.get(&frames) else {
                return Ok(None);
            };
            let length = lengths.get(unit as usize).copied().ok_or_else(|| {
                Error::Invalid("a condition unit lies outside its condition".into())
            })?;
            shapes.extend([length, height, width, 3]);
        }
        Ok(Some(shapes))
    }

    /// The input shapes of a call encoding tracks of `lengths` samples.
    fn track_shapes(lengths: &[u64]) -> Vec<i64> {
        lengths
            .iter()
            .flat_map(|&length| [length as i64, 2])
            .collect()
    }

    /// Record into `job` what the encodings of `inputs` lack: the unit
    /// frames of conditions whose frame count is new, and once those are
    /// known the construction of the encoders of new input shapes. `None`
    /// when the encodings can be recorded.
    ///
    /// Both are calls of the library, so the job makes them; a constructed
    /// encoder is an argument of its forward, which is why the forward is
    /// recorded only from the job's completion ([`Self::accept`]).
    fn setup(&self, inputs: &[ConditionInput], job: &Job) -> Result<Option<ConditionSetup>> {
        let mut frames: Vec<i64> = inputs
            .iter()
            .filter_map(|input| match input {
                ConditionInput::Units { units, .. } => Some(units),
                ConditionInput::Tracks { .. } => None,
            })
            .flatten()
            .map(|unit| unit[1])
            .filter(|count| !self.frames.contains_key(count))
            .collect();
        frames.sort_unstable();
        frames.dedup();
        if !frames.is_empty() {
            for &count in &frames {
                job.call_returning(MethodCall::new(
                    &self.unit_frames,
                    vec![
                        Argument::Handle(self.loaded.clone()),
                        Argument::Integer(count),
                    ],
                ));
            }
            return Ok(Some(ConditionSetup::Frames(frames)));
        }

        let mut runners: Vec<(ConditionMedium, Vec<i64>)> = Vec::new();
        for input in inputs {
            let (medium, shapes) = match input {
                ConditionInput::Units { units, .. } => {
                    let shapes = self.unit_shapes(units)?.ok_or_else(|| {
                        Error::Invariant("a condition's unit frames were not asked for".into())
                    })?;
                    (ConditionMedium::Video, shapes)
                }
                ConditionInput::Tracks { lengths, .. } => {
                    (ConditionMedium::Audio, Self::track_shapes(lengths))
                }
            };
            let (class, constructed) = match medium {
                ConditionMedium::Video => (&self.video, &self.video_runners),
                ConditionMedium::Audio => (&self.audio, &self.audio_runners),
            };
            let recorded = runners
                .iter()
                .any(|(other, known)| *other == medium && *known == shapes);
            if constructed.contains_key(&shapes) || recorded {
                continue;
            }
            // A video runner replays the captured still-frame tile.
            let mut arguments = vec![Argument::Integers(shapes.clone())];
            if medium == ConditionMedium::Video {
                arguments.extend(self.tile_arguments()?);
            }
            job.call_handle(class.construction(arguments));
            runners.push((medium, shapes));
        }
        if !runners.is_empty() {
            return Ok(Some(ConditionSetup::Runners(runners)));
        }

        // A condition size's posterior draw is sized, then filled, once;
        // both are calls of the library, so the job makes them.
        let sizes = Self::unit_sizes(inputs);
        let unknown: Vec<[i64; 3]> = sizes
            .iter()
            .copied()
            .filter(|size| !self.posteriors.contains_key(size))
            .collect();
        if !unknown.is_empty() {
            for &[frames, width, height] in &unknown {
                job.call_returning(MethodCall::new(
                    &self.posterior_elements,
                    vec![
                        Argument::Handle(self.loaded.clone()),
                        Argument::Integer(frames),
                        Argument::Integer(width),
                        Argument::Integer(height),
                    ],
                ));
            }
            return Ok(Some(ConditionSetup::Posteriors(unknown)));
        }
        let mut undrawn = Vec::new();
        for size in sizes {
            if let Some(Posterior::Undrawn(noise)) = self.posteriors.get(&size) {
                let [frames, width, height] = size;
                job.call(MethodCall::new(
                    &self.draw_posterior,
                    vec![
                        Argument::Handle(self.loaded.clone()),
                        Argument::Integer(frames),
                        Argument::Integer(width),
                        Argument::Integer(height),
                        Argument::Tensor(noise.clone()),
                    ],
                ));
                undrawn.push(size);
            }
        }
        Ok((!undrawn.is_empty()).then_some(ConditionSetup::Draws(undrawn)))
    }

    /// Keep what the job of `setup` returned.
    fn accept(&mut self, setup: ConditionSetup, completion: &mut Completion) -> Result<()> {
        match setup {
            ConditionSetup::Frames(frames) => {
                for count in frames {
                    self.frames.insert(count, completion.returned()?);
                }
            }
            ConditionSetup::Runners(runners) => {
                for (medium, shapes) in runners {
                    let runner = completion.handle()?;
                    match medium {
                        ConditionMedium::Video => self
                            .video_runners
                            .insert(shapes, self.video.constructed(runner)),
                        ConditionMedium::Audio => self
                            .audio_runners
                            .insert(shapes, self.audio.constructed(runner)),
                    };
                }
            }
            ConditionSetup::Posteriors(sizes) => {
                for &size in &sizes {
                    let elements = completion.returned()?.first().copied().ok_or_else(|| {
                        Error::Interface(format!(
                            "{} returned no count",
                            self.posterior_elements.name()
                        ))
                    })?;
                    let posterior = if elements == 0 {
                        Posterior::None
                    } else {
                        let memory = Storage::pinned(self.device, elements as usize * 4)?;
                        Posterior::Undrawn(StorageView::contiguous(
                            memory as Arc<dyn Backing>,
                            0,
                            DType::F32,
                            &[elements],
                        )?)
                    };
                    self.hold_posterior(size, posterior, &sizes);
                }
            }
            ConditionSetup::Draws(sizes) => {
                // The job that filled them completed, so later encodings read
                // complete draws.
                for size in sizes {
                    if let Some(Posterior::Undrawn(noise)) = self.posteriors.remove(&size) {
                        self.posteriors.insert(size, Posterior::Drawn(noise));
                    }
                }
            }
        }
        Ok(())
    }

    /// Encode visual units ([`ConditionUnit`]) into consecutive rows of
    /// `out`.
    fn encode_video(
        &mut self,
        pixels: &StorageView,
        units: &[ConditionUnit],
        out: &StorageView,
        job: &Job,
    ) -> Result<()> {
        let runner = self
            .unit_shapes(units)?
            .and_then(|shapes| self.video_runners.get(&shapes))
            .ok_or(MISSING_ENCODER)?;
        let mut arguments = vec![
            Argument::Handle(runner.runner().clone()),
            Argument::Tensor(pixels.clone()),
            Argument::Tensor(out.clone()),
            Argument::Integers(units.iter().flatten().copied().collect()),
        ];
        // Each unit reads its condition size's posterior draw; an encoder
        // that takes no draw reads none.
        for &[_, frames, width, height, ..] in units {
            match self.posteriors.get(&[frames, width, height]) {
                Some(Posterior::Drawn(noise)) => arguments.push(Argument::Tensor(noise.clone())),
                Some(Posterior::None) => {}
                _ => return Err(MISSING_POSTERIOR),
            }
        }
        job.call(MethodCall::new(&self.video.forward, arguments));
        Ok(())
    }

    /// Encode audio tracks of `lengths` samples into consecutive rows of
    /// `out`.
    fn encode_audio(
        &mut self,
        samples: &StorageView,
        lengths: &[u64],
        out: &StorageView,
        job: &Job,
    ) -> Result<()> {
        let runner = self
            .audio_runners
            .get(&Self::track_shapes(lengths))
            .ok_or(MISSING_ENCODER)?;
        job.call(MethodCall::new(
            &self.audio.forward,
            vec![
                Argument::Handle(runner.runner().clone()),
                Argument::Tensor(samples.clone()),
                Argument::Tensor(out.clone()),
                Argument::Integers(lengths.iter().map(|&length| length as i64).collect()),
            ],
        ));
        Ok(())
    }

    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        let mut calls = self.video.kernel_calls(self.video_runners.values());
        calls.extend(self.audio.kernel_calls(self.audio_runners.values()));
        calls
    }

    /// Close the runners, then the tile runner and its graph, which the
    /// video runners replay. The caller drained the encoders' stream.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let video = self.video.close(aborted, self.video_runners.values());
        let mut result = video.and(self.audio.close(aborted, self.audio_runners.values()));
        if let Some(tile) = &mut self.tile {
            result = result.and(self.video.close(aborted, [&tile.runner]));
            result = result.and(tile.graphs.close(aborted));
        }
        result
    }
}

/// What a video model's text encoder component is bound with.
pub(crate) struct TextSetup<'a> {
    pub(crate) loaded: &'a Handle,
    pub(crate) component: &'a str,
    /// The component's root module path.
    pub(crate) path: &'a str,
    pub(crate) device: DeviceId,
    pub(crate) stream: Stream,
    /// The stream's shared work areas, a Python `Scratch`.
    pub(crate) scratch: Handle,
    pub(crate) attention: String,
    /// Prompt capacities, the longest admitted prompt and, as the row
    /// budget, the most vision patch rows one request carries.
    pub(crate) config: EncoderConfig,
    /// Whether the component also encodes condition media with its vision
    /// tower.
    pub(crate) vision: bool,
    /// The model dtype the encoders compute in.
    pub(crate) dtype: DType,
}

impl EncoderRunner {
    /// The runners of a video model's text encoder component: the prompt
    /// encoder, which owns the execution context, and, for conditioned
    /// tasks, the vision tower bound with it.
    pub(crate) fn text(setup: TextSetup<'_>) -> Result<(Self, Option<Self>)> {
        let TextSetup {
            loaded,
            component,
            path,
            device,
            stream,
            scratch,
            attention,
            config,
            vision,
            dtype,
        } = setup;
        let text_info = EncoderInfo::describe(loaded, path, dtype)?;
        let vision_info = if vision {
            Some(EncoderInfo::describe(
                loaded,
                &format!("{path}.vision"),
                dtype,
            )?)
        } else {
            None
        };
        // One vision call encodes every vision block of a request. Their
        // patch rows together are within the row budget, and a block holds
        // at least one feature, so the budget also bounds the blocks.
        let config = match &vision_info {
            Some(EncoderInfo::Vision(tower)) => EncoderConfig {
                max_images: (config.row_budget / (tower.downsample as usize).pow(2)).max(1),
                ..config
            },
            _ => config,
        };
        // The context serves both encoders, so its operators are prepared
        // for the larger of their workloads: a prompt padded to the largest
        // capacity, and a request's vision blocks.
        let size = [
            text_info.context_size(&config),
            vision_info.and_then(|info| info.context_size(&config)),
        ]
        .into_iter()
        .flatten()
        .reduce(|left, right| TextSize {
            tokens: left.tokens.max(right.tokens),
            rows: left.rows.max(right.rows),
        })
        .ok_or_else(|| Error::Invariant("a text encoder declares its size".into()))?;
        let method = |name: &str| Interpreter::import(&format!("{ENCODER}.{name}"));
        let runner = Handle::new(Interpreter::import(ENCODER)?.call_values(&[
            loaded.to_any(),
            Any::from(tvm_ffi::String::from(path)),
            Any::from(stream.raw() as i64),
            Any::from(tvm_ffi::String::from(attention.as_str())),
            scratch.to_any(),
            Any::from(integer_list([size.tokens as i64, size.rows as i64])),
        ])?)?;
        let device = Device::Cuda(device);
        let vision = vision_info
            .map(|info| {
                let tower = Handle::new(method("tower")?.call_values(&[
                    runner.to_any(),
                    Any::from(tvm_ffi::String::from("vision")),
                ])?)?;
                EncoderRunner::over(
                    component,
                    device,
                    tower,
                    "vision",
                    info,
                    &config,
                    stream.clone(),
                )
            })
            .transpose()?;
        let mut text =
            EncoderRunner::over(component, device, runner, "", text_info, &config, stream)?;
        text.context = Some(OwnedContext {
            kernels: method("kernels")?,
            close: method("close")?,
        });
        Ok((text, vision))
    }

    /// The runner of a video model's condition latent encoders. `video`
    /// and `audio` name the encoders' module paths; the other fields of
    /// the setups agree.
    pub(crate) fn conditions(video: RunnerSetup<'_>, audio: RunnerSetup<'_>) -> Result<Self> {
        Ok(Self {
            owner: video.path.to_owned(),
            info: EncoderInfo::Condition,
            device: Device::Cuda(video.device),
            stream: Some(video.stream.clone()),
            methods: EncoderMethods::new(Handle::none())?,
            packed: None,
            pinned: PinnedPool::new(Some(video.device)),
            resident: None,
            noise: None,
            text: None,
            text_capacities: Vec::new(),
            config: EncoderConfig::default(),
            graphs: None,
            capturing: false,
            context: None,
            conditions: Some(Box::new(Conditions::new(video, audio)?)),
        })
    }

    /// Capture the condition encoders' still-frame tile encoding into
    /// `pool`, each step a job `perform` performs on the encoders' lane
    /// ([`Conditions::capture_tile`]).
    pub(crate) fn capture_tile(
        &mut self,
        pool: &GraphPool,
        perform: &mut dyn FnMut(Job) -> Result<Completion>,
    ) -> Result<()> {
        let (Some(stream), Device::Cuda(device)) = (self.stream.clone(), self.device) else {
            return Ok(());
        };
        if pool.device != device {
            return Err(Error::Invariant(
                "an encoder captures into a pool of its device".into(),
            ));
        }
        self.condition_encoders()?
            .capture_tile(pool, &stream, device, perform)
    }

    fn condition_encoders(&mut self) -> Result<&mut Conditions> {
        self.conditions
            .as_deref_mut()
            .ok_or_else(|| Error::Invalid(format!("{} encodes no conditions", self.owner)))
    }

    /// Record into `job` what the condition encodings of `inputs` lack
    /// before they can be recorded; see [`Conditions::setup`].
    fn condition_setup(
        &mut self,
        inputs: &[ConditionInput],
        job: &Job,
    ) -> Result<Option<ConditionSetup>> {
        self.condition_encoders()?.setup(inputs, job)
    }

    /// Keep what the job of `setup` returned.
    fn accept_setup(&mut self, setup: ConditionSetup, completion: &mut Completion) -> Result<()> {
        self.condition_encoders()?.accept(setup, completion)
    }

    /// Encode a condition's visual units ([`ConditionUnit`]) into
    /// consecutive rows of `out`.
    pub(crate) fn encode_condition_units(
        &mut self,
        pixels: &StorageView,
        units: &[ConditionUnit],
        out: &StorageView,
        job: &Job,
    ) -> Result<()> {
        self.condition_encoders()?
            .encode_video(pixels, units, out, job)
    }

    /// Encode condition audio tracks of `lengths` samples into consecutive
    /// rows of `out`.
    pub(crate) fn encode_condition_tracks(
        &mut self,
        samples: &StorageView,
        lengths: &[u64],
        out: &StorageView,
        job: &Job,
    ) -> Result<()> {
        self.condition_encoders()?
            .encode_audio(samples, lengths, out, job)
    }
}

/// The patch grids of a request's vision blocks, images and videos apart,
/// each in request order as the text encoder places them.
fn vision_grids(conditions: &[VideoCondition]) -> (Vec<[u32; 3]>, Vec<[u32; 3]>) {
    let mut images = Vec::new();
    let mut videos = Vec::new();
    for condition in conditions {
        let Some(vision) = &condition.vision else {
            continue;
        };
        let grid = [vision.grid.t, vision.grid.h, vision.grid.w];
        match condition.media {
            uniserve_core::ConditionMedia::Video { .. } => videos.push(grid),
            _ => images.push(grid),
        }
    }
    (images, videos)
}

/// Encode a request's prompt into its `conditioning` rows, splicing the
/// vision features of its condition media in at their placeholders when the
/// call reads them. Every rank of the text encoder computes the features
/// and publishes its copy.
pub(crate) fn encode_text(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    if call.outputs.len() != 1 || call.inputs.len() > 1 {
        return Err(Error::Invalid(
            "text encoding reads at most one vision product and writes its conditioning".into(),
        ));
    }
    let (tokens, conditions) = {
        let request = parts.video.request(slot, call.request_key)?;
        (Arc::clone(&request.tokens), Arc::clone(&request.conditions))
    };
    if tokens.is_empty() {
        return Err(Error::Invalid(
            "text conditioning requires admitted prompt tokens".into(),
        ));
    }
    let width = row_width(&call.outputs[0])?;
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let visual = call
        .inputs
        .first()
        .map(|input| batch.read(parts.owners().store, input, call.call_id, issue))
        .transpose()?;
    let write = bind_rows(
        parts,
        batch,
        &call.outputs[0],
        vec![tokens.len() as u64, width],
        placement,
    )?;

    let text = &mut parts
        .encoders
        .text
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no text encoder".into()))?
        .runner;
    let (image_grids, video_grids) = vision_grids(&conditions);
    let vision = visual.as_ref().map(|read| PromptVision {
        image_grids: &image_grids,
        video_grids: &video_grids,
        rows: read.value(),
    });
    let mode = text.encode_text(&tokens, &write.view(), vision, job)?;
    batch.record(mode);
    publish(
        parts,
        batch,
        call,
        index,
        &write,
        &[tokens.len() as u64, width],
        issue,
    )
}

/// Encode the vision blocks of a request's condition media into one
/// `vision_features` row per vision placeholder, in request order.
pub(crate) fn encode_vision(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "vision encoding requires one input and one output".into(),
        ));
    }
    let conditions = Arc::clone(&parts.video.request(slot, call.request_key)?.conditions);
    let blocks: Vec<([u32; 3], u64)> = conditions
        .iter()
        .filter_map(|condition| condition.vision.as_ref())
        .map(|vision| {
            (
                [vision.grid.t, vision.grid.h, vision.grid.w],
                u64::from(vision.tokens),
            )
        })
        .collect();
    if blocks.is_empty() {
        return Err(Error::Invalid(
            "vision encoding requires a vision block".into(),
        ));
    }
    let width = row_width(&call.outputs[0])?;
    let tokens: u64 = blocks.iter().map(|block| block.1).sum();
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let patches = batch.read(parts.owners().store, &call.inputs[0], call.call_id, issue)?;
    let write = bind_rows(
        parts,
        batch,
        &call.outputs[0],
        vec![tokens, width],
        placement,
    )?;

    let vision = parts
        .encoders
        .vision
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank's text encoder reads no vision".into()))?;
    // Each block's features are consecutive rows of the product.
    let grids: Vec<[u32; 3]> = blocks.iter().map(|block| block.0).collect();
    vision.encode_patches(patches.value(), &grids, &write.view(), job)?;
    publish(parts, batch, call, index, &write, &[tokens, width], issue)
}

/// What condition encoding `call` encodes on this rank: a round of visual
/// units, this rank's of them, or every audio track. `None` for an audio
/// call on a rank that is not the component's first.
fn condition_input(
    parts: &Parts<'_>,
    plan: &BatchPlan,
    call: &uniserve_ipc::Call,
    slot: usize,
) -> Result<Option<ConditionInput>> {
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "condition encoding requires one input and one output".into(),
        ));
    }
    let name = product_name(parts.info, call, &call.outputs[0])?;
    let conditions = &parts.video.request(slot, call.request_key)?.conditions;
    if conditions.is_empty() {
        return Err(Error::Invalid(
            "condition encoding requires admitted conditions".into(),
        ));
    }
    let latent = parts
        .encoders
        .latent
        .as_ref()
        .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
    match name {
        "condition_video_latents" => {
            let range = decode_range(plan, call)?;
            // Every visual unit as (condition, unit, rows), in encoding order.
            let all: Vec<(usize, u32, u64)> = conditions
                .iter()
                .enumerate()
                .flat_map(|(index, condition)| {
                    condition
                        .latent_units
                        .iter()
                        .enumerate()
                        .map(move |(unit, &rows)| (index, unit as u32, u64::from(rows)))
                })
                .collect();
            let (cursor, count) = (range.cursor as usize, range.max_units as usize);
            if count == 0 || cursor + count > all.len() {
                return Err(Error::Invalid(
                    "condition encoding exceeds the admitted units".into(),
                ));
            }
            let round = &all[cursor..cursor + count];
            let run = latent.units(&range.ranks, parts.rank, 0, count as u32)?;
            if run.is_empty() {
                return Err(Error::Invalid(
                    "condition encoding assigns this rank no unit".into(),
                ));
            }
            let rows: u64 = round.iter().map(|unit| unit.2).sum();
            let first: u64 = round[..run.start as usize].iter().map(|unit| unit.2).sum();
            let last: u64 = round[..run.end as usize].iter().map(|unit| unit.2).sum();
            let mut offsets = Vec::new();
            let mut cursor_pixels = 0u64;
            for condition in conditions.iter() {
                offsets.push(cursor_pixels);
                cursor_pixels += condition.pixel_bytes() / 3;
            }
            let units: Vec<ConditionUnit> = round[run.start as usize..run.end as usize]
                .iter()
                .map(|&(condition, unit, _)| {
                    let condition_ref = &conditions[condition];
                    let (frames, canvas) = condition_ref.media.pixels().unwrap_or((
                        0,
                        Canvas {
                            width: 0,
                            height: 0,
                        },
                    ));
                    let (window, patch_rows) = unit_rows(condition_ref, unit, frames, canvas)?;
                    Ok([
                        offsets[condition] as i64,
                        i64::from(frames),
                        i64::from(canvas.width),
                        i64::from(canvas.height),
                        i64::from(window),
                        i64::from(patch_rows.start),
                        i64::from(patch_rows.end),
                    ])
                })
                .collect::<Result<_>>()?;
            Ok(Some(ConditionInput::Units {
                units,
                rows,
                region: first..last,
            }))
        }
        "condition_audio_latents" => {
            // The audio call is one unit, the first of its dealt ranks'.
            let range = decode_range(plan, call)?;
            if latent.units(&range.ranks, parts.rank, 0, 1)?.is_empty() {
                return Ok(None);
            }
            let lengths: Vec<u64> = conditions
                .iter()
                .filter_map(|condition| condition.media.audio().map(|track| track.samples))
                .collect();
            let rows: u64 = conditions
                .iter()
                .map(|condition| u64::from(condition.audio_rows))
                .sum();
            Ok(Some(ConditionInput::Tracks { lengths, rows }))
        }
        other => Err(Error::Invalid(format!(
            "latent encoding does not produce {other}"
        ))),
    }
}

/// The encoder window and the rows of 32-pixel latent patches that unit
/// `unit` of `condition` encodes, its pixels being `frames` frames of
/// `canvas`.
///
/// A still frame planned as several units is encoded in bands of its patch
/// rows, each band reading the whole frame: the head deals the patch rows in
/// order, so a band's first row follows the rows of the bands before it
/// (`uniserve_core::video::plan::ConditionPlan::latent_units`). Any other
/// unit is its encoder window, over every patch row.
fn unit_rows(
    condition: &VideoCondition,
    unit: u32,
    frames: u32,
    canvas: Canvas,
) -> Result<(u32, std::ops::Range<u32>)> {
    let patch_rows = canvas.height / CANVAS_MULTIPLE;
    let units = &condition.latent_units;
    if frames != 1 || units.len() < 2 {
        return Ok((unit, 0..patch_rows));
    }
    let patches_per_row = canvas.width / CANVAS_MULTIPLE;
    let unit = unit as usize;
    let rows_before: u32 = units[..unit].iter().sum();
    let rows = units[unit];
    if patches_per_row == 0
        || !rows_before.is_multiple_of(patches_per_row)
        || !rows.is_multiple_of(patches_per_row)
    {
        return Err(Error::Invalid(
            "an image band is not whole rows of latent patches".into(),
        ));
    }
    let first = rows_before / patches_per_row;
    Ok((0, first..first + rows / patches_per_row))
}

/// Record into `job` what the batch's condition encodings lack before
/// their forwards can be recorded: answers and encoders the library gives
/// for input shapes this rank has not encoded before. `None` when the
/// batch's forwards can be recorded; otherwise the job ends here and
/// [`accept_conditions`] takes its completion.
pub(crate) fn setup_conditions(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    calls: &[CallState<VideoCall>],
    job: &Job,
) -> Result<Option<ConditionSetup>> {
    let mut inputs = Vec::new();
    for (call, state) in plan.calls.iter().zip(calls) {
        // A skipped call encodes nothing.
        if call.kind != CallKind::Media(MediaCall::LatentEncoding) || !state.active {
            continue;
        }
        inputs.extend(condition_input(parts, plan, call, state.slot)?);
    }
    if inputs.is_empty() {
        return Ok(None);
    }
    let latent = parts
        .encoders
        .latent
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
    latent.runner.condition_setup(&inputs, job)
}

/// Keep what the job of `setup` returned for the batch's condition
/// encodings.
pub(crate) fn accept_conditions(
    parts: &mut Parts<'_>,
    setup: ConditionSetup,
    completion: &mut Completion,
) -> Result<()> {
    let latent = parts
        .encoders
        .latent
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
    latent.runner.accept_setup(setup, completion)
}

/// Encode a request's conditions into denoiser rows: a round of visual
/// units, this rank's of them, or every audio track.
pub(crate) fn encode_conditions(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    let Some(input) = condition_input(parts, plan, call, slot)? else {
        return Ok(());
    };
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let width = row_width(&call.outputs[0])?;
    let (rows, region) = match &input {
        ConditionInput::Units { rows, region, .. } => (*rows, Some(vec![region.clone(), 0..width])),
        ConditionInput::Tracks { rows, .. } => (*rows, None),
    };
    let read = batch.read(owners.store, &call.inputs[0], call.call_id, issue)?;
    let write = bind_outputs(
        owners.store,
        &[OutputShape {
            product: call.outputs[0].clone(),
            shape: vec![rows, width],
            region,
        }],
        batch.target(),
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("condition rows were not bound".into()))?;
    batch.batch.writes.push(write.clone());

    let latent = parts
        .encoders
        .latent
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
    match &input {
        ConditionInput::Units { units, .. } => {
            latent
                .runner
                .encode_condition_units(read.value(), units, &write.view(), job)?;
        }
        ConditionInput::Tracks { lengths, .. } => {
            latent
                .runner
                .encode_condition_tracks(read.value(), lengths, &write.view(), job)?;
        }
    }
    publish_bound(parts, batch, call, index, &write, false, issue)
}

// Image vision and latent encoding batches (`VisionEncoding`,
// `LatentEncoding`): each active call encodes one image into its
// `encoder_output` feature product, an inline request image the host lane
// prepared before the batch launched, or a resident image product (feedback
// of a generated image), which the encoder prepares on its stream. Features
// are written in place into the product's reserved storage and published
// with the image's canvas dimensions (`Metadata::Feature`).
