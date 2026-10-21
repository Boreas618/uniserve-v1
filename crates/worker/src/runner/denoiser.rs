//! The denoiser runner, and a video model's sizing.
//!
//! [`DenoiserRunner`] is the Rust half of
//! `uniserve.runner.denoiser.DenoiserRunner` for one denoiser of a model:
//! a video model's denoiser ([`VideoDenoiser`], in `video`) or an image model's
//! cache-conditioned image denoiser ([`ImageDenoiser`], in [`image`]). The
//! two families share the runner's owner, device, stream, the Python
//! half with the methods both call (`draw_noise`, `prepare_latents`,
//! `encode_conditions`, `denoise`, `kernels`, `close`), the graph capture
//! functions and the runner's lifecycle (`seal`, `kernels`, `close`); they
//! differ in the state their forwards read and write. [`VideoModel`] asks
//! the sizing methods of the Python runners what admission, capacity
//! planning and the runners need.
//!
//! A video request's samples live in the latent pool: two banks of pages
//! whose slot owns `pages` consecutive pages after page 0. A solver step
//! gathers the committed bank's pages into the runner's sample scratch,
//! copies the request's state (its slot rows of every state table and, on
//! the pipeline stage holding the text conditioner, of the retained
//! conditioning) into the fixed state buffers, writes the
//! evaluation index, launches the layout's captured graph (or runs
//! `DenoiserRunner.denoise` of `uniserve.runner.denoiser` on the
//! builder thread) and scatters the scratch into the other bank. Every one of
//! those buffers is allocated here; the Python half receives the tensors of
//! a forward as its arguments.
//!
//! Layouts ([`VideoLayout`]) fix the shapes a step evaluates. Startup
//! layouts are prepared, warmed and captured before serving and stay
//! resident; a conditioned request whose layout was not prepared at startup
//! prepares a transient layout, evaluated eagerly, which the least recently
//! used transient layout makes room for once no request uses it.

pub(crate) mod image;
pub(crate) mod image_model;
pub(crate) mod runners;
mod samples;
mod start;
mod trajectory;
mod video;
mod video_model;
pub(crate) mod warmup;

use cuda::{DeviceId, Stream};
use tvm_ffi::Any;

use crate::config::Device;
use crate::ffi::{Function, Handle, Interpreter};
use crate::parallel::joins::GraphFunctions;
use crate::profiling::kernels::KernelRecord;
use crate::runner::denoiser::image::ImageDenoiser;
use crate::runner::methods::{MethodCall, kernels_call, list_kernels};
use crate::{Error, Result};

pub(crate) use samples::{SampleLayout, SamplePages, StateEnvelope};
pub(crate) use trajectory::{denoise, prepare};
pub(crate) use video::{Conditioning, DenoiserSetup, Evicted, VideoDenoiser};
use video_model::size_values;
pub(crate) use video_model::{LayoutFields, SampleRows, VideoModel, invalid_size, placed_denoiser};

/// The Python half: a `uniserve.runner.denoiser.DenoiserRunner` and the
/// methods both families call, each with the runner first.
struct Methods {
    runner: Handle,
    draw_noise: Function,
    prepare_latents: Function,
    encode_conditions: Function,
    denoise: Function,
    kernels: Function,
    close: Function,
}

impl Methods {
    /// The methods of the Python half `runner`.
    fn import(runner: Handle) -> Result<Self> {
        let method = |name: &str| Interpreter::import(&format!("{DENOISER}.{name}"));
        Ok(Self {
            runner,
            draw_noise: method("draw_noise")?,
            prepare_latents: method("prepare_latents")?,
            encode_conditions: method("encode_conditions")?,
            denoise: method("denoise")?,
            kernels: method("kernels")?,
            close: method("close")?,
        })
    }
}

/// The runner of one denoiser of a model.
pub(crate) struct DenoiserRunner {
    /// The component the runner serves, naming it in messages.
    owner: String,
    device: Device,
    /// The stream every forward and sample update runs on; `None` for an
    /// image denoiser on the host. A video denoiser computes on CUDA.
    stream: Option<Stream>,
    methods: Methods,
    /// Captures a forward between its begin and end.
    capture: GraphFunctions,
    family: Family,
}

/// The denoiser family a runner serves, with the state of its forwards.
enum Family {
    Video(Box<VideoDenoiser>),
    Image(Box<ImageDenoiser>),
}

impl Family {
    fn video(&self) -> Result<&VideoDenoiser> {
        match self {
            Self::Video(video) => Ok(video),
            Self::Image(_) => Err(Error::Invariant("the denoiser denoises no video".into())),
        }
    }

    fn video_mut(&mut self) -> Result<&mut VideoDenoiser> {
        match self {
            Self::Video(video) => Ok(video),
            Self::Image(_) => Err(Error::Invariant("the denoiser denoises no video".into())),
        }
    }

    fn image(&self) -> Result<&ImageDenoiser> {
        match self {
            Self::Image(image) => Ok(image),
            Self::Video(_) => Err(Error::Invariant("the denoiser denoises no images".into())),
        }
    }

    fn image_mut(&mut self) -> Result<&mut ImageDenoiser> {
        match self {
            Self::Image(image) => Ok(image),
            Self::Video(_) => Err(Error::Invariant("the denoiser denoises no images".into())),
        }
    }
}

impl DenoiserRunner {
    /// The device the runner computes on.
    pub(crate) fn device(&self) -> Device {
        self.device
    }

    /// The runner's stream; `None` for an image denoiser on the host.
    pub(crate) fn stream(&self) -> Option<&Stream> {
        self.stream.as_ref()
    }

    /// The CUDA device and stream of a video denoiser, which always
    /// computes on CUDA.
    fn cuda(&self) -> Result<(DeviceId, Stream)> {
        match (self.device, &self.stream) {
            (Device::Cuda(device), Some(stream)) => Ok((device, stream.clone())),
            _ => Err(Error::Invariant(format!(
                "{}: a video denoiser computes on a CUDA stream",
                self.owner
            ))),
        }
    }

    /// Graphs of the denoiser's steps captured so far.
    pub(crate) fn captured(&self) -> usize {
        match &self.family {
            Family::Video(video) => video.graphs.len(),
            Family::Image(image) => image.graphs.len(),
        }
    }

    /// Refuse further captures: serving replays what startup captured.
    pub(crate) fn seal(&mut self) {
        match &mut self.family {
            Family::Video(video) => {
                video.graphs.seal();
                if let Some(conditioner) = &mut video.conditioner {
                    conditioner.graphs.seal();
                }
            }
            Family::Image(image) => image.graphs.seal(),
        }
    }

    /// The calls that list the kernels the operators of the denoiser's
    /// Python runners selected: the denoiser's, then a video denoiser's
    /// conditioner's at each capacity.
    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        let mut calls = vec![kernels_call(&self.methods.kernels, &self.methods.runner)];
        if let Family::Video(video) = &self.family
            && let Some(conditioner) = &video.conditioner
        {
            for runner in conditioner.runners.values() {
                calls.push(kernels_call(&conditioner.kernels, runner));
            }
        }
        calls
    }

    /// The kernels the operators of the denoiser's Python runners selected.
    pub(crate) fn kernels(&self) -> Result<Vec<KernelRecord>> {
        list_kernels(self.kernel_calls())
    }

    /// Release graphs and the Python halves; the caller drained the
    /// runner's streams. An aborted close keeps everything device work may
    /// still address until the process exits. Every release is attempted;
    /// the first failure is returned.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let mut failure = None;
        let mut keep = |result: Result<()>| {
            if let Err(error) = result {
                failure.get_or_insert(error);
            }
        };
        let flag = Any::from(aborted);
        match &mut self.family {
            Family::Video(video) => {
                keep(video.graphs.close(aborted));
                if let Some(conditioner) = &mut video.conditioner {
                    keep(conditioner.graphs.close(aborted));
                    for runner in conditioner.runners.values() {
                        keep(conditioner.close.invoke(&[runner.view(), (&flag).into()]));
                    }
                }
            }
            Family::Image(image) => keep(image.graphs.close(aborted)),
        }
        keep(
            self.methods
                .close
                .invoke(&[self.methods.runner.view(), (&flag).into()]),
        );
        failure.map_or(Ok(()), Err)
    }
}

const DENOISER: &str = "uniserve.runner.denoiser.DenoiserRunner";
const DECODER: &str = "uniserve.runner.decoder.DecoderRunner";
const ENCODER: &str = "uniserve.runner.encoder.EncoderRunner";
const OUTPUTS: &str = "uniserve.runner.denoiser.Outputs";
