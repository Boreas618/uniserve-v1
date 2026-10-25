//! The image denoiser runner.
//!
//! [`DenoiserRunner`] is the Rust half of
//! `uniserve.runner.denoiser.DenoiserRunner` for an image model's
//! cache-conditioned image denoiser ([`ImageDenoiser`], in [`image`]); a
//! video model's denoiser is the executor's (`uniserve.executor.video`). The
//! runner holds its owner, device, stream, the Python half with the methods
//! it calls (`draw_noise`, `prepare_latents`, `encode_conditions`,
//! `denoise`, `kernels`, `close`), the graph capture functions and its
//! lifecycle (`seal`, `kernels`, `close`).

pub(crate) mod image;
pub(crate) mod image_model;
pub(crate) mod runners;
pub(crate) mod warmup;

use cuda::Stream;
use tvm_ffi::Any;

use crate::Result;
use crate::config::Device;
use crate::ffi::{Function, Handle, Interpreter};
use crate::parallel::joins::GraphFunctions;
use crate::runner::denoiser::image::ImageDenoiser;
use crate::runner::methods::{MethodCall, kernels_call};

/// The Python half: a `uniserve.runner.denoiser.DenoiserRunner` and the
/// methods the image denoiser calls, each with the runner first.
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
    /// image denoiser on the host.
    stream: Option<Stream>,
    methods: Methods,
    /// Captures a forward between its begin and end.
    capture: GraphFunctions,
    /// The state the image denoiser's forwards read and write.
    image: Box<ImageDenoiser>,
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

    /// Graphs of the denoiser's steps captured so far.
    pub(crate) fn captured(&self) -> usize {
        self.image.graphs.len()
    }

    /// Refuse further captures: serving replays what startup captured.
    pub(crate) fn seal(&mut self) {
        self.image.graphs.seal();
        self.image.condition_graphs.seal();
    }

    /// The call that lists the kernels the operators of the denoiser's
    /// Python runner selected.
    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        vec![kernels_call(&self.methods.kernels, &self.methods.runner)]
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
        keep(self.image.graphs.close(aborted));
        keep(self.image.condition_graphs.close(aborted));
        keep(
            self.methods
                .close
                .invoke(&[self.methods.runner.view(), (&flag).into()]),
        );
        failure.map_or(Ok(()), Err)
    }
}

const DENOISER: &str = "uniserve.runner.denoiser.DenoiserRunner";
