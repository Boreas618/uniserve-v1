//! The Rust half of the worker's numerical runners and what they share.

pub(crate) mod attention;
pub(crate) mod causal_lm;
pub(crate) mod decoder;
pub(crate) mod denoiser;
pub(crate) mod diffusion_lm;
pub(crate) mod encoder;
pub(crate) mod input_buffers;
pub(crate) mod methods;
pub(crate) mod scratch;
pub(crate) mod start;
pub(crate) mod vision;

use cuda::Stream;
use tvm_ffi::Any;

use crate::ffi::Handle;

/// Per-sequence language output, independently of the input's cache writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TokenSelection {
    LastLogits,
    AllLogits,
    Cache,
}

impl TokenSelection {
    pub(crate) fn logit_rows(self, tokens: usize) -> usize {
        match self {
            Self::LastLogits => tokens.min(1),
            Self::AllLogits => tokens,
            Self::Cache => 0,
        }
    }
}

/// What an image runner's Python half is constructed from: the module at
/// `path` of the loaded model, the stream its forwards enqueue on and that
/// stream's shared work areas. The half constructs its own execution
/// context and releases it when the runner closes.
pub struct ImageSetup<'a> {
    pub loaded: &'a Handle,
    /// The component the module belongs to.
    pub component: &'a str,
    /// The module's path in the model.
    pub path: &'a str,
    /// The device `stream` belongs to. A module placed on a generation
    /// tower computes there between the stream's steps.
    pub device: crate::config::Device,
    /// The stream every forward enqueues on; `None` on the host.
    pub stream: Option<Stream>,
    /// The stream's shared work areas, a Python `Scratch`.
    pub scratch: Handle,
}

impl ImageSetup<'_> {
    /// The runner's name: its component, with the module path below it.
    pub(crate) fn owner(&self) -> String {
        if self.path.is_empty() {
            self.component.to_owned()
        } else {
            format!("{}.{}", self.component, self.path)
        }
    }

    /// The leading constructor arguments of every image runner: the loaded
    /// model, the module path, the stream handle (0 on the host), the
    /// device ordinal (-1 for the host) and the scratch.
    pub(crate) fn arguments(&self) -> Vec<Any> {
        vec![
            self.loaded.to_any(),
            Any::from(tvm_ffi::String::from(self.path)),
            Any::from(self.stream.as_ref().map_or(0, Stream::raw) as i64),
            Any::from(self.device.cuda().map_or(-1, |device| i64::from(device.0))),
            self.scratch.to_any(),
        ]
    }
}
