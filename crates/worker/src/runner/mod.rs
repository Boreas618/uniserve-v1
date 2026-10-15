//! Routing and homogeneous batches for the worker's numerical runners.

pub(crate) mod attention;
pub(crate) mod causal_lm;
pub(crate) mod component;
pub(crate) mod decoder;
pub(crate) mod denoiser;
pub(crate) mod diffusion_lm;
pub(crate) mod encoder;
pub(crate) mod input_buffers;
pub(crate) mod media;
pub(crate) mod methods;
pub(crate) mod scratch;
pub(crate) mod start;
pub(crate) mod vision;

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_cuda::Stream;
use uniserve_ipc::{Call, CallKind};

use crate::ffi::{Backing, DType, Handle};
use crate::storage::memory::{Memory, StorageView};
use crate::{Error, Result};

/// Per-sequence language output, independently of the input's cache writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenSelection {
    LastLogits,
    AllLogits,
    Hidden,
    Cache,
}

impl TokenSelection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LastLogits => "last_logits",
            Self::AllLogits => "all_logits",
            Self::Hidden => "hidden",
            Self::Cache => "cache",
        }
    }

    pub const fn projects(self) -> bool {
        matches!(self, Self::LastLogits | Self::AllLogits)
    }

    pub fn logit_rows(self, tokens: usize) -> usize {
        match self {
            Self::LastLogits => tokens.min(1),
            Self::AllLogits => tokens,
            Self::Hidden | Self::Cache => 0,
        }
    }
}

/// One numerical invocation and the rows whose outputs it produces.
pub struct ModelBatch<'a, R> {
    pub runner: &'a R,
    pub rows: Vec<usize>,
    /// A later batch reuses this runner's graph or output buffers.
    pub preserve_output: bool,
}

/// Component operations bound to numerical runners at worker startup.
/// Runners own their device, buffers, and graphs; this table owns routing and
/// preserves first-appearance order when combining compatible input rows.
pub struct ModelRunners<R> {
    routes: HashMap<String, HashMap<CallKind, usize>>,
    runners: Vec<R>,
}

impl<R> Default for ModelRunners<R> {
    fn default() -> Self {
        Self {
            routes: HashMap::new(),
            runners: Vec::new(),
        }
    }
}

impl<R> ModelRunners<R> {
    pub fn iter(&self) -> impl Iterator<Item = &R> {
        self.runners.iter()
    }

    pub fn clear(&mut self) {
        self.routes.clear();
        self.runners.clear();
    }

    /// Bind every operation of one runner, rejecting ambiguous dispatch.
    pub fn bind(&mut self, component: &str, kinds: &[CallKind], runner: R) -> Result<()> {
        let routes = self.routes.entry(component.into()).or_default();
        if let Some(kind) = kinds.iter().find(|kind| routes.contains_key(kind)) {
            return Err(Error::Invalid(format!(
                "computation {component}.{} has multiple lane bindings",
                kind.as_str()
            )));
        }

        let index = self.runners.len();
        self.runners.push(runner);
        routes.extend(kinds.iter().map(|&kind| (kind, index)));
        Ok(())
    }

    pub fn get(&self, component: &str, kind: CallKind) -> Option<&R> {
        self.route(component, kind)
            .map(|index| &self.runners[index])
    }

    /// First runner of an operation in startup binding order.
    pub fn first(&self, kind: CallKind) -> Option<&R> {
        self.routes
            .values()
            .filter_map(|routes| routes.get(&kind))
            .min()
            .map(|&index| &self.runners[index])
    }

    /// Group rows by their runner, operation, and numerical input format.
    /// The backend supplies `K` (row type and shape); native calls determine
    /// whether context segments require per-row attention causality.
    /// Unbound rows are reported separately so their errors precede execution.
    pub fn group<'a, K: Eq + Hash>(
        &self,
        rows: impl IntoIterator<Item = (&'a Call, CallKind, K, Option<bool>)>,
    ) -> (Vec<usize>, Vec<ModelBatch<'_, R>>) {
        let mut missing = Vec::new();
        let mut positions = HashMap::new();
        let mut batches: Vec<ModelBatch<'_, R>> = Vec::new();
        let mut runners = Vec::new();

        for (index, (call, kind, format, causal)) in rows.into_iter().enumerate() {
            let Some(runner) = self.route(&call.component, kind) else {
                missing.push(index);
                continue;
            };
            // Interdependent text and vision segments write all their K/V
            // before attending. Keep them together with device causality flags.
            let causal = if !call.vision_inputs.is_empty() && call.completion_output.is_none() {
                None
            } else {
                causal
            };
            let position = *positions
                .entry((runner, kind, format, causal))
                .or_insert_with(|| {
                    let position = batches.len();
                    batches.push(ModelBatch {
                        runner: &self.runners[runner],
                        rows: Vec::new(),
                        preserve_output: false,
                    });
                    runners.push(runner);
                    position
                });
            batches[position].rows.push(index);
        }

        // Only the final invocation can lend its output directly. Earlier
        // invocations must survive reuse of the same backing by later batches.
        let mut seen = HashSet::new();
        for (batch, runner) in batches.iter_mut().zip(runners).rev() {
            batch.preserve_output = !seen.insert(runner);
        }
        (missing, batches)
    }

    fn route(&self, component: &str, kind: CallKind) -> Option<usize> {
        self.routes.get(component)?.get(&kind).copied()
    }
}

/// What the Python runners of one module are constructed with.
pub(crate) struct RunnerSetup<'a> {
    pub(crate) loaded: &'a Handle,
    /// The module's path in the model.
    pub(crate) path: &'a str,
    pub(crate) stream: Stream,
    pub(crate) device: i32,
    pub(crate) attention: String,
    /// The stream's shared work areas, a Python `Scratch`.
    pub(crate) scratch: Handle,
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
    pub fn owner(&self) -> String {
        if self.path.is_empty() {
            self.component.to_owned()
        } else {
            format!("{}.{}", self.component, self.path)
        }
    }

    /// The leading constructor arguments of every image runner: the loaded
    /// model, the module path, the stream handle (0 on the host), the
    /// device ordinal (-1 for the host) and the scratch.
    pub fn arguments(&self) -> Vec<Any> {
        vec![
            self.loaded.to_any(),
            Any::from(tvm_ffi::String::from(self.path)),
            Any::from(self.stream.as_ref().map_or(0, Stream::raw) as i64),
            Any::from(self.device.cuda_index().map_or(-1, i64::from)),
            self.scratch.to_any(),
        ]
    }
}

impl RunnerSetup<'_> {
    /// The arguments every runner of the module is constructed with before
    /// its size: the loaded model, the module path, the stream handle, the
    /// attention backend and the stream's scratch.
    pub(crate) fn arguments(&self) -> Vec<methods::Argument> {
        use methods::Argument;
        vec![
            Argument::Handle(self.loaded.clone()),
            Argument::Text(self.path.to_owned()),
            Argument::Integer(self.stream.raw() as i64),
            Argument::Text(self.attention.clone()),
            Argument::Handle(self.scratch.clone()),
        ]
    }
}

pub(crate) fn device_buffer(device: i32, dtype: DType, shape: &[i64]) -> Result<StorageView> {
    let elements: i64 = shape.iter().product();
    let memory = Memory::device(device, (elements as usize * dtype.bytes()).max(1))?;
    StorageView::contiguous(memory as Arc<dyn Backing>, 0, dtype, shape)
}
