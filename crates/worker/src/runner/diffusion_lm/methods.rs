//! The Python half of the diffusion LM runner:
//! `uniserve_models.runner.diffusion_lm.DiffusionLMRunner`.
//!
//! [`Methods`] holds the Python runner and its methods, reached by import
//! path. A method takes the runner first, then its scalar arguments and the
//! tensors it reads and writes; the Python contract of each is its
//! docstring. [`Facts`] is what the rank sizes its buffers and graphs from
//! before a runner exists.

use tvm_ffi::Any;

use crate::cuda_graph::CUDAGraph;
use crate::ffi::{DType, Function, Handle, Record, Shape};
use crate::model::interface::Interface;
use crate::storage::memory::StorageView;
use crate::{Error, Result};

const MODULE: &str = "uniserve_models.runner.diffusion_lm";
const RUNNER: &str = "uniserve_models.runner.diffusion_lm.DiffusionLMRunner";

/// What a diffusion LM's rank computes (`describe`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Facts {
    /// Tokens of one generated block (`C`).
    pub(crate) sequence_length: usize,
    /// Real vocabulary entries (`V`), on every pipeline stage.
    pub(crate) vocab_size: usize,
    /// Hidden width of the token embeddings (`H`).
    pub(crate) hidden_size: usize,
    /// The self-conditioning embedding dtype.
    pub(crate) dtype: DType,
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) pad_token_id: u32,
    /// Per-token widths of the final stage's attend state; empty elsewhere.
    pub(crate) attend_widths: Vec<usize>,
    pub(crate) attend_dtype: DType,
    /// This rank denoises resident blocks (one pipeline stage holding the
    /// unsharded embedding table); otherwise it serves readouts only.
    pub(crate) generates: bool,
    /// `readout` exchanges no tokens across ranks, so it runs once per slot
    /// chunk and is captured per slot bucket.
    pub(crate) local_tail: bool,
    pub(crate) pipeline_rank: usize,
    pub(crate) pipeline_size: usize,
}

fn dtype(record: &Record, key: &str) -> Result<DType> {
    let name = record.string(key)?;
    DType::from_name(&name)
        .ok_or_else(|| Error::Interface(format!("{MODULE}.describe reports the {key} {name:?}")))
}

fn text(value: &str) -> Any {
    Any::from(tvm_ffi::String::from(value))
}

impl Facts {
    /// Describe the diffusion LM at `path` of the loaded model. `exchange`
    /// says routed experts use an expert exchange, and `attention_ranks`
    /// counts the attention ranks of a disaggregated expert union (zero
    /// when every rank holds its experts).
    pub(crate) fn describe(
        interface: &Interface,
        loaded: &Handle,
        path: &str,
        exchange: bool,
        attention_ranks: usize,
    ) -> Result<Self> {
        let function = format!("{MODULE}.describe");
        let record = Record::new(
            function.as_str(),
            interface.import(&function)?.call_values(&[
                loaded.to_any(),
                text(path),
                Any::from(exchange),
                Any::from(attention_ranks as i64),
            ])?,
        )?;
        let tokens = |values: Vec<i64>| values.into_iter().map(|value| value as u32).collect();
        Ok(Self {
            sequence_length: record.count("sequence_length")?,
            vocab_size: record.count("vocab_size")?,
            hidden_size: record.count("hidden_size")?,
            dtype: dtype(&record, "dtype")?,
            eos_token_ids: tokens(record.shape("eos_token_ids")?),
            pad_token_id: record.count("pad_token_id")? as u32,
            attend_widths: record
                .shape("attend_widths")?
                .into_iter()
                .map(|width| width as usize)
                .collect(),
            attend_dtype: dtype(&record, "attend_dtype")?,
            generates: record.flag("generates")?,
            local_tail: record.flag("local_tail")?,
            pipeline_rank: record.count("pipeline_rank")?,
            pipeline_size: record.count("pipeline_size")?,
        })
    }

    /// The stage that holds the head and completes readouts.
    pub(crate) fn last_stage(&self) -> bool {
        self.pipeline_rank + 1 == self.pipeline_size
    }

    /// Device bytes of the sampler workspace the Python runner allocates to
    /// step `chunk_rows` blocks at a time, on CUDA or the host.
    pub(crate) fn workspace_bytes(
        &self,
        interface: &Interface,
        chunk_rows: usize,
        cuda: bool,
    ) -> Result<u64> {
        let function = format!("{MODULE}.workspace_bytes");
        let bytes = interface.import(&function)?.call_values(&[
            Any::from(chunk_rows as i64),
            Any::from(self.sequence_length as i64),
            Any::from(self.vocab_size as i64),
            Any::from(self.hidden_size as i64),
            text(self.dtype.name()),
            Any::from(cuda),
        ])?;
        i64::try_from(bytes)
            .map(|bytes| bytes as u64)
            .map_err(|_| Error::Interface(format!("{function} returns an integer")))
    }
}

/// One argument of a method call.
#[derive(Clone)]
pub(crate) enum Argument {
    Integer(i64),
    Float(f64),
    Integers(Vec<i64>),
    Tensor(StorageView),
    Handle(Handle),
}

impl From<usize> for Argument {
    fn from(value: usize) -> Self {
        Self::Integer(value as i64)
    }
}

impl From<bool> for Argument {
    fn from(value: bool) -> Self {
        Self::Integer(i64::from(value))
    }
}

impl From<&[i64]> for Argument {
    fn from(values: &[i64]) -> Self {
        Self::Integers(values.to_vec())
    }
}

impl From<&StorageView> for Argument {
    fn from(view: &StorageView) -> Self {
        Self::Tensor(view.clone())
    }
}

/// Convert arguments for a call. Tensors are exported here, on the calling
/// thread.
fn values(arguments: &[Argument]) -> Result<Vec<Any>> {
    arguments
        .iter()
        .map(|argument| {
            Ok(match argument {
                Argument::Integer(value) => Any::from(*value),
                Argument::Float(value) => Any::from(*value),
                Argument::Integers(values) => Any::from(Shape::from(values.as_slice())),
                Argument::Tensor(view) => Any::from(view.to_tensor()?),
                Argument::Handle(handle) => handle.to_any(),
            })
        })
        .collect()
}

/// The forwards of the Python runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Method {
    Attend,
    Readout,
    Denoise,
}

impl Method {
    const fn name(self) -> &'static str {
        match self {
            Self::Attend => "attend",
            Self::Readout => "readout",
            Self::Denoise => "denoise",
        }
    }
}

/// The Python runner and its methods.
pub(crate) struct Methods {
    runner: Handle,
    attend: Function,
    readout: Function,
    denoise: Function,
    plan_attention: Function,
    plans_attention: Function,
    capture: Function,
    replay: Function,
    kernels: Function,
    /// `graphs.release(graph)`, for the runner's graph table.
    pub(crate) release: Function,
}

impl Methods {
    /// Construct the Python runner with `arguments`: the prepared
    /// component, then the constructor's parameters in order.
    pub(crate) fn construct(interface: &Interface, arguments: &[Argument]) -> Result<Self> {
        let method = |name: &str| interface.import(&format!("{RUNNER}.{name}"));
        let runner = interface.import(RUNNER)?.call_values(&values(arguments)?)?;
        Ok(Self {
            runner: Handle::new(runner)?,
            attend: method("attend")?,
            readout: method("readout")?,
            denoise: method("denoise")?,
            plan_attention: method("plan_attention")?,
            plans_attention: method("plans_attention")?,
            capture: method("capture")?,
            replay: method("replay")?,
            kernels: method("kernels")?,
            release: interface.function("graphs.release")?,
        })
    }

    fn invoke(&self, function: &Function, leading: &[Any], arguments: &[Argument]) -> Result<Any> {
        let mut all = Vec::with_capacity(1 + leading.len() + arguments.len());
        all.push(self.runner.to_any());
        all.extend_from_slice(leading);
        all.extend(values(arguments)?);
        function.call_values(&all)
    }

    /// Run `method` eagerly on the Python thread.
    pub(crate) fn call(&self, method: Method, arguments: &[Argument]) -> Result<()> {
        let function = match method {
            Method::Attend => &self.attend,
            Method::Readout => &self.readout,
            Method::Denoise => &self.denoise,
        };
        self.invoke(function, &[], arguments).map(drop)
    }

    /// Capture `method` called with `arguments` into `pool`. The caller ran
    /// the same call eagerly first.
    pub(crate) fn capture(
        &self,
        pool: &Handle,
        method: Method,
        arguments: &[Argument],
    ) -> Result<CUDAGraph> {
        let leading = [pool.to_any(), text(method.name())];
        CUDAGraph::captured(self.invoke(&self.capture, &leading, arguments)?)
    }

    /// Run a captured forward whose segments have graph breaks.
    pub(crate) fn replay(&self, graph: &CUDAGraph) -> Result<()> {
        self.replay
            .invoke(&[self.runner.view(), graph.handle().view()])
    }

    /// Build the plans a captured `attend` or `denoise` reads, for
    /// attention operators that plan each launch on the host.
    pub(crate) fn plan_attention(&self, arguments: &[Argument]) -> Result<()> {
        self.invoke(&self.plan_attention, &[], arguments).map(drop)
    }

    /// Whether an attention operator plans each launch on the host.
    pub(crate) fn plans_attention(&self) -> Result<bool> {
        self.plans_attention.call_as(&[self.runner.view()])
    }

    /// The runner's kernel records.
    pub(crate) fn kernels(&self) -> Result<Vec<Record>> {
        let value: tvm_ffi::Array<Any> = self.kernels.call_as(&[self.runner.view()])?;
        value
            .iter()
            .enumerate()
            .map(|(index, record)| Record::new(format!("{RUNNER}.kernels[{index}]"), record))
            .collect()
    }
}
