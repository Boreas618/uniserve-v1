//! [`Facts`]: what the rank sizes a diffusion LM runner's buffers and graphs
//! from before the runner exists.
//!
//! Each fact is a function of `uniserve.runner.diffusion_lm` over the
//! loaded model and the diffusion LM's module path, returning a scalar or a
//! flat list: the model's own read-only method of the same name. The module
//! itself does not cross: a callable object would cross as a function.

use tvm_ffi::Any;

use crate::ffi::{DType, Handle, Interpreter};
use crate::{Error, Result};

const MODULE: &str = "uniserve.runner.diffusion_lm";

/// What a diffusion LM's rank computes.
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

fn text(value: &str) -> Any {
    Any::from(tvm_ffi::String::from(value))
}

/// The error of a library function that returned another type than its
/// docstring states.
fn mismatch(function: &str) -> impl Fn(tvm_ffi::Error) -> Error + '_ {
    move |error| Error::numerical(function, &error)
}

impl Facts {
    /// Read the facts of the diffusion LM at `path` of the loaded model.
    /// `exchange` says routed experts use an expert exchange, and
    /// `attention_ranks` counts the attention ranks of a disaggregated
    /// expert union (zero when every rank holds its experts).
    pub(crate) fn read(
        loaded: &Handle,
        path: &str,
        exchange: bool,
        attention_ranks: usize,
    ) -> Result<Self> {
        let call = |name: &str, more: &[Any]| -> Result<Any> {
            let mut arguments = vec![loaded.to_any(), text(path)];
            arguments.extend_from_slice(more);
            Interpreter::import(&format!("{MODULE}.{name}"))?.call_values(&arguments)
        };
        let count = |name: &str| -> Result<usize> {
            let value = i64::try_from(call(name, &[])?).map_err(mismatch(name))?;
            Ok(value as usize)
        };
        let dtype = |name: &str| -> Result<DType> {
            let value = tvm_ffi::String::try_from(call(name, &[])?).map_err(mismatch(name))?;
            DType::from_name(value.as_str()).ok_or_else(|| {
                Error::Unsupported(format!("{MODULE}.{name} names {:?}", value.as_str()))
            })
        };
        let counts = |name: &str| -> Result<Vec<usize>> {
            let values = crate::ffi::integers(call(name, &[])?).map_err(mismatch(name))?;
            Ok(values.into_iter().map(|value| value as usize).collect())
        };
        let flag = |name: &str, more: &[Any]| -> Result<bool> {
            bool::try_from(call(name, more)?).map_err(mismatch(name))
        };

        let stage = counts("stage")?;
        let exchanges = flag(
            "exchanges_readout",
            &[Any::from(exchange), Any::from(attention_ranks as i64)],
        )?;
        Ok(Self {
            sequence_length: count("sequence_length")?,
            vocab_size: count("vocab_size")?,
            hidden_size: count("hidden_size")?,
            dtype: dtype("embedding_dtype")?,
            attend_widths: counts("attend_widths")?,
            attend_dtype: dtype("compute_dtype")?,
            generates: flag("generates", &[])?,
            local_tail: !exchanges,
            pipeline_rank: stage[0],
            pipeline_size: stage[1],
        })
    }

    /// The stage that holds the head and completes readouts.
    pub(crate) fn last_stage(&self) -> bool {
        self.pipeline_rank + 1 == self.pipeline_size
    }

    /// Device bytes of the sampler workspace the Python runner allocates to
    /// step `chunk_rows` blocks at a time, on CUDA or the host.
    pub(crate) fn workspace_bytes(&self, chunk_rows: usize, cuda: bool) -> Result<u64> {
        let function = format!("{MODULE}.workspace_bytes");
        let bytes = Interpreter::import(&function)?.call_values(&[
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
