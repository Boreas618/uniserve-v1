//! [`Facts`]: what the rank sizes a diffusion LM runner's buffers and graphs
//! from before the runner exists (`uniserve_models.runner.diffusion_lm`).

use tvm_ffi::Any;

use crate::ffi::{DType, Handle, Record};
use crate::model::interface::Interface;
use crate::{Error, Result};

const MODULE: &str = "uniserve_models.runner.diffusion_lm";

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
