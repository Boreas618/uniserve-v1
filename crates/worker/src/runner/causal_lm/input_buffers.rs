//! The input and output tensors of one text runner's forwards.
//!
//! One allocation per lane binding holds every column. A runner method
//! receives the columns it reads and writes as arguments, and a captured
//! graph addresses the columns its capture was given, so the storage is
//! fixed for the runner's life. `R` is the row capacity, `T` the token
//! capacity and `Tb` the number of numerical block tables.
//!
//! | Column | Element | Shape | Writer |
//! | --- | --- | --- | --- |
//! | `input_ids` | i64 | `[T]` | Rust (host and relay tokens of a prefill) |
//! | `positions` | i64 | `[3, T]` | Rust (axis 0 of a prefill) |
//! | `request_pool_indices` | i64 | `[R]` | Rust: each decode row's slot, 0 for padding |
//! | `decode_force_finish` | u8 | `[R]` | Rust |
//! | `finish_ids` | i64 | `[R, FINISH_IDS]` | Rust: each decode row's finish tokens, padded with -1 |
//! | `row_columns` | i64 | `[(6 + Tb) * (R + 1) + T]` | Rust (row gather input) |
//! | `causal_values` | i32 | `[R]` | Rust (mixed causality) |
//! | `logit_rows` | i64 | `[T]` | Rust: token positions projected to logits |
//! | `draws` | f32 | `[T]` | Rust: per candidate row |
//! | `params` | f32 | `[T, 8]` | Rust: per candidate row |
//! | `tasks` | i64 | `[2, R]` | Rust: per task slot and gate |
//! | `commit` | i64 | `[8, R]` | Rust: per commit row |
//! | `logits` | model | `[R, local vocab]` | steps: candidate logits |
//! | `hidden` | model | `[T, hidden]` | prefill graphs (when captured) |
//! | `completion` | i64 | `[4, R]` | steps: valid, active, token, accepted |
//! | `continuation`, `transition` | u8 | `[R]` | steps |
//! | `tagged`, `points` | i64 | `[R]` | steps: relay value, emitted tokens |
//! | `logprobs` | i64 | `[L]` | steps: packed logprob words |
//! | `prompt_tail` | model | `[R, vocab]` | steps: last logits of scored chunks |
//! | `input_embeddings` | model | `[T, hidden]` | Rust: vision features (multimodal) |
//! | `embedding_mask` | u8 | `[T]` | Rust: tokens whose embedding is replaced (multimodal) |

use std::sync::Arc;

use uniserve_cuda::Stream;

use crate::ffi::{Backing, DType};
use crate::storage::memory::{Memory, StorageView};
use crate::{Error, Result};

/// Byte alignment of each carved column.
const ALIGNMENT: usize = 256;

/// Host sections of the row gather's column array before the page counts.
pub(crate) const ROW_SECTIONS: usize = 6;

/// Finish tokens a decode row passes to the decode forward. A row with more
/// is selected by the runner's `sample`.
pub(crate) const FINISH_IDS: usize = 8;

/// Logprob words per token of capacity: room for 20 top and 20 requested
/// tokens per scored position.
const LOGPROB_WORDS_PER_TOKEN: usize = 3 + 3 * 20 + 2 * 20;

/// The dimensions of one binding's columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dimensions {
    pub(crate) rows: usize,
    pub(crate) tokens: usize,
    pub(crate) tables: usize,
    /// Pages a row selects from one block table, at most.
    pub(crate) width: usize,
    /// Local vocabulary columns of the logits.
    pub(crate) local_vocab: usize,
    /// Global vocabulary of sampling and prompt scoring.
    pub(crate) vocab: usize,
    pub(crate) hidden: usize,
    /// Allocate precomputed embeddings replacing token embeddings (vision
    /// features of multimodal prompts).
    pub(crate) embeddings: bool,
    pub(crate) dtype: DType,
}

impl Dimensions {
    /// The row gather's host column words: `ROW_SECTIONS + tables` sections
    /// of `rows + 1` words, then one word per token.
    pub(crate) fn row_words(&self) -> usize {
        (ROW_SECTIONS + self.tables) * (self.rows + 1) + self.tokens
    }

    pub(crate) fn logprob_words(&self) -> usize {
        self.tokens * LOGPROB_WORDS_PER_TOKEN
    }

    fn columns(&self) -> Vec<(&'static str, DType, Vec<i64>)> {
        let rows = self.rows as i64;
        let tokens = self.tokens as i64;
        let mut columns = vec![
            ("input_ids", DType::I64, vec![tokens]),
            ("positions", DType::I64, vec![3, tokens]),
            ("request_pool_indices", DType::I64, vec![rows]),
            ("decode_force_finish", DType::U8, vec![rows]),
            ("finish_ids", DType::I64, vec![rows, FINISH_IDS as i64]),
            ("row_columns", DType::I64, vec![self.row_words() as i64]),
            ("causal_values", DType::I32, vec![rows]),
            ("logit_rows", DType::I64, vec![tokens]),
            ("draws", DType::F32, vec![tokens]),
            ("params", DType::F32, vec![tokens, 8]),
            ("tasks", DType::I64, vec![2, rows]),
            ("commit", DType::I64, vec![8, rows]),
            ("logits", self.dtype, vec![rows, self.local_vocab as i64]),
            ("completion", DType::I64, vec![4, rows]),
            ("continuation", DType::U8, vec![rows]),
            ("transition", DType::U8, vec![rows]),
            ("tagged", DType::I64, vec![rows]),
            ("points", DType::I64, vec![rows]),
            ("logprobs", DType::I64, vec![self.logprob_words() as i64]),
            ("prompt_tail", self.dtype, vec![rows, self.vocab as i64]),
            ("hidden", self.dtype, vec![tokens, self.hidden as i64]),
        ];
        if self.embeddings {
            columns.push((
                "input_embeddings",
                self.dtype,
                vec![tokens, self.hidden as i64],
            ));
            columns.push(("embedding_mask", DType::U8, vec![tokens]));
        }
        columns
    }

    /// Device bytes of the columns.
    pub(crate) fn bytes(&self) -> u64 {
        self.columns()
            .iter()
            .map(|(_, dtype, shape)| column_bytes(*dtype, shape).next_multiple_of(ALIGNMENT))
            .sum::<usize>() as u64
    }
}

fn column_bytes(dtype: DType, shape: &[i64]) -> usize {
    shape.iter().product::<i64>().max(1) as usize * dtype.bytes()
}

/// A binding's columns in one allocation.
pub(crate) struct InputBuffers {
    pub(crate) dimensions: Dimensions,
    columns: Vec<(&'static str, StorageView)>,
}

impl InputBuffers {
    /// Allocate the columns on `device` (host memory when `None`) and zero
    /// them on `stream`.
    pub(crate) fn allocate(
        dimensions: Dimensions,
        device: Option<i32>,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        let layout = dimensions.columns();
        let bytes = dimensions.bytes() as usize;
        let memory = match device {
            Some(device) => Memory::device(device, bytes)?,
            None => Memory::host(bytes),
        };
        let mut columns = Vec::with_capacity(layout.len());
        let mut offset = 0;
        for (name, dtype, shape) in layout {
            columns.push((
                name,
                StorageView::contiguous(
                    Arc::clone(&memory) as Arc<dyn Backing>,
                    offset,
                    dtype,
                    &shape,
                )?,
            ));
            offset += column_bytes(dtype, &shape).next_multiple_of(ALIGNMENT);
        }
        let buffers = Self {
            dimensions,
            columns,
        };
        match (device, stream) {
            (Some(_), Some(stream)) => {
                // SAFETY: the new allocation spans `bytes`. The stream
                // drains below.
                unsafe { uniserve_cuda::memset_d32(memory.address(), 0, bytes / 4, stream)? };
                stream.synchronize()?;
            }
            // Host memory starts zeroed.
            (None, None) => {}
            _ => {
                return Err(Error::Invalid(
                    "text buffers are reset on a stream exactly for device storage".into(),
                ));
            }
        }
        Ok(buffers)
    }

    pub(crate) fn column(&self, name: &str) -> Result<&StorageView> {
        self.columns
            .iter()
            .find(|(column, _)| *column == name)
            .map(|(_, view)| view)
            .ok_or_else(|| Error::Invariant(format!("text buffers have no column {name}")))
    }
}
