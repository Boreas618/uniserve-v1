//! The fixed device columns a text binding computes over.
//!
//! One allocation per lane binding holds every column; Python receives
//! DLPack views of them once, at bind time, so captured graphs address fixed
//! storage. `R` is the row capacity, `T` the token capacity, `Tb` the number
//! of numerical block tables and `W` their widest graph width.
//!
//! | Column | Element | Shape | Writer |
//! | --- | --- | --- | --- |
//! | `input_ids` | i64 | `[T]` | Rust (host and relay tokens), decode gather |
//! | `positions` | i64 | `[3, T]` | Rust (axis 0), decode gather |
//! | `request_pool_indices` | i64 | `[R]` | Rust; the decode gather zeroes padding |
//! | `decode_force_finish` | u8 | `[R]` | Rust |
//! | `row_columns` | i64 | `[(6 + Tb) * (R + 1) + T]` | Rust (row gather input) |
//! | `block_tables` | i32 | `[Tb, R, W]` | gathers |
//! | `start_pages` | i32 | `[Tb, R]` | gathers |
//! | `cache_lengths`, `query_lengths` | i32 | `[R]` | gathers |
//! | `causal_values` | i32 | `[R]` | Rust (mixed causality) |
//! | `query_offsets`, `prefix_offsets` | i32 | `[R + 1]` | gathers |
//! | `write_indices` | i64 | `[Tb, T]` | gathers |
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

use std::sync::Arc;

use uniserve_cuda::Stream;

use crate::ffi::{Backing, DType};
use crate::storage::memory::{Memory, StorageView};
use crate::{Error, Result};

/// Byte alignment of each carved column.
const ALIGNMENT: usize = 256;

/// Host sections of the row gather's column array before the page counts.
pub(crate) const ROW_SECTIONS: usize = 6;

/// Logprob words per token of capacity: room for 20 top and 20 requested
/// tokens per scored position.
const LOGPROB_WORDS_PER_TOKEN: usize = 3 + 3 * 20 + 2 * 20;

/// The dimensions of one binding's columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dimensions {
    pub(crate) rows: usize,
    pub(crate) tokens: usize,
    pub(crate) tables: usize,
    pub(crate) width: usize,
    /// Local vocabulary columns of the logits.
    pub(crate) local_vocab: usize,
    /// Global vocabulary of sampling and prompt scoring.
    pub(crate) vocab: usize,
    pub(crate) hidden: usize,
    /// Allocate the prefill graph output.
    pub(crate) hidden_output: bool,
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
        let tables = self.tables as i64;
        let mut columns = vec![
            ("input_ids", DType::I64, vec![tokens]),
            ("positions", DType::I64, vec![3, tokens]),
            ("request_pool_indices", DType::I64, vec![rows]),
            ("decode_force_finish", DType::U8, vec![rows]),
            ("row_columns", DType::I64, vec![self.row_words() as i64]),
            (
                "block_tables",
                DType::I32,
                vec![tables, rows, self.width as i64],
            ),
            ("start_pages", DType::I32, vec![tables, rows]),
            ("cache_lengths", DType::I32, vec![rows]),
            ("query_lengths", DType::I32, vec![rows]),
            ("causal_values", DType::I32, vec![rows]),
            ("query_offsets", DType::I32, vec![rows + 1]),
            ("prefix_offsets", DType::I32, vec![rows + 1]),
            ("write_indices", DType::I64, vec![tables, tokens]),
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
        ];
        if self.hidden_output {
            columns.push(("hidden", self.dtype, vec![tokens, self.hidden as i64]));
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
pub(crate) struct TextBuffers {
    pub(crate) dimensions: Dimensions,
    columns: Vec<(&'static str, StorageView)>,
}

impl TextBuffers {
    /// Allocate the columns on `device` (host memory when `None`) and reset
    /// them on `stream`: `input_ids` to 1, `write_indices` to -1 and every
    /// other column to zero.
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
        let input_ids = buffers.column("input_ids")?.clone();
        let writes = buffers.column("write_indices")?.clone();
        match (device, stream) {
            (Some(_), Some(stream)) => {
                // SAFETY: the new allocation spans `bytes`; `input_ids` and
                // `write_indices` are columns of it. The stream drains below.
                unsafe {
                    uniserve_cuda::memset_d32(memory.address(), 0, bytes / 4, stream)?;
                    fill_i64(&input_ids, 1, stream)?;
                    fill_i64(&writes, -1, stream)?;
                }
                stream.synchronize()?;
            }
            (None, None) => {
                // Host memory starts zeroed.
                for (view, value) in [(&input_ids, 1_i64), (&writes, -1)] {
                    let count = view.numel() as usize;
                    // SAFETY: a column of this new host allocation.
                    unsafe {
                        std::slice::from_raw_parts_mut(view.address() as *mut i64, count)
                            .fill(value);
                    }
                }
            }
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

    /// Every column keyed as the text binding receives it.
    pub(crate) fn views(&self) -> Vec<(String, StorageView)> {
        self.columns
            .iter()
            .map(|(name, view)| ((*name).to_owned(), view.clone()))
            .collect()
    }
}

/// Fill an int64 column with `value` as two 32-bit word patterns.
///
/// # Safety
/// `view` is contiguous device storage valid until the stream passes.
unsafe fn fill_i64(view: &StorageView, value: i64, stream: &Stream) -> Result<()> {
    let count = view.numel() as usize;
    let low = value as u64 as u32;
    let high = (value as u64 >> 32) as u32;
    // SAFETY: forwarded; a 2D fill writes every other word.
    unsafe {
        uniserve_cuda::memset_2d_d32(view.address(), 8, low, 1, count, stream)?;
        uniserve_cuda::memset_2d_d32(view.address() + 4, 8, high, 1, count, stream)?;
    }
    Ok(())
}
