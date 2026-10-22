//! The input and output tensors of one text runner's forwards.
//!
//! One allocation per lane binding holds every column. A runner method
//! receives the columns it reads and writes as arguments, and a captured
//! graph addresses the columns its capture was given, so the storage is
//! fixed for the runner's life. `R` is the row capacity, `T` the token
//! capacity, `Tb` the number of numerical block tables, `B` the words of a
//! token set over the vocabulary, `L` the logprob words a forward may
//! report and `P` the prompt rows a forward may score, padded to whole
//! calls.
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
//! | `draws` | f32 | `[T]` | Rust: per candidate row, or per row of a decode that samples its rows |
//! | `params` | f32 | `[T, 8]` | Rust: likewise |
//! | `tasks` | i64 | `[2, R]` | Rust: per task slot and gate |
//! | `select_rows` | i64 | `[T]` | Rust: each candidate row's logits row |
//! | `select_paths` | i64 | `[R]` | Rust: each task's sampler |
//! | `select_masks` | i32 | `[R, B]` | Rust: token sets masked rows select from |
//! | `select_mask_rows` | i64 | `[T]` | Rust: each candidate row's set, or -1 |
//! | `select_biased`, `select_force` | u8 | `[R]` | Rust: tasks reading their slot's logit biases, tasks forced to finish |
//! | `select_finish`, `select_transitions` | i32 | `[R, B]` | Rust: each task's finish and transition token sets |
//! | `select_drafts` | i64 | `[T]` | Rust: a verification's draft tokens, `[tasks, width - 1]` |
//! | `select_terminal` | i64 | `[R]` | Rust: each task's terminal draft prefix |
//! | `logprob_tasks`, `logprob_counts` | i64 | `[R]` | Rust: the tasks reporting logprobs and their top counts, -1 for padding |
//! | `prompt_rows` | i64 | `[8, P]` | Rust: per prompt row its hidden position, retained slot, target token, chunk words, index, rows and top count, and tail row |
//! | `prompt_block` | i32 | `[1]` | Rust: the first prompt row of a `score_prompt` call |
//! | `commit` | i64 | `[8, R]` | Rust: per commit row |
//! | `logits` | model | `[R, local vocab]` | steps: candidate logits |
//! | `hidden` | model | `[T, hidden]` | prefill graphs (when captured) |
//! | `completion` | i64 | `[4, R]` | steps: valid, active, token, accepted |
//! | `continuation`, `transition` | u8 | `[R]` | steps |
//! | `tagged`, `points` | i64 | `[R]` | steps: relay value, emitted tokens |
//! | `scored` | i64 | `[R]` | steps: the logits row each task's reported logprobs score |
//! | `logprobs` | i64 | `[L + 1]` | steps: compact logprob words, then a word for those no one reports |
//! | `prompt_tail` | model | `[R + 1, vocab]` | steps: last logits of scored chunks, then the logits of the other prompt rows |
//! | `input_embeddings` | model | `[T, hidden]` | Rust: vision features (multimodal) |
//! | `embedding_mask` | u8 | `[T]` | Rust: tokens whose embedding is replaced (multimodal) |

use crate::ffi::DType;
use crate::runner::attention::ROW_SECTIONS;
use crate::runner::input_buffers::Columns;
use crate::sampling::{bitmask_words, least_top, logprob_words};

/// Finish tokens a decode row passes to the decode forward. A row with more
/// is selected by the runner's `select`.
pub(crate) const FINISH_IDS: usize = 8;

/// Logprob words per token of capacity: room for 20 top tokens per scored
/// position.
const LOGPROB_WORDS_PER_TOKEN: usize = 3 + 3 * 20;

/// Logprob words a forward of a binding of `tokens` tokens may report:
/// planning refuses a forward whose logprobs exceed them.
pub(crate) const fn logprob_capacity(tokens: usize) -> usize {
    tokens * LOGPROB_WORDS_PER_TOKEN
}

/// The bytes of transient logits, scores and words one `score_prompt` call
/// may hold.
const PROMPT_BLOCK_BYTES: usize = 512 << 20;

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

    /// Logprob words a forward may report.
    pub(crate) fn logprob_words(&self) -> usize {
        logprob_capacity(self.tokens)
    }

    /// Words of one token set over the vocabulary.
    pub(crate) fn set_words(&self) -> usize {
        bitmask_words(self.vocab)
    }

    /// Prompt rows one `score_prompt` call scores at top bucket `top`: as
    /// many as keep its transient logits, scores and words within
    /// [`PROMPT_BLOCK_BYTES`] (24 bytes per vocabulary entry: the projected,
    /// gathered and retained logits and their FP32 scores; 36 per top
    /// token: its values, indices and words), no more than report the
    /// bucket's least top tokens within the forward's words, and no more
    /// than a forward's prompt rows; at least one.
    pub(crate) fn prompt_block(&self, top: usize) -> usize {
        let bytes = PROMPT_BLOCK_BYTES / (24 * self.vocab + 36 * top).max(1);
        let reporting = self.logprob_words() / logprob_words(1, least_top(top));
        bytes.min(reporting).min(self.tokens + self.rows).max(1)
    }

    /// Prompt rows of the binding: one per position and one retained row
    /// per chunk, at most, padded to a whole call of the widest block.
    pub(crate) fn prompt_rows(&self) -> usize {
        self.tokens + self.rows + self.prompt_block(0)
    }

    /// The word of the `logprobs` column that the scores no one reports
    /// (padding rows, top tokens past a row's count) are written to.
    pub(crate) fn dump_words(&self) -> usize {
        self.logprob_words()
    }
}

impl Columns for Dimensions {
    type Name = &'static str;

    fn columns(&self) -> Vec<(&'static str, DType, Vec<i64>)> {
        let rows = self.rows as i64;
        let tokens = self.tokens as i64;
        let words = self.set_words() as i64;
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
            ("select_rows", DType::I64, vec![tokens]),
            ("select_paths", DType::I64, vec![rows]),
            ("select_masks", DType::I32, vec![rows, words]),
            ("select_mask_rows", DType::I64, vec![tokens]),
            ("select_biased", DType::U8, vec![rows]),
            ("select_force", DType::U8, vec![rows]),
            ("select_finish", DType::I32, vec![rows, words]),
            ("select_transitions", DType::I32, vec![rows, words]),
            ("select_drafts", DType::I64, vec![tokens]),
            ("select_terminal", DType::I64, vec![rows]),
            ("logprob_tasks", DType::I64, vec![rows]),
            ("logprob_counts", DType::I64, vec![rows]),
            (
                "prompt_rows",
                DType::I64,
                vec![8, self.prompt_rows() as i64],
            ),
            ("prompt_block", DType::I32, vec![1]),
            ("commit", DType::I64, vec![8, rows]),
            ("logits", self.dtype, vec![rows, self.local_vocab as i64]),
            ("completion", DType::I64, vec![4, rows]),
            ("continuation", DType::U8, vec![rows]),
            ("transition", DType::U8, vec![rows]),
            ("tagged", DType::I64, vec![rows]),
            ("points", DType::I64, vec![rows]),
            ("scored", DType::I64, vec![rows]),
            ("logprobs", DType::I64, vec![self.dump_words() as i64 + 1]),
            ("prompt_tail", self.dtype, vec![rows + 1, self.vocab as i64]),
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
}
