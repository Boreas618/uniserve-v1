//! The checkpoint's tokenizer and incremental decoding.
//!
//! `HuggingFaceTokenizer` loads a checkpoint's tokenizer directory through `fastokens`, with
//! the tokens its `tokenizer_config.json` declares, as `transformers` loads a checkpoint
//! (`HuggingFaceTokenizer::load`), and serves both directions of the text path: input
//! processing (`input::processor`, `input::sampling` and the families' request processing)
//! encodes prompts into token IDs, and the output stages in `output::text` and
//! `output::assembly` turn generated IDs back into streamed text through `IncrementalDecoder`,
//! whose stream steps are `fastokens`' own.

use std::sync::Arc;

#[macro_use]
mod error;
mod hf;
mod incremental;

pub use error::{Result, TokenizerError};
pub use hf::HuggingFaceTokenizer;
pub use incremental::IncrementalDecoder;

/// The configured tokenizer instance shared by one resolved model.
pub type DynTokenizer = Arc<HuggingFaceTokenizer>;
