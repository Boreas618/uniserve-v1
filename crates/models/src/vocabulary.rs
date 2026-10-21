//! What a model family's rules read of the model's tokenizer.
//!
//! Family rules resolve control tokens by their text and encode the prompt
//! text they frame; the daemon supplies its tokenizer through
//! [`Vocabulary`], so the rules stay pure functions over its answers.

/// A model's tokenizer as family rules read it.
pub trait Vocabulary {
    /// The id of the token whose text is exactly `token`, if the vocabulary
    /// has one.
    fn token_id(&self, token: &str) -> Option<u32>;

    /// Encodes `text` into token ids without adding special tokens.
    ///
    /// # Errors
    ///
    /// Returns the tokenizer's own failure to encode `text`.
    fn encode(&self, text: &str) -> Result<Vec<u32>, VocabularyError>;
}

/// A tokenizer's failure to encode text, carried as the tokenizer's own
/// error.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct VocabularyError(pub Box<dyn std::error::Error + Send + Sync>);
