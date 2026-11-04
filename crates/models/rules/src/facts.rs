//! Why a model family's facts cannot be resolved.

/// A family rule's refusal of what a checkpoint or a request states: the
/// checkpoint's metadata lacks a field the family needs, or a value violates
/// one of the family's invariants.
#[derive(Debug, thiserror::Error)]
pub enum FactsError {
    /// Required model metadata is absent.
    #[error("model metadata is missing required field `{field}`")]
    MissingField {
        /// Required metadata field name.
        field: &'static str,
    },
    /// Model content violates a family invariant.
    #[error("invalid model asset: {0}")]
    Invalid(String),
}

impl FactsError {
    /// A violated invariant, described by `message`.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

/// Result of a family rule.
pub type Result<T> = std::result::Result<T, FactsError>;
