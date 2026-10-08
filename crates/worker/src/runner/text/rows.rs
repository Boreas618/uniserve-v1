//! Token rows of a text invocation and the host columns describing them.

use crate::runner::TokenSelection;
use crate::{Error, Result};

/// Where a row's input tokens and positions come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RowSource {
    /// Host tokens at consecutive positions from `position`.
    Host { tokens: Vec<u32>, position: u64 },
    /// The slot's committed next token (on the device) followed by host
    /// draft tokens, at consecutive host positions from `position`: a
    /// continuation whose predecessor's token was never observed.
    Relay { drafts: Vec<u32>, position: u64 },
    /// One token and its position read from the slot's continuation state
    /// on the device: a decode row of a call that also has other rows.
    Resident,
    /// One token, its position and the slot's verified KV length gathered on
    /// the device by the decode gather: a decode row of an all-indexed call.
    Indexed,
    /// Host tokens with explicit rotary positions (temporal, row, column):
    /// a vision block, or text of a prompt holding one, whose positions do
    /// not follow from the token count. Without `spatial`, only the
    /// temporal axis is meaningful and the model reads one axis.
    Placed {
        tokens: Vec<u32>,
        positions: [Vec<i64>; 3],
        spatial: bool,
    },
}

/// One row of a text invocation: one call's query over its slot's KV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenRow {
    pub(crate) slot: u32,
    pub(crate) source: RowSource,
    /// Visible KV tokens before the query. A relayed decode's value bounds
    /// the device's verified length.
    pub(crate) prefix: u32,
    pub(crate) query: u32,
    pub(crate) selection: TokenSelection,
    /// The call is gated by its request's continuation flag.
    pub(crate) gated: bool,
    /// Grammar state forces the call's selection to finish.
    pub(crate) force_finish: bool,
    /// The query attends causally; a vision block attends to all of
    /// itself.
    pub(crate) causal: bool,
}

impl TokenRow {
    /// Logits rows this row contributes to the invocation's candidates.
    pub(crate) fn logit_rows(&self) -> usize {
        self.selection.logit_rows(self.query as usize)
    }

    /// Validate the source against the query length.
    pub(crate) fn validate(&self) -> Result<()> {
        let tokens = match &self.source {
            RowSource::Host { tokens, .. } => tokens.len(),
            RowSource::Relay { drafts, .. } => drafts.len() + 1,
            RowSource::Resident | RowSource::Indexed => 1,
            RowSource::Placed {
                tokens, positions, ..
            } => {
                if positions.iter().any(|axis| axis.len() != tokens.len()) {
                    return Err(Error::Invalid(
                        "placed row positions disagree with its tokens".into(),
                    ));
                }
                tokens.len()
            }
        };
        if tokens != self.query as usize || self.query == 0 {
            return Err(Error::Invalid(
                "text row tokens disagree with its query length".into(),
            ));
        }
        if self.slot == 0 {
            return Err(Error::Invalid("text rows require a request slot".into()));
        }
        Ok(())
    }
}

/// Candidate logits rows per invocation row: `(first, count)` in row order.
pub(crate) fn logit_layout(rows: &[TokenRow]) -> Vec<(usize, usize)> {
    let mut first = 0;
    rows.iter()
        .map(|row| {
            let count = row.logit_rows();
            let span = (first, count);
            first += count;
            span
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(query: u32, selection: TokenSelection) -> TokenRow {
        TokenRow {
            slot: 1,
            source: RowSource::Host {
                tokens: vec![5; query as usize],
                position: 0,
            },
            prefix: 0,
            query,
            selection,
            gated: false,
            force_finish: false,
            causal: true,
        }
    }

    #[test]
    fn logit_layout_follows_each_row_selection() -> Result<()> {
        let rows = [
            row(3, TokenSelection::LastLogits),
            row(2, TokenSelection::Cache),
            row(4, TokenSelection::AllLogits),
            row(1, TokenSelection::LastLogits),
        ];
        for value in &rows {
            value.validate()?;
        }
        assert_eq!(logit_layout(&rows), vec![(0, 1), (1, 0), (1, 4), (5, 1)]);

        let mut relay = row(3, TokenSelection::AllLogits);
        relay.source = RowSource::Relay {
            drafts: vec![9, 9],
            position: 4,
        };
        relay.validate()?;
        relay.query = 2;
        assert!(relay.validate().is_err());
        Ok(())
    }
}
