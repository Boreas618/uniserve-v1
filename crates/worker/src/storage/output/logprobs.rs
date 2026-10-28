//! Host decoding for the sampler's packed score columns.

use std::collections::{HashMap, HashSet};

use ipc::TokenLogprob;

use crate::{Error, Result};

/// Row selection of one compact logprob column: field by field over its
/// rows, each row's selected token, its score and its rank, then each
/// row's own top tokens, scores and ranks, `counts[i]` of them for row `i`
/// ([`Self::words`]). Packed row `i` reports for `rows[i]`. Scores carry
/// float32 bits in the low half of each int64 word.
pub struct LogprobLayout {
    pub rows: Vec<usize>,
    pub counts: Vec<usize>,
}

impl LogprobLayout {
    /// Words of the column: three per row, and three per top token.
    pub fn words(&self) -> Option<usize> {
        let tops = self
            .counts
            .iter()
            .try_fold(self.rows.len(), |total, &count| total.checked_add(count))?;
        tops.checked_mul(3)
    }

    pub(super) fn validate(&self, words: usize) -> Result<()> {
        if self.counts.len() != self.rows.len() || self.words() != Some(words) {
            return Err(Error::Invariant(
                "logprob completion does not match its row layout".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn decode(&self, words: &[i64]) -> Result<HashMap<usize, Vec<TokenLogprob>>> {
        let rows = self.rows.len();
        let total: usize = self.counts.iter().sum();

        // Each field contains all rows before the next field begins; a
        // row's top tokens follow those of the rows before it.
        let top_ids = rows * 3;
        let top_values = top_ids + total;
        let top_ranks = top_values + total;
        let mut result = HashMap::with_capacity(rows);

        let mut first = 0;
        for (local, &row) in self.rows.iter().enumerate() {
            let mut entries = Vec::with_capacity(1 + self.counts[local]);
            let mut seen = HashSet::with_capacity(entries.capacity());
            let mut append = |token, score, rank| -> Result<()> {
                let token_id = u32::try_from(token)
                    .map_err(|_| Error::State("logprob token is outside the token id range"))?;
                if seen.insert(token_id) {
                    entries.push(TokenLogprob {
                        token_id,
                        logprob: f32::from_bits(score as u32),
                        rank: u32::try_from(rank)
                            .map_err(|_| Error::State("logprob rank is outside the rank range"))?,
                    });
                }
                Ok(())
            };

            append(words[local], words[rows + local], words[rows * 2 + local])?;
            for column in first..first + self.counts[local] {
                append(
                    words[top_ids + column],
                    words[top_values + column],
                    words[top_ranks + column],
                )?;
            }
            first += self.counts[local];
            result.insert(row, entries);
        }
        Ok(result)
    }
}
