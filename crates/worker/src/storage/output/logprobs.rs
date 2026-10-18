//! Host decoding for the sampler's packed score columns.

use std::collections::{HashMap, HashSet};

use uniserve_core::TokenLogprob;

use crate::{Error, Result};

/// Row selection and padded width of one packed logprob column: per row
/// the selected token, its score and rank, then `max_count` top tokens,
/// scores and ranks, of which a row's own count is visible. Scores carry
/// float32 bits in the low half of each int64 word.
pub struct LogprobLayout {
    pub rows: Vec<usize>,
    pub counts: Vec<usize>,
    pub max_count: usize,
}

impl LogprobLayout {
    pub(super) fn validate(&self, words: usize) -> Result<()> {
        let width = self
            .max_count
            .checked_mul(3)
            .and_then(|top| top.checked_add(3));
        if width.and_then(|width| self.rows.len().checked_mul(width)) != Some(words)
            || self.counts.len() != self.rows.len()
            || self.counts.iter().any(|&count| count > self.max_count)
        {
            return Err(Error::Invariant(
                "logprob completion does not match its row layout".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn decode(&self, words: &[i64]) -> Result<HashMap<usize, Vec<TokenLogprob>>> {
        let rows = self.rows.len();

        // Each field contains all rows before the next field begins. Top
        // fields are padded; only a row's own count of them is visible.
        let top_ids = rows * 3;
        let top_values = top_ids + rows * self.max_count;
        let top_ranks = top_values + rows * self.max_count;
        let mut result = HashMap::with_capacity(rows);

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
            for index in 0..self.counts[local] {
                let column = local * self.max_count + index;
                append(
                    words[top_ids + column],
                    words[top_values + column],
                    words[top_ranks + column],
                )?;
            }
            result.insert(row, entries);
        }
        Ok(result)
    }
}
