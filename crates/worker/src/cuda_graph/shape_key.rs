//! Shape keys of text graphs: what identifies a captured graph, the sizes
//! startup captures and the bucket serving selects for a forward.

use std::collections::BTreeSet;

use crate::{Error, Result};

/// What identifies a captured text graph besides the runner's fixed table
/// widths and tensors: the method and its scalar arguments.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ShapeKey {
    /// `decode` of a row bucket, selecting greedily or, `sampled`, each
    /// row from its top-k candidates.
    Decode { rows: usize, sampled: bool },
    /// `prefill` of a row and token bucket, with outputs or writing only
    /// the prefix cache.
    Prefill {
        rows: usize,
        tokens: usize,
        outputs: bool,
    },
    /// `commit` of a row bucket.
    Commit { rows: usize },
}

/// One prefill capture, including its extra padding row and attention mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PrefillShape {
    /// Flat tokens and rows held by the captured numerical input.
    pub(crate) token_bucket: usize,
    pub(crate) row_bucket: usize,
    /// Least rows served by this bucket: one, or the preceding row bucket.
    /// Startup uses this count to fit scratch prefix cache within the physical pool.
    pub(crate) live_rows: usize,
    /// None selects per-row device causality flags.
    pub(crate) causal: Option<bool>,
    pub(crate) embeddings: bool,
    /// False captures cache writes alone, without computing hidden outputs.
    pub(crate) outputs: bool,
}

impl PrefillShape {
    /// Fit capture rows into the scratch cache pool without exceeding a row's
    /// token limit. Tokens that do not fit become graph padding.
    ///
    /// Cache groups supply `(tokens per page, units per page)`. Block sizes are
    /// positive powers of two; the pool holds at least one page of each group.
    pub(crate) fn scratch_lengths(
        &self,
        row_tokens: usize,
        pages: &[(usize, usize)],
        units: usize,
    ) -> Vec<usize> {
        let row_units: usize = pages.iter().map(|&(_, count)| count).sum();
        let rows = self
            .live_rows
            .max(self.token_bucket.div_ceil(row_tokens))
            .min(self.row_bucket.saturating_sub(1))
            .min(units / row_units)
            .max(1);
        let mut fitted = vec![1; rows];
        let mut low = rows;
        let mut high = self.token_bucket.min(rows * row_tokens);

        // One token in every row fits. Search for the fullest numerical batch
        // whose whole pages fit all groups, leaving only graph padding unused.
        while low < high {
            let tokens = low + (high - low).div_ceil(2);
            let candidate = spread(rows, tokens, row_tokens, pages);
            if let Some(lengths) = candidate.filter(|lengths| {
                let count: usize = lengths
                    .iter()
                    .flat_map(|&length| {
                        pages
                            .iter()
                            .map(move |&(page, count)| count * length.div_ceil(page))
                    })
                    .sum();
                count <= units
            }) {
                low = tokens;
                fitted = lengths;
            } else {
                high = tokens - 1;
            }
        }
        fitted
    }
}

/// Distribute tokens using the minimum whole pages in every cache group.
/// Overfilled groups have smaller pages; their largest page divides the
/// others. Rows use whole such pages, with the final row taking the remainder.
fn spread(
    rows: usize,
    tokens: usize,
    row_tokens: usize,
    pages: &[(usize, usize)],
) -> Option<Vec<usize>> {
    let step = pages
        .iter()
        .filter_map(|&(page, _)| (tokens.div_ceil(page) > rows).then_some(page))
        .max()
        .unwrap_or(1);
    let cap = pages
        .iter()
        .filter_map(|&(page, _)| (tokens.div_ceil(page) <= rows).then_some(page))
        .min()
        .unwrap_or(row_tokens)
        .min(row_tokens);
    if cap < step {
        return None;
    }

    let steps = tokens / step;
    let base = steps / rows;
    let extra = steps % rows;
    let lengths: Vec<_> = (0..rows)
        .map(|index| {
            (base + usize::from(index < extra)) * step
                + if index + 1 == rows { tokens % step } else { 0 }
        })
        .collect();
    lengths
        .iter()
        .all(|&length| length > 0 && length <= cap)
        .then_some(lengths)
}

/// Fewest units held by independent rows over all cache groups.
/// Each group supplies `(tokens per page, pool units per page)` in `pages`.
pub(crate) fn prefill_units(pages: &[(usize, usize)], rows: usize, tokens: usize) -> usize {
    pages
        .iter()
        .map(|&(page, units)| units * rows.max(tokens.div_ceil(page)))
        .sum()
}

/// Build configured prefill captures that can serve an admitted batch.
/// A row bucket needs an extra sequence for padding. Its first token bucket
/// serves one token per live row; later buckets begin above the previous size.
pub(crate) fn prefill_shapes(
    token_sizes: &[usize],
    row_sizes: &[usize],
    max_rows: usize,
    max_tokens: usize,
    variants: &[(Option<bool>, bool)],
    outputs: bool,
    pool: Option<(&[(usize, usize)], usize)>,
) -> Vec<PrefillShape> {
    let mut tokens: BTreeSet<_> = token_sizes
        .iter()
        .copied()
        .filter(|&size| size <= max_tokens)
        .collect();
    tokens.insert(max_tokens);
    let rows: BTreeSet<_> = row_sizes.iter().copied().filter(|&size| size > 1).collect();
    let mut shapes = Vec::new();
    let mut live_rows = 1;

    for row_bucket in rows {
        if live_rows > max_rows {
            break;
        }
        let minimum = if live_rows == 1 { 1 } else { live_rows + 1 };
        let mut least = live_rows;
        for &token_bucket in tokens.range(minimum..) {
            if pool.is_some_and(|(pages, units)| prefill_units(pages, live_rows, least) > units) {
                break;
            }
            shapes.extend(variants.iter().map(|&(causal, embeddings)| PrefillShape {
                token_bucket,
                row_bucket,
                live_rows,
                causal,
                embeddings,
                outputs,
            }));
            least = token_bucket + 1;
        }
        live_rows = row_bucket;
    }
    shapes
}

/// The forward a text batch asks a graph for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TextForward<'a> {
    /// Query tokens of each row.
    pub queries: &'a [usize],
    /// Query tokens of the batch.
    pub tokens: usize,
    /// Whether the batch is a decode step.
    pub decode: bool,
    /// The attention's causality the batch's rows share; `None` when they
    /// differ.
    pub causal: Option<bool>,
    /// Whether the rows carry input embeddings.
    pub embeddings: bool,
    /// Whether only each row's last logits are computed.
    pub last_logits: bool,
    /// Whether the forward only writes the cache.
    pub cache_only: bool,
}

/// Immutable capture sizes of one text runner. The numerical backend owns
/// graph objects and fixed buffers; this owner chooses their serving shape.
pub(crate) struct TextShapes {
    decode: Vec<usize>,
    prefill: Vec<PrefillShape>,
    cache_only: bool,
    outputs: bool,
}

impl TextShapes {
    pub(crate) fn new(decode: Vec<usize>, prefill: Vec<PrefillShape>) -> Self {
        Self {
            cache_only: prefill.iter().any(|shape| !shape.outputs),
            outputs: prefill.iter().any(|shape| shape.outputs),
            decode,
            prefill,
        }
    }

    pub(crate) fn decode(&self) -> &[usize] {
        &self.decode
    }

    pub(crate) fn prefill(&self) -> &[PrefillShape] {
        &self.prefill
    }

    /// Return `(rows, tokens, decode, outputs)` of the smallest captured
    /// bucket that holds the forward, or None when no bucket holds it: the
    /// forward then runs eagerly, as every forward does without graphs. The
    /// capture catalog is therefore no bound on the batches the engine
    /// forms. Decode never borrows a prefill bucket. A cache-only prefill
    /// uses a hidden-state graph only when no cache-only graph family was
    /// captured.
    pub(crate) fn select(
        &self,
        forward: &TextForward<'_>,
    ) -> Result<Option<(usize, usize, bool, bool)>> {
        let TextForward {
            queries,
            tokens,
            decode,
            causal,
            embeddings,
            last_logits,
            cache_only,
        } = *forward;
        let rows = queries.len();
        if rows == 0 {
            return Err(Error::Invalid(
                "text graphs require a nonempty query batch".into(),
            ));
        }
        if decode && queries.iter().all(|&length| length == 1) {
            if causal != Some(true) || !last_logits {
                return Ok(None);
            }
            return Ok(self
                .decode
                .iter()
                .find(|&&size| size >= rows)
                .map(|&size| (size, size, true, true)));
        }

        let outputs = !(cache_only && self.cache_only);
        if (outputs && !self.outputs) || queries.contains(&0) {
            return Ok(None);
        }
        Ok(self
            .prefill
            .iter()
            .filter(|shape| {
                shape.outputs == outputs
                    && shape.causal == causal
                    && shape.embeddings == embeddings
                    && shape.row_bucket > rows
                    && shape.token_bucket >= tokens
            })
            .min_by_key(|shape| (shape.row_bucket, shape.token_bucket))
            .map(|shape| (shape.row_bucket, shape.token_bucket, false, outputs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_capture_fits_hybrid_pages_and_per_request_limits() {
        let shape = PrefillShape {
            token_bucket: 257,
            row_bucket: 9,
            live_rows: 4,
            causal: Some(true),
            embeddings: false,
            outputs: true,
        };
        let lengths = shape.scratch_lengths(128, &[(16, 2), (64, 1)], 24);
        assert_eq!(lengths.len(), 4);
        // Twenty-four units hold ten small pages and four large pages. The
        // small pages therefore bound the useful capture to 160 tokens.
        assert_eq!(lengths.iter().sum::<usize>(), 160);
        assert!(lengths.iter().all(|&length| length <= 128));
        let used: usize = lengths
            .iter()
            .map(|length| 2 * length.div_ceil(16) + length.div_ceil(64))
            .sum();
        assert_eq!(used, 24);

        let limited = PrefillShape {
            token_bucket: 64,
            row_bucket: 5,
            live_rows: 1,
            ..shape
        };
        let lengths = limited.scratch_lengths(8, &[(16, 1)], 8);
        assert_eq!(lengths.len(), 4);
        assert_eq!(lengths.iter().sum::<usize>(), 32);
        assert!(lengths.iter().all(|&length| length <= 8));
    }

    /// A causal prefill forward without embeddings, outputs at every position.
    const PREFILL: TextForward<'static> = TextForward {
        queries: &[],
        tokens: 0,
        decode: false,
        causal: Some(true),
        embeddings: false,
        last_logits: false,
        cache_only: false,
    };

    /// A causal decode step computing the last logits.
    const DECODE: TextForward<'static> = TextForward {
        decode: true,
        last_logits: true,
        ..PREFILL
    };

    fn prefill(rows: usize, tokens: usize, causal: Option<bool>, outputs: bool) -> PrefillShape {
        PrefillShape {
            row_bucket: rows,
            token_bucket: tokens,
            live_rows: 1,
            causal,
            embeddings: false,
            outputs,
        }
    }

    /// A decode takes the first configured size that holds it and never a
    /// prefill bucket; a decode no size holds, or one no decode graph
    /// computes (non-causal rows, outputs other than the last logits), runs
    /// eagerly.
    #[test]
    fn decode_keeps_configured_order_and_never_uses_prefill_capacity() -> Result<()> {
        let shapes = TextShapes::new(vec![8, 4, 16], vec![prefill(32, 64, Some(true), true)]);
        assert_eq!(
            shapes.select(&TextForward {
                queries: &[1; 3],
                tokens: 3,
                ..DECODE
            })?,
            Some((8, 8, true, true))
        );
        for forward in [
            TextForward {
                queries: &[1; 17],
                tokens: 17,
                ..DECODE
            },
            TextForward {
                queries: &[1],
                tokens: 1,
                causal: Some(false),
                ..DECODE
            },
            TextForward {
                queries: &[1],
                tokens: 1,
                last_logits: false,
                ..DECODE
            },
        ] {
            assert_eq!(shapes.select(&forward)?, None);
        }
        Ok(())
    }

    #[test]
    fn prefill_selects_padding_capacity_and_matching_numerical_semantics() -> Result<()> {
        let shapes = TextShapes::new(
            vec![],
            vec![
                prefill(8, 16, Some(true), true),
                prefill(4, 64, Some(true), true),
                prefill(4, 32, None, true),
                prefill(4, 16, Some(true), false),
            ],
        );
        assert_eq!(
            shapes.select(&TextForward {
                queries: &[2, 3],
                tokens: 5,
                ..PREFILL
            })?,
            Some((4, 64, false, true))
        );
        assert_eq!(
            shapes.select(&TextForward {
                queries: &[2, 3],
                tokens: 5,
                cache_only: true,
                ..PREFILL
            })?,
            Some((4, 16, false, false))
        );
        assert_eq!(
            shapes.select(&TextForward {
                queries: &[2, 3],
                tokens: 5,
                causal: None,
                ..PREFILL
            })?,
            Some((4, 32, false, true))
        );
        assert_eq!(
            shapes.select(&TextForward {
                queries: &[1; 4],
                tokens: 4,
                last_logits: true,
                ..PREFILL
            })?,
            Some((8, 16, false, true))
        );
        // Embedding rows and rows beyond every bucket have no graph and run
        // eagerly.
        for forward in [
            TextForward {
                queries: &[2],
                tokens: 2,
                embeddings: true,
                ..PREFILL
            },
            TextForward {
                queries: &[1; 8],
                tokens: 8,
                ..PREFILL
            },
            TextForward {
                queries: &[0],
                tokens: 0,
                ..PREFILL
            },
        ] {
            assert_eq!(shapes.select(&forward)?, None);
        }
        assert!(
            shapes
                .select(&TextForward {
                    queries: &[],
                    tokens: 0,
                    ..PREFILL
                })
                .is_err()
        );

        let output_only = TextShapes::new(vec![], vec![prefill(2, 16, Some(true), true)]);
        assert_eq!(
            output_only.select(&TextForward {
                queries: &[2],
                tokens: 2,
                cache_only: true,
                ..PREFILL
            })?,
            Some((2, 16, false, true))
        );
        let cache_only = TextShapes::new(vec![], vec![prefill(2, 16, Some(true), false)]);
        assert_eq!(
            cache_only.select(&TextForward {
                queries: &[2],
                tokens: 2,
                last_logits: true,
                ..PREFILL
            })?,
            None
        );
        Ok(())
    }

    #[test]
    fn captures_exclude_buckets_no_admitted_batch_can_reach() {
        let shapes = prefill_shapes(
            &[16, 32],
            &[1, 2, 4, 8],
            3,
            64,
            &[(Some(true), false)],
            true,
            Some((&[(4, 1)], 4)),
        );
        assert_eq!(
            shapes
                .iter()
                .map(|shape| (shape.row_bucket, shape.token_bucket, shape.live_rows))
                .collect::<Vec<_>>(),
            vec![(2, 16, 1), (4, 16, 2)]
        );
    }
}
