//! The vocabulary embedding: a masked gather of the vocabulary rows one rank holds.
//!
//! The vocabulary is sharded across tensor-parallel ranks by [`vocabulary_shard`]. A token
//! outside a rank's shard embeds to zeros on that rank, every rank contributes its own rows, and
//! the sum over the ranks is the token's embedding
//! ([`Embedding`](crate::modules::Embedding) reduces it). The operator is therefore a gather with
//! a mask, and two implementations of it agree bit for bit.

use ops::embedding::{Args, Descriptor, Embedding};
use uniserve_tensor::Tensor;

use super::{PDL, columns, on_cpu};
use crate::collective::Split;
use crate::error::{Error, Result};

/// Tokens one `embedding` kernel handles.
const TOKENS: u32 = 8;

/// The rows of a `vocabulary`-row table `split`'s rank holds, as `(first id, rows)`.
///
/// Every rank holds `ceil(vocabulary / size)` contiguous rows, and the last ones the remainder,
/// so a rank past the vocabulary's end holds none. The embedding and the vocabulary head shard
/// by this one rule, so a token's row and its logit column are on the same rank.
pub fn vocabulary_shard(vocabulary: u32, split: Split) -> (u32, u32) {
    let per_rank = vocabulary.div_ceil(split.size.max(1));
    let first = split.rank.saturating_mul(per_rank).min(vocabulary);

    (first, per_rank.min(vocabulary - first))
}

/// The embedding rows of `ids` from `weight`, the `[rows, hidden]` shard of a vocabulary table
/// that opens at vocabulary id `first`: `[tokens, hidden]` in the table's element type, with
/// zeros for every token outside `first..first + rows`.
///
/// `ids` is one column of token ids, `U32`.
///
/// # Errors
///
/// When `ids` is not one column or `weight` is not a matrix, or the operator refuses its
/// arguments.
pub fn embedding(ids: &Tensor, weight: &Tensor, first: u32) -> Result<Tensor> {
    const OWNER: &str = "embedding";

    let &[tokens] = ids.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("ids are a column, not {:?}", ids.dims()),
        ));
    };
    let &[rows, _] = weight.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("a table is [rows, hidden], not {:?}", weight.dims()),
        ));
    };
    let hidden = columns(OWNER, weight)?;
    let rows = u32::try_from(rows)
        .map_err(|_| Error::shape(OWNER, "more vocabulary rows than a descriptor holds"))?;

    if on_cpu(ids) {
        // A rank past the vocabulary's end holds no row: every token is outside its shard.
        if rows == 0 {
            return Ok(Tensor::zeros(
                (tokens, hidden as usize),
                weight.dtype(),
                ids.device(),
            )?);
        }

        // A token below the shard or past its end has no row here: it reads the nearest row the
        // shard holds, and the mask writes zeros in its place.
        // The scalar operators of the tensor library evaluate in floating point, so the shard's
        // first id is subtracted as a tensor of the ids' own type.
        let opening = Tensor::new(first, ids.device())?.to_dtype(ids.dtype())?;
        let (first, last) = (f64::from(first), f64::from(first) + f64::from(rows) - 1.0);
        let local = ids.maximum(first)?.minimum(last)?.broadcast_sub(&opening)?;
        let gathered = weight.index_select(&local, 0)?;

        let outside = (ids.lt(first)? + ids.gt(last)?)?;
        let mask = outside.unsqueeze(1)?.broadcast_as(gathered.shape())?;
        return Ok(mask.where_cond(&gathered.zeros_like()?, &gathered)?);
    }

    let out = backend::empty((tokens, hidden as usize), weight.dtype())?;
    backend::launch::<Embedding>(
        // The token count is not in the descriptor: the ids' length is a launch value.
        &Descriptor {
            dtype: weight.dtype(),
            ids: ids.dtype(),
            hidden,
            first,
            rows,
            block: TOKENS,
            pdl: PDL,
        },
        Args {
            ids: backend::operand(ids)?,
            weight: backend::operand(weight)?,
            out: backend::operand(&out)?,
        },
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::vocabulary_shard;
    use crate::collective::Split;

    #[test]
    fn the_shards_tile_the_vocabulary_in_order() {
        for (vocabulary, size) in [(151_936_u32, 4_u32), (151_937, 4), (30, 2), (7, 4), (5, 1)] {
            let mut next = 0;
            for rank in 0..size {
                let (first, rows) = vocabulary_shard(vocabulary, Split { size, rank });
                assert_eq!(first, next.min(vocabulary), "{vocabulary} over {size}");
                assert!(rows <= vocabulary.div_ceil(size));
                next = first + rows;
            }
            assert_eq!(
                next, vocabulary,
                "{vocabulary} over {size} covers every row"
            );
        }

        // The last rank of an uneven vocabulary keeps the remainder.
        assert_eq!(
            vocabulary_shard(151_937, Split { size: 4, rank: 3 }),
            (113_955, 37_982)
        );
        // A rank past the end of a small vocabulary holds no row.
        assert_eq!(vocabulary_shard(5, Split { size: 4, rank: 3 }), (5, 0));
    }
}
