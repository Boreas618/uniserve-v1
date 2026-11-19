//! The vocabulary embedding: a masked gather of the vocabulary rows one rank holds.
//!
//! The vocabulary is sharded across tensor-parallel ranks by [`vocabulary_shard`]. A token
//! outside a rank's shard embeds to zeros on that rank, every rank contributes its own rows, and
//! the sum over the ranks is the token's embedding
//! ([`Embedding`](crate::modules::Embedding) reduces it). The operator is therefore a gather with
//! a mask, and two implementations of it agree bit for bit.

use ops::embedding::{
    Args, Descriptor, Embedding, EmbeddingMxfp8, MXFP8_GROUP, Mxfp8Args, Mxfp8Descriptor,
};
use tensor::{DType, Device, Tensor};

use super::{Fused, columns, compute};
use crate::collective::Split;
use crate::error::{Error, Result};

/// The rows of a `vocabulary`-row table `split`'s rank holds, as `(first id, rows)`.
///
/// Every rank holds `ceil(vocabulary / size)` contiguous rows, and the last ones the remainder,
/// so a rank past the vocabulary's end holds none. The embedding and the vocabulary head shard
/// by this one rule, so a token's row and its logit column are on the same rank.
pub fn vocabulary_shard(vocabulary: u32, split: Split) -> (u32, u32) {
    let per_rank = vocabulary.div_ceil(split.size());
    let first = split.rank().saturating_mul(per_rank).min(vocabulary);

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
    const OWNER: &str = EmbeddingCall::OWNER;

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

    compute(EmbeddingCall {
        ids,
        weight,
        first,
        tokens,
        hidden,
        rows,
    })
}

/// A checked call of [`embedding`]: `tokens` ids into a shard of `rows` rows of `hidden`.
struct EmbeddingCall<'t> {
    ids: &'t Tensor,
    weight: &'t Tensor,
    first: u32,
    tokens: usize,
    hidden: u32,
    rows: u32,
}

impl EmbeddingCall<'_> {
    const OWNER: &'static str = "embedding";
}

impl Fused for EmbeddingCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.ids.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            ids,
            weight,
            first,
            tokens,
            hidden,
            rows,
        } = self;

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
        Ok(mask.where_cond(&gathered.zeros_like()?, &gathered)?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            ids,
            weight,
            first,
            tokens,
            hidden,
            rows,
        } = self;

        let out = ops::empty((tokens, hidden as usize), weight.dtype())?;
        ops::launch::<Embedding>(
            // The token count is not in the descriptor: the ids' length is a launch value.
            &Descriptor::new(weight.dtype(), ids.dtype(), hidden, first, rows),
            Args {
                ids: ops::operand(ids)?,
                weight: ops::operand(weight)?,
                out: ops::operand(&out)?,
            },
        )?;
        Ok(out)
    }
}

/// The rows of `ids` from an MXFP8 table's shard, dequantized: `[ids, hidden]` of `output`
/// elements, each E4M3 value of `values` times its group's UE8M0 scale of `scales`, the product
/// in FP32 rounded once, and zeros for an id outside the shard.
///
/// `ids` is one column of I64 ids of the whole table; `values` is the shard's `[rows, hidden]`
/// E4M3 rows, whose first is row `first` of the table, and `scales` its `[rows, hidden / 32]`
/// U8 exponent bytes, `e` meaning `2^(e − 127)`.
///
/// # Errors
///
/// When an operand has other extents or types, or the operator refuses its arguments.
pub fn embedding_mxfp8(
    ids: &Tensor,
    values: &Tensor,
    scales: &Tensor,
    first: u64,
    output: DType,
) -> Result<Tensor> {
    const OWNER: &str = Mxfp8Call::OWNER;

    let group = MXFP8_GROUP as usize;
    let (&[_], &[rows, hidden]) = (ids.dims(), values.dims()) else {
        return Err(Error::shape(
            OWNER,
            format!(
                "ids of {:?} and rows of {:?}: a column of ids and a [rows, hidden] table",
                ids.dims(),
                values.dims()
            ),
        ));
    };
    if ids.dtype() != DType::I64
        || values.dtype() != DType::F8E4M3
        || scales.dtype() != DType::U8
        || hidden == 0
        || !hidden.is_multiple_of(group)
        || scales.dims() != [rows, hidden / group]
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "ids {} {:?}, values {} {:?} and scales {} {:?}: I64 ids, E4M3 rows of whole \
                 groups of {group} and one U8 scale per group",
                ops::abi::name(ids.dtype()),
                ids.dims(),
                ops::abi::name(values.dtype()),
                values.dims(),
                ops::abi::name(scales.dtype()),
                scales.dims()
            ),
        ));
    }

    compute(Mxfp8Call {
        ids,
        values,
        scales,
        first,
        output,
    })
}

/// A checked call of [`embedding_mxfp8`].
struct Mxfp8Call<'t> {
    ids: &'t Tensor,
    values: &'t Tensor,
    scales: &'t Tensor,
    first: u64,
    output: DType,
}

impl Mxfp8Call<'_> {
    const OWNER: &'static str = "embedding_mxfp8";
}

/// The FP32 value of a UE8M0 exponent byte: `2^(e − 127)`, the subnormal `2^-127` for 0 and not a
/// number for 255.
fn ue8m0(byte: u8) -> f32 {
    match byte {
        0 => f32::from_bits(0x0040_0000),
        u8::MAX => f32::NAN,
        exponent => f32::from_bits(u32::from(exponent) << 23),
    }
}

impl Fused for Mxfp8Call<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.ids.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            ids,
            values,
            scales,
            first,
            output,
        } = self;
        let (rows, hidden) = values.dims2()?;
        let count = ids.elem_count();
        let groups = hidden / MXFP8_GROUP as usize;
        if rows == 0 {
            return Ok(Tensor::zeros((count, hidden), output, ids.device())?);
        }

        // An id outside the shard reads its nearest row, and the mask writes zeros in its place.
        // The ids are below 2^53, where the scalar comparisons' floating point is exact.
        let (low, high) = (first as f64, (first + rows as u64 - 1) as f64);
        let opening = Tensor::new(first as i64, ids.device())?;
        let local = ids
            .maximum(low)?
            .minimum(high)?
            .broadcast_sub(&opening)?
            .to_dtype(DType::U32)?;

        let exponents = Tensor::from_vec(
            (0..=u8::MAX).map(ue8m0).collect::<Vec<f32>>(),
            256,
            ids.device(),
        )?;
        let factors = exponents
            .index_select(
                &scales
                    .index_select(&local, 0)?
                    .to_dtype(DType::U32)?
                    .flatten_all()?,
                0,
            )?
            .reshape((count, groups, 1))?;
        let dequantized = values
            .index_select(&local, 0)?
            .to_dtype(DType::F32)?
            .reshape((count, groups, MXFP8_GROUP as usize))?
            .broadcast_mul(&factors)?
            .reshape((count, hidden))?
            .to_dtype(output)?;

        let outside = (ids.lt(low)? + ids.gt(high)?)?
            .unsqueeze(1)?
            .broadcast_as(dequantized.shape())?;
        Ok(outside.where_cond(&dequantized.zeros_like()?, &dequantized)?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            ids,
            values,
            scales,
            first,
            output,
        } = self;
        let hidden = values.dim(1)?;
        let out = ops::empty((ids.elem_count(), hidden), output)?;
        if ids.elem_count() == 0 {
            return Ok(out);
        }

        let width = u32::try_from(hidden)
            .map_err(|_| Error::shape(Self::OWNER, format!("rows of {hidden} elements")))?;
        ops::launch::<EmbeddingMxfp8>(
            &Mxfp8Descriptor::new(output, ids.dtype(), width),
            Mxfp8Args {
                ids: ops::operand(&ids.contiguous()?)?,
                values: ops::operand(values)?,
                scales: ops::operand(scales)?,
                out: ops::operand(&out)?,
                first,
            },
        )?;
        Ok(out)
    }
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
                let (first, rows) = vocabulary_shard(vocabulary, Split::new(size, rank).unwrap());
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
            vocabulary_shard(151_937, Split::new(4, 3).unwrap()),
            (113_955, 37_982)
        );
        // A rank past the end of a small vocabulary holds no row.
        assert_eq!(vocabulary_shard(5, Split::new(4, 3).unwrap()), (5, 0));
    }
}
