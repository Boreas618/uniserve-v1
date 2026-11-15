//! Video sparse attention (VSA): the operators that choose the key tiles each query tile
//! attends, and the compressed branch that stands for the tiles it does not.
//!
//! The sequence is cut into `tile_rows`-row tiles, the last one of a segment possibly partial:
//! `sizes` holds each tile's valid rows, and a row past it is padding. A layer runs
//!
//! ```text
//! vsa_prepare_tiles_       normalize and rotate Q and K, zero each tile's padding rows of Q, K
//!                          and V, and pool each tile's per-head means in FP32
//! vsa_scores               one dot product per (query tile, key tile) and head of the means
//! vsa_select_tiles         choose each query tile's key tiles from its pooled scores
//! sparse_attention         attend the selected tiles (`super::sparse_attention`)
//! vsa_zero_padding_        zero the padding rows of the attended queries
//! vsa_compress_pooled      each (head, query tile)'s softmax over the live key tiles, weighting
//!                          their pooled values
//! vsa_add_gated_tiles_     fold a tile's gated compression into its attended rows
//! vsa_compose_gated_rows   the same fold for one row interval of several members' rows, laid
//!                          out member by member for a sequence-parallel return
//! ```
//!
//! Every table stays on the device, so a captured graph replays the selection. Sizes, counts and
//! tile indices are `U32`; a tile's segment is `I32`, because a negative entry marks a tile
//! outside every segment: one every live query attends.
//!
//! A function whose name ends in an underscore writes the tensors it takes by value, as its
//! kernel does, and returns them.

use std::ops::Range;

use ops::descriptor::Float;
use ops::vsa::{
    AddGatedTilesArgs, AddGatedTilesDescriptor, BLOCK_ROWS, ComposeGatedRowsArgs,
    ComposeGatedRowsDescriptor, CompressPooledArgs, CompressPooledDescriptor, MAX_MEMBERS,
    PrepareTilesArgs, PrepareTilesDescriptor, ScoresArgs, ScoresDescriptor, SelectTilesArgs,
    SelectTilesDescriptor, VsaAddGatedTiles, VsaComposeGatedRows, VsaCompressPooled,
    VsaPrepareTiles, VsaScores, VsaSelectTiles, VsaZeroPadding, ZeroPaddingArgs,
    ZeroPaddingDescriptor,
};
use tensor::{D, DType, Device, Tensor};

use super::on_cpu;
use crate::error::{Error, Result};

/// The rows one program of a VSA kernel handles: a tile is whole blocks of them, and on the
/// device so are each member's rows and the row interval of [`vsa_compose_gated_rows`], so that
/// a block lies within one tile. A sequence split that returns its rows through the composition
/// lays out whole blocks for every member.
pub const VSA_BLOCK_ROWS: u32 = BLOCK_ROWS;

/// The fewest tiles a selection's kernel is compiled to rank, however few a layout holds.
const SELECT_FLOOR: usize = 256;

/// The `[rows, heads, width]` extents of tile rows, with their tile count.
fn tile_rows_of(
    owner: &'static str,
    rows: &Tensor,
    tile_rows: u32,
) -> Result<(usize, usize, usize, usize)> {
    let tile = tile_rows as usize;

    match rows.dims() {
        &[count, heads, width] if tile > 0 && count.is_multiple_of(tile) => {
            Ok((count / tile, tile, heads, width))
        }
        dims => Err(Error::shape(
            owner,
            format!("{dims:?} are not [rows, heads, width] in whole {tile_rows}-row tiles"),
        )),
    }
}

/// Checks that `table`, the per-tile table `name`, is one `dtype` entry for each of `tiles`.
fn tile_table(
    owner: &'static str,
    name: &str,
    table: &Tensor,
    tiles: usize,
    dtype: DType,
) -> Result<()> {
    if table.dims() != [tiles] || table.dtype() != dtype {
        return Err(Error::shape(
            owner,
            format!(
                "the {name} are {} {:?}, and {tiles} tiles take one {} entry each",
                ops::abi::name(table.dtype()),
                table.dims(),
                ops::abi::name(dtype)
            ),
        ));
    }
    Ok(())
}

/// `x` as a 32-bit descriptor field.
fn field(owner: &'static str, x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| Error::shape(owner, format!("an extent of {x}")))
}

/// Refuses a strided view as the operand `name` a kernel writes in place.
fn written(owner: &'static str, name: &str, tensor: &Tensor) -> Result<()> {
    if !tensor.is_contiguous() {
        return Err(Error::shape(
            owner,
            format!("the {name} are written in place, and a strided view is not their rows"),
        ));
    }
    Ok(())
}

/// `x` rounded into `dtype` and widened back: one rounding point of an FP32 formula.
fn rounded(x: &Tensor, dtype: DType) -> Result<Tensor> {
    Ok(x.to_dtype(dtype)?.to_dtype(DType::F32)?)
}

/// Which rows of `tiles` tiles of `tile` rows are padding: `[tiles · tile, 1, 1]` `U8`, one
/// where a row is at or past its tile's valid size. `sizes` is a CPU tensor.
fn padding_rows(sizes: &Tensor, tiles: usize, tile: usize) -> Result<Tensor> {
    let local = Tensor::arange(0_u32, tile as u32, &Device::Cpu)?
        .unsqueeze(0)?
        .broadcast_as((tiles, tile))?;
    let valid = sizes.unsqueeze(1)?.broadcast_as((tiles, tile))?;

    Ok(local.ge(&valid)?.reshape((tiles * tile, 1, 1))?)
}

/// `rows`, a CPU tensor `[tiles · tile, heads, width]`, with its padding rows zero.
fn without_padding(rows: &Tensor, sizes: &Tensor, tiles: usize, tile: usize) -> Result<Tensor> {
    let padding = padding_rows(sizes, tiles, tile)?.broadcast_as(rows.shape())?;

    Ok(padding.where_cond(&rows.zeros_like()?, rows)?)
}

/// The stepwise normalization and rotation of the FP32 rows `x`, `[rows, heads, width]` of
/// `dtype` values: each head scaled by its inverse RMS and the gain, rounded; then the leading
/// `rotated` features turned split-half by the rounded factors, each product and each sum
/// rounded once.
fn normalized_rotated(
    x: &Tensor,
    gain: &Tensor,
    factors: Option<(&Tensor, &Tensor)>,
    rotated: usize,
    eps: f32,
    dtype: DType,
) -> Result<Tensor> {
    let inverse = (x.sqr()?.mean_keepdim(D::Minus1)? + f64::from(eps))?
        .sqrt()?
        .recip()?;
    let normalized = rounded(
        &x.broadcast_mul(&inverse)?
            .broadcast_mul(&gain.to_dtype(DType::F32)?)?,
        dtype,
    )?;

    let Some((cosine, sine)) = factors.filter(|_| rotated > 0) else {
        return Ok(normalized);
    };
    let half = rotated / 2;
    let width = x.dims()[2];

    // One factor per row and pair, shared by the row's heads.
    let factor = |table: &Tensor| -> Result<Tensor> {
        Ok(rounded(&table.to_dtype(DType::F32)?, dtype)?.unsqueeze(1)?)
    };
    let (cosine, sine) = (factor(cosine)?, factor(sine)?);
    let leading = normalized.narrow(2, 0, half)?;
    let trailing = normalized.narrow(2, half, half)?;
    let product = |features: &Tensor, factor: &Tensor| -> Result<Tensor> {
        rounded(&features.broadcast_mul(factor)?, dtype)
    };

    // Feature `i` of the leading half pairs with feature `i + half`:
    // `(a, b) → (a·cos − b·sin, b·cos + a·sin)`.
    let turned_leading = rounded(
        &(product(&leading, &cosine)? - product(&trailing, &sine)?)?,
        dtype,
    )?;
    let turned_trailing = rounded(
        &(product(&trailing, &cosine)? + product(&leading, &sine)?)?,
        dtype,
    )?;
    let rest = normalized.narrow(2, rotated, width - rotated)?;

    Ok(Tensor::cat(&[turned_leading, turned_trailing, rest], 2)?)
}

/// Normalizes and rotates the query and key rows, zeroes the padding rows of all three
/// projections, and pools each tile's per-head means, as one operator. Returns the three
/// projections, written in place, and the pooled means.
///
/// `rows` is the query, key and value projections, each `[tiles · tile_rows, heads, width]` of
/// one 16-bit element type. With `gains`, the `[width]` query and key gains, each head of Q and
/// K is scaled by its inverse RMS (variance floor `eps`) and its gain; with `factors`, the
/// `[rows, rotated / 2]` cosine and sine tables, its leading `rotated` features are then turned
/// split-half, every step rounded into the rows' type as the eager expression rounds it.
/// `sizes` is `[tiles]` `U32`, each tile's valid rows.
///
/// The pooled means are `[3, tiles, heads, width]` in FP32, the query's, the key's and the
/// value's: each tile's sum of its stored rows over its valid size, or over one for an empty
/// tile.
///
/// # Errors
///
/// When the projections differ or are not whole tiles, the rotation is odd, wider than a head
/// or without the gains, the factors are absent for a rotation, a projection is a strided view,
/// or the operator refuses its arguments.
pub fn vsa_prepare_tiles_(
    rows: [Tensor; 3],
    gains: Option<(&Tensor, &Tensor)>,
    factors: Option<(&Tensor, &Tensor)>,
    sizes: &Tensor,
    tile_rows: u32,
    rotated: u32,
    eps: f32,
) -> Result<([Tensor; 3], Tensor)> {
    const OWNER: &str = "vsa_prepare_tiles_";

    let [query, key, value] = rows;
    let (tiles, tile, heads, width) = tile_rows_of(OWNER, &query, tile_rows)?;
    for (name, other) in [("key", &key), ("value", &value)] {
        if other.dims() != query.dims() || other.dtype() != query.dtype() {
            return Err(Error::shape(
                OWNER,
                format!(
                    "the {name} rows are {} {:?} against query rows of {} {:?}",
                    ops::abi::name(other.dtype()),
                    other.dims(),
                    ops::abi::name(query.dtype()),
                    query.dims()
                ),
            ));
        }
    }
    for (name, rows) in [("query", &query), ("key", &key), ("value", &value)] {
        written(OWNER, name, rows)?;
    }
    tile_table(OWNER, "tile sizes", sizes, tiles, DType::U32)?;

    let turned = rotated as usize;
    if turned > width || !turned.is_multiple_of(2) {
        return Err(Error::shape(
            OWNER,
            format!("a rotation of {rotated} features of {width}"),
        ));
    }
    if turned > 0 && (gains.is_none() || factors.is_none()) {
        return Err(Error::shape(
            OWNER,
            "a rotation is applied with the normalization and reads both factor tables".to_owned(),
        ));
    }

    if on_cpu(&query) {
        let dtype = query.dtype();
        let mut prepared = Vec::with_capacity(3);

        for (rows, gain) in [
            (&query, gains.map(|(query, _)| query)),
            (&key, gains.map(|(_, key)| key)),
            (&value, None),
        ] {
            let rows32 = rows.to_dtype(DType::F32)?;
            let stored = match gain {
                Some(gain) => normalized_rotated(&rows32, gain, factors, turned, eps, dtype)?,
                None => rows32,
            };
            prepared.push(without_padding(&stored, sizes, tiles, tile)?);
        }

        // A tile's mean is the sum of its stored rows over its valid size.
        let divisor = sizes
            .to_dtype(DType::F32)?
            .maximum(1.0)?
            .reshape((tiles, 1, 1))?;
        let means = prepared
            .iter()
            .map(|rows| {
                Ok(rows
                    .reshape((tiles, tile, heads, width))?
                    .sum(1)?
                    .broadcast_div(&divisor)?)
            })
            .collect::<Result<Vec<Tensor>>>()?;

        for (rows, stored) in [&query, &key, &value].into_iter().zip(&prepared) {
            rows.slice_set(&stored.to_dtype(dtype)?, 0, 0)?;
        }
        return Ok(([query, key, value], Tensor::stack(&means, 0)?));
    }

    let pooled = ops::empty((3, tiles, heads, width), DType::F32)?;
    let (query_rows, key_rows, value_rows) = (
        ops::operand(&query)?,
        ops::operand(&key)?,
        ops::operand(&value)?,
    );
    let cosine = factors
        .map(|(cosine, _)| ops::operand(cosine))
        .transpose()?;
    let means = |index: usize| -> Result<ops::Operand> { Ok(ops::operand(&pooled.get(index)?)?) };

    ops::launch::<VsaPrepareTiles>(
        // The strides are compiled into the kernel, so they are stated from the layouts the
        // projections have.
        &PrepareTilesDescriptor {
            dtype: query.dtype(),
            heads: field(OWNER, heads)?,
            width: field(OWNER, width)?,
            rotated,
            tile_rows,
            block_rows: BLOCK_ROWS,
            epsilon: Float::from(eps),
            norm: gains.is_some(),
            query_row_stride: query_rows.stride(0),
            query_head_stride: query_rows.stride(1),
            key_row_stride: key_rows.stride(0),
            key_head_stride: key_rows.stride(1),
            value_row_stride: value_rows.stride(0),
            value_head_stride: value_rows.stride(1),
            factor_stride: cosine.as_ref().map_or(0, |table| table.stride(0)),
        },
        PrepareTilesArgs {
            query: query_rows,
            key: key_rows,
            value: value_rows,
            query_weight: gains.map(|(query, _)| ops::operand(query)).transpose()?,
            key_weight: gains.map(|(_, key)| ops::operand(key)).transpose()?,
            cosine,
            sine: factors.map(|(_, sine)| ops::operand(sine)).transpose()?,
            sizes: ops::operand(sizes)?,
            pooled_query: means(0)?,
            pooled_key: means(1)?,
            pooled_value: means(2)?,
        },
    )?;
    Ok(([query, key, value], pooled))
}

/// The pooled tile similarities a selection ranks and a compression weighs:
/// `scores[head, i, j] = Σ_w query[i, head, w] · key[j, head, w] / sqrt(width)`, in FP32.
///
/// `pooled_query` is `[query tiles, heads, width]` and `pooled_key` `[tiles, heads, width]`,
/// both FP32, the layout the preparation pools in; the scores are `[heads, query tiles, tiles]`.
///
/// # Errors
///
/// When the means are not FP32 tiles of one head shape, or the operator refuses its arguments.
pub fn vsa_scores(pooled_query: &Tensor, pooled_key: &Tensor) -> Result<Tensor> {
    const OWNER: &str = "vsa_scores";

    let (&[query_tiles, heads, width], &[tiles, key_heads, key_width]) =
        (pooled_query.dims(), pooled_key.dims())
    else {
        return Err(Error::shape(
            OWNER,
            format!(
                "the pooled means are [tiles, heads, width], not {:?} and {:?}",
                pooled_query.dims(),
                pooled_key.dims()
            ),
        ));
    };
    if (heads, width) != (key_heads, key_width)
        || pooled_query.dtype() != DType::F32
        || pooled_key.dtype() != DType::F32
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "pooled queries of {:?} and keys of {:?} are not FP32 means of one head shape",
                pooled_query.dims(),
                pooled_key.dims()
            ),
        ));
    }

    if on_cpu(pooled_query) {
        let queries = pooled_query.transpose(0, 1)?.contiguous()?;
        let keys = pooled_key.permute((1, 2, 0))?.contiguous()?;
        let root = Tensor::new((width as f32).sqrt(), &Device::Cpu)?;

        return Ok(queries.matmul(&keys)?.broadcast_div(&root)?);
    }

    let scores = ops::empty((heads, query_tiles, tiles), DType::F32)?;
    let width = field(OWNER, width)?;

    ops::launch::<VsaScores>(
        &ScoresDescriptor::new(
            field(OWNER, heads)?,
            width,
            width.next_power_of_two(),
            false,
        ),
        ScoresArgs {
            pooled_query: ops::operand(pooled_query)?,
            pooled_key: ops::operand(pooled_key)?,
            out: ops::operand(&scores)?,
        },
    )?;
    Ok(scores)
}

/// The ascending order key of a score: its sign-flipped bits, with −0 as +0 and every NaN above
/// +∞, as a stable sort orders FP32 values.
fn order_key(score: f32) -> u32 {
    if score.is_nan() {
        return u32::MAX;
    }

    let bits = (score + 0.0).to_bits();
    if bits >= 0x8000_0000 {
        u32::MAX - bits
    } else {
        bits | 0x8000_0000
    }
}

/// Each query tile's key tiles, chosen from its pooled scores. Returns `(indices, counts)`:
/// `indices` is `[heads, query tiles, tiles]` `U32`, every key tile with the kept ones first,
/// each part in ascending order, and `counts` is `[heads, query tiles]` `U32`, how many are
/// kept.
///
/// `scores` is `[heads, query tiles, tiles]` FP32, query tile `q` being sequence tile
/// `query_offset + q`: a rank of an all-gather group ranks its own interval of query tiles over
/// every key tile. `sizes` is `[tiles]` `U32`, each tile's valid rows; `segments` is `[tiles]`
/// `I32`, the
/// segment each tile belongs to, negative for a tile of none; `keep` is `[tiles]` `U32`, read by
/// segment: how many tiles of each segment a query keeps.
///
/// A live query tile of a segment keeps every live tile outside the segments and, of each
/// segment, its `keep` tiles of the highest scores, ties going to the lower tile. A live query
/// tile outside the segments keeps every live tile, and an empty query tile keeps none.
///
/// # Errors
///
/// When the scores are not `[heads, query tiles, tiles]` FP32 whose query tiles from
/// `query_offset` lie among the tiles, a table is not one entry per tile, or the operator refuses
/// its arguments.
pub fn vsa_select_tiles(
    scores: &Tensor,
    sizes: &Tensor,
    segments: &Tensor,
    keep: &Tensor,
    query_offset: usize,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = "vsa_select_tiles";

    let &[heads, query_tiles, tiles] = scores.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "the pooled scores are [heads, query tiles, tiles], not {:?}",
                scores.dims()
            ),
        ));
    };
    if scores.dtype() != DType::F32 || query_offset + query_tiles > tiles {
        return Err(Error::shape(
            OWNER,
            format!(
                "{} scores of {query_tiles} query tiles from tile {query_offset} over {tiles} \
                 tiles",
                ops::abi::name(scores.dtype())
            ),
        ));
    }
    tile_table(OWNER, "tile sizes", sizes, tiles, DType::U32)?;
    tile_table(OWNER, "tile segments", segments, tiles, DType::I32)?;
    tile_table(OWNER, "segment keep counts", keep, tiles, DType::U32)?;

    if on_cpu(scores) {
        let values: Vec<f32> = scores.flatten_all()?.to_vec1()?;
        let sizes: Vec<u32> = sizes.to_vec1()?;
        let segments: Vec<i32> = segments.to_vec1()?;
        let keep: Vec<u32> = keep.to_vec1()?;
        let last_segment = segments.iter().copied().max().unwrap_or(-1);
        if last_segment >= tiles as i32 {
            return Err(Error::shape(
                OWNER,
                format!("segment {last_segment} has no entry in a keep table of {tiles}"),
            ));
        }

        let mut indices = Vec::with_capacity(values.len());
        let mut counts = Vec::with_capacity(heads * query_tiles);
        for row in values.chunks(tiles) {
            let query = query_offset + counts.len() % query_tiles;
            let query_live = sizes[query] > 0;

            let mut kept: Vec<bool> = sizes.iter().map(|size| query_live && *size > 0).collect();
            if query_live && segments[query] >= 0 {
                // Every live tile outside the segments, and each segment's highest scores.
                for (tile, flag) in kept.iter_mut().enumerate() {
                    *flag = (sizes[tile] > 0 && segments[tile] == -1) || segments[tile] < -1;
                }
                for segment in 0..=last_segment {
                    let mut members: Vec<usize> = (0..tiles)
                        .filter(|tile| segments[*tile] == segment)
                        .collect();
                    members.sort_by_key(|tile| (std::cmp::Reverse(order_key(row[*tile])), *tile));

                    for tile in members.into_iter().take(keep[segment as usize] as usize) {
                        kept[tile] = true;
                    }
                }
            }

            let ascending = |wanted: bool| {
                let kept = &kept;
                (0..tiles as u32).filter(move |tile| kept[*tile as usize] == wanted)
            };
            indices.extend(ascending(true).chain(ascending(false)));
            counts.push(kept.iter().filter(|flag| **flag).count() as u32);
        }

        return Ok((
            Tensor::from_vec(indices, (heads, query_tiles, tiles), &Device::Cpu)?,
            Tensor::from_vec(counts, (heads, query_tiles), &Device::Cpu)?,
        ));
    }

    let indices = ops::empty(scores.shape(), DType::U32)?;
    let counts = ops::empty((heads, query_tiles), DType::U32)?;
    let scores = scores.contiguous()?;

    ops::launch::<VsaSelectTiles>(
        // The kernel masks the tiles past the layout's own, so one compilation ranks every
        // layout up to its power of two.
        &SelectTilesDescriptor {
            block: field(OWNER, tiles.max(SELECT_FLOOR).next_power_of_two())?,
        },
        SelectTilesArgs {
            scores: ops::operand(&scores)?,
            segments: ops::operand(segments)?,
            sizes: ops::operand(sizes)?,
            keep: ops::operand(keep)?,
            indices: ops::operand(&indices)?,
            counts: ops::operand(&counts)?,
            query_offset: query_offset as u64,
        },
    )?;
    Ok((indices, counts))
}

/// The compressed branch: each (head, query tile)'s softmax over the live key tiles, weighting
/// their pooled values.
///
/// ```text
/// weights[head, i, j] = softmax_j(scores[head, i, j])      j over the tiles of a valid row
/// out[head, i, w]     = Σ_j weights[head, i, j] · values[j, head, w]
/// ```
///
/// `scores` is `[heads, query tiles, tiles]` FP32 and `values` `[tiles, heads, width]` FP32,
/// the value means the preparation pools; `sizes` is `[tiles]` `U32`. A tile of no valid row is
/// masked out of the softmax and not zeroed: a zeroed value under a live score would still take
/// weight. The result is `[heads, query tiles, width]`, one row per (head, query tile), summed
/// in FP32 and rounded once into `output`. `max_tiles` is the deployment's bound on the key
/// tiles one reduction covers.
///
/// # Errors
///
/// When the scores and values are not FP32 of one head and tile count, the sizes are not one
/// per tile, or the operator refuses its arguments.
pub fn vsa_compress_pooled(
    scores: &Tensor,
    values: &Tensor,
    sizes: &Tensor,
    max_tiles: u32,
    output: DType,
) -> Result<Tensor> {
    const OWNER: &str = "vsa_compress_pooled";

    let (&[heads, query_tiles, tiles], &[value_tiles, value_heads, width]) =
        (scores.dims(), values.dims())
    else {
        return Err(Error::shape(
            OWNER,
            format!(
                "the scores are [heads, query tiles, tiles] and the values [tiles, heads, \
                 width], not {:?} and {:?}",
                scores.dims(),
                values.dims()
            ),
        ));
    };
    if (heads, tiles) != (value_heads, value_tiles)
        || scores.dtype() != DType::F32
        || values.dtype() != DType::F32
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "scores of {:?} and pooled values of {:?} are not FP32 over one head and tile \
                 count",
                scores.dims(),
                values.dims()
            ),
        ));
    }
    tile_table(OWNER, "tile sizes", sizes, tiles, DType::U32)?;

    if on_cpu(scores) {
        let live: Vec<u32> = sizes.to_vec1()?;
        if live.iter().all(|size| *size == 0) {
            return Ok(Tensor::zeros(
                (heads, query_tiles, width),
                output,
                &Device::Cpu,
            )?);
        }

        let masked = sizes
            .gt(0.0)?
            .reshape((1, 1, tiles))?
            .broadcast_as(scores.shape())?
            .where_cond(
                scores,
                &Tensor::full(f32::NEG_INFINITY, scores.shape(), &Device::Cpu)?,
            )?;
        let weights = masked.broadcast_sub(&masked.max_keepdim(2)?)?.exp()?;
        let by_head = values.transpose(0, 1)?.contiguous()?;

        let compressed = weights
            .matmul(&by_head)?
            .broadcast_div(&weights.sum_keepdim(2)?)?;
        return Ok(compressed.to_dtype(output)?);
    }

    let out = ops::empty((heads, query_tiles, width), output)?;
    let scores = scores.contiguous()?;
    let width = field(OWNER, width)?;

    ops::launch::<VsaCompressPooled>(
        &CompressPooledDescriptor::new(
            output,
            field(OWNER, heads)?,
            width,
            max_tiles,
            width.next_power_of_two(),
            false,
        ),
        CompressPooledArgs {
            scores: ops::operand(&scores)?,
            values: ops::operand(values)?,
            sizes: ops::operand(sizes)?,
            out: ops::operand(&out)?,
        },
    )?;
    Ok(out)
}

/// Zeroes the rows of every tile past its valid size, in place, and returns `values`.
///
/// `values` is `[tiles · tile_rows, heads, width]` and `sizes` `[tiles]` `U32`. Only the padding
/// rows are stored.
///
/// # Errors
///
/// When the rows are not whole tiles or a strided view, the sizes are not one per tile, or the
/// operator refuses its arguments.
pub fn vsa_zero_padding_(values: Tensor, sizes: &Tensor, tile_rows: u32) -> Result<Tensor> {
    const OWNER: &str = "vsa_zero_padding_";

    let (tiles, tile, heads, width) = tile_rows_of(OWNER, &values, tile_rows)?;
    tile_table(OWNER, "tile sizes", sizes, tiles, DType::U32)?;
    written(OWNER, "rows", &values)?;

    if on_cpu(&values) {
        values.slice_set(&without_padding(&values, sizes, tiles, tile)?, 0, 0)?;
        return Ok(values);
    }

    ops::launch::<VsaZeroPadding>(
        &ZeroPaddingDescriptor {
            dtype: values.dtype(),
            heads: field(OWNER, heads)?,
            tile_rows,
            width: field(OWNER, width)?,
            block_rows: BLOCK_ROWS,
        },
        ZeroPaddingArgs {
            values: ops::operand(&values)?,
            sizes: ops::operand(sizes)?,
        },
    )?;
    Ok(values)
}

/// Checks the operands a gated fold shares: `fine` and `gate` rows of one shape and element
/// type in whole tiles, and one compressed row per (head, tile) in that type.
fn check_gated(
    owner: &'static str,
    fine: &Tensor,
    compressed: &Tensor,
    gate: &Tensor,
    tile_rows: u32,
) -> Result<(usize, usize, usize, usize)> {
    let (tiles, tile, heads, width) = tile_rows_of(owner, fine, tile_rows)?;

    if gate.dims() != fine.dims()
        || compressed.dims() != [heads, tiles, width]
        || gate.dtype() != fine.dtype()
        || compressed.dtype() != fine.dtype()
    {
        return Err(Error::shape(
            owner,
            format!(
                "a gate of {:?} and a compression of {:?} for {:?} rows of {tiles} tiles",
                gate.dims(),
                compressed.dims(),
                fine.dims()
            ),
        ));
    }
    Ok((tiles, tile, heads, width))
}

/// The gated fold of CPU tensors, `fine + compressed[:, tile] · gate`, in the rows' type: the
/// product rounds, then the sum rounds, as two eager elementwise operations round them.
fn gated_sum(
    fine: &Tensor,
    compressed: &Tensor,
    gate: &Tensor,
    (tiles, tile, heads, width): (usize, usize, usize, usize),
) -> Result<Tensor> {
    let dtype = fine.dtype();

    // A tile's compressed row, broadcast over every row of the tile.
    let per_row = compressed
        .to_dtype(DType::F32)?
        .transpose(0, 1)?
        .unsqueeze(1)?
        .broadcast_as((tiles, tile, heads, width))?
        .reshape((tiles * tile, heads, width))?;
    let product = rounded(&(per_row * gate.to_dtype(DType::F32)?)?, dtype)?;

    Ok((fine.to_dtype(DType::F32)? + product)?.to_dtype(dtype)?)
}

/// Folds each tile's gated compression into its attended rows, in place:
/// `fine += compressed[:, tile] · gate`, and returns `fine`.
///
/// `fine` and `gate` are `[tiles · tile_rows, heads, width]` and `compressed`
/// `[heads, tiles, width]`, of one element type. A compressed row is broadcast over every row of
/// its tile; the product and the sum each round into the rows' type.
///
/// # Errors
///
/// When the operands are not one shape and type in whole tiles, `fine` is a strided view, or the
/// operator refuses its arguments.
pub fn vsa_add_gated_tiles_(
    fine: Tensor,
    compressed: &Tensor,
    gate: &Tensor,
    tile_rows: u32,
) -> Result<Tensor> {
    const OWNER: &str = "vsa_add_gated_tiles_";

    let extents = check_gated(OWNER, &fine, compressed, gate, tile_rows)?;
    written(OWNER, "fine rows", &fine)?;

    if on_cpu(&fine) {
        fine.slice_set(&gated_sum(&fine, compressed, gate, extents)?, 0, 0)?;
        return Ok(fine);
    }

    ops::launch::<VsaAddGatedTiles>(
        &AddGatedTilesDescriptor {
            dtype: fine.dtype(),
            tile_rows,
            width: field(OWNER, extents.3)?,
            block_rows: BLOCK_ROWS,
        },
        AddGatedTilesArgs {
            output: ops::operand(&fine)?,
            compressed: ops::operand(compressed)?,
            gate: ops::operand(gate)?,
        },
    )?;
    Ok(fine)
}

/// One row interval of every member's gated rows, member by member: the send buffer of a
/// sequence-parallel return.
///
/// `fine` and `gate` hold every member's rows of this member's heads,
/// `[members · owner rows, heads, width]`, so each of the `slots.len()` members owns
/// `rows / members` of them; `rows` is the interval of each owner's rows the result carries.
/// `slots` names the block of the result each member's rows land in, in member order. The
/// result is `[members, rows.len(), heads, width]`: block `slots[m]` holds
/// `fine + compressed[:, tile] · gate` over rows `m · owner rows + rows` (the arithmetic of
/// [`vsa_add_gated_tiles_`]).
///
/// On the device a member's owner rows and `rows` are whole [`VSA_BLOCK_ROWS`]-row blocks: the
/// kernel folds one tile's compression into each block it handles, so a block that started off
/// a block boundary could reach into the next tile.
///
/// # Errors
///
/// When there is no member or more than the operator covers, the slots are not one per member,
/// the interval is outside a member's rows, the operands are not one shape and type, or the
/// operator refuses its arguments (on the device, owner rows or an interval that are not whole
/// blocks).
pub fn vsa_compose_gated_rows(
    fine: &Tensor,
    compressed: &Tensor,
    gate: &Tensor,
    tile_rows: u32,
    rows: Range<usize>,
    slots: &[u32],
) -> Result<Tensor> {
    const OWNER: &str = "vsa_compose_gated_rows";

    let extents = check_gated(OWNER, fine, compressed, gate, tile_rows)?;
    let (tiles, tile, heads, width) = extents;
    let members = slots.len();

    let mut ordered = slots.to_vec();
    ordered.sort_unstable();
    if members == 0 || members > MAX_MEMBERS || !ordered.iter().copied().eq(0..members as u32) {
        return Err(Error::shape(
            OWNER,
            format!("the slots {slots:?} are not one per member of one to {MAX_MEMBERS}"),
        ));
    }
    let owner_rows = tiles * tile / members;
    if rows.end > owner_rows || rows.start > rows.end || !(tiles * tile).is_multiple_of(members) {
        return Err(Error::shape(
            OWNER,
            format!(
                "rows {rows:?} of {owner_rows} rows per member, {members} members in {} rows",
                tiles * tile
            ),
        ));
    }

    if on_cpu(fine) {
        let gated = gated_sum(fine, compressed, gate, extents)?;

        // Block `slots[m]` of the result is member `m`'s interval.
        let mut blocks = vec![None; members];
        for (member, slot) in slots.iter().enumerate() {
            let first = member * owner_rows + rows.start;
            blocks[*slot as usize] = Some(gated.narrow(0, first, rows.len())?);
        }
        let blocks: Vec<Tensor> = blocks.into_iter().flatten().collect();
        return Ok(Tensor::stack(&blocks, 0)?);
    }

    let payload = ops::empty((members, rows.len(), heads, width), fine.dtype())?;
    let mut table = [0_u32; MAX_MEMBERS];
    table[..members].copy_from_slice(slots);

    ops::launch::<VsaComposeGatedRows>(
        &ComposeGatedRowsDescriptor {
            dtype: fine.dtype(),
            heads: field(OWNER, heads)?,
            tile_rows,
            width: field(OWNER, width)?,
            block_rows: BLOCK_ROWS,
            slots: table,
            members: members as u32,
        },
        ComposeGatedRowsArgs {
            output: ops::operand(&payload)?,
            fine: ops::operand(fine)?,
            compressed: ops::operand(compressed)?,
            gate: ops::operand(gate)?,
            owner_rows: owner_rows as u64,
            start: rows.start as u64,
            count: rows.len() as u64,
        },
    )?;
    Ok(payload)
}
