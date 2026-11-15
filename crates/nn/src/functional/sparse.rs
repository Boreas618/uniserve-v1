//! Block-sparse attention: every query tile attends the key blocks a selection names for it.
//!
//! The rows are cut into `block`-row tiles. For query row `t` of head `h` inside query tile `q`,
//! with `s_j = q_t,h · k_j,h / sqrt(head_dim)` over the keys `j` of the blocks
//! `indices[h, q, 0..counts[h, q]]` that are visible (a key row past its block's `valid` entry
//! is not):
//!
//! ```text
//! p_j     = exp(s_j − max_j s_j)
//! out_t,h = Σ_j p_j · v_j,h / Σ_j p_j
//! ```
//!
//! accumulated in FP32 and rounded once into the output type. A query tile that selects no
//! visible key writes zeros. The operator reads the selection from the device, so its cost
//! follows the selection and not the sequence length, and one captured call serves any
//! selection of the same shape.

use ops::attention::sparse::{SparseArgs, SparseAttention, SparseDescriptor};
use tensor::{DType, Device, Tensor};

use super::{Fused, compute};
use crate::error::{Error, Result};

/// The rows `query` attends in the key blocks its tile selects: `[query rows, heads, head_dim]`
/// in `output` elements.
///
/// `query` is `[query rows, heads, head_dim]` and `key` and `value` `[key rows, heads,
/// head_dim]`, each whole `block`-row tiles of one element type. `indices` is
/// `[heads, query tiles, selected]` `U32`, the key blocks each query tile reads, of which the
/// first `counts[head, query tile]` are read (`counts` is `[heads, query tiles]` `U32`);
/// `valid` is `[key blocks]` `U32`, the visible rows of each key block. `keys` is the largest
/// key extent, in rows, the deployment serves.
///
/// # Errors
///
/// When the rows are not whole tiles of three axes, a table is not `U32` of its extent, or the
/// operator refuses its arguments.
pub fn sparse_attention(
    (query, key, value): (&Tensor, &Tensor, &Tensor),
    (indices, counts): (&Tensor, &Tensor),
    valid: &Tensor,
    block: u32,
    keys: u32,
    output: DType,
) -> Result<Tensor> {
    const OWNER: &str = SparseAttentionCall::OWNER;

    let (&[query_rows, heads, head_dim], &[key_rows, ..]) = (query.dims(), key.dims()) else {
        return Err(Error::shape(
            OWNER,
            format!(
                "the queries and keys are [rows, heads, head_dim], not {:?} and {:?}",
                query.dims(),
                key.dims()
            ),
        ));
    };
    let tile = block as usize;
    if tile == 0
        || !query_rows.is_multiple_of(tile)
        || !key_rows.is_multiple_of(tile)
        || key.dims() != [key_rows, heads, head_dim]
        || value.dims() != key.dims()
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "queries of {:?}, keys of {:?} and values of {:?} are not whole {block}-row \
                 tiles of one head shape",
                query.dims(),
                key.dims(),
                value.dims()
            ),
        ));
    }

    let (query_tiles, key_blocks) = (query_rows / tile, key_rows / tile);
    let tables = [
        (
            "indices",
            indices,
            indices.rank() == 3 && indices.dims()[..2] == [heads, query_tiles],
        ),
        ("counts", counts, counts.dims() == [heads, query_tiles]),
        ("valid sizes", valid, valid.dims() == [key_blocks]),
    ];
    for (name, table, fits) in tables {
        if !fits || table.dtype() != DType::U32 {
            return Err(Error::shape(
                OWNER,
                format!(
                    "the {name} are {} {:?} for {heads} heads, {query_tiles} query tiles and \
                     {key_blocks} key blocks of u32 entries",
                    ops::abi::name(table.dtype()),
                    table.dims()
                ),
            ));
        }
    }
    let scale = (1.0 / (head_dim as f64).sqrt()) as f32;

    compute(SparseAttentionCall {
        query,
        key,
        value,
        indices,
        counts,
        valid,
        block,
        keys,
        output,
        heads,
        head_dim,
        tile,
        query_tiles,
        key_blocks,
        scale,
    })
}

/// A checked call of [`sparse_attention`]: the heads, the tiles and the softmax scale are its rows'.
struct SparseAttentionCall<'t> {
    query: &'t Tensor,
    key: &'t Tensor,
    value: &'t Tensor,
    indices: &'t Tensor,
    counts: &'t Tensor,
    valid: &'t Tensor,
    block: u32,
    keys: u32,
    output: DType,
    heads: usize,
    head_dim: usize,
    tile: usize,
    query_tiles: usize,
    key_blocks: usize,
    scale: f32,
}

impl SparseAttentionCall<'_> {
    const OWNER: &'static str = "sparse_attention";
}

impl Fused for SparseAttentionCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.query.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            query,
            key,
            value,
            indices,
            counts,
            valid,
            output,
            heads,
            head_dim,
            tile,
            query_tiles,
            key_blocks,
            scale,
            ..
        } = self;
        let selected: Vec<Vec<Vec<u32>>> = indices.to_vec3()?;
        let read: Vec<Vec<u32>> = counts.to_vec2()?;
        let visible: Vec<u32> = valid.to_vec1()?;

        // Each head's rows as a packed matrix: `[heads, rows, head_dim]` in FP32.
        let by_head = |rows: &Tensor| -> Result<Tensor> {
            Ok(rows.to_dtype(DType::F32)?.transpose(0, 1)?.contiguous()?)
        };
        let (queries, keys, values) = (by_head(query)?, by_head(key)?, by_head(value)?);
        let scale = Tensor::new(scale, &Device::Cpu)?;

        let mut heads_out = Vec::with_capacity(heads);
        for head in 0..heads {
            let (head_keys, head_values) = (keys.get(head)?, values.get(head)?);
            let mut tiles_out = Vec::with_capacity(query_tiles);

            for query_tile in 0..query_tiles {
                // The visible key rows of the blocks this query tile reads, in selection order.
                let count = read[head][query_tile] as usize;
                let mut rows: Vec<u32> = Vec::new();
                for block in selected[head][query_tile].iter().take(count) {
                    let Some(visible) = visible.get(*block as usize) else {
                        return Err(Error::shape(
                            Self::OWNER,
                            format!("key block {block} of {key_blocks} is selected"),
                        ));
                    };
                    let first = block * tile as u32;
                    rows.extend(first..first + visible);
                }
                if rows.is_empty() {
                    tiles_out.push(Tensor::zeros((tile, head_dim), DType::F32, &Device::Cpu)?);
                    continue;
                }

                let visible_rows = rows.len();
                let rows = Tensor::from_vec(rows, visible_rows, &Device::Cpu)?;
                let tile_queries = queries.get(head)?.narrow(0, query_tile * tile, tile)?;
                let scores = tile_queries
                    .matmul(&head_keys.index_select(&rows, 0)?.t()?)?
                    .broadcast_mul(&scale)?;

                let weights = scores.broadcast_sub(&scores.max_keepdim(1)?)?.exp()?;
                let attended = weights
                    .matmul(&head_values.index_select(&rows, 0)?)?
                    .broadcast_div(&weights.sum_keepdim(1)?)?;
                tiles_out.push(attended);
            }
            heads_out.push(Tensor::cat(&tiles_out, 0)?);
        }
        Ok(Tensor::stack(&heads_out, 1)?.to_dtype(output)?)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            query,
            key,
            value,
            indices,
            counts,
            valid,
            block,
            keys,
            output,
            head_dim,
            scale,
            ..
        } = self;
        let out = ops::empty(query.shape(), output)?;
        ops::launch::<SparseAttention>(
            &SparseDescriptor {
                block,
                dtype: query.dtype(),
                output,
                head_dim: u32::try_from(head_dim).map_err(|_| {
                    Error::shape(Self::OWNER, format!("heads of {head_dim} elements"))
                })?,
                lse: false,
                keys,
            },
            SparseArgs {
                query: ops::operand(query)?,
                key: ops::operand(key)?,
                value: ops::operand(value)?,
                indices: ops::operand(indices)?,
                counts: ops::operand(counts)?,
                valid: ops::operand(valid)?,
                lse: None,
                out: ops::operand(&out)?,
                scale,
            },
        )?;
        Ok(out)
    }
}
