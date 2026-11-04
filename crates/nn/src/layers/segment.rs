//! Block-sparse attention over a video transformer's packed tiles.
//!
//! A sparse attention layout packs its rows into tiles of a fixed block size: `[text | conditions |
//! audio | generated video | padding]`. The tiles before the generated video's first tile are
//! **dense** — every query row reads all of them — and each query tile of the video then reads the
//! fine tiles its selection names. Both sets share one softmax normalizer per query row, so the
//! attended result cannot be assembled from two calls added together; it is one call over the union
//! of the two sets, which is what the selection tables already say (`packing::SegmentExtent::
//! key_bounds` gives every prefix tile all keys, so the selection's kept-tiles-first order *is* the
//! union — the layer reads it as given and adds nothing).
//!
//! Two zeroing rules keep the padded rows out of the result, and they belong on different sides:
//!
//! - invalid **key** rows of a partial tile are zeroed by the caller before the call
//!   (`VsaTiles::zero_padding` on k and v), so no mask exists and none is needed;
//! - invalid **query** rows of a partial tile are the packed sequence's padding, and their output
//!   must be zero so nothing leaks into a later residual. The layer zeroes them after the attention,
//!   inside the same captured launch ([`VsaTiles::zero_padding`] on the output).
//!
//! The compressed-tile branch is deliberately not here: it is one elementwise gate the model folds
//! in with [`Rows::add`](crate::layers::Rows), where its arithmetic stays visible.

use ops::attention::sparse::{SparseArgs, SparseAttention, SparseDescriptor};
use ops::segments::{self, VsaZeroPadding};
use ops::{DType, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;

/// The rows one zeroing program handles at a time.
const BLOCK_ROWS: u32 = segments::BLOCK_ROWS;

/// The selection a model's packing prepared for one call.
///
/// Every view is the model's: the layer neither assembles nor edits the tables, because the union of
/// the dense prefix and the fine selection is exactly what the packing's key bounds already order.
#[derive(Clone, Copy)]
pub struct SegmentTables {
    /// `[heads, query_tiles, selected]` i32: each query tile's key tiles, in read order.
    pub indices: TensorView,
    /// `[heads, query_tiles]` i32: how many of each row's tiles are read.
    pub counts: TensorView,
    /// `[tiles]` i32: the valid rows of each key tile, which is also each query tile's.
    pub sizes: TensorView,
}

/// Attention over a sparse layout's whole tiles.
#[derive(Clone, Copy)]
pub struct SegmentAttention {
    block: u32,
    heads: u32,
    head_dim: u32,
    keys: u32,
    output: DType,
}

impl SegmentAttention {
    /// Attention over `block`-row tiles of `heads` heads `head_dim` wide, serving up to `keys` key
    /// rows.
    ///
    /// `block` is the layout's tile size and it is checkpoint-specific (64 for the FastH3 students,
    /// 128 for OmniRef); query and key tiles are the same size, because a tile is one block of
    /// packed rows on both sides. `keys` is the largest key extent the compiled program serves, from
    /// the same packing. `output` is the element type the attended rows are written in.
    pub fn new(block: u64, heads: u64, head_dim: u64, keys: u64, output: DType) -> Result<Self> {
        let narrow = |name: &'static str, value: u64| {
            u32::try_from(value).map_err(|_| {
                Error::shape(
                    "SegmentAttention",
                    format!("{name} {value} is wider than a descriptor holds"),
                )
            })
        };
        let block = narrow("block", block)?;
        let (heads, head_dim, keys) = (
            narrow("heads", heads)?,
            narrow("head_dim", head_dim)?,
            narrow("keys", keys)?,
        );
        if block == 0 || heads == 0 || head_dim == 0 || keys == 0 {
            return Err(Error::shape(
                "SegmentAttention",
                "a tile, a head and a key bound are all positive".to_owned(),
            ));
        }
        if !keys.is_multiple_of(block) {
            return Err(Error::shape(
                "SegmentAttention",
                format!("a key bound of {keys} rows is not whole {block}-row tiles"),
            ));
        }
        Ok(Self {
            block,
            heads,
            head_dim,
            keys,
            output,
        })
    }

    /// The tile's row count.
    pub const fn block(&self) -> u32 {
        self.block
    }

    /// The heads one row carries.
    pub const fn heads(&self) -> u32 {
        self.heads
    }

    /// The elements of one head.
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// Attend `query` against the tiles `tables` name, writing zeroed padding rows for the rest.
    ///
    /// `query`, `key` and `value` are `[tiles · block, heads, head_dim]`; `out` has the query's
    /// shape. The softmax scale is `1/sqrt(head_dim)`, applied once to `q · k`: the sparse path has
    /// no learned softmax scale.
    pub fn forward(
        &self,
        f: &Forward,
        query: TensorView,
        key: TensorView,
        value: TensorView,
        tables: &SegmentTables,
        out: TensorView,
    ) -> Result<()> {
        let head_dim = f64::from(self.head_dim);
        let descriptor = SparseDescriptor {
            block: self.block,
            dtype: query.dtype,
            output: self.output,
            head_dim: self.head_dim,
            lse: false,
            keys: self.keys,
        };
        f.operate::<SparseAttention>(
            &descriptor,
            SparseArgs {
                query,
                key,
                value,
                indices: tables.indices,
                counts: tables.counts,
                valid: tables.sizes,
                lse: None,
                out,
                scale: (1.0 / head_dim.sqrt()) as f32,
            },
        )?;
        // The padding rows of a partial last tile are the packed sequence's, and nothing may leak
        // from them into the residual that adds this output; the same sizes that bound the key
        // tiles bound the query tiles.
        let zeroing = segments::ZeroPaddingDescriptor {
            dtype: self.output,
            heads: self.heads,
            tile_rows: self.block,
            width: self.head_dim,
            block_rows: BLOCK_ROWS,
        };
        f.operate::<VsaZeroPadding>(
            &zeroing,
            segments::ZeroPaddingArgs {
                values: out,
                sizes: tables.sizes,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tile_and_its_key_bound_are_whole_and_positive() {
        assert!(SegmentAttention::new(64, 12, 128, 4096, DType::BF16).is_ok());
        assert!(SegmentAttention::new(128, 12, 128, 4096, DType::BF16).is_ok());
        assert!(SegmentAttention::new(0, 12, 128, 4096, DType::BF16).is_err());
        assert!(SegmentAttention::new(12, 12, 128, 4096, DType::BF16).is_err());
        assert!(SegmentAttention::new(64, 0, 128, 4096, DType::BF16).is_err());
        assert!(SegmentAttention::new(64, 12, 0, 4096, DType::BF16).is_err());
        assert!(
            SegmentAttention::new(64, 12, 128, 4100, DType::BF16).is_err(),
            "a key bound that is not whole tiles cannot be addressed by tiles"
        );
    }

    #[test]
    fn the_geometry_states_the_tile_the_heads_and_the_head() {
        let attention = SegmentAttention::new(64, 12, 128, 4096, DType::BF16).unwrap();
        assert_eq!(
            (attention.block(), attention.heads(), attention.head_dim()),
            (64, 12, 128)
        );
    }
}
