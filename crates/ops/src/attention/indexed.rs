//! Attention over the tokens an indexer selects: the index scores and the attention itself.
//!
//! A sparse attention layer keeps, beside each page of keys and values, one compressed **index
//! key** per `ratio` consecutive tokens. A query row scores the index keys of its request with a
//! few index heads, keeps the best groups, and attends only the tokens of those groups and of its
//! own unfinished group (vLLM's Qwen sparse attention,
//! `refs/vllm/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py:20-219` and
//! `refs/vllm/vllm/models/qwen4_exp/nvidia/ops/qsa.py:75-330`):
//!
//! ```text
//! index_scores        s[t, g] = Σ_h relu(q[t, h] · key[g])        g < visible[t], −inf otherwise
//! indexed_attention   o[t, i] = Σ_n softmax_n(q[t, i] · k[n, i / group] · scale) · v[n, i / group]
//!                                                                  n ∈ selection[t], n ≥ 0
//! ```
//!
//! Both read a request's pages through the page table the paged attention reads
//! ([`super`]): `rows[t]` names the table row of token `t`, group `g` of a request is row
//! `g mod groups` of index-key page `table[rows[t], g / groups]`, and token `n` is row
//! `n mod page` of key and value page `table[rows[t], n / page]`. Neither checks a table entry:
//! the invariant is the table's producer's, as it is for paged attention.
//!
//! The index keys are selected between, so the scores are ranked and never rounded: FP32. The
//! attention computes its scores, its softmax and its sum in FP32; with caches of a 16-bit type
//! each probability is rounded into that type before it multiplies its value, as the reference
//! casts them (`qsa.py:141-253`), and the output rounds once.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::{Float, Key};
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The largest magnitude a cached value is declared to have, for the attention's absolute bound:
/// the envelope paged attention declares for the same caches.
const VALUE_ENVELOPE: f32 = 64.0;

/// Checks the columns both operators address a request's pages by: the `[rows, max_pages]` u32
/// page table and the `[tokens]` u32 table row of each token.
fn check_table(
    operator: &'static str,
    table: &Operand,
    rows: &Operand,
    tokens: u64,
) -> Result<u64> {
    table.require_rank(operator, 2)?;
    let max_pages = table.dim(1);
    table
        .require_row(operator, DType::U32, max_pages)
        .map_err(|error| Error::argument(operator, format!("page table: {error}")))?;
    if !table.is_row_major() || max_pages == 0 {
        return Err(Error::argument(
            operator,
            "the page table is not contiguous rows of pages".to_owned(),
        ));
    }

    rows.require_shape(operator, &[tokens])
        .and_then(|()| rows.require_row(operator, DType::U32, tokens))
        .map_err(|error| Error::argument(operator, format!("rows: {error}")))?;
    Ok(max_pages)
}

/// One compilation of `index_scores`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IndexScoresDescriptor {
    /// Element type of the index queries.
    pub dtype: DType,
    /// Element type of the index keys.
    pub keys: DType,
    /// Index heads of one query row.
    pub heads: u32,
    /// Elements of one index head and of one index key.
    pub width: u32,
    /// Index keys of one page: the page's tokens over the tokens one key stands for.
    pub groups: u32,
    /// Index keys one program scores.
    pub block: u32,
}

impl fmt::Display for IndexScoresDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("keys", abi::name(self.keys))
            .field("heads", self.heads)
            .field("width", self.width)
            .field("groups", self.groups)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of `index_scores`.
#[derive(Clone)]
pub struct IndexScoresArgs {
    /// `[tokens, heads, width]`, contiguous: the index queries.
    pub query: Operand,
    /// `[pages, groups, width]`, contiguous: the index keys of one layer.
    pub keys: Operand,
    /// `[rows, max_pages]` u32: each row's pages.
    pub page_table: Operand,
    /// `[tokens]` u32: each token's row of the page table.
    pub rows: Operand,
    /// `[tokens]` u32: the groups each token scores, the leading ones of its request.
    pub visible: Operand,
    /// `[tokens, max_pages · groups]` FP32, contiguous: receives the scores, `−inf` from a
    /// token's visible count on.
    pub out: Operand,
}

/// The scores of the index keys a query row may select among.
///
/// One program scores [`IndexScoresDescriptor::block`] keys of one token: the products and the
/// sum over the heads are FP32.
pub struct IndexScores;

impl Op for IndexScores {
    const NAME: &'static str = proto::module::operator::INDEX_SCORES;

    type Descriptor = IndexScoresDescriptor;
    type Args<'a> = IndexScoresArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let d = descriptor;
        if d.heads == 0 || d.width == 0 || d.groups == 0 || d.block == 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} index heads of {} elements over pages of {} keys, in blocks of {}",
                    d.heads, d.width, d.groups, d.block
                ),
            ));
        }
        let (heads, width, groups) = (u64::from(d.heads), u64::from(d.width), u64::from(d.groups));

        args.query.require_rank(Self::NAME, 3)?;
        let tokens = args.query.dim(0);
        args.query
            .require_shape(Self::NAME, &[tokens, heads, width])?;
        args.query.require_row(Self::NAME, d.dtype, width)?;

        args.keys.require_rank(Self::NAME, 3)?;
        let pages = args.keys.dim(0);
        args.keys
            .require_shape(Self::NAME, &[pages, groups, width])?;
        args.keys.require_row(Self::NAME, d.keys, width)?;

        let max_pages = check_table(Self::NAME, &args.page_table, &args.rows, tokens)?;
        args.visible
            .require_shape(Self::NAME, &[tokens])
            .and_then(|()| args.visible.require_row(Self::NAME, DType::U32, tokens))
            .map_err(|error| Error::argument(Self::NAME, format!("visible: {error}")))?;

        args.out
            .require_shape(Self::NAME, &[tokens, max_pages * groups])?;
        args.out
            .require_row(Self::NAME, DType::F32, max_pages * groups)?;
        for (name, view) in [
            ("query", &args.query),
            ("keys", &args.keys),
            ("out", &args.out),
        ] {
            if !view.is_row_major() {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} is not contiguous"),
                ));
            }
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.query.dim(0);
        let max_pages = args.page_table.dim(1);
        let columns = max_pages * u64::from(descriptor.groups);

        let mut arguments = Arguments::new();
        arguments
            .view(&args.query)
            .view(&args.keys)
            .view(&args.page_table)
            .view(&args.rows)
            .view(&args.visible)
            .view(&args.out)
            .u64(tokens)
            .u64(max_pages);
        kernel.launch(
            LaunchGrid::new([columns.div_ceil(u64::from(descriptor.block)), tokens, 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        IndexScoresArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Each score sums `heads` rectified dot products of `width` terms in FP32 and is stored
        // as FP32: a dot product's bound over `heads · width` terms.
        Tolerance::dot(
            u64::from(descriptor.heads) * u64::from(descriptor.width),
            DType::F32,
            DType::F32,
        )
    }
}

/// One compilation of `indexed_attention`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IndexedAttentionDescriptor {
    /// Element type of the queries and the output.
    pub dtype: DType,
    /// Element type of the cached keys and values.
    pub cache: DType,
    /// Query heads.
    pub heads: u32,
    /// Key/value heads; each serves `heads / kv_heads` consecutive query heads.
    pub kv_heads: u32,
    /// Elements of one head.
    pub head_dim: u32,
    /// Tokens of one cache page.
    pub page: u32,
    /// Columns of a selection row: the reduction length of one query's softmax.
    pub columns: u32,
    /// The scale the scores are multiplied by.
    pub scale: Float,
}

impl fmt::Display for IndexedAttentionDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("cache", abi::name(self.cache))
            .field("heads", self.heads)
            .field("kv_heads", self.kv_heads)
            .field("head_dim", self.head_dim)
            .field("page", self.page)
            .field("columns", self.columns)
            .field("scale", self.scale)
            .finish(formatter)
    }
}

/// The arguments of `indexed_attention`.
#[derive(Clone)]
pub struct IndexedAttentionArgs {
    /// `[tokens, heads, head_dim]`, contiguous: the queries.
    pub query: Operand,
    /// `[pages, page, kv_heads, head_dim]`, contiguous: the cached keys of one layer.
    pub key: Operand,
    /// `[pages, page, kv_heads, head_dim]`, contiguous: the cached values of the layer.
    pub value: Operand,
    /// `[rows, max_pages]` u32: each row's pages.
    pub page_table: Operand,
    /// `[tokens]` u32: each token's row of the page table.
    pub rows: Operand,
    /// `[tokens, columns]` i32, contiguous: the positions each token attends, in its request's
    /// token order; a negative entry names no token.
    pub selection: Operand,
    /// `[tokens, heads, head_dim]`, contiguous: receives the attended values.
    pub out: Operand,
}

/// Paged attention over the tokens each query row's selection names.
///
/// A query whose selection names no token receives zeros.
pub struct IndexedAttention;

impl Op for IndexedAttention {
    const NAME: &'static str = proto::module::operator::INDEXED_ATTENTION;

    type Descriptor = IndexedAttentionDescriptor;
    type Args<'a> = IndexedAttentionArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let d = descriptor;
        if d.kv_heads == 0
            || !d.heads.is_multiple_of(d.kv_heads)
            || d.heads == 0
            || d.head_dim == 0
            || d.page == 0
            || d.columns == 0
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} query heads over {} key/value heads of {}, pages of {} tokens and \
                     selections of {}",
                    d.heads, d.kv_heads, d.head_dim, d.page, d.columns
                ),
            ));
        }
        let (heads, kv_heads) = (u64::from(d.heads), u64::from(d.kv_heads));
        let (head_dim, page) = (u64::from(d.head_dim), u64::from(d.page));

        args.query.require_rank(Self::NAME, 3)?;
        let tokens = args.query.dim(0);
        for (name, view) in [("query", &args.query), ("out", &args.out)] {
            view.require_shape(Self::NAME, &[tokens, heads, head_dim])
                .and_then(|()| view.require_row(Self::NAME, d.dtype, head_dim))
                .map_err(|error| Error::argument(Self::NAME, format!("{name}: {error}")))?;
        }

        args.key.require_rank(Self::NAME, 4)?;
        let pages = args.key.dim(0);
        for (name, view) in [("key", &args.key), ("value", &args.value)] {
            view.require_shape(Self::NAME, &[pages, page, kv_heads, head_dim])
                .and_then(|()| view.require_row(Self::NAME, d.cache, head_dim))
                .map_err(|error| Error::argument(Self::NAME, format!("{name}: {error}")))?;
        }

        check_table(Self::NAME, &args.page_table, &args.rows, tokens)?;
        let columns = u64::from(d.columns);
        args.selection
            .require_shape(Self::NAME, &[tokens, columns])
            .and_then(|()| args.selection.require_row(Self::NAME, DType::I32, columns))
            .map_err(|error| Error::argument(Self::NAME, format!("selection: {error}")))?;

        for (name, view) in [
            ("query", &args.query),
            ("key", &args.key),
            ("value", &args.value),
            ("selection", &args.selection),
            ("out", &args.out),
        ] {
            if !view.is_row_major() {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} is not contiguous"),
                ));
            }
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.query.dim(0);

        let mut arguments = Arguments::new();
        arguments
            .view(&args.query)
            .view(&args.key)
            .view(&args.value)
            .view(&args.page_table)
            .view(&args.rows)
            .view(&args.selection)
            .view(&args.out)
            .u64(tokens)
            .u64(args.page_table.dim(1));
        kernel.launch(
            LaunchGrid::new([u64::from(descriptor.heads), tokens, 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        IndexedAttentionArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    /// Paged attention's bound with the selection's width as the reduction length: a softmax
    /// over at most `columns` scores accumulated in FP32, `rtol = 2γₙ + u_out`, and the
    /// normalized weights' value term `(2γₙ + u) · max|v|` as the absolute part. With caches of
    /// a 16-bit type each probability also rounds once into it, a relative `u_cache` per term.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = u64::from(descriptor.columns);
        let gamma = tolerance::gamma(reduction, DType::F32);
        let value_term = 2.0 * gamma + tolerance::unit_roundoff(DType::F32);
        let rounded = Tolerance::rounding(reduction, DType::F32, descriptor.dtype);

        Tolerance {
            rtol: rounded.rtol + tolerance::unit_roundoff(descriptor.cache) as f32,
            ..rounded
        }
        .with_atol((value_term * f64::from(VALUE_ENVELOPE)) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_keys_state_the_heads_the_pages_and_the_selection() {
        let scores = IndexScoresDescriptor {
            dtype: DType::BF16,
            keys: DType::BF16,
            heads: 4,
            width: 128,
            groups: 16,
            block: 256,
        };
        assert_eq!(
            scores.to_string(),
            "dtype_bf16_keys_bf16_heads_4_width_128_groups_16_block_256"
        );

        let attention = IndexedAttentionDescriptor {
            dtype: DType::BF16,
            cache: DType::BF16,
            heads: 24,
            kv_heads: 2,
            head_dim: 256,
            page: 64,
            columns: 2052,
            scale: 0.0625_f32.into(),
        };
        assert_eq!(
            attention.to_string(),
            "dtype_bf16_cache_bf16_heads_24_kv_heads_2_head_dim_256_page_64_columns_2052_\
             scale_0x3d800000"
        );
    }

    #[test]
    fn the_attention_bound_follows_the_selection_width() {
        let wide = IndexedAttentionDescriptor {
            dtype: DType::BF16,
            cache: DType::BF16,
            heads: 24,
            kv_heads: 2,
            head_dim: 256,
            page: 64,
            columns: 2052,
            scale: 0.0625_f32.into(),
        };
        let narrow = IndexedAttentionDescriptor {
            columns: 36,
            ..wide
        };

        let (wide, narrow) = (
            IndexedAttention::tolerance(&wide),
            IndexedAttention::tolerance(&narrow),
        );
        assert!(narrow.rtol < wide.rtol && wide.rtol < 0.02);
        assert!(wide.allows(1.0, 1.0) && !wide.allows(0.0, 1.0));
    }
}
