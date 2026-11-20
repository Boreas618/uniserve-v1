//! Attention over the tokens an indexer selects: the scores of a request's index keys, the best
//! of them, and the attention over the tokens a selection names.
//!
//! A sparse attention layer keeps one compressed index key per `ratio` consecutive tokens beside
//! its pages of keys and values, the plane of its [`INDEX_KEY`] field ([`KvCache::plane`]). A query
//! row scores the keys of the groups it may see ([`index_scores`]), keeps the best
//! ([`super::top_k`]), and attends the tokens of those groups and of its own unfinished group
//! ([`indexed_attention`]):
//!
//! ```text
//! index_scores        s[t, g] = Σ_h relu(q[t, h] · key[g])        g < visible[t], −inf otherwise
//! indexed_attention   o[t, i] = Σ_n softmax_n(q[t, i] · k[n, i / group] · scale) · v[n, i / group]
//!                                                                  n ∈ selection[t], n ≥ 0
//! ```
//!
//! The scoring and the attention read a request's pages through the cache's page table:
//! `rows[t]` names the table row of token `t`, which is the token's own row in a batch of one
//! token per row and its request's row in a prefill.

use ops::attention::{
    IndexScores, IndexScoresArgs, IndexScoresDescriptor, IndexedAttention, IndexedAttentionArgs,
    IndexedAttentionDescriptor,
};
use tensor::{DType, Device, Tensor};

use super::{ACCUMULATE, Fused, compute};
use crate::capability::KvCache;
use crate::error::{Error, Result};
use crate::family::{INDEX_KEY, KEY, VALUE};

/// The values of `column`, a U32 index column of a CPU batch.
fn column(column: &Tensor) -> Result<Vec<u32>> {
    Ok(column.flatten_all()?.to_vec1::<u32>()?)
}

/// The cache layer `layer` names, as a plane states its owner.
fn cache_layer(layer: usize) -> Result<u32> {
    u32::try_from(layer).map_err(|_| {
        Error::shape(
            "KvCache",
            format!("layer {layer} is beyond the cache's layers"),
        )
    })
}

/// Checks `rows`, the `[tokens]` U32 table row of each token.
fn check_rows(owner: &'static str, rows: &Tensor, tokens: usize) -> Result<()> {
    if rows.dims() != [tokens] || rows.dtype() != DType::U32 {
        return Err(Error::shape(
            owner,
            format!(
                "the tokens' table rows are {} {:?}, [{tokens}] u32 expected",
                ops::abi::name(rows.dtype()),
                rows.dims()
            ),
        ));
    }
    Ok(())
}

/// The scores of the index keys each query row may select among: `[tokens, max_pages · groups]`
/// FP32, entry `g` of a row the score of its request's group `g` and `−inf` from the row's
/// visible count on.
///
/// `query` is `[tokens, heads, width]`, the rows' index queries. Layer `layer`'s plane of the
/// [`INDEX_KEY`] field of `cache` holds the index keys, `[pages, groups, 1, width]`, and
/// `rows`, `[tokens]` U32, names each token's row of the cache's page table. `visible`,
/// `[tokens]` U32, is the number of leading groups of its request a token scores: the groups
/// complete before it.
///
/// # Errors
///
/// A layer without index keys, queries of another width than the keys, columns that are not one
/// entry per token, or the operator's refusal.
pub fn index_scores(
    query: &Tensor,
    cache: &KvCache,
    layer: usize,
    rows: &Tensor,
    visible: &Tensor,
) -> Result<Tensor> {
    const OWNER: &str = IndexScoresCall::OWNER;

    let plane = cache.plane(INDEX_KEY, cache_layer(layer)?)?;
    let keys = &plane.entries;
    let (&[tokens, heads, width], &[pages, groups, 1, key_width], &[_, max_pages]) =
        (query.dims(), keys.dims(), cache.block_tables.dims())
    else {
        return Err(Error::shape(
            OWNER,
            format!(
                "queries of {:?}, index keys of {:?} and a page table of {:?}: [tokens, heads, \
                 width], [pages, groups, 1, width] and [rows, max_pages] expected",
                query.dims(),
                keys.dims(),
                cache.block_tables.dims()
            ),
        ));
    };
    if key_width != width || heads == 0 || width == 0 || groups == 0 {
        return Err(Error::shape(
            OWNER,
            format!("index queries of {width} elements against keys of {key_width}"),
        ));
    }
    check_rows(OWNER, rows, tokens)?;
    if visible.dims() != [tokens] || visible.dtype() != DType::U32 {
        return Err(Error::shape(
            OWNER,
            format!(
                "the visible counts are {:?}, [{tokens}] u32 expected",
                visible.dims()
            ),
        ));
    }
    let columns = max_pages * groups;
    // One key a group: the kernel reads the plane as `[pages, groups, width]`.
    let keys = &keys.reshape((pages, groups, key_width))?;

    compute(IndexScoresCall {
        query,
        cache,
        keys,
        rows,
        visible,
        tokens,
        heads,
        width,
        groups,
        columns,
    })
}

/// A checked call of [`index_scores`]: `keys` are the layer's index keys.
struct IndexScoresCall<'t> {
    query: &'t Tensor,
    cache: &'t KvCache,
    keys: &'t Tensor,
    rows: &'t Tensor,
    visible: &'t Tensor,
    tokens: usize,
    heads: usize,
    width: usize,
    groups: usize,
    columns: usize,
}

impl IndexScoresCall<'_> {
    const OWNER: &'static str = "index_scores";
}

impl Fused for IndexScoresCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.query.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            query,
            cache,
            keys,
            rows,
            visible,
            tokens,
            width,
            groups,
            columns,
            ..
        } = self;
        let table = cache.block_tables.to_vec2::<u32>()?;
        let (rows, visible) = (column(rows)?, column(visible)?);
        let queries = query.to_dtype(ACCUMULATE)?;
        let keys = keys.to_dtype(ACCUMULATE)?;

        let mut scores = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let seen = visible[token] as usize;
            let hidden =
                Tensor::full(f32::NEG_INFINITY, columns - seen.min(columns), &Device::Cpu)?;
            if seen == 0 {
                scores.push(hidden);
                continue;
            }
            if seen > columns {
                return Err(Error::shape(
                    Self::OWNER,
                    format!("token {token} sees {seen} groups of a table of {columns}"),
                ));
            }

            // The leading groups of the token's request, page by page.
            let pages = &table[rows[token] as usize][..seen.div_ceil(groups)];
            let pages = Tensor::from_vec(pages.to_vec(), pages.len(), &Device::Cpu)?;
            let seen_keys = keys
                .index_select(&pages, 0)?
                .reshape((pages.elem_count() * groups, width))?
                .narrow(0, 0, seen)?;

            let products = queries.get(token)?.matmul(&seen_keys.t()?)?;
            scores.push(Tensor::cat(&[products.relu()?.sum(0)?, hidden], 0)?);
        }
        Ok(Tensor::stack(&scores, 0)?)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            query,
            cache,
            keys,
            rows,
            visible,
            tokens,
            heads,
            width,
            groups,
            columns,
        } = self;
        let out = ops::empty((tokens, columns), DType::F32)?;
        ops::launch::<IndexScores>(
            &IndexScoresDescriptor::new(
                query.dtype(),
                keys.dtype(),
                heads as u32,
                width as u32,
                groups as u32,
            ),
            IndexScoresArgs {
                query: ops::operand(query)?,
                keys: ops::operand(keys)?,
                page_table: ops::operand(&cache.block_tables)?,
                rows: ops::operand(rows)?,
                visible: ops::operand(visible)?,
                out: ops::operand(&out)?,
            },
        )?;
        Ok(out)
    }
}

/// Attention of each query row over the tokens its selection names, through the cache's pages:
/// `[tokens, heads, head_dim]` in the queries' element type.
///
/// `query` is `[tokens, heads, head_dim]`. Layer `layer` of `cache` holds the keys and values,
/// `[pages, page_size, kv_heads, head_dim]` each in a 16-bit or FP32 type, each key/value head
/// serving `heads / kv_heads` consecutive query heads. `rows`, `[tokens]` U32, names each
/// token's row of the cache's page table, and `selection`, `[tokens, columns]` I32, the
/// positions the token attends in its request's token order, a negative entry naming none. A
/// row that names no token is zeros.
///
/// The scores (times `scale`), the softmax and the sum are FP32; each probability is rounded
/// into the cache's element type before it multiplies its value, and the output rounds once.
///
/// # Errors
///
/// Planes that are not the queries' heads' width or are FP8 (a scaled cache is decoded by the
/// paged attention alone), query heads the key/value heads do not group, columns that are not
/// one entry per token, or the operator's refusal.
pub fn indexed_attention(
    query: &Tensor,
    cache: &KvCache,
    layer: usize,
    rows: &Tensor,
    selection: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    const OWNER: &str = IndexedAttentionCall::OWNER;

    let layer = cache_layer(layer)?;
    let (key, value) = (
        &cache.plane(KEY, layer)?.entries,
        &cache.plane(VALUE, layer)?.entries,
    );
    let (&[tokens, heads, head_dim], &[pages, page, kv_heads, key_dim]) =
        (query.dims(), key.dims())
    else {
        return Err(Error::shape(
            OWNER,
            format!(
                "queries of {:?} and keys of {:?}: [tokens, heads, head_dim] and [pages, \
                 page_size, kv_heads, head_dim] expected",
                query.dims(),
                key.dims()
            ),
        ));
    };
    if value.dims() != key.dims()
        || value.dtype() != key.dtype()
        || key_dim != head_dim
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || page != cache.page_size as usize
        || matches!(key.dtype(), DType::F8E4M3 | DType::U8)
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "{heads} query heads of {head_dim} over {} keys of {:?} and {} values of {:?}, \
                 in pages of {}",
                ops::abi::name(key.dtype()),
                key.dims(),
                ops::abi::name(value.dtype()),
                value.dims(),
                cache.page_size
            ),
        ));
    }
    check_rows(OWNER, rows, tokens)?;
    let &[selected_rows, columns] = selection.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "a selection of {:?} is not [tokens, columns]",
                selection.dims()
            ),
        ));
    };
    if selected_rows != tokens || columns == 0 || selection.dtype() != DType::I32 {
        return Err(Error::shape(
            OWNER,
            format!(
                "a selection of {} {:?} for {tokens} tokens",
                ops::abi::name(selection.dtype()),
                selection.dims()
            ),
        ));
    }

    compute(IndexedAttentionCall {
        query,
        cache,
        key,
        value,
        rows,
        selection,
        scale,
        tokens,
        heads,
        head_dim,
        pages,
        page,
        kv_heads,
        columns,
    })
}

/// A checked call of [`indexed_attention`]: `key` and `value` are the layer's planes.
struct IndexedAttentionCall<'t> {
    query: &'t Tensor,
    cache: &'t KvCache,
    key: &'t Tensor,
    value: &'t Tensor,
    rows: &'t Tensor,
    selection: &'t Tensor,
    scale: f32,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    pages: usize,
    page: usize,
    kv_heads: usize,
    columns: usize,
}

impl IndexedAttentionCall<'_> {
    const OWNER: &'static str = "indexed_attention";
}

impl Fused for IndexedAttentionCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.query.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            query,
            cache,
            key,
            value,
            rows,
            selection,
            scale,
            tokens,
            heads,
            head_dim,
            pages,
            page,
            kv_heads,
            ..
        } = self;
        let table = cache.block_tables.to_vec2::<u32>()?;
        let rows = column(rows)?;
        let selected = selection.to_vec2::<i32>()?;
        let group = heads / kv_heads;

        let cached = (pages * page, kv_heads, head_dim);
        let (keys, values) = (key.reshape(cached)?, value.reshape(cached)?);
        let queries = query.to_dtype(ACCUMULATE)?;

        let mut out = Vec::with_capacity(tokens);
        for token in 0..tokens {
            // The cache rows of the tokens the row names.
            let pages = &table[rows[token] as usize];
            let slots: Vec<u32> = selected[token]
                .iter()
                .filter(|position| **position >= 0)
                .map(|position| {
                    let position = *position as usize;
                    pages[position / page] * page as u32 + (position % page) as u32
                })
                .collect();
            if slots.is_empty() {
                out.push(Tensor::zeros((heads, head_dim), ACCUMULATE, &Device::Cpu)?);
                continue;
            }

            let slots = Tensor::from_vec(slots.clone(), slots.len(), &Device::Cpu)?;
            let token_keys = keys.index_select(&slots, 0)?.to_dtype(ACCUMULATE)?;
            let token_values = values.index_select(&slots, 0)?.to_dtype(ACCUMULATE)?;

            let mut attended = Vec::with_capacity(kv_heads);
            for head in 0..kv_heads {
                let head_queries = queries.get(token)?.narrow(0, head * group, group)?;
                let head_keys = token_keys.narrow(1, head, 1)?.squeeze(1)?;
                let head_values = token_values.narrow(1, head, 1)?.squeeze(1)?;

                let scores = head_queries
                    .matmul(&head_keys.t()?)?
                    .affine(f64::from(scale), 0.0)?;
                let shifted = scores.broadcast_sub(&scores.max_keepdim(1)?)?.exp()?;
                let weights = shifted.broadcast_div(&shifted.sum_keepdim(1)?)?;

                // The reference casts the probabilities to the values' type before the product.
                let weights = weights.to_dtype(key.dtype())?.to_dtype(ACCUMULATE)?;
                attended.push(weights.matmul(&head_values)?);
            }
            out.push(Tensor::cat(&attended, 0)?);
        }
        Ok(Tensor::stack(&out, 0)?.to_dtype(query.dtype())?)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            query,
            cache,
            key,
            value,
            rows,
            selection,
            scale,
            tokens,
            heads,
            head_dim,
            kv_heads,
            columns,
            ..
        } = self;
        let out = ops::empty((tokens, heads, head_dim), query.dtype())?;
        ops::launch::<IndexedAttention>(
            &IndexedAttentionDescriptor {
                dtype: query.dtype(),
                cache: key.dtype(),
                heads: heads as u32,
                kv_heads: kv_heads as u32,
                head_dim: head_dim as u32,
                page: cache.page_size,
                columns: columns as u32,
                scale: scale.into(),
            },
            IndexedAttentionArgs {
                query: ops::operand(query)?,
                key: ops::operand(key)?,
                value: ops::operand(value)?,
                page_table: ops::operand(&cache.block_tables)?,
                rows: ops::operand(rows)?,
                selection: ops::operand(selection)?,
                out: ops::operand(&out)?,
            },
        )?;
        Ok(out)
    }
}
