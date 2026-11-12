//! Attention over the tokens an indexer selects: the scores of a request's index keys, the best
//! of them, and the attention over the tokens a selection names.
//!
//! A sparse attention layer keeps one compressed index key per `ratio` consecutive tokens beside
//! its pages of keys and values ([`KvPlanes::index_keys`](crate::KvPlanes::index_keys)). A query
//! row scores the keys of the groups it may see ([`index_scores`]), keeps the best
//! ([`super::top_k`]), and attends the tokens of those groups and of its own unfinished group
//! ([`indexed_attention`]), as vLLM's Qwen sparse attention does
//! (`refs/vllm/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py:20-277`,
//! `refs/vllm/vllm/models/qwen4_exp/nvidia/ops/qsa.py:75-330`):
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
use uniserve_tensor::{DType, Device, Tensor};

use super::{ACCUMULATE, on_cpu};
use crate::capability::KvCache;
use crate::error::{Error, Result};

/// Index keys one `index_scores` kernel scores.
const SCORES_BLOCK: u32 = 256;

/// The values of `column`, a U32 index column of a CPU batch.
fn column(column: &Tensor) -> Result<Vec<u32>> {
    Ok(column.flatten_all()?.to_vec1::<u32>()?)
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
/// `query` is `[tokens, heads, width]`, the rows' index queries. Layer `layer` of `cache` holds
/// the index keys, `[pages, groups, width]`, and `rows`, `[tokens]` U32, names each token's row
/// of the cache's page table. `visible`, `[tokens]` U32, is the number of leading groups of its
/// request a token scores: the groups complete before it.
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
    const OWNER: &str = "index_scores";

    let keys =
        cache.layer(layer)?.index_keys.as_ref().ok_or_else(|| {
            Error::shape(OWNER, format!("cache layer {layer} holds no index keys"))
        })?;
    let (&[tokens, heads, width], &[_, groups, key_width], &[_, max_pages]) =
        (query.dims(), keys.dims(), cache.block_tables.dims())
    else {
        return Err(Error::shape(
            OWNER,
            format!(
                "queries of {:?}, index keys of {:?} and a page table of {:?}: [tokens, heads, \
                 width], [pages, groups, width] and [rows, max_pages] expected",
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

    if on_cpu(query) {
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
                    OWNER,
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
        return Ok(Tensor::stack(&scores, 0)?);
    }

    let out = backend::empty((tokens, columns), DType::F32)?;
    backend::launch::<IndexScores>(
        &IndexScoresDescriptor {
            dtype: query.dtype(),
            keys: keys.dtype(),
            heads: heads as u32,
            width: width as u32,
            groups: groups as u32,
            block: SCORES_BLOCK,
        },
        IndexScoresArgs {
            query: backend::operand(query)?,
            keys: backend::operand(keys)?,
            page_table: backend::operand(&cache.block_tables)?,
            rows: backend::operand(rows)?,
            visible: backend::operand(visible)?,
            out: backend::operand(&out)?,
        },
    )?;
    Ok(out)
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
    const OWNER: &str = "indexed_attention";

    let planes = cache.layer(layer)?;
    let (key, value) = (&planes.key, &planes.value);
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

    if on_cpu(query) {
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
        return Ok(Tensor::stack(&out, 0)?.to_dtype(query.dtype())?);
    }

    let out = backend::empty((tokens, heads, head_dim), query.dtype())?;
    backend::launch::<IndexedAttention>(
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
            query: backend::operand(query)?,
            key: backend::operand(key)?,
            value: backend::operand(value)?,
            page_table: backend::operand(&cache.block_tables)?,
            rows: backend::operand(rows)?,
            selection: backend::operand(selection)?,
            out: backend::operand(&out)?,
        },
    )?;
    Ok(out)
}
