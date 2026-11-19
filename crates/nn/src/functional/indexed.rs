//! Attention over the tokens an indexer selects: the scores of a request's index keys, the best
//! of them, and the attention over the tokens a selection names.
//!
//! A sparse attention layer keeps one compressed index key per `ratio` consecutive tokens beside
//! its cache, the plane of its [`crate::family::INDEX_KEY`] field ([`KvCache::plane`]). A query
//! row scores the keys of the groups it may see ([`index_scores`]), keeps the best
//! ([`index_select`]), and attends the tokens of those groups ([`indexed_attention`], or the
//! compressed latents themselves, [`super::latent_attention`]):
//!
//! ```text
//! index_scores        s[t, g] = Σ_h relu(q[t, h] · key[g]) · w[t, h]   g < visible[t], −inf otherwise
//! index_select        sel[t]  = the k groups of the largest scores, −1 for a −inf one
//! indexed_attention   o[t, i] = Σ_n softmax_n(q[t, i] · k[n, i / group] · scale) · v[n, i / group]
//!                                                                  n ∈ selection[t], n ≥ 0
//! ```
//!
//! The scoring and the attention read a request's pages through a page table: `rows[t]` names
//! the table row of token `t`, which is the token's own row in a batch of one token per row and
//! its request's row in a prefill. [`ops::attention::indexed`] states the candidate blocks the
//! scores may be restricted to and the block mode of the selection.

use ops::attention::{
    IndexScores, IndexScoresArgs, IndexScoresDescriptor, IndexSelect, IndexSelectArgs,
    IndexSelectDescriptor, IndexedAttention, IndexedAttentionArgs, IndexedAttentionDescriptor,
};
use ops::record::{Format, Layout};
use tensor::{DType, Device, Tensor};

use super::latent::Plane;
use super::{ACCUMULATE, Fused, compute};
use crate::capability::KvCache;
use crate::error::{Error, Result};
use crate::family::{KEY, VALUE};

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

/// How [`index_scores`] scores beyond the plain sum of rectified products.
#[derive(Clone, Copy)]
pub struct IndexScoring<'t> {
    /// `[tokens, heads]` F32: each row's head weights, when the heads are weighted.
    pub weights: Option<&'t Tensor>,
    /// How a query is read: as it is ([`Format::Bf16`]), or rounded to [`Format::Mxfp4`] blocks
    /// as the reference rounds its index queries.
    pub query: Format,
    /// `[tokens, blocks]` I32 candidate blocks of the given number of groups each, when the
    /// scores are restricted to them: column `c` then scores group `candidates[t, c / B] · B +
    /// c mod B`, a negative block naming none.
    pub candidates: Option<(&'t Tensor, u32)>,
}

impl Default for IndexScoring<'_> {
    /// The unweighted sum over every visible group, each query read as it is.
    fn default() -> Self {
        Self {
            weights: None,
            query: Format::Bf16,
            candidates: None,
        }
    }
}

/// The monotone key of an FP32 score, the order a radix selection ranks by: larger keys for
/// larger scores, and `−0` below `+0`.
fn monotone(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// The scores of the index keys each query row may select among: `[tokens, columns]` FP32,
/// `−inf` for a column of no visible group.
///
/// `query` is `[tokens, heads, width]`, the rows' index queries. `keys` is the layer's index-key
/// plane: `[pages, groups, width]` plain values for [`Format::Bf16`] (BF16, or F16/F32 on the CPU),
/// or packed records whose pages keep their scales after every record's values
/// ([`Layout::Page`]). `rows`, `[tokens]` U32, names each token's row of the plane's table, and
/// `visible`, `[tokens]` U32, the number of leading groups of its request a token sees. Without
/// candidates a row has `max_pages · groups` columns, column `g` the request's group `g`; with
/// candidates of `B` groups it has `blocks · B`.
///
/// # Errors
///
/// A plane of another width than the queries or of another layout than a packed one's, columns
/// that are not one entry per token, weights or candidates of other shapes, or the operator's
/// refusal.
pub fn index_scores(
    query: &Tensor,
    keys: Plane<'_>,
    rows: &Tensor,
    visible: &Tensor,
    scoring: IndexScoring<'_>,
) -> Result<Tensor> {
    const OWNER: &str = IndexScoresCall::OWNER;

    let (&[tokens, heads, width], &[_, groups, _], &[_, max_pages]) =
        (query.dims(), keys.records.dims(), keys.table.dims())
    else {
        return Err(Error::shape(
            OWNER,
            format!(
                "queries of {:?}, index keys of {:?} and a page table of {:?}: [tokens, heads, \
                 width], [pages, groups, record] and [rows, max_pages] expected",
                query.dims(),
                keys.records.dims(),
                keys.table.dims()
            ),
        ));
    };
    let (_, last) = keys.format.plane(width as u64);
    let plain = !keys.format.packed()
        && matches!(keys.records.dtype(), DType::BF16 | DType::F16 | DType::F32);
    let packed =
        keys.format.packed() && keys.layout == Layout::Page && keys.records.dtype() == DType::U8;
    if keys.records.dims()[2] as u64 != last
        || !(plain || packed)
        || heads == 0
        || width == 0
        || groups == 0
        || keys.table.dtype() != DType::U32
        || keys.format.check_width(OWNER, width as u64).is_err()
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "index queries of {width} elements against {} {} keys of {:?} in the {} layout",
                keys.format,
                ops::abi::name(keys.records.dtype()),
                keys.records.dims(),
                keys.layout
            ),
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
    if let Some(weights) = scoring.weights
        && (weights.dims() != [tokens, heads] || weights.dtype() != DType::F32)
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "head weights of {:?}, [{tokens}, {heads}] f32 expected",
                weights.dims()
            ),
        ));
    }
    let columns = match scoring.candidates {
        Some((candidates, block)) => match *candidates.dims() {
            [candidate_rows, blocks]
                if candidate_rows == tokens
                    && blocks > 0
                    && block > 0
                    && candidates.dtype() == DType::I32 =>
            {
                blocks * block as usize
            }
            _ => {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "candidate blocks of {:?} and {block} groups for {tokens} tokens",
                        candidates.dims()
                    ),
                ));
            }
        },
        None => max_pages * groups,
    };

    compute(IndexScoresCall {
        query,
        keys,
        rows,
        visible,
        scoring,
        tokens,
        heads,
        width,
        groups,
        columns,
    })
}

/// A checked call of [`index_scores`].
struct IndexScoresCall<'t> {
    query: &'t Tensor,
    keys: Plane<'t>,
    rows: &'t Tensor,
    visible: &'t Tensor,
    scoring: IndexScoring<'t>,
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
            keys,
            rows,
            visible,
            scoring,
            tokens,
            heads,
            width,
            groups,
            columns,
        } = self;
        let table = keys.table.to_vec2::<u32>()?;
        let (rows, visible) = (column(rows)?, column(visible)?);
        let decoded = if keys.format.packed() {
            keys_decoded(&keys, width)?
        } else {
            let pages = keys.records.dims()[0];
            keys.records
                .reshape((pages * groups, width))?
                .to_dtype(ACCUMULATE)?
        };

        // A query rounded to MXFP4 is each head's blocks encoded and decoded, as the reference
        // rounds it in place before it scores.
        let queries = query.to_dtype(ACCUMULATE)?;
        let queries = if scoring.query == Format::Bf16 {
            queries
        } else {
            let values = queries.flatten_all()?.to_vec1::<f32>()?;
            let rounded: Vec<f32> = values
                .chunks(width)
                .flat_map(|head| {
                    let (codes, scales) = ops::record::encode(scoring.query, head);
                    ops::record::decode(scoring.query, width, &codes, &scales)
                })
                .collect();
            Tensor::from_vec(rounded, (tokens, heads, width), &Device::Cpu)?
        };
        let weights = match scoring.weights {
            Some(weights) => Some(weights.to_vec2::<f32>()?),
            None => None,
        };
        let candidates = match scoring.candidates {
            Some((candidates, block)) => Some((candidates.to_vec2::<i32>()?, block as usize)),
            None => None,
        };

        let mut scores = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let row = &table[rows[token] as usize];
            let seen = visible[token] as usize;

            // The group each column scores, `None` for a column of no visible group.
            let named: Vec<Option<usize>> = (0..columns)
                .map(|column| {
                    let group = match &candidates {
                        Some((candidates, block)) => {
                            let candidate = candidates[token][column / block];
                            usize::try_from(candidate).ok()? * block + column % block
                        }
                        None => column,
                    };
                    (group < seen && group / groups < row.len()).then_some(group)
                })
                .collect();
            let slots: Vec<u32> = named
                .iter()
                .flatten()
                .map(|group| row[group / groups] * groups as u32 + (group % groups) as u32)
                .collect();

            let mut out = vec![f32::NEG_INFINITY; columns];
            if !slots.is_empty() {
                let seen_keys = decoded.index_select(
                    &Tensor::from_vec(slots.clone(), slots.len(), &Device::Cpu)?,
                    0,
                )?;
                // `[heads, n]` rectified products, weighted per head and summed over the heads.
                let products = queries.get(token)?.matmul(&seen_keys.t()?)?.relu()?;
                let products = match &weights {
                    Some(weights) => products.broadcast_mul(&Tensor::from_vec(
                        weights[token].clone(),
                        (heads, 1),
                        &Device::Cpu,
                    )?)?,
                    None => products,
                };
                let summed = products.sum(0)?.to_vec1::<f32>()?;
                let mut next = summed.into_iter();
                for (column, group) in named.iter().enumerate() {
                    if group.is_some() {
                        out[column] = next.next().unwrap_or(f32::NEG_INFINITY);
                    }
                }
            }
            scores.push(Tensor::from_vec(out, columns, &Device::Cpu)?);
        }
        Ok(Tensor::stack(&scores, 0)?)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            query,
            keys,
            rows,
            visible,
            scoring,
            tokens,
            heads,
            width,
            groups,
            columns,
        } = self;
        let out = ops::empty((tokens, columns), DType::F32)?;
        ops::launch::<IndexScores>(
            &IndexScoresDescriptor {
                weighted: scoring.weights.is_some(),
                query: scoring.query,
                candidates: scoring.candidates.map_or(0, |(_, block)| block),
                ..IndexScoresDescriptor::new(
                    query.dtype(),
                    keys.format,
                    heads as u32,
                    width as u32,
                    groups as u32,
                )
            },
            IndexScoresArgs {
                query: ops::operand(query)?,
                weights: scoring.weights.map(ops::operand).transpose()?,
                keys: ops::operand(keys.records)?,
                page_table: ops::operand(keys.table)?,
                rows: ops::operand(rows)?,
                visible: ops::operand(visible)?,
                candidates: scoring
                    .candidates
                    .map(|(candidates, _)| ops::operand(candidates))
                    .transpose()?,
                out: ops::operand(&out)?,
            },
        )?;
        Ok(out)
    }
}

/// The decoded index keys of a packed CPU plane, FP32 `[pages · groups, width]`.
fn keys_decoded(keys: &Plane<'_>, width: usize) -> Result<Tensor> {
    let (pages, groups) = (keys.records.dims()[0], keys.records.dims()[1]);
    let bytes = keys.records.flatten_all()?.to_vec1::<u8>()?;
    let page_bytes = bytes.len() / pages;
    let (value_bytes, scale_bytes) = (
        keys.format.value_bytes(width as u64) as usize,
        keys.format.scale_bytes(width as u64) as usize,
    );

    let mut values = Vec::with_capacity(pages * groups * width);
    for page in bytes.chunks(page_bytes) {
        for slot in 0..groups {
            let (at, scales_at) =
                keys.layout
                    .offsets(keys.format, groups as u64, width as u64, slot as u64);
            let (at, scales_at) = (at as usize, scales_at as usize);
            values.extend(ops::record::decode(
                keys.format,
                width,
                &page[at..at + value_bytes],
                &page[scales_at..scales_at + scale_bytes],
            ));
        }
    }
    Ok(Tensor::from_vec(
        values,
        (pages * groups, width),
        &Device::Cpu,
    )?)
}

/// The `kept` best groups (or blocks) of each row of `scores`, `[tokens, columns]` F32:
/// `[tokens, kept]` I32, `-1` for a kept slot whose score is `−inf` ([`ops::attention::IndexSelect`]).
///
/// With `pool > 1` a block of `pool` columns ranks by its largest score, the block holding the
/// row's newest visible group (`(visible − 1) / pool`) ranks first, and the kept entries are
/// blocks: the candidates of a later [`index_scores`]. With `candidates` (`[tokens, blocks]` I32
/// blocks of the given number of groups, the restriction the scores were computed under) a kept
/// column maps back to its group. A tie keeps the lower column; the kept entries leave the
/// strictly greater ones first, then the ties, each by column.
///
/// # Errors
///
/// A kept count of none or above the row's blocks, columns that are not one entry per token,
/// candidates beside the block mode, or the operator's refusal.
pub fn index_select(
    scores: &Tensor,
    visible: &Tensor,
    candidates: Option<(&Tensor, u32)>,
    pool: u32,
    kept: usize,
) -> Result<Tensor> {
    const OWNER: &str = IndexSelectCall::OWNER;

    let &[tokens, columns] = scores.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("scores of {:?}, [tokens, columns] expected", scores.dims()),
        ));
    };
    let blocks = columns.div_ceil(pool.max(1) as usize);
    if scores.dtype() != DType::F32
        || pool == 0
        || kept == 0
        || kept > blocks
        || (pool > 1 && candidates.is_some())
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "{kept} kept of {} scores of {:?} in blocks of {pool}{}",
                ops::abi::name(scores.dtype()),
                scores.dims(),
                if candidates.is_some() {
                    " restricted to candidates"
                } else {
                    ""
                }
            ),
        ));
    }
    check_rows(OWNER, visible, tokens)?;
    if let Some((candidates, block)) = candidates
        && (candidates.dims().len() != 2
            || candidates.dims()[0] != tokens
            || candidates.dims()[1] * block as usize != columns
            || candidates.dtype() != DType::I32)
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "candidate blocks of {:?} and {block} groups for scores of {:?}",
                candidates.dims(),
                scores.dims()
            ),
        ));
    }

    compute(IndexSelectCall {
        scores,
        visible,
        candidates,
        pool,
        kept,
    })
}

/// A checked call of [`index_select`].
struct IndexSelectCall<'t> {
    scores: &'t Tensor,
    visible: &'t Tensor,
    candidates: Option<(&'t Tensor, u32)>,
    pool: u32,
    kept: usize,
}

impl IndexSelectCall<'_> {
    const OWNER: &'static str = "index_select";
}

impl Fused for IndexSelectCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.scores.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            scores,
            visible,
            candidates,
            pool,
            kept,
        } = self;
        let (scores, visible) = (scores.to_vec2::<f32>()?, column(visible)?);
        let candidates = match candidates {
            Some((candidates, block)) => Some((candidates.to_vec2::<i32>()?, block as usize)),
            None => None,
        };
        let pool = pool as usize;

        let mut out = Vec::with_capacity(scores.len() * kept);
        for (token, row) in scores.iter().enumerate() {
            // The pooled keys, a block's largest, the newest visible block pinned above all.
            let mut keys: Vec<u32> = row
                .chunks(pool)
                .map(|block| {
                    block
                        .iter()
                        .map(|score| monotone(*score))
                        .max()
                        .unwrap_or(0)
                })
                .collect();
            if pool > 1 && visible[token] > 0 {
                let newest = (visible[token] as usize - 1) / pool;
                if let Some(key) = keys.get_mut(newest) {
                    *key = monotone(f32::INFINITY);
                }
            }

            // The k-th largest key, then the strictly greater ones by column and the ties at
            // it by column until the row is full.
            let mut ranked = keys.clone();
            ranked.sort_unstable_by(|a, b| b.cmp(a));
            let threshold = ranked[kept - 1];
            let above: Vec<usize> = (0..keys.len()).filter(|&b| keys[b] > threshold).collect();
            let ties = (0..keys.len())
                .filter(|&b| keys[b] == threshold)
                .take(kept - above.len());
            let none = monotone(f32::NEG_INFINITY);
            for block in above.into_iter().chain(ties) {
                let entry = if keys[block] == none {
                    -1
                } else {
                    match &candidates {
                        Some((candidates, size)) => {
                            let candidate = candidates[token][block / size];
                            if candidate < 0 {
                                -1
                            } else {
                                candidate * *size as i32 + (block % size) as i32
                            }
                        }
                        None => block as i32,
                    }
                };
                out.push(entry);
            }
        }
        Ok(Tensor::from_vec(out, (scores.len(), kept), &Device::Cpu)?)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            scores,
            visible,
            candidates,
            pool,
            kept,
        } = self;
        let out = ops::empty((scores.dims()[0], kept), DType::I32)?;
        ops::launch::<IndexSelect>(
            &IndexSelectDescriptor::new(pool, candidates.map_or(0, |(_, block)| block)),
            IndexSelectArgs {
                scores: ops::operand(scores)?,
                visible: ops::operand(visible)?,
                candidates: candidates
                    .map(|(candidates, _)| ops::operand(candidates))
                    .transpose()?,
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
