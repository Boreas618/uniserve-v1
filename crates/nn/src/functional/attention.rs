//! Attention: dense over packed sequences, and paged over the key/value cache.
//!
//! ```text
//! s[i, j]   = softmax_scale · q[i, h] · k[j, g]          g = h / (q_heads / kv_heads)
//! p[i, j]   = exp(s[i, j] − max_j s[i, j])               over the keys j visible to query i
//! out[i, h] = Σ_j p[i, j] · v[j, g] / Σ_j p[i, j]
//! ```
//!
//! Every sum is formed in FP32 and the output is rounded once into the query's element type. A
//! query with no visible key has an output row of zeros.
//!
//! [`dense_attention`] attends packed queries to packed keys as one sequence, and
//! [`dense_attention_varlen`] as the sequences two offset tables name, in the form of flash
//! attention's variable-length call. Under the causal mask a key is visible to the queries at
//! its own place in its sequence and after it, both counted from their sequence's start; a
//! history window bounds how far back a query sees.
//!
//! [`paged_attention`] reads the keys and values a batch's rows address in the cache's pages.
//! Row `r` holds `seq_lens[r]` keys, and its queries are its last ones, so the mask is causal
//! by position; its key `p` is at
//!
//! ```text
//! slot = block_tables[r, p / page_size] · page_size + p mod page_size
//! ```
//!
//! [`kv_write_`] is the write that precedes it: it scatters a batch's keys and values into the
//! slots the scheduler assigned them. A dense cache may hold another of BF16, F16 and F32 than
//! the projections, and the write rounds each element into it to nearest even. A cache whose
//! planes are E4M3 holds each page divided by the page's own FP32 scale, which the write derives
//! from the values it stores and the read multiplies back.
//!
//! The element types of the tensors decide the kernel, so an FP32 attention is the same function
//! over FP32 tensors; the widths and head counts are the tensors' own extents.

use ops::attention::dense::{DenseArgs, DenseAttention, DenseDescriptor};
use ops::attention::{Args, Descriptor, Mask, PagedAttention, Phase};
use ops::fp8::E4M3_MAX;
use ops::kv::{self, KvWrite, KvWriteScaled};
use tensor::{D, DType, Device, Shape, Tensor};

use super::nvfp4::integer;
use super::rotary::heads_of;
use super::{ACCUMULATE, Fused, compute};
use crate::capability::{KvCache, KvPlanes, NO_SLOT, TokenBatch};
use crate::error::{Error, Result};

/// The floor under a page's largest magnitude, so a write of zeros still has a scale to divide
/// by.
const SCALE_FLOOR: f32 = 1e-12;

/// A history window as a launch states it: the keys before a query's own it sees, or `-1` for
/// no window.
fn window_left(window: Option<u32>) -> i32 {
    window.map_or(-1, |window| window as i32)
}

/// Refuses query heads that do not group the key heads, or heads of two widths.
fn grouped(owner: &'static str, query: (u32, u32), key: (u32, u32)) -> Result<()> {
    let ((heads, head_dim), (kv_heads, kv_dim)) = (query, key);

    if !heads.is_multiple_of(kv_heads) || head_dim != kv_dim {
        return Err(Error::shape(
            owner,
            format!(
                "{heads} query heads of {head_dim} do not group {kv_heads} key heads of {kv_dim}"
            ),
        ));
    }
    Ok(())
}

/// `value` as a rank-0 FP32 CPU tensor: a scalar operand of a true FP32 division, where the
/// tensor library's scalar division multiplies by a reciprocal.
fn scalar(value: f32) -> Result<Tensor> {
    Ok(Tensor::new(value, &Device::Cpu)?)
}

/// The causal visibility of keys at `key_positions`, `[keys]`, to queries at `query_positions`,
/// `[queries]`, both I64 places within one sequence: `[queries, keys]` U8, key `j` visible to
/// query `i` when `j ≤ i` and, under a window, `j ≥ i − window`.
fn causal_mask(
    query_positions: &Tensor,
    key_positions: &Tensor,
    window: Option<u32>,
) -> Result<Tensor> {
    let query = query_positions.unsqueeze(1)?;
    let key = key_positions.unsqueeze(0)?;
    let past = key.broadcast_le(&query)?;

    match window {
        None => Ok(past),
        Some(window) => {
            let first = (query - f64::from(window))?;
            Ok((past * key.broadcast_ge(&first)?)?)
        }
    }
}

/// The softmax attention of the module documentation, in FP32.
///
/// `query` is `[queries, q_heads, head_dim]`; `key` and `value` are `[queries, keys, kv_heads,
/// head_dim]`, each query's own keys, or `[1, keys, kv_heads, head_dim]` when every query reads
/// the same ones; `visible` is `[queries, keys]` U8. The result is `[queries, q_heads,
/// head_dim]` FP32.
fn attend(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    visible: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let (queries, heads, head_dim) = query.dims3()?;
    let (_, keys, kv_heads, _) = key.dims4()?;
    let group = heads / kv_heads;

    // Each key head's query heads side by side, so one product serves the group.
    let grouped = query.reshape((queries, kv_heads, group, head_dim))?;
    let keys_by_head = key.permute((0, 2, 3, 1))?.contiguous()?;
    let scores = grouped
        .broadcast_matmul(&keys_by_head)?
        .affine(f64::from(softmax_scale), 0.0)?;

    // The largest visible score of each row, then the weights relative to it. An invisible key
    // weighs nothing whatever its score, so a row that sees no key sums to zero.
    let visible = visible
        .reshape((queries, 1, 1, keys))?
        .broadcast_as(scores.shape())?;
    let floor = Tensor::full(f32::MIN, scores.shape(), &Device::Cpu)?;
    let peak = visible
        .where_cond(&scores, &floor)?
        .max_keepdim(D::Minus1)?;
    let weights =
        visible.where_cond(&scores.broadcast_sub(&peak)?.exp()?, &scores.zeros_like()?)?;

    let total = weights.sum_keepdim(D::Minus1)?;
    let total = total.gt(0.0)?.where_cond(&total, &total.ones_like()?)?;
    let values_by_head = value.permute((0, 2, 1, 3))?.contiguous()?;
    let summed = weights.broadcast_matmul(&values_by_head)?;

    Ok(summed
        .broadcast_div(&total)?
        .reshape((queries, heads, head_dim))?)
}

/// Dense attention of `query`, `[q_tokens, heads, head_dim]`, to `key` and `value`,
/// `[kv_tokens, kv_heads, head_dim]`, as **one sequence** of every query and every key:
/// `[q_tokens, heads, head_dim]` in the query's element type.
///
/// `softmax_scale` multiplies `q · k`. With `causal` a query sees the keys at its own row and
/// before it, and with a `window` at most that many keys before its own; without `causal` every
/// query sees every key, which is the form of a cross-attention, whose queries and keys differ
/// in count.
///
/// # Errors
///
/// Tensors that are not heads of one width, query heads that do not group the key heads, a
/// window without the causal mask, a scale that is zero or not finite, or the operator's
/// refusal.
pub fn dense_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    softmax_scale: f32,
    causal: bool,
    window: Option<u32>,
) -> Result<Tensor> {
    let keys = key.dims().first().copied().unwrap_or(0);
    let bound = u32::try_from(keys).map_err(|_| {
        Error::shape(
            "dense_attention",
            format!("{keys} keys exceed a launch's bound"),
        )
    })?;

    dense(
        (query, key, value),
        None,
        bound,
        softmax_scale,
        causal,
        window,
    )
}

/// Dense attention over the variable-length sequences two offset tables name, as flash
/// attention's variable-length call states them.
///
/// `cu_seqlens_q` and `cu_seqlens_k` are `[sequences + 1]` U32 cumulative row counts of the
/// queries and of the keys, each accumulated from zero; a call whose sequences attend
/// themselves passes one table as both. A row past a table's last entry belongs to no sequence:
/// it sees nothing and its output is zeros.
///
/// `max_seqlen_q` is the longest query sequence, when the caller wrote the tables and knows it:
/// the launch sizes its schedule by it instead of by the packed row count, which for many equal
/// sequences asks for that many times the kernels it needs. `max_kv_len` is the bound on the
/// keys this launch reduces over, a launch value because a deployment's buckets differ by
/// thousands of rows; it covers every key passed.
///
/// # Errors
///
/// As [`dense_attention`], and for tables that are not two U32 vectors of one length or a key
/// bound below the keys passed.
#[expect(
    clippy::too_many_arguments,
    reason = "the operands of flash attention's variable-length call, in its order"
)]
pub fn dense_attention_varlen(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    max_seqlen_q: Option<u32>,
    max_kv_len: u32,
    softmax_scale: f32,
    causal: bool,
    window: Option<u32>,
) -> Result<Tensor> {
    dense(
        (query, key, value),
        Some((cu_seqlens_q, cu_seqlens_k, max_seqlen_q)),
        max_kv_len,
        softmax_scale,
        causal,
        window,
    )
}

/// Both forms of dense attention: `sequences` holds the offset tables of the variable-length
/// form, the queries' and the keys', with the longest query sequence when the caller knows it.
fn dense(
    (query, key, value): (&Tensor, &Tensor, &Tensor),
    sequences: Option<(&Tensor, &Tensor, Option<u32>)>,
    max_kv_len: u32,
    softmax_scale: f32,
    causal: bool,
    window: Option<u32>,
) -> Result<Tensor> {
    const OWNER: &str = DenseCall::OWNER;

    let (heads, head_dim) = heads_of(OWNER, "the query", query)?;
    let (kv_heads, kv_dim) = heads_of(OWNER, "the key", key)?;
    let (q_tokens, kv_tokens) = (query.dims()[0], key.dims()[0]);
    grouped(OWNER, (heads, head_dim), (kv_heads, kv_dim))?;
    if value.dims() != key.dims() || value.dtype() != key.dtype() {
        return Err(Error::shape(
            OWNER,
            format!(
                "the value is {} {:?} against keys of {} {:?}",
                ops::abi::name(value.dtype()),
                value.dims(),
                ops::abi::name(key.dtype()),
                key.dims()
            ),
        ));
    }
    if q_tokens == 0 || kv_tokens == 0 {
        return Err(Error::shape(
            OWNER,
            "a call attends from at least one query to at least one key".to_owned(),
        ));
    }
    if window.is_some() && !causal {
        return Err(Error::shape(
            OWNER,
            "a history window reads the causal mask only".to_owned(),
        ));
    }
    if !softmax_scale.is_finite() || softmax_scale == 0.0 {
        return Err(Error::shape(
            OWNER,
            format!("a softmax scale of {softmax_scale}"),
        ));
    }
    if (max_kv_len as usize) < kv_tokens {
        return Err(Error::shape(
            OWNER,
            format!("a {max_kv_len}-key bound for a launch of {kv_tokens} keys"),
        ));
    }
    if let Some((queries, keys, _)) = sequences {
        for (name, table) in [("cu_seqlens_q", queries), ("cu_seqlens_k", keys)] {
            if table.rank() != 1
                || table.dims() != queries.dims()
                || table.elem_count() < 2
                || table.dtype() != DType::U32
            {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "{name} is {} {:?}, and the offsets are two u32 vectors of one entry \
                         per sequence and its end",
                        ops::abi::name(table.dtype()),
                        table.dims()
                    ),
                ));
            }
        }
    }

    compute(DenseCall {
        heads: (query, key, value),
        sequences,
        max_kv_len,
        softmax_scale,
        causal,
        window,
        extents: (heads, kv_heads, head_dim),
    })
}

/// A checked call of [`dense_attention`] or [`dense_attention_varlen`]: `extents` are the query
/// heads, the key heads and their width.
struct DenseCall<'t> {
    heads: (&'t Tensor, &'t Tensor, &'t Tensor),
    sequences: Option<(&'t Tensor, &'t Tensor, Option<u32>)>,
    max_kv_len: u32,
    softmax_scale: f32,
    causal: bool,
    window: Option<u32>,
    extents: (u32, u32, u32),
}

impl DenseCall<'_> {
    const OWNER: &'static str = "dense_attention";
}

impl Fused for DenseCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.heads.0.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            heads: (query, key, value),
            sequences,
            softmax_scale,
            causal,
            window,
            ..
        } = self;
        let (q_tokens, kv_tokens) = (query.dims()[0], key.dims()[0]);

        let visible = match sequences {
            Some((queries, keys, _)) => {
                sequence_mask((queries, q_tokens), (keys, kv_tokens), causal, window)?
            }
            None if causal => causal_mask(
                &Tensor::arange(0_i64, q_tokens as i64, &Device::Cpu)?,
                &Tensor::arange(0_i64, kv_tokens as i64, &Device::Cpu)?,
                window,
            )?,
            None => Tensor::ones((q_tokens, kv_tokens), DType::U8, &Device::Cpu)?,
        };
        let attended = attend(
            &query.to_dtype(ACCUMULATE)?,
            &key.to_dtype(ACCUMULATE)?.unsqueeze(0)?,
            &value.to_dtype(ACCUMULATE)?.unsqueeze(0)?,
            &visible,
            softmax_scale,
        )?;

        Ok(attended.to_dtype(query.dtype())?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            heads: (query, key, value),
            sequences,
            max_kv_len,
            softmax_scale,
            causal,
            window,
            extents: (heads, kv_heads, head_dim),
        } = self;

        let out = ops::empty(query.shape(), query.dtype())?;
        ops::launch::<DenseAttention>(
            // The widths are the tensors' and the key bound the call's; whether offset tables
            // are read selects one of two kernels, so a launch never reads a table at a null
            // address.
            &DenseDescriptor {
                query: query.dtype(),
                key_value: key.dtype(),
                output: out.dtype(),
                head_dim,
                q_heads: heads,
                kv_heads,
                mask: if causal { Mask::Causal } else { Mask::Dense },
                windowed: window.is_some(),
                max_kv_len,
                cu_seq_lens: sequences.is_some(),
            },
            DenseArgs {
                query: ops::operand(query)?,
                key: ops::operand(key)?,
                value: ops::operand(value)?,
                cu_seq_lens_q: sequences
                    .map(|(queries, _, _)| ops::operand(queries))
                    .transpose()?,
                cu_seq_lens_k: sequences
                    .map(|(_, keys, _)| ops::operand(keys))
                    .transpose()?,
                out: ops::operand(&out)?,
                scale: softmax_scale,
                window_left: window_left(window),
                causal,
                longest: sequences.and_then(|(_, _, longest)| longest).map(u64::from),
            },
        )?;
        Ok(out)
    }
}

/// The sequence and the place within it of each of `rows` packed rows, by the cumulative
/// offsets `offsets`, `[sequences + 1]` U32: `(sequence, place)`, both `[rows]` I64. A row past
/// the last offset has the sequence `sequences`, which no table names.
fn sequence_places(offsets: &Tensor, rows: usize) -> Result<(Tensor, Tensor)> {
    let sequences = offsets.elem_count() - 1;
    let offsets = offsets.to_dtype(DType::I64)?;
    let row = Tensor::arange(0_i64, rows as i64, &Device::Cpu)?;

    // A row's sequence is the number of sequence ends at or before it.
    let ends = offsets.narrow(0, 1, sequences)?;
    let sequence = row
        .unsqueeze(1)?
        .broadcast_ge(&ends.unsqueeze(0)?)?
        .to_dtype(DType::I64)?
        .sum(1)?;

    // A row past every sequence reads the last start, and its place is never read.
    let start = offsets.index_select(&sequence.minimum((sequences - 1) as f64)?, 0)?;
    Ok((sequence, (row - start)?))
}

/// The visibility of packed keys to packed queries under two offset tables: `[q_tokens,
/// kv_tokens]` U8. A key is visible to the queries of its own sequence, under the causal mask
/// by their places within it.
fn sequence_mask(
    (query_offsets, q_tokens): (&Tensor, usize),
    (key_offsets, kv_tokens): (&Tensor, usize),
    causal: bool,
    window: Option<u32>,
) -> Result<Tensor> {
    let sequences = (query_offsets.elem_count() - 1) as f64;
    let (query_sequence, query_place) = sequence_places(query_offsets, q_tokens)?;
    let (key_sequence, key_place) = sequence_places(key_offsets, kv_tokens)?;

    let same = query_sequence
        .unsqueeze(1)?
        .broadcast_eq(&key_sequence.unsqueeze(0)?)?;
    let named = query_sequence.lt(sequences)?.unsqueeze(1)?;
    let visible = same.broadcast_mul(&named)?;

    if causal {
        Ok((visible * causal_mask(&query_place, &key_place, window)?)?)
    } else {
        Ok(visible)
    }
}

/// The tokens one page of `cache` holds, which the batch's columns address too.
///
/// The page size enters every descriptor of the cache's kernels, and the cache is laid out in
/// it, so a batch that states another one or a page of no tokens is refused before anything is
/// addressed.
fn page_size(owner: &'static str, batch: &TokenBatch, cache: &KvCache) -> Result<u32> {
    let page = cache.page_size;

    if page == 0 || batch.page_size != page {
        return Err(Error::shape(
            owner,
            format!(
                "the cache has {page} tokens per page and the batch {}",
                batch.page_size
            ),
        ));
    }
    Ok(page)
}

/// The key/value heads and the head width of `planes`, `[pages, page, kv_heads, head_dim]`
/// each, checked against the cache's page and against each other.
fn plane_heads(owner: &'static str, planes: &KvPlanes, page: u32) -> Result<(u32, u32)> {
    if let [_, tokens, heads, head_dim] = *planes.key.dims()
        && let (Ok(heads), Ok(head_dim)) = (u32::try_from(heads), u32::try_from(head_dim))
        && tokens == page as usize
        && heads > 0
        && head_dim > 0
        && planes.value.dims() == planes.key.dims()
    {
        return Ok((heads, head_dim));
    }

    Err(Error::shape(
        owner,
        format!(
            "the cache planes are {:?} keys and {:?} values, [pages, {page}, kv_heads, head_dim] \
             expected of both",
            planes.key.dims(),
            planes.value.dims()
        ),
    ))
}

/// The per-page scale a plane is read through: one FP32 entry per page when the plane is E4M3,
/// and none otherwise.
///
/// An E4M3 plane without its scale would be read unnormalized, a scaled plane of another type
/// would be multiplied by a scale the cache never divided by, and one scalar where the table
/// belongs would scale every page of a sequence by one number. The write divides each page by
/// the page's own scale, so the read takes a table of the plane's pages.
fn page_scale<'t>(plane: &Tensor, scale: Option<&'t Tensor>) -> Result<Option<&'t Tensor>> {
    const OWNER: &str = PagedAttentionCall::OWNER;

    match (plane.dtype() == DType::F8E4M3, scale) {
        (true, Some(scale)) => {
            let pages = plane.dims().first().copied().unwrap_or(0);

            if scale.dtype() != DType::F32 || scale.dims() != [pages, 1, 1, 1] {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "an FP8 plane of {pages} pages has {pages} f32 page scales, \
                         [{pages}, 1, 1, 1]; the scale is {} {:?}",
                        ops::abi::name(scale.dtype()),
                        scale.dims()
                    ),
                ));
            }
            Ok(Some(scale))
        }
        (true, None) => Err(Error::shape(
            OWNER,
            "an FP8 cache plane needs its per-page scale".to_owned(),
        )),
        (false, Some(_)) => Err(Error::shape(
            OWNER,
            format!(
                "a {} cache plane has no scale; the scale belongs to the page's encoding",
                ops::abi::name(plane.dtype())
            ),
        )),
        (false, None) => Ok(None),
    }
}

/// Paged attention of `query`, `[tokens, heads, head_dim]`, over the keys and values the batch's
/// rows hold in layer `layer` of `cache`: `[tokens, heads, head_dim]` in the query's element
/// type.
///
/// A prefill's rows are packed by the batch's query offsets and each row's queries are the last
/// tokens of its `seq_lens` keys; a decode has one query per row, its row's last token. The
/// mask is causal, and a `window` bounds how many keys before its own a query sees.
///
/// `max_kv_len` is the longest key reduction the deployment serves. It enters the descriptor,
/// so it is one value for every batch of a deployment, and the page table addresses no more
/// keys than it.
///
/// # Errors
///
/// A batch whose columns disagree ([`TokenBatch::check`]) or whose phase is not an attention
/// phase, a cache of another page size than the batch's or with a page longer than
/// `max_kv_len`, a page table that is not one U32 row per row of the batch, an FP8 plane
/// without its page scales or a scale on any other plane, heads the planes' do not group, or
/// the operator's refusal.
pub fn paged_attention(
    query: &Tensor,
    batch: &TokenBatch,
    cache: &KvCache,
    layer: usize,
    softmax_scale: f32,
    window: Option<u32>,
    max_kv_len: u32,
) -> Result<Tensor> {
    const OWNER: &str = PagedAttentionCall::OWNER;

    // The batch's columns are read by address arithmetic inside the kernels, so they are
    // checked against each other once here, before any of them is addressed.
    batch.check()?;
    let page = page_size(OWNER, batch, cache)?;
    if max_kv_len < page {
        return Err(Error::shape(
            OWNER,
            format!("a {max_kv_len}-key bound is shorter than a {page}-token page"),
        ));
    }
    let phase = match batch.phase {
        proto::module::Phase::Prefill => Phase::Prefill,
        proto::module::Phase::Decode => Phase::Decode,
        other => {
            return Err(Error::Unsupported(format!(
                "paged_attention: phase {other} is not an attention phase"
            )));
        }
    };

    let planes = cache.layer(layer)?;
    let (heads, head_dim) = heads_of(OWNER, "the query", query)?;
    let (kv_heads, kv_dim) = plane_heads(OWNER, planes, page)?;
    let tokens = query.dims()[0];
    grouped(OWNER, (heads, head_dim), (kv_heads, kv_dim))?;

    // The kernel sees only the table, so a table of another rank or row count would have it
    // read another batch's pages; the batch's rows are known here, and the refusal names them.
    let rows = batch.rows();
    let table = &cache.block_tables;
    if table.rank() != 2 || table.dims()[0] != rows || table.dtype() != DType::U32 {
        return Err(Error::shape(
            OWNER,
            format!(
                "the page table is {} {:?}, u32 [{rows}, pages] expected: one row per row of \
                 this batch",
                ops::abi::name(table.dtype()),
                table.dims()
            ),
        ));
    }
    if phase == Phase::Decode && tokens != rows {
        return Err(Error::shape(
            OWNER,
            format!("a decode has one query per row: {tokens} queries for {rows} rows"),
        ));
    }

    // The plane's element type decides whether the call has scales, so a mismatch in either
    // direction is refused instead of read unscaled. The scales are this layer's own.
    let key_scale = page_scale(&planes.key, planes.key_scale.as_ref())?;
    let value_scale = page_scale(&planes.value, planes.value_scale.as_ref())?;

    compute(PagedAttentionCall {
        query,
        batch,
        cache,
        planes,
        phase,
        softmax_scale,
        window,
        max_kv_len,
        scales: (key_scale, value_scale),
        geometry: (page, heads, kv_heads, head_dim),
    })
}

/// A checked call of [`paged_attention`]: `planes` are the layer's, `scales` their page scales
/// when they are E4M3, and `geometry` the page size, the query heads, the key heads and their
/// width.
struct PagedAttentionCall<'t> {
    query: &'t Tensor,
    batch: &'t TokenBatch,
    cache: &'t KvCache,
    planes: &'t KvPlanes,
    phase: Phase,
    softmax_scale: f32,
    window: Option<u32>,
    max_kv_len: u32,
    scales: (Option<&'t Tensor>, Option<&'t Tensor>),
    geometry: (u32, u32, u32, u32),
}

impl PagedAttentionCall<'_> {
    const OWNER: &'static str = "paged_attention";
}

impl Fused for PagedAttentionCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.query.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            query,
            batch,
            cache,
            planes,
            phase,
            softmax_scale,
            window,
            ..
        } = self;

        let attended = paged_formula(query, batch, cache, planes, phase, softmax_scale, window)?;
        Ok(attended.to_dtype(query.dtype())?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            query,
            batch,
            cache,
            planes,
            phase,
            softmax_scale,
            window,
            max_kv_len,
            scales: (key_scale, value_scale),
            geometry: (page, heads, kv_heads, head_dim),
        } = self;

        let out = ops::empty(query.shape(), query.dtype())?;
        ops::launch::<PagedAttention>(
            // The phase is the batch's, because a prefill reads a whole sequence and a decode
            // one token; the three element types are the query's, the planes' and the
            // output's, because the kernel is compiled for the combination.
            &Descriptor {
                phase,
                query: query.dtype(),
                key_value: planes.key.dtype(),
                output: out.dtype(),
                head_dim,
                page_size: page,
                q_heads: heads,
                kv_heads,
                mask: Mask::Causal,
                windowed: window.is_some(),
                max_kv_len,
            },
            Args {
                query: ops::operand(query)?,
                key: ops::operand(&planes.key)?,
                value: ops::operand(&planes.value)?,
                key_scale: key_scale.map(ops::operand).transpose()?,
                value_scale: value_scale.map(ops::operand).transpose()?,
                page_table: ops::operand(&cache.block_tables)?,
                seq_lens: ops::operand(&batch.seq_lens)?,
                cum_seq_lens_q: ops::operand(&batch.query_offsets)?,
                out: ops::operand(&out)?,
                workspace: ops::operand(&cache.workspace)?,
                scale: softmax_scale,
                window_left: window_left(window),
                causal: true,
            },
        )?;
        Ok(out)
    }
}

/// The FP32 result of [`paged_attention`] on CPU tensors, before its one rounding.
fn paged_formula(
    query: &Tensor,
    batch: &TokenBatch,
    cache: &KvCache,
    planes: &KvPlanes,
    phase: Phase,
    softmax_scale: f32,
    window: Option<u32>,
) -> Result<Tensor> {
    let (tokens, rows) = (query.dims()[0], batch.rows());
    let page = cache.page_size;
    let capacity = cache.block_tables.dims()[1] * page as usize;

    // 1. The slot of every key a row may hold: its page's entry of the table, then the place
    //    within the page. A place the row has not stored reads slot zero and is never visible.
    let place = Tensor::arange(0_u32, capacity as u32, &Device::Cpu)?;
    let page_slot = integer(page, &place)?;
    let slots = cache
        .block_tables
        .index_select(&place.broadcast_div(&page_slot)?, 1)?
        .broadcast_mul(&page_slot)?
        .broadcast_add(&place.broadcast_remainder(&integer(page, &place)?)?)?;
    let stored = place
        .unsqueeze(0)?
        .broadcast_lt(&batch.seq_lens.reshape((rows, 1))?)?;
    let slots = stored
        .where_cond(&slots, &slots.zeros_like()?)?
        .flatten_all()?;

    // 2. The row and the position of each query. A prefill's query is one of its row's last
    //    tokens; a token past the last row is the bucket's padding and sees nothing.
    let seq_lens = batch.seq_lens.to_dtype(DType::I64)?;
    let (row, position, live) = match phase {
        Phase::Decode => (
            Tensor::arange(0_i64, rows as i64, &Device::Cpu)?,
            (seq_lens - 1.0)?,
            None,
        ),
        Phase::Prefill => {
            let offsets = batch.query_offsets.to_dtype(DType::I64)?;
            let ends = offsets.narrow(0, 1, rows)?;
            let token = Tensor::arange(0_i64, tokens as i64, &Device::Cpu)?;
            let counted = token
                .unsqueeze(1)?
                .broadcast_ge(&ends.unsqueeze(0)?)?
                .to_dtype(DType::I64)?
                .sum(1)?;

            let row = counted.minimum((rows - 1) as f64)?;
            let before_end = (token - ends.index_select(&row, 0)?)?;
            let position = (seq_lens.index_select(&row, 0)? + before_end)?;
            (row, position, Some(counted.lt(rows as f64)?))
        }
    };

    // 3. Each query's keys and values, decoded: an E4M3 page times its own scale.
    let gathered = |plane: &Tensor, scale: Option<&Tensor>| -> Result<Tensor> {
        let (pages, _, kv_heads, head_dim) = plane.dims4()?;
        let mut decoded = plane.to_dtype(ACCUMULATE)?;
        if let Some(scale) = scale {
            decoded = decoded.broadcast_mul(scale)?;
        }

        Ok(decoded
            .reshape((pages * page as usize, kv_heads, head_dim))?
            .index_select(&slots, 0)?
            .reshape((rows, capacity, kv_heads, head_dim))?
            .index_select(&row, 0)?)
    };
    let key = gathered(&planes.key, planes.key_scale.as_ref())?;
    let value = gathered(&planes.value, planes.value_scale.as_ref())?;

    // 4. A query sees the keys up to its own position, which its row has stored.
    let mut visible = causal_mask(&position, &place.to_dtype(DType::I64)?, window)?;
    if let Some(live) = live {
        visible = visible.broadcast_mul(&live.unsqueeze(1)?)?;
    }

    attend(
        &query.to_dtype(ACCUMULATE)?,
        &key,
        &value,
        &visible,
        softmax_scale,
    )
}

/// Writes the batch's `key` and `value`, `[tokens, kv_heads, head_dim]` each, into layer
/// `layer` of `cache`, in place: token `t` into slot `slot_mapping[t]` of both planes, raising
/// the flag of every page that receives a row. A slot of [`NO_SLOT`] writes nothing.
///
/// The planes' element type decides the write. Dense planes, BF16, F16 or F32 in both, receive
/// the rows rounded once into their type, to nearest even, both planes in one launch: planes of
/// the projections' own type receive them as they are, and a BF16 model's F16 cache receives
/// each element at F16's nearest value. The rounding is the scatter's own store, so a cache of
/// another type than the projections costs no pass of its own. E4M3 planes receive the rows
/// quantized, one launch per plane, each page by its own scale:
///
/// ```text
/// proposed = max(|rows written to the page|) / 448
/// scale    = max(scale, proposed) for an initialized page, and proposed otherwise
/// page     = e4m3(page · old scale / scale)             when an initialized page's scale grew
/// row      = e4m3(clamp(row / scale, ±448))
/// ```
///
/// so the rows a page already holds keep their decoded values, and the scale and the flag are
/// written with the rows.
///
/// # Errors
///
/// A cache of another page size than the batch's, projections that are not the batch's tokens
/// of the planes' heads, that are strided views (the kernels address a token's row from the
/// tensor's first element) or that differ in type, planes that are neither one dense type in
/// both nor E4M3 in both, an E4M3 plane without its page scales, or the operator's refusal.
pub fn kv_write_(
    cache: &KvCache,
    layer: usize,
    batch: &TokenBatch,
    key: &Tensor,
    value: &Tensor,
) -> Result<()> {
    const OWNER: &str = KvWriteCall::OWNER;

    let page = page_size(OWNER, batch, cache)?;
    let planes = cache.layer(layer)?;
    let (heads, head_dim) = plane_heads(OWNER, planes, page)?;
    let row = [batch.tokens(), heads as usize, head_dim as usize];

    for (name, projection) in [("key", key), ("value", value)] {
        if projection.dims() != row {
            return Err(Error::shape(
                OWNER,
                format!(
                    "the {name} is {:?}, {row:?} expected: the batch's tokens of the cache's \
                     heads",
                    projection.dims()
                ),
            ));
        }

        // Both writes address a token's row from the tensor's first element and are given no
        // stride, so a strided projection would be read as packed rows and land in the wrong
        // slots.
        if !projection.is_contiguous() {
            return Err(Error::shape(
                OWNER,
                format!(
                    "the {name} is a strided view of {:?}: the cache write reads packed rows of \
                     {heads} heads of {head_dim}",
                    projection.dims()
                ),
            ));
        }
    }
    let slots = &batch.slot_mapping;
    if slots.dims() != [row[0]] || slots.dtype() != DType::U32 {
        return Err(Error::shape(
            OWNER,
            format!(
                "the slot mapping is {} {:?}, one u32 slot per token expected",
                ops::abi::name(slots.dtype()),
                slots.dims()
            ),
        ));
    }

    // Both writes take one source type for the keys and the values, as each kernel is compiled
    // for one.
    if key.dtype() != value.dtype() {
        return Err(Error::shape(
            OWNER,
            format!(
                "the keys are {} and the values {}: the cache write reads one element type",
                ops::abi::name(key.dtype()),
                ops::abi::name(value.dtype())
            ),
        ));
    }

    // Dense planes of one type are a scatter of both planes in one launch, rounding into the
    // planes' type, and E4M3 planes are the quantizing write. Planes of two types, or of a type
    // that is neither, have no kernel that fills them and no reader that could interpret them.
    let dense = |dtype| matches!(dtype, DType::BF16 | DType::F16 | DType::F32);
    match (planes.key.dtype(), planes.value.dtype()) {
        (DType::F8E4M3, DType::F8E4M3) => return write_scaled(planes, slots, page, key, value),
        (keys, values) if keys == values && dense(keys) => {}
        (keys, values) => {
            return Err(Error::shape(
                OWNER,
                format!(
                    "the cache is {} keys and {} values over {} projections: a cache is bf16, \
                     f16 or f32 in both planes, or E4M3 in both",
                    ops::abi::name(keys),
                    ops::abi::name(values),
                    ops::abi::name(key.dtype())
                ),
            ));
        }
    }

    compute(KvWriteCall {
        planes,
        slots,
        page,
        heads: (heads, head_dim),
        projections: (key, value),
    })
}

/// A checked call of [`kv_write_`] into dense planes of one type: `heads` are the planes' key
/// heads and their width, and `projections` the keys and the values written.
struct KvWriteCall<'t> {
    planes: &'t KvPlanes,
    slots: &'t Tensor,
    page: u32,
    heads: (u32, u32),
    projections: (&'t Tensor, &'t Tensor),
}

impl KvWriteCall<'_> {
    const OWNER: &'static str = "kv_write_";
}

impl Fused for KvWriteCall<'_> {
    type Output = ();

    fn device(&self) -> &Device {
        self.projections.0.device()
    }

    fn cpu(self) -> Result<()> {
        let Self {
            planes,
            slots,
            page,
            projections: (key, value),
            ..
        } = self;

        for (plane, flags, rows) in [
            (&planes.key, &planes.key_initialized, key),
            (&planes.value, &planes.value_initialized, value),
        ] {
            // The kernel's store: each element rounded once, to nearest even, from its exact
            // value into the plane's type, which leaves a row of the plane's own type as it is.
            scatter_rows(plane, slots, &rows.to_dtype(plane.dtype())?)?;
            raise_flags(flags, &pages_of(slots, page)?)?;
        }
        Ok(())
    }

    fn launch(self) -> Result<()> {
        let Self {
            planes,
            slots,
            page,
            heads: (heads, head_dim),
            projections: (key, value),
        } = self;

        ops::launch::<KvWrite>(
            // The projection's type and the plane's are both key fields.
            &kv::Descriptor::new(key.dtype(), planes.key.dtype(), page, heads, head_dim),
            kv::Args {
                key: ops::operand(key)?,
                value: ops::operand(value)?,
                key_cache: ops::operand(&planes.key)?,
                value_cache: ops::operand(&planes.value)?,
                slots: ops::operand(slots)?,
                key_initialized: ops::operand(&planes.key_initialized)?,
                value_initialized: ops::operand(&planes.value_initialized)?,
            },
        )?;
        Ok(())
    }
}

/// The quantizing write into E4M3 planes: one launch per plane, each with the page scales its
/// reader multiplies back.
fn write_scaled(
    planes: &KvPlanes,
    slots: &Tensor,
    page: u32,
    key: &Tensor,
    value: &Tensor,
) -> Result<()> {
    let (tokens, heads, head_dim) = key.dims3()?;

    // One descriptor serves both planes: the projections are one element type, which
    // `kv_write_` has checked.
    let descriptor = kv::ScaledWriteDescriptor::new(
        key.dtype(),
        DType::F32,
        page,
        heads as u32,
        head_dim as u32,
    );

    for (name, plane, scale, flags, source) in [
        (
            "key",
            &planes.key,
            &planes.key_scale,
            &planes.key_initialized,
            key,
        ),
        (
            "value",
            &planes.value,
            &planes.value_scale,
            &planes.value_initialized,
            value,
        ),
    ] {
        // A reader of an E4M3 plane multiplies by a scale this write must have stored, in the
        // reader's own buffer.
        let scale = scale.as_ref().ok_or_else(|| {
            Error::shape(
                ScaledWriteCall::OWNER,
                format!(
                    "the {name} cache is {} and has no page scale",
                    ops::abi::name(plane.dtype())
                ),
            )
        })?;
        let rows = source.reshape((tokens, heads * head_dim))?;

        compute(ScaledWriteCall {
            plane,
            scale,
            flags,
            slots,
            rows: &rows,
            page,
            descriptor,
        })?;
    }
    Ok(())
}

/// A checked quantizing write of [`kv_write_`] into one E4M3 plane: `rows` are its projection's
/// `[tokens, width]` rows, `scale` and `flags` the plane's page scales and flags, and
/// `descriptor` the write's, which both planes share.
struct ScaledWriteCall<'t> {
    plane: &'t Tensor,
    scale: &'t Tensor,
    flags: &'t Tensor,
    slots: &'t Tensor,
    rows: &'t Tensor,
    page: u32,
    descriptor: kv::ScaledWriteDescriptor,
}

impl ScaledWriteCall<'_> {
    const OWNER: &'static str = "kv_write_";
}

impl Fused for ScaledWriteCall<'_> {
    type Output = ();

    fn device(&self) -> &Device {
        self.rows.device()
    }

    fn cpu(self) -> Result<()> {
        write_scaled_formula(
            self.plane, self.scale, self.flags, self.slots, self.rows, self.page,
        )
    }

    fn launch(self) -> Result<()> {
        ops::launch::<KvWriteScaled>(
            &self.descriptor,
            kv::ScaledWriteArgs {
                source: ops::operand(self.rows)?,
                slots: ops::operand(self.slots)?,
                scales: ops::operand(self.scale)?,
                cache: ops::operand(self.plane)?,
                initialized: ops::operand(self.flags)?,
            },
        )?;
        Ok(())
    }
}

/// The page of each slot of `slots`, `[tokens]` U32, with [`NO_SLOT`] kept as it is.
fn pages_of(slots: &Tensor, page: u32) -> Result<Tensor> {
    let stored = slots.broadcast_ne(&integer(NO_SLOT, slots)?)?;
    let pages = slots.broadcast_div(&integer(page, slots)?)?;

    Ok(stored.where_cond(&pages, slots)?)
}

/// `plane`, a contiguous CPU tensor of the cache, in `shape` over the same storage: what an
/// in-place write through the view changes is the plane's.
fn plane_view(plane: &Tensor, shape: impl Into<Shape>) -> Result<Tensor> {
    if !plane.is_contiguous() {
        return Err(Error::shape(
            "kv_write_",
            format!("a cache plane of {:?} is not contiguous", plane.dims()),
        ));
    }
    Ok(plane.reshape(shape.into())?)
}

/// Writes `rows`, `[tokens, kv_heads, head_dim]` or `[tokens, width]` in the plane's element
/// type, into the slots `slots` of the CPU plane `plane`, in place. An index of [`NO_SLOT`] is
/// one the scatter skips.
fn scatter_rows(plane: &Tensor, slots: &Tensor, rows: &Tensor) -> Result<()> {
    let tokens = slots.elem_count();
    let width = rows.elem_count() / tokens.max(1);
    let index = slots
        .unsqueeze(1)?
        .broadcast_as((tokens, width))?
        .contiguous()?;

    plane_view(plane, (plane.elem_count() / width.max(1), width))?.scatter_set(
        &index,
        &rows.reshape((tokens, width))?,
        0,
    )?;
    Ok(())
}

/// Raises the flag of each page `pages` names, `[entries]` U32 with [`NO_SLOT`] for none, in
/// the CPU flags `flags`, `[pages, 1, 1, 1]` U8, in place.
fn raise_flags(flags: &Tensor, pages: &Tensor) -> Result<()> {
    let raised = Tensor::ones(pages.elem_count(), DType::U8, &Device::Cpu)?;

    plane_view(flags, flags.elem_count())?.scatter_set(pages, &raised, 0)?;
    Ok(())
}

/// `x`, FP32, rounded to E4M3 after the clamp that keeps a value the format cannot hold from
/// becoming an infinity.
fn e4m3(x: &Tensor) -> Result<Tensor> {
    let limit = f64::from(E4M3_MAX);

    Ok(x.clamp(-limit, limit)?.to_dtype(DType::F8E4M3)?)
}

/// The quantizing write of [`kv_write_`] into one CPU plane: `rows` is `[tokens, width]`, the
/// plane `[pages, page, kv_heads, head_dim]` E4M3, and `scales` and `flags` its `[pages, 1, 1,
/// 1]` page scales and flags.
fn write_scaled_formula(
    plane: &Tensor,
    scales: &Tensor,
    flags: &Tensor,
    slots: &Tensor,
    rows: &Tensor,
    page: u32,
) -> Result<()> {
    let pages = plane.dims()[0];
    let width = rows.dims()[1];
    let page_width = page as usize * width;
    let rows = rows.to_dtype(ACCUMULATE)?;

    let page_ids = Tensor::arange(0_u32, pages as u32, &Device::Cpu)?;
    let none = Tensor::full(NO_SLOT, pages, &Device::Cpu)?;
    let stored = slots.broadcast_ne(&integer(NO_SLOT, slots)?)?;
    let token_pages = slots.broadcast_div(&integer(page, slots)?)?;

    // 1. Each page's proposal: the largest magnitude among the rows this write brings to it,
    //    floored, over the largest E4M3 magnitude.
    let brings = page_ids
        .unsqueeze(1)?
        .broadcast_eq(&token_pages.unsqueeze(0)?)?
        .broadcast_mul(&stored.unsqueeze(0)?)?
        .to_dtype(ACCUMULATE)?;
    let largest = rows.abs()?.max(1)?;
    let touched = brings.max(1)?.gt(0.0)?;
    let proposed = brings
        .broadcast_mul(&largest.unsqueeze(0)?)?
        .max(1)?
        .maximum(f64::from(SCALE_FLOOR))?
        .broadcast_div(&scalar(E4M3_MAX)?)?;

    // 2. Each page's scale: the proposal on a fresh page, and never below the stored one,
    //    because the rows already written were divided by it.
    let raised = flags.reshape(pages)?.ne(0.0)?;
    let old = scales.reshape(pages)?;
    let scale = raised.where_cond(&old.maximum(&proposed)?, &proposed)?;

    // 3. The rows an initialized page holds, decoded at its old scale and encoded at the one
    //    it grew to. Every other page keeps its bytes.
    let grew = ((&touched * &raised)? * scale.gt(&old)?)?;
    let per_page = (pages, 1, 1, 1);
    let rewritten = e4m3(
        &plane
            .to_dtype(ACCUMULATE)?
            .broadcast_mul(&old.reshape(per_page)?)?
            .broadcast_div(&scale.reshape(per_page)?)?,
    )?;
    let grown = grew
        .where_cond(&page_ids, &none)?
        .unsqueeze(1)?
        .broadcast_as((pages, page_width))?
        .contiguous()?;
    plane_view(plane, (pages, page_width))?.scatter_set(
        &grown,
        &rewritten.reshape((pages, page_width))?,
        0,
    )?;

    // 4. The rows this write brings, at their page's scale.
    let read_pages = stored.where_cond(&token_pages, &token_pages.zeros_like()?)?;
    let divisor = scale.index_select(&read_pages, 0)?.unsqueeze(1)?;
    scatter_rows(plane, slots, &e4m3(&rows.broadcast_div(&divisor)?)?)?;

    // 5. The scale, then the flag, of every page that received a row.
    let written = touched.where_cond(&page_ids, &none)?;
    plane_view(scales, pages)?.scatter_set(&written, &scale, 0)?;
    raise_flags(flags, &written)
}

/// Zeroes the arrival counters at the head of `workspace`, a cache's `[bytes]` U8 kernel
/// workspace, in place: once per forward of `rows` batch rows over attention layers of
/// `q_heads` query heads, before its first [`paged_attention`].
///
/// The decode kernel splits a row's keys over thread blocks and counts their arrivals at the
/// head of its workspace; it needs that count at zero on entry and leaves it at zero itself. A
/// workspace holds whatever an earlier forward of another shape left there, a prefill's scratch
/// among it, and a stale count reduces a row before its blocks are in, so a forward clears the
/// counters its decode reads ([`ops::attention::arrival_counter_bytes`]) once and every layer
/// of the forward then finds them as the layer before left them. The CPU attention keeps no
/// counters, so on the CPU there is nothing to clear.
///
/// # Errors
///
/// When the workspace is not a vector of bytes or is smaller than the counters, or the fill
/// cannot be issued.
pub fn zero_arrival_counters_(workspace: &Tensor, rows: usize, q_heads: u32) -> Result<()> {
    const OWNER: &str = ZeroArrivalCountersCall::OWNER;

    if workspace.rank() != 1 || workspace.dtype() != DType::U8 {
        return Err(Error::shape(
            OWNER,
            format!(
                "the workspace is {} {:?}, a vector of bytes expected",
                ops::abi::name(workspace.dtype()),
                workspace.dims()
            ),
        ));
    }
    compute(ZeroArrivalCountersCall {
        workspace,
        rows,
        q_heads,
    })
}

/// A checked call of [`zero_arrival_counters_`].
struct ZeroArrivalCountersCall<'t> {
    workspace: &'t Tensor,
    rows: usize,
    q_heads: u32,
}

impl ZeroArrivalCountersCall<'_> {
    const OWNER: &'static str = "zero_arrival_counters_";
}

impl Fused for ZeroArrivalCountersCall<'_> {
    type Output = ();

    fn device(&self) -> &Device {
        self.workspace.device()
    }

    /// The CPU attention keeps no arrival counters, so there is nothing to clear.
    fn cpu(self) -> Result<()> {
        Ok(())
    }

    fn launch(self) -> Result<()> {
        let Self {
            workspace,
            rows,
            q_heads,
        } = self;

        let counters =
            ops::attention::arrival_counter_bytes(workspace.device(), rows as u64, q_heads.into())?;
        let counters = usize::try_from(counters)
            .ok()
            .filter(|counters| *counters <= workspace.elem_count())
            .ok_or_else(|| {
                Error::shape(
                    Self::OWNER,
                    format!(
                        "a workspace of {} bytes holds no {counters} bytes of arrival counters \
                         for {rows} rows of {q_heads} query heads",
                        workspace.elem_count()
                    ),
                )
            })?;
        Ok(workspace.narrow(0, 0, counters)?.zero_set()?)
    }
}
