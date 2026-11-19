//! Attention over one shared latent that is both key and value, and the cache fields it reads:
//! the record write of a paged field, the compression of tokens into latents, and the attention
//! over a window and a selection of compressed entries.
//!
//! ```text
//! record_write_      plane[e] = encode(rotate(norm(x)))          e = entries[t] ≥ 0
//! latent_compress    latent   = rms_norm(Σ_m softmax_m(score) · kv)   for each completed group
//! latent_attention   o[t, h]  = Σ_n softmax(s)·c[n] / (Σ e^(s − m) + e^(sink[h] − m))
//!                                n over the window and the selection of row t
//! ```
//!
//! A forward borrows each field's plane as a [`Plane`]: the `[pages, entries, record]` records,
//! the `[rows, max_pages]` U32 page table and the records' format ([`ops::record`]). Entry `e`
//! of a row lies at slot `e mod entries` of page `table[row, (e / entries) mod max_pages]`, so a
//! window table narrower than the request is a ring of pages. The operators' documentation in
//! `ops` (`ops::record::RecordWrite`, `ops::compress`, `ops::attention::latent`) states each
//! formula; the CPU implementations here compute it in FP32, rounding where the kernels round.

use ops::attention::{
    Compressed, CompressedArgs, LatentAttention, LatentAttentionArgs, LatentAttentionDescriptor,
};
use ops::compress::{LatentCompress, LatentCompressArgs, LatentCompressDescriptor};
use ops::record::{Format, Layout, RecordWrite, RecordWriteArgs, RecordWriteDescriptor};
use tensor::{DType, Device, Tensor};

use super::{ACCUMULATE, Fused, compute};
use crate::capability::NO_SLOT;
use crate::error::{Error, Result};

/// The finite floor of an attention row's maximum score: a row that reads no entry attends its
/// sink alone and receives zeros, as the reference's kernel floors it.
const SCORE_FLOOR: f32 = -1e30;

/// One paged plane of a cache field, as a forward borrows it.
#[derive(Clone, Copy)]
pub struct Plane<'t> {
    /// `[pages, entries, record]`: the records, BF16 elements for [`Format::Bf16`] and bytes
    /// otherwise.
    pub records: &'t Tensor,
    /// `[rows, max_pages]` U32: each row's pages.
    pub table: &'t Tensor,
    /// How a record encodes its values.
    pub format: Format,
    /// Where a page keeps its records' scales.
    pub layout: Layout,
}

impl Plane<'_> {
    /// Records per page.
    fn entries(&self) -> usize {
        self.records.dims().get(1).copied().unwrap_or(0)
    }

    /// Checks the plane holds records of `width` values in its format and a table of whole
    /// rows: `[pages, entries, record]` and `[rows, max_pages]` U32.
    fn check(&self, owner: &'static str, name: &str, width: usize) -> Result<()> {
        let (dtype, last) = self.format.plane(width as u64);
        let records = self.records.dims();
        let table = self.table.dims();
        if records.len() != 3
            || records[0] == 0
            || records[1] == 0
            || records[2] as u64 != last
            || self.records.dtype() != dtype
            || table.len() != 2
            || table[1] == 0
            || self.table.dtype() != DType::U32
            || self.format.check_width(owner, width as u64).is_err()
        {
            return Err(Error::shape(
                owner,
                format!(
                    "the {name} plane is {} {:?} with a table of {} {:?}: [pages, entries, {last}] \
                     {} records of {width} values and a [rows, max_pages] u32 table expected",
                    ops::abi::name(self.records.dtype()),
                    records,
                    ops::abi::name(self.table.dtype()),
                    table,
                    self.format
                ),
            ));
        }
        Ok(())
    }

    /// The decoded records of a CPU plane, FP32 `[pages · entries, width]`, entry by entry.
    fn decoded(&self, width: usize) -> Result<Tensor> {
        let entries = self.entries();
        let pages = self.records.dims()[0];
        if self.format == Format::Bf16 {
            return Ok(self
                .records
                .reshape((pages * entries, width))?
                .to_dtype(ACCUMULATE)?);
        }

        let bytes = self.records.flatten_all()?.to_vec1::<u8>()?;
        let page_bytes = bytes.len() / pages;
        let (value_bytes, scale_bytes) = (
            self.format.value_bytes(width as u64) as usize,
            self.format.scale_bytes(width as u64) as usize,
        );
        let mut values = Vec::with_capacity(pages * entries * width);
        for page in 0..pages {
            let page = &bytes[page * page_bytes..(page + 1) * page_bytes];
            for slot in 0..entries {
                let (at, scales_at) =
                    self.layout
                        .offsets(self.format, entries as u64, width as u64, slot as u64);
                let (at, scales_at) = (at as usize, scales_at as usize);
                values.extend(ops::record::decode(
                    self.format,
                    width,
                    &page[at..at + value_bytes],
                    &page[scales_at..scales_at + scale_bytes],
                ));
            }
        }
        Ok(Tensor::from_vec(
            values,
            (pages * entries, width),
            &Device::Cpu,
        )?)
    }

    /// The record of entry `entry` of table row `row`: its index among the plane's records.
    fn slot(&self, table: &[Vec<u32>], row: usize, entry: usize) -> usize {
        let entries = self.entries();
        let columns = table[row].len();
        table[row][(entry / entries) % columns] as usize * entries + entry % entries
    }
}

/// The values of a CPU index column as `T`.
fn column<T: tensor::WithDType>(column: &Tensor) -> Result<Vec<T>> {
    Ok(column.flatten_all()?.to_vec1::<T>()?)
}

/// Checks `column`, the `[tokens]` column `name` of `dtype`.
fn check_column(
    owner: &'static str,
    name: &str,
    column: &Tensor,
    dtype: DType,
    tokens: usize,
) -> Result<()> {
    if column.dims() != [tokens] || column.dtype() != dtype {
        return Err(Error::shape(
            owner,
            format!(
                "the {name} are {} {:?}, [{tokens}] {} expected",
                ops::abi::name(column.dtype()),
                column.dims(),
                ops::abi::name(dtype)
            ),
        ));
    }
    Ok(())
}

/// `values` rounded into `dtype` and back to FP32, as a kernel stores an intermediate row.
fn rounded(values: &[f32], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::BF16 => values
            .iter()
            .map(|value| half::bf16::from_f32(*value).to_f32())
            .collect(),
        _ => values.to_vec(),
    }
}

/// `x · rsqrt(mean(x²) + eps)`, then times `gain`, each product in FP32, as the reference's
/// `RMSNorm` computes it.
fn rms_norm_row(x: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    let inverse = 1.0 / (mean + eps).sqrt();
    x.iter()
        .zip(gain)
        .map(|(value, gain)| gain * (value * inverse))
        .collect()
}

/// The rotation of `row`'s trailing `2 · inv_freq.len()` channels in adjacent pairs by the
/// angles of `position`: the pair `(a, b)` turns into `(a·cos − b·sin, b·cos + a·sin)`, the
/// angle `f32(position) · inv_freq[j]` in FP32, as the factors' operator evaluates it.
fn rotate_trailing(row: &mut [f32], position: f32, inv_freq: &[f32], inverse: bool) {
    let first = row.len() - 2 * inv_freq.len();
    for (pair, frequency) in inv_freq.iter().enumerate() {
        let angle = position * frequency;
        let (cos, sin) = (
            angle.cos(),
            if inverse { -angle.sin() } else { angle.sin() },
        );
        let (a, b) = (row[first + 2 * pair], row[first + 2 * pair + 1]);
        row[first + 2 * pair] = a * cos - b * sin;
        row[first + 2 * pair + 1] = b * cos + a * sin;
    }
}

// ---- record_write_ ------------------------------------------------------------------------

/// How [`record_write_`] prepares a row before it is encoded.
#[derive(Clone, Copy)]
pub struct RecordRow<'t> {
    /// The RMS normalization's gain `[width]` and epsilon, when the row is normalized first.
    pub norm: Option<(&'t Tensor, f32)>,
    /// `[pairs]` F32: the inverse frequencies the trailing `2 · pairs` channels rotate by, when
    /// the row is rotated.
    pub inv_freq: Option<&'t Tensor>,
    /// Tokens one entry stands for: entry `e` rotates by the angles of position `e · ratio`.
    pub ratio: u32,
}

/// Writes each token's row of `x`, `[tokens, width]` BF16 or F32, into its entry of `plane`:
/// normalized, its trailing channels rotated at its entry's position, rounded into `x`'s type
/// after each step and encoded in the plane's format ([`ops::record::RecordWrite`]). `entries`
/// (`[tokens]` I32) names each token's entry, a negative one writing nothing, and `rows`
/// (`[tokens]` U32) its row of the plane's table. The tokens of one call name distinct entries.
///
/// # Errors
///
/// Rows, columns or a plane of other shapes than these, or the operator's refusal.
pub fn record_write_(
    x: &Tensor,
    row: RecordRow<'_>,
    plane: Plane<'_>,
    entries: &Tensor,
    rows: &Tensor,
) -> Result<()> {
    const OWNER: &str = RecordWriteCall::OWNER;

    let &[tokens, width] = x.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("rows of {:?}, [tokens, width] expected", x.dims()),
        ));
    };
    if !matches!(x.dtype(), DType::BF16 | DType::F32) {
        return Err(Error::shape(
            OWNER,
            format!("{} rows: BF16 or F32 expected", ops::abi::name(x.dtype())),
        ));
    }
    plane.check(OWNER, "cache", width)?;
    check_column(OWNER, "entries", entries, DType::I32, tokens)?;
    check_column(OWNER, "rows", rows, DType::U32, tokens)?;
    if let Some((gain, _)) = row.norm
        && gain.dims() != [width]
    {
        return Err(Error::shape(
            OWNER,
            format!("a gain of {:?} for rows of {width}", gain.dims()),
        ));
    }
    let rotated = match row.inv_freq {
        Some(frequencies) => match *frequencies.dims() {
            [pairs] if 2 * pairs <= width && frequencies.dtype() == DType::F32 => 2 * pairs,
            _ => {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "inverse frequencies of {:?} for rows of {width}: [pairs] f32 expected",
                        frequencies.dims()
                    ),
                ));
            }
        },
        None => 0,
    };
    if row.ratio == 0 {
        return Err(Error::shape(OWNER, "entries of no token".to_owned()));
    }

    compute(RecordWriteCall {
        x,
        row,
        plane,
        entries,
        rows,
        width,
        rotated,
    })
}

/// A checked call of [`record_write_`].
struct RecordWriteCall<'t> {
    x: &'t Tensor,
    row: RecordRow<'t>,
    plane: Plane<'t>,
    entries: &'t Tensor,
    rows: &'t Tensor,
    width: usize,
    rotated: usize,
}

impl RecordWriteCall<'_> {
    const OWNER: &'static str = "record_write";
}

impl Fused for RecordWriteCall<'_> {
    type Output = ();

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<()> {
        let Self {
            x,
            row,
            plane,
            entries,
            rows,
            width,
            ..
        } = self;
        let dtype = x.dtype();
        let values = x.to_dtype(ACCUMULATE)?.to_vec2::<f32>()?;
        let gain = match row.norm {
            Some((gain, eps)) => Some((column::<f32>(&gain.to_dtype(ACCUMULATE)?)?, eps)),
            None => None,
        };
        let inv_freq = match row.inv_freq {
            Some(frequencies) => column::<f32>(frequencies)?,
            None => Vec::new(),
        };
        let table = plane.table.to_vec2::<u32>()?;
        let (entries, rows) = (column::<i32>(entries)?, column::<u32>(rows)?);

        let page_entries = plane.entries();
        let pages = plane.records.dims()[0];
        let flat = if plane.format.packed() {
            plane.records.flatten_all()?
        } else {
            plane.records.reshape((pages * page_entries, width))?
        };
        let page_bytes = plane.format.record_bytes(width as u64) as usize * page_entries;

        for (token, values) in values.into_iter().enumerate() {
            let Ok(entry) = usize::try_from(entries[token]) else {
                continue;
            };

            // The normalization and the rotation each round into the rows' type, as the
            // reference's modules return their rows in it.
            let mut prepared = match &gain {
                Some((gain, eps)) => rounded(&rms_norm_row(&values, gain, *eps), dtype),
                None => values,
            };
            if !inv_freq.is_empty() {
                let position = (entry as u64 * u64::from(row.ratio)) as f32;
                rotate_trailing(&mut prepared, position, &inv_freq, false);
                prepared = rounded(&prepared, dtype);
            }

            let slot = plane.slot(&table, rows[token] as usize, entry);
            if !plane.format.packed() {
                let record =
                    Tensor::from_vec(prepared, (1, width), &Device::Cpu)?.to_dtype(DType::BF16)?;
                flat.slice_set(&record, 0, slot)?;
                continue;
            }

            let (codes, scales) = ops::record::encode(plane.format, &prepared);
            let (page, within) = (slot / page_entries, slot % page_entries);
            let (at, scales_at) = plane.layout.offsets(
                plane.format,
                page_entries as u64,
                width as u64,
                within as u64,
            );
            let start = page * page_bytes;
            flat.slice_set(
                &Tensor::from_vec(codes.clone(), codes.len(), &Device::Cpu)?,
                0,
                start + at as usize,
            )?;
            flat.slice_set(
                &Tensor::from_vec(scales.clone(), scales.len(), &Device::Cpu)?,
                0,
                start + scales_at as usize,
            )?;
        }
        Ok(())
    }

    fn launch(self) -> Result<()> {
        let Self {
            x,
            row,
            plane,
            entries,
            rows,
            width,
            rotated,
        } = self;
        ops::launch::<RecordWrite>(
            &RecordWriteDescriptor {
                dtype: x.dtype(),
                width: width as u32,
                norm: row.norm.is_some(),
                rotated: rotated as u32,
                ratio: row.ratio,
                format: plane.format,
                layout: plane.layout,
                entries: plane.entries() as u32,
            },
            RecordWriteArgs {
                x: ops::operand(x)?,
                weight: row.norm.map(|(gain, _)| ops::operand(gain)).transpose()?,
                inv_freq: row.inv_freq.map(ops::operand).transpose()?,
                entries: ops::operand(entries)?,
                rows: ops::operand(rows)?,
                table: ops::operand(plane.table)?,
                plane: ops::operand(plane.records)?,
                eps: row.norm.map_or(0.0, |(_, eps)| eps),
            },
        )?;
        Ok(())
    }
}

// ---- latent_compress ----------------------------------------------------------------------

/// The state rows of a compression, as a recurrent state lends them: each request's ring and
/// the columns that place a batch's tokens in it.
#[derive(Clone, Copy)]
pub struct Ring<'t> {
    /// `[slots, ring, 2 · width]` F32: each request's last projections, position `q` in row
    /// `q mod ring`; read here, written by the caller from the answered ring rows.
    pub rows: &'t Tensor,
    /// `[state rows]` U32: each state row's request slot, [`NO_SLOT`] for a padding row.
    pub slots: &'t Tensor,
    /// `[state rows + 1]` U32: the packed offsets of each state row's tokens.
    pub query_offsets: &'t Tensor,
    /// `[state rows]` U32: the tokens each state row has seen, its new ones included.
    pub seq_lens: &'t Tensor,
}

/// What [`latent_compress`] answers: the latents of the groups the batch completes, and where
/// they and the projections go.
pub struct Compression {
    /// `[tokens, width]` BF16: row `t` the normalized latent of the group token `t` completes;
    /// the row of a token that completes none is unspecified.
    pub latent: Tensor,
    /// `[tokens]` I32: the entry token `t` completes, `p / ratio`, or `-1`.
    pub entries: Tensor,
    /// `[tokens]` I32: the ring row token `t`'s projection is kept in (`slot · ring + p mod
    /// ring`, for a state row's last `ring` tokens), or `-1`; the caller places the projection
    /// rows there once the batch has read the ring ([`super::place_rows_`]).
    pub ring_rows: Tensor,
}

/// The softmax-gated pooling of each group of `ratio` tokens into one RMS-normalized latent
/// ([`ops::compress`]): `projected` is `[tokens, 2 · width]` F32, each token's values then its
/// gates, `gain` the normalization's `[width]` gain and `positions` the tokens' `[tokens]` I32
/// positions. A group's earlier tokens are read from the batch or from the request's ring.
///
/// # Errors
///
/// Operands of other shapes than these, a ratio below two or a ring smaller than a group, or the
/// operator's refusal.
pub fn latent_compress(
    projected: &Tensor,
    (gain, eps): (&Tensor, f32),
    ring: Ring<'_>,
    positions: &Tensor,
    ratio: u32,
) -> Result<Compression> {
    const OWNER: &str = LatentCompressCall::OWNER;

    let &[tokens, doubled] = projected.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "a projection of {:?}, [tokens, 2 · width] expected",
                projected.dims()
            ),
        ));
    };
    let width = doubled / 2;
    let &[_, ring_rows, ring_width] = ring.rows.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "a ring of {:?}, [slots, ring, 2 · width] expected",
                ring.rows.dims()
            ),
        ));
    };
    if doubled == 0
        || doubled % 2 != 0
        || projected.dtype() != DType::F32
        || ring_width != doubled
        || ring.rows.dtype() != DType::F32
        || gain.dims() != [width]
        || ratio < 2
        || ring_rows < ratio as usize
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "a {} projection of {:?}, a {} ring of {:?} and a gain of {:?} at a ratio of \
                 {ratio}: F32 rows of values and gates, a ring of at least a group",
                ops::abi::name(projected.dtype()),
                projected.dims(),
                ops::abi::name(ring.rows.dtype()),
                ring.rows.dims(),
                gain.dims()
            ),
        ));
    }
    let state_rows = ring.slots.dims().first().copied().unwrap_or(0);
    check_column(OWNER, "slots", ring.slots, DType::U32, state_rows)?;
    check_column(
        OWNER,
        "query offsets",
        ring.query_offsets,
        DType::U32,
        state_rows + 1,
    )?;
    check_column(
        OWNER,
        "sequence lengths",
        ring.seq_lens,
        DType::U32,
        state_rows,
    )?;
    check_column(OWNER, "positions", positions, DType::I32, tokens)?;

    compute(LatentCompressCall {
        projected,
        gain,
        eps,
        ring,
        positions,
        ratio,
        tokens,
        width,
        ring_rows,
    })
}

/// A checked call of [`latent_compress`].
struct LatentCompressCall<'t> {
    projected: &'t Tensor,
    gain: &'t Tensor,
    eps: f32,
    ring: Ring<'t>,
    positions: &'t Tensor,
    ratio: u32,
    tokens: usize,
    width: usize,
    ring_rows: usize,
}

impl LatentCompressCall<'_> {
    const OWNER: &'static str = "latent_compress";
}

impl Fused for LatentCompressCall<'_> {
    type Output = Compression;

    fn device(&self) -> &Device {
        self.projected.device()
    }

    fn cpu(self) -> Result<Compression> {
        let Self {
            projected,
            gain,
            eps,
            ring,
            positions,
            ratio,
            tokens,
            width,
            ring_rows,
        } = self;
        let ratio = ratio as usize;
        let projected = projected.to_vec2::<f32>()?;
        let gain = column::<f32>(&gain.to_dtype(ACCUMULATE)?)?;
        let stored = ring.rows.flatten_all()?.to_vec1::<f32>()?;
        let slots = column::<u32>(ring.slots)?;
        let offsets = column::<u32>(ring.query_offsets)?;
        let seq_lens = column::<u32>(ring.seq_lens)?;
        let positions = column::<i32>(positions)?;

        let mut latent = vec![0.0_f32; tokens * width];
        let mut entries = vec![-1_i32; tokens];
        let mut kept = vec![-1_i32; tokens];
        for (state, window) in offsets.windows(2).enumerate() {
            let (start, end) = (window[0] as usize, window[1] as usize);
            if slots[state] == NO_SLOT || start >= end {
                continue;
            }
            let slot = slots[state] as usize;
            // The state row's tokens are consecutive positions ending at `seq_lens - 1`.
            let first = seq_lens[state] as usize - (end - start);

            for token in start..end {
                let position = positions[token] as usize;
                if seq_lens[state] as usize - position <= ring_rows {
                    kept[token] = (slot * ring_rows + position % ring_rows) as i32;
                }
                if !(position + 1).is_multiple_of(ratio) {
                    continue;
                }

                // The group's members, oldest first: from the batch when it holds them, else
                // from the request's ring.
                let members: Vec<&[f32]> = (position + 1 - ratio..=position)
                    .map(|member| {
                        if member >= first {
                            projected[token - (position - member)].as_slice()
                        } else {
                            let at = (slot * ring_rows + member % ring_rows) * 2 * width;
                            &stored[at..at + 2 * width]
                        }
                    })
                    .collect();
                let pooled: Vec<f32> = (0..width)
                    .map(|channel| {
                        let peak = members
                            .iter()
                            .map(|member| member[width + channel])
                            .fold(f32::NEG_INFINITY, f32::max);
                        let weights: Vec<f32> = members
                            .iter()
                            .map(|member| (member[width + channel] - peak).exp())
                            .collect();
                        let total: f32 = weights.iter().sum();
                        members
                            .iter()
                            .zip(&weights)
                            .map(|(member, weight)| member[channel] * (weight / total))
                            .sum()
                    })
                    .collect();

                // The pooled row rounds into BF16 before the normalization, as the reference
                // casts it back to the model's type.
                let normalized = rms_norm_row(&rounded(&pooled, DType::BF16), &gain, eps);
                latent[token * width..(token + 1) * width].copy_from_slice(&normalized);
                entries[token] = (position / ratio) as i32;
            }
        }

        Ok(Compression {
            latent: Tensor::from_vec(latent, (tokens, width), &Device::Cpu)?
                .to_dtype(DType::BF16)?,
            entries: Tensor::from_vec(entries, tokens, &Device::Cpu)?,
            ring_rows: Tensor::from_vec(kept, tokens, &Device::Cpu)?,
        })
    }

    fn launch(self) -> Result<Compression> {
        let Self {
            projected,
            gain,
            eps,
            ring,
            positions,
            ratio,
            tokens,
            width,
            ring_rows,
        } = self;
        let latent = ops::empty((tokens, width), DType::BF16)?;
        let entries = ops::empty(tokens, DType::I32)?;
        let kept = ops::empty(tokens, DType::I32)?;
        ops::launch::<LatentCompress>(
            &LatentCompressDescriptor {
                ratio,
                width: width as u32,
                ring: ring_rows as u32,
                gain: gain.dtype(),
            },
            LatentCompressArgs {
                projected: ops::operand(projected)?,
                weight: ops::operand(gain)?,
                ring: ops::operand(ring.rows)?,
                slots: ops::operand(ring.slots)?,
                query_offsets: ops::operand(ring.query_offsets)?,
                seq_lens: ops::operand(ring.seq_lens)?,
                positions: ops::operand(positions)?,
                latent: ops::operand(&latent)?,
                entries: ops::operand(&entries)?,
                ring_rows: ops::operand(&kept)?,
                eps,
            },
        )?;
        Ok(Compression {
            latent,
            entries,
            ring_rows: kept,
        })
    }
}

// ---- latent_attention ---------------------------------------------------------------------

/// The window side of a [`latent_attention`]: the plane and the entries a query reads of it.
#[derive(Clone, Copy)]
pub struct Window<'t> {
    /// The window entries' plane, one entry per token.
    pub plane: Plane<'t>,
    /// Entries a query attends: its own position and the `size − 1` before it.
    pub size: u32,
}

/// The compressed side of a [`latent_attention`]: the plane and each query's selection of it.
#[derive(Clone, Copy)]
pub struct Selected<'t> {
    /// The compressed entries' plane.
    pub plane: Plane<'t>,
    /// `[tokens, columns]` I32: the compressed entries each query attends, in its request's
    /// order; a negative one names none.
    pub selection: &'t Tensor,
}

/// Attention of each rotated query over its window and its selected compressed entries, one
/// latent as key and value, with a per-head sink ([`ops::attention::latent`]): `query` is
/// `[tokens, heads, head_dim]` BF16, `rows` (`[tokens]` U32) each token's row of the tables,
/// `positions` (`[tokens]` I32) its position and `sink` the `[heads]` F32 sink logits. Answers
/// `[tokens, heads, head_dim]` BF16.
///
/// # Errors
///
/// Operands of other shapes than these, planes of another width than the heads, or the
/// operator's refusal.
pub fn latent_attention(
    query: &Tensor,
    window: Window<'_>,
    compressed: Option<Selected<'_>>,
    (rows, positions): (&Tensor, &Tensor),
    sink: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    const OWNER: &str = LatentAttentionCall::OWNER;

    let &[tokens, heads, head_dim] = query.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "queries of {:?}, [tokens, heads, head_dim] expected",
                query.dims()
            ),
        ));
    };
    if query.dtype() != DType::BF16 || sink.dims() != [heads] || sink.dtype() != DType::F32 {
        return Err(Error::shape(
            OWNER,
            format!(
                "{} queries of {:?} and a {} sink of {:?}: BF16 queries and one F32 sink per head",
                ops::abi::name(query.dtype()),
                query.dims(),
                ops::abi::name(sink.dtype()),
                sink.dims()
            ),
        ));
    }
    window.plane.check(OWNER, "window", head_dim)?;
    if window.size == 0 {
        return Err(Error::shape(OWNER, "a window of no entry".to_owned()));
    }
    if let Some(side) = compressed {
        side.plane.check(OWNER, "compressed", head_dim)?;
        let selected = side.selection.dims();
        if selected.len() != 2
            || selected[0] != tokens
            || selected[1] == 0
            || side.selection.dtype() != DType::I32
            || side.plane.table.dims()[0] != window.plane.table.dims()[0]
        {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a {} selection of {:?} for {tokens} tokens beside tables of {:?} and {:?}",
                    ops::abi::name(side.selection.dtype()),
                    selected,
                    window.plane.table.dims(),
                    side.plane.table.dims()
                ),
            ));
        }
    }
    check_column(OWNER, "rows", rows, DType::U32, tokens)?;
    check_column(OWNER, "positions", positions, DType::I32, tokens)?;

    compute(LatentAttentionCall {
        query,
        window,
        compressed,
        rows,
        positions,
        sink,
        scale,
        tokens,
        heads,
        head_dim,
    })
}

/// A checked call of [`latent_attention`].
struct LatentAttentionCall<'t> {
    query: &'t Tensor,
    window: Window<'t>,
    compressed: Option<Selected<'t>>,
    rows: &'t Tensor,
    positions: &'t Tensor,
    sink: &'t Tensor,
    scale: f32,
    tokens: usize,
    heads: usize,
    head_dim: usize,
}

impl LatentAttentionCall<'_> {
    const OWNER: &'static str = "latent_attention";
}

impl Fused for LatentAttentionCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.query.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            query,
            window,
            compressed,
            rows,
            positions,
            sink,
            scale,
            tokens,
            heads,
            head_dim,
        } = self;
        let window_latents = window.plane.decoded(head_dim)?;
        let window_table = window.plane.table.to_vec2::<u32>()?;
        let compressed = match compressed {
            Some(side) => Some((
                side.plane,
                side.plane.decoded(head_dim)?,
                side.plane.table.to_vec2::<u32>()?,
                side.selection.to_vec2::<i32>()?,
            )),
            None => None,
        };
        let (rows, positions) = (column::<u32>(rows)?, column::<i32>(positions)?);
        let sink = column::<f32>(sink)?;
        let queries = query.to_dtype(ACCUMULATE)?;

        let mut out = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let row = rows[token] as usize;

            // The window's entries, then the selection's, as latents of their planes; a
            // padding token's negative position names no window entry.
            let mut latents = Vec::with_capacity(2);
            if let Ok(position) = usize::try_from(positions[token]) {
                let oldest = (position + 1).saturating_sub(window.size as usize);
                let slots: Vec<u32> = (oldest..=position)
                    .map(|entry| window.plane.slot(&window_table, row, entry) as u32)
                    .collect();
                latents.push(window_latents.index_select(
                    &Tensor::from_vec(slots.clone(), slots.len(), &Device::Cpu)?,
                    0,
                )?);
            }
            if let Some((plane, decoded, table, selection)) = &compressed {
                let slots: Vec<u32> = selection[token]
                    .iter()
                    .filter_map(|entry| usize::try_from(*entry).ok())
                    .map(|entry| plane.slot(table, row, entry) as u32)
                    .collect();
                if !slots.is_empty() {
                    latents.push(decoded.index_select(
                        &Tensor::from_vec(slots.clone(), slots.len(), &Device::Cpu)?,
                        0,
                    )?);
                }
            }
            if latents.is_empty() {
                // A query that reads no entry attends its sink alone: zeros.
                out.push(Tensor::zeros((heads, head_dim), ACCUMULATE, &Device::Cpu)?);
                continue;
            }
            let latents = Tensor::cat(&latents, 0)?;

            // FP32 scores and softmax over the entries; the sink enters the denominator alone.
            let scores = queries
                .get(token)?
                .matmul(&latents.t()?)?
                .affine(f64::from(scale), 0.0)?;
            let peak = scores
                .max_keepdim(1)?
                .clamp(f64::from(SCORE_FLOOR), f64::INFINITY)?;
            let weights = scores.broadcast_sub(&peak)?.exp()?;
            let sinks = Tensor::from_vec(sink.clone(), (heads, 1), &Device::Cpu)?
                .sub(&peak)?
                .exp()?;
            let denominator = weights.sum_keepdim(1)?.add(&sinks)?;

            // The probabilities round into BF16 before they multiply the latents, as the
            // reference's tensor-core product reads them.
            let rounded = weights.to_dtype(DType::BF16)?.to_dtype(ACCUMULATE)?;
            out.push(rounded.matmul(&latents)?.broadcast_div(&denominator)?);
        }
        Ok(Tensor::stack(&out, 0)?.to_dtype(DType::BF16)?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            query,
            window,
            compressed,
            rows,
            positions,
            sink,
            scale,
            tokens,
            heads,
            head_dim,
        } = self;
        let out = ops::empty((tokens, heads, head_dim), DType::BF16)?;
        let descriptor = LatentAttentionDescriptor::new(
            DType::BF16,
            heads as u32,
            head_dim as u32,
            (
                window.plane.format,
                window.size,
                window.plane.entries() as u32,
            ),
            compressed.map(|side| Compressed {
                format: side.plane.format,
                entries: side.plane.entries() as u32,
                columns: side.selection.dims()[1] as u32,
            }),
        );
        let compressed = match compressed {
            Some(side) => Some(CompressedArgs {
                plane: ops::operand(side.plane.records)?,
                table: ops::operand(side.plane.table)?,
                selection: ops::operand(side.selection)?,
            }),
            None => None,
        };
        ops::launch::<LatentAttention>(
            &descriptor,
            LatentAttentionArgs {
                query: ops::operand(query)?,
                window: ops::operand(window.plane.records)?,
                window_table: ops::operand(window.plane.table)?,
                compressed,
                rows: ops::operand(rows)?,
                positions: ops::operand(positions)?,
                sink: ops::operand(sink)?,
                out: ops::operand(&out)?,
                scale,
            },
        )?;
        Ok(out)
    }
}
