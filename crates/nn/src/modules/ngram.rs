//! A hashed n-gram memory: rows of one large table, named by hashes of the n-grams that end at a
//! token, gated into the streams of a hyper-connected residual (the conditional memory of
//! Engram, Cheng et al., 2026).
//!
//! ```text
//! rows      = ngram_hash(context)                   [tokens, (N − 1)·H]: one row per head of
//!                                                    every n-gram length 2..=N
//! recalled  = table[rows]                           [tokens, (N − 1)·H·r], MXFP8 rows dequantized
//! projected = recalled · Wᵀ                         [tokens, (n + 1)·d]: a key per stream, a value
//! h'        = ngram_gate(h, projected, q, κ)        each stream plus the value under its gate
//! ```
//!
//! The hashing ([`NgramHashing`]) is the model's statement: its normalization of the vocabulary,
//! its multipliers and its prime moduli and offsets. The table ([`NgramTable`]) is one weight of
//! hundreds of millions of rows, which a rank may hold a shard of ([`TableShard`]): it reads the
//! rows it holds and the forward's group completes the rest, by a sum of every rank's rows when
//! the table is split by rows, or by a gather of every rank's columns when it is split by hash
//! heads, whose rows are contiguous ranges of it. Where the rows live (device memory or memory
//! of the host the device reads) is the loader's placement: the lookup reads the shard's rows
//! wherever its tensor is.

use tensor::{DType, Tensor};

use crate::collective::{self, Split};
use crate::error::{Error, Result};
use crate::functional;
use crate::modules::Linear;

/// What a checkpoint states of a hashed n-gram memory.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct NgramConfig {
    /// The longest n-gram, `N`: a token's context is the `N − 1` tokens before it and its own.
    pub order: u32,
    /// Heads of each n-gram length, `H`.
    pub heads: u32,
    /// Elements of one table row, `r`.
    pub row: u32,
    /// Streams of the residual the memory adds into, `n`.
    pub streams: u32,
    /// Elements of one stream, `d`.
    pub hidden: u32,
    /// The epsilon of the gate's two mean squares.
    pub eps: f32,
}

impl NgramConfig {
    /// The table rows one token reads: `(N − 1)·H`.
    pub const fn columns(&self) -> u32 {
        (self.order - 1) * self.heads
    }
}

/// How one model hashes a layer's n-grams into table rows: the normalization of the vocabulary,
/// the multipliers and the prime ranges of the heads (`functional::ngram_hash`).
pub struct NgramHashing {
    /// `[vocabulary]` I32: each vocabulary id's normalized id.
    normalize: Tensor,
    /// `[N]` I64: the multiplier of the token `i` back, the token's own first.
    multipliers: Tensor,
    /// `[N − 1, H]` I64: each head's prime modulus.
    moduli: Tensor,
    /// `[N − 1, H]` I64: each head's first row of the table.
    offsets: Tensor,
}

impl NgramHashing {
    /// The hashing over the tensors given: `normalize` `[vocabulary]` I32, `multipliers` `[N]`
    /// I64, and `moduli` and `offsets` `[N − 1, H]` I64.
    ///
    /// # Errors
    ///
    /// When a tensor has other extents or types.
    pub fn new(
        normalize: Tensor,
        multipliers: Tensor,
        moduli: Tensor,
        offsets: Tensor,
    ) -> Result<Self> {
        let order = multipliers.dims().first().copied().unwrap_or(0);
        let heads = moduli.dims().get(1).copied().unwrap_or(0);
        if order < 2 || heads == 0 || normalize.rank() != 1 || normalize.dtype() != DType::I32 {
            return Err(Error::shape(
                "NgramHashing",
                format!(
                    "multipliers of {:?} and moduli of {:?} over a normalization of {} {:?}",
                    multipliers.dims(),
                    moduli.dims(),
                    ops::abi::name(normalize.dtype()),
                    normalize.dims()
                ),
            ));
        }
        for (name, tensor, dims) in [
            ("multipliers", &multipliers, [order].as_slice()),
            ("moduli", &moduli, &[order - 1, heads]),
            ("offsets", &offsets, &[order - 1, heads]),
        ] {
            if tensor.dims() != dims || tensor.dtype() != DType::I64 {
                return Err(Error::shape(
                    "NgramHashing",
                    format!(
                        "the {name} are {} {:?}, I64 {dims:?} expected",
                        ops::abi::name(tensor.dtype()),
                        tensor.dims()
                    ),
                ));
            }
        }

        Ok(Self {
            normalize,
            multipliers,
            moduli,
            offsets,
        })
    }

    /// The longest n-gram, `N`, and the heads of each length, `H`.
    pub fn extents(&self) -> (u32, u32) {
        (
            self.multipliers.elem_count() as u32,
            self.moduli.dims()[1] as u32,
        )
    }

    /// The table rows of the n-grams ending at each token of `context`, `[tokens, N]` vocabulary
    /// ids (the `N − 1` before each token, oldest first, then its own): `[tokens, (N − 1)·H]` I64.
    ///
    /// # Errors
    ///
    /// A context of another shape, or the operator's refusal.
    pub fn rows(&self, context: &Tensor) -> Result<Tensor> {
        functional::ngram_hash(
            context,
            &self.normalize,
            &self.multipliers,
            &self.moduli,
            &self.offsets,
        )
    }
}

/// The rows of a table a rank holds, and how the forward's group completes the others.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TableShard {
    /// The rank holds the whole table.
    Whole,
    /// The rank holds a contiguous range of rows; every rank reads the rows it holds, zeros for
    /// the others, and the group sums them.
    Rows(Split),
    /// The rank holds the rows of the hash heads `split` assigns it: its share of each token's
    /// row columns (`columns / size` of them, in order), whose ranges of the table are contiguous.
    /// It reads only those, and the group gathers every rank's columns.
    Heads(Split),
}

/// An MXFP8 table of n-gram rows, or this rank's shard of it.
pub struct NgramTable {
    /// `[rows, r]` E4M3: the rows held.
    values: Tensor,
    /// `[rows, r / 32]` U8: their UE8M0 scales.
    scales: Tensor,
    /// The table row of the first row held.
    first: u64,
    shard: TableShard,
}

impl NgramTable {
    /// The table over the rows held: `values` `[rows, r]` E4M3, `scales` `[rows, r / 32]` U8
    /// exponent bytes, the first of them row `first` of the whole table, held as `shard` states.
    /// For a shard of hash heads the rows are exactly the ranges of the heads it assigns.
    ///
    /// # Errors
    ///
    /// When the tensors' extents or types disagree.
    pub fn new(values: Tensor, scales: Tensor, first: u64, shard: TableShard) -> Result<Self> {
        let groups =
            values.dims().get(1).copied().unwrap_or(0) / ops::embedding::MXFP8_GROUP as usize;
        if values.rank() != 2
            || values.dtype() != DType::F8E4M3
            || scales.dtype() != DType::U8
            || scales.dims() != [values.dims()[0], groups]
        {
            return Err(Error::shape(
                "NgramTable",
                format!(
                    "values {} {:?} and scales {} {:?}: E4M3 rows and a U8 scale per 32",
                    ops::abi::name(values.dtype()),
                    values.dims(),
                    ops::abi::name(scales.dtype()),
                    scales.dims()
                ),
            ));
        }
        if shard == TableShard::Whole && first != 0 {
            return Err(Error::shape(
                "NgramTable",
                format!("a whole table opens at row 0, not {first}"),
            ));
        }

        Ok(Self {
            values,
            scales,
            first,
            shard,
        })
    }

    /// The rows `rows` names, `[tokens, columns]` I64, dequantized into `dtype`:
    /// `[tokens, columns · r]`, every token's rows side by side.
    ///
    /// # Errors
    ///
    /// Rows of another shape, a shard of heads that does not divide the columns, a split table in
    /// a forward without its group, or what the lookup refuses.
    pub fn lookup(&self, rows: &Tensor, dtype: DType) -> Result<Tensor> {
        let (tokens, columns) = rows.dims2()?;
        let width = self.values.dim(1)?;
        let read = |ids: &Tensor| -> Result<Tensor> {
            functional::embedding_mxfp8(ids, &self.values, &self.scales, self.first, dtype)
        };

        match self.shard {
            TableShard::Whole => {
                Ok(read(&rows.flatten_all()?)?.reshape((tokens, columns * width))?)
            }
            TableShard::Rows(_) => {
                let recalled = read(&rows.flatten_all()?)?.reshape((tokens, columns * width))?;
                grouped("a table split by rows sums every rank's rows over its group")?;
                collective::all_reduce(&recalled)?;
                Ok(recalled)
            }
            TableShard::Heads(split) => {
                let size = split.size() as usize;
                if !columns.is_multiple_of(size) {
                    return Err(Error::shape(
                        "NgramTable",
                        format!("{columns} row columns are not whole shares of {size} ranks"),
                    ));
                }
                // This rank's columns, column-major, so the gather of every rank's block in rank
                // order is every column in order.
                let share = columns / size;
                let held = rows
                    .narrow(1, split.rank() as usize * share, share)?
                    .t()?
                    .contiguous()?;
                let block = read(&held.flatten_all()?)?.reshape((share, tokens, width))?;
                grouped("a table split by heads gathers every rank's columns over its group")?;
                let gathered = collective::gathered(&block)?;
                Ok(gathered
                    .transpose(0, 1)?
                    .contiguous()?
                    .reshape((tokens, columns * width))?)
            }
        }
    }
}

/// Refuses a split table's lookup in a forward that has no group to complete it over.
fn grouped(what: &str) -> Result<()> {
    if collective::split().is_none() {
        return Err(Error::Unsupported(what.to_owned()));
    }
    Ok(())
}

/// A hashed n-gram memory of one layer (module documentation).
pub struct NgramMemory {
    hashing: NgramHashing,
    table: NgramTable,
    /// The projection of a token's recalled rows onto a key per stream and a value:
    /// `(N − 1)·H·r → (n + 1)·d`.
    projection: Linear,
    /// `[n, d]`: the gate's query weight.
    query: Tensor,
    /// `[n, d]`: the gate's key weight.
    key: Tensor,
    config: NgramConfig,
}

impl NgramMemory {
    /// The memory over its parts: the hashing, the table (or this rank's shard), the projection
    /// of the recalled rows and the gate's `[n, d]` query and key weights, in the streams' type.
    ///
    /// # Errors
    ///
    /// When the gate's weights have other extents.
    pub fn new(
        config: &NgramConfig,
        hashing: NgramHashing,
        table: NgramTable,
        projection: Linear,
        query: Tensor,
        key: Tensor,
    ) -> Result<Self> {
        if hashing.extents() != (config.order, config.heads) {
            return Err(Error::shape(
                "NgramMemory",
                format!(
                    "a hashing of (order, heads) {:?} for a memory of ({}, {})",
                    hashing.extents(),
                    config.order,
                    config.heads
                ),
            ));
        }
        let dims = [config.streams as usize, config.hidden as usize];
        if query.dims() != dims || key.dims() != dims {
            return Err(Error::shape(
                "NgramMemory",
                format!(
                    "gate weights of {:?} and {:?}, {dims:?} expected",
                    query.dims(),
                    key.dims()
                ),
            ));
        }

        Ok(Self {
            hashing,
            table,
            projection,
            query,
            key,
            config: *config,
        })
    }

    /// The table rows of each token's n-grams ([`NgramHashing::rows`]).
    ///
    /// # Errors
    ///
    /// As [`NgramHashing::rows`].
    pub fn rows(&self, context: &Tensor) -> Result<Tensor> {
        self.hashing.rows(context)
    }

    /// The rows `rows` names, dequantized into `dtype` ([`NgramTable::lookup`]).
    ///
    /// # Errors
    ///
    /// As [`NgramTable::lookup`].
    pub fn lookup(&self, rows: &Tensor, dtype: DType) -> Result<Tensor> {
        self.table.lookup(rows, dtype)
    }

    /// `streams`, `[tokens, n, d]`, each plus the value `recalled` projects to, under the gate of
    /// its key's agreement with the stream: `[tokens, n, d]` in the streams' type.
    ///
    /// # Errors
    ///
    /// Recalled rows of another width, or what the projection or the gate refuses.
    pub fn gate(&self, streams: &Tensor, recalled: &Tensor) -> Result<Tensor> {
        let projected = self.projection.forward(recalled)?;
        functional::ngram_gate(streams, &projected, &self.query, &self.key, self.config.eps)
    }

    /// The memory added into `streams`, `[tokens, n, d]`, for the tokens of `context`
    /// (`[tokens, N]` vocabulary ids): [`Self::rows`], [`Self::lookup`] and [`Self::gate`].
    ///
    /// # Errors
    ///
    /// What one of them refuses.
    pub fn forward(&self, streams: &Tensor, context: &Tensor) -> Result<Tensor> {
        let recalled = self.lookup(&self.rows(context)?, streams.dtype())?;
        self.gate(streams, &recalled)
    }
}

impl std::fmt::Debug for NgramMemory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NgramMemory")
            .field("config", &self.config)
            .field("shard", &self.table.shard)
            .finish_non_exhaustive()
    }
}
