//! Attention over the tokens an indexer selects: the index scores, the selection of the best of
//! them, and the attention itself.
//!
//! A sparse attention layer keeps, beside its cache, one compressed **index key** per `ratio`
//! consecutive tokens. A query row scores the index keys of its request with a few index heads,
//! keeps the best groups, and attends only the tokens of those groups:
//!
//! ```text
//! index_scores        s[t, g] = Σ_h relu(q[t, h] · key[g]) · w[t, h]   g < visible[t], −inf otherwise
//! index_select        sel[t]  = the k columns of s[t] of the largest scores, −1 for a −inf one
//! indexed_attention   o[t, i] = Σ_n softmax_n(q[t, i] · k[n, i / group] · scale) · v[n, i / group]
//!                                                                  n ∈ selection[t], n ≥ 0
//! ```
//!
//! The weights `w` are optional (one for every head without them). The scores may be restricted
//! to **candidate blocks**: the row then names `[blocks]` blocks of `B` consecutive groups,
//! column `c` scores group `candidates[t, c / B] · B + c mod B`, and the selection maps a kept
//! column back to that group. The selection's **block mode** ranks blocks instead: a block of `B`
//! columns scores its largest column, the block holding the row's newest visible group scores
//! `+∞`, and the kept blocks are the candidates of a later scoring.
//!
//! All read a request's pages through a page table: `rows[t]` names the table row of token `t`,
//! group `g` of a request is row `g mod groups` of index-key page `table[rows[t], g / groups]`,
//! and token `n` is row `n mod page` of key and value page `table[rows[t], n / page]`. None
//! checks a table entry: the invariant is the table's producer's, as it is for paged attention.
//!
//! Index keys are BF16 rows or packed records ([`crate::record`]), a packed page keeping all its
//! records' values and then all their scales ([`crate::record::Layout::Page`], DeepGEMM's paged
//! layout). The index queries may be rounded to MXFP4 as they are read, as the reference rounds
//! them before it scores (`fp4_act_quant` of DeepSeek's `Indexer`). The index keys are selected
//! between, so the scores are ranked and never rounded: FP32. The attention computes its scores, its softmax and
//! its sum in FP32; with caches of a 16-bit type each probability is rounded into that type
//! before it multiplies its value, as the reference casts them (`qsa.py:141-253`), and the
//! output rounds once.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::{Float, Key};
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::record::Format;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The largest magnitude a cached value is declared to have, for the indexed attention's absolute
/// bound: a declaration about the cache's data, as `tolerance::VALUE_ENVELOPE` is about the
/// activations.
const CACHED_VALUE_ENVELOPE: f32 = 64.0;

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
    /// The index keys' records; a packed page keeps its scales after every record's values
    /// ([`crate::record::Layout::Page`]).
    pub keys: Format,
    /// Index heads of one query row.
    pub heads: u32,
    /// Elements of one index head and of one index key.
    pub width: u32,
    /// Index keys of one page: the page's tokens over the tokens one key stands for.
    pub groups: u32,
    /// Index keys one program scores.
    pub block: u32,
    /// Whether each head's rectified product is weighted by the row's weight of that head.
    pub weighted: bool,
    /// How a query is read: as it is ([`Format::Bf16`]), or rounded to [`Format::Mxfp4`] blocks.
    pub query: Format,
    /// Groups of one candidate block, when the scores are restricted to candidate blocks; zero
    /// scores every group of the request.
    pub candidates: u32,
}

/// The index keys one `index_scores` program scores.
const SCORES_BLOCK: u32 = 256;

impl IndexScoresDescriptor {
    /// The compilation that scores `dtype` queries of `heads` heads of `width` against index
    /// keys of `keys` in groups of `groups` per page: every visible group, unweighted, each
    /// query read as it is.
    pub fn new(dtype: DType, keys: Format, heads: u32, width: u32, groups: u32) -> Self {
        Self {
            dtype,
            keys,
            heads,
            width,
            groups,
            block: SCORES_BLOCK,
            weighted: false,
            query: Format::Bf16,
            candidates: 0,
        }
    }

    /// The columns of a score row over `max_pages` pages, or over `blocks` candidate blocks.
    pub fn columns(&self, max_pages: u64, blocks: u64) -> u64 {
        if self.candidates == 0 {
            max_pages * u64::from(self.groups)
        } else {
            blocks * u64::from(self.candidates)
        }
    }
}

impl fmt::Display for IndexScoresDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("keys", self.keys)
            .field("heads", self.heads)
            .field("width", self.width)
            .field("groups", self.groups)
            .field("block", self.block);
        // The weighting, the query's rounding and the candidates are written after `block`, and
        // only when they are not the unweighted dense scoring every earlier key states.
        let key = if self.weighted {
            key.field("weighted", true)
        } else {
            key
        };
        let key = if self.query == Format::Bf16 {
            key
        } else {
            key.field("query", self.query)
        };
        let key = if self.candidates == 0 {
            key
        } else {
            key.field("candidates", self.candidates)
        };
        key.finish(formatter)
    }
}

/// The arguments of `index_scores`.
#[derive(Clone)]
pub struct IndexScoresArgs {
    /// `[tokens, heads, width]`, contiguous: the index queries.
    pub query: Operand,
    /// `[tokens, heads]` F32, contiguous: each row's head weights, exactly when the descriptor
    /// weights.
    pub weights: Option<Operand>,
    /// `[pages, groups, record]`, contiguous: the index keys of one layer.
    pub keys: Operand,
    /// `[rows, max_pages]` u32: each row's pages.
    pub page_table: Operand,
    /// `[tokens]` u32: each token's row of the page table.
    pub rows: Operand,
    /// `[tokens]` u32: the groups each token scores, the leading ones of its request.
    pub visible: Operand,
    /// `[tokens, blocks]` I32, contiguous: each row's candidate blocks, a negative one naming
    /// none, exactly when the descriptor restricts to candidates.
    pub candidates: Option<Operand>,
    /// `[tokens, columns]` FP32, contiguous: receives the scores, `−inf` for a column of no
    /// visible group ([`IndexScoresDescriptor::columns`]).
    pub out: Operand,
}

/// The scores of the index keys a query row may select among.
///
/// One program scores [`IndexScoresDescriptor::block`] keys of one token: the products, the
/// weighting and the sum over the heads are FP32.
pub struct IndexScores;

impl Op for IndexScores {
    const NAME: &'static str = proto::module::operator::INDEX_SCORES;

    type Descriptor = IndexScoresDescriptor;
    type Args<'a> = IndexScoresArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let d = descriptor;
        if d.heads == 0
            || d.width == 0
            || d.groups == 0
            || d.block == 0
            || !matches!(d.query, Format::Bf16 | Format::Mxfp4)
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} index heads of {} elements over pages of {} keys, in blocks of {}, \
                     queries read as {}",
                    d.heads, d.width, d.groups, d.block, d.query
                ),
            ));
        }
        let (heads, width, groups) = (u64::from(d.heads), u64::from(d.width), u64::from(d.groups));
        d.keys.check_width(Self::NAME, width)?;
        d.query.check_width(Self::NAME, width)?;

        args.query.require_rank(Self::NAME, 3)?;
        let tokens = args.query.dim(0);
        args.query
            .require_shape(Self::NAME, &[tokens, heads, width])?;
        args.query.require_row(Self::NAME, d.dtype, width)?;

        d.keys
            .check_plane(Self::NAME, "index key", &args.keys, groups, width)?;

        let max_pages = check_table(Self::NAME, &args.page_table, &args.rows, tokens)?;
        args.visible
            .require_shape(Self::NAME, &[tokens])
            .and_then(|()| args.visible.require_row(Self::NAME, DType::U32, tokens))
            .map_err(|error| Error::argument(Self::NAME, format!("visible: {error}")))?;

        match (&args.weights, d.weighted) {
            (Some(weights), true) => {
                weights
                    .require_shape(Self::NAME, &[tokens, heads])
                    .and_then(|()| weights.require_row(Self::NAME, DType::F32, heads))
                    .map_err(|error| Error::argument(Self::NAME, format!("weights: {error}")))?;
                if !weights.is_row_major() {
                    return Err(Error::argument(
                        Self::NAME,
                        "the weights are not contiguous".to_owned(),
                    ));
                }
            }
            (None, false) => {}
            _ => {
                return Err(Error::argument(
                    Self::NAME,
                    "weights are passed exactly when the compilation weights".to_owned(),
                ));
            }
        }
        let blocks = match (&args.candidates, d.candidates) {
            (Some(candidates), block) if block > 0 => {
                candidates.require_rank(Self::NAME, 2)?;
                let blocks = candidates.dim(1);
                candidates
                    .require_shape(Self::NAME, &[tokens, blocks])
                    .and_then(|()| candidates.require_row(Self::NAME, DType::I32, blocks))
                    .map_err(|error| Error::argument(Self::NAME, format!("candidates: {error}")))?;
                if !candidates.is_row_major() || blocks == 0 {
                    return Err(Error::argument(
                        Self::NAME,
                        "the candidates are not contiguous rows of blocks".to_owned(),
                    ));
                }
                blocks
            }
            (None, 0) => 0,
            _ => {
                return Err(Error::argument(
                    Self::NAME,
                    "candidates are passed exactly when the compilation restricts to them"
                        .to_owned(),
                ));
            }
        };

        let columns = d.columns(max_pages, blocks);
        args.out.require_shape(Self::NAME, &[tokens, columns])?;
        args.out.require_row(Self::NAME, DType::F32, columns)?;
        for (name, view) in [("query", &args.query), ("out", &args.out)] {
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
        let blocks = args.candidates.as_ref().map_or(0, |view| view.dim(1));
        let columns = descriptor.columns(max_pages, blocks);

        // An unweighted or unrestricted call passes the unread operand, which its program,
        // compiled without it, never dereferences.
        let unread = Operand::unread();
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("query", &args.query)
            .view("weights", args.weights.as_ref().unwrap_or(&unread))
            .view("keys", &args.keys)
            .view("page_table", &args.page_table)
            .view("rows", &args.rows)
            .view("visible", &args.visible)
            .view("candidates", args.candidates.as_ref().unwrap_or(&unread))
            .view("out", &args.out)
            .u64("tokens", tokens)
            .u64("max_pages", max_pages)
            .u64("blocks", blocks);
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
        // Each score sums `heads` rectified dot products of `width` terms in FP32, each weighted
        // by one FP32 product, and is stored as FP32: a dot product's bound over `heads · width`
        // terms, one rounding more per term for the weight.
        Tolerance::split_dot(
            u64::from(descriptor.heads) * u64::from(descriptor.width),
            u64::from(descriptor.weighted),
            DType::F32,
            DType::F32,
        )
    }
}

/// One compilation of `index_select`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IndexSelectDescriptor {
    /// Score columns one block ranks by its largest: one for a selection of columns, `B` for the
    /// block mode, which pins each row's newest visible block.
    pub pool: u32,
    /// Groups of one candidate block the scores were restricted to, which a kept column maps
    /// back through; zero for scores of the request's groups themselves.
    pub candidates: u32,
    /// Lanes one pass of the program reads.
    pub tile: u32,
}

/// The lanes one `index_select` pass reads: one chunk of a row's scores at a time, so the
/// program serves rows of any length.
const SELECT_TILE: u32 = 4096;

impl IndexSelectDescriptor {
    /// The compilation that ranks blocks of `pool` columns of scores restricted to candidate
    /// blocks of `candidates` groups (zero for none).
    pub fn new(pool: u32, candidates: u32) -> Self {
        Self {
            pool,
            candidates,
            tile: SELECT_TILE,
        }
    }
}

impl fmt::Display for IndexSelectDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("pool", self.pool)
            .field("candidates", self.candidates)
            .field("tile", self.tile)
            .finish(formatter)
    }
}

/// The arguments of `index_select`.
#[derive(Clone)]
pub struct IndexSelectArgs {
    /// `[tokens, columns]` F32, contiguous: the scores ([`IndexScores`]).
    pub scores: Operand,
    /// `[tokens]` U32: the groups each token sees, which bound the blocks a row of its request's
    /// own groups ranks and whose newest the block mode pins.
    pub visible: Operand,
    /// `[tokens, blocks]` I32, contiguous: the candidate blocks the scores were restricted to,
    /// exactly when the descriptor maps through them.
    pub candidates: Option<Operand>,
    /// `[tokens, k]` I32, contiguous: receives the kept groups (or blocks), `-1` for a slot whose
    /// score is `−inf`.
    pub out: Operand,
}

/// The `k` best groups (or blocks) of each score row.
///
/// ```text
/// pooled[t, b] = max_{c < pool} s[t, b · pool + c]          +∞ for b = (visible[t] − 1) / pool
///                                                           when pool > 1 and visible[t] > 0
/// kept[t]      = the k blocks of the largest pooled scores, a tie keeping the lower block
/// out[t, i]    = −1 where pooled = −∞; else the block, or, mapped through candidates,
///                candidates[t, c / B] · B + c mod B for the kept column c
/// ```
///
/// The kept entries leave in the order the program compacts them: the strictly greater ones by
/// column, then the ties at the k-th score by column. A selection is a set, so a consumer
/// reads it in any order.
///
/// A row of the request's own groups (no candidates) ranks only the blocks below
/// `⌈visible[t] / pool⌉`, the ones its visible groups lie in, whose scores past them
/// [`IndexScores`] states `−∞`: a row costs its request's length rather than the table's, and the
/// slots past its ranked blocks are `−1`.
pub struct IndexSelect;

impl Op for IndexSelect {
    const NAME: &'static str = proto::module::operator::INDEX_SELECT;

    type Descriptor = IndexSelectDescriptor;
    type Args<'a> = IndexSelectArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let d = descriptor;
        if d.pool == 0 || d.tile == 0 || (d.pool > 1 && d.candidates > 0) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "blocks of {} columns over candidates of {}, in passes of {}: the block mode \
                     ranks unrestricted scores",
                    d.pool, d.candidates, d.tile
                ),
            ));
        }

        args.scores.require_rank(Self::NAME, 2)?;
        let (tokens, columns) = (args.scores.dim(0), args.scores.dim(1));
        args.scores
            .require_row(Self::NAME, DType::F32, columns)
            .map_err(|error| Error::argument(Self::NAME, format!("scores: {error}")))?;
        args.visible
            .require_shape(Self::NAME, &[tokens])
            .and_then(|()| args.visible.require_row(Self::NAME, DType::U32, tokens))
            .map_err(|error| Error::argument(Self::NAME, format!("visible: {error}")))?;

        args.out.require_rank(Self::NAME, 2)?;
        let kept = args.out.dim(1);
        let blocks = columns.div_ceil(u64::from(d.pool));
        args.out
            .require_shape(Self::NAME, &[tokens, kept])
            .and_then(|()| args.out.require_row(Self::NAME, DType::I32, kept))
            .map_err(|error| Error::argument(Self::NAME, format!("out: {error}")))?;
        if kept == 0 || kept > blocks || columns == 0 || columns > i32::MAX as u64 {
            return Err(Error::argument(
                Self::NAME,
                format!("{kept} kept of {blocks} blocks of {columns} columns"),
            ));
        }

        match (&args.candidates, d.candidates) {
            (Some(candidates), block) if block > 0 => {
                let groups = candidates.dim(1) * u64::from(block);
                candidates
                    .require_shape(Self::NAME, &[tokens, candidates.dim(1)])
                    .and_then(|()| {
                        candidates.require_row(Self::NAME, DType::I32, candidates.dim(1))
                    })
                    .map_err(|error| Error::argument(Self::NAME, format!("candidates: {error}")))?;
                if groups != columns || !candidates.is_row_major() {
                    return Err(Error::argument(
                        Self::NAME,
                        format!(
                            "{} candidate blocks of {block} for scores of {columns} columns",
                            candidates.dim(1)
                        ),
                    ));
                }
            }
            (None, 0) => {}
            _ => {
                return Err(Error::argument(
                    Self::NAME,
                    "candidates are passed exactly when the compilation maps through them"
                        .to_owned(),
                ));
            }
        }
        for (name, view) in [("scores", &args.scores), ("out", &args.out)] {
            if !view.is_row_major() {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} are not contiguous"),
                ));
            }
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        _descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.scores.dim(0);
        let unread = Operand::unread();

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("scores", &args.scores)
            .view("visible", &args.visible)
            .view("candidates", args.candidates.as_ref().unwrap_or(&unread))
            .view("out", &args.out)
            .u64("columns", args.scores.dim(1))
            .u64("kept", args.out.dim(1));
        kernel.launch(LaunchGrid::linear(tokens)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        IndexSelectArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // A selection by an exact order with a stated tie rule: two implementations keep the same
        // blocks in the same slots.
        Tolerance::exact()
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

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("query", &args.query)
            .view("key", &args.key)
            .view("value", &args.value)
            .view("page_table", &args.page_table)
            .view("rows", &args.rows)
            .view("selection", &args.selection)
            .view("out", &args.out)
            .u64("tokens", tokens)
            .u64("max_pages", args.page_table.dim(1));
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
        .with_atol((value_term * f64::from(CACHED_VALUE_ENVELOPE)) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_keys_state_the_heads_the_pages_and_the_selection() {
        let scores = IndexScoresDescriptor::new(DType::BF16, Format::Bf16, 4, 128, 16);
        assert_eq!(
            scores.to_string(),
            "dtype_bf16_keys_bf16_heads_4_width_128_groups_16_block_256"
        );

        // A weighted scoring of MXFP4 keys by MXFP4-rounded queries inside candidate blocks.
        let restricted = IndexScoresDescriptor {
            weighted: true,
            query: Format::Mxfp4,
            candidates: 8,
            ..IndexScoresDescriptor::new(DType::BF16, Format::Mxfp4, 32, 128, 64)
        };
        assert_eq!(
            restricted.to_string(),
            "dtype_bf16_keys_mxfp4_heads_32_width_128_groups_64_block_256_weighted_true_\
             query_mxfp4_candidates_8"
        );
        assert_eq!(restricted.columns(100, 2048), 2048 * 8);
        assert_eq!(
            IndexSelectDescriptor::new(8, 0).to_string(),
            "pool_8_candidates_0_tile_4096"
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
