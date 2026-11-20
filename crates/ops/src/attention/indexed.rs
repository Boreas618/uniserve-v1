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
//! them before it scores (`fp4_act_quant` of DeepSeek's `Indexer`). The scores are ranked, so
//! where they round decides which entries a selection keeps: FP32 scores are the FP32 sums of
//! the products, and BF16 scores are DeepSeek's indexer's own arithmetic, which rounds each
//! head's product, its weighted value and the heads' sum to BF16 (`inference/model.py`,
//! `Indexer.forward`: a BF16 `einsum`, a BF16 product with the weights and a BF16 `sum`), so a
//! selection of them keeps the entries the reference ranks first, near ties included. The
//! attention computes its scores, its softmax and its sum in FP32; with caches of a 16-bit type
//! each probability is rounded into that type before it multiplies its value, as the reference
//! casts them (`qsa.py:141-253`), and the output rounds once.
//!
//! Each operator is served by Triton programs and by native modules; a native module takes the
//! operator's call structure ([`ScoresCall`], [`SelectCall`]).

use std::ffi::{c_char, c_void};
use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi::{self, MESSAGE};
use crate::descriptor::{Float, Key};
use crate::kernel::{Arguments, Kernel, LaunchGrid, NativeCall};
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
    /// The type the scores round into and are stored in: [`DType::F32`], each score the FP32 sum
    /// of the weighted rectified products; or [`DType::BF16`], DeepSeek's indexer's arithmetic,
    /// [`IndexScores`]'s rounded formula.
    pub scores: DType,
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
            scores: DType::F32,
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
        // The weighting, the query's rounding, the candidates and the scores' type are written
        // after `block`, and only when they are not the unweighted dense FP32 scoring every
        // earlier key states.
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
        let key = if self.scores == DType::F32 {
            key
        } else {
            key.field("scores", abi::name(self.scores))
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
    /// `[tokens, columns]` of the descriptor's score type, contiguous: receives the scores,
    /// `−inf` for a column of no visible group ([`IndexScoresDescriptor::columns`]).
    pub out: Operand,
    /// `[bytes]` U8, at least [`IndexScores::workspace_bytes`]: the call's transient workspace,
    /// whatever it held before. A native module keeps there the operands it derives for its
    /// kernels (the rounded queries, the weights in the kernels' type, its schedule); a Triton
    /// program reads none of it.
    pub workspace: Operand,
}

/// The multiprocessors an `index_scores` workspace holds a schedule for
/// ([`IndexScores::workspace_bytes`]): more than any device the operator launches on has.
const SCHEDULED_MULTIPROCESSORS: u64 = 1024;

/// The alignment of each region a native module carves from an `index_scores` workspace: a
/// tensor map's base address.
const WORKSPACE_ALIGNMENT: u64 = 16;

/// The call structure `index_scores`'s native entry takes, declared to the package as
/// `uniserve_index_scores_call` (`uniserve_index_scores.h`, generated with its layout asserted:
/// [`abi`]).
///
/// The operands' device addresses, the launch's extents, the descriptor's values the module is
/// compiled for (which the entry holds the call to), and the multiprocessors of the device the
/// launch runs on. An unweighted or unrestricted call passes a null `weights` or `candidates`.
#[repr(C)]
#[derive(abi::Layout)]
pub struct ScoresCall {
    /// `[tokens, heads, width]` queries of the descriptor's element type.
    pub query: *const c_void,
    /// `[tokens, heads]` F32 head weights, or null.
    pub weights: *const c_void,
    /// `[pages, groups, record]`: the index-key plane.
    pub keys: *const c_void,
    /// `[table_rows, max_pages]` U32: each row's pages.
    pub page_table: *const c_void,
    /// `[tokens]` U32: each token's row of the page table.
    pub rows: *const c_void,
    /// `[tokens]` U32: the groups each token sees.
    pub visible: *const c_void,
    /// `[tokens, blocks]` I32 candidate blocks, or null.
    pub candidates: *const c_void,
    /// `[tokens, columns]` scores of the descriptor's score type.
    pub out: *mut c_void,
    /// The call's transient workspace and its bytes ([`IndexScoresArgs::workspace`]).
    pub workspace: *mut c_void,
    pub workspace_bytes: i64,
    pub tokens: i64,
    /// Pages of the key plane.
    pub pages: i64,
    /// Rows of the page table.
    pub table_rows: i64,
    /// Pages of one table row.
    pub max_pages: i64,
    /// Candidate blocks of one row; zero for a dense call.
    pub blocks: i64,
    /// Columns of a score row ([`IndexScoresDescriptor::columns`]).
    pub columns: i64,
    /// The descriptor's index heads, elements of a head, groups of a page and groups of a
    /// candidate block.
    pub heads: i64,
    pub width: i64,
    pub groups: i64,
    pub candidate_block: i64,
    /// The descriptor's score type (`uniserve_dtype`).
    pub scores_dtype: i64,
    /// The multiprocessors of the device the launch runs on.
    pub multiprocessors: i64,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for ScoresCall {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// The scores of the index keys a query row may select among.
///
/// ```text
/// FP32   s[t, g] = Σ_h relu(q[t, h] · key[g]) · w[t, h]
/// BF16   s[t, g] = bf16(Σ_h bf16(relu(bf16(q[t, h] · key[g])) · bf16(w[t, h])))   Σ_h in FP32
/// ```
///
/// Each head's product is FP32 either way (codes under power-of-two scales: the products are
/// exact and only their sum rounds), its weighting one FP32 product. The BF16 scores round where
/// DeepSeek's indexer rounds, whose head weights are BF16 values, so that their order is the
/// reference's own, whose BF16 `topk` keeps the lower entry of a tie: equal BF16 scores are ties
/// a selection breaks by its own rule. A weighted value is the product of two BF16 values, exact
/// in FP32 before it rounds.
pub struct IndexScores;

impl IndexScores {
    /// The bytes of the workspace a call of `tokens` query rows passes
    /// ([`IndexScoresArgs::workspace`]): room for each row's queries as MXFP4 records (`width / 2`
    /// value bytes and `width / 32` scale bytes a head), its weights as a BF16 row padded to 16
    /// bytes, four 32-bit words of it (its table row, its visible count, and the first row and
    /// the end of the run of rows it is scheduled with), a schedule of two 32-bit words for each
    /// of up to 1024 multiprocessors and one more, and the alignment of each of those eight
    /// regions to 16 bytes.
    pub fn workspace_bytes(descriptor: &IndexScoresDescriptor, tokens: u64) -> u64 {
        let (heads, width) = (u64::from(descriptor.heads), u64::from(descriptor.width));
        let records = heads * (width / 2 + width.div_ceil(32));
        let weights = (2 * heads).next_multiple_of(WORKSPACE_ALIGNMENT);
        let schedule = 8 * (SCHEDULED_MULTIPROCESSORS + 1);
        tokens * (records + weights + 4 * 4) + schedule + 8 * WORKSPACE_ALIGNMENT
    }
}

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
            || !matches!(d.scores, DType::F32 | DType::BF16)
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} index heads of {} elements over pages of {} keys, in blocks of {}, \
                     queries read as {}, {} scores",
                    d.heads,
                    d.width,
                    d.groups,
                    d.block,
                    d.query,
                    abi::name(d.scores)
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
        args.out.require_row(Self::NAME, d.scores, columns)?;
        let needed = Self::workspace_bytes(d, tokens);
        if args.workspace.dtype != DType::U8
            || args.workspace.rank() != 1
            || args.workspace.numel() < needed
            || !args.workspace.is_row_major()
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the workspace is a contiguous u8 vector of at least {needed} bytes for \
                     {tokens} rows"
                ),
            ));
        }
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

        // A native module keeps the call structure; a Triton program takes the arguments its
        // package declares. The descriptor names no backend: the store decides.
        if !kernel.is_cubin() {
            let address = |operand: &Operand| operand.ptr.address() as usize as *const c_void;
            let optional =
                |operand: &Option<Operand>| operand.as_ref().map_or(std::ptr::null(), address);
            let mut call = ScoresCall {
                query: address(&args.query),
                weights: optional(&args.weights),
                keys: address(&args.keys),
                page_table: address(&args.page_table),
                rows: address(&args.rows),
                visible: address(&args.visible),
                candidates: optional(&args.candidates),
                out: args.out.ptr.address() as usize as *mut c_void,
                workspace: args.workspace.ptr.address() as usize as *mut c_void,
                workspace_bytes: args.workspace.numel() as i64,
                tokens: tokens as i64,
                pages: args.keys.dim(0) as i64,
                table_rows: args.page_table.dim(0) as i64,
                max_pages: max_pages as i64,
                blocks: blocks as i64,
                columns: columns as i64,
                heads: i64::from(descriptor.heads),
                width: i64::from(descriptor.width),
                groups: i64::from(descriptor.groups),
                candidate_block: i64::from(descriptor.candidates),
                scores_dtype: abi::code(descriptor.scores)?,
                multiprocessors: i64::from(stream.device().multiprocessors()?),
                error: [0; MESSAGE],
            };
            // SAFETY: `ScoresCall` is the structure this package's entry declares, generated
            // from this definition with its layout asserted.
            return unsafe { kernel.native(&mut call, stream) };
        }

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

    /// FP32 scores: each sums `heads` rectified dot products of `width` terms in FP32, each
    /// weighted by one FP32 product, and is stored as FP32: a dot product's bound over `heads ·
    /// width` terms, one rounding more per term for the weight.
    ///
    /// BF16 scores add the format's roundings to that bound. Two implementations whose FP32 head
    /// products differ by the dot product's bound may round a head's product, and then its
    /// weighted value, to neighbouring BF16 values: each of the two roundings moves a head's term
    /// by at most one BF16 spacing, `2u` of it (`u = 2⁻⁸`), and the `heads` terms' errors add
    /// as a dot product's do, `√heads` times the largest. The FP32 sum of the heads is a sum of
    /// `heads` terms (`2γ_heads √heads`), and its rounding to BF16 may again land on neighbours,
    /// `2u` of the score. Like the FP32 bound it is relative to the row's largest score, which
    /// the weights' signs let a score cancel below.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let heads = u64::from(descriptor.heads);
        let products = Tolerance::split_dot(
            heads * u64::from(descriptor.width),
            u64::from(descriptor.weighted),
            DType::F32,
            DType::F32,
        );
        if descriptor.scores == DType::F32 {
            return products;
        }

        let spacing = 2.0 * tolerance::unit_roundoff(DType::BF16);
        let spread = (heads as f64).sqrt();
        let rounded = f64::from(products.rtol)
            + 2.0 * spacing * spread
            + 2.0 * tolerance::gamma(heads, DType::F32) * spread
            + spacing;
        Tolerance {
            atol: tolerance::underflow(DType::BF16) as f32,
            rtol: rounded as f32,
            cancels: true,
        }
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
    /// Element type of the scores: [`DType::F32`] or [`DType::BF16`].
    pub scores: DType,
}

/// The lanes one `index_select` pass reads: one chunk of a row's scores at a time, so the
/// program serves rows of any length.
const SELECT_TILE: u32 = 4096;

impl IndexSelectDescriptor {
    /// The compilation that ranks blocks of `pool` columns of FP32 scores restricted to
    /// candidate blocks of `candidates` groups (zero for none).
    pub fn new(pool: u32, candidates: u32) -> Self {
        Self {
            pool,
            candidates,
            tile: SELECT_TILE,
            scores: DType::F32,
        }
    }
}

impl fmt::Display for IndexSelectDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = Key::new()
            .field("pool", self.pool)
            .field("candidates", self.candidates)
            .field("tile", self.tile);
        // The scores' type is written only for scores other than the FP32 ones every earlier key
        // ranks.
        let key = if self.scores == DType::F32 {
            key
        } else {
            key.field("scores", abi::name(self.scores))
        };
        key.finish(formatter)
    }
}

/// The arguments of `index_select`.
#[derive(Clone)]
pub struct IndexSelectArgs {
    /// `[tokens, columns]` of the descriptor's score type, contiguous: the scores
    /// ([`IndexScores`]).
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

/// The call structure `index_select`'s native entry takes, declared to the package as
/// `uniserve_index_select_call` (`uniserve_index_select.h`, generated with its layout asserted:
/// [`abi`]). A selection of the request's own groups passes a null `candidates`.
#[repr(C)]
#[derive(abi::Layout)]
pub struct SelectCall {
    /// `[tokens, columns]` scores of the descriptor's score type.
    pub scores: *const c_void,
    /// `[tokens]` U32: the groups each token sees.
    pub visible: *const c_void,
    /// `[tokens, columns / candidate_block]` I32 candidate blocks, or null.
    pub candidates: *const c_void,
    /// `[tokens, kept]` I32: the kept entries.
    pub out: *mut c_void,
    pub tokens: i64,
    pub columns: i64,
    pub kept: i64,
    /// The descriptor's pool and candidate block, which the module is compiled for.
    pub pool: i64,
    pub candidate_block: i64,
    /// The descriptor's score type (`uniserve_dtype`).
    pub scores_dtype: i64,
    /// The multiprocessors of the device the launch runs on.
    pub multiprocessors: i64,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for SelectCall {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
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
        if d.pool == 0
            || d.tile == 0
            || (d.pool > 1 && d.candidates > 0)
            || !matches!(d.scores, DType::F32 | DType::BF16)
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "blocks of {} columns over candidates of {}, in passes of {}, of {} scores: \
                     the block mode ranks unrestricted FP32 or BF16 scores",
                    d.pool,
                    d.candidates,
                    d.tile,
                    abi::name(d.scores)
                ),
            ));
        }

        args.scores.require_rank(Self::NAME, 2)?;
        let (tokens, columns) = (args.scores.dim(0), args.scores.dim(1));
        args.scores
            .require_row(Self::NAME, d.scores, columns)
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
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.scores.dim(0);

        // A native module keeps the call structure; a Triton program takes its package's
        // arguments.
        if !kernel.is_cubin() {
            let address = |operand: &Operand| operand.ptr.address() as usize as *const c_void;
            let mut call = SelectCall {
                scores: address(&args.scores),
                visible: address(&args.visible),
                candidates: args.candidates.as_ref().map_or(std::ptr::null(), address),
                out: args.out.ptr.address() as usize as *mut c_void,
                tokens: tokens as i64,
                columns: args.scores.dim(1) as i64,
                kept: args.out.dim(1) as i64,
                pool: i64::from(descriptor.pool),
                candidate_block: i64::from(descriptor.candidates),
                scores_dtype: abi::code(descriptor.scores)?,
                multiprocessors: i64::from(stream.device().multiprocessors()?),
                error: [0; MESSAGE],
            };
            // SAFETY: `SelectCall` is the structure this package's entry declares, generated
            // from this definition with its layout asserted.
            return unsafe { kernel.native(&mut call, stream) };
        }

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

        // DeepSeek's BF16 scores, and their selection.
        let rounded = IndexScoresDescriptor {
            scores: DType::BF16,
            ..restricted
        };
        assert_eq!(
            rounded.to_string(),
            "dtype_bf16_keys_mxfp4_heads_32_width_128_groups_64_block_256_weighted_true_\
             query_mxfp4_candidates_8_scores_bf16"
        );
        assert_eq!(
            IndexSelectDescriptor {
                scores: DType::BF16,
                ..IndexSelectDescriptor::new(1, 8)
            }
            .to_string(),
            "pool_1_candidates_8_tile_4096_scores_bf16"
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
