//! Attention over one shared latent that is both key and value: a sliding window of the
//! request's latest entries and the compressed entries an indexer selected, in one softmax with a
//! per-head sink.
//!
//! ```text
//! entries(t)  = window(t) ∪ selected(t)
//!             window(t)   = the window entries at positions max(start, p_t − W + 1) ..= p_t,
//!                           start = starts[row(t)], none for a padding token's negative
//!                           position
//!             selected(t) = the compressed entries selection[t, :] names, a negative one none
//! s[t,h,n]    = scale · q[t,h] · c[n]                         n ∈ entries(t)
//! m[t,h]      = max(−1e30, max_n s[t,h,n])
//! o[t,h]      = Σ_n e^(s − m) · c[n]  /  (Σ_n e^(s − m) + e^(sink[h] − m))
//! ```
//!
//! `q` is `[tokens, heads, head_dim]` and every entry `c` is one `head_dim`-wide latent read from
//! its plane's record ([`crate::record`]), the key and the value at once. The window holds one
//! entry per token, so the window entry of position `p` is entry `p` of the row's window table;
//! the compressed plane holds one entry per `ratio` tokens, and the selection names compressed
//! entries by their index in the request's order. Entry `e` of a table lies on page
//! `table[row, (e / entries) mod max_pages]`, at slot `e mod entries`: a table as wide as the
//! request addresses every page directly, and a narrower window table is a ring of pages. A
//! row's window holds its entries from `starts[row]` on: the position a request resumed from a
//! prefix hit recomputes its window from (zero for one that computed every token), before
//! which the window group holds nothing.
//!
//! The finite floor of the maximum is the reference's (`inference/kernel.py`, `sparse_attn`):
//! a query that reads no entry attends the sink alone and receives zeros. The scores, the
//! softmax and the sums are FP32; each probability is rounded into the queries' type before it
//! multiplies its latent, as the reference's tensor-core product does, and the output rounds
//! once. The queries arrive rotated and the output leaves as attended: the rotation and its
//! inverse are [`crate::rope::RopeApply`]'s.
//!
//! # The packages
//!
//! A Triton program walks each token's entries itself, from the tables, the positions and the
//! selection above, and takes the arguments its package declares. A native module instead
//! receives [`LatentAttentionCall`] and owns its whole host side: the `flashmla` package reads
//! the entries by their slot in the planes, which it materializes from the same tables into the
//! call's workspace before its attention. One operator is served by two kinds of program, which
//! is why the descriptor names no backend and the store decides.
//!
//! The workspace is transient scratch the caller lends, as paged attention's is:
//! [`workspace_bytes`] is what a call of a descriptor needs at most, and a module refuses a
//! shorter one. A Triton launch reads none of it.

use std::ffi::{c_char, c_void};
use std::fmt;

use platform::Stream;
use tensor::DType;

use super::{address, to_i64};
use crate::abi::{self, MESSAGE};
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid, NativeCall};
use crate::operand::Operand;
use crate::record::Format;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// The largest magnitude a cached latent is declared to have, for the attention's absolute
/// bound: an RMS-normalized row times its learned gain, as `tolerance::VALUE_ENVELOPE` is about
/// the activations.
const LATENT_ENVELOPE: f32 = 64.0;

/// The query heads one program attends: the rows of its products.
const HEAD_BLOCK: u32 = 16;

/// The entries a native module reads as one block, which its workspace holds whole rows of:
/// FlashMLA's `B_TOPK`, the entries one of its tile loads gathers.
const SLOT_BLOCK: u64 = 64;

/// The bytes a package may carve its workspace into, and the alignment of each region it holds
/// ([`crate::allocator::ALIGNMENT`], the alignment every operand has).
const WORKSPACE_ALIGNMENT: u64 = 256;

/// `value` rounded up to a whole multiple of `granule`.
const fn aligned(value: u64, granule: u64) -> u64 {
    value.div_ceil(granule) * granule
}

/// The workspace bytes a call of `descriptor` over `tokens` query tokens needs at most: what a
/// native module carves out of it, each region aligned to [`WORKSPACE_ALIGNMENT`].
///
/// A module reads the entries of a token by their slot in the planes, so the regions are, in
/// order: one row of `SLOT_BLOCK`-aligned window slots per token and one window length per
/// token, then, for a compilation with a compressed side, the same two over the selection's
/// columns, and last one FP32 normalizer per query head of a token. A Triton launch reads none
/// of them, and a module that needs fewer bytes refuses nothing.
pub fn workspace_bytes(descriptor: &LatentAttentionDescriptor, tokens: u64) -> u64 {
    const WORD: u64 = 4;

    let slots = |columns: u64| {
        aligned(
            tokens * aligned(columns, SLOT_BLOCK) * WORD,
            WORKSPACE_ALIGNMENT,
        ) + aligned(tokens * WORD, WORKSPACE_ALIGNMENT)
    };
    let compressed = descriptor
        .compressed
        .map_or(0, |side| slots(u64::from(side.columns)));

    slots(u64::from(descriptor.window))
        + compressed
        + aligned(
            tokens * u64::from(descriptor.heads) * WORD,
            WORKSPACE_ALIGNMENT,
        )
}

/// The compressed side of a `latent_attention` compilation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Compressed {
    /// The format of the compressed entries' records.
    pub format: Format,
    /// Compressed entries per page.
    pub entries: u32,
    /// Columns of a selection row: the most compressed entries one query attends.
    pub columns: u32,
}

/// One compilation of `latent_attention`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LatentAttentionDescriptor {
    /// Element type of the queries and the output.
    pub dtype: DType,
    /// Query heads of a token.
    pub heads: u32,
    /// Elements of a head and of a latent.
    pub head_dim: u32,
    /// The format of the window entries' records.
    pub window_format: Format,
    /// Window entries a query attends: its own position and the `window − 1` before it.
    pub window: u32,
    /// Window entries per page.
    pub page: u32,
    /// The compressed entries a query attends beside its window; `None` for a window-only layer.
    pub compressed: Option<Compressed>,
    /// Query heads one program attends.
    pub block: u32,
}

impl LatentAttentionDescriptor {
    /// The compilation over `heads` query heads of `head_dim` in `dtype`, a window of `window`
    /// entries in pages of `page` records of `window_format`, and the compressed side, if any.
    pub fn new(
        dtype: DType,
        heads: u32,
        head_dim: u32,
        (window_format, window, page): (Format, u32, u32),
        compressed: Option<Compressed>,
    ) -> Self {
        Self {
            dtype,
            heads,
            head_dim,
            window_format,
            window,
            page,
            compressed,
            block: HEAD_BLOCK.min(heads),
        }
    }

    /// The most entries one query's softmax reduces over: the window and the selection.
    pub fn reduction(&self) -> u64 {
        u64::from(self.window) + self.compressed.map_or(0, |side| u64::from(side.columns))
    }
}

impl fmt::Display for LatentAttentionDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("heads", self.heads)
            .field("head_dim", self.head_dim)
            .field("window_format", self.window_format)
            .field("window", self.window)
            .field("page", self.page);
        let key = match self.compressed {
            Some(side) => key
                .field("compressed", side.format)
                .field("entries", side.entries)
                .field("columns", side.columns),
            None => key.field("compressed", "none"),
        };
        key.field("block", self.block).finish(formatter)
    }
}

/// The compressed operands of a `latent_attention` call.
#[derive(Clone)]
pub struct CompressedArgs {
    /// `[pages, entries, record]`: the compressed entries' records.
    pub plane: Operand,
    /// `[rows, max_pages]` U32: each row's compressed pages.
    pub table: Operand,
    /// `[tokens, columns]` I32, contiguous: the compressed entries each query attends, in its
    /// request's order; a negative entry names none.
    pub selection: Operand,
}

/// The arguments of `latent_attention`.
#[derive(Clone)]
pub struct LatentAttentionArgs {
    /// `[tokens, heads, head_dim]`, contiguous: the rotated queries.
    pub query: Operand,
    /// `[pages, page, record]`: the window entries' records.
    pub window: Operand,
    /// `[rows, max_pages]` U32: each row's window pages.
    pub window_table: Operand,
    /// `[rows]` U32: the first position whose window entry each row holds.
    pub window_starts: Operand,
    /// The compressed side, exactly when the descriptor has one.
    pub compressed: Option<CompressedArgs>,
    /// `[tokens]` U32: each token's row of the tables.
    pub rows: Operand,
    /// `[tokens]` I32: each token's position, the newest window entry it attends.
    pub positions: Operand,
    /// `[heads]` F32: each head's sink logit.
    pub sink: Operand,
    /// `[tokens, heads, head_dim]`, contiguous: receives the attended latents.
    pub out: Operand,
    /// `[bytes]` U8: transient scratch a native module carves ([`workspace_bytes`]), which it
    /// refuses when it is short; a Triton launch reads none of it.
    pub workspace: Operand,
    /// The factor the scores are multiplied by.
    pub scale: f32,
}

/// The call structure a native package's entry receives, declared to the packages as
/// `uniserve_latent_attention_call` (`uniserve_latent_attention.h`, generated with its layout
/// asserted: [`abi`]).
///
/// The planes are `[pages, entries, record]` bytes of the descriptor's formats, the tables
/// `[rows, pages]` page ids, and the module addresses an entry as the operator states it: page
/// `table[row, (e / entries) mod pages]`, slot `e mod entries`. A window-only compilation is
/// handed null for every compressed operand and zero widths.
#[repr(C)]
#[derive(Debug, abi::Layout)]
pub struct LatentAttentionCall {
    /// `[tokens, heads, head_dim]`, contiguous: the rotated queries.
    pub query: *const c_void,
    /// `[tokens, heads, head_dim]`, contiguous: receives the attended latents.
    pub out: *mut c_void,
    /// `[pages, page, window_record]` bytes: the window entries' records.
    pub window: *const c_void,
    /// `[rows, window_pages]` u32: each row's window pages.
    pub window_table: *const c_void,
    /// `[rows]` u32: the first position whose window entry each row holds.
    pub window_starts: *const c_void,
    /// `[pages, entries, compressed_record]` bytes, or null for a window-only compilation.
    pub compressed: *const c_void,
    /// `[rows, compressed_pages]` u32, or null.
    pub compressed_table: *const c_void,
    /// `[tokens, columns]` i32: the compressed entries each query attends, a negative one none;
    /// null for a window-only compilation.
    pub selection: *const c_void,
    /// `[tokens]` u32: each token's row of the tables.
    pub rows: *const c_void,
    /// `[tokens]` i32: each token's position, the newest window entry it attends.
    pub positions: *const c_void,
    /// `[heads]` f32: each head's sink logit.
    pub sink: *const c_void,
    /// The scratch the module carves, and its bytes; the entry refuses a short one.
    pub workspace: *mut c_void,
    /// Bytes of `workspace`.
    pub workspace_bytes: i64,
    /// Query tokens of the call.
    pub tokens: i64,
    /// Query heads of a token.
    pub heads: i64,
    /// Elements of a head and of a latent.
    pub head_dim: i64,
    /// Window entries a query attends: its own position and the ones before it.
    pub window_size: i64,
    /// Window entries per page.
    pub page: i64,
    /// Columns of the window table: a ring of pages when it is narrower than the request.
    pub window_pages: i64,
    /// Bytes of one window record.
    pub window_record: i64,
    /// Pages of the window plane.
    pub window_plane_pages: i64,
    /// Compressed entries per page; zero for a window-only compilation.
    pub entries: i64,
    /// Columns of a selection row; zero for a window-only compilation.
    pub columns: i64,
    /// Columns of the compressed table; zero for a window-only compilation.
    pub compressed_pages: i64,
    /// Bytes of one compressed record; zero for a window-only compilation.
    pub compressed_record: i64,
    /// Pages of the compressed plane; zero for a window-only compilation.
    pub compressed_plane_pages: i64,
    /// The multiprocessors of the device the call runs on.
    pub multiprocessors: i64,
    /// The factor the scores are multiplied by.
    pub scale: f32,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for LatentAttentionCall {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// Checks a `[rows, max_pages]` U32 page table and answers its width.
fn check_table(operator: &'static str, name: &str, table: &Operand) -> Result<u64> {
    table
        .require_rank(operator, 2)
        .and_then(|()| table.require_row(operator, DType::U32, table.dim(1)))
        .map_err(|error| Error::argument(operator, format!("the {name} table: {error}")))?;
    if !table.is_row_major() || table.dim(1) == 0 {
        return Err(Error::argument(
            operator,
            format!("the {name} table is not contiguous rows of pages"),
        ));
    }
    Ok(table.dim(1))
}

/// Checks a `[tokens]` index column of `dtype`.
fn check_column(
    operator: &'static str,
    name: &str,
    column: &Operand,
    dtype: DType,
    tokens: u64,
) -> Result<()> {
    column
        .require_shape(operator, &[tokens])
        .and_then(|()| column.require_row(operator, dtype, tokens))
        .map_err(|error| Error::argument(operator, format!("the {name}: {error}")))
}

/// Sparse latent attention with a per-head sink; see the module documentation.
pub struct LatentAttention;

impl Op for LatentAttention {
    const NAME: &'static str = proto::module::operator::LATENT_ATTENTION;

    type Descriptor = LatentAttentionDescriptor;
    type Args<'a> = LatentAttentionArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let d = descriptor;
        if d.heads == 0
            || d.head_dim == 0
            || d.window == 0
            || d.page == 0
            || d.block == 0
            || !d.heads.is_multiple_of(d.block)
            || d.dtype != DType::BF16
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} {} query heads of {} in blocks of {}, a window of {} in pages of {}: BF16 \
                     queries whose heads the block divides",
                    d.heads,
                    abi::name(d.dtype),
                    d.head_dim,
                    d.block,
                    d.window,
                    d.page
                ),
            ));
        }
        let (heads, head_dim) = (u64::from(d.heads), u64::from(d.head_dim));
        d.window_format.check_width(Self::NAME, head_dim)?;

        args.query.require_rank(Self::NAME, 3)?;
        let tokens = args.query.dim(0);
        for (name, view) in [("queries", &args.query), ("output", &args.out)] {
            view.require_shape(Self::NAME, &[tokens, heads, head_dim])
                .and_then(|()| view.require_row(Self::NAME, d.dtype, head_dim))
                .map_err(|error| Error::argument(Self::NAME, format!("the {name}: {error}")))?;
            if !view.is_row_major() {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} are not contiguous"),
                ));
            }
        }

        d.window_format.check_plane(
            Self::NAME,
            "window",
            &args.window,
            u64::from(d.page),
            head_dim,
        )?;
        check_table(Self::NAME, "window", &args.window_table)?;
        let table_rows = args.window_table.dim(0);

        match (d.compressed, &args.compressed) {
            (None, None) => {}
            (Some(side), Some(operands)) => {
                if side.entries == 0 || side.columns == 0 {
                    return Err(Error::argument(
                        Self::NAME,
                        format!(
                            "compressed pages of {} entries and selections of {}",
                            side.entries, side.columns
                        ),
                    ));
                }
                side.format.check_width(Self::NAME, head_dim)?;
                side.format.check_plane(
                    Self::NAME,
                    "compressed",
                    &operands.plane,
                    u64::from(side.entries),
                    head_dim,
                )?;
                check_table(Self::NAME, "compressed", &operands.table)?;
                if operands.table.dim(0) != table_rows {
                    return Err(Error::argument(
                        Self::NAME,
                        format!(
                            "a window table of {table_rows} rows beside a compressed one of {}",
                            operands.table.dim(0)
                        ),
                    ));
                }
                let columns = u64::from(side.columns);
                operands
                    .selection
                    .require_shape(Self::NAME, &[tokens, columns])
                    .and_then(|()| {
                        operands
                            .selection
                            .require_row(Self::NAME, DType::I32, columns)
                    })
                    .map_err(|error| Error::argument(Self::NAME, format!("selection: {error}")))?;
                if !operands.selection.is_row_major() {
                    return Err(Error::argument(
                        Self::NAME,
                        "the selection is not contiguous".to_owned(),
                    ));
                }
            }
            (Some(_), None) => {
                return Err(Error::argument(
                    Self::NAME,
                    "a compilation with a compressed side called without its plane".to_owned(),
                ));
            }
            (None, Some(_)) => {
                return Err(Error::argument(
                    Self::NAME,
                    "a window-only compilation called with a compressed plane".to_owned(),
                ));
            }
        }

        check_column(Self::NAME, "rows", &args.rows, DType::U32, tokens)?;
        check_column(Self::NAME, "positions", &args.positions, DType::I32, tokens)?;
        check_column(
            Self::NAME,
            "window starts",
            &args.window_starts,
            DType::U32,
            table_rows,
        )?;
        args.sink
            .require_shape(Self::NAME, &[heads])
            .and_then(|()| args.sink.require_row(Self::NAME, DType::F32, heads))
            .map_err(|error| Error::argument(Self::NAME, format!("the sink: {error}")))?;
        args.workspace
            .require_rank(Self::NAME, 1)
            .and_then(|()| {
                args.workspace
                    .require_row(Self::NAME, DType::U8, args.workspace.dim(0))
            })
            .map_err(|error| Error::argument(Self::NAME, format!("the workspace: {error}")))?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.query.dim(0);

        // A window-only call passes the null address for every compressed operand and a zero
        // width, which its program, compiled without the compressed side, never reads.
        let null = Operand::unread();
        let (plane, table, selection) = match &args.compressed {
            Some(side) => (&side.plane, &side.table, &side.selection),
            None => (&null, &null, &null),
        };

        // A native module keeps the `LatentAttentionCall` structure and owns its own grid and
        // its workspace; a Triton program is this crate's own kernel over the arguments its
        // package declares, one program per head block of a token.
        if !kernel.is_cubin() {
            return Self::launch_module(kernel, descriptor, &args, stream);
        }

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("query", &args.query)
            .view("window", &args.window)
            .view("window_table", &args.window_table)
            .view("window_starts", &args.window_starts)
            .view("compressed", plane)
            .view("compressed_table", table)
            .view("selection", selection)
            .view("rows", &args.rows)
            .view("positions", &args.positions)
            .view("sink", &args.sink)
            .view("out", &args.out)
            .u64("tokens", tokens)
            .u64("window_pages", args.window_table.dim(1))
            .u64("compressed_pages", table.dim(1))
            .f32("scale", args.scale);
        kernel.launch(
            LaunchGrid::new([u64::from(descriptor.heads / descriptor.block), tokens, 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        LatentAttentionArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    /// Paged attention's bound over the window and the selection: a softmax over at most
    /// [`LatentAttentionDescriptor::reduction`] scores accumulated in FP32, `rtol = 2γₙ + u_out`,
    /// and as the absolute part the normalized weights' value term over [`LATENT_ENVELOPE`]:
    /// `(2γₙ + u + u_p) · max|c|`. Rounding each probability into the queries' type (`u_p`)
    /// perturbs every term of the weighted sum by its own share of `Σp|c| / Σp ≤ max|c|`, which
    /// does not scale with an output that cancels, so it belongs to the absolute part.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        let reduction = descriptor.reduction();
        let gamma = tolerance::gamma(reduction, DType::F32);
        let value_term = 2.0 * gamma
            + tolerance::unit_roundoff(DType::F32)
            + tolerance::unit_roundoff(descriptor.dtype);

        Tolerance::rounding(reduction, DType::F32, descriptor.dtype)
            .with_atol((value_term * f64::from(LATENT_ENVELOPE)) as f32)
    }
}

impl LatentAttention {
    /// Calls a native module's entry with [`LatentAttentionCall`], the call `check` passed.
    ///
    /// The module reads the extents of the planes and the tables from the call, so one module
    /// serves every page count, table width and token count of its descriptor; the widths the
    /// descriptor fixes travel with it, and the entry refuses a call that disagrees with the
    /// compilation.
    fn launch_module(
        kernel: &Kernel,
        descriptor: &LatentAttentionDescriptor,
        args: &LatentAttentionArgs,
        stream: &Stream,
    ) -> Result<()> {
        let name = Self::NAME;
        let head_dim = u64::from(descriptor.head_dim);
        let tokens = args.query.dim(0);
        let null = Operand::unread();
        let (plane, table, selection) = match &args.compressed {
            Some(side) => (&side.plane, &side.table, &side.selection),
            None => (&null, &null, &null),
        };
        let side = descriptor.compressed;

        let mut call = LatentAttentionCall {
            query: address(args.query.ptr) as usize as *const c_void,
            out: address(args.out.ptr) as usize as *mut c_void,
            window: address(args.window.ptr) as usize as *const c_void,
            window_table: address(args.window_table.ptr) as usize as *const c_void,
            window_starts: address(args.window_starts.ptr) as usize as *const c_void,
            compressed: address(plane.ptr) as usize as *const c_void,
            compressed_table: address(table.ptr) as usize as *const c_void,
            selection: address(selection.ptr) as usize as *const c_void,
            rows: address(args.rows.ptr) as usize as *const c_void,
            positions: address(args.positions.ptr) as usize as *const c_void,
            sink: address(args.sink.ptr) as usize as *const c_void,
            workspace: address(args.workspace.ptr) as usize as *mut c_void,
            workspace_bytes: to_i64(name, args.workspace.numel())?,
            tokens: to_i64(name, tokens)?,
            heads: i64::from(descriptor.heads),
            head_dim: i64::from(descriptor.head_dim),
            window_size: i64::from(descriptor.window),
            page: i64::from(descriptor.page),
            window_pages: to_i64(name, args.window_table.dim(1))?,
            window_record: to_i64(name, descriptor.window_format.record_bytes(head_dim))?,
            window_plane_pages: to_i64(name, args.window.dim(0))?,
            entries: side.map_or(0, |side| i64::from(side.entries)),
            columns: side.map_or(0, |side| i64::from(side.columns)),
            compressed_pages: match side {
                Some(_) => to_i64(name, table.dim(1))?,
                None => 0,
            },
            compressed_record: match side {
                Some(side) => to_i64(name, side.format.record_bytes(head_dim))?,
                None => 0,
            },
            compressed_plane_pages: match side {
                Some(_) => to_i64(name, plane.dim(0))?,
                None => 0,
            },
            multiprocessors: i64::from(stream.device().multiprocessors()?),
            scale: args.scale,
            error: [0; MESSAGE],
        };
        // SAFETY: `LatentAttentionCall` is the call structure this package's entry declares: the
        // declaration is generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_states_the_window_and_the_compressed_side() {
        let attention = LatentAttentionDescriptor::new(
            DType::BF16,
            64,
            512,
            (Format::Mxfp8, 128, 64),
            Some(Compressed {
                format: Format::Nvfp4,
                entries: 32,
                columns: 512,
            }),
        );
        assert_eq!(
            attention.to_string(),
            "dtype_bf16_heads_64_head_dim_512_window_format_mxfp8_window_128_page_64_\
             compressed_nvfp4_entries_32_columns_512_block_16"
        );
        assert_eq!(attention.reduction(), 640);

        let window_only =
            LatentAttentionDescriptor::new(DType::BF16, 16, 512, (Format::Bf16, 128, 64), None);
        assert_eq!(
            window_only.to_string(),
            "dtype_bf16_heads_16_head_dim_512_window_format_bf16_window_128_page_64_\
             compressed_none_block_16"
        );
    }

    #[test]
    fn the_bound_follows_the_entries_one_softmax_reads() {
        let wide = LatentAttentionDescriptor::new(
            DType::BF16,
            64,
            512,
            (Format::Mxfp8, 128, 64),
            Some(Compressed {
                format: Format::Nvfp4,
                entries: 32,
                columns: 512,
            }),
        );
        let narrow = LatentAttentionDescriptor {
            compressed: None,
            ..wide
        };
        let (wide, narrow) = (
            LatentAttention::tolerance(&wide),
            LatentAttention::tolerance(&narrow),
        );
        assert!(narrow.rtol < wide.rtol && wide.rtol < 0.02);
        assert!(narrow.atol < wide.atol);
    }
}
