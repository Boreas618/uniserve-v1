//! Writing the keys and values of the current tokens into the paged KV cache.
//!
//! The cache of one attention type is laid out as
//!
//! ```text
//! key   [pages, page_size, kv_heads, head_dim]   elements of the cache's dtype
//! value [pages, page_size, kv_heads, head_dim]
//! key.initialized, value.initialized  [pages, 1, 1, 1]   u8, one flag per page
//! ```
//!
//! which is the layout `paged_attention` reads (specs/worker.md 9, KV 写入; the current store's
//! `[blocks, block_size, local_kv_heads, head_dim]` with a per-block `initialized` marker). A write
//! scatters one row per token into the slot the scheduler assigned it; a slot of `-1` writes nothing
//! and initializes nothing, so a token that is not persisted leaves the page as it was. A flag is
//! raised when a page receives its first element: the owner of a page zeroes only storage it
//! created, and a reader must not read a page before it is initialized (specs/worker.md 8, "只清零
//! 新存储").
//!
//! An **FP8 cache** is filled by a scaled write instead: the same scatter, with each element divided by
//! the page's own scale and rounded to E4M3 on the way in. That scale is one `[pages, 1, 1, 1]` f32
//! per plane, the same buffer the read side multiplies back, and the write derives it from the values
//! it is about to store rather than receiving it.
//! Splitting the two writes is what keeps the storage decision (which element type the cache holds)
//! out of the scatter that does not care: `kv_write` stays exact and has no scale operand.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::{Stream, address};
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// One compilation of `kv_write`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the keys and values being written.
    pub dtype: DType,
    /// Element type of the cache.
    ///
    /// It is the source's own: this operator moves values, it does not encode them. An FP8 cache is
    /// filled by [`KvWriteScaled`], which divides by the page's scale on the way in, and a launch that
    /// asks this one for a narrower cache is refused rather than writing values no reader could
    /// interpret (a reader of an FP8 plane multiplies by a scale this operator would not have
    /// applied).
    pub kv: DType,
    /// Tokens per page.
    pub page: u32,
    /// Key/value heads this rank holds.
    pub heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Elements of a cache row one program writes.
    pub tile: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("kv", self.kv)
            .field("page", self.page)
            .field("heads", self.heads)
            .field("head_dim", self.head_dim)
            .field("tile", self.tile)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `kv_write`.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[tokens, heads, head_dim]` keys.
    pub key: TensorView,
    /// `[tokens, heads, head_dim]` values.
    pub value: TensorView,
    /// `[pages, page, heads, head_dim]`.
    pub key_cache: TensorView,
    /// `[pages, page, heads, head_dim]`.
    pub value_cache: TensorView,
    /// `[tokens]` i32: the flat slot `page · page_size + offset` of each token, or `-1`.
    pub slots: TensorView,
    /// `[pages, 1, 1, 1]` u8 initialization flags of the key cache.
    pub key_initialized: TensorView,
    /// `[pages, 1, 1, 1]` u8 initialization flags of the value cache.
    pub value_initialized: TensorView,
}

/// The addresses and extents of a write, off unless `UNISERVE_ATTENTION_TRACE` asks for them.
///
/// A write and the read that follows it are separate launches, and the only thing that says they
/// agree about *which* pages hold this layer is the cache address each was handed. A zero attention
/// output that is not a harness fault is the shape of a reader looking at pages nothing wrote, so the
/// two launches print the same field in the same form and the pair can be compared line by line
/// without a device read-back.
fn trace(descriptor: &Descriptor, args: &Args) {
    if std::env::var_os("UNISERVE_ATTENTION_TRACE").is_none_or(|value| value.is_empty()) {
        return;
    }
    eprintln!(
        "kv_write trace: page {} heads {} head_dim {} pdl {} tokens {} | key {} at {:#x} value {} at {:#x}",
        descriptor.page,
        descriptor.heads,
        descriptor.head_dim,
        descriptor.pdl,
        args.key.shape.dim(0),
        args.key.dtype,
        address(args.key.ptr),
        args.value.dtype,
        address(args.value.ptr),
    );
    eprintln!(
        "kv_write trace: key_cache {} at {:#x} pages {} | value_cache {} at {:#x} pages {} | flags at \
         {:#x} and {:#x}, slots at {:#x}",
        args.key_cache.dtype,
        address(args.key_cache.ptr),
        args.key_cache.shape.dim(0),
        args.value_cache.dtype,
        address(args.value_cache.ptr),
        args.value_cache.shape.dim(0),
        address(args.key_initialized.ptr),
        address(args.value_initialized.ptr),
        address(args.slots.ptr),
    );
}

/// Scatter of the current tokens into the paged cache.
pub struct KvWrite;

impl Op for KvWrite {
    const NAME: &'static str = "kv_write";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let heads = u64::from(descriptor.heads);
        let head_dim = u64::from(descriptor.head_dim);
        let row = heads * head_dim;
        // The token count is the first extent of a `[tokens, heads, head_dim]` view; `rows()` is the
        // product of every extent but the last and would count tokens times heads.
        let tokens = args.key.shape.dim(0);
        // The cache is filled in the source's type. An FP8 cache needs the page's scale applied
        // first, which is the scaled write's job: allowing it here would write values that a reader
        // multiplies by a scale they never had applied, which is wrong by orders of magnitude
        // rather than by a rounding.
        if descriptor.kv != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "this writes a {} cache from {} values; an FP8 cache is filled by kv_write_scaled",
                    descriptor.kv, descriptor.dtype
                ),
            ));
        }
        args.key
            .require_shape(Self::NAME, Shape::new(&[tokens, heads, head_dim]))?;
        args.value
            .require_shape(Self::NAME, Shape::new(&[tokens, heads, head_dim]))?;
        if args.key.dtype != descriptor.dtype || args.value.dtype != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} keys and values, found {} and {}",
                    descriptor.dtype, args.key.dtype, args.value.dtype
                ),
            ));
        }
        args.slots
            .require_shape(Self::NAME, Shape::new(&[tokens]))?;
        if args.slots.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 slots, found {}", args.slots.dtype),
            ));
        }

        let page = u64::from(descriptor.page);
        for cache in [&args.key_cache, &args.value_cache] {
            cache.require_rank(Self::NAME, 4)?;
            if cache.dtype != descriptor.kv {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected a {} cache, found {}", descriptor.kv, cache.dtype),
                ));
            }
            if cache.shape.dim(1) != page
                || cache.shape.dim(2) != heads
                || cache.shape.dim(3) != head_dim
            {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "a cache page is {:?}, the descriptor says {page} tokens × {heads} heads × {head_dim}",
                        cache.shape.dims()
                    ),
                ));
            }
        }
        for initialized in [&args.key_initialized, &args.value_initialized] {
            initialized.require_rank(Self::NAME, 4)?;
            // One byte per page, because the plane is one byte per page: the executor's
            // conversion, units and Storage Server all carry `u8`, and `cache_copy_region` already
            // loads and stores these same flags as `u8`. A flag is a byte of the page's plane, and
            // a four-byte store into it would clobber the three pages that follow.
            if initialized.dtype != DType::U8 {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected u8 initialization flags, one byte per page, found {}",
                        initialized.dtype
                    ),
                ));
            }
        }

        trace(descriptor, &args);
        let mut arguments = Arguments::new();
        arguments
            .view(args.key)
            .view(args.value)
            .view(args.key_cache)
            .view(args.value_cache)
            .view(args.slots)
            .view(args.key_initialized)
            .view(args.value_initialized)
            .u64(tokens)
            .u32(descriptor.page)
            .u32(descriptor.heads)
            .u32(descriptor.head_dim)
            .flag(descriptor.pdl);
        kernel.launch(
            LaunchGrid::new([tokens, row.div_ceil(u64::from(descriptor.tile)), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        // The pages are read-modify-write containers: the new kernel writes its twin pages, and the
        // comparison sees the untouched rows as well as the written ones.
        KvWriteArgs {
            key_cache: twins.in_place(args.key_cache, stream),
            value_cache: twins.in_place(args.value_cache, stream),
            key_initialized: twins.in_place(args.key_initialized, stream),
            value_initialized: twins.in_place(args.value_initialized, stream),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // A scatter of values, possibly rounded once into the cache's element type by the kernel
        // itself; two implementations that use the same rounding agree bit for bit.
        Tolerance::exact()
    }
}

/// The argument struct, named for the shadow implementation's clarity.
type KvWriteArgs = Args;

/// The write that derives a page's scale instead of being given one. A write that is handed a scale
/// can agree with another to the encoding's own bound; this one *decides* the scale from the values it
/// is about to store, grows the page's scale when it must, and
/// rewrites the page so that rows written earlier keep their decoded values. That is a different
/// launch over different operands - the page is read and rewritten, the scale is an output as well as
/// an input - and its rule is the reference's (`uniserve/prefix_cache/mha.py`'s quantizing write):
/// `proposed = max|v| / 448`, `updated = max(scale, proposed)` for an initialized page and `proposed`
/// otherwise, the page re-encoded from the old scale to the new when it grew, then the new rows stored
/// at `updated`. It is a separate operator rather than a flag on the write that takes a scale,
/// because a descriptor must name one program: a field selecting between "apply the scale" and
/// "derive the scale" would make one key cover two kernels and one tolerance cover two error sources.
///
/// **The scale is derived, never supplied.** The reference derives it from the values it is about to
/// store (`uniserve/prefix_cache/mha.py`'s quantizing write), and an FP8 cache takes its scales per
/// block: a checkpoint's per-tensor `k_scale`/`v_scale` floats have no reader here, and a block whose
/// initialized flag is clear discards whatever its scale buffer held, so such a scalar could not
/// influence it even as an initial value. A per-tensor scale would be a different layout with a
/// different reader, not a mode of this operator.
///
/// **A block is a page.** The scale this operator maintains is one number per page, indexed by the
/// page of each token's slot, and a page holds [`page`](Self::page) tokens - the unit this write
/// fills and `paged_attention` reads. That is the layout the reference states
/// (`shape = (num_blocks, block_size, heads, head_dim)`, `scale = (num_blocks, 1, 1, 1)` over one
/// stream), where a block *is* a page; a deployment that laid one scale over several pages would need
/// a block field here and a different index in every reader, and that is a change of layout rather
/// than a parameter.
///
/// The cache's element type is not a field: an FP8 page is E4M3 by the layout every reader agrees
/// on, so no key may ask for another one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScaledWriteDescriptor {
    /// Element type of the keys and values being written: a full-precision accumulation's output.
    pub dtype: DType,
    /// Element type of the page's scale.
    pub scale: DType,
    /// Tokens per page, which is tokens per block: one scale covers one page.
    pub page: u32,
    /// Key/value heads this rank holds.
    pub heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Elements of a row one program re-encodes in one pass.
    pub block: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for ScaledWriteDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("scale", self.scale)
            .field("page", self.page)
            .field("heads", self.heads)
            .field("head_dim", self.head_dim)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `kv_write_scaled`.
#[derive(Clone, Copy)]
pub struct ScaledWriteArgs {
    /// `[tokens, heads · head_dim]` the layer's rotated keys or values, in full precision.
    pub source: TensorView,
    /// `[tokens]` i32: the flat slot `page · page_size + offset` of each token, or `-1`.
    pub slots: TensorView,
    /// `[pages, 1, 1, 1]` f32: each page's scale, read for the growth rule and written back.
    pub scales: TensorView,
    /// `[pages, page, heads, head_dim]` E4M3: the cache, read and rewritten for a page whose scale
    /// grew and written at the new rows for every other.
    pub cache: TensorView,
    /// `[pages, 1, 1, 1]` u8 initialization flags, written with the scale in the same launch.
    pub initialized: TensorView,
}

/// The scale-deriving write into an FP8 cache.
pub struct KvWriteScaled;

impl Op for KvWriteScaled {
    const NAME: &'static str = "kv_write_scaled";

    type Descriptor = ScaledWriteDescriptor;
    type Args<'a> = ScaledWriteArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let heads = u64::from(descriptor.heads);
        let head_dim = u64::from(descriptor.head_dim);
        let row = heads * head_dim;
        let tokens = args.source.rows();

        if !matches!(descriptor.dtype, DType::BF16 | DType::F32) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected a bf16 or f32 source, the descriptor says {}",
                    descriptor.dtype
                ),
            ));
        }
        if args.source.dtype != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} source, found {}",
                    descriptor.dtype, args.source.dtype
                ),
            ));
        }
        args.source
            .require_shape(Self::NAME, Shape::new(&[tokens, row]))?;
        // A page whose scale would be derived from rows it does not own is the wrong cache, so the
        // source's row width is stated rather than inferred from a reshape's remainder.
        args.source
            .require_row(Self::NAME, descriptor.dtype, args.source.shape.dim(1))?;
        args.slots
            .require_shape(Self::NAME, Shape::new(&[tokens]))?;
        if args.slots.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 slots, found {}", args.slots.dtype),
            ));
        }

        let page = u64::from(descriptor.page);
        args.cache.require_rank(Self::NAME, 4)?;
        if args.cache.dtype != DType::F8E4M3 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected an e4m3 cache, found {}", args.cache.dtype),
            ));
        }
        if args.cache.shape.dim(1) != page
            || args.cache.shape.dim(2) != heads
            || args.cache.shape.dim(3) != head_dim
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a cache page is {:?}, the descriptor says {page} tokens × {heads} heads × {head_dim}",
                    args.cache.shape.dims()
                ),
            ));
        }

        // The scale and the flag are the page's, so both are exactly one entry per page: a shorter
        // tensor would derive a scale for a page it cannot then record.
        let pages = args.cache.shape.dim(0);
        args.scales
            .require_shape(Self::NAME, Shape::new(&[pages, 1, 1, 1]))?;
        if args.scales.dtype != descriptor.scale {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} page scales, found {}",
                    descriptor.scale, args.scales.dtype
                ),
            ));
        }
        if descriptor.scale != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                "a page scale is accumulated and compared in f32; another type is a different rule",
            ));
        }
        args.initialized
            .require_shape(Self::NAME, Shape::new(&[pages, 1, 1, 1]))?;
        // One byte per page, as in `kv_write`: the flag is the page's own byte of its plane, and a
        // four-byte store would clobber the three pages after it.
        if args.initialized.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected u8 initialization flags, one byte per page, found {}",
                    args.initialized.dtype
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.source)
            .view(args.slots)
            .view(args.scales)
            .view(args.cache)
            .view(args.initialized)
            .u64(tokens)
            .u32(descriptor.page)
            .u32(descriptor.heads)
            .u32(descriptor.head_dim)
            .flag(descriptor.pdl);
        // One program per page, not per token: the page's maximum is over every row it holds, the
        // re-encode is over every row already written, and a per-token grid could not compute either
        // without a second pass. A page no token of this launch touches still re-encodes nothing and
        // costs one program that reads its flag and returns.
        kernel.launch(
            LaunchGrid::new([pages, row.div_ceil(u64::from(descriptor.block)), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        // Every page is read-modify-write here, and so is its scale: the comparison must see the
        // rewritten rows and the grown scale, not only the rows this launch stored.
        ScaledWriteArgs {
            scales: twins.in_place(args.scales, stream),
            cache: twins.in_place(args.cache, stream),
            initialized: twins.in_place(args.initialized, stream),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // The encoding's own bound, in the page's scaled units - with one
        // difference this operator owns: it also *decides* the scale. Two implementations given the
        // same source and the same rule derive the same maximum and therefore the same scale, so the
        // bound stays the encoding's. Two implementations that derive different scales hold two
        // different caches, and that is a wrong cache rather than a rounded one: the check for it is
        // the end-to-end one (the same token ids at the same weights), not a wider tolerance here.
        Tolerance::new(2f32.powi(-6), 2f32.powi(-3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scaled() -> ScaledWriteDescriptor {
        ScaledWriteDescriptor {
            dtype: DType::BF16,
            scale: DType::F32,
            page: 16,
            heads: 8,
            head_dim: 128,
            block: 1024,
            pdl: false,
        }
    }

    #[test]
    fn the_scaled_write_names_its_page_and_its_scale_in_the_key() {
        assert_eq!(
            crate::descriptor::key(&scaled()),
            "dtype_bf16_scale_f32_page_16_heads_8_head_dim_128_block_1024_pdl_false"
        );
        // The page is the block the scale covers, so a different page size is a different program:
        // an implementation that derived one scale over another geometry would store a plausible
        // wrong number everywhere, which is the failure this key exists to prevent.
        let wider = ScaledWriteDescriptor {
            page: 32,
            ..scaled()
        };
        assert_ne!(
            crate::descriptor::key(&scaled()),
            crate::descriptor::key(&wider)
        );
    }

    #[test]
    fn the_scaled_write_bounds_the_encoding_and_says_a_wrong_scale_is_not_a_rounding() {
        let tolerance = <KvWriteScaled as Op>::tolerance(&scaled());
        assert_eq!(tolerance.rtol, 2f32.powi(-3));
        assert_eq!(tolerance.atol, 2f32.powi(-6));
        assert!(tolerance.allows(0.0, 2f32.powi(-6)));
        assert!(!tolerance.allows(0.0, 2f32.powi(-5)));
    }
}
