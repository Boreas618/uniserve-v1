//! Writing the keys and values of the current tokens into the paged KV cache.
//!
//! The cache of one attention type is laid out as
//!
//! ```text
//! key   [pages, page_size, kv_heads, head_dim]   elements of the cache's dtype
//! value [pages, page_size, kv_heads, head_dim]
//! key.initialized, value.initialized  [pages, 1, 1, 1]   i32, one flag per page
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
//! An **FP8 cache** is filled by `kv_quantize` instead: the same scatter, with each element divided by
//! the page's own scale and rounded to E4M3 on the way in. The scale is the deployment's per-page
//! static value (one `[pages, 1, 1, 1]` f32 per plane), supplied as an input rather than derived, so
//! the write needs no maximum and no second pass, and the read side multiplies it back per page.
//! Splitting the two writes is what keeps the storage decision (which element type the cache holds)
//! out of the scatter that does not care: `kv_write` stays exact and has no scale operand.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
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
    /// filled by [`KvQuantize`], which divides by the page's scale on the way in, and a launch that
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
    /// `[pages, 1, 1, 1]` i32 initialization flags of the key cache.
    pub key_initialized: TensorView,
    /// `[pages, 1, 1, 1]` i32 initialization flags of the value cache.
    pub value_initialized: TensorView,
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
        let tokens = args.key.rows();
        // The cache is filled in the source's type. An FP8 cache needs the page's scale applied
        // first, which is `kv_quantize`'s job: allowing it here would write values that a reader
        // multiplies by a scale they never had applied, which is wrong by orders of magnitude
        // rather than by a rounding.
        if descriptor.kv != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "this writes a {} cache from {} values; an FP8 cache is filled by kv_quantize",
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
            if initialized.dtype != DType::I32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected i32 initialization flags, found {}",
                        initialized.dtype
                    ),
                ));
            }
        }

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

/// One compilation of `kv_quantize`.
///
/// The cache's element type is not a field: an FP8 cache is E4M3 by the layout every reader agrees
/// on, so the storage type is a constant of this operator rather than a variant of it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QuantizeDescriptor {
    /// Element type of the keys and values being written: a full-precision accumulation's output.
    pub dtype: DType,
    /// Element type of the page's scale.
    pub scale: DType,
    /// Tokens per page.
    pub page: u32,
    /// Key/value heads this rank holds.
    pub heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Elements of a cache row one program writes.
    pub block: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for QuantizeDescriptor {
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

/// The arguments of `kv_quantize`.
#[derive(Clone, Copy)]
pub struct QuantizeArgs {
    /// `[tokens, heads · head_dim]` the layer's rotated keys or values.
    pub source: TensorView,
    /// `[tokens]` i32: the flat slot `page · page_size + offset` of each token, or `-1`.
    pub slots: TensorView,
    /// `[pages, 1, 1, 1]` f32: each page's static scale, the same tensor the read side multiplies by.
    pub scales: TensorView,
    /// `[pages, page, heads, head_dim]` E4M3: the cache the launch writes.
    pub out: TensorView,
    /// `[pages, 1, 1, 1]` i32 initialization flags of this cache.
    pub initialized: TensorView,
}

/// The quantizing write into an FP8 cache.
pub struct KvQuantize;

impl Op for KvQuantize {
    const NAME: &'static str = "kv_quantize";

    type Descriptor = QuantizeDescriptor;
    type Args<'a> = QuantizeArgs;

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

        // The source is a full-precision layer output; a cache in a narrower type is what this
        // operator produces, never what it reads.
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

        args.out.require_rank(Self::NAME, 4)?;
        if args.out.dtype != DType::F8E4M3 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected an e4m3 cache, found {}", args.out.dtype),
            ));
        }
        let page = u64::from(descriptor.page);
        if args.out.shape.dim(1) != page
            || args.out.shape.dim(2) != heads
            || args.out.shape.dim(3) != head_dim
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a cache page is {:?}, the descriptor says {page} tokens × {heads} heads × {head_dim}",
                    args.out.shape.dims()
                ),
            ));
        }

        let pages = args.out.shape.dim(0);
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
        args.initialized
            .require_shape(Self::NAME, Shape::new(&[pages, 1, 1, 1]))?;
        if args.initialized.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected i32 initialization flags, found {}",
                    args.initialized.dtype
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.source)
            .view(args.slots)
            .view(args.scales)
            .view(args.out)
            .view(args.initialized)
            .u64(tokens)
            .u32(descriptor.page)
            .u32(descriptor.heads)
            .u32(descriptor.head_dim)
            .flag(descriptor.pdl);
        // One block per (token, element block of its row): a token's page is `slot / page_size`, so
        // every program derives its own page from its slot and no second pass initializes anything.
        kernel.launch(
            LaunchGrid::new([tokens, row.div_ceil(u64::from(descriptor.block)), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        // The pages are read-modify-write containers, as in `kv_write`: the flag and the untouched
        // rows are part of what the comparison sees.
        QuantizeArgs {
            out: twins.in_place(args.out, stream),
            initialized: twins.in_place(args.initialized, stream),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // This operator is **lossy by contract**, so 7.3's accumulation bound does not apply to it:
        // it stores each element as E4M3 after dividing by the page's scale, and no order of the same
        // arithmetic would give the source back. The bound is the encoding's own, stated per unit of
        // that scale because the scale is supplied by the deployment and is not part of the error (a
        // wrong scale is a wrong cache, not a rounded one): an E4M3 value is within `2^-3` relative
        // of the scaled magnitude it rounds, and a subnormal is within `2^-6` of the scale, which is
        // an absolute term in the same scaled units.
        Tolerance::new(2f32.powi(-6), 2f32.powi(-3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> QuantizeDescriptor {
        QuantizeDescriptor {
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
    fn the_quantizing_write_is_compiled_for_the_page_and_the_head_geometry() {
        assert_eq!(
            crate::descriptor::key(&descriptor()),
            "dtype_bf16_scale_f32_page_16_heads_8_head_dim_128_block_1024_pdl_false"
        );
        // The cache's element type is not a field, so no key can ask for another one.
        let f32_source = QuantizeDescriptor {
            dtype: DType::F32,
            ..descriptor()
        };
        assert_ne!(
            crate::descriptor::key(&descriptor()),
            crate::descriptor::key(&f32_source)
        );
    }

    #[test]
    fn the_bound_is_the_encoding_s_own_and_not_an_accumulation_bound() {
        let tolerance = <KvQuantize as Op>::tolerance(&descriptor());
        // Two terms in the page's scaled units: the relative error of an E4M3 value, and the
        // subnormal floor.
        assert_eq!(tolerance.rtol, 2f32.powi(-3));
        assert_eq!(tolerance.atol, 2f32.powi(-6));
        // The subnormal floor is what makes the bound hold near zero, so it must be there.
        assert!(tolerance.allows(0.0, 2f32.powi(-6)));
        assert!(!tolerance.allows(0.0, 2f32.powi(-5)));
        // And the relative term is the one that holds at full scale.
        assert!(tolerance.allows(1.0, 1.0 + 2f32.powi(-3)));
        assert!(!tolerance.allows(1.0, 1.0 + 2f32.powi(-2)));
    }
}
