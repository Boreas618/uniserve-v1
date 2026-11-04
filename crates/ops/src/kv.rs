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
    /// Element type of the cache: the same, or FP8 for an FP8 cache.
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

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
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
