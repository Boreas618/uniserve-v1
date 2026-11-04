//! Attention over the paged cache: write the batch's keys and values, then attend.
//!
//! One call covers both halves because the batch's tokens are written exactly once and read in the
//! same step: the scheduler's mechanism assigns each token a slot (`-1` for a token it does not
//! persist), [`ops::kv::KvWrite`] scatters the projections into the pages, and
//! [`ops::attention::PagedAttention`] reads the pages each row addresses (specs/worker.md 9, "KV
//! 写入" and "注意力"). Both operators are compiled per geometry, so the descriptor pins the element
//! types, the head and page geometry and the mask variant, while the batch, the lengths, the scale
//! and the window travel in the call.

use ops::attention::{Args, Descriptor, Mask, PagedAttention, Phase};
use ops::kv::{self, Descriptor as KvDescriptor, KvWrite};
use ops::{Op, PhaseKind, TensorView};

use crate::capability::{KvView, TokenBatch};
use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::{PDL, kv_tile};

/// Which implementation reads the keys and values.
///
/// The backend names the program the epoch record fills the operator with (`attention_backend`);
/// the layer states which attention its model composes so a runner can refuse a block/backend
/// mismatch before serving, not during a forward.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttentionBackend {
    /// FlashAttention-4's dense kernel through the CuTe DSL program.
    Dense,
    /// The sparse attention of the H3 diffusion block (VSA).
    Sparse,
    /// FP32 attention through a GEMM and a softmax program.
    Fp32,
    /// Paged attention over the prefix cache, read and written per token.
    Paged,
}

/// Paged attention over a cache's planes.
pub struct Attention {
    backend: AttentionBackend,
    head_dim: u32,
    q_heads: u32,
    kv_heads: u32,
    page_size: u32,
    scale: f32,
    window: Option<u32>,
    max_kv_len: u32,
    causal: bool,
}

impl Attention {
    /// Paged attention of `head_dim` elements per head over `page_size`-token pages.
    ///
    /// `max_kv_len` is the longest key reduction the captured bucket serves: it enters the
    /// descriptor, and with it the tolerance bound. `window` is the inclusive history window, when
    /// the model has one.
    pub fn paged(
        head_dim: u32,
        q_heads: u32,
        kv_heads: u32,
        page_size: u32,
        max_kv_len: u32,
        window: Option<u32>,
        scale: f32,
    ) -> Result<Self> {
        if head_dim == 0 || q_heads == 0 || kv_heads == 0 || page_size == 0 {
            return Err(Error::shape(
                "Attention",
                "a head, a page and both head counts must be nonzero".to_owned(),
            ));
        }
        if !q_heads.is_multiple_of(kv_heads) {
            return Err(Error::shape(
                "Attention",
                format!("{q_heads} query heads do not group {kv_heads} key heads"),
            ));
        }
        if max_kv_len < page_size {
            return Err(Error::shape(
                "Attention",
                format!("a {max_kv_len}-key bound is shorter than a {page_size}-token page"),
            ));
        }
        Ok(Self {
            backend: AttentionBackend::Paged,
            head_dim,
            q_heads,
            kv_heads,
            page_size,
            scale,
            window,
            max_kv_len,
            causal: true,
        })
    }

    /// The backend this layer expects.
    pub const fn backend(&self) -> AttentionBackend {
        self.backend
    }

    /// The elements per head.
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// The longest key reduction the descriptor serves.
    pub const fn max_kv_len(&self) -> u32 {
        self.max_kv_len
    }

    /// Write `key` and `value` into `planes`, then attend.
    ///
    /// The argument list is the design's: the batch, the cache, the layer's place in it, the three
    /// projections, the output and the program's workspace.
    ///
    /// `query` and `out` are `[tokens, q_heads, head_dim]`, `key` and `value`
    /// `[tokens, kv_heads, head_dim]`; `workspace` is `[bytes]` of the attention program's transient
    /// scratch (512 MiB for the trtllm-gen program), zeroed by the executor before its first use.
    #[allow(
        clippy::too_many_arguments,
        reason = "the design's attention call carries the batch, the cache, the layer index and the \
                  three projections; grouping them into a struct would name the same values twice"
    )]
    pub fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        group: usize,
        layer: usize,
        query: TensorView,
        key: TensorView,
        value: TensorView,
        out: TensorView,
        workspace: TensorView,
    ) -> Result<()> {
        if self.backend != AttentionBackend::Paged {
            return Err(Error::Unsupported(format!(
                "Attention: the {:?} backend does not read the paged cache",
                self.backend
            )));
        }
        let planes = cache.planes(group, layer)?;
        self.write(f, batch, planes, key, value)?;
        self.attend(f, batch, cache, group, query, out, workspace)
    }

    /// Write the batch's keys and values into `planes`; slots of `-1` write nothing.
    pub fn write(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        planes: &crate::capability::KvPlanes,
        key: TensorView,
        value: TensorView,
    ) -> Result<()> {
        let head_dim = u64::from(self.head_dim);
        let heads = u64::from(self.kv_heads);
        for (name, view) in [("key", key), ("value", value)] {
            if view.shape.rank() != 3 || view.shape.dim(1) != heads || view.shape.dim(2) != head_dim
            {
                return Err(Error::shape(
                    "Attention::write",
                    format!(
                        "{name} is {:?}, [tokens, {heads}, {head_dim}] expected",
                        view.shape.dims()
                    ),
                ));
            }
        }
        if key.shape.dim(0) != batch.tokens() || value.shape.dim(0) != batch.tokens() {
            return Err(Error::shape(
                "Attention::write",
                format!(
                    "the batch has {} tokens but the projections have {} and {}",
                    batch.tokens(),
                    key.shape.dim(0),
                    value.shape.dim(0)
                ),
            ));
        }
        let descriptor = KvDescriptor {
            dtype: key.dtype,
            kv: planes.key.dtype,
            page: self.page_size,
            heads: self.kv_heads,
            head_dim: self.head_dim,
            tile: kv_tile(self.head_dim),
            pdl: PDL,
        };
        let args = kv::Args {
            key,
            value,
            key_cache: planes.key,
            value_cache: planes.value,
            slots: batch.slot_mapping,
            key_initialized: planes.key_initialized,
            value_initialized: planes.value_initialized,
        };
        let kernel = f.kernels().op::<KvWrite>(&descriptor)?;
        KvWrite::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
    }

    /// Read the cache and write this batch's attended output.
    #[allow(
        clippy::too_many_arguments,
        reason = "the design's attention call carries the batch, the cache, the group index, the \
                  query, the output and the program's workspace"
    )]
    pub fn attend(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: &KvView<'_>,
        group: usize,
        query: TensorView,
        out: TensorView,
        workspace: TensorView,
    ) -> Result<()> {
        let planes = cache.planes(group, 0)?;
        let group_view = cache.group(group)?;
        if group_view.page_size != self.page_size || batch.page_size != self.page_size {
            return Err(Error::shape(
                "Attention::attend",
                format!(
                    "the layer is built for {} tokens per page, the cache and batch have {} and {}",
                    self.page_size, group_view.page_size, batch.page_size
                ),
            ));
        }
        let phase = match batch.phase {
            PhaseKind::Prefill => Phase::Prefill,
            PhaseKind::Decode => Phase::Decode,
            other => {
                return Err(Error::Unsupported(format!(
                    "Attention: phase {other} is not an attention phase"
                )));
            }
        };
        let head_dim = u64::from(self.head_dim);
        let q_heads = u64::from(self.q_heads);
        for (name, view) in [("query", query), ("out", out)] {
            if view.shape.rank() != 3
                || view.shape.dim(1) != q_heads
                || view.shape.dim(2) != head_dim
            {
                return Err(Error::shape(
                    "Attention::attend",
                    format!(
                        "{name} is {:?}, [tokens, {q_heads}, {head_dim}] expected",
                        view.shape.dims()
                    ),
                ));
            }
        }
        let descriptor = Descriptor {
            phase,
            query: query.dtype,
            key_value: planes.key.dtype,
            output: out.dtype,
            head_dim: self.head_dim,
            page_size: self.page_size,
            q_heads: self.q_heads,
            kv_heads: self.kv_heads,
            mask: Mask::Causal,
            windowed: self.window.is_some(),
            max_kv_len: self.max_kv_len,
        };
        // A quantized cache divides each page by its own FP32 scale; the page layout of those
        // scales is the FP8 cache step's, so a quantized cache is refused here rather than read
        // without them.
        let (key_scale, value_scale) = match (planes.key.dtype, group_view.scales) {
            (ops::DType::F8E4M3, Some(_)) => {
                return Err(Error::Unsupported(
                    "Attention: an FP8 cache's per-page key and value scales are the FP8 KV-page \
                     step; this attention reads only caches without them"
                        .to_owned(),
                ));
            }
            _ => (None, None),
        };
        let args = Args {
            query,
            key: planes.key,
            value: planes.value,
            key_scale,
            value_scale,
            page_table: group_view.block_tables,
            seq_lens: batch.seq_lens,
            cum_seq_lens_q: batch.query_offsets,
            out,
            workspace,
            scale: self.scale,
            window_left: self.window.map_or(-1, |window| window as i32),
            causal: self.causal,
        };
        let kernel = f.kernels().op::<PagedAttention>(&descriptor)?;
        PagedAttention::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
    }
}

/// The bytes the trtllm-gen attention program needs as its workspace.
///
/// The program's `workspace_size` argument (specs/worker.md 9); the executor carves it from its
/// scratch pool and zeroes it once (specs/worker.md 5.4, [D4]).
pub const TRTLLM_WORKSPACE_BYTES: u64 = 512 << 20;
