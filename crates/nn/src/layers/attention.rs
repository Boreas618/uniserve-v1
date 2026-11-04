//! Attention over the paged cache: write the batch's keys and values, then attend.
//!
//! One call covers both halves because the batch's tokens are written exactly once and read in the
//! same step: the scheduler's mechanism assigns each token a slot (`-1` for a token it does not
//! persist), [`ops::kv::KvWrite`] scatters the projections into the pages, and
//! [`ops::attention::PagedAttention`] reads the pages each row addresses (specs/worker.md 9, "KV
//! 写入" and "注意力"). Both operators are compiled per geometry, so the descriptor pins the element
//! types, the head and page geometry and the mask variant, while the batch, the lengths, the scale
//! and the window travel in the call.

use ops::attention::dense::{DenseArgs, DenseAttention, DenseDescriptor};
use ops::attention::sparse::{SparseArgs, SparseAttention, SparseDescriptor};
use ops::attention::{Args, Descriptor, Mask, PagedAttention, Phase};
use ops::kv::{self, Descriptor as KvDescriptor, KvWrite};
use ops::{DType, DevicePtr, PhaseKind, Shape, TensorView};

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
    /// Rows of one query tile and of one key block, for the sparse backend.
    block: u32,
    /// The largest key extent the sparse module serves.
    max_keys: u32,
    /// Whether the sparse call writes the log-sum-exp normalizer.
    lse: bool,
}

impl Attention {
    /// Paged attention of `head_dim` elements per head over `page_size`-token pages.
    ///
    /// `max_kv_len` is the longest key reduction the deployment serves (40960 for the Qwen3-MoE
    /// slice): it enters the descriptor and with it the tolerance bound, so a deployment that
    /// serves longer sequences needs another module, not another launch. `window` is the inclusive history window, when
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
            block: 0,
            max_keys: 0,
            lse: false,
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

    /// Dense attention over a packed sequence: the language encoder, a cross-attention, the video
    /// block's attention without the cache.
    ///
    /// `max_seq_len` is the longest key sequence the deployment serves and enters the descriptor
    /// (one value per deployment, like the paged backend's bound). `causal` states the mask the
    /// model's attention declares; `window` is an inclusive history window, which the causal mask
    /// requires.
    pub fn dense(
        head_dim: u32,
        q_heads: u32,
        kv_heads: u32,
        max_seq_len: u32,
        causal: bool,
        window: Option<u32>,
        scale: f32,
    ) -> Result<Self> {
        Self::dense_of(
            AttentionBackend::Dense,
            head_dim,
            q_heads,
            kv_heads,
            max_seq_len,
            causal,
            window,
            scale,
        )
    }

    /// The FP32 dense attention of the `ref2va` audio encoder and the sparse compression branch:
    /// the same call, with cuBLASLt SGEMM and a softmax kernel inside the package.
    pub fn fp32(
        head_dim: u32,
        q_heads: u32,
        kv_heads: u32,
        max_seq_len: u32,
        scale: f32,
    ) -> Result<Self> {
        // The FP32 paths read a whole sequence at once, so their mask is dense.
        Self::dense_of(
            AttentionBackend::Fp32,
            head_dim,
            q_heads,
            kv_heads,
            max_seq_len,
            false,
            None,
            scale,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "a dense attention is named by the mask's, the head geometry and the bound's values"
    )]
    fn dense_of(
        backend: AttentionBackend,
        head_dim: u32,
        q_heads: u32,
        kv_heads: u32,
        max_seq_len: u32,
        causal: bool,
        window: Option<u32>,
        scale: f32,
    ) -> Result<Self> {
        if head_dim == 0 || q_heads == 0 || kv_heads == 0 || max_seq_len == 0 {
            return Err(Error::shape(
                "Attention::dense",
                "a head, a sequence bound and both head counts must be nonzero".to_owned(),
            ));
        }
        if !q_heads.is_multiple_of(kv_heads) {
            return Err(Error::shape(
                "Attention::dense",
                format!("{q_heads} query heads do not group {kv_heads} key heads"),
            ));
        }
        if window.is_some() && !causal {
            return Err(Error::shape(
                "Attention::dense",
                "a history window reads the causal mask only".to_owned(),
            ));
        }
        if !scale.is_finite() || scale == 0.0 {
            return Err(Error::shape(
                "Attention::dense",
                format!("a softmax scale of {scale}"),
            ));
        }
        Ok(Self {
            backend,
            head_dim,
            q_heads,
            kv_heads,
            page_size: 0,
            scale,
            window,
            max_kv_len: max_seq_len,
            causal,
            block: 0,
            max_keys: 0,
            lse: false,
        })
    }

    /// Video Sparse Attention: query tiles of `block` rows read the key blocks the selection names.
    ///
    /// `max_keys` is the longest key extent the deployment serves; `lse` asks for the log-sum-exp
    /// normalizer of every query row.
    pub fn sparse(
        block: u32,
        head_dim: u32,
        heads: u32,
        max_keys: u32,
        lse: bool,
        scale: f32,
    ) -> Result<Self> {
        if block == 0 || head_dim == 0 || heads == 0 || max_keys == 0 {
            return Err(Error::shape(
                "Attention::sparse",
                "a tile, a head, a key bound and a head count must be nonzero".to_owned(),
            ));
        }
        if !max_keys.is_multiple_of(block) {
            return Err(Error::shape(
                "Attention::sparse",
                format!("a {max_keys}-key bound is not whole {block}-row tiles"),
            ));
        }
        if !scale.is_finite() || scale == 0.0 {
            return Err(Error::shape(
                "Attention::sparse",
                format!("a softmax scale of {scale}"),
            ));
        }
        Ok(Self {
            backend: AttentionBackend::Sparse,
            head_dim,
            q_heads: heads,
            kv_heads: heads,
            page_size: 0,
            scale,
            window: None,
            max_kv_len: max_keys,
            causal: false,
            block,
            max_keys,
            lse,
        })
    }

    /// Dense attention from packed queries to packed keys: `query [q_tokens, q_heads, head_dim]`,
    /// `key`/`value [kv_tokens, kv_heads, head_dim]`, `out [q_tokens, q_heads, head_dim]`.
    ///
    /// `cu_seq_lens_q`/`cu_seq_lens_k` are the `[sequences + 1]` i32 row offsets the scheduler
    /// writes, given together or both absent. Absent means **one sequence**, `[0, q_tokens]` and
    /// `[0, kv_tokens]`: a decoder that runs one sequence per tile owns no table, and the launch is
    /// the one it would make with that two-element table — the tables' addresses travel to the
    /// program as zero, which a package reads as the single-sequence offsets. A batch that packs
    /// several sequences always passes both, and giving one without the other is refused here
    /// rather than left to a program.
    ///
    /// Serves both the `Dense` and the `Fp32` backend; the element types the views carry decide the
    /// descriptor, and the package behind it decides the tile and split heuristics.
    #[allow(
        clippy::too_many_arguments,
        reason = "the call's operands and its two offset tables are the design's argument list"
    )]
    pub fn attend_dense(
        &self,
        f: &Forward,
        query: TensorView,
        key: TensorView,
        value: TensorView,
        cu_seq_lens_q: Option<TensorView>,
        cu_seq_lens_k: Option<TensorView>,
        out: TensorView,
    ) -> Result<()> {
        if !matches!(
            self.backend,
            AttentionBackend::Dense | AttentionBackend::Fp32
        ) {
            return Err(Error::Unsupported(format!(
                "Attention::attend_dense: the {:?} backend is not a dense attention",
                self.backend
            )));
        }
        let descriptor = DenseDescriptor {
            query: query.dtype,
            key_value: key.dtype,
            output: out.dtype,
            head_dim: self.head_dim,
            q_heads: self.q_heads,
            kv_heads: self.kv_heads,
            mask: if self.causal {
                Mask::Causal
            } else {
                Mask::Dense
            },
            windowed: self.window.is_some(),
            max_kv_len: self.max_kv_len,
        };
        let (cu_seq_lens_q, cu_seq_lens_k) = single_sequence(cu_seq_lens_q, cu_seq_lens_k)?;
        let args = DenseArgs {
            query,
            key,
            value,
            cu_seq_lens_q,
            cu_seq_lens_k,
            out,
            scale: self.scale,
            window_left: self.window.map_or(-1, |window| window as i32),
            causal: self.causal,
        };

        f.operate::<DenseAttention>(&descriptor, args)
    }

    /// Video Sparse Attention: `query [query_tiles · block, heads, head_dim]`,
    /// `key`/`value [key_blocks · block, heads, head_dim]`, the selection `indices [heads,
    /// query_tiles, selected]` i32 and its per-tile `counts [heads, query_tiles]` i32, the valid rows
    /// of each key block `valid [key_blocks]` i32, the optional normalizer and `out`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the selection is three device-written tables and the normalizer is a fourth \
                  optional output; grouping them would name the same values twice"
    )]
    pub fn attend_sparse(
        &self,
        f: &Forward,
        query: TensorView,
        key: TensorView,
        value: TensorView,
        indices: TensorView,
        counts: TensorView,
        valid: TensorView,
        lse: Option<TensorView>,
        out: TensorView,
    ) -> Result<()> {
        if self.backend != AttentionBackend::Sparse {
            return Err(Error::Unsupported(format!(
                "Attention::attend_sparse: the {:?} backend is not the sparse attention",
                self.backend
            )));
        }
        match (self.lse, lse) {
            (true, None) => {
                return Err(Error::shape(
                    "Attention::attend_sparse",
                    "the layer writes a normalizer and the call passes none".to_owned(),
                ));
            }
            (false, Some(_)) => {
                return Err(Error::shape(
                    "Attention::attend_sparse",
                    "the call passes a normalizer the layer does not write".to_owned(),
                ));
            }
            _ => {}
        }
        let descriptor = SparseDescriptor {
            block: self.block,
            dtype: query.dtype,
            output: out.dtype,
            head_dim: self.head_dim,
            lse: self.lse,
            keys: self.max_keys,
        };
        let args = SparseArgs {
            query,
            key,
            value,
            indices,
            counts,
            valid,
            lse,
            out,
            scale: self.scale,
        };

        f.operate::<SparseAttention>(&descriptor, args)
    }

    /// The elements between consecutive rows of the query a sparse call reads.
    pub const fn block(&self) -> u32 {
        self.block
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

        f.operate::<KvWrite>(&descriptor, args)
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
        // An FP8 plane stores each page divided by its own FP32 scale; the plane's element type
        // decides whether the call carries one, so a mismatch in either direction is an argument
        // error rather than a silently unscaled read. The scales are this layer's, so two layers of
        // one group reach the launch with their own.
        let key_scale = page_scale(planes.key, planes.key_scale)?;
        let value_scale = page_scale(planes.value, planes.value_scale)?;
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

        f.operate::<PagedAttention>(&descriptor, args)
    }
}

/// The offset tables a dense call packs by, or the absent pair that means one sequence.
///
/// One sequence needs no table — its offsets are `[0, tokens]` by construction — so both tables are
/// absent together and the call carries a zero address and the shape a two-element table has. The
/// program reads a zero address as those offsets, and nothing on the host writes anything.
fn single_sequence(
    queries: Option<TensorView>,
    keys: Option<TensorView>,
) -> Result<(TensorView, TensorView)> {
    match (queries, keys) {
        (Some(queries), Some(keys)) => Ok((queries, keys)),
        (None, None) => {
            let absent = TensorView::new(DevicePtr::new(0), DType::I32, Shape::new(&[2]));
            Ok((absent, absent))
        }
        _ => Err(Error::shape(
            "Attention::attend_dense",
            "the query and key offsets are given together, or both absent for one sequence"
                .to_owned(),
        )),
    }
}

/// The per-page scale a plane's call carries: exactly one when the plane is FP8.
///
/// An FP8 plane without its scale would be read unnormalized and a scaled plane of another type
/// would multiply a scale the cache never divided by, so both mismatches are refused here rather
/// than at the launch.
fn page_scale(plane: TensorView, scale: Option<TensorView>) -> Result<Option<TensorView>> {
    match (plane.dtype == ops::DType::F8E4M3, scale) {
        (true, Some(scale)) => Ok(Some(scale)),
        (true, None) => Err(Error::shape(
            "Attention::attend",
            "an FP8 cache plane needs its per-page scale".to_owned(),
        )),
        (false, Some(_)) => Err(Error::shape(
            "Attention::attend",
            format!(
                "a {} cache plane carries no scale; the scale belongs to the page's encoding",
                plane.dtype
            ),
        )),
        (false, None) => Ok(None),
    }
}

/// The bytes the trtllm-gen attention program needs as its workspace.
///
/// The program's `workspace_size` argument (specs/worker.md 9); the executor carves it from its
/// scratch pool and zeroes it once (specs/worker.md 5.4, [D4]).
pub const TRTLLM_WORKSPACE_BYTES: u64 = 512 << 20;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_paged_layer_pins_the_page_geometry() {
        let attention = Attention::paged(128, 32, 4, 64, 40_960, None, 0.088).unwrap();
        assert_eq!(attention.backend(), AttentionBackend::Paged);
        assert_eq!(attention.max_kv_len(), 40_960);
        // 32 query heads do not group 5 key heads.
        assert!(Attention::paged(128, 32, 5, 64, 40_960, None, 0.088).is_err());
        // A bound shorter than one page cannot address a page.
        assert!(Attention::paged(128, 32, 4, 64, 32, None, 0.088).is_err());
    }

    #[test]
    fn a_dense_layer_states_its_mask_and_its_bound() {
        let encoder = Attention::dense(128, 32, 4, 8_192, true, None, 0.088).unwrap();
        assert_eq!(encoder.backend(), AttentionBackend::Dense);
        assert_eq!(encoder.max_kv_len(), 8_192);
        // The FP32 paths read a whole sequence at once: dense mask, no window.
        let audio = Attention::fp32(64, 8, 8, 4_096, 0.125).unwrap();
        assert_eq!(audio.backend(), AttentionBackend::Fp32);
        // A history window reads the causal mask only.
        assert!(Attention::dense(128, 32, 4, 8_192, false, Some(1_024), 0.088).is_err());
        assert!(Attention::dense(0, 32, 4, 8_192, true, None, 0.088).is_err());
        assert!(Attention::dense(128, 32, 4, 8_192, true, None, 0.0).is_err());
    }

    #[test]
    fn a_sparse_layer_names_its_tile_and_its_key_bound() {
        let vsa = Attention::sparse(64, 128, 12, 65_536, true, 0.088).unwrap();
        assert_eq!(vsa.backend(), AttentionBackend::Sparse);
        assert_eq!(vsa.block(), 64);
        assert_eq!(vsa.max_keys, 65_536);
        // The descriptor's key bound is whole tiles of the block size.
        assert!(Attention::sparse(64, 128, 12, 1_000, false, 0.088).is_err());
        assert!(Attention::sparse(0, 128, 12, 1_024, false, 0.088).is_err());
        assert!(Attention::sparse(128, 128, 12, 1_024, false, 0.088).is_ok());
    }

    /// A page's scale is its own layer's, so two layers of one group carry different ones and the
    /// call refuses a plane whose type and scale disagree in either direction.
    #[test]
    fn a_planes_per_page_scale_follows_the_planes_element_type() {
        let pages = Shape::new(&[16, 1, 1, 1]);
        let fp8 = TensorView::new(
            DevicePtr::new(0x1000),
            DType::F8E4M3,
            Shape::new(&[16, 64, 8, 128]),
        );
        let bf16 = TensorView::new(
            DevicePtr::new(0x2000),
            DType::BF16,
            Shape::new(&[16, 64, 8, 128]),
        );
        let first = TensorView::new(DevicePtr::new(0x3000), DType::F32, pages);
        let second = TensorView::new(DevicePtr::new(0x4000), DType::F32, pages);

        assert_eq!(
            page_scale(fp8, Some(first)).unwrap().unwrap().ptr,
            first.ptr
        );
        assert_eq!(
            page_scale(fp8, Some(second)).unwrap().unwrap().ptr,
            second.ptr
        );
        assert!(
            page_scale(fp8, None).is_err(),
            "an FP8 plane without its scale"
        );
        assert!(
            page_scale(bf16, Some(first)).is_err(),
            "a scale on a plane that has none"
        );
        assert!(page_scale(bf16, None).unwrap().is_none());
    }

    /// A call of one sequence carries the same argument shapes as the one with a two-element table,
    /// so `None` is the same launch and not a second path with its own assumptions.
    #[test]
    fn one_sequence_needs_no_offset_table() {
        let tokens = 1_024_u64;
        let table = TensorView::new(DevicePtr::new(0x1_000), DType::I32, Shape::new(&[2]));
        let (queries, keys) = single_sequence(Some(table), Some(table)).unwrap();
        assert_eq!(queries.ptr, table.ptr);
        assert_eq!(keys.ptr, table.ptr);

        let (queries, keys) = single_sequence(None, None).unwrap();
        // The shape a program reads its offsets with is the one a two-element table has, and the
        // address is zero: the program takes `[0, tokens]` from it, and the host writes nothing.
        assert_eq!(queries.shape.dims(), &[2]);
        assert_eq!(keys.shape.dims(), &[2]);
        assert_eq!(queries.dtype, DType::I32);
        assert_eq!(queries.ptr, DevicePtr::new(0));
        assert_eq!(keys.ptr, DevicePtr::new(0));

        // Half a table would leave one side with an address and the other with none.
        assert!(single_sequence(Some(table), None).is_err());
        assert!(single_sequence(None, Some(table)).is_err());
        // A table of any other length is the batch path's, and the descriptor's bound is what
        // decides it, not this rule.
        let odd = TensorView::new(DevicePtr::new(0x2_000), DType::I32, Shape::new(&[tokens]));
        assert!(single_sequence(Some(odd), Some(odd)).is_ok());
    }
}
