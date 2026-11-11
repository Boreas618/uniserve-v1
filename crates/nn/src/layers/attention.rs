//! Attention: dense over packed sequences, and paged over the prefix cache.
//!
//! Two layers, one per mathematics a model composes. [`DenseAttention`] attends packed queries to
//! packed keys in one launch of `ops::attention::dense::DenseAttention`, either as one sequence or
//! as the variable-length sequences a [`Varlen`] names; the element types of the views decide the
//! program, so an FP32 attention is the same layer over FP32 operands. [`PagedAttention`] writes a
//! batch's keys and values into the cache's pages and reads the pages each row addresses: the
//! scheduler assigns each token a slot (`-1` for a token it does not persist), [`ops::kv::KvWrite`]
//! scatters the projections into the pages - or, for an FP8 cache, [`ops::kv::KvWriteScaled`]
//! derives the page's own scale on the way in, the tensor the read side multiplies back - and
//! [`ops::attention::PagedAttention`] reads them. Both operators are compiled per geometry, so a
//! descriptor pins the element types, the head and page geometry and the mask, while the batch,
//! the lengths, the scale and the window travel in the call.

use ops::attention::dense::{DenseArgs, DenseAttention as DenseOp, DenseDescriptor};
use ops::attention::{Args, Descriptor, Mask, PagedAttention as PagedOp, Phase};
use ops::kv::{self, Descriptor as KvDescriptor, KvWrite, KvWriteScaled};
use ops::{DType, Shape, Operand};
use proto::program::PhaseKind;

use crate::capability::{KvGroup, KvLayer, TokenBatch};
#[cfg(any(doc, test))]
use crate::capability::{KvPlanes, KvView};
use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::{PDL, kv_tile};

/// The geometry and mask of a [`DenseAttention`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseSpec {
    /// Channels of one head.
    pub head_dim: u32,
    /// Query heads one token carries.
    pub heads: u32,
    /// Key and value heads one token carries; they group the query heads.
    pub kv_heads: u32,
    /// The longest key reduction the deployment serves: the ceiling every call's bound stays under.
    pub max_keys: u32,
    /// Whether each query sees only its own past.
    pub causal: bool,
    /// An inclusive history window, which only the causal mask reads.
    pub window: Option<u32>,
    /// The softmax scale applied to `q · k`.
    pub scale: f32,
}

/// The variable-length sequences one dense call packs, as flash attention's varlen call states
/// them.
///
/// Both tables are `[sequences + 1]` i32 cumulative row counts, of the queries and of the keys; a
/// call whose sequences attend themselves passes one table as both.
#[derive(Clone, Copy, Debug)]
pub struct Varlen {
    /// The query rows of each sequence, accumulated from zero.
    pub cu_q: Operand,
    /// The key rows of each sequence, accumulated from zero.
    pub cu_k: Operand,
    /// The longest query sequence, when the caller wrote the tables and knows it: the launch sizes
    /// its schedule by it rather than by the packed row count, which for many equal sequences asks
    /// for that many times the programs it needs.
    pub max_q: Option<u32>,
    /// The longest key reduction this launch makes: a launch value, because a deployment's buckets
    /// differ by thousands of rows. It covers every key passed and stays within the layer's
    /// `max_keys`.
    pub max_k: u32,
}

/// Dense attention from packed queries to packed keys.
#[derive(Clone, Copy, Debug)]
pub struct DenseAttention {
    spec: DenseSpec,
}

impl DenseAttention {
    /// The attention of `spec`.
    ///
    /// # Errors
    ///
    /// A zero head, bound or head count, query heads that do not group the key heads, a window
    /// over a dense mask, or a scale that is zero or not finite.
    pub fn new(spec: DenseSpec) -> Result<Self> {
        const OWNER: &str = "DenseAttention";
        if spec.head_dim == 0 || spec.heads == 0 || spec.kv_heads == 0 || spec.max_keys == 0 {
            return Err(Error::shape(
                OWNER,
                "a head, a key bound and both head counts must be nonzero".to_owned(),
            ));
        }
        if !spec.heads.is_multiple_of(spec.kv_heads) {
            return Err(Error::shape(
                OWNER,
                format!(
                    "{} query heads do not group {} key heads",
                    spec.heads, spec.kv_heads
                ),
            ));
        }
        if spec.window.is_some() && !spec.causal {
            return Err(Error::shape(
                OWNER,
                "a history window reads the causal mask only".to_owned(),
            ));
        }
        if !spec.scale.is_finite() || spec.scale == 0.0 {
            return Err(Error::shape(
                OWNER,
                format!("a softmax scale of {}", spec.scale),
            ));
        }
        Ok(Self { spec })
    }

    /// The geometry and mask this attention was built for.
    pub const fn spec(&self) -> &DenseSpec {
        &self.spec
    }

    /// The descriptor one launch takes: the geometry is the layer's, the element types and the key
    /// bound the call's, and whether offset tables are read selects one of two programs.
    fn descriptor(
        &self,
        query: DType,
        key_value: DType,
        output: DType,
        keys: u32,
        varlen: bool,
    ) -> DenseDescriptor {
        DenseDescriptor {
            query,
            key_value,
            output,
            head_dim: self.spec.head_dim,
            q_heads: self.spec.heads,
            kv_heads: self.spec.kv_heads,
            mask: if self.spec.causal {
                Mask::Causal
            } else {
                Mask::Dense
            },
            windowed: self.spec.window.is_some(),
            max_kv_len: keys,
            cu_seq_lens: varlen,
        }
    }

    /// `query [q_tokens, heads, head_dim]` attended to `key`/`value [kv_tokens, kv_heads,
    /// head_dim]`: a fresh `[q_tokens, heads, head_dim]` tensor of the query's element type.
    ///
    /// `varlen` names the packed sequences; absent, the call is **one sequence** of every query and
    /// every key, its key bound the key count, and the single-sequence program is launched rather
    /// than one reading a table at a null address.
    ///
    /// # Errors
    ///
    /// A key bound below the keys passed or above the layer's ceiling, an arena that cannot hold the
    /// output, or the operator's refusal.
    pub fn forward(
        &self,
        f: &Forward,
        query: Operand,
        key: Operand,
        value: Operand,
        varlen: Option<Varlen>,
    ) -> Result<Operand> {
        const OWNER: &str = "DenseAttention";
        let kv_tokens = key.shape.dim(0);
        let keys = match varlen {
            Some(varlen) => varlen.max_k,
            None => u32::try_from(kv_tokens).map_err(|_| {
                Error::shape(OWNER, format!("{kv_tokens} keys exceed a launch's bound"))
            })?,
        };
        if u64::from(keys) < kv_tokens {
            return Err(Error::shape(
                OWNER,
                format!("a {keys}-key bound for a launch of {kv_tokens} keys"),
            ));
        }
        if keys > self.spec.max_keys {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a {keys}-key bound beyond the {} this layer was accepted for",
                    self.spec.max_keys
                ),
            ));
        }
        let out = f.empty(query.dtype, query.shape)?;
        let descriptor = self.descriptor(query.dtype, key.dtype, out.dtype, keys, varlen.is_some());
        let args = DenseArgs {
            query,
            key,
            value,
            cu_seq_lens_q: varlen.map(|varlen| varlen.cu_q),
            cu_seq_lens_k: varlen.map(|varlen| varlen.cu_k),
            out,
            scale: self.spec.scale,
            window_left: self.spec.window.map_or(-1, |window| window as i32),
            causal: self.spec.causal,
            longest: varlen.and_then(|varlen| varlen.max_q).map(u64::from),
        };

        f.operate::<DenseOp>(&descriptor, args)?;
        Ok(out)
    }
}

/// The geometry of a [`PagedAttention`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PagedSpec {
    /// Channels of one head.
    pub head_dim: u32,
    /// Query heads one token carries on this rank.
    pub heads: u32,
    /// Key and value heads one token carries on this rank.
    pub kv_heads: u32,
    /// The longest key reduction the deployment serves; it enters the descriptor, so a deployment
    /// that serves longer sequences needs another layer, not another launch.
    pub max_kv_len: u32,
    /// An inclusive history window, when the model has one.
    pub window: Option<u32>,
    /// The softmax scale applied to `q · k`.
    pub scale: f32,
}

/// Bytes at the head of the paged program's workspace that can hold the decode's arrival count:
/// four bytes per row and query head, at least one per multiprocessor, for any bucket.
const COUNTER_BYTES: u64 = 1 << 20;

/// Paged attention over a cache's planes: causal, written and read per token.
pub struct PagedAttention {
    spec: PagedSpec,
}

impl PagedAttention {
    /// The attention of `spec`.
    ///
    /// # Errors
    ///
    /// A zero head or head count, or query heads that do not group the key heads.
    pub fn new(spec: PagedSpec) -> Result<Self> {
        const OWNER: &str = "PagedAttention";
        if spec.head_dim == 0 || spec.heads == 0 || spec.kv_heads == 0 {
            return Err(Error::shape(
                OWNER,
                "a head and both head counts must be nonzero".to_owned(),
            ));
        }
        if !spec.heads.is_multiple_of(spec.kv_heads) {
            return Err(Error::shape(
                OWNER,
                format!(
                    "{} query heads do not group {} key heads",
                    spec.heads, spec.kv_heads
                ),
            ));
        }
        Ok(Self { spec })
    }

    /// The geometry this attention was built for.
    pub const fn spec(&self) -> &PagedSpec {
        &self.spec
    }

    /// Zero the decode program's arrival counters at the head of `workspace`, once per forward
    /// before the first paged call.
    ///
    /// The decode program splits a row's keys over CTAs and counts their arrivals in the head of its
    /// workspace; it needs that count at zero on entry and leaves it at zero itself. A workspace
    /// carved from an arena holds whatever another bucket's tensors wrote there, and a stale count
    /// reduces a row before its CTAs are in, so a forward clears it once and every layer of the
    /// forward then finds it as the one before left it.
    ///
    /// # Errors
    ///
    /// When the fill cannot be issued.
    pub fn zero_counters(f: &Forward, workspace: Operand) -> Result<()> {
        let bytes = workspace.numel() * workspace.dtype.bytes().unwrap_or(1);
        f.zero(workspace, bytes.min(COUNTER_BYTES))
    }

    /// Write `key` and `value` into the layer's `cache`, then attend: a fresh
    /// `[tokens, q_heads, head_dim]` tensor of the query's element type.
    ///
    /// `query` is `[tokens, q_heads, head_dim]`, `key` and `value` `[tokens, kv_heads, head_dim]`.
    /// The page size is the cache group's, which the batch states too; the program's transient
    /// scratch is the cache's workspace ([`KvView::workspace`]).
    ///
    /// # Errors
    ///
    /// Projections or a cache the operators refuse, or an arena that cannot hold the output.
    pub fn forward(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: KvLayer<'_>,
        query: Operand,
        key: Operand,
        value: Operand,
    ) -> Result<Operand> {
        // One selection for both halves: the read is handed the planes the write was handed.
        let page = self.page_of(batch, cache.group)?;
        self.write(f, batch, cache.planes, page, key, value)?;
        let out = f.empty(query.dtype, query.shape)?;
        self.attend(f, batch, cache, page, query, out)?;
        Ok(out)
    }

    /// The tokens one page of `group` holds, which the batch's views address too.
    ///
    /// The page size enters every descriptor this layer launches, and the cache is laid out in it, so
    /// a batch that states another one, a page of no tokens, or a page longer than the layer's key
    /// bound is refused before anything is addressed.
    fn page_of(&self, batch: &TokenBatch, group: &KvGroup<'_>) -> Result<u32> {
        let page = group.page_size;
        if page == 0 || batch.page_size != page {
            return Err(Error::shape(
                "PagedAttention",
                format!(
                    "the cache has {page} tokens per page and the batch {}",
                    batch.page_size
                ),
            ));
        }
        if self.spec.max_kv_len < page {
            return Err(Error::shape(
                "PagedAttention",
                format!(
                    "a {}-key bound is shorter than a {page}-token page",
                    self.spec.max_kv_len
                ),
            ));
        }
        Ok(page)
    }

    /// Write the batch's keys and values into `planes`; slots of `-1` write nothing.
    fn write(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        planes: &crate::capability::KvPlanes,
        page: u32,
        key: Operand,
        value: Operand,
    ) -> Result<()> {
        let head_dim = u64::from(self.spec.head_dim);
        let heads = u64::from(self.spec.kv_heads);
        for (name, view) in [("key", key), ("value", value)] {
            if view.shape.rank() != 3 || view.shape.dim(1) != heads || view.shape.dim(2) != head_dim
            {
                return Err(Error::shape(
                    "PagedAttention::write",
                    format!(
                        "{name} is {:?}, [tokens, {heads}, {head_dim}] expected",
                        view.shape.dims()
                    ),
                ));
            }
        }
        // Both writers address a token's row from the buffer pointer: `kv_write` scatters a row and
        // `kv_write_scaled` reads one, and neither is handed a stride. A projection whose rows are
        // contiguous is what a layer's output is; a strided one would be read as packed rows and
        // land in the wrong slots, so it is refused here instead of silently.
        for (name, view) in [("key", key), ("value", value)] {
            let contiguous = view.strides.get(0) == heads * head_dim
                && view.strides.get(1) == head_dim
                && view.strides.last() == 1;
            if !contiguous {
                return Err(Error::shape(
                    "PagedAttention::write",
                    format!(
                        "{name} is {:?} at strides {}/{}/{}: the cache write addresses a token's \
                         row from its pointer, so the projections must be packed rows of {heads} \
                         heads × {head_dim}",
                        view.shape.dims(),
                        view.strides.get(0),
                        view.strides.get(1),
                        view.strides.last()
                    ),
                ));
            }
        }
        if key.shape.dim(0) != batch.tokens() || value.shape.dim(0) != batch.tokens() {
            return Err(Error::shape(
                "PagedAttention::write",
                format!(
                    "the batch has {} tokens but the projections have {} and {}",
                    batch.tokens(),
                    key.shape.dim(0),
                    value.shape.dim(0)
                ),
            ));
        }
        // The cache's element type decides the program, and there are two: a cache of the
        // projection's own type is a scatter of both planes in one launch, and an E4M3 cache is the
        // quantizing write, one launch per plane, dividing each page by the scale its reader
        // multiplies back. A cache of any other type, or one whose two planes are in different
        // types, is refused here rather than launched into a reader that would misread it: the
        // scatter writes two planes of one type and the quantizing write fills one plane at a time,
        // so a mixed pair needs a per-plane scatter that no operator provides.
        let quantized = matches!(planes.key.dtype, DType::F8E4M3);
        if planes.key.dtype != key.dtype || planes.value.dtype != value.dtype {
            if !quantized
                || planes.value.dtype != planes.key.dtype
                || planes.key.dtype != DType::F8E4M3
            {
                return Err(Error::shape(
                    "PagedAttention::write",
                    format!(
                        "the cache is {} keys and {} values over {} projections: a cache is the \
                         projections' own type or E4M3 in both planes",
                        planes.key.dtype, planes.value.dtype, key.dtype
                    ),
                ));
            }
            return self.write_quantized(f, batch, planes, page, key, value);
        }
        let descriptor = self.write_descriptor(key.dtype, planes.key.dtype, page);
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

    /// The quantizing write into an E4M3 cache: two launches, one per plane, each with the page
    /// scale its reader multiplies back.
    ///
    /// The operator takes one plane at a time — its scale is per plane and per page — so the keys
    /// and the values are two calls. Both read the same slot mapping, which is the batch's: a slot
    /// of `-1` writes nothing in either.
    fn write_quantized(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        planes: &crate::capability::KvPlanes,
        page: u32,
        key: Operand,
        value: Operand,
    ) -> Result<()> {
        let descriptor = self.scaled_write_descriptor(key.dtype, page);
        let rows = u64::from(self.spec.kv_heads) * u64::from(self.spec.head_dim);
        for (name, cache, scale, initialized, source) in [
            (
                "key",
                planes.key,
                planes.key_scale,
                planes.key_initialized,
                key,
            ),
            (
                "value",
                planes.value,
                planes.value_scale,
                planes.value_initialized,
                value,
            ),
        ] {
            // A reader of an E4M3 plane multiplies by a scale this launch must have applied, so a
            // cache without one is a cache nobody can read: refused here, before the launch.
            let scale = scale.ok_or_else(|| {
                Error::shape(
                    "PagedAttention::write",
                    format!(
                        "the {name} cache is {} and carries no page scale",
                        cache.dtype
                    ),
                )
            })?;
            f.operate::<KvWriteScaled>(
                &descriptor,
                kv::ScaledWriteArgs {
                    // The token count, not `source.rows()`: a source arrives as `[tokens, heads,
                    // head_dim]` as often as `[tokens, heads · head_dim]`, and `rows()` flattens the
                    // leading dimensions into tokens × heads - a width the operator then compares
                    // against a slot list of tokens.
                    source: source.reshape(Shape::new(&[source.shape.dim(0), rows])),
                    slots: batch.slot_mapping,
                    // The reader's own buffer: `attend` multiplies by `planes.key_scale`, so a
                    // second allocation of the right shape would produce a cache that reads as
                    // zeros with no error anywhere.
                    scales: scale,
                    cache,
                    initialized,
                },
            )?;
        }
        Ok(())
    }

    /// Check that the cache's page table is the batch's: one row per sequence, whole i32 pages.
    ///
    /// The operator sees only the table, so a table of another rank or another row count would read
    /// another batch's pages there and the operator could not say so; the batch's arity is known
    /// here, and the refusal names it. This is the check the first decode run needed and did not
    /// have: a table built as a vector surfaced as `paged_attention: expected 2 axes, found 1`.
    fn check_page_table(&self, table: Operand, sequences: u64) -> Result<()> {
        if table.shape.rank() != 2 || table.shape.dim(0) != sequences {
            return Err(Error::shape(
                "PagedAttention::attend",
                format!(
                    "the page table is {:?}, [{sequences}, pages] expected: one row per sequence of \
                     this batch",
                    table.shape.dims()
                ),
            ));
        }
        if table.dtype != DType::I32 {
            return Err(Error::shape(
                "PagedAttention::attend",
                format!("the page table is {} pages, not i32", table.dtype),
            ));
        }
        Ok(())
    }

    /// The descriptor one paged attention launches with.
    ///
    /// `phase` is the batch's — a prefill reads a whole sequence and a decode one token — and the
    /// three element types are the query's, the cache plane's and the output's, because the program
    /// is compiled for the combination. The window and the deployment's bound are the layer's: a
    /// windowed attention and a full one are different programs, and `max_kv_len` is the reduction
    /// length the 7.3 tolerance is written against.
    fn descriptor(
        &self,
        phase: Phase,
        query: DType,
        key_value: DType,
        output: DType,
        page: u32,
    ) -> Descriptor {
        Descriptor {
            phase,
            query,
            key_value,
            output,
            head_dim: self.spec.head_dim,
            page_size: page,
            q_heads: self.spec.heads,
            kv_heads: self.spec.kv_heads,
            mask: Mask::Causal,
            windowed: self.spec.window.is_some(),
            max_kv_len: self.spec.max_kv_len,
        }
    }

    /// The descriptor the cache write launches with: the projection's type and the plane's.
    ///
    /// Both are key fields and they differ in an E4M3 deployment, which is why the projection's own
    /// dtype cannot stand in for the cache's.
    fn write_descriptor(&self, projection: DType, cache: DType, page: u32) -> KvDescriptor {
        KvDescriptor {
            dtype: projection,
            kv: cache,
            page,
            heads: self.spec.kv_heads,
            head_dim: self.spec.head_dim,
            tile: kv_tile(self.spec.head_dim),
            pdl: PDL,
        }
    }

    /// The descriptor the quantizing cache write launches with: the projection's type, one plane's
    /// page scales and the geometry both share.
    fn scaled_write_descriptor(&self, projection: DType, page: u32) -> kv::ScaledWriteDescriptor {
        kv::ScaledWriteDescriptor {
            dtype: projection,
            scale: DType::F32,
            page,
            heads: self.spec.kv_heads,
            head_dim: self.spec.head_dim,
            block: kv_tile(self.spec.head_dim),
            pdl: PDL,
        }
    }

    /// Read one layer's cached keys and values and write this batch's attended output into `out`.
    ///
    /// The planes are the layer's own selection ([`KvLayer`]); the group travels for the page table
    /// and the page size, which are the group's and not a layer's.
    fn attend(
        &self,
        f: &Forward,
        batch: &TokenBatch,
        cache: KvLayer<'_>,
        page: u32,
        query: Operand,
        out: Operand,
    ) -> Result<()> {
        let (group, planes, workspace) = (cache.group, cache.planes, cache.workspace);
        // The batch's own parts are read by pointer arithmetic inside the programs, so they are
        // checked against each other once here, before any of them is addressed.
        batch.check()?;
        let phase = match batch.phase {
            PhaseKind::Prefill => Phase::Prefill,
            PhaseKind::Decode => Phase::Decode,
            other => {
                return Err(Error::Unsupported(format!(
                    "PagedAttention: phase {other} is not an attention phase"
                )));
            }
        };
        let head_dim = u64::from(self.spec.head_dim);
        let q_heads = u64::from(self.spec.heads);
        for (name, view) in [("query", query), ("out", out)] {
            if view.shape.rank() != 3
                || view.shape.dim(1) != q_heads
                || view.shape.dim(2) != head_dim
            {
                return Err(Error::shape(
                    "PagedAttention::attend",
                    format!(
                        "{name} is {:?}, [tokens, {q_heads}, {head_dim}] expected",
                        view.shape.dims()
                    ),
                ));
            }
        }
        // The page table is the cache's and it carries one row per sequence of this batch.
        self.check_page_table(
            group.block_tables,
            batch.query_offsets.numel().saturating_sub(1),
        )?;
        let descriptor = self.descriptor(phase, query.dtype, planes.key.dtype, out.dtype, page);
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
            page_table: group.block_tables,
            seq_lens: batch.seq_lens,
            cum_seq_lens_q: batch.query_offsets,
            out,
            workspace,
            scale: self.spec.scale,
            window_left: self.spec.window.map_or(-1, |window| window as i32),
            causal: true,
        };

        f.operate::<PagedOp>(&descriptor, args)
    }
}

/// The per-page scale a plane's call carries: a table with **one entry per page** when the plane is
/// FP8, and absent otherwise.
///
/// An FP8 plane without its scale would be read unnormalized, a scaled plane of another type would
/// multiply a scale the cache never divided by, and a *scalar* where the page table belongs would
/// scale every page of the sequence by one number — the last is the quiet one, since it type-checks,
/// launches, and answers a plausible tensor. The write divides each page by the page's own FP32
/// scale, so the reader's operand is a table of the same length: the pages of the plane in F32.
fn page_scale(plane: Operand, scale: Option<Operand>) -> Result<Option<Operand>> {
    match (plane.dtype == ops::DType::F8E4M3, scale) {
        (true, Some(scale)) => {
            let pages = plane.shape.dim(0);
            if scale.dtype != ops::DType::F32 || scale.shape.numel() != pages {
                return Err(Error::shape(
                    "PagedAttention::attend",
                    format!(
                        "an FP8 plane of {pages} pages carries {pages} F32 page scales; the operand \
                         is {} of {} element(s)",
                        scale.dtype,
                        scale.shape.numel()
                    ),
                ));
            }
            Ok(Some(scale))
        }
        (true, None) => Err(Error::shape(
            "PagedAttention::attend",
            "an FP8 cache plane needs its per-page scale".to_owned(),
        )),
        (false, Some(_)) => Err(Error::shape(
            "PagedAttention::attend",
            format!(
                "a {} cache plane carries no scale; the scale belongs to the page's encoding",
                plane.dtype
            ),
        )),
        (false, None) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use ops::{DType, DevicePtr, Shape};

    use super::*;

    /// The reader's scale operand is a page table, not a number: the write divides each page by its
    /// own scale, so a reader holding one scalar would scale the whole sequence by it — a wrong
    /// answer that type-checks and launches.
    #[test]
    fn a_page_scale_table_has_one_entry_per_page() {
        let pages = 16_u64;
        let plane = |dtype| {
            Operand::new(
                DevicePtr::new(0x1_0000),
                dtype,
                Shape::new(&[pages, 64, 8, 128]),
            )
        };
        let scale = |elements: u64, dtype| {
            Operand::new(DevicePtr::new(0x2_0000), dtype, Shape::new(&[elements]))
        };

        assert!(page_scale(plane(DType::F8E4M3), Some(scale(pages, DType::F32))).is_ok());
        assert!(
            page_scale(plane(DType::F8E4M3), None).is_err(),
            "no scale at all"
        );
        assert!(
            page_scale(plane(DType::F8E4M3), Some(scale(1, DType::F32))).is_err(),
            "one scalar where a page table belongs"
        );
        assert!(
            page_scale(plane(DType::F8E4M3), Some(scale(pages - 1, DType::F32))).is_err(),
            "a table short of the plane's pages"
        );
        assert!(
            page_scale(plane(DType::F8E4M3), Some(scale(pages, DType::BF16))).is_err(),
            "page scales are FP32"
        );
        assert!(
            page_scale(plane(DType::BF16), Some(scale(pages, DType::F32))).is_err(),
            "a plane of the projection's own type carries no scale"
        );
        assert!(page_scale(plane(DType::BF16), None).is_ok());
    }

    /// The paged geometry of a layer of 32 query heads of 128 at one rank.
    fn paged(kv_heads: u32, max_kv_len: u32) -> Result<PagedAttention> {
        PagedAttention::new(PagedSpec {
            head_dim: 128,
            heads: 32,
            kv_heads,
            max_kv_len,
            window: None,
            scale: 0.088,
        })
    }

    #[test]
    fn a_paged_layer_pins_the_page_geometry() {
        assert_eq!(paged(4, 40_960).unwrap().spec().max_kv_len, 40_960);
        // 32 query heads do not group 5 key heads.
        assert!(paged(5, 40_960).is_err());
    }

    #[test]
    fn a_page_table_is_one_i32_row_per_sequence_of_its_batch() {
        let attention = paged(4, 40_960).unwrap();
        let table = |dtype: DType, shape: &[u64]| {
            Operand::new(DevicePtr::new(0x2_0000), dtype, Shape::new(shape))
        };
        assert!(
            attention
                .check_page_table(table(DType::I32, &[15, 4]), 15)
                .is_ok()
        );
        // A table built as a vector: the shape a harness gets wrong, named with the batch's own
        // sequence count rather than as the operator's axis count.
        let error = attention
            .check_page_table(table(DType::I32, &[60]), 15)
            .expect_err("a page table of rank one");
        assert!(
            format!("{error}").contains("one row per sequence"),
            "{error}"
        );
        // Another batch's table, and pages that are not whole i32 words.
        assert!(
            attention
                .check_page_table(table(DType::I32, &[8, 4]), 15)
                .is_err()
        );
        assert!(
            attention
                .check_page_table(table(DType::BF16, &[15, 4]), 15)
                .is_err()
        );
    }

    /// The programs one paged forward of `tokens` tokens in one sequence launches on a recording
    /// forward, over a cache whose planes are `cache`.
    fn launched(
        attention: &PagedAttention,
        phase: PhaseKind,
        tokens: u64,
        cache: DType,
    ) -> Vec<String> {
        let arena = crate::Arena::counting();
        let f = Forward::recording(&arena);
        let tensor = |dtype: DType, dims: &[u64]| arena.tensor(dtype, Shape::new(dims)).unwrap();
        let batch = TokenBatch {
            positions: tensor(DType::I32, &[tokens]),
            slot_mapping: tensor(DType::I32, &[tokens]),
            seq_lens: tensor(DType::I32, &[1]),
            query_offsets: tensor(DType::I32, &[2]),
            logits_rows: tensor(DType::I32, &[1]),
            phase,
            page_size: 64,
            num_tokens: tokens as u32,
            max_kv_len: 64,
            draft_lens: None,
            rope_positions: None,
        };
        let pages = [1, 64, 4, 128];
        let scale = (cache == DType::F8E4M3).then(|| tensor(DType::F32, &[1, 1, 1, 1]));
        let planes = [KvPlanes {
            key: tensor(cache, &pages),
            value: tensor(cache, &pages),
            key_initialized: tensor(DType::U8, &[1, 1, 1, 1]),
            value_initialized: tensor(DType::U8, &[1, 1, 1, 1]),
            key_scale: scale,
            value_scale: scale,
        }];
        let groups = [KvGroup {
            block_tables: tensor(DType::I32, &[1, 1]),
            page_size: 64,
            dtype: cache,
            layers: &planes,
        }];
        let query = tensor(DType::BF16, &[tokens, 32, 128]);
        let key = tensor(DType::BF16, &[tokens, 4, 128]);
        let value = tensor(DType::BF16, &[tokens, 4, 128]);
        let workspace = tensor(DType::U8, &[512 << 20]);
        let cache = KvView {
            groups: &groups,
            workspace,
            state: None,
        };
        let (walked, launches) = crate::forward::recorded(|| {
            attention.forward(&f, &batch, cache.layer(0, 0)?, query, key, value)
        });
        walked.unwrap();
        launches
            .into_iter()
            .map(|(operator, key)| format!("{operator}/{key}"))
            .collect()
    }

    #[test]
    fn the_cache_writes_key_the_plane_type_and_the_reads_key_the_phase() {
        let attention = paged(4, 40_960).unwrap();
        let writes = |launches: &[String]| -> Vec<String> {
            launches
                .iter()
                .filter(|launch| launch.starts_with("kv_write"))
                .cloned()
                .collect()
        };
        let reads = |launches: &[String]| -> Vec<String> {
            launches
                .iter()
                .filter(|launch| launch.starts_with("paged_attention/"))
                .cloned()
                .collect()
        };
        // A cache of the projection's own type and an E4M3 cache are two programs, because the
        // quantizing write applies a page scale the scatter never sees.
        let plain = launched(&attention, PhaseKind::Prefill, 8, DType::BF16);
        let quantized = launched(&attention, PhaseKind::Prefill, 8, DType::F8E4M3);
        assert_eq!(writes(&plain).len(), 1, "{plain:?}");
        assert!(!writes(&quantized).is_empty(), "{quantized:?}");
        assert!(
            writes(&quantized)
                .iter()
                .all(|write| !writes(&plain).contains(write)),
            "{plain:?} {quantized:?}"
        );
        // A prefill reads a whole sequence and a decode one token: one attention program each.
        let decode = launched(&attention, PhaseKind::Decode, 1, DType::BF16);
        assert_eq!(reads(&plain).len(), 1, "{plain:?}");
        assert_eq!(reads(&decode).len(), 1, "{decode:?}");
        assert_ne!(reads(&plain), reads(&decode));
    }

    /// A dense layer of `heads` over `kv_heads`, `causal` or not, with a `window`.
    fn dense(
        kv_heads: u32,
        causal: bool,
        window: Option<u32>,
        scale: f32,
    ) -> Result<DenseAttention> {
        DenseAttention::new(DenseSpec {
            head_dim: 128,
            heads: 32,
            kv_heads,
            max_keys: 8_192,
            causal,
            window,
            scale,
        })
    }

    #[test]
    fn a_dense_layer_states_its_mask_and_refuses_what_it_cannot_launch() {
        assert!(dense(4, true, None, 0.088).unwrap().spec().causal);
        assert!(dense(4, true, Some(1_024), 0.088).is_ok());
        // A history window reads the causal mask only.
        assert!(dense(4, false, Some(1_024), 0.088).is_err());
        assert!(dense(5, false, None, 0.088).is_err());
        assert!(dense(4, false, None, 0.0).is_err());
    }

    /// A call's key bound covers its keys and stays under the layer's ceiling, and a call with
    /// sequence tables is another program than the single-sequence one.
    #[test]
    fn a_dense_call_reads_sequences_under_its_bound() {
        let attention = dense(4, false, None, 0.088).unwrap();
        let arena = crate::Arena::counting();
        let f = Forward::recording(&arena);
        let tensor = |dtype: DType, dims: &[u64]| arena.tensor(dtype, Shape::new(dims)).unwrap();
        let query = tensor(DType::BF16, &[16, 32, 128]);
        let key = tensor(DType::BF16, &[16, 4, 128]);
        let offsets = tensor(DType::I32, &[3]);
        let varlen = |max_k| Varlen {
            cu_q: offsets,
            cu_k: offsets,
            max_q: Some(8),
            max_k,
        };
        let call = |varlen: Option<Varlen>| {
            crate::forward::recorded(|| attention.forward(&f, query, key, key, varlen))
        };
        let (single, one) = call(None);
        single.unwrap();
        let (packed, two) = call(Some(varlen(16)));
        packed.unwrap();
        assert_eq!((one.len(), two.len()), (1, 1));
        assert_ne!(one[0].1, two[0].1, "sequence tables select another program");
        assert!(
            call(Some(varlen(8))).0.is_err(),
            "a bound below the keys passed"
        );
        assert!(
            call(Some(varlen(16_384))).0.is_err(),
            "a bound above the layer's ceiling"
        );
    }

    /// A page's scale is its own layer's, so two layers of one group carry different ones and the
    /// call refuses a plane whose type and scale disagree in either direction.
    #[test]
    fn a_planes_per_page_scale_follows_the_planes_element_type() {
        let pages = Shape::new(&[16, 1, 1, 1]);
        let fp8 = Operand::new(
            DevicePtr::new(0x1000),
            DType::F8E4M3,
            Shape::new(&[16, 64, 8, 128]),
        );
        let bf16 = Operand::new(
            DevicePtr::new(0x2000),
            DType::BF16,
            Shape::new(&[16, 64, 8, 128]),
        );
        let first = Operand::new(DevicePtr::new(0x3000), DType::F32, pages);
        let second = Operand::new(DevicePtr::new(0x4000), DType::F32, pages);

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
}
