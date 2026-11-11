//! Video sparse attention (VSA): pooled tile scores select the key tiles each query tile attends.
//!
//! The sequence is cut into `tile_rows`-row tiles. A layer prepares the projected rows, selects each
//! query tile's key tiles from their pooled scores, attends the selected tiles, and compresses each
//! (head, query tile) over its pooled values; the compressed branch is folded into the attended rows
//! by a gate the model projects. Every stage leaves its tables on the device, so a captured graph
//! replays the selection:
//!
//! ```text
//! prepare     normalize and rotate Q and K in place, zero each tile's padding rows, pool the means
//! scores      one dot product per (query tile, key tile) and head, over the pooled means
//! select      choose each query tile's key tiles from its pooled scores
//! fine        attend the selected tiles, then zero the query padding rows
//! compress    reduce each (head, query tile) over the tiles it selects, pooled value in, rows out
//! add_gated   fold the compressed branch's gated rows into a tile's rows, in place
//! compose     the member-major payload of one row interval from several members' heads
//! ```
//!
//! [`VsaAttention::attend`] runs the first five in order and answers the two [`Branches`], every
//! intermediate reserved from the forward's arena; the gated fold is
//! the model's to place, because under sequence parallelism it happens on the return path, interval
//! by interval, as [`VsaAttention::compose`], and on one rank in place as
//! [`VsaAttention::add_gated`]. A compressed row is broadcast over every row of its query tile, so
//! the fold is not a residual add of one row per row.
//!
//! A sparse layout packs `[text | conditions | audio | generated video | padding]`: the tiles before
//! the generated video's are read by every query row, and each video query tile reads the tiles its
//! selection names. Both sets share one softmax per query row, so the fine attention is one call over
//! their union, which the selection's kept-tiles-first order already is. Invalid key rows of a
//! partial tile are zeroed by the preparation, and invalid query rows by the fine attention after it
//! launches, so nothing leaks into a later residual.

use ops::attention::sparse::{SparseArgs, SparseAttention, SparseDescriptor};
use ops::descriptor::Float;
use ops::vsa::{self, MAX_MEMBERS, VsaZeroPadding};
use ops::{DType, Operand, Shape};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::Factors;
use crate::weights::Weight;

/// The fewest lanes a selection is compiled with, however few tiles a layout holds.
const SELECT_FLOOR: u32 = 256;

/// The geometry and arithmetic of a [`VsaAttention`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VsaSpec {
    /// Rows of one tile: the checkpoint's block size, a whole number of programme rows.
    pub tile_rows: u32,
    /// Heads per row.
    pub heads: u32,
    /// Channels per head.
    pub head_dim: u32,
    /// Whether the preparation normalizes Q and K by learned per-head gains.
    pub qk_norm: bool,
    /// The variance floor of that normalization.
    pub epsilon: f32,
    /// Leading channels of each head the preparation rotates; zero rotates none. A rotation implies
    /// the normalization, as the reference applies them together.
    pub rotated: u32,
    /// The deployment's bound on the key tiles one compression reduces over.
    pub max_tiles: u32,
    /// The largest key extent the fine attention serves, in rows: whole tiles.
    pub max_keys: u32,
    /// The element type the attended rows are written in: the stack's.
    pub output: DType,
}

/// The selection tables a packing prepared for one call.
#[derive(Clone, Copy)]
pub struct VsaTables {
    /// `[tiles]` i32: the valid rows of each tile.
    pub sizes: Operand,
    /// `[tiles]` i32: the segment each tile belongs to.
    pub segments: Operand,
    /// The segment-keep table, over-allocated to the tile count and read by segment index.
    pub keep: Operand,
}

/// The two branches [`VsaAttention::attend`] answers, which the gated fold combines.
#[derive(Clone, Copy)]
pub struct Branches {
    /// `[rows, heads · head_dim]`: the fine attention's rows, in the layer's output type.
    pub fine: Operand,
    /// `[heads, tiles, head_dim]`: one compressed row per (head, query tile), in the output type.
    pub compressed: Operand,
}

/// Each tile's pooled query, key and value means, `[tiles, heads, head_dim]` FP32.
struct Pooled {
    query: Operand,
    key: Operand,
    value: Operand,
}

/// One layer's video sparse attention.
pub struct VsaAttention {
    tile_rows: u32,
    heads: u32,
    width: u32,
    eps: f32,
    rotated: u32,
    norm: bool,
    block_rows: u32,
    /// The bound on the key tiles one compression reduces over.
    max_tiles: u32,
    /// The fine attention's key bound, in rows.
    keys: u32,
    /// The element type of the attended rows.
    output: DType,
}

impl VsaAttention {
    /// The attention of `spec`.
    ///
    /// # Errors
    ///
    /// A tile that is not whole programme rows, a zero head, width, tile bound or key bound, a key
    /// bound that is not whole tiles, or a rotation that is odd, wider than a head, or without the
    /// normalization.
    pub fn new(spec: VsaSpec) -> Result<Self> {
        const OWNER: &str = "VsaAttention";
        if spec.tile_rows == 0 || spec.heads == 0 || spec.head_dim == 0 {
            return Err(Error::shape(
                OWNER,
                "a tile, a head count and a feature count are each at least one".to_owned(),
            ));
        }
        if !spec.tile_rows.is_multiple_of(vsa::BLOCK_ROWS) {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a tile of {} rows is not a whole number of {} programmes",
                    spec.tile_rows,
                    vsa::BLOCK_ROWS
                ),
            ));
        }
        if spec.rotated > spec.head_dim || !spec.rotated.is_multiple_of(2) {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a rotation of {} features of {}",
                    spec.rotated, spec.head_dim
                ),
            ));
        }
        if spec.rotated > 0 && !spec.qk_norm {
            return Err(Error::shape(
                OWNER,
                "a rotation is applied with the normalization, as the reference does".to_owned(),
            ));
        }
        if spec.max_tiles == 0 {
            return Err(Error::shape(
                OWNER,
                "a tile bound of zero leaves the compression nothing to reduce".to_owned(),
            ));
        }
        if spec.max_keys == 0 || !spec.max_keys.is_multiple_of(spec.tile_rows) {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a key bound of {} rows is not whole {}-row tiles",
                    spec.max_keys, spec.tile_rows
                ),
            ));
        }
        Ok(Self {
            tile_rows: spec.tile_rows,
            heads: spec.heads,
            width: spec.head_dim,
            eps: spec.epsilon,
            rotated: spec.rotated,
            norm: spec.qk_norm,
            block_rows: vsa::BLOCK_ROWS,
            max_tiles: spec.max_tiles,
            keys: spec.max_keys,
            output: spec.output,
        })
    }

    /// Prepare, score, select, attend and compress `query`, `key` and `value`
    /// `[tiles · tile_rows, heads, head_dim]`: the fine and compressed [`Branches`].
    ///
    /// The projections are normalized and rotated in place; `gains` and `factors` are present
    /// exactly when the layer normalizes and rotates. The selection reads `tables`, and every query
    /// tile starts at sequence tile zero. The pooled means, scores and selection are reserved from
    /// the forward's arena before the branches.
    ///
    /// # Errors
    ///
    /// Rows that are not whole tiles, views or tables a stage refuses, or an arena that cannot hold
    /// the stages' views.
    pub fn attend(
        &self,
        f: &Forward,
        (query, key, value): (Operand, Operand, Operand),
        gains: Option<(&Weight, &Weight)>,
        factors: Option<Factors>,
        tables: &VsaTables,
    ) -> Result<Branches> {
        let rows = query.shape.dim(0);
        let tile = u64::from(self.tile_rows);
        if !rows.is_multiple_of(tile) {
            return Err(Error::shape(
                "VsaAttention",
                format!("{rows} rows are not whole {tile}-row tiles"),
            ));
        }
        let pooled = self.prepare(f, (query, key, value), gains, factors, tables.sizes)?;
        // The preparation already stored zeros past each tile's valid size in all three
        // projections, so the fine attention reads the same zeroed rows without a second pass.
        let scores = self.scores(f, pooled.query, pooled.key)?;
        let selection = self.select(f, scores, tables, 0)?;
        let fine = self.fine(f, (query, key, value), selection, tables.sizes)?;
        // One compression per (head, query tile): a dense FP32 softmax over every live key tile,
        // rounded once into the rows' type. A tile whose rows are all invalid is masked out of the
        // softmax rather than zeroed.
        let compressed = self.compress(f, scores, pooled.value, tables.sizes)?;
        Ok(Branches { fine, compressed })
    }

    /// The fine attention's descriptor, from the tile geometry and the query's element type.
    fn fine_descriptor(&self, dtype: DType) -> SparseDescriptor {
        SparseDescriptor {
            block: self.tile_rows,
            dtype,
            output: self.output,
            head_dim: self.width,
            lse: false,
            keys: self.keys,
        }
    }

    /// The descriptor the padding-row zeroing takes: the output's element type and the tile geometry.
    fn zero_descriptor(&self) -> vsa::ZeroPaddingDescriptor {
        vsa::ZeroPaddingDescriptor {
            dtype: self.output,
            heads: self.heads,
            tile_rows: self.tile_rows,
            width: self.width,
            block_rows: vsa::BLOCK_ROWS,
        }
    }

    /// `query` attended against the tiles the selection's `indices` and `counts` name, padding rows
    /// zeroed: a fresh `[rows, heads · head_dim]` tensor of the output type. The softmax scale is
    /// `1/sqrt(head_dim)`.
    fn fine(
        &self,
        f: &Forward,
        (query, key, value): (Operand, Operand, Operand),
        (indices, counts): (Operand, Operand),
        sizes: Operand,
    ) -> Result<Operand> {
        let rows = query.shape.dim(0);
        let fine = f.empty(
            self.output,
            Shape::new(&[rows, u64::from(self.heads) * u64::from(self.width)]),
        )?;
        let out = fine.reshape(query.shape);
        let head_dim = f64::from(self.width);
        f.operate::<SparseAttention>(
            &self.fine_descriptor(query.dtype),
            SparseArgs {
                query,
                key,
                value,
                indices,
                counts,
                valid: sizes,
                lse: None,
                out,
                scale: (1.0 / head_dim.sqrt()) as f32,
            },
        )?;
        // The padding rows of a partial last tile are the packed sequence's; the same sizes that
        // bound the key tiles bound the query tiles.
        f.operate::<VsaZeroPadding>(
            &self.zero_descriptor(),
            vsa::ZeroPaddingArgs { values: out, sizes },
        )?;
        Ok(fine)
    }

    /// Rows of one tile.
    pub const fn tile_rows(&self) -> u32 {
        self.tile_rows
    }

    /// Heads per row.
    pub const fn heads(&self) -> u32 {
        self.heads
    }

    /// Features per head.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The descriptor one `vsa_select_tiles` launch takes, for a key-tile **bound**.
    ///
    /// `block` is the lane width the compiled program ranks in, and the package's own signature
    /// makes it a capacity rather than a count: `tiles` is a runtime argument
    /// (`build.py:28-29`) and the kernel masks every lane at or past it (`keys < tiles`), so a
    /// program compiled for the deployment's bound serves **every** request up to it. A descriptor
    /// built from a request's tile count instead compiles a fresh module per distinct count whose
    /// extra lanes are all masked off - the same arithmetic at the cost of a class per request.
    fn select_descriptor(&self, tiles: u64) -> Result<vsa::SelectTilesDescriptor> {
        Ok(vsa::SelectTilesDescriptor {
            block: u32::try_from(tiles.max(1).next_power_of_two()).map_err(|_| {
                Error::shape(
                    "VsaAttention",
                    format!("{tiles} key tiles in one query tile"),
                )
            })?,
        })
    }

    /// What a preparation needs of its at-rest arguments: [`Self::prepare`] calls this before it
    /// composes a descriptor.
    ///
    /// It is public because a host test cannot build a [`Forward`] and this is the only part of a
    /// preparation a test can exercise; a second copy of the rule inside `prepare` would be the
    /// untested path.
    fn check_prepare(
        &self,
        query_weight: bool,
        key_weight: bool,
        cosine: Option<Operand>,
        sine: Option<Operand>,
    ) -> Result<()> {
        if query_weight != key_weight {
            return Err(Error::shape(
                "VsaAttention",
                format!(
                    "the Q and K normalization weights are a pair: {} without {}",
                    if query_weight { "Q" } else { "K" },
                    if query_weight { "K" } else { "Q" }
                ),
            ));
        }
        if self.norm != query_weight {
            return Err(Error::shape(
                "VsaAttention",
                "the Q/K normalization weights are present exactly when the layer normalizes"
                    .to_owned(),
            ));
        }
        if self.rotated != 0 && (cosine.is_none() || sine.is_none()) {
            return Err(Error::shape(
                "VsaAttention",
                "a rotated preparation needs both factor tables".to_owned(),
            ));
        }
        Ok(())
    }

    /// The descriptor one `vsa_prepare_tiles` launch takes, from the geometry and the strides.
    ///
    /// The strides are **compiled in** (`crates/ops/kernels/vsa_prepare_tiles/triton/src/build.py`
    /// hands them to the build as constants), so they are in the key, taken from the layout the
    /// model's own buffers have rather than from a caller's choice.
    fn prepare_descriptor(
        dtype: DType,
        geometry: [u32; 5],
        norm: bool,
        strides: [u64; 7],
        epsilon: Float,
    ) -> vsa::PrepareTilesDescriptor {
        let [heads, width, rotated, tile_rows, block_rows] = geometry;
        let [
            query_row,
            query_head,
            key_row,
            key_head,
            value_row,
            value_head,
            factor,
        ] = strides;
        vsa::PrepareTilesDescriptor {
            dtype,
            heads,
            width,
            rotated,
            tile_rows,
            block_rows,
            epsilon,
            norm,
            query_row_stride: query_row,
            query_head_stride: query_head,
            key_row_stride: key_row,
            key_head_stride: key_head,
            value_row_stride: value_row,
            value_head_stride: value_head,
            factor_stride: factor,
        }
    }

    /// Normalize, rotate, zero and pool whole tiles of projected Q/K/V rows: the pooled means.
    ///
    /// `query`, `key` and `value` are `[tiles · tile_rows, heads, width]` and are modified in place;
    /// `sizes` is `[tiles]` i32. The normalization weights and the factor tables are present exactly
    /// when the layer normalizes and rotates.
    fn prepare(
        &self,
        f: &Forward,
        (query, key, value): (Operand, Operand, Operand),
        gains: Option<(&Weight, &Weight)>,
        factors: Option<Factors>,
        sizes: Operand,
    ) -> Result<Pooled> {
        let (query_weight, key_weight) = (gains.map(|(query, _)| query), gains.map(|(_, key)| key));
        let (cosine, sine) = (
            factors.map(|table| table.cosine),
            factors.map(|table| table.sine),
        );
        self.check_prepare(query_weight.is_some(), key_weight.is_some(), cosine, sine)?;
        let tiles = query.shape.dim(0) / u64::from(self.tile_rows);
        let pooled_shape = Shape::new(&[tiles, u64::from(self.heads), u64::from(self.width)]);
        let pooled = Pooled {
            query: f.empty(DType::F32, pooled_shape)?,
            key: f.empty(DType::F32, pooled_shape)?,
            value: f.empty(DType::F32, pooled_shape)?,
        };
        let descriptor = Self::prepare_descriptor(
            query.dtype,
            [
                self.heads,
                self.width,
                self.rotated,
                self.tile_rows,
                self.block_rows,
            ],
            self.norm,
            [
                query.strides.get(0),
                query.strides.get(1),
                key.strides.get(0),
                key.strides.get(1),
                value.strides.get(0),
                value.strides.get(1),
                cosine.map(|table| table.strides.get(0)).unwrap_or(0),
            ],
            Float::from(self.eps),
        );
        let args = vsa::PrepareTilesArgs {
            query,
            key,
            value,
            query_weight: query_weight.map(Weight::view),
            key_weight: key_weight.map(Weight::view),
            cosine,
            sine,
            sizes,
            pooled_query: pooled.query,
            pooled_key: pooled.key,
            pooled_value: pooled.value,
        };
        f.operate::<vsa::VsaPrepareTiles>(&descriptor, args)?;
        Ok(pooled)
    }

    /// Fold the compressed branch's gated rows into each tile's rows of `inout` (the fine branch),
    /// in place.
    ///
    /// # Errors
    ///
    /// The operator's refusal.
    pub fn add_gated(
        &self,
        f: &Forward,
        inout: Operand,
        compressed: Operand,
        gate: Operand,
    ) -> Result<()> {
        let descriptor = vsa::AddGatedTilesDescriptor {
            dtype: inout.dtype,
            tile_rows: self.tile_rows,
            width: self.width,
            block_rows: self.block_rows,
        };
        let args = vsa::AddGatedTilesArgs {
            output: inout,
            compressed,
            gate,
        };
        f.operate::<vsa::VsaAddGatedTiles>(&descriptor, args)
    }

    /// How many members a composition may name: [`Self::compose`] calls this before it composes a
    /// descriptor.
    ///
    /// A tensor-parallel group's membership is the caller's, so the count is the caller's argument
    /// and its refusal belongs where a host can see it — public for the same reason as
    /// [`Self::check_prepare`].
    fn check_members(&self, members: usize) -> Result<()> {
        if members == 0 || members > MAX_MEMBERS {
            return Err(Error::shape(
                "VsaAttention",
                format!("{members} members, one to {MAX_MEMBERS}"),
            ));
        }
        Ok(())
    }

    /// Write one row interval of every member's gated rows, member-major, into `payload`, the
    /// interval's send buffer of a sequence-parallel return.
    ///
    /// `branches.fine` holds every member's rows of this member's heads, `members · owner rows`
    /// rows, so each member owns `fine rows / members` of them; `interval` is the rows of each
    /// owner this payload carries. `slots` names the payload slot of each member, in member order;
    /// a sequence-parallel group has one member per rank.
    ///
    /// # Errors
    ///
    /// No member or more than the operator's table, or the operator's refusal.
    pub fn compose(
        &self,
        f: &Forward,
        payload: Operand,
        branches: Branches,
        gate: Operand,
        interval: crate::layers::Interval,
        slots: &[u32],
    ) -> Result<()> {
        self.check_members(slots.len())?;
        let Branches { fine, compressed } = branches;
        let owner_rows = fine.shape.dim(0) / slots.len() as u64;
        let mut table = [0_u32; MAX_MEMBERS];
        table[..slots.len()].copy_from_slice(slots);
        let descriptor = vsa::ComposeGatedRowsDescriptor {
            dtype: payload.dtype,
            heads: self.heads,
            tile_rows: self.tile_rows,
            width: self.width,
            block_rows: self.block_rows,
            slots: table,
            members: slots.len() as u32,
        };
        let args = vsa::ComposeGatedRowsArgs {
            output: payload,
            fine,
            compressed,
            gate,
            owner_rows,
            start: interval.first,
            count: interval.count,
        };
        f.operate::<vsa::VsaComposeGatedRows>(&descriptor, args)
    }

    /// The descriptor one score launch takes: the head count, the features per head and the bound on
    /// the dot product's length. A walk states it from the layer's own geometry.
    fn scores_descriptor(&self) -> vsa::ScoresDescriptor {
        vsa::ScoresDescriptor::new(
            self.heads,
            self.width,
            self.width.next_power_of_two(),
            false,
        )
    }

    /// The descriptor one compression launch takes: the element type, the head count, the features
    /// per head, and the bound on the key tiles one reduction covers.
    fn compress_descriptor(&self, dtype: DType) -> vsa::CompressPooledDescriptor {
        vsa::CompressPooledDescriptor::new(
            dtype,
            self.heads,
            self.width,
            self.max_tiles,
            self.width.next_power_of_two(),
            false,
        )
    }

    /// Multiply the pooled query and key tiles into the scores `select` ranks.
    ///
    /// `pooled_query` is `[query tiles, heads, width]` and `pooled_key` `[tiles, heads, width]`, both
    /// FP32 — the layout and the type [`Self::prepare`] writes; the scores are a fresh
    /// `[heads, query tiles, tiles]` FP32 tensor read by [`Self::select`]. One dot product per
    /// (query tile, key tile) and head — no scale and no softmax, because the selection only
    /// compares and the attention scales its own products.
    fn scores(&self, f: &Forward, pooled_query: Operand, pooled_key: Operand) -> Result<Operand> {
        if pooled_query.shape.rank() != 3 {
            return Err(Error::shape(
                "VsaAttention",
                format!(
                    "the pooled query is [query tiles, heads, width], not {:?}",
                    pooled_query.shape.dims()
                ),
            ));
        }
        // The pooled features are the layer's own width: the program sums that many per head, and a
        // wider query would be read through a stride the descriptor never declared.
        if pooled_query.shape.dim(2) != u64::from(self.width) {
            return Err(Error::shape(
                "VsaAttention",
                format!(
                    "the pooled query holds {} features per head, the layer {}",
                    pooled_query.shape.dim(2),
                    self.width
                ),
            ));
        }
        let scores = f.empty(
            DType::F32,
            Shape::new(&[
                u64::from(self.heads),
                pooled_query.shape.dim(0),
                pooled_key.shape.dim(0),
            ]),
        )?;
        let descriptor = self.scores_descriptor();
        let args = vsa::ScoresArgs {
            pooled_query,
            pooled_key,
            out: scores,
        };
        f.operate::<vsa::VsaScores>(&descriptor, args)?;
        Ok(scores)
    }

    /// Each query tile's softmax over the live key tiles weights their pooled values.
    ///
    /// `scores` is `[heads, query tiles, tiles]` FP32 contiguous, exactly what [`Self::scores`]
    /// writes; `values` is `[tiles, heads, width]` FP32, what the pooling writes; `sizes` is
    /// `[tiles]` i32, each key tile's live rows. A tile of no live rows is masked out of the softmax
    /// rather than zeroed, because a zeroed pooled value under a live score would still take weight
    /// and scale the others down.
    ///
    /// The answer is a fresh `[heads, query tiles, width]` tensor in the attention's own element
    /// type — the layout the gate step reads, and one row per (head, query tile) rather than one
    /// per row, because every row of a query tile compresses to the same vector.
    fn compress(
        &self,
        f: &Forward,
        scores: Operand,
        values: Operand,
        sizes: Operand,
    ) -> Result<Operand> {
        self.check_compress(scores, values, sizes)?;
        let out = f.empty(
            self.output,
            Shape::new(&[
                u64::from(self.heads),
                scores.shape.dim(1),
                u64::from(self.width),
            ]),
        )?;
        let descriptor = self.compress_descriptor(out.dtype);
        let args = vsa::CompressPooledArgs {
            scores,
            values,
            sizes,
            out,
        };
        f.operate::<vsa::VsaCompressPooled>(&descriptor, args)?;
        Ok(out)
    }

    /// The checks [`Self::compress`] makes before it launches, reachable without a device.
    ///
    /// Every refusal the layer can raise lives here rather than behind `&Forward`, because a
    /// `Forward` needs a live stream and context and no host test can build one; the layer's own
    /// test asserts these, and [`Self::compress`] calls this first so the two cannot drift.
    fn check_compress(&self, scores: Operand, values: Operand, sizes: Operand) -> Result<()> {
        if self.max_tiles == 0 {
            return Err(Error::shape(
                "VsaAttention",
                "a compression needs the deployment's key-tile bound: call max_tiles".to_owned(),
            ));
        }
        if scores.shape.rank() != 3 || values.shape.rank() != 3 {
            return Err(Error::shape(
                "VsaAttention",
                format!(
                    "the scores are [heads, query tiles, tiles] and the values [tiles, heads, \
                     width], not {:?} and {:?}",
                    scores.shape.dims(),
                    values.shape.dims()
                ),
            ));
        }
        if scores.shape.dim(0) != u64::from(self.heads)
            || values.shape.dim(1) != u64::from(self.heads)
        {
            return Err(Error::shape(
                "VsaAttention",
                format!(
                    "the compression is over {} heads, and the views carry {} and {}",
                    self.heads,
                    scores.shape.dim(0),
                    values.shape.dim(1)
                ),
            ));
        }
        if values.shape.dim(2) != u64::from(self.width) {
            return Err(Error::shape(
                "VsaAttention",
                format!(
                    "the pooled values carry {} features, and the block's heads carry {}",
                    values.shape.dim(2),
                    self.width
                ),
            ));
        }
        if sizes.shape.numel() != values.shape.dim(0) {
            return Err(Error::shape(
                "VsaAttention",
                format!(
                    "{} key tiles of pooled values against {} live sizes",
                    values.shape.dim(0),
                    sizes.shape.numel()
                ),
            ));
        }
        Ok(())
    }

    /// Choose each query tile's key tiles from its pooled scores.
    ///
    /// `scores` is `[heads, query tiles, tiles]` FP32 contiguous; the tables' `segments` and
    /// `sizes` are `[tiles]` i32 and `keep` `[segments]` i32. The answer is the fresh `indices`
    /// `[heads, query tiles, tiles]` i32 (kept tiles first, in ascending order) and `counts`
    /// `[heads, query tiles]` i32. `query_offset` is the sequence tile of the first query tile.
    fn select(
        &self,
        f: &Forward,
        scores: Operand,
        tables: &VsaTables,
        query_offset: u64,
    ) -> Result<(Operand, Operand)> {
        if scores.shape.rank() != 3 {
            return Err(Error::shape(
                "VsaAttention",
                format!(
                    "the pooled scores are [heads, query tiles, tiles], not {:?}",
                    scores.shape.dims()
                ),
            ));
        }
        let tiles = scores.shape.dim(2);
        // A layout past the deployment's bound has no scratch sized for it, so it is refused here
        // rather than compiled for.
        if self.max_tiles == 0 {
            return Err(Error::shape(
                "VsaAttention",
                "a selection needs the deployment's key-tile bound: call max_tiles".to_owned(),
            ));
        }
        if tiles > u64::from(self.max_tiles) {
            return Err(Error::shape(
                "VsaAttention",
                format!(
                    "a request ranks {tiles} key tiles, past the deployment's bound of {}",
                    self.max_tiles
                ),
            ));
        }
        // The lanes are this layout's own tiles, not the deployment's bound: a graph is captured for
        // one layout, and a program over the bound's lanes ranks and masks them all for every query
        // tile of every head.
        let descriptor = self.select_descriptor(tiles.max(u64::from(SELECT_FLOOR)))?;
        // The selection is as wide as the scores it comes from, one entry per key tile with the
        // kept ones first.
        let indices = f.empty(DType::I32, scores.shape)?;
        let counts = f.empty(
            DType::I32,
            Shape::new(&[scores.shape.dim(0), scores.shape.dim(1)]),
        )?;
        let args = vsa::SelectTilesArgs {
            scores,
            segments: tables.segments,
            sizes: tables.sizes,
            keep: tables.keep,
            indices,
            counts,
            query_offset,
        };
        f.operate::<vsa::VsaSelectTiles>(&descriptor, args)?;
        Ok((indices, counts))
    }
}

#[cfg(test)]
mod tests {
    use ops::{DType, DevicePtr, Shape};

    use super::*;

    /// A view of `shape` at a host address: these tests never launch, so the address is a marker.
    fn view(dtype: DType, shape: &[u64]) -> Operand {
        Operand::new(DevicePtr::new(0x1000), dtype, Shape::new(shape))
    }

    /// A layer of `heads` heads `width` wide over `tile_rows`-row tiles, normalized and rotated as
    /// `rotated` says, with a tile bound of 4096.
    fn spec(tile_rows: u32, heads: u32, width: u32, qk_norm: bool, rotated: u32) -> VsaSpec {
        VsaSpec {
            tile_rows,
            heads,
            head_dim: width,
            qk_norm,
            epsilon: 1e-6,
            rotated,
            max_tiles: 4_096,
            max_keys: 64 * tile_rows,
            output: DType::BF16,
        }
    }

    fn layer(tile_rows: u32, heads: u32, width: u32) -> VsaAttention {
        VsaAttention::new(spec(tile_rows, heads, width, false, 0)).unwrap()
    }

    /// The selection's block is the layout's own key-tile count rounded up to a power of two: the
    /// package's `tiles` is a runtime argument and the kernel masks the lanes past it, so one
    /// compiled class serves every layout of a class, up to the deployment's bound.
    #[test]
    fn the_select_block_is_the_layouts_tiles_rounded_up() {
        let layer = VsaAttention::new(VsaSpec {
            max_tiles: 5_400,
            ..spec(64, 4, 128, false, 0)
        })
        .unwrap();
        let block = |tiles: u64| {
            let arena = crate::Arena::counting();
            let f = Forward::recording(&arena);
            let table = || arena.tensor(DType::I32, Shape::new(&[tiles])).unwrap();
            let scores = arena
                .tensor(DType::F32, Shape::new(&[4, 1, tiles]))
                .unwrap();
            let tables = VsaTables {
                sizes: table(),
                segments: table(),
                keep: table(),
            };
            let (walked, launches) =
                crate::forward::recorded(|| layer.select(&f, scores, &tables, 0));
            walked.unwrap();
            assert_eq!(launches.len(), 1, "{launches:?}");
            launches[0].1.clone()
        };
        // 5 400 rounds up to 8 192, and a layout below the floor is compiled at the floor.
        assert_eq!(block(5_400), "block_8192");
        assert_eq!(block(1), block(u64::from(SELECT_FLOOR)));
        // Past the deployment's bound there is no scratch, so the layout is refused.
        let arena = crate::Arena::counting();
        let f = Forward::recording(&arena);
        let past = arena
            .tensor(DType::F32, Shape::new(&[4, 1, 5_401]))
            .unwrap();
        let table = arena.tensor(DType::I32, Shape::new(&[5_401])).unwrap();
        let tables = VsaTables {
            sizes: table,
            segments: table,
            keep: table,
        };
        assert!(layer.select(&f, past, &tables, 0).is_err());
    }

    #[test]
    fn a_tile_is_whole_programme_rows_and_every_bound_is_positive() {
        // A tile of 64 rows and a 128-row checkpoint both serve: tile rows are the checkpoint's.
        assert!(VsaAttention::new(spec(64, 8, 256, false, 0)).is_ok());
        assert!(VsaAttention::new(spec(128, 8, 256, false, 0)).is_ok());
        assert!(VsaAttention::new(spec(0, 8, 256, false, 0)).is_err());
        assert!(VsaAttention::new(spec(64, 0, 256, false, 0)).is_err());
        assert!(VsaAttention::new(spec(64, 8, 0, false, 0)).is_err());
        // 96 is six whole 16-row programmes, so it serves; 100 is not.
        assert!(VsaAttention::new(spec(96, 8, 256, false, 0)).is_ok());
        let text = match VsaAttention::new(spec(100, 8, 256, false, 0)) {
            Ok(_) => panic!("100 rows is not a whole number of programme rows"),
            Err(error) => format!("{error}"),
        };
        assert!(text.contains("100"), "{text}");
        // The compression's tile bound and the fine attention's key bound are stated, the latter
        // in whole tiles.
        let base = spec(64, 8, 256, false, 0);
        assert!(
            VsaAttention::new(VsaSpec {
                max_tiles: 0,
                ..base
            })
            .is_err()
        );
        assert!(
            VsaAttention::new(VsaSpec {
                max_keys: 0,
                ..base
            })
            .is_err()
        );
        assert!(
            VsaAttention::new(VsaSpec {
                max_keys: 100,
                ..base
            })
            .is_err()
        );
    }

    /// A rotation reads the first features of a head in pairs, with the normalization the reference
    /// applies with it.
    #[test]
    fn a_rotation_is_even_bounded_by_the_width_and_needs_the_norm() {
        assert!(VsaAttention::new(spec(64, 8, 256, true, 96)).is_ok());
        assert!(
            VsaAttention::new(spec(64, 8, 256, false, 96)).is_err(),
            "a rotation without the norm"
        );
        assert!(
            VsaAttention::new(spec(64, 8, 256, true, 97)).is_err(),
            "a rotation reads pairs of features"
        );
        assert!(
            VsaAttention::new(spec(64, 8, 256, true, 512)).is_err(),
            "a rotation of more features than a head has"
        );
    }

    /// The weights are present exactly when the layer normalizes, and a rotation needs both factor
    /// tables: a preparation missing either would read a null view, so both are refused by name.
    #[test]
    fn a_preparation_names_its_weights_and_tables_exactly() {
        let plain = layer(64, 8, 256);
        plain
            .check_prepare(false, false, None, None)
            .expect("a layer that does not normalize reads no weights");
        assert!(plain.check_prepare(true, true, None, None).is_err());
        assert!(plain.check_prepare(true, false, None, None).is_err());

        let normed = VsaAttention::new(spec(64, 8, 256, true, 0)).unwrap();
        normed
            .check_prepare(true, true, None, None)
            .expect("a normalizing layer reads both weights");
        assert!(normed.check_prepare(false, false, None, None).is_err());
        assert!(
            normed.check_prepare(true, false, None, None).is_err(),
            "one weight is not both halves of the normalization"
        );

        let rotated = VsaAttention::new(spec(64, 8, 256, true, 96)).unwrap();
        let table = view(DType::F32, &[4096, 96]);
        rotated
            .check_prepare(true, true, Some(table), Some(table))
            .expect("a rotated layer reads both tables");
        let missing = rotated.check_prepare(true, true, Some(table), None);
        assert!(
            format!("{}", missing.unwrap_err()).contains("factor tables"),
            "both tables are the rotation's"
        );
    }

    /// A composition names one member per rank of a sequence-parallel group: at least one, at most
    /// the operator's table.
    #[test]
    fn a_composition_names_one_member_per_rank_and_no_more_than_the_table() {
        let tiles = layer(64, 8, 256);
        tiles.check_members(1).expect("one member");
        tiles.check_members(MAX_MEMBERS).expect("the table's size");
        let none = tiles.check_members(0);
        assert!(
            format!("{}", none.unwrap_err()).contains("one to"),
            "no member is not a composition"
        );
        let too_many = tiles.check_members(MAX_MEMBERS + 1);
        assert!(
            format!("{}", too_many.unwrap_err()).contains(&MAX_MEMBERS.to_string()),
            "the table's size is named"
        );
    }

    #[test]
    fn a_compression_refuses_views_that_would_sum_another_block() {
        let tiles = layer(128, 4, 64);
        let scores = view(DType::F32, &[4, 8, 16]);
        let values = view(DType::F32, &[16, 4, 64]);
        let sizes = view(DType::I32, &[16]);
        assert!(tiles.check_compress(scores, values, sizes).is_ok());
        // Another head count on either side would weight another block's rows.
        assert!(
            tiles
                .check_compress(view(DType::F32, &[2, 8, 16]), values, sizes)
                .is_err()
        );
        assert!(
            tiles
                .check_compress(scores, view(DType::F32, &[16, 2, 64]), sizes)
                .is_err()
        );
        // Another feature width is another head's arithmetic.
        assert!(
            tiles
                .check_compress(scores, view(DType::F32, &[16, 4, 32]), sizes)
                .is_err()
        );
        // The live sizes are one per key tile.
        assert!(
            tiles
                .check_compress(scores, values, view(DType::I32, &[12]))
                .is_err()
        );
    }

    /// One attend launches the five stages in order and refuses rows that are not whole tiles.
    #[test]
    fn an_attend_prepares_scores_selects_attends_and_compresses() {
        let layer = VsaAttention::new(spec(64, 4, 128, true, 96)).unwrap();
        let arena = crate::Arena::counting();
        let f = Forward::recording(&arena);
        let rows = 4 * 64;
        let tensor = |dtype: DType, dims: &[u64]| arena.tensor(dtype, Shape::new(dims)).unwrap();
        let projection = || tensor(DType::BF16, &[rows, 4, 128]);
        let (query, key, value) = (projection(), projection(), projection());
        let gain = || Weight {
            backing: crate::Backing::new(DevicePtr::new(0x8000), 256),
            view: Operand::new(DevicePtr::new(0x8000), DType::BF16, Shape::new(&[128])),
        };
        let (query_gain, key_gain) = (gain(), gain());
        let factors = Factors {
            cosine: tensor(DType::F32, &[rows, 48]),
            sine: tensor(DType::F32, &[rows, 48]),
        };
        let tables = VsaTables {
            sizes: tensor(DType::I32, &[4]),
            segments: tensor(DType::I32, &[4]),
            keep: tensor(DType::I32, &[4]),
        };
        let (walked, launches) = crate::forward::recorded(|| {
            layer.attend(
                &f,
                (query, key, value),
                Some((&query_gain, &key_gain)),
                Some(factors),
                &tables,
            )
        });
        walked.unwrap();
        let ragged = tensor(DType::BF16, &[rows + 1, 4, 128]);
        assert!(
            layer
                .attend(
                    &f,
                    (ragged, ragged, ragged),
                    Some((&query_gain, &key_gain)),
                    Some(factors),
                    &tables
                )
                .is_err()
        );
        let operators: Vec<&str> = launches.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            operators,
            [
                "vsa_prepare_tiles",
                "vsa_scores",
                "vsa_select_tiles",
                "sparse_attention",
                "vsa_zero_padding",
                "vsa_compress_pooled"
            ]
        );
    }
}
