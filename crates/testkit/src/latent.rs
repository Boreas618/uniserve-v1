//! A stub of DeepSeek-V4.1-Flash's cache: a causal language model of no numerics that declares
//! the cache fields and the per-request state the model keeps, and records what each forward is
//! lent, so the engine's tests serve that cache through the engine on the meta path.
//!
//! The fields, over pages of [`PAGE`] tokens ([`fields`]):
//!
//! | Field | Owners | History | Entries per page | Record |
//! | --- | --- | --- | --- | --- |
//! | [`LATENT`], read as key and value | the 40 target layers and the 3 draft layers (40–42) | the last [`WINDOW`] tokens and the draft block's rollback | 64 | 512 E4M3 + 16 UE8M0, 528 B |
//! | [`COMPRESSED`] | layers 2, 8, 14 (one entry per two tokens) and 20 (one per token), read by 3–7, 9–13, 15–19 and 21–39 | full | 32 / 64 | NVFP4 (288 B) or MXFP8 (528 B) |
//! | [`INDEX_KEY`] | the same four, read by 24, 28, 32 and 36 from 20 | full | 32 / 64 | MXFP4 (68 B, a page's values then their scales) or MXFP8 (132 B) |
//!
//! and the compressor's remainder, a per-request ring of eight FP32 rows of 1024 on layers 2, 8
//! and 14 ([`remainder`]), its groups the ratio-2 compressor's pairs.
//!
//! [`LatentModel::drafting`] gives the stub a drafter of the block form (`nn::BlockDrafter`), as
//! DeepSeek-V4.1's DSpark drafter is, whose entries are the draft layers' window latent: it
//! records what each of its forwards is lent ([`LentDraft`]).

use std::sync::Mutex;

use nn::family::{
    CacheField, ExpertShape, FieldOwner, FieldReader, Format, History, INDEX_KEY, Layout, Record,
    StateField, StateRows, StateShape, TextModel, TokenHistory,
};
use nn::functional::{WeightOrder, argmax, embedding};
use nn::modules::Linear;
use nn::{
    BlockDrafter, CausalLm, DType, DraftBlock, DraftContext, Drafter, ExpertExchange, KvCache,
    RecurrentState, Tensor, TokenBatch,
};

/// Tokens of one page.
pub const PAGE: u32 = 64;
/// The tokens before a token the window latent's readers read, the token included.
pub const WINDOW: u32 = 128;
/// The target model's layers, and the drafter's that keep the window latent beside them.
pub const LAYERS: u32 = 40;
pub const DRAFT_LAYERS: u32 = 3;
/// The field of the window latent, both key and value of every layer's sliding window.
pub const LATENT: &str = "latent";
/// The field of the compressed latent the sparse attention reads.
pub const COMPRESSED: &str = "compressed_latent";
/// The width of a latent and of an index key.
const LATENT_WIDTH: u32 = 512;
const INDEX_WIDTH: u32 = 128;
/// The widths of the stub's activations and vocabulary.
const HIDDEN: usize = 16;
const VOCABULARY: usize = 64;

/// The record formats a deployment chooses for the compressed latent and the index keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Formats {
    /// NVFP4 or MXFP8.
    pub compressed: Format,
    /// MXFP4 or MXFP8.
    pub index: Format,
}

impl Formats {
    /// The checkpoint's own: an NVFP4 compressed latent and MXFP4 index keys.
    pub const REFERENCE: Self = Self {
        compressed: Format::NVFP4,
        index: Format::MXFP4,
    };

    /// The index keys' record: an MXFP4 page keeps every key's values and then their scales,
    /// as the paged index logits read it; an MXFP8 key keeps its scales after its values.
    pub const fn index_record(&self) -> Record {
        let layout = match self.index {
            Format::MXFP4 => Layout::Page,
            _ => Layout::Entry,
        };
        Record {
            layout,
            ..Record::rows(self.index, 1, INDEX_WIDTH)
        }
    }
}

/// The owners of the compressed latent and of the index keys: layers 2, 8 and 14 compress pairs
/// of tokens, layer 20 every token.
fn compressors() -> Vec<FieldOwner> {
    [(2, 2), (8, 2), (14, 2), (20, 1)]
        .into_iter()
        .map(|(layer, ratio)| FieldOwner { layer, ratio })
        .collect()
}

/// The layers `first..=last` as readers of `owner`.
fn readers(layers: std::ops::RangeInclusive<u32>, owner: u32) -> Vec<FieldReader> {
    layers.map(|layer| FieldReader { layer, owner }).collect()
}

/// DeepSeek-V4.1-Flash's cache fields (module documentation) for a deployment that drafts
/// `draft_tokens` tokens a step, whose rollback the window keeps, in `formats`.
pub fn fields(draft_tokens: u32, formats: Formats) -> Vec<CacheField> {
    let one = |layer| FieldOwner { layer, ratio: 1 };
    let compressed_readers = [
        readers(3..=7, 2),
        readers(9..=13, 8),
        readers(15..=19, 14),
        readers(21..=39, 20),
    ]
    .concat();
    let index_readers = [24, 28, 32, 36]
        .into_iter()
        .map(|layer| FieldReader { layer, owner: 20 })
        .collect();

    vec![
        CacheField {
            name: LATENT,
            owners: (0..LAYERS + DRAFT_LAYERS).map(one).collect(),
            readers: Vec::new(),
            history: History::Window {
                tokens: WINDOW + draft_tokens,
            },
            record: Record::rows(Format::MXFP8, 1, LATENT_WIDTH),
        },
        CacheField {
            name: COMPRESSED,
            owners: compressors(),
            readers: compressed_readers,
            history: History::Full,
            record: Record::rows(formats.compressed, 1, LATENT_WIDTH),
        },
        CacheField {
            name: INDEX_KEY,
            owners: compressors(),
            readers: index_readers,
            history: History::Full,
            record: formats.index_record(),
        },
    ]
}

/// The compressor's remainder on layers 2, 8 and 14: a ring of eight FP32 rows of 1024 per
/// request, in groups of the two tokens a compressed entry stands for.
pub fn remainder() -> StateShape {
    StateShape {
        layers: 3,
        fields: vec![StateField {
            name: "compressor_remainder",
            dtype: DType::F32,
            dims: vec![8, 1024],
            stepwise: false,
            rows: StateRows::Ring { group: 2 },
        }],
    }
}

/// What one forward was lent of one cache group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LentGroup {
    /// Each plane's field, owner, readers and extents.
    pub planes: Vec<(&'static str, u32, Vec<u32>, Vec<usize>)>,
    /// The extents of the group's page table rows.
    pub block_tables: Vec<usize>,
    /// The extents of its slot mapping.
    pub slot_mapping: Vec<usize>,
    /// The extents of its rows' starts, for a sliding window's group.
    pub starts: Option<Vec<usize>>,
}

/// What one forward of the stub's block drafter was lent: the context's groups and tokens, then
/// the blocks' groups and tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LentDraft {
    /// The groups the context is written through, and its tokens.
    pub context: (Vec<LentGroup>, usize),
    /// The groups the blocks attend through, and their tokens.
    pub block: (Vec<LentGroup>, usize),
}

/// What a forward was lent of `group`.
fn record(group: &KvCache) -> LentGroup {
    LentGroup {
        planes: group
            .planes
            .iter()
            .map(|plane| {
                (
                    plane.field,
                    plane.layer,
                    plane.readers.clone(),
                    plane.entries.dims().to_vec(),
                )
            })
            .collect(),
        block_tables: group.block_tables.dims().to_vec(),
        slot_mapping: group.slot_mapping.dims().to_vec(),
        starts: group.starts.as_ref().map(|starts| starts.dims().to_vec()),
    }
}

/// The stub model (module documentation).
pub struct LatentModel {
    draft_tokens: u32,
    formats: Formats,
    /// The tokens of a block its drafter of the block form drafts, zero for a stub without one.
    block: u32,
    table: Tensor,
    head: Linear,
    /// What each forward was lent, group by group, in the order of the forwards.
    pub lent: Mutex<Vec<Vec<LentGroup>>>,
    /// What each forward of its block drafter was lent, in their order.
    pub drafted: Mutex<Vec<LentDraft>>,
}

/// A meta tensor of `shape` and `dtype`: extents and an element type at an address nothing
/// reads.
fn meta(shape: impl Into<nn::Shape>, dtype: DType) -> nn::Result<Tensor> {
    Ok(ops::retained(0x1_0000, shape, dtype, &ops::meta())?)
}

impl LatentModel {
    /// The stub of a deployment that drafts `draft_tokens` tokens a step, in `formats`, over
    /// meta weights.
    ///
    /// # Errors
    ///
    /// When a meta tensor cannot be made.
    pub fn new(draft_tokens: u32, formats: Formats) -> nn::Result<Self> {
        Ok(Self {
            draft_tokens,
            formats,
            block: 0,
            table: meta((VOCABULARY, HIDDEN), DType::BF16)?,
            head: Linear::new(
                meta((VOCABULARY, HIDDEN), DType::BF16)?,
                None,
                WeightOrder::NxK,
            )?,
            lent: Mutex::new(Vec::new()),
            drafted: Mutex::new(Vec::new()),
        })
    }

    /// The stub with a drafter of the block form that drafts blocks of `block` tokens in its own
    /// cache fields, the window latent's draft layers, as DeepSeek-V4.1's DSpark drafter does.
    ///
    /// # Errors
    ///
    /// When a meta tensor cannot be made.
    pub fn drafting(block: u32, formats: Formats) -> nn::Result<Self> {
        Ok(Self {
            block,
            ..Self::new(block, formats)?
        })
    }

    /// What its block drafter's forwards so far were lent.
    ///
    /// # Panics
    ///
    /// When a forward panicked holding the record.
    pub fn drafts(&self) -> Vec<LentDraft> {
        #[expect(clippy::unwrap_used, reason = "a test's record")]
        self.drafted.lock().unwrap().clone()
    }

    /// What the forwards so far were lent.
    ///
    /// # Panics
    ///
    /// When a forward panicked holding the record.
    pub fn forwards(&self) -> Vec<Vec<LentGroup>> {
        #[expect(clippy::unwrap_used, reason = "a test's record")]
        self.lent.lock().unwrap().clone()
    }
}

impl CausalLm for LatentModel {
    fn embed_input_ids(&self, ids: &Tensor) -> nn::Result<Tensor> {
        embedding(ids, &self.table, 0)
    }

    /// Records the groups it is lent and returns the embeddings: the full history's and the
    /// window's, each group's columns checked against the batch, and the remainder's state.
    fn forward(
        &self,
        batch: &TokenBatch,
        cache: &[KvCache],
        state: Option<&RecurrentState>,
        _: Option<&ExpertExchange>,
        hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        let [full, window] = cache else {
            return Err(nn::Error::Unsupported(format!(
                "the stub reads the full history's group and the window's, and was lent {}",
                cache.len()
            )));
        };
        if state.is_none_or(|state| state.live.len() != 3) {
            return Err(nn::Error::Unsupported(
                "the stub's compressors keep their remainder's state, which was not lent".into(),
            ));
        }
        for group in [full, window] {
            group.check(batch)?;
        }

        let mut lent = self
            .lent
            .lock()
            .map_err(|_| nn::Error::Unsupported("the record of a panicked forward".into()))?;
        lent.push(vec![record(full), record(window)]);
        Ok(hidden.clone())
    }

    fn compute_logits(&self, hidden: &Tensor) -> nn::Result<Tensor> {
        Ok(self.head.forward(hidden)?.to_dtype(DType::F32)?)
    }

    fn join_expert_step(&self, exchange: &ExpertExchange) -> nn::Result<()> {
        exchange.check()
    }
}

impl TextModel for LatentModel {
    fn hidden(&self) -> u32 {
        HIDDEN as u32
    }

    fn vocabulary(&self) -> u32 {
        VOCABULARY as u32
    }

    fn dtype(&self) -> DType {
        DType::BF16
    }

    fn cache(&self) -> Vec<CacheField> {
        fields(self.draft_tokens, self.formats)
    }

    fn state_layers(&self) -> Vec<StateShape> {
        vec![remainder()]
    }

    fn experts(&self) -> Option<ExpertShape> {
        None
    }

    fn drafter(&self) -> Option<&dyn Drafter> {
        None
    }

    fn block_drafter(&self) -> Option<&dyn BlockDrafter> {
        (self.block > 0).then_some(self as &dyn BlockDrafter)
    }

    fn token_history(&self) -> Option<TokenHistory> {
        None
    }
}

/// The stub's drafter of the block form: it checks each group it is lent against the batch it
/// is lent for, records them, and returns the blocks' embeddings as their rows.
impl BlockDrafter for LatentModel {
    fn block(&self) -> u32 {
        self.block
    }

    fn fill(&self) -> u32 {
        VOCABULARY as u32 - 1
    }

    fn forward(
        &self,
        context: &DraftContext<'_>,
        block: &DraftBlock<'_>,
        _: Option<&ExpertExchange>,
    ) -> nn::Result<Tensor> {
        for group in context.cache {
            group.check(context.batch)?;
        }
        for group in block.cache {
            group.check(block.batch)?;
        }
        let (tokens, blocks) = (context.batch.tokens(), block.batch.tokens());
        if context.hidden.dims() != [tokens, HIDDEN]
            || block.embedded.dims() != [blocks, HIDDEN]
            || !blocks.is_multiple_of(self.block as usize)
        {
            return Err(nn::Error::Unsupported(format!(
                "a context of {tokens} tokens with rows {:?} and blocks of {blocks} tokens with \
                 embeddings {:?}",
                context.hidden.dims(),
                block.embedded.dims()
            )));
        }

        let mut drafted = self
            .drafted
            .lock()
            .map_err(|_| nn::Error::Unsupported("the record of a panicked forward".into()))?;
        drafted.push(LentDraft {
            context: (context.cache.iter().map(record).collect(), tokens),
            block: (block.cache.iter().map(record).collect(), blocks),
        });
        Ok(block.embedded.clone())
    }

    /// The argmax of each block token's logits.
    fn propose(&self, logits: &Tensor, anchors: &Tensor) -> nn::Result<Tensor> {
        let rows = anchors.dim(0)?;
        Ok(argmax(logits)?.reshape((rows, self.block as usize))?)
    }
}
