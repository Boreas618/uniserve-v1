//! What a causal language model's forward is lent, and the model itself loaded as execution
//! loads it.
//!
//! The cache and the state buffers a forward is lent are built from the model's own shapes by
//! `nn::KvCache::planes_of` and `nn::RecurrentState::buffers`; the values here are the batch and
//! the state of one forward, assembled from the columns a test states.

use std::sync::Arc;

use nn::family::{
    CacheField, ExpertOrder, Family, KEY, Model, Registry, TextCheckpoint, TextModel,
    TextPlacement, VALUE,
};
use nn::{
    Checkpoint, DType, KvCache, Phase, RecurrentState, Shape, Tensor, TokenBatch, VerifySlots,
    WeightLoader,
};
use proto::config::Quantization;

/// Tokens of one cache page: the page a deployment's cache holds when its
/// `kv_cache_config.block_size` states none, the largest the attention packages read. It holds
/// whole chunks of the gated delta rule and whole groups of an indexer.
pub const PAGE_TOKENS: u32 = 64;

/// The most (row, query head) pairs a test's batch attends: a decode bucket of 4096 rows over
/// 64 query heads.
const ATTENDED_PAIRS: u64 = 4096 * 64;

/// Bytes of the attention workspace a test lends a cache group (`nn::KvCache::workspace`).
///
/// Execution reserves the workspace the loaded kernel needs; what a forward itself touches is
/// the decode's arrival counters at its head, which it clears once per forward. A workspace of
/// this size holds the counters of every batch a test attends
/// ([`ops::attention::counter_bytes`] of [`ATTENDED_PAIRS`]), the meta device stating no
/// multiprocessors of its own, so a profile run clears them as a deployment does.
pub const WORKSPACE_BYTES: usize = ops::attention::counter_bytes(ATTENDED_PAIRS) as usize;

/// The causal language model of `checkpoint`, opened ([`open`]) and loaded as execution does:
/// the opened checkpoint loads `placement`'s share through `loader`.
///
/// # Errors
///
/// The family's refusal to open or to load the checkpoint.
pub fn load(
    families: &[Family],
    checkpoint: &Checkpoint,
    loader: &mut WeightLoader<'_>,
    placement: &TextPlacement,
) -> nn::Result<Arc<dyn TextModel>> {
    open(families, checkpoint)?.load(checkpoint, loader, placement, &ExpertOrder::new())
}

/// The checkpoint opened by the family of `families` it declares, as execution opens it: the
/// registry of `families` resolves the class the checkpoint's `config.json` declares, and the
/// family reads the checkpoint's configuration under no precision policy.
///
/// # Errors
///
/// The family's refusal to open the checkpoint, such as a representation no operator serves.
pub fn open(families: &[Family], checkpoint: &Checkpoint) -> nn::Result<Box<dyn TextCheckpoint>> {
    let config = checkpoint
        .sidecar("config.json")?
        .unwrap_or_else(|| panic!("{} holds no config.json", checkpoint.root().display()));
    let class = config["architectures"][0]
        .as_str()
        .unwrap_or_else(|| panic!("{} declares no class", checkpoint.root().display()));

    let registry = Registry::new(families.iter().copied())?;
    let family = registry
        .by_class(class)
        .unwrap_or_else(|| panic!("no family of {families:?} claims {class}"));
    let Model::Text(family) = family.model else {
        panic!("{class} is not a text family");
    };

    family.open(checkpoint, &Quantization::default())
}

/// The batch of one forward of `phase` over pages of `page_size` tokens whose rows read at most
/// `max_kv_len` keys: `columns` holds its `positions`, `seq_lens`, `query_offsets` and
/// `logits_rows`, in that order, and `ngram` the tokens' ids and the ids before each row
/// (`TokenBatch::token_history`) for a model that reads them. The slots its tokens write are
/// each cache group's ([`nn::KvCache::slot_mapping`]).
pub fn batch(
    phase: Phase,
    columns: [Tensor; 4],
    ngram: Option<(Tensor, Tensor)>,
    page_size: u32,
    max_kv_len: u32,
) -> TokenBatch {
    let [positions, seq_lens, query_offsets, logits_rows] = columns;
    let num_tokens = positions.elem_count() as u32;
    let (token_ids, token_history) = ngram.unzip();

    TokenBatch {
        positions,
        seq_lens,
        query_offsets,
        logits_rows,
        phase,
        page_size,
        num_tokens,
        max_kv_len,
        draft_lens: None,
        rope_positions: None,
        live_tokens: None,
        token_ids,
        token_history,
    }
}

/// The cache group of `fields` over `pages` pages of `page_size` tokens in `dtype`, as one
/// forward is lent it: rows address pages through `block_tables` and its tokens write the slots
/// `slot_mapping` names, beside the attention `workspace`. `tensor` makes each plane's tensors
/// ([`KvCache::planes_of`]); the group is the full history's, so it states no window starts.
///
/// # Errors
///
/// Fields `KvCache::planes_of` refuses, or what `tensor` returns.
pub fn cache(
    fields: &[CacheField],
    dtype: DType,
    (pages, page_size): (usize, u32),
    (block_tables, slot_mapping, workspace): (Tensor, Tensor, Tensor),
    tensor: impl FnMut(Shape, DType) -> nn::Result<Tensor>,
) -> nn::Result<KvCache> {
    Ok(KvCache {
        block_tables,
        slot_mapping,
        starts: None,
        page_size,
        planes: KvCache::planes_of(fields, dtype, pages, page_size, tensor)?,
        workspace,
    })
}

/// The attention layers, the key/value heads of each and their width, as a model of separate
/// keys and values states its [`KEY`] and [`VALUE`] fields, the first two of `fields`.
///
/// # Panics
///
/// When the first two fields are not the keys and the values, or state other owners or records.
pub fn key_value_shape(fields: &[CacheField]) -> (u32, u32, u32) {
    let [key, value, ..] = fields else {
        panic!("fields {fields:?} hold no keys and values");
    };
    assert_eq!((key.name, value.name), (KEY, VALUE), "{fields:?}");
    assert_eq!((&key.owners, key.record), (&value.owners, value.record));

    (key.owners.len() as u32, key.record.heads, key.record.width)
}

/// The recurrent state of one forward over the buffers `live` and `checkpoint`: `columns` holds
/// the state rows' `query_offsets`, `seq_lens`, `slots` and `checkpoints`, in that order, and
/// `verify` the rows' slots and standing tokens when a request owns several.
pub fn state(
    columns: [Tensor; 4],
    live: &[Vec<Tensor>],
    checkpoint: &[Vec<Tensor>],
    verify: Option<(Tensor, Tensor)>,
) -> RecurrentState {
    let [query_offsets, seq_lens, slots, checkpoints] = columns;

    RecurrentState {
        query_offsets,
        seq_lens,
        slots,
        checkpoints,
        live: live.to_vec(),
        checkpoint: checkpoint.to_vec(),
        trail: None,
        verify: verify.map(|(state_indices, num_accepted)| VerifySlots {
            state_indices,
            num_accepted,
        }),
    }
}
