//! What the crate's integration tests share: a checkpoint loaded through the family, and the
//! batch, cache and state tensors a forward is lent.
//!
//! Each test file that declares `mod support;` compiles its own copy and uses part of it.
#![allow(dead_code, reason = "each test crate uses part of the shared support")]

use std::path::Path;
use std::sync::Arc;

use nn::family::{CacheShape, ExpertOrder, Model, Registry, StateShape, TextModel, TextPlacement};
use nn::{
    Checkpoint, DType, KvCache, KvPlanes, PhaseKind, RecurrentState, Shape, Split, Tensor,
    TokenBatch, VerifySlots, WeightLoader,
};

/// The causal language model of `checkpoint`, opened and loaded as execution does: the
/// checkpoint's declared class names its family, the family opens it, and the opened
/// checkpoint loads `placement`'s share through `loader`.
pub(crate) fn load(
    checkpoint: &Checkpoint,
    loader: &mut WeightLoader<'_>,
    placement: &TextPlacement,
) -> nn::Result<Arc<dyn TextModel>> {
    let config = checkpoint
        .sidecar("config.json")
        .expect("a readable config.json")
        .expect("a config.json");
    let class = config["architectures"][0]
        .as_str()
        .expect("a declared class");

    let registry = Registry::new([qwen4_exp::family::FAMILY]).expect("one family");
    let family = registry
        .by_class(class)
        .unwrap_or_else(|| panic!("no family claims {class}"));
    let Model::Text(family) = family.model else {
        panic!("{class} is a text family");
    };

    let opened = family.open(checkpoint)?;
    opened.load(checkpoint, loader, placement, &ExpertOrder::new())
}

/// The checkpoint at `root`.
pub(crate) fn checkpoint(root: &Path) -> Checkpoint {
    Checkpoint::open(root).unwrap_or_else(|error| panic!("{}: {error}", root.display()))
}

/// A deployment at `tensor`'s rank that drafts `draft_tokens` tokens a step with the
/// checkpoint's drafter, or none without one.
pub(crate) fn placement(tensor: Split, draft_tokens: u32) -> TextPlacement {
    TextPlacement {
        tensor,
        experts: Split::WHOLE,
        drafter: draft_tokens > 0,
        draft_tokens,
    }
}

/// The index columns of one batch of `phase` over pages of `page_size` tokens whose rows read
/// at most `max_kv_len` keys: `positions`, `slot_mapping`, `seq_lens`, `query_offsets` and
/// `logits_rows`, in that order, as `columns` holds them, with the tokens' ids and the two ids
/// before each row as `ngram` holds them.
pub(crate) fn batch(
    phase: PhaseKind,
    columns: [Tensor; 5],
    ngram: Option<(Tensor, Tensor)>,
    page_size: u32,
    max_kv_len: u32,
) -> TokenBatch {
    let [
        positions,
        slot_mapping,
        seq_lens,
        query_offsets,
        logits_rows,
    ] = columns;
    let num_tokens = positions.elem_count() as u32;
    let (token_ids, ngram_context) = ngram.unzip();

    TokenBatch {
        positions,
        slot_mapping,
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
        ngram_context,
    }
}

/// One cache group of `shape`'s layers over `pages` pages of `page_size` tokens, its planes of
/// `dtype`, each tensor made by `tensor` from its extents and element type; `block_tables` is
/// the `[rows, row_pages]` table of the rows' pages.
pub(crate) fn cache(
    shape: CacheShape,
    dtype: DType,
    (pages, page_size): (usize, u32),
    block_tables: Tensor,
    tensor: &dyn Fn(Shape, DType) -> Tensor,
) -> KvCache {
    let plane = Shape::from((
        pages,
        page_size as usize,
        shape.kv_heads as usize,
        shape.head_dim as usize,
    ));
    let flags = Shape::from((pages, 1, 1, 1));

    KvCache {
        block_tables,
        page_size,
        layers: (0..shape.layers)
            .map(|_| KvPlanes {
                key: tensor(plane.clone(), dtype),
                value: tensor(plane.clone(), dtype),
                key_initialized: tensor(flags.clone(), DType::U8),
                value_initialized: tensor(flags.clone(), DType::U8),
                key_scale: None,
                value_scale: None,
                index_keys: shape.index_keys.map(|keys| {
                    let groups = (page_size / keys.ratio) as usize;
                    tensor(Shape::from((pages, groups, keys.width as usize)), dtype)
                }),
            })
            .collect(),
        workspace: tensor(Shape::from(1 << 20), DType::U8),
    }
}

/// The buffers of every layer of `shapes` with `leading` rows each (request slots), one list
/// of fields per recurrent layer, in the state's layer order.
pub(crate) fn state_buffers(
    shapes: &[StateShape],
    leading: usize,
    tensor: &dyn Fn(Shape, DType) -> Tensor,
) -> Vec<Vec<Tensor>> {
    shapes
        .iter()
        .flat_map(|shape| {
            (0..shape.layers).map(|_| {
                shape
                    .fields
                    .iter()
                    .map(|field| {
                        let dims = [&[leading][..], field.dims.as_slice()].concat();
                        tensor(Shape::from(dims), field.dtype)
                    })
                    .collect()
            })
        })
        .collect()
}

/// The recurrent state of one batch over the buffers `live` and `checkpoint`: `columns` holds
/// the state rows' `query_offsets`, `seq_lens`, `slots` and `checkpoints`, in that order, and
/// `verify` the rows' slots and standing tokens when a request owns several.
pub(crate) fn state(
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
