//! What the family's tests share: a checkpoint loaded through the family, and the
//! batch, cache and state tensors a forward is lent.

use std::path::Path;
use std::sync::Arc;

use nn::family::{CacheShape, ExpertOrder, Model, Registry, StateShape, TextModel, TextPlacement};
use nn::{
    Checkpoint, DType, KvCache, KvPlanes, Phase, RecurrentState, Shape, Tensor, TokenBatch,
    WeightLoader,
};
use proto::config::Quantization;

/// The causal language model of the checkpoint at `root`, opened and loaded as execution does:
/// the checkpoint's declared class names its family, the family opens it, and the opened
/// checkpoint loads `placement`'s share through `loader`.
pub(crate) fn load(
    checkpoint: &Checkpoint,
    loader: &mut WeightLoader<'_>,
    placement: &TextPlacement,
) -> Arc<dyn TextModel> {
    let config = checkpoint
        .sidecar("config.json")
        .expect("a readable config.json")
        .expect("a config.json");
    let class = config["architectures"][0]
        .as_str()
        .expect("a declared class");

    let registry = Registry::new([
        models::qwen3::family::FAMILY,
        models::qwen3::family::HYBRID_FAMILY,
    ])
    .expect("two families of distinct classes");
    let family = registry
        .by_class(class)
        .unwrap_or_else(|| panic!("no family claims {class}"));
    let Model::Text(family) = family.model else {
        panic!("{class} is a text family");
    };

    let opened = family
        .open(checkpoint, &Quantization::default())
        .expect("the configuration");
    opened
        .load(checkpoint, loader, placement, &ExpertOrder::new())
        .expect("the load")
}

/// The checkpoint at `root`.
pub(crate) fn checkpoint(root: &Path) -> Checkpoint {
    Checkpoint::open(root).unwrap_or_else(|error| panic!("{}: {error}", root.display()))
}

/// A deployment of one rank, with the drafter when `drafter` asks for it.
pub(crate) fn whole(drafter: bool) -> TextPlacement {
    TextPlacement {
        tensor: nn::Split::WHOLE,
        pipeline: nn::Split::WHOLE,
        sequence: nn::family::SequenceSplit::WHOLE,
        experts: nn::Split::WHOLE,
        tensor_experts: false,
        drafter,
        draft_tokens: 0,
    }
}

/// The index columns of one batch of `phase` over pages of `page_size` tokens whose rows read
/// at most `max_kv_len` keys: `positions`, `slot_mapping`, `seq_lens`, `query_offsets` and
/// `logits_rows`, in that order, as `columns` holds them.
pub(crate) fn batch(
    phase: Phase,
    columns: [Tensor; 5],
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
        token_ids: None,
        ngram_context: None,
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
                index_keys: None,
            })
            .collect(),
        workspace: tensor(Shape::from(1 << 20), DType::U8),
    }
}

/// The buffers of `shape`'s fields with `leading` rows each (request slots, or the tokens of a
/// trail), one list of fields per recurrent layer.
pub(crate) fn state_buffers(
    shape: &StateShape,
    leading: usize,
    tensor: &dyn Fn(Shape, DType) -> Tensor,
) -> Vec<Vec<Tensor>> {
    (0..shape.layers)
        .map(|_| {
            shape
                .fields
                .iter()
                .map(|field| {
                    let dims = [&[leading][..], field.dims.as_slice()].concat();
                    tensor(Shape::from(dims), field.dtype)
                })
                .collect()
        })
        .collect()
}

/// The recurrent state of one batch over the buffers `live` and `checkpoint`: `columns` holds
/// the state rows' `query_offsets`, `seq_lens`, `slots` and `checkpoints`, in that order.
pub(crate) fn state(
    columns: [Tensor; 4],
    live: &[Vec<Tensor>],
    checkpoint: &[Vec<Tensor>],
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
        verify: None,
    }
}
