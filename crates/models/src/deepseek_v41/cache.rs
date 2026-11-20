//! What DeepSeek-V4.1's layers keep per request: its cache fields and the compressor's
//! remainder, as the model declares them to execution ([`nn::family::CacheField`],
//! [`nn::family::StateShape`]).
//!
//! | Field | Owners | Readers | History | Entries per page of 64 | Record |
//! | --- | --- | --- | --- | --- | --- |
//! | [`WINDOW`], read as key and value | every layer, and each DSpark layer after them | — | the last `sliding_window` tokens | 64 | MXFP8: 512 E4M3 then 16 UE8M0 (528 B) |
//! | [`COMPRESSED`] | the compressed sources (2, 8, 14 of ratio 2; 20 of ratio 1) | the layers after each source of its ratio | full | 32 / 64 | NVFP4: 256 B of E2M1 pairs then 32 E4M3 scales (288 B) |
//! | [`INDEX_KEY`] | the same sources | the index sources that read a source's keys (24, 28, 32, 36 read 20's) | full | 32 / 64 | MXFP4 in the page layout: a page's 64 B of E2M1 pairs per key, then its 4 UE8M0 scales per key (68 B a key) |
//!
//! The formats are the reference's own (`inference/model.py:705-767`): the window latent is
//! fake-quantized by `act_quant` (MXFP8 per 32), a compressed latent by `fp4_act_quant` per 16
//! with E4M3 scales, an index key by `fp4_act_quant` per 32 with UE8M0 scales. Decoding a record
//! is that fake quantization exactly, so the attention reads the values the reference attends.
//! The index keys' page layout is the one their writer and their reader take from this
//! declaration ([`nn::family::Record::layout`]), the layout DeepGEMM's paged index logits read.
//!
//! A ratio-2 source keeps a ring of the projections of its request's last tokens, values then
//! gates in FP32, from which the group its next tokens complete is pooled ([`remainder`]).
//!
//! The DSpark layers (`crate::deepseek_v41::dspark`) keep their windows in the same field, as
//! owners `num_hidden_layers + i`: each holds the latents its layer projects from the decoder's
//! rows of the request's tokens, and its next block's while the block is drafted. A checkpoint
//! that carries the drafter declares them whether or not a placement loads it, so the window's
//! pages are laid out alike for a deployment that speculates and one that does not, and a
//! kernel generation that starts drafting finds them. The block's positions lie at most a
//! verification's drafts past its tokens, which the window's pages a verification holds cover.

use nn::family::{
    CacheField, FieldOwner, FieldReader, Format, History, INDEX_KEY, Layout, Record, StateField,
    StateRows, StateShape,
};
use nn::{DType, Result};

use crate::deepseek_v41::ID;
use crate::deepseek_v41::config::Config;

/// The field of every layer's window latent: the key and the value of its sliding window.
pub const WINDOW: &str = "window_latent";

/// The field of the compressed latents the sparse attention selects among.
pub const COMPRESSED: &str = "compressed_latent";

/// A window latent: one MXFP8 row of the head width.
pub const WINDOW_RECORD: Record = Record::rows(Format::MXFP8, 1, 512);

/// A compressed latent: one NVFP4 row of the head width.
pub const COMPRESSED_RECORD: Record = Record::rows(Format::NVFP4, 1, 512);

/// An index key: one MXFP4 row of the index head width, its page's scales after its values.
pub const INDEX_RECORD: Record = Record {
    format: Format::MXFP4,
    layout: Layout::Page,
    heads: 1,
    width: 128,
};

/// The rows of a source's compressor ring: a whole number of groups of either ratio that holds
/// the open group of a step's first token and the tokens a step adds before it closes.
const RING: usize = 8;

/// The model's cache fields (module documentation).
///
/// # Errors
///
/// A configuration whose widths are not the records' (the checkpoint's 512-wide latent and
/// 128-wide index keys).
pub fn fields(config: &Config) -> Result<Vec<CacheField>> {
    if config.attention.head_dim != WINDOW_RECORD.width
        || config.indexer.head_dim != INDEX_RECORD.width
    {
        return Err(ID.config(format!(
            "latents of {} and index keys of {}; the cache's records hold {} and {}",
            config.attention.head_dim,
            config.indexer.head_dim,
            WINDOW_RECORD.width,
            INDEX_RECORD.width
        )));
    }

    let layers = config.num_hidden_layers;
    let owners: Vec<FieldOwner> = config
        .compressors()
        .into_iter()
        .map(|(layer, ratio)| FieldOwner { layer, ratio })
        .collect();
    let compressed_readers = (0..layers)
        .filter_map(|layer| match config.layers[layer as usize].compressed {
            Some(owner) if owner != layer => Some(FieldReader { layer, owner }),
            _ => None,
        })
        .collect();
    let index_readers = (0..layers)
        .filter_map(|layer| {
            let sources = &config.layers[layer as usize];
            match sources.compressed {
                Some(owner) if sources.indexes && owner != layer => {
                    Some(FieldReader { layer, owner })
                }
                _ => None,
            }
        })
        .collect();

    Ok(vec![
        CacheField {
            name: WINDOW,
            owners: (0..layers)
                .chain(config.draft_layers())
                .map(|layer| FieldOwner { layer, ratio: 1 })
                .collect(),
            readers: Vec::new(),
            history: History::Window {
                tokens: config.attention.window,
            },
            record: WINDOW_RECORD,
        },
        CacheField {
            name: COMPRESSED,
            owners: owners.clone(),
            readers: compressed_readers,
            history: History::Full,
            record: COMPRESSED_RECORD,
        },
        CacheField {
            name: INDEX_KEY,
            owners,
            readers: index_readers,
            history: History::Full,
            record: INDEX_RECORD,
        },
    ])
}

/// The sources that pool groups of tokens, in layer order: the layers whose rings the
/// recurrent state holds, its first entry the first of them.
pub fn pooling(config: &Config) -> Vec<u32> {
    config
        .compressors()
        .into_iter()
        .filter(|&(_, ratio)| ratio > 1)
        .map(|(layer, _)| layer)
        .collect()
}

/// The compressor's remainder on each source that pools groups of tokens ([`pooling`]): a ring of
/// [`RING`] FP32 rows of `2 · head_dim` (the values, then the gates) per request, position `p` at
/// row `p mod 8`, in groups of the ratio. A request resumed at a page boundary reads none of it:
/// every group of a page is complete there.
pub fn remainder(config: &Config) -> Vec<StateShape> {
    let pooling = pooling(config);
    let Some(&first) = pooling.first() else {
        return Vec::new();
    };
    vec![StateShape {
        layers: pooling.len() as u32,
        fields: vec![StateField {
            name: "compressor_remainder",
            dtype: DType::F32,
            dims: vec![RING, 2 * config.attention.head_dim as usize],
            stepwise: false,
            rows: StateRows::Ring {
                group: config.layers[first as usize].ratio as usize,
            },
        }],
    }]
}
