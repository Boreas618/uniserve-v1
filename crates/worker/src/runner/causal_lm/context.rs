//! Prefill calls that place image features in a request's sequence.
//!
//! A *context* prefill (`Call::writes_context`) holds a prompt with vision
//! blocks: its rows alternate runs of prompt text (causal) and one block
//! per `VisionInput` (non-causal), at increasing prefix cache and position
//! coordinates; a block may advance the rotary position by less than its
//! token count. A *visual-state* prefill (`Call::writes_visual_state`)
//! places exactly one feature after a generated or ingested image: a vision
//! feature closes the image as one block row; a latent feature (an image
//! denoiser's conditioning) writes its framed latent sequence's K/V through
//! the image denoiser.
//!
//! Every row of these calls carries explicit positions on the three rotary
//! axes (`RowSource::Placed`; text and markers take spatial zeros), so they
//! run eagerly with their features replacing the block's placeholder
//! embeddings. The logical position after the call is the temporal position
//! after its last row; a visual-state row advances it by the image
//! denoiser's rope advance when it completes a feedback.

use std::sync::Arc;
use uniserve_cuda::DeviceId;

use uniserve_ipc::{Call, ForwardBatch};

use crate::executor::PendingOutput;
use crate::executor::image::ImageParts;
use crate::ffi::MemoryKind;
use crate::lane::Recorder;
use crate::runner::TokenSelection;
use crate::runner::causal_lm::{RowSource, TokenRow};
use crate::runner::denoiser::image::ConditionRow;
use crate::runner::denoiser::image_model::GridKey;
use crate::runner::vision::{BlockPositions, PositionLayout, VisionBlock};
use crate::storage::memory::StorageView;
use crate::storage::{BlockTables, Metadata, ProductRead, ProductStore};
use crate::{Error, Result};

/// How a media call's rows advance its request.
pub(super) struct Media {
    /// Rows before the call's last row, in sequence order.
    pub(super) rows: Vec<TokenRow>,
    /// Vision features by their first token, counted from the call's first
    /// row.
    pub(super) embeddings: Vec<(usize, StorageView)>,
    /// The feature leases, held until the batch completes.
    pub(super) reads: Vec<Arc<ProductRead>>,
    /// The logical position after the call.
    pub(super) position: u64,
    /// The position the slot continues from when the call samples.
    pub(super) continued: u64,
    /// The semantic position of the call's sample.
    pub(super) sample: u64,
    /// A latent feature conditioning the image denoiser writes instead of
    /// text rows.
    pub(super) latent: Option<ConditionRow>,
}

/// What a media call borrows from the rank.
pub(super) struct Sources<'a> {
    pub(super) tensors: &'a ProductStore,
    pub(super) image: Option<&'a ImageParts<'a>>,
    pub(super) tables: &'a BlockTables,
    pub(super) device: Option<DeviceId>,
    /// What the batch issues its stream work to: the stream that consumes
    /// the features waits on their producers.
    pub(super) stream: Recorder<'a>,
    /// The request's admitted input images.
    pub(super) input_images: u32,
}

impl Sources<'_> {
    /// Whether rows carry three rotary axes
    /// ([`crate::runner::denoiser::image_model::ImageModel::spatial_positions`]).
    fn spatial(&self) -> bool {
        self.image
            .is_some_and(|image| image.facts().spatial_positions())
    }
}

/// One consumed feature and its image dimensions.
struct Feature {
    read: Arc<ProductRead>,
    height: u32,
    width: u32,
}

fn consume(sources: &Sources<'_>, call: &Call, index: usize, latent: bool) -> Result<Feature> {
    let reference = if latent {
        call.latent_feature_input.as_ref()
    } else {
        call.vision_inputs.get(index).map(|input| &input.feature)
    }
    .ok_or_else(|| Error::Invalid("visual input names no feature".into()))?;
    let read = sources.tensors.consume(
        reference,
        call.call_id,
        Some(sources.device.map_or(MemoryKind::Host, MemoryKind::Cuda)),
        sources.stream,
    )?;
    match read.metadata() {
        Some(Metadata::Feature { height, width }) => Ok(Feature {
            read,
            height,
            width,
        }),
        _ => Err(Error::Invalid(
            "visual input requires encoder feature metadata".into(),
        )),
    }
}

/// The temporal position after a row.
pub(super) fn next_position(row: &TokenRow) -> Result<u64> {
    let temporal = match &row.source {
        RowSource::Placed { positions, .. } => positions[0].iter().copied().max(),
        RowSource::Host { position, tokens } => Some(*position as i64 + tokens.len() as i64 - 1),
        _ => None,
    };
    temporal
        .map(|value| value as u64 + 1)
        .ok_or_else(|| Error::Invariant("a context row has no positions".into()))
}

/// The row of prompt `tokens` after `prefix` visible tokens of `slot`, at
/// consecutive temporal positions from `position` and zero spatial ones.
pub(super) fn text_row(
    slot: u32,
    tokens: &[u32],
    prefix: u64,
    position: u64,
    selection: TokenSelection,
    spatial: bool,
) -> Result<TokenRow> {
    let count = tokens.len();
    let start = position as i64;
    Ok(TokenRow {
        slot,
        source: RowSource::Placed {
            tokens: tokens.to_vec(),
            positions: [
                (start..start + count as i64).collect(),
                vec![0; count],
                vec![0; count],
            ],
            spatial,
        },
        prefix: narrow(prefix)?,
        query: narrow(count as u64)?,
        selection,
        gated: false,
        force_finish: false,
        causal: true,
    })
}

fn narrow(value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Invalid("context row exceeds the token range".into()))
}

/// A vision block row of `feature` at `position` after `prefix` visible
/// tokens; returns the row and the block's features.
fn vision_row(
    sources: &Sources<'_>,
    feature: &Feature,
    slot: u32,
    prefix: u64,
    position: u64,
    close_image: bool,
    selection: TokenSelection,
) -> Result<(TokenRow, usize, StorageView)> {
    let image = sources
        .image
        .ok_or_else(|| Error::Unsupported("vision input requires the image runtime".into()))?;
    let injection = image
        .facts()
        .processing
        .as_ref()
        .and_then(|processing| processing.injection)
        .ok_or_else(|| {
            Error::Invalid("vision state preparation requires declared feature injection".into())
        })?;
    if prefix > u64::from(sources.tables.allocated_length(slot)) {
        return Err(Error::Invalid(
            "vision row prefix lies outside the request's cache extent".into(),
        ));
    }
    if !close_image && sources.input_images == 0 {
        return Err(Error::Invalid(
            "request admission declares no input images".into(),
        ));
    }
    let values = feature.read.value();
    let [tokens, _] = values.shape()[..] else {
        return Err(Error::Invalid(
            "vision features must have shape [tokens, hidden]".into(),
        ));
    };
    // Spatial positions place the features on the image's patch grid, a
    // fact of the model asked when the batch's inputs were admitted.
    let grid = match injection.positions {
        PositionLayout::TemporalSpatial => {
            let key = grid_key(
                feature.height,
                feature.width,
                close_image,
                sources.input_images,
            );
            Some(image.policy().grid(key).ok_or_else(|| {
                Error::Invariant("an image's patch grid was not asked at admission".into())
            })?)
        }
        PositionLayout::Temporal | PositionLayout::Sequential => None,
    };
    let block = injection.block(tokens as usize, position, close_image, grid)?;
    let first = block.features.start;
    let row = block_row(block, slot, prefix, sources.spatial(), selection)?;
    Ok((row, first, values.clone()))
}

/// The row of `block` after `prefix` visible tokens of `slot`: its marker
/// and placeholder tokens at the block's positions, attending to all of
/// itself. `spatial` gives the row three rotary axes.
pub(super) fn block_row(
    block: VisionBlock,
    slot: u32,
    prefix: u64,
    spatial: bool,
    selection: TokenSelection,
) -> Result<TokenRow> {
    let count = block.len();
    let positions = match block.positions {
        BlockPositions::Linear(values) => [values, vec![0; count], vec![0; count]],
        BlockPositions::Spatial(values) => values,
    };
    let tokens: Vec<u32> = block
        .token_ids
        .iter()
        .map(|&token| {
            u32::try_from(token).map_err(|_| Error::Invalid("vision marker overflows".into()))
        })
        .collect::<Result<_>>()?;
    Ok(TokenRow {
        slot,
        source: RowSource::Placed {
            tokens,
            positions,
            spatial,
        },
        prefix: narrow(prefix)?,
        query: narrow(count as u64)?,
        selection,
        gated: false,
        force_finish: false,
        causal: false,
    })
}

/// The rows of a context prefill: its prompt runs and vision blocks, the
/// last returned separately. `descriptors` are the scheduler's forward rows
/// of the call, which must match its rows. A call that `scores` its prompt
/// projects every text token and each block's last token.
pub(super) fn context(
    sources: &Sources<'_>,
    call: &Call,
    output: &PendingOutput,
    slot: u32,
    descriptors: &[usize],
    forward: &ForwardBatch,
    scores: bool,
) -> Result<(TokenRow, Media)> {
    let mut position = output.progress.logical_position;
    let (_, mut visible, _) = sources
        .tables
        .coordinates(slot, output.progress.cache_visible_len)?;
    let mut rows = Vec::new();
    let mut embeddings = Vec::new();
    let mut reads = Vec::new();
    let mut offset = 0;
    let mut begin = 0;
    let tokens = &call.input_token_ids;
    let blocks = call.vision_inputs.len();
    for index in 0..=blocks {
        let end = call
            .vision_inputs
            .get(index)
            .map_or(tokens.len(), |block| block.offset as usize);
        if end < begin || end > tokens.len() {
            return Err(Error::Invalid(
                "context blocks must lie in order within the call's tokens".into(),
            ));
        }
        if end > begin {
            // A scored prompt projects every text token; its last row
            // also samples.
            let selection = if scores {
                TokenSelection::AllLogits
            } else if index == blocks && call.token_output.is_some() {
                TokenSelection::LastLogits
            } else {
                TokenSelection::Cache
            };
            let row = text_row(
                slot,
                &tokens[begin..end],
                visible,
                position,
                selection,
                sources.spatial(),
            )?;
            position += (end - begin) as u64;
            visible += (end - begin) as u64;
            offset += row.query as usize;
            rows.push(row);
        }
        if index < blocks {
            let last = index + 1 == blocks && end == tokens.len();
            // A block's last logits predict the prompt's next token.
            let selection = if scores || (last && call.token_output.is_some()) {
                TokenSelection::LastLogits
            } else {
                TokenSelection::Cache
            };
            let feature = consume(sources, call, index, false)?;
            let built = vision_row(sources, &feature, slot, visible, position, false, selection);
            reads.push(Arc::clone(&feature.read));
            let (row, first, values) = built?;
            position = next_position(&row)?;
            visible += u64::from(row.query);
            embeddings.push((offset + first, values));
            offset += row.query as usize;
            rows.push(row);
        }
        begin = end;
    }

    // The scheduler's forward rows of the call describe the same rows: one
    // per prompt run and vision block, on the call's slot, each with its
    // query. Their prefix lengths come from the same request state as the
    // call's visible length, so they are not compared with it.
    let count: u64 = rows.iter().map(|row| u64::from(row.query)).sum();
    let aligned = descriptors.len() == rows.len()
        && descriptors.iter().zip(&rows).all(|(&descriptor, row)| {
            forward.request_pool_indices[descriptor] == slot
                && forward.query_lens[descriptor] == row.query
        });
    if !aligned || count > u64::from(call.bounds.max_tokens) {
        return Err(Error::Invalid(
            "context rows disagree with the call's forward rows".into(),
        ));
    }
    let last = rows
        .pop()
        .ok_or_else(|| Error::Invalid("context prefill has no rows".into()))?;
    let next = next_position(&last)?;
    Ok((
        last,
        Media {
            rows,
            embeddings,
            reads,
            position: next,
            continued: next,
            sample: next,
            latent: None,
        },
    ))
}

/// The row of a visual-state prefill: one vision block closing an image,
/// or a latent feature's conditioning.
pub(super) fn visual(
    sources: &Sources<'_>,
    call: &Call,
    output: &PendingOutput,
    slot: u32,
) -> Result<(TokenRow, Media)> {
    let latent = call.latent_feature_input.is_some();
    if call.vision_inputs.len() + usize::from(latent) != 1 {
        return Err(Error::Invalid(
            "visual extend requires exactly one feature tensor".into(),
        ));
    }
    let start = output.progress.logical_position;
    let (_, visible, _) = sources
        .tables
        .coordinates(slot, output.progress.cache_visible_len)?;
    let feature = consume(sources, call, 0, latent)?;
    let read = Arc::clone(&feature.read);
    let image = sources
        .image
        .ok_or_else(|| Error::Unsupported("visual input requires the image runtime".into()));

    // A completed feedback moves the language position past the image.
    let builder = image.as_ref().ok().and_then(|image| image.facts().denoiser);
    let advance = builder.map_or(1, |layout| u64::from(layout.rope_advance));
    let position = start
        + if call.completion_output.is_some() {
            advance
        } else {
            0
        };
    let continued = start
        + if call.completion_output.is_some() {
            advance
        } else {
            1
        };
    let selection = if call.token_output.is_some() {
        TokenSelection::LastLogits
    } else {
        TokenSelection::Cache
    };

    if latent {
        if call.token_output.is_some() {
            return Err(Error::Invalid(
                "a latent state prefill samples no token".into(),
            ));
        }
        let image = image?;
        let denoiser = image.denoiser().ok_or_else(|| {
            Error::Unsupported("latent state prefill requires an image denoiser".into())
        })?;
        let layout = denoiser.layout()?;
        let query = layout.sequence_length(feature.height, feature.width)?;
        if query > call.bounds.max_tokens {
            return Err(Error::Invalid(
                "image state query span exceeds the call token bound".into(),
            ));
        }
        let features = read.value().clone();
        let row = TokenRow {
            slot,
            source: RowSource::Placed {
                tokens: vec![0; query as usize],
                positions: layout.conditioning_positions(feature.height, feature.width, start)?,
                spatial: true,
            },
            prefix: narrow(visible)?,
            query,
            selection,
            gated: false,
            force_finish: false,
            causal: false,
        };
        return Ok((
            row,
            Media {
                rows: Vec::new(),
                embeddings: Vec::new(),
                reads: vec![read],
                position,
                continued,
                sample: start + advance,
                latent: Some(ConditionRow {
                    features,
                    height: feature.height,
                    width: feature.width,
                    slot,
                    prefix: narrow(visible)?,
                    position: start,
                }),
            },
        ));
    }

    let built = vision_row(sources, &feature, slot, visible, start, true, selection);
    let (row, first, values) = built?;
    if row.query > call.bounds.max_tokens {
        return Err(Error::Invalid(
            "image state query span exceeds the call token bound".into(),
        ));
    }
    Ok((
        row,
        Media {
            rows: Vec::new(),
            embeddings: vec![(first, values)],
            reads: vec![read],
            position,
            continued,
            sample: start + advance,
            latent: None,
        },
    ))
}

/// The patch grid key of an image of `height` x `width`: a generated image
/// (`close_image`) keeps the single-image pixel bound, a request's input
/// image counts the request's `input_images`.
pub(crate) fn grid_key(height: u32, width: u32, close_image: bool, input_images: u32) -> GridKey {
    (height, width, if close_image { 0 } else { input_images })
}
