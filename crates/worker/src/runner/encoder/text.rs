//! Text encoding: a prompt zero-padded to the smallest configured text capacity
//! that holds it ([`text_capacity`]), the buffers of a text encoder's forward,
//! and the encode of a prompt, natively through its capacity's graph or, with
//! vision tokens ([`PromptVision`]), on the Python thread.

use cuda::{CopyBatch, SourceOrder};

use super::{EncoderGraph, EncoderRunner};
use crate::cuda_graph::GraphMode;
use crate::lane::Job;
use crate::memory::allocation::StorageView;
use crate::runner::methods::{Argument, MethodCall};
use crate::stream::Issue;
use crate::{Error, Result};

/// The text capacity a prompt of `tokens` is encoded at: the smallest
/// configured capacity holding it, or the prompt's own length without
/// configured capacities.
pub(crate) fn text_capacity(capacities: &[usize], tokens: usize) -> Result<usize> {
    if tokens == 0 {
        return Err(Error::Invalid("text encoding requires a token".into()));
    }
    if capacities.is_empty() {
        return Ok(tokens);
    }
    crate::cuda_graph::bucket(capacities, tokens).ok_or_else(|| {
        Error::Invalid(format!(
            "prompt of {tokens} tokens exceeds the configured text capacities"
        ))
    })
}

/// The vision tokens of a prompt, for a text encoder that places them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PromptVision<'a> {
    /// `(time, height, width)` patch grids of the prompt's image blocks, in
    /// prompt order.
    pub image_grids: &'a [[u32; 3]],
    /// The same for its video blocks.
    pub video_grids: &'a [[u32; 3]],
    /// The vision rows of its placeholders, in placeholder order.
    pub rows: &'a StorageView,
}

/// The buffers of a text encoder's forward.
pub(super) struct TextBuffers {
    /// The `[max capacity]` int64 prompt, zero-padded.
    pub(super) tokens: StorageView,
    /// The `[max capacity, hidden]` features of the padded prompt.
    pub(super) features: StorageView,
    /// The `[3, max capacity]` int64 rotary coordinates of a prompt with
    /// vision tokens, for an encoder that places them.
    pub(super) positions: Option<StorageView>,
    /// Whether this rank's pipeline stage emits the features.
    pub(super) final_stage: bool,
}

impl EncoderRunner {
    /// The arguments of the Python half's `encode` of a prompt padded to
    /// `capacity`, without vision tokens.
    pub(super) fn text_arguments(&self, capacity: usize) -> Result<Vec<Argument>> {
        let Some(text) = &self.text else {
            return Err(Error::Invalid(format!("{} encodes no text", self.owner)));
        };
        Ok(vec![
            Argument::Handle(self.methods.runner.clone()),
            Argument::Tensor(text.tokens.prefix(capacity as i64)?),
            Argument::Tensor(text.features.prefix(capacity as i64)?),
        ])
    }

    /// Record the encoding of one prompt of `tokens` ids into `out`, its
    /// `[tokens, hidden]` feature rows in the product's dtype, as operations
    /// of `job` on the encoder's stream: the token upload, the launch of
    /// the prompt capacity's graph or the eager call, and the copy of the
    /// prompt's rows. Returns the mode the forward runs in.
    ///
    /// `vision` carries the prompt's vision tokens, for an encoder that
    /// places them. A pipeline stage that emits no features writes nothing.
    pub(crate) fn encode_text(
        &mut self,
        tokens: &[i64],
        out: &StorageView,
        vision: Option<PromptVision<'_>>,
        job: &Job,
    ) -> Result<GraphMode> {
        let (encode, position_ids) = (&self.methods.encode, &self.methods.position_ids);
        let (Some(text), Some(stream)) = (&self.text, &self.stream) else {
            return Err(Error::Invalid(format!("{} encodes no text", self.owner)));
        };
        let (encode, position_ids, stream) = (encode.clone(), position_ids.clone(), stream.clone());
        let issue = job.on(Some(&stream));
        let capacity = text_capacity(&self.text_capacities, tokens.len())?;
        if capacity as i64 > text.tokens.shape()[0] {
            return Err(Error::Invalid(
                "text encoder input exceeds its token capacity".into(),
            ));
        }
        // Causal attention keeps the zero padding out of the prompt's rows.
        let mut padded = tokens.to_vec();
        padded.resize(capacity, 0);
        let bytes: Vec<u8> = padded
            .iter()
            .flat_map(|token| token.to_ne_bytes())
            .collect();
        // SAFETY: the text encoder's token input, kept until it closes after
        // its stream drained; the forwards reading it run on `issue`'s
        // stream, the earlier ones before this write and this one after it.
        unsafe { text.tokens.upload(&bytes, issue) }?;
        let (features, final_stage) = (text.features.clone(), text.final_stage);

        let runner = Argument::Handle(self.methods.runner.clone());
        let mut arguments = self.text_arguments(capacity)?;
        let mode = match vision {
            Some(PromptVision {
                image_grids,
                video_grids,
                rows,
            }) => {
                let positions = self
                    .text
                    .as_ref()
                    .and_then(|text| text.positions.as_ref())
                    .ok_or_else(|| {
                        Error::Invalid(format!("{} places no vision tokens", self.owner))
                    })?;
                let grids = |grids: &[[u32; 3]]| {
                    Argument::Integers(
                        grids
                            .iter()
                            .flatten()
                            .map(|&extent| i64::from(extent))
                            .collect(),
                    )
                };
                job.call(MethodCall::new(
                    &position_ids,
                    vec![
                        runner,
                        Argument::Integers(padded.clone()),
                        grids(image_grids),
                        grids(video_grids),
                        Argument::Tensor(positions.clone()),
                    ],
                ));
                arguments.push(Argument::Tensor(positions.clone()));
                arguments.push(Argument::Tensor(rows.clone()));
                job.call(MethodCall::new(&encode, arguments));
                GraphMode::Eager
            }
            // Without configured capacities a prompt is encoded at its
            // exact length, which no graph serves.
            None if self.text_capacities.is_empty() => {
                job.call(MethodCall::new(&encode, arguments));
                GraphMode::Eager
            }
            None => self.forward(EncoderGraph::Text { capacity }, arguments, None, job)?,
        };
        if final_stage {
            // The features and the product share the model dtype, so the
            // prompt's rows are copied as they are.
            let count = out.numel() * out.dtype().bytes() as u64;
            let mut rows = CopyBatch::new(SourceOrder::Stream);
            rows.push(out.address(), features.address(), count as usize);
            // SAFETY: the features hold at least the prompt's rows, the
            // caller bound `out` for them, both outlive the batch, and the
            // copy follows the encode on the stream.
            unsafe { issue.copy(&rows) }?;
        }
        Ok(mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_take_the_smallest_holding_text_capacity() -> Result<()> {
        assert_eq!(text_capacity(&[256, 512, 1024], 300)?, 512);
        assert_eq!(text_capacity(&[256, 512, 1024], 256)?, 256);
        assert!(text_capacity(&[256, 512], 513).is_err());
        assert!(text_capacity(&[256], 0).is_err());
        // Without configured capacities a prompt keeps its own length.
        assert_eq!(text_capacity(&[], 37)?, 37);
        Ok(())
    }
}
