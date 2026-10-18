//! Diffusion LM calls: validation and their rows.
//!
//! A token-denoising call carries either a readout or a denoising step
//! (`Call::validate`). Its forward rows read the request's unchanged cached
//! prefix (`seq_len - query_len` is the accepted visible prefix cache
//! length) and write no prefix cache. A readout's rows partition its
//! tokens, back to back, and every row reads at least one answer slot. A
//! denoising step is one row of the block length over the request's prefix,
//! stepping the block its request keeps resident with the sampling the
//! deployment serves.

use std::ops::Range;

use uniserve_core::CanvasSampling;
use uniserve_ipc::{Batch, Call, CanvasStep, ForwardBatch, NewRequest};

use crate::storage::{AttentionRow, TablePages};
use crate::{Error, Result};

/// One sequence of a forward: a block of tokens over its request's prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sequence {
    /// Block tokens.
    pub(crate) query: usize,
    /// Cached prefix tokens the block reads: its request's visible prefix cache.
    pub(crate) prefix: u32,
    /// Position of the block's first token.
    pub(crate) position: u64,
}

/// A readout forward: the sequences of one or more calls, back to back.
pub(crate) struct ReadoutPass<'a> {
    pub(crate) rows: Vec<Sequence>,
    /// Packed block tokens of every row.
    pub(crate) tokens: Vec<i64>,
    /// Packed token index of every answer slot, strictly increasing.
    pub(crate) slots: Vec<i64>,
    /// `slots + 1` offsets of each slot's candidates in `candidates`.
    pub(crate) slot_candidates: Vec<usize>,
    /// Candidate token ids, in row, slot and candidate order.
    pub(crate) candidates: Vec<i64>,
    /// The rows' selected pages, one entry per numerical table.
    pub(crate) pages: &'a [TablePages],
}

impl ReadoutPass<'_> {
    /// Tokens of the longest block.
    pub(crate) fn longest(&self) -> usize {
        self.rows.iter().map(|row| row.query).max().unwrap_or(0)
    }
}

/// One denoising step: the request slot whose resident block it denoises and
/// its sampling coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenoiseRow {
    pub(crate) slot: u32,
    /// The request's sampling seed, carried as its two's-complement bits.
    pub(crate) seed: u64,
    pub(crate) block: u32,
    pub(crate) step: u32,
    pub(crate) prefix: u32,
    pub(crate) position: u64,
}

/// The forward rows of each call of `plan`, in row order.
pub(crate) fn rows_by_call(plan: &Batch) -> Result<Vec<Vec<usize>>> {
    let mut rows = vec![Vec::new(); plan.calls.len()];
    for (row, &call) in plan.forward.call_indices.iter().enumerate() {
        rows.get_mut(call as usize)
            .ok_or_else(|| Error::Invalid("a forward row names no call".into()))?
            .push(row);
    }
    Ok(rows)
}

/// Check that `descriptors` read `slot`'s prefix without writing the
/// prefix cache. A row's prefix length is attention geometry the engine
/// builds from the same request state as the call's visible length, so the
/// two are not compared.
fn read_only(forward: &ForwardBatch, descriptors: &[usize], slot: u32) -> Result<()> {
    if descriptors.is_empty()
        || descriptors.iter().any(|&row| {
            forward.write_cache[row]
                || forward.request_pool_indices[row] != slot
                || forward.query_lens[row] == 0
        })
    {
        return Err(Error::Invalid(
            "block rows must read the request's unchanged prefix".into(),
        ));
    }
    Ok(())
}

/// A readout forward under construction: the sequences, answer slots
/// and candidates of its calls, back to back.
#[derive(Default)]
pub(crate) struct Readouts {
    pub(crate) rows: Vec<Sequence>,
    pub(crate) attention: Vec<AttentionRow>,
    pub(crate) tokens: Vec<i64>,
    pub(crate) slots: Vec<i64>,
    pub(crate) slot_candidates: Vec<usize>,
    pub(crate) candidates: Vec<i64>,
}

impl Readouts {
    pub(crate) fn new() -> Self {
        Self {
            slot_candidates: vec![0],
            ..Self::default()
        }
    }

    /// Append the readout `call` of the request in `slot`, whose accepted
    /// visible prefix cache is `visible` tokens and whose blocks start at logical
    /// position `position`. Returns the call's range of the invocation's
    /// candidates.
    pub(crate) fn add(
        &mut self,
        forward: &ForwardBatch,
        call: &Call,
        descriptors: &[usize],
        slot: u32,
        visible: u64,
        position: u64,
    ) -> Result<Range<usize>> {
        let readout = call
            .readout
            .as_ref()
            .ok_or_else(|| Error::Invalid("a readout forward requires a readout".into()))?;
        read_only(forward, descriptors, slot)?;
        let lengths: Vec<usize> = descriptors
            .iter()
            .map(|&row| forward.query_lens[row] as usize)
            .collect();
        if lengths.iter().sum::<usize>() != call.input_token_ids.len() {
            return Err(Error::Invalid(
                "block rows must partition the call's tokens over its prefix".into(),
            ));
        }
        let prefix = u32::try_from(visible)
            .map_err(|_| Error::Invalid("a block prefix exceeds the token range".into()))?;

        // Every row reads at least one of the call's slots, which address
        // the call's own tokens in increasing order.
        let base = self.tokens.len();
        let mut next = 0;
        let mut first = 0;
        for &length in &lengths {
            let end = first + length;
            let start = next;
            while next < readout.slot_tokens.len() && (readout.slot_tokens[next] as usize) < end {
                next += 1;
            }
            if next == start {
                return Err(Error::Invalid("every block row must read a slot".into()));
            }
            self.rows.push(Sequence {
                query: length,
                prefix,
                position,
            });
            self.attention.push(AttentionRow {
                slot,
                prefix,
                query: length as u32,
                write: false,
            });
            first = end;
        }
        if next != readout.slot_tokens.len() {
            return Err(Error::Invalid(
                "a readout slot lies outside its block rows".into(),
            ));
        }

        let candidates = self.candidates.len()..self.candidates.len() + readout.candidate_ids.len();
        self.tokens
            .extend(call.input_token_ids.iter().map(|&token| i64::from(token)));
        self.slots.extend(
            readout
                .slot_tokens
                .iter()
                .map(|&token| (base + token as usize) as i64),
        );
        let offset = self.candidates.len();
        self.slot_candidates.extend(
            readout.candidate_offsets[1..]
                .iter()
                .map(|&end| offset + end as usize),
        );
        self.candidates
            .extend(readout.candidate_ids.iter().map(|&id| i64::from(id)));
        Ok(candidates)
    }
}

/// The denoising step `call` of `admission`'s request in `slot`, whose
/// accepted visible prefix cache is `visible` tokens and whose block starts at
/// logical position `position`. `served` is the sampling this worker steps
/// blocks with, the deployment's (`None` on a rank that serves readouts
/// only); the request carries none of its own.
pub(crate) fn step_row(
    forward: &ForwardBatch,
    call: &Call,
    descriptors: &[usize],
    admission: &NewRequest,
    served: Option<&CanvasSampling>,
    visible: u64,
    position: u64,
) -> Result<(CanvasStep, DenoiseRow)> {
    let step = call
        .canvas
        .ok_or_else(|| Error::Invalid("a block step requires its coordinates".into()))?;
    let generation = admission
        .generation
        .as_ref()
        .ok_or_else(|| Error::Invalid("a block step requires an admitted generation".into()))?;
    let seed = generation
        .sampling
        .seed
        .ok_or_else(|| Error::Invalid("a block step requires a seeded request".into()))?;
    let sampling =
        served.ok_or_else(|| Error::Invalid("this worker keeps no generating blocks".into()))?;
    let slot = admission.request_pool_idx;
    read_only(forward, descriptors, slot)?;
    let length = sampling.canvas_length;
    if descriptors.len() != 1
        || call.bounds.max_tokens != length
        || forward.query_lens[descriptors[0]] != length
    {
        return Err(Error::Invalid(
            "a block step is one read-only block row over its prefix".into(),
        ));
    }
    if step.step >= sampling.max_steps {
        return Err(Error::Invalid(
            "the block step exceeds its step limit".into(),
        ));
    }
    let prefix = u32::try_from(visible)
        .map_err(|_| Error::Invalid("a block prefix exceeds the token range".into()))?;
    Ok((
        step,
        DenoiseRow {
            slot,
            seed,
            block: step.block,
            step: step.step,
            prefix,
            position,
        },
    ))
}

#[cfg(test)]
mod tests {
    use uniserve_core::{CallId, RequestId, SamplingParams};
    use uniserve_ipc::{CallKind, ForwardMode, GenerationParams, Readout, RequestKey};

    use super::*;

    fn call(tokens: usize, readout: Option<Readout>, canvas: Option<CanvasStep>) -> Call {
        Call {
            request_key: RequestKey::new(1, RequestId(7), 1),
            call_id: CallId::new(1, 0),
            coordinates: Default::default(),
            component: "diffusion_gemma".into(),
            kind: CallKind::Forward(ForwardMode::TokenDenoising),
            bounds: uniserve_ipc::Bounds {
                max_tokens: tokens as u32,
                ..Default::default()
            },
            inputs: Vec::new(),
            outputs: Vec::new(),
            consumer_slots: Vec::new(),
            token_input: None,
            token_output: None,
            vision_inputs: Vec::new(),
            latent_feature_input: None,
            encoder_output: None,
            latent_input: None,
            latent_output: None,
            image_input: None,
            image_output: None,
            completion_output: None,
            transition_output: None,
            predicate: None,
            image_index: None,
            sampling_state: None,
            input_token_ids: (0..tokens as u32).map(|token| 100 + token).collect(),
            readout,
            canvas,
            input_image: None,
            cache_input: None,
            cache_output: None,
        }
    }

    /// Rows of `queries` tokens reading `prefix` tokens of `slot`.
    fn forward(slot: u32, prefix: u32, queries: &[u32]) -> ForwardBatch {
        let mut forward = ForwardBatch::default();
        for &query in queries {
            forward.push(0, slot, prefix + query, query, false);
        }
        forward
    }

    fn sampling() -> CanvasSampling {
        CanvasSampling {
            canvas_length: 4,
            max_steps: 3,
            entropy_bound: 0.5,
            t_min: 0.4,
            t_max: 0.8,
            confidence_threshold: 0.2,
            stability_threshold: 1,
        }
    }

    fn admission(seed: Option<u64>) -> NewRequest {
        NewRequest::new(
            RequestKey::new(1, RequestId(7), 1),
            2,
            Some(GenerationParams {
                sampling: SamplingParams {
                    seed,
                    ..SamplingParams::default()
                },
                negative_token_ids: Vec::new(),
                finish_token_ids: Vec::new(),
                initial_position: 0,
            }),
            None,
            0,
        )
        .expect("a valid admission")
    }

    #[test]
    fn readouts_pack_rows_slots_and_candidates_of_every_call() -> Result<()> {
        let mut readouts = Readouts::new();
        let first = call(
            5,
            Some(Readout {
                slot_tokens: vec![1, 3, 4],
                candidate_offsets: vec![0, 2, 3, 5],
                candidate_ids: vec![7, 8, 9, 10, 11],
            }),
            None,
        );
        // Two rows of 3 and 2 tokens; each reads a slot.
        let range = readouts.add(&forward(2, 6, &[3, 2]), &first, &[0, 1], 2, 6, 40)?;
        assert_eq!(range, 0..5);
        let second = call(
            2,
            Some(Readout {
                slot_tokens: vec![0],
                candidate_offsets: vec![0, 1],
                candidate_ids: vec![12],
            }),
            None,
        );
        let range = readouts.add(&forward(3, 1, &[2]), &second, &[0], 3, 1, 9)?;
        assert_eq!(range, 5..6);

        assert_eq!(readouts.tokens, [100, 101, 102, 103, 104, 100, 101]);
        // Slots address the invocation's packed tokens.
        assert_eq!(readouts.slots, [1, 3, 4, 5]);
        assert_eq!(readouts.slot_candidates, [0, 2, 3, 5, 6]);
        assert_eq!(readouts.candidates, [7, 8, 9, 10, 11, 12]);
        assert_eq!(
            readouts.rows,
            [
                Sequence {
                    query: 3,
                    prefix: 6,
                    position: 40
                },
                Sequence {
                    query: 2,
                    prefix: 6,
                    position: 40
                },
                Sequence {
                    query: 2,
                    prefix: 1,
                    position: 9
                },
            ]
        );
        assert!(readouts.attention.iter().all(|row| !row.write));
        Ok(())
    }

    #[test]
    fn readouts_refuse_rows_that_write_skip_slots_or_misread_the_prefix() {
        let readout = Readout {
            slot_tokens: vec![4],
            candidate_offsets: vec![0, 1],
            candidate_ids: vec![1],
        };
        let readout_call = call(5, Some(readout), None);
        // The first row reads no slot.
        assert!(
            Readouts::new()
                .add(&forward(2, 6, &[3, 2]), &readout_call, &[0, 1], 2, 6, 0)
                .is_err()
        );
        // Another slot, a write, or rows not partitioning the tokens.
        assert!(
            Readouts::new()
                .add(&forward(3, 6, &[5]), &readout_call, &[0], 2, 6, 0)
                .is_err()
        );
        let mut writing = forward(2, 6, &[5]);
        writing.write_cache[0] = true;
        assert!(
            Readouts::new()
                .add(&writing, &readout_call, &[0], 2, 6, 0)
                .is_err()
        );
        assert!(
            Readouts::new()
                .add(&forward(2, 6, &[4]), &readout_call, &[0], 2, 6, 0)
                .is_err()
        );
    }

    /// A block step runs under the sampling this rank was launched with and
    /// the request's seed.
    #[test]
    fn steps_require_the_served_sampling_and_a_seeded_request() -> Result<()> {
        let served = sampling();
        let step = CanvasStep { block: 2, step: 1 };
        let rows = forward(2, 10, &[4]);
        let canvas = call(4, None, Some(step));
        let seed = u64::MAX - 1;
        let (coordinates, row) = step_row(
            &rows,
            &canvas,
            &[0],
            &admission(Some(seed)),
            Some(&served),
            10,
            12,
        )?;
        assert_eq!(coordinates, step);
        assert_eq!(
            row,
            DenoiseRow {
                slot: 2,
                seed,
                block: 2,
                step: 1,
                prefix: 10,
                position: 12,
            }
        );

        let refused = |admitted: &NewRequest, served: Option<&CanvasSampling>, call: &Call| {
            step_row(&rows, call, &[0], admitted, served, 10, 12).is_err()
        };
        let seeded = admission(Some(1));
        assert!(refused(&admission(None), Some(&served), &canvas));
        assert!(refused(&seeded, None, &canvas));
        // The last step of a block is the step limit's predecessor.
        assert!(refused(
            &seeded,
            Some(&served),
            &call(4, None, Some(CanvasStep { block: 0, step: 3 }))
        ));
        // A step is exactly one block-length row.
        assert!(refused(&seeded, Some(&served), &call(3, None, Some(step))));
        Ok(())
    }
}
