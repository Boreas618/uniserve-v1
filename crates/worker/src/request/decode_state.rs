//! Per-slot continuation state of token-generating requests.
//!
//! A request slot's continuation state is what the request's next token
//! call reads on the device without a host round trip:
//!
//! | Field | Element | Shape | Meaning |
//! | --- | --- | --- | --- |
//! | `future_input_tokens` | `i64` | `[S, 1]` | next input token |
//! | `predicates` | `u8` | `[S]` | the last selection continues |
//! | `logical_lengths` | `i64` | `[S]` | logical position of the next token |
//! | `sampling_positions` | `i64` | `[S]` | semantic draw position |
//! | `penalty_counts` | `i32` | `[S, V]` | committed generated-token counts |
//! | `prompt_logits` | model dtype | `[S, V]` | last logits of a scored prompt chunk |
//! | `logit_bias` | `f32` | `[S, V]` | the logit biases of the slot's request |
//!
//! The slot's verified cache length is the block tables' `verified_lengths`,
//! shared with attention gathers. `S` is the request slot count plus one:
//! slot 0 is padding. Rust allocates, resets and copies; the runner's
//! `commit` installs selections. A slot's logit biases are installed when a
//! request that has any takes the slot, and read only by its tasks.

use std::sync::Arc;

use cuda::{CopyBatch, DeviceId, SourceOrder, Stream, memset_d32};

use crate::ffi::{Backing, DType};
use crate::memory::allocation::{Storage, StorageView};
use crate::stream::{Fill, Issue};
use crate::{Error, Result};

/// Byte alignment of each carved field.
const ALIGNMENT: usize = 256;

/// The continuation state of every request slot of a rank.
pub(crate) struct DecodeState {
    memory: Arc<Storage>,
    rows: usize,
    vocab: usize,
    future_input_tokens: StorageView,
    predicates: StorageView,
    logical_lengths: StorageView,
    sampling_positions: StorageView,
    penalty_counts: StorageView,
    prompt_logits: StorageView,
    logit_bias: StorageView,
    /// Borrowed from the block tables.
    verified_lengths: StorageView,
}

/// Byte offsets of the carved fields and the total length.
fn layout(rows: usize, vocab: usize, logits: DType) -> ([usize; 7], usize) {
    let sizes = [
        rows * 8,
        rows,
        rows * 8,
        rows * 8,
        rows * vocab * 4,
        rows * vocab * logits.bytes(),
        rows * vocab * 4,
    ];
    let mut offsets = [0; 7];
    let mut cursor = 0;
    for (offset, size) in offsets.iter_mut().zip(sizes) {
        *offset = cursor;
        cursor += size.next_multiple_of(ALIGNMENT);
    }
    (offsets, cursor)
}

impl DecodeState {
    /// Device bytes of the state of `request_slots` slots over `vocab`
    /// tokens with `logits`-typed prompt logits.
    pub(crate) fn bytes(request_slots: usize, vocab: usize, logits: DType) -> u64 {
        layout(request_slots + 1, vocab, logits).1 as u64
    }

    /// Allocate the state on `device` (host memory when `None`) and reset
    /// every slot on `stream`, which is synchronized before returning.
    /// `verified_lengths` is the block tables' `[S]` int32 column.
    pub(crate) fn allocate(
        request_slots: usize,
        vocab: usize,
        logits: DType,
        verified_lengths: StorageView,
        device: Option<DeviceId>,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        let rows = request_slots + 1;
        if vocab == 0
            || verified_lengths.dtype() != DType::I32
            || verified_lengths.shape() != [rows as i64]
        {
            return Err(Error::Invalid(
                "decode state requires a vocabulary and per-slot verified lengths".into(),
            ));
        }
        if device.is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "device decode state is reset on a stream of its device".into(),
            ));
        }
        let (offsets, bytes) = layout(rows, vocab, logits);
        let memory = match device {
            Some(device) => Storage::device(device, bytes)?,
            None => Storage::host(bytes),
        };
        let view = |index: usize, dtype: DType, shape: &[i64]| {
            StorageView::contiguous(
                Arc::clone(&memory) as Arc<dyn Backing>,
                offsets[index],
                dtype,
                shape,
            )
        };
        let (slots, width) = (rows as i64, vocab as i64);
        let state = Self {
            future_input_tokens: view(0, DType::I64, &[slots, 1])?,
            predicates: view(1, DType::U8, &[slots])?,
            logical_lengths: view(2, DType::I64, &[slots])?,
            sampling_positions: view(3, DType::I64, &[slots])?,
            penalty_counts: view(4, DType::I32, &[slots, width])?,
            prompt_logits: view(5, logits, &[slots, width])?,
            logit_bias: view(6, DType::F32, &[slots, width])?,
            verified_lengths,
            memory,
            rows,
            vocab,
        };
        if let Some(stream) = stream {
            // SAFETY: the allocation spans `bytes` and is new; the stream
            // drains before the state is shared.
            unsafe { memset_d32(state.memory.address(), 0, bytes / 4, stream) }?;
        }
        let all: Vec<usize> = (0..rows).collect();
        state.reset(&all, stream)?;
        if let Some(stream) = stream {
            stream.synchronize()?;
        }
        Ok(state)
    }

    pub(crate) fn future_input_tokens(&self) -> &StorageView {
        &self.future_input_tokens
    }

    pub(crate) fn predicates(&self) -> &StorageView {
        &self.predicates
    }

    pub(crate) fn logical_lengths(&self) -> &StorageView {
        &self.logical_lengths
    }

    pub(crate) fn sampling_positions(&self) -> &StorageView {
        &self.sampling_positions
    }

    pub(crate) fn penalty_counts(&self) -> &StorageView {
        &self.penalty_counts
    }

    pub(crate) fn prompt_logits(&self) -> &StorageView {
        &self.prompt_logits
    }

    pub(crate) fn logit_bias(&self) -> &StorageView {
        &self.logit_bias
    }

    pub(crate) fn verified_lengths(&self) -> &StorageView {
        &self.verified_lengths
    }

    /// The address of slot `slot`'s next input token.
    pub(crate) fn future_token(&self, slot: usize) -> Result<u64> {
        self.require_slot(slot)?;
        Ok(self.future_input_tokens.address() + 8 * slot as u64)
    }

    /// The address of slot `slot`'s next logical position.
    pub(crate) fn logical_length(&self, slot: usize) -> Result<u64> {
        self.require_slot(slot)?;
        Ok(self.logical_lengths.address() + 8 * slot as u64)
    }

    /// The address of slot `slot`'s retained prompt logits row.
    pub(crate) fn prompt_row(&self, slot: usize) -> Result<u64> {
        self.require_slot(slot)?;
        Ok(self.prompt_logits.address()
            + (slot * self.vocab * self.prompt_logits.dtype().bytes()) as u64)
    }

    fn require_slot(&self, slot: usize) -> Result<()> {
        if slot >= self.rows {
            return Err(Error::Invalid(
                "request slot is outside the decode state".into(),
            ));
        }
        Ok(())
    }

    /// Reset newly admitted slots on `stream`: next token 1, no
    /// continuation, zero coordinates, cache length and penalty counts.
    /// The retained prompt logits are read only after a scored chunk wrote
    /// them, so they are not reset.
    pub(crate) fn reset(&self, slots: &[usize], stream: impl Issue) -> Result<()> {
        for &slot in slots {
            self.require_slot(slot)?;
        }
        let word =
            |view: &StorageView, slot: usize, width: u64| view.address() + width * slot as u64;

        // Eight-byte fields take one source word per slot; the payload is
        // read during the call.
        let ones: Vec<i64> = vec![1; slots.len()];
        let zeros: Vec<i64> = vec![0; slots.len()];
        let mut batch = CopyBatch::new(SourceOrder::DuringCall);
        for (index, &slot) in slots.iter().enumerate() {
            batch.push(
                word(&self.future_input_tokens, slot, 8),
                ones[index..].as_ptr() as u64,
                8,
            );
            batch.push(
                word(&self.logical_lengths, slot, 8),
                zeros[index..].as_ptr() as u64,
                8,
            );
            batch.push(
                word(&self.sampling_positions, slot, 8),
                zeros[index..].as_ptr() as u64,
                8,
            );
        }
        // SAFETY: every destination is one element of this state for a
        // distinct slot; sources are read during the call.
        unsafe { stream.copy(&batch) }?;

        for &slot in slots {
            let zero = |address: u64, width: usize, count: usize| Fill {
                address,
                value: 0,
                width,
                count,
            };
            // SAFETY: one byte, one word and one row of this state.
            unsafe {
                stream.fill(zero(word(&self.predicates, slot, 1), 1, 1))?;
                stream.fill(zero(word(&self.verified_lengths, slot, 4), 4, 1))?;
                stream.fill(zero(
                    word(&self.penalty_counts, slot, 4 * self.vocab as u64),
                    4,
                    self.vocab,
                ))?;
            }
        }
        Ok(())
    }

    /// Count `tokens` in slot `slot`'s penalty counts on `stream`, after
    /// `reset` zeroed them: the tokens its request generated on a
    /// predecessor engine (`GenerationParams::generated_token_ids`), which
    /// its penalties count as generated. Fails for a token outside the
    /// vocabulary.
    pub(crate) fn count_generated(
        &self,
        slot: usize,
        tokens: &[u32],
        stream: impl Issue,
    ) -> Result<()> {
        self.require_slot(slot)?;
        if tokens.is_empty() {
            return Ok(());
        }
        let mut counts = vec![0_i32; self.vocab];
        for &token in tokens {
            let count = counts.get_mut(token as usize).ok_or_else(|| {
                Error::Invalid("a generated token is outside the vocabulary".into())
            })?;
            *count = count.saturating_add(1);
        }

        // The slot's whole row is written from the host counts.
        let row = self.penalty_counts.address() + (4 * self.vocab * slot) as u64;
        let mut batch = CopyBatch::new(SourceOrder::DuringCall);
        batch.push(row, counts.as_ptr() as u64, 4 * self.vocab);
        // SAFETY: the destination is one row of this state; the source is
        // read during the call.
        unsafe { stream.copy(&batch) }?;
        Ok(())
    }

    /// Install `biases` as slot `slot`'s logit biases on `stream`: the
    /// row's tokens outside the list are zero. A bias of a token outside the
    /// vocabulary has no logit to shift.
    pub(crate) fn install_bias(
        &self,
        slot: usize,
        biases: &[(u32, f32)],
        stream: impl Issue,
    ) -> Result<()> {
        self.require_slot(slot)?;
        let mut row = vec![0.0_f32; self.vocab];
        for &(token, bias) in biases {
            if let Some(entry) = row.get_mut(token as usize) {
                *entry = bias;
            }
        }
        let address = self.logit_bias.address() + (4 * self.vocab * slot) as u64;
        let mut batch = CopyBatch::new(SourceOrder::DuringCall);
        batch.push(address, row.as_ptr() as u64, 4 * self.vocab);
        // SAFETY: the destination is one row of this state; the source is
        // read during the call.
        unsafe { stream.copy(&batch) }?;
        Ok(())
    }

    /// Set `slots`' continuation flags to `value` on `stream` (warmup graph
    /// capture runs its decode rows as active continuations).
    pub(crate) fn set_predicates(
        &self,
        slots: &[usize],
        value: bool,
        stream: impl Issue,
    ) -> Result<()> {
        for &slot in slots {
            self.require_slot(slot)?;
            // SAFETY: one byte of this state.
            unsafe {
                stream.fill(Fill {
                    address: self.predicates.address() + slot as u64,
                    value: u32::from(value),
                    width: 1,
                    count: 1,
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read<T: Copy>(view: &StorageView) -> Vec<T> {
        let count = view.numel() as usize;
        // SAFETY: host state of the test, not written concurrently.
        unsafe { std::slice::from_raw_parts(view.address() as *const T, count).to_vec() }
    }

    #[test]
    fn reset_restores_a_slot_without_touching_its_neighbors() -> Result<()> {
        let lengths = StorageView::contiguous(
            Storage::host(4 * 4) as Arc<dyn Backing>,
            0,
            DType::I32,
            &[4],
        )?;
        let state = DecodeState::allocate(3, 5, DType::BF16, lengths.clone(), None, None)?;
        assert_eq!(read::<i64>(&state.future_input_tokens), vec![1; 4]);

        // Dirty slots 1 and 2, then reset slot 2 only.
        for slot in [1_usize, 2] {
            // SAFETY: host state of the test.
            unsafe {
                (state.future_token(slot)? as *mut i64).write(77);
                (state.logical_length(slot)? as *mut i64).write(9);
                ((lengths.address() + 4 * slot as u64) as *mut i32).write(9);
                ((state.penalty_counts.address() + (4 * 5 * slot) as u64) as *mut i32).write(3);
            }
        }
        state.set_predicates(&[1, 2], true, None)?;
        state.reset(&[2], None)?;
        assert_eq!(read::<i64>(&state.future_input_tokens), vec![1, 77, 1, 1]);
        assert_eq!(read::<i64>(&state.logical_lengths), vec![0, 9, 0, 0]);
        assert_eq!(read::<i32>(&lengths), vec![0, 9, 0, 0]);
        assert_eq!(read::<u8>(&state.predicates), vec![0, 1, 0, 0]);
        let counts = read::<i32>(&state.penalty_counts);
        assert_eq!(counts[5], 3);
        assert_eq!(counts[10], 0);
        assert!(state.reset(&[4], None).is_err());
        Ok(())
    }

    #[test]
    fn a_request_installs_its_logit_biases_in_its_slot() -> Result<()> {
        let lengths = StorageView::contiguous(
            Storage::host(4 * 4) as Arc<dyn Backing>,
            0,
            DType::I32,
            &[4],
        )?;
        let state = DecodeState::allocate(3, 5, DType::BF16, lengths, None, None)?;
        state.install_bias(2, &[(1, 0.5), (4, -2.0)], None)?;
        let biases = read::<f32>(&state.logit_bias);
        assert_eq!(&biases[10..15], &[0.0, 0.5, 0.0, 0.0, -2.0]);
        assert!(
            biases[..10]
                .iter()
                .chain(&biases[15..])
                .all(|&bias| bias == 0.0)
        );
        // Token 5 is outside the vocabulary of five tokens.
        state.install_bias(1, &[(5, 1.0)], None)?;
        assert!(
            read::<f32>(&state.logit_bias)[5..10]
                .iter()
                .all(|&bias| bias == 0.0)
        );
        assert!(state.install_bias(4, &[(0, 1.0)], None).is_err());
        Ok(())
    }

    #[test]
    fn a_resumed_request_counts_its_generated_tokens() -> Result<()> {
        let lengths = StorageView::contiguous(
            Storage::host(4 * 4) as Arc<dyn Backing>,
            0,
            DType::I32,
            &[4],
        )?;
        let state = DecodeState::allocate(3, 5, DType::BF16, lengths, None, None)?;
        state.count_generated(2, &[4, 1, 4], None)?;
        let counts = read::<i32>(&state.penalty_counts);
        assert_eq!(&counts[10..15], &[0, 1, 0, 0, 2]);
        assert!(
            counts[..10]
                .iter()
                .chain(&counts[15..])
                .all(|count| *count == 0)
        );
        assert!(state.count_generated(1, &[5], None).is_err());
        Ok(())
    }
}
