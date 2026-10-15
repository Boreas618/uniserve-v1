use std::sync::Arc;

use uniserve_core::TokenLogprob;
use uniserve_worker::ffi::{Backing, DType};
use uniserve_worker::storage::memory::{Memory, StorageView};
use uniserve_worker::{EventPool, LogprobLayout, OutputBuffer, OutputStorage, Result};

/// A host column holding `values`, as a CPU lane's sampler produces it.
fn column<T: Copy>(values: &[T], dtype: DType) -> Result<StorageView> {
    let memory = Memory::host(size_of_val(values));
    // SAFETY: fresh host memory of exactly this size, owned by the test.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), memory.address() as *mut T, values.len());
    }
    StorageView::contiguous(memory as Arc<dyn Backing>, 0, dtype, &[values.len() as i64])
}

/// An output buffer whose token words are a host capture of `words`.
fn buffer(words: Vec<i64>) -> Result<OutputBuffer> {
    let mut buffer = OutputBuffer::new(OutputStorage::allocate(None, words.len())?, 1, &[], false);
    let (offset, count) =
        buffer.capture(&column(&words, DType::I64)?, None, &EventPool::default())?;
    assert_eq!((offset, count), (0, words.len()));
    Ok(buffer)
}

fn complete(buffer: &mut OutputBuffer) -> Result<()> {
    // Host captures are sealed on no stream.
    buffer.seal::<&uniserve_cuda::Stream>(&[], &EventPool::default())?;
    buffer.copies_finished();
    buffer.readback()
}

fn score(value: f32) -> i64 {
    i64::from(value.to_bits() as i32)
}

fn entry(token_id: u32, logprob: f32, rank: u32) -> TokenLogprob {
    TokenLogprob {
        token_id,
        logprob,
        rank,
    }
}

#[test]
fn scores_preserve_requested_row_order_ranks_and_unique_tokens() -> Result<()> {
    // Two scored rows from a larger batch. The second requests fewer top
    // tokens, while the first repeats its selection in both other fields.
    let words = [
        vec![5, 9],
        vec![score(-0.5), score(-2.0)],
        vec![1, 4],
        vec![5, 7, 2, 8],
        vec![score(-0.5), score(-0.5), score(-1.0), score(-1.5)],
        vec![1, 1, 1, 2],
        vec![
            score(-0.5),
            score(f32::NEG_INFINITY),
            score(-2.0),
            score(-1.0),
        ],
        vec![1, 10, 4, 1],
    ]
    .concat();
    let count = words.len();
    let mut output = buffer(words)?;
    output.register_logprobs(
        0,
        count,
        LogprobLayout {
            rows: vec![4, 1],
            counts: vec![2, 1],
            requested_ids: vec![vec![5, 3], vec![9, 2]],
            max_count: 2,
            max_requested: 2,
        },
    )?;
    complete(&mut output)?;

    assert_eq!(output.logprob_entries((0, count, 4))?, 5);
    assert_eq!(output.logprob_bytes(&[(0, count, 4), (0, count, 1)])?, 116);
    assert_eq!(
        output.logprob_values((0, count, 1))?,
        [entry(9, -2.0, 4), entry(2, -1.0, 1)]
    );
    assert_eq!(
        output.logprob_values((0, count, 4))?,
        [
            entry(5, -0.5, 1),
            entry(7, -0.5, 1),
            entry(3, f32::NEG_INFINITY, 10)
        ]
    );
    Ok(())
}

#[test]
fn selected_score_needs_no_top_or_requested_columns() -> Result<()> {
    let mut output = buffer(vec![12, 0, score(-3.5), 8])?;
    output.register_logprobs(
        1,
        3,
        LogprobLayout {
            rows: vec![6],
            counts: vec![0],
            requested_ids: vec![vec![]],
            max_count: 0,
            max_requested: 0,
        },
    )?;
    complete(&mut output)?;

    assert_eq!(output.logprob_entries((1, 3, 6))?, 1);
    assert_eq!(output.logprob_values((1, 3, 6))?, [entry(0, -3.5, 8)]);
    Ok(())
}

#[test]
fn readback_rejects_unfinished_copies_and_out_of_range_reads() -> Result<()> {
    let mut output = buffer(vec![3, 7])?;
    assert!(output.readback().is_err());
    complete(&mut output)?;

    assert_eq!(output.read_tokens(1, 1)?, [7]);
    assert!(output.read_tokens(2, 1).is_err());
    assert!(output.read_tokens(usize::MAX, 1).is_err());
    Ok(())
}

#[test]
fn score_layout_must_fit_its_capture_and_row_widths() -> Result<()> {
    let mut output = buffer(vec![0; 6])?;
    for (counts, requested_ids, max_count, max_requested) in [
        (vec![], vec![vec![]], 1, 0),
        (vec![2], vec![vec![]], 1, 0),
        (vec![1], vec![vec![3]], 1, 0),
        (vec![0], vec![vec![]], 0, 0),
    ] {
        assert!(
            output
                .register_logprobs(
                    0,
                    6,
                    LogprobLayout {
                        rows: vec![0],
                        counts,
                        requested_ids,
                        max_count,
                        max_requested,
                    },
                )
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn narrow_token_columns_widen_on_readback() -> Result<()> {
    let mut output = OutputBuffer::new(OutputStorage::allocate(None, 8)?, 1, &[], false);
    let events = EventPool::default();
    let (signed, signed_count) =
        output.capture(&column(&[-3_i32, 7, i32::MIN], DType::I32)?, None, &events)?;
    let (unsigned, unsigned_count) =
        output.capture(&column(&[u32::MAX, 1], DType::U32)?, None, &events)?;
    let (wide, wide_count) = output.capture(&column(&[i64::MAX], DType::I64)?, None, &events)?;
    assert!(
        output
            .capture(&column(&[1.0_f32], DType::F32)?, None, &events)
            .is_err()
    );
    complete(&mut output)?;

    assert_eq!(
        output.read_tokens(signed, signed_count)?,
        [-3, 7, i64::from(i32::MIN)]
    );
    assert_eq!(
        output.read_tokens(unsigned, unsigned_count)?,
        [i64::from(u32::MAX), 1]
    );
    assert_eq!(output.read_tokens(wide, wide_count)?, [i64::MAX]);
    Ok(())
}

#[test]
fn byte_captures_fill_from_the_back_and_bound_token_capacity() -> Result<()> {
    let mut output = OutputBuffer::new(OutputStorage::allocate(None, 2)?, 1, &[], false);
    let events = EventPool::default();
    let bytes = output.capture_bytes(&column(&[1_u8, 2, 3], DType::U8)?, None, &events)?;
    // Three bytes from the back leave one whole token word.
    output.capture(&column(&[9_i64], DType::I64)?, None, &events)?;
    assert!(
        output
            .capture(&column(&[9_i64], DType::I64)?, None, &events)
            .is_err()
    );
    complete(&mut output)?;

    // SAFETY: the capture is host memory that no copy writes any more.
    let captured = unsafe { std::slice::from_raw_parts(bytes.address() as *const u8, 3) };
    assert_eq!(captured, [1, 2, 3]);
    assert_eq!(output.read_tokens(0, 1)?, [9]);
    Ok(())
}
