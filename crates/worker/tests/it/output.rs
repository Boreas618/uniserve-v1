//! Output readback of the words a producer writes into its lease: token
//! words and the logprob scores packed among them.

use ipc::TokenLogprob;
use worker::storage::{LogprobLayout, OutputBuffer, OutputStorage};
use worker::{EventPool, Result};

/// A host output buffer whose lent words hold `words`, as the executor
/// writes them.
fn buffer(words: &[i64]) -> Result<OutputBuffer> {
    let mut buffer = OutputBuffer::new(OutputStorage::allocate(None, words.len())?, 1, &[], false);
    let (offset, lent) = buffer.lend_words()?;
    assert_eq!((offset, lent.shape()), (0, &[words.len() as i64][..]));
    // SAFETY: the lent host words belong to this buffer and nothing else
    // writes them.
    unsafe {
        std::ptr::copy_nonoverlapping(words.as_ptr(), lent.address() as *mut i64, words.len());
    }
    Ok(buffer)
}

/// Seal a host buffer, on no stream.
fn seal(buffer: &mut OutputBuffer) -> Result<()> {
    buffer.seal::<&cuda::Stream>(&[], &EventPool::default())?;
    Ok(())
}

/// The written words' copies finished: they are read back.
fn complete(buffer: &mut OutputBuffer) -> Result<()> {
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
    // Two scored rows from a larger batch. The second reports fewer top
    // tokens, which follow the first's, while the first repeats its
    // selection among its top tokens.
    let words = [
        vec![5, 9],
        vec![score(-0.5), score(-2.0)],
        vec![1, 4],
        vec![5, 7, 2],
        vec![score(-0.5), score(-0.5), score(-1.0)],
        vec![1, 1, 1],
    ]
    .concat();
    let count = words.len();
    let mut output = buffer(&words)?;
    seal(&mut output)?;
    output.register_lent(
        count,
        vec![(
            0,
            count,
            LogprobLayout {
                rows: vec![4, 1],
                counts: vec![2, 1],
            },
        )],
    )?;
    complete(&mut output)?;

    assert_eq!(
        output.logprob_values((0, count, 1))?,
        [entry(9, -2.0, 4), entry(2, -1.0, 1)]
    );
    assert_eq!(
        output.logprob_values((0, count, 4))?,
        [entry(5, -0.5, 1), entry(7, -0.5, 1)]
    );
    Ok(())
}

#[test]
fn selected_score_needs_no_top_columns() -> Result<()> {
    let mut output = buffer(&[12, 0, score(-3.5), 8])?;
    seal(&mut output)?;
    output.register_lent(
        4,
        vec![(
            1,
            3,
            LogprobLayout {
                rows: vec![6],
                counts: vec![0],
            },
        )],
    )?;
    complete(&mut output)?;

    assert_eq!(output.logprob_values((1, 3, 6))?, [entry(0, -3.5, 8)]);
    Ok(())
}

#[test]
fn readback_rejects_unfinished_copies_and_out_of_range_reads() -> Result<()> {
    let mut output = buffer(&[3, 7])?;
    assert!(output.readback().is_err());
    seal(&mut output)?;
    output.register_lent(2, Vec::new())?;
    complete(&mut output)?;

    assert_eq!(output.read_tokens(1, 1)?, [7]);
    assert!(output.read_tokens(2, 1).is_err());
    assert!(output.read_tokens(usize::MAX, 1).is_err());
    Ok(())
}

#[test]
fn score_layout_must_fit_its_column() -> Result<()> {
    let mut output = buffer(&[0; 6])?;
    seal(&mut output)?;
    // A row of one top token fills the column; a missing count, or a count
    // whose words do not fill it, is refused.
    for counts in [vec![], vec![2], vec![0], vec![1, 0]] {
        assert!(
            output
                .register_lent(
                    6,
                    vec![(
                        0,
                        6,
                        LogprobLayout {
                            rows: vec![0],
                            counts,
                        },
                    )],
                )
                .is_err()
        );
    }
    Ok(())
}
