//! The completion records the executor writes for a token batch, their
//! check before the batch's commit point and their application to the
//! calls' pending outputs after it.
//!
//! The executor's `execute` writes into the batch's `records` host tensor
//! and returns how many words it wrote. The layout, in `int64` words:
//!
//! - a header of [`HEADER`] words: [`MAGIC`], the call count, the logprob
//!   segment count, the output words written, the graph mode (-1 none, 0
//!   eager, 1 replay, 2 capture), the live tokens and the padding tokens
//!   the forward evaluated beside them, its rows and draft tokens, the
//!   agreed expert capacity (0 none) and two reserved words;
//! - one record of [`CALL_WORDS`] words per call, call `i` at word
//!   `HEADER + CALL_WORDS * i`: its [`flag`] bits, its progress (logical position, draws, visible
//!   and computed lengths, its prompt logits' readiness in a flag), its
//!   sampling completion (offset, extent and its task among the extent's
//!   four fields), its sampled logprobs (segment and packed row; a
//!   verification's rows for its drafts' positions follow that row), a range
//!   of candidate, canvas or proposed draft words (offset, count), its tail
//!   (record word index and words, zero without one), and two reserved
//!   words;
//! - the logprob segments, back to back after the calls: `[offset, count,
//!   n, rows[n], counts[n]]`, each one packed column of the output words
//!   (`LogprobLayout`), named by their ordinal;
//! - the tails, after every segment: `[n_prompt, (segment, packed row) ×
//!   n_prompt, n_drafts, drafts…, one-based terminal draft prefix or -1]`.
//!
//! `uniserve.executor.text.protocol` writes them (`RecordWriter`).
//!
//! Output word offsets count from the first word the batch lent the
//! executor. [`Records::parse`] refuses a record that does not hold: its
//! ranges lie within the output words written, its columns match their
//! layouts, inactive calls report nothing and active ones that they ran,
//! progress never moves backwards and stays within the slot's tables.

use std::ops::Range;

use crate::cuda_graph::GraphMode;
use crate::executor::PendingOutput;
use crate::storage::LogprobLayout;
use crate::{Error, Result};

/// The first header word: `TKRC` in the high word, the record layout's
/// version in the low word.
pub(crate) const MAGIC: i64 = (0x544B_5243 << 32) | 3;

/// Words of the header.
pub(crate) const HEADER: usize = 12;

/// Words of one call's record.
pub(crate) const CALL_WORDS: usize = 16;

/// Fields of a sampling completion: valid, active, token and accepted
/// drafts, each one word per task, field-major
/// (`uniserve.executor.text.protocol.COMPLETION_FIELDS`).
pub(crate) const COMPLETION_FIELDS: usize = 5;

/// Bits of a call record's flags.
pub(crate) mod flag {
    /// The call's forward ran.
    pub(crate) const RAN: i64 = 1;
    /// Words 1-4 hold the call's progress.
    pub(crate) const PROGRESS: i64 = 1 << 1;
    /// Words 5-7 hold its sampling completion.
    pub(crate) const SAMPLING: i64 = 1 << 2;
    /// Words 8-9 hold its sampled logprobs.
    pub(crate) const LOGPROB: i64 = 1 << 3;
    /// Its tail holds its prompt logprob rows, possibly none.
    pub(crate) const PROMPT: i64 = 1 << 4;
    /// Words 10-11 hold its candidate scores.
    pub(crate) const CANDIDATES: i64 = 1 << 5;
    /// Words 10-11 hold its canvas step's outcome and tokens.
    pub(crate) const CANVAS: i64 = 1 << 6;
    /// Its tail holds the drafts it verified and their terminal prefix.
    pub(crate) const SPECULATION: i64 = 1 << 7;
    /// The request's retained prompt logits are ready after the call.
    pub(crate) const PROMPT_LOGITS_READY: i64 = 1 << 8;
    /// Words 10-11 hold the draft tokens the call proposed, one per word.
    pub(crate) const PROPOSALS: i64 = 1 << 9;

    pub(super) const ALL: i64 = (1 << 10) - 1;
}

/// The record words the worker allocates for a batch of `calls` calls and
/// `tokens` flat input tokens, which every batch's records fit; the
/// executor refuses a batch whose records would not.
pub(crate) fn capacity(calls: usize, tokens: usize) -> usize {
    // Beside the header, a call takes at most its record, its tail's three
    // counts, one sampling task's segment row (2), one prompt chunk's
    // segment header (3) and one sampled-logprob segment header (3; a
    // microbatch's segment has a task, so there are no more segments than
    // calls).
    const PER_CALL: usize = CALL_WORDS + 3 + 2 + 3 + 3;
    // A token takes at most a scored prompt token's segment row (2), its
    // tail entry (2) and another chunk's header (3), more than a verified
    // position's task segment row (2) and draft (1).
    const PER_TOKEN: usize = 2 + 2 + 3;
    HEADER + PER_CALL * calls + PER_TOKEN * tokens
}

/// What a call's record is checked against: the call's column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CallLimits {
    pub(crate) active: bool,
    pub(crate) logical: u64,
    pub(crate) rng: u64,
    /// The tokens the slot's block tables hold.
    pub(crate) allocated: u64,
    pub(crate) max_tokens: u64,
    /// The draft tokens the call may propose.
    pub(crate) proposals: u64,
}

/// A call's progress as its record states it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Progress {
    pub(crate) logical: u64,
    pub(crate) rng: u64,
    pub(crate) visible: u64,
    pub(crate) computed: u64,
    pub(crate) prompt_logits_ready: bool,
}

/// One call's parsed record.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CallRecord {
    pub(crate) ran: bool,
    pub(crate) progress: Option<Progress>,
    /// Offset, extent and row of its sampling completion.
    pub(crate) sampling: Option<(usize, usize, usize)>,
    /// Segment and packed row of its sampled logprobs. A verification's
    /// rows follow it in the segment, one per candidate position: its
    /// current token's, then each draft's.
    pub(crate) logprob: Option<(usize, usize)>,
    /// Segment and packed row of each scored prompt token, in prompt order;
    /// `Some` (possibly empty) for a call that scored its prompt.
    pub(crate) prompt: Option<Vec<(usize, usize)>>,
    pub(crate) candidates: Option<(usize, usize)>,
    pub(crate) canvas: Option<(usize, usize)>,
    /// Offset and count of the draft tokens the call proposed.
    pub(crate) proposals: Option<(usize, usize)>,
    /// The drafts a verification scored and the prefix ending at a terminal
    /// draft.
    pub(crate) speculation: Option<(Vec<u32>, Option<usize>)>,
}

/// One packed logprob column of the output words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Segment {
    pub(crate) offset: usize,
    pub(crate) count: usize,
    pub(crate) rows: Vec<usize>,
    pub(crate) counts: Vec<usize>,
}

impl Segment {
    fn layout(&self) -> LogprobLayout {
        LogprobLayout {
            rows: self.rows.clone(),
            counts: self.counts.clone(),
        }
    }
}

/// A batch's parsed completion records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Records {
    /// Output words the executor wrote from the first lent word.
    pub(crate) written: usize,
    /// How the batch's forward ran; `None` when none ran.
    pub(crate) mode: Option<GraphMode>,
    pub(crate) live_tokens: u64,
    /// Tokens the forward evaluated beside the live ones: its graph's
    /// padding.
    pub(crate) padding_tokens: u64,
    pub(crate) rows: u64,
    pub(crate) draft_tokens: u64,
    /// The expert step capacity the union agreed; zero without one.
    pub(crate) capacity: u64,
    pub(crate) calls: Vec<CallRecord>,
    pub(crate) segments: Vec<Segment>,
}

/// Reads words of the records the executor wrote, refusing any outside
/// them.
struct Reader<'a> {
    words: &'a [i64],
}

impl Reader<'_> {
    fn word(&self, index: usize) -> Result<i64> {
        self.words.get(index).copied().ok_or_else(|| {
            Error::Interface(format!(
                "a token batch's records end at word {}, before word {index}",
                self.words.len()
            ))
        })
    }

    /// A word that counts or locates something: never negative.
    fn count(&self, index: usize) -> Result<usize> {
        let value = self.word(index)?;
        usize::try_from(value).map_err(|_| {
            Error::Interface(format!(
                "word {index} of a token batch's records is negative ({value})"
            ))
        })
    }

    fn counts(&self, range: Range<usize>) -> Result<Vec<usize>> {
        range.map(|index| self.count(index)).collect()
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Interface(message.into())
}

impl Records {
    /// Parse and check the first `used` words of `words`, the records of a
    /// batch whose calls' columns state `limits` and which lent the
    /// executor `lent` output words.
    pub(crate) fn parse(
        words: &[i64],
        used: usize,
        limits: &[CallLimits],
        lent: usize,
    ) -> Result<Self> {
        let words = words.get(..used).ok_or_else(|| {
            invalid(format!(
                "the executor wrote {used} record words into {}",
                words.len()
            ))
        })?;
        let reader = Reader { words };
        if reader.word(0)? != MAGIC {
            return Err(invalid("a token batch's records lack their magic word"));
        }
        if reader.count(1)? != limits.len() {
            return Err(invalid(format!(
                "a token batch of {} calls has records of {}",
                limits.len(),
                reader.count(1)?
            )));
        }
        let written = reader.count(3)?;
        if written > lent {
            return Err(invalid(format!(
                "the executor wrote {written} output words into {lent}"
            )));
        }
        let mode = match reader.word(4)? {
            -1 => None,
            code => Some(
                GraphMode::from_code(code)
                    .ok_or_else(|| invalid(format!("a token batch ran in mode {code}")))?,
            ),
        };
        // An output range: its words lie within those written. An empty
        // range may sit at any offset up to the end of the written words.
        let range = |offset: usize, count: usize, what: &str| -> Result<(usize, usize)> {
            match offset.checked_add(count) {
                Some(end) if end <= written => Ok((offset, count)),
                _ => Err(invalid(format!(
                    "a {what} of {count} words at {offset} lies outside the {written} output words written"
                ))),
            }
        };

        // Segments, after the call records.
        let mut segments = Vec::new();
        let mut at = HEADER + CALL_WORDS * limits.len();
        for _ in 0..reader.count(2)? {
            let (offset, count) =
                range(reader.count(at)?, reader.count(at + 1)?, "logprob column")?;
            let rows = reader.count(at + 2)?;
            let start = at + 3;
            let segment = Segment {
                offset,
                count,
                rows: reader.counts(start..start + rows)?,
                counts: reader.counts(start + rows..start + 2 * rows)?,
            };
            if segment.layout().words() != Some(count) {
                return Err(invalid(
                    "a logprob column's words disagree with its row layout",
                ));
            }
            segments.push(segment);
            at = start + 2 * rows;
        }
        // A reference to packed row `row` of segment `segment`.
        let logprob_row = |segment: usize, row: usize| -> Result<(usize, usize)> {
            match segments.get(segment) {
                Some(found) if row < found.rows.len() => Ok((segment, row)),
                _ => Err(invalid(format!(
                    "a call reports logprob row {row} of segment {segment}, which has no such row"
                ))),
            }
        };

        let mut calls = Vec::with_capacity(limits.len());
        for (index, limit) in limits.iter().enumerate() {
            let base = HEADER + CALL_WORDS * index;
            let word = |offset: usize| reader.word(base + offset);
            let count = |offset: usize| reader.count(base + offset);
            let flags = word(0)?;
            if flags & !flag::ALL != 0 {
                return Err(invalid(format!(
                    "call {index}'s record has unknown flags {flags:#x}"
                )));
            }
            let has = |bit: i64| flags & bit != 0;
            if !limit.active {
                if flags != 0 {
                    return Err(invalid(format!(
                        "call {index} did not execute but its record reports results"
                    )));
                }
                calls.push(CallRecord::default());
                continue;
            }
            if !has(flag::RAN) {
                return Err(invalid(format!(
                    "call {index} executes but its record says it did not run"
                )));
            }
            let mut record = CallRecord {
                ran: true,
                ..CallRecord::default()
            };
            if has(flag::PROGRESS) {
                let progress = Progress {
                    logical: count(1)? as u64,
                    rng: count(2)? as u64,
                    visible: count(3)? as u64,
                    computed: count(4)? as u64,
                    prompt_logits_ready: has(flag::PROMPT_LOGITS_READY),
                };
                if progress.logical < limit.logical || progress.rng < limit.rng {
                    return Err(invalid(format!(
                        "call {index}'s progress moves its position or draws backwards"
                    )));
                }
                if progress.visible > progress.computed || progress.computed > limit.allocated {
                    return Err(invalid(format!(
                        "call {index} reports {} visible and {} computed tokens of {} allocated",
                        progress.visible, progress.computed, limit.allocated
                    )));
                }
                record.progress = Some(progress);
            } else if has(flag::PROMPT_LOGITS_READY) {
                return Err(invalid(format!(
                    "call {index} reports ready prompt logits without its progress"
                )));
            }
            if has(flag::SAMPLING) {
                let (offset, extent) = range(count(5)?, count(6)?, "sampling completion")?;
                let row = count(7)?;
                if extent % COMPLETION_FIELDS != 0 || row >= extent / COMPLETION_FIELDS {
                    return Err(invalid(format!(
                        "call {index}'s sampling row {row} lies outside its {extent}-word completion"
                    )));
                }
                record.sampling = Some((offset, extent, row));
            }
            if has(flag::LOGPROB) {
                record.logprob = Some(logprob_row(count(8)?, count(9)?)?);
            }
            if [flag::CANDIDATES, flag::CANVAS, flag::PROPOSALS]
                .into_iter()
                .filter(|&bit| has(bit))
                .count()
                > 1
            {
                return Err(invalid(format!(
                    "call {index} reports two kinds of words in one range"
                )));
            }
            if has(flag::PROPOSALS) {
                let (offset, proposed) = range(count(10)?, count(11)?, "draft proposal")?;
                if proposed as u64 > limit.proposals {
                    return Err(invalid(format!(
                        "call {index} proposed {proposed} drafts beyond its {}",
                        limit.proposals
                    )));
                }
                record.proposals = Some((offset, proposed));
            }
            if has(flag::CANDIDATES) {
                record.candidates = Some(range(count(10)?, count(11)?, "candidate range")?);
            }
            if has(flag::CANVAS) {
                record.canvas = Some(range(count(10)?, count(11)?, "canvas range")?);
            }
            if has(flag::PROMPT) || has(flag::SPECULATION) {
                let start = count(12)?;
                let length = count(13)?;
                let tail = start
                    .checked_add(length)
                    .and_then(|end| words.get(start..end))
                    .ok_or_else(|| {
                        invalid(format!("call {index}'s tail lies outside the records"))
                    })?;
                let tail = Reader { words: tail };
                let prompts = tail.count(0)?;
                let mut prompt = Vec::with_capacity(prompts);
                for entry in 0..prompts {
                    prompt.push(logprob_row(
                        tail.count(1 + 2 * entry)?,
                        tail.count(2 + 2 * entry)?,
                    )?);
                }
                let drafts_at = 1 + 2 * prompts;
                let drafts = tail.count(drafts_at)?;
                if drafts as u64 > limit.max_tokens {
                    return Err(invalid(format!(
                        "call {index} verified {drafts} drafts beyond its {} tokens",
                        limit.max_tokens
                    )));
                }
                let tokens = (drafts_at + 1..drafts_at + 1 + drafts)
                    .map(|position| {
                        u32::try_from(tail.word(position)?)
                            .map_err(|_| invalid("a draft lies outside the token id range"))
                    })
                    .collect::<Result<Vec<u32>>>()?;
                let terminal = match tail.word(drafts_at + 1 + drafts)? {
                    -1 => None,
                    prefix => Some(
                        usize::try_from(prefix)
                            .ok()
                            .filter(|&prefix| prefix <= drafts)
                            .ok_or_else(|| {
                                invalid(format!(
                                    "call {index}'s terminal prefix {prefix} lies outside its drafts"
                                ))
                            })?,
                    ),
                };
                if has(flag::PROMPT) {
                    record.prompt = Some(prompt);
                } else if !prompt.is_empty() {
                    return Err(invalid(format!(
                        "call {index}'s tail reports prompt rows it does not flag"
                    )));
                }
                if has(flag::SPECULATION) {
                    // A verification's logprob rows cover every position
                    // it may commit.
                    if let Some((segment, row)) = record.logprob {
                        logprob_row(segment, row + tokens.len())?;
                    }
                    record.speculation = Some((tokens, terminal));
                } else if drafts > 0 {
                    return Err(invalid(format!(
                        "call {index}'s tail reports drafts it does not flag"
                    )));
                }
            }
            calls.push(record);
        }

        Ok(Self {
            written,
            mode,
            live_tokens: reader.count(5)? as u64,
            padding_tokens: reader.count(6)? as u64,
            rows: reader.count(7)? as u64,
            draft_tokens: reader.count(8)? as u64,
            capacity: reader.count(9)? as u64,
            calls,
            segments,
        })
    }

    /// Apply call `index`'s record to its pending output, whose output
    /// words start at word `lent` of the batch's lease: its progress, then
    /// the ranges its result decodes from, then a verification's drafts
    /// over the progress it reported.
    pub(crate) fn apply(
        &self,
        index: usize,
        output: &mut PendingOutput,
        lent: usize,
    ) -> Result<()> {
        let record = self
            .calls
            .get(index)
            .ok_or_else(|| Error::Invariant("a token call has no record".into()))?;
        // A packed row decodes under the row its layout reports it for.
        let segment = |(segment, row): (usize, usize)| -> Result<(usize, usize, usize)> {
            let found = self
                .segments
                .get(segment)
                .ok_or_else(|| Error::Invariant("a logprob row names no segment".into()))?;
            let reported = *found
                .rows
                .get(row)
                .ok_or_else(|| Error::Invariant("a logprob row lies outside its segment".into()))?;
            Ok((lent + found.offset, found.count, reported))
        };
        if let Some(progress) = record.progress {
            output.progress.logical_position = progress.logical;
            output.progress.rng_counter = progress.rng;
            output.progress.cache_visible_len = progress.visible;
            output.progress.cache_computed_len = progress.computed;
            output.progress.prompt_logits_ready = progress.prompt_logits_ready;
        }
        output.sampling_range = record
            .sampling
            .map(|(offset, extent, row)| (lent + offset, extent, row));
        let positions = record
            .speculation
            .as_ref()
            .map_or(1, |(drafts, _)| drafts.len() + 1);
        output.logprob_ranges = match record.logprob {
            Some((first, row)) => (row..row + positions)
                .map(|row| segment((first, row)))
                .collect::<Result<_>>()?,
            None => Vec::new(),
        };
        if let Some(prompt) = &record.prompt {
            output.prompt_logprob_ranges = prompt
                .iter()
                .map(|&entry| segment(entry))
                .collect::<Result<_>>()?;
        }
        output.candidate_range = record
            .candidates
            .map(|(offset, count)| (lent + offset, count));
        output.canvas_range = record.canvas.map(|(offset, count)| (lent + offset, count));
        output.proposal_range = record
            .proposals
            .map(|(offset, count)| (lent + offset, count));
        if let Some((drafts, terminal)) = &record.speculation {
            let progress = output.progress;
            output.set_speculation(
                drafts.clone(),
                *terminal,
                progress.cache_visible_len,
                progress.cache_computed_len,
            );
        }
        Ok(())
    }

    /// The logprob columns among the written output words, as
    /// `OutputBuffer::register_lent` takes them.
    pub(crate) fn columns(&self) -> Vec<(usize, usize, LogprobLayout)> {
        self.segments
            .iter()
            .map(|segment| (segment.offset, segment.count, segment.layout()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ipc::{Call, CallId, CallKind, ForwardMode, RequestId, RequestKey};

    use super::*;
    use crate::request::RequestProgress;

    /// Records of a batch as the executor writes them, built by call.
    struct Writer {
        words: Vec<i64>,
        calls: usize,
        tails: Vec<Vec<i64>>,
    }

    impl Writer {
        fn new(calls: usize, written: i64) -> Self {
            let mut words = vec![0; HEADER + CALL_WORDS * calls];
            words[0] = MAGIC;
            words[1] = calls as i64;
            words[3] = written;
            words[4] = 1;
            words[5] = 3;
            words[6] = 8;
            Self {
                words,
                calls,
                tails: vec![Vec::new(); calls],
            }
        }

        fn call(&mut self, index: usize, fields: &[(usize, i64)]) {
            for &(offset, value) in fields {
                self.words[HEADER + CALL_WORDS * index + offset] = value;
            }
        }

        fn segment(&mut self, offset: i64, rows: &[i64], counts: &[i64]) {
            let words = 3 * (rows.len() as i64 + counts.iter().sum::<i64>());
            self.words.extend([offset, words, rows.len() as i64]);
            self.words.extend(rows);
            self.words.extend(counts);
            self.words[2] += 1;
        }

        fn tail(&mut self, index: usize, tail: &[i64]) {
            self.tails[index] = tail.to_vec();
        }

        /// The records and the words used.
        fn finish(mut self) -> (Vec<i64>, usize) {
            for index in 0..self.calls {
                let tail = std::mem::take(&mut self.tails[index]);
                if tail.is_empty() {
                    continue;
                }
                let start = self.words.len() as i64;
                self.call(index, &[(12, start), (13, tail.len() as i64)]);
                self.words.extend(tail);
            }
            let used = self.words.len();
            // Capacity beyond the words used, as a batch allocates it.
            self.words.resize(used + 8, 0);
            (self.words, used)
        }
    }

    fn active(allocated: u64, max_tokens: u64) -> CallLimits {
        CallLimits {
            active: true,
            logical: 10,
            rng: 2,
            allocated,
            max_tokens,
            proposals: 0,
        }
    }

    /// A decode with sampled logprobs, a skipped call, and a verification
    /// with prompt-free speculation and a logprob row per position it may
    /// commit: records parse into each call's ranges and apply to its
    /// pending output at the lent offset.
    #[test]
    fn records_parse_and_apply_each_calls_ranges() -> Result<()> {
        use flag::*;
        let mut writer = Writer::new(3, 32);
        writer.call(
            0,
            &[
                (0, RAN | PROGRESS | SAMPLING | LOGPROB),
                (1, 11),
                (2, 3),
                (3, 11),
                (4, 11),
                (5, 0),
                (6, 2 * COMPLETION_FIELDS as i64),
                (7, 0),
                (8, 0),
                (9, 0),
            ],
        );
        writer.call(
            2,
            &[
                (
                    0,
                    RAN | PROGRESS | SAMPLING | SPECULATION | PROMPT_LOGITS_READY | LOGPROB,
                ),
                (1, 10),
                (2, 2),
                (3, 10),
                (4, 14),
                (5, 0),
                (6, 2 * COMPLETION_FIELDS as i64),
                (7, 1),
                (8, 1),
                (9, 0),
            ],
        );
        // One sampled row (request row 5) with two top tokens: 9 words,
        // after the two tasks' completion words; then the verification's
        // four rows, its current token's and its three drafts', without top
        // tokens: 12 words.
        writer.segment(10, &[5], &[2]);
        writer.segment(19, &[0, 1, 2, 3], &[0, 0, 0, 0]);
        writer.tail(2, &[0, 3, 21, 22, 23, 2]);
        let (words, used) = writer.finish();
        let limits = [active(16, 1), CallLimits::default(), active(16, 3)];
        let records = Records::parse(&words, used, &limits, 32)?;
        assert_eq!(records.written, 32);
        assert_eq!(records.mode, Some(GraphMode::Replay));
        assert_eq!((records.live_tokens, records.padding_tokens), (3, 8));
        assert_eq!(records.calls[1], CallRecord::default());
        assert_eq!(records.calls[0].logprob, Some((0, 0)));
        assert_eq!(
            records.calls[2].speculation,
            Some((vec![21, 22, 23], Some(2)))
        );
        assert_eq!(records.columns()[0].0, 10);

        // Applied at lent word 4: ranges shift by it, progress replaces the
        // coordinates', and the verification keeps its drafts' base.
        let call = Call::new(
            RequestKey::new(1, RequestId(1), 1),
            CallId::new(1, 0),
            CallKind::Forward(ForwardMode::Verify),
        );
        let mut output = PendingOutput::new(&call, RequestProgress::default(), 2)?;
        records.apply(2, &mut output, 4)?;
        assert_eq!(output.sampling_range, Some((4, 2 * COMPLETION_FIELDS, 1)));
        assert_eq!(
            output.logprob_ranges,
            (0..4).map(|row| (23, 12, row)).collect::<Vec<_>>()
        );
        assert_eq!(output.progress.cache_visible_len, 10);
        assert_eq!(output.progress.cache_computed_len, 14);
        assert!(output.progress.prompt_logits_ready);
        let mut decoded = PendingOutput::new(&call, RequestProgress::default(), 0)?;
        records.apply(0, &mut decoded, 4)?;
        assert_eq!(decoded.logprob_ranges, vec![(14, 9, 5)]);
        assert!(decoded.prompt_logprob_ranges.is_empty());
        Ok(())
    }

    /// A prompt chunk's rows reach the call in prompt order through its
    /// tail; a readout's empty candidate range may sit at the end of the
    /// written words.
    #[test]
    fn prompt_rows_and_empty_ranges_are_accepted() -> Result<()> {
        use flag::*;
        let mut writer = Writer::new(2, 6);
        writer.call(0, &[(0, RAN | PROMPT)]);
        writer.call(1, &[(0, RAN | CANDIDATES), (10, 6), (11, 0)]);
        // Two scored prompt rows without top tokens: 6 words.
        writer.segment(0, &[0, 1], &[0, 0]);
        writer.tail(0, &[2, 0, 1, 0, 0, 0, -1]);
        let (words, used) = writer.finish();
        let records = Records::parse(&words, used, &[active(4, 4), active(4, 0)], 6)?;
        assert_eq!(records.calls[0].prompt, Some(vec![(0, 1), (0, 0)]));
        assert_eq!(records.calls[1].candidates, Some((6, 0)));
        Ok(())
    }

    /// Each broken promise of a record refuses the batch's records.
    #[test]
    fn malformed_records_are_refused() {
        use flag::*;
        let refused = |edit: &dyn Fn(&mut Writer), limits: &[CallLimits], lent: usize| {
            let mut writer = Writer::new(limits.len(), 4);
            edit(&mut writer);
            let (words, used) = writer.finish();
            Records::parse(&words, used, limits, lent).is_err()
        };
        let one = [active(8, 2)];
        let ran = |writer: &mut Writer| writer.call(0, &[(0, RAN)]);
        assert!(!refused(&ran, &one, 4));

        // The words written exceed those lent.
        assert!(refused(&ran, &one, 3));
        // An active call that did not run; a skipped call that reports.
        assert!(refused(&|_: &mut Writer| {}, &one, 4));
        assert!(refused(&ran, &[CallLimits::default()], 4));
        // Unknown flags.
        assert!(refused(
            &|writer: &mut Writer| writer.call(0, &[(0, RAN | 1 << 12)]),
            &one,
            4
        ));
        // Progress backwards, or beyond the allocated tokens.
        assert!(refused(
            &|writer: &mut Writer| writer.call(0, &[(0, RAN | PROGRESS), (1, 9), (2, 2)]),
            &one,
            4
        ));
        assert!(refused(
            &|writer: &mut Writer| writer
                .call(0, &[(0, RAN | PROGRESS), (1, 10), (2, 2), (3, 4), (4, 9)]),
            &one,
            4
        ));
        // A sampling completion outside the words written, or misaligned.
        assert!(refused(
            &|writer: &mut Writer| writer.call(0, &[(0, RAN | SAMPLING), (5, 2), (6, 4)]),
            &one,
            4
        ));
        assert!(refused(
            &|writer: &mut Writer| writer.call(0, &[(0, RAN | SAMPLING), (6, 3)]),
            &one,
            4
        ));
        // A packed logprob row its segment lacks.
        assert!(refused(
            &|writer: &mut Writer| {
                writer.call(0, &[(0, RAN | LOGPROB), (8, 0), (9, 1)]);
                writer.segment(0, &[7], &[0]);
            },
            &one,
            4
        ));
        // A segment whose words disagree with its layout.
        assert!(refused(
            &|writer: &mut Writer| {
                ran(writer);
                writer.words.extend([0, 2, 1, 0, 0]);
                writer.words[2] = 1;
            },
            &one,
            4
        ));
        // More drafts than the call's tokens.
        assert!(refused(
            &|writer: &mut Writer| {
                writer.call(0, &[(0, RAN | SPECULATION)]);
                writer.tail(0, &[0, 3, 1, 2, 3, -1]);
            },
            &one,
            4
        ));
        // Another call count, or no magic.
        assert!(refused(
            &|writer: &mut Writer| {
                ran(writer);
                writer.words[1] = 2;
            },
            &one,
            4
        ));
        assert!(refused(
            &|writer: &mut Writer| {
                ran(writer);
                writer.words[0] = 0;
            },
            &one,
            4
        ));
    }
}
