//! The sampling metadata of one forward: its sampled calls' selection
//! inputs, in task order.
//!
//! [`SamplingMetadata::build`] validates every task on the host and
//! produces the columns the runner's `select` (or, for a verification,
//! `verify`) reads besides the logits: per task its sampler
//! ([`SamplingPath`]), slot, gate, forced finish, finish and transition
//! token sets and whether its request's logit biases apply; per candidate
//! row its logits row, draw, parameters and token set. A task's output
//! position is its index.
//!
//! Token sets cross as bitmasks of [`bitmask_words`] int32 words, bit `b` of
//! word `w` holding token `32 * w + b`, with tokens outside the vocabulary
//! dropped. The candidate rows' sets (allowed tokens, less the suppressed
//! ones) are deduplicated into a table the rows index, so the rows of one
//! request share one set.
//!
//! The tasks reporting logprobs pack their words into a column of
//! [`logprob_shape`] rows and top tokens, the padded shape a captured
//! scoring graph holds.
//!
//! A decode forward whose rows all take the top-k sampler or the unshaped
//! greedy path selects in its own graph instead: [`RowSampling`] holds its
//! rows' draws and parameters in row order, and no selection follows.

use std::collections::HashMap;

use ipc::SamplingParams;

use super::{SamplingControls, SamplingPath, TOKEN_VALUE_MASK, candidates, top_k_width};
use crate::memory::LogprobLayout;
use crate::{Error, Result};

/// Parameters per candidate row: temperature, top-p, min-p, typical-p,
/// repetition, frequency and presence penalties, and the top-k width
/// (`uniserve.sampling.top_k`'s columns).
pub(crate) const PARAMETER_COLUMNS: usize = 8;

/// The parameter row of a neutral row: a padding row of a forward that
/// selects its rows (zero temperature, no cutoff, no penalty).
pub(crate) const NEUTRAL: [f32; PARAMETER_COLUMNS] = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];

/// A task's sampler as the runner's `select` names it
/// (`uniserve.sampling.select`).
pub(crate) const GREEDY: i64 = 0;
pub(crate) const TOP_K: i64 = 1;
pub(crate) const CATEGORICAL: i64 = 2;

/// Words of one token set over a `vocab`-token vocabulary.
pub(crate) fn bitmask_words(vocab: usize) -> usize {
    vocab.div_ceil(32)
}

/// The task bucket of `count` tasks of a binding of `capacity` rows: the
/// next power of two, at most the capacity
/// (`crate::cuda_graph::task_buckets`).
pub(crate) fn task_bucket(count: usize, capacity: usize) -> usize {
    count.next_power_of_two().min(capacity)
}

/// The top bucket of `top` reported tokens over a `vocab`-token
/// vocabulary: zero when none is reported, else the next power of two, at
/// most the vocabulary.
pub(crate) fn top_bucket(top: usize, vocab: usize) -> usize {
    if top == 0 {
        0
    } else {
        top.next_power_of_two().min(vocab)
    }
}

/// The fewest top tokens top bucket `top` holds: one past the power of two
/// below it, or the bucket itself for zero and one.
pub(crate) fn least_top(top: usize) -> usize {
    if top < 2 {
        top
    } else {
        (1 << (top - 1).ilog2()) + 1
    }
}

/// The packed shape `(rows, top)` of the logprob column of `rows` tasks
/// reporting at most `top` top tokens each, of a binding of `capacity` rows
/// over a `vocab`-token vocabulary: the rows' task bucket and the top
/// bucket. A shape at most doubles each count, so its words are at most
/// four times the unpadded ones.
pub(crate) fn logprob_shape(
    rows: usize,
    top: usize,
    vocab: usize,
    capacity: usize,
) -> (usize, usize) {
    (task_bucket(rows, capacity), top_bucket(top, vocab))
}

/// One sampled call of an invocation.
pub(crate) struct SampleTask<'a> {
    pub(crate) params: &'a SamplingParams,
    pub(crate) controls: SamplingControls,
    /// Per candidate row; absent for the unshaped greedy path.
    pub(crate) draws: Option<Vec<f32>>,
    pub(crate) slot: u32,
    /// First candidate row in the invocation's logits.
    pub(crate) row: usize,
    /// The call is gated by its request's continuation flag.
    pub(crate) gated: bool,
}

impl SampleTask<'_> {
    fn rows(&self) -> usize {
        self.controls.allowed.len()
    }
}

/// The selection inputs of one forward's tasks, in task order.
pub(crate) struct SamplingMetadata {
    /// Tasks, and candidate rows per task: one, or the most rows of a
    /// verification's tasks.
    pub(crate) tasks: usize,
    pub(crate) width: usize,
    /// `[tasks * width]` each candidate row's logits row; a task's rows past
    /// its own repeat its first row, its draw, parameters and set.
    pub(crate) rows: Vec<i64>,
    /// Each task's sampler.
    pub(crate) paths: Vec<i64>,
    pub(crate) draws: Vec<f32>,
    pub(crate) params: Vec<f32>,
    pub(crate) slots: Vec<i64>,
    pub(crate) gates: Vec<i64>,
    /// `[sets, words]` distinct token sets masked rows select from, and each
    /// candidate row's set (-1 for a row every token may take).
    pub(crate) masks: Vec<i32>,
    pub(crate) mask_rows: Vec<i64>,
    /// Tasks whose request's logit biases apply.
    pub(crate) biased: Vec<u8>,
    /// Tasks that finish whatever they select.
    pub(crate) force: Vec<u8>,
    /// `[tasks, words]` finish and transition token sets; a task without a
    /// transition output has an empty set.
    pub(crate) finish: Vec<i32>,
    pub(crate) transitions: Vec<i32>,
    /// `[tasks, width - 1]` draft tokens, -1 past a task's own.
    pub(crate) drafts: Vec<i64>,
    /// Per task, the one-based length of its draft prefix ending at the
    /// first finish token, or zero.
    pub(crate) terminal: Vec<i64>,
    /// Some task takes the top-k sampler, or the general one; `sorts` when
    /// the general one sorts a whole vocabulary row for some task (a top-k
    /// width beyond the top-k sampler's candidates, a nucleus cutoff without
    /// a top-k width, a typical-probability cutoff).
    pub(crate) top_k: bool,
    pub(crate) categorical: bool,
    pub(crate) sorts: bool,
    /// The tasks reporting logprobs, in task order, and the compact layout
    /// of their words, which start the binding's logprob column.
    pub(crate) logprob_tasks: Vec<i64>,
    pub(crate) logprobs: Option<LogprobLayout>,
    /// The bucket of reporting tasks and the top width their `score` graph
    /// scores at ([`logprob_shape`]); zero without logprobs.
    pub(crate) logprob_bucket: (usize, usize),
}

/// Set `tokens` in `set`, a bitmask over `vocab` tokens; tokens outside
/// the vocabulary are dropped.
fn insert(set: &mut [i32], tokens: &[u32], vocab: usize) {
    for &token in tokens {
        let token = token as usize;
        if token < vocab {
            set[token / 32] |= 1 << (token % 32);
        }
    }
}

/// The set of a row allowed `allowed` (every token without one) and never
/// `suppressed`.
fn permitted(allowed: Option<&[u32]>, suppressed: &[u32], vocab: usize) -> Vec<i32> {
    let words = bitmask_words(vocab);
    let mut set = vec![0; words];
    match allowed {
        Some(tokens) => insert(&mut set, tokens, vocab),
        None => {
            set.fill(-1);
            // Bits past the vocabulary stay clear.
            if !vocab.is_multiple_of(32) {
                set[words - 1] = ((1_u32 << (vocab % 32)) - 1) as i32;
            }
        }
    }
    for &token in suppressed {
        let token = token as usize;
        if token < vocab {
            set[token / 32] &= !(1 << (token % 32));
        }
    }
    set
}

/// Whether a general-path row of `params` sorts its whole vocabulary row.
fn sorts(params: &SamplingParams, vocab: usize) -> bool {
    let top_k = top_k_width(params, vocab);
    let width = if top_k < vocab { top_k } else { 0 };
    width > candidates(vocab)
        || (width == 0 && params.top_p > 0.0 && params.top_p < 1.0)
        || params.typical_p < 1.0
}

impl SamplingMetadata {
    /// Validate `tasks` and lay out their selection inputs. `vocab` is the
    /// global vocabulary, `capacity` the binding's rows (the tasks and the
    /// token sets it holds) and `logprob_capacity` the logprob words a
    /// forward may report.
    pub(crate) fn build(
        tasks: &[SampleTask<'_>],
        vocab: usize,
        capacity: usize,
        logprob_capacity: usize,
    ) -> Result<Self> {
        if vocab > TOKEN_VALUE_MASK as usize {
            return Err(Error::Unsupported(
                "vocabulary exceeds the device token decision range".into(),
            ));
        }
        for task in tasks {
            validate(task, vocab)?;
        }
        if tasks.len() > capacity {
            return Err(Error::Invariant(
                "a forward samples more tasks than its rows".into(),
            ));
        }
        let words = bitmask_words(vocab);
        let width = tasks.iter().map(SampleTask::rows).max().unwrap_or(1);
        let count = tasks.len();
        let mut metadata = Self {
            tasks: count,
            width,
            rows: Vec::with_capacity(count * width),
            paths: Vec::with_capacity(count),
            draws: Vec::with_capacity(count * width),
            params: Vec::with_capacity(count * width * PARAMETER_COLUMNS),
            slots: Vec::with_capacity(count),
            gates: Vec::with_capacity(count),
            masks: Vec::new(),
            mask_rows: Vec::with_capacity(count * width),
            biased: Vec::with_capacity(count),
            force: Vec::with_capacity(count),
            finish: vec![0; count * words],
            transitions: vec![0; count * words],
            drafts: vec![-1; count * (width - 1)],
            terminal: Vec::with_capacity(count),
            top_k: false,
            categorical: false,
            sorts: false,
            logprob_tasks: Vec::new(),
            logprobs: None,
            logprob_bucket: (0, 0),
        };
        // Distinct token sets by their allowed and suppressed tokens.
        let mut sets: HashMap<(Option<&[u32]>, &[u32]), i64> = HashMap::new();
        let mut counts = Vec::new();
        for (index, task) in tasks.iter().enumerate() {
            let params = task.params;
            let controls = &task.controls;
            let path = controls.path(params, vocab);
            metadata.paths.push(match path {
                SamplingPath::Greedy { .. } => GREEDY,
                SamplingPath::TopK => {
                    metadata.top_k = true;
                    TOP_K
                }
                SamplingPath::Categorical => {
                    metadata.categorical = true;
                    metadata.sorts |= sorts(params, vocab);
                    CATEGORICAL
                }
            });
            metadata.slots.push(i64::from(task.slot));
            metadata.gates.push(i64::from(task.gated));
            metadata
                .biased
                .push(u8::from(!params.logit_bias.is_empty()));
            metadata.force.push(u8::from(controls.force_finish));
            insert(
                &mut metadata.finish[index * words..(index + 1) * words],
                &controls.finish_token_ids,
                vocab,
            );
            if controls.return_transition {
                insert(
                    &mut metadata.transitions[index * words..(index + 1) * words],
                    &controls.transition_token_ids,
                    vocab,
                );
            }
            for (column, &token) in controls.draft_token_ids.iter().enumerate() {
                metadata.drafts[index * (width - 1) + column] = i64::from(token);
            }
            metadata
                .terminal
                .push(controls.terminal_draft_prefix.unwrap_or(0) as i64);
            if params.generated_logprobs_requested() {
                metadata.logprob_tasks.push(index as i64);
                counts.push(logprob_count(params.n_logprobs, vocab));
            }

            // The task's candidate rows, then copies of its first.
            let row_parameters = parameters(params, vocab);
            let rows = task.rows();
            for offset in 0..width {
                let own = if offset < rows { offset } else { 0 };
                metadata.rows.push((task.row + own) as i64);
                metadata
                    .draws
                    .push(task.draws.as_ref().map_or(0.0, |values| values[own]));
                metadata.params.extend_from_slice(&row_parameters);
                let allowed = controls.allowed[own].as_deref();
                let set = if allowed.is_none() && controls.suppress.is_empty() {
                    -1
                } else {
                    let next = sets.len() as i64;
                    let set = *sets
                        .entry((allowed, controls.suppress.as_slice()))
                        .or_insert(next);
                    if set == next {
                        if sets.len() > capacity {
                            return Err(Error::Resource(
                                "token sets exceed the binding's capacity",
                            ));
                        }
                        metadata
                            .masks
                            .extend(permitted(allowed, &controls.suppress, vocab));
                    }
                    set
                };
                metadata.mask_rows.push(set);
            }
        }

        if !counts.is_empty() {
            let reporting = counts.len();
            let top = counts.iter().copied().max().unwrap_or(0);
            let layout = LogprobLayout {
                rows: (0..reporting).collect(),
                counts,
            };
            if layout.words().is_none_or(|words| words > logprob_capacity) {
                return Err(Error::Resource(
                    "logprob words exceed the binding's capacity",
                ));
            }
            metadata.logprob_bucket = logprob_shape(reporting, top, vocab, capacity);
            metadata.logprobs = Some(layout);
        }
        Ok(metadata)
    }

    /// The output position of input task `index`: its index.
    pub(crate) fn position(&self, index: usize) -> usize {
        index
    }

    /// Input task `index`'s packed logprob row when it reports logprobs.
    pub(crate) fn logprob_row(&self, index: usize) -> Option<usize> {
        self.logprob_tasks
            .iter()
            .position(|&task| task == index as i64)
    }

    /// Words of the logprob column the tasks' logprobs pack.
    pub(crate) fn logprob_end(&self) -> usize {
        self.logprobs
            .as_ref()
            .and_then(LogprobLayout::words)
            .unwrap_or(0)
    }
}

/// The draws and parameters of a decode forward whose rows the forward
/// samples itself: one row per task, in task order, which is the forward's
/// row order.
pub(crate) struct RowSampling {
    /// One draw per row; zero for a row that selects by argmax.
    pub(crate) draws: Vec<f32>,
    /// [`PARAMETER_COLUMNS`] parameters per row.
    pub(crate) params: Vec<f32>,
}

impl RowSampling {
    /// The rows of `tasks`, each of which takes the top-k sampler or the
    /// unshaped greedy path with one candidate row.
    pub(crate) fn build(tasks: &[SampleTask<'_>], vocab: usize) -> Result<Self> {
        let mut draws = Vec::with_capacity(tasks.len());
        let mut params = Vec::with_capacity(tasks.len() * PARAMETER_COLUMNS);
        for task in tasks {
            validate(task, vocab)?;
            let own = matches!(
                task.controls.path(task.params, vocab),
                SamplingPath::TopK | SamplingPath::Greedy { suppressed: false }
            );
            if !own || task.rows() != 1 {
                return Err(Error::Invariant(
                    "a decode forward samples only top-k and greedy rows".into(),
                ));
            }
            draws.push(task.draws.as_ref().map_or(0.0, |values| values[0]));
            params.extend_from_slice(&parameters(task.params, vocab));
        }
        Ok(Self { draws, params })
    }

    /// `rows` neutral rows: zero draws, and parameters that select by
    /// argmax without a cutoff or penalty.
    pub(crate) fn neutral(rows: usize) -> Self {
        Self {
            draws: vec![0.0; rows],
            params: NEUTRAL.repeat(rows),
        }
    }

    /// The draws and parameters of `rows` forward rows: these rows, then
    /// neutral padding rows.
    pub(crate) fn padded(&self, rows: usize) -> (Vec<f32>, Vec<f32>) {
        let mut draws = self.draws.clone();
        draws.resize(rows.max(draws.len()), 0.0);
        let mut params = self.params.clone();
        while params.len() < draws.len() * PARAMETER_COLUMNS {
            params.extend_from_slice(&NEUTRAL);
        }
        (draws, params)
    }
}

/// The top count a call reports per position, of its `count` top tokens
/// (the generated tokens' `n_logprobs` or the prompt's
/// `n_prompt_logprobs`) within the vocabulary.
pub(crate) fn logprob_count(count: u32, vocab: usize) -> usize {
    (count as usize).min(vocab)
}

/// Words of `rows` logprob rows of `count` top tokens each: the selected
/// token, score and rank, then the top tokens, scores and ranks.
pub(crate) fn logprob_words(rows: usize, count: usize) -> usize {
    rows * (3 + 3 * count)
}

fn parameters(params: &SamplingParams, vocab: usize) -> [f32; PARAMETER_COLUMNS] {
    // The top-k width inside the vocabulary, or zero without one; an
    // integer far below 2^24, which f32 holds exactly.
    let top_k = top_k_width(params, vocab);
    [
        params.temperature,
        params.top_p,
        params.min_p,
        params.typical_p,
        params.repetition_penalty,
        params.frequency_penalty,
        params.presence_penalty,
        if top_k < vocab { top_k as f32 } else { 0.0 },
    ]
}

/// Host-only checks; no device value is read.
fn validate(task: &SampleTask<'_>, vocab: usize) -> Result<()> {
    let rows = task.rows();
    if rows == 0 || rows != task.controls.draft_token_ids.len() + 1 {
        return Err(Error::Invalid(
            "sampling rows must cover the draft chain and bonus token".into(),
        ));
    }
    if task.controls.device_greedy(task.params) {
        if task.draws.is_some() || rows != 1 {
            return Err(Error::Invalid(
                "greedy sampling task has shaped metadata".into(),
            ));
        }
    } else if task.draws.as_ref().is_none_or(|draws| draws.len() != rows) {
        return Err(Error::Invalid(
            "sampling task draws must align with its rows".into(),
        ));
    }
    if task
        .controls
        .draft_token_ids
        .iter()
        .any(|&token| token as usize >= vocab)
    {
        return Err(Error::Invalid(
            "speculative draft token is outside the model vocabulary".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SamplingParams {
        SamplingParams::default()
    }

    fn task<'a>(params: &'a SamplingParams, row: usize, draws: Option<Vec<f32>>) -> SampleTask<'a> {
        SampleTask {
            params,
            controls: SamplingControls {
                allowed: vec![None],
                ..SamplingControls::default()
            },
            draws,
            slot: 3,
            row,
            gated: true,
        }
    }

    fn bits(set: &[i32], vocab: usize) -> Vec<usize> {
        (0..vocab)
            .filter(|&token| set[token / 32] & (1 << (token % 32)) != 0)
            .collect()
    }

    /// Tasks keep their input order, each with its sampler, slot, gate and
    /// finish set; reporting tasks pack their logprobs into a padded shape.
    #[test]
    fn tasks_keep_their_order_and_report_logprobs_in_a_padded_shape() -> Result<()> {
        let greedy = params();
        let mut stochastic = params();
        stochastic.temperature = 0.7;
        stochastic.n_logprobs = 3;
        let mut first = task(&stochastic, 0, Some(vec![0.25]));
        first.controls.finish_token_ids = vec![2, 40, 70];
        let tasks = [
            first,
            task(&greedy, 1, None),
            task(&stochastic, 2, Some(vec![0.5])),
        ];
        let metadata = SamplingMetadata::build(&tasks, 50, 8, 1000)?;
        assert_eq!((metadata.tasks, metadata.width), (3, 1));
        assert_eq!(metadata.paths, vec![CATEGORICAL, GREEDY, CATEGORICAL]);
        assert_eq!(
            (metadata.top_k, metadata.categorical, metadata.sorts),
            (false, true, false)
        );
        assert_eq!(metadata.rows, vec![0, 1, 2]);
        assert_eq!(metadata.draws, vec![0.25, 0.0, 0.5]);
        assert_eq!(metadata.params.len(), 3 * PARAMETER_COLUMNS);
        assert_eq!(metadata.slots, vec![3, 3, 3]);
        assert_eq!(metadata.mask_rows, vec![-1, -1, -1]);
        // Token 70 is outside the 50-token vocabulary.
        let words = bitmask_words(50);
        assert_eq!(bits(&metadata.finish[..words], 50), vec![2, 40]);
        assert!(metadata.finish[words..].iter().all(|&word| word == 0));
        assert_eq!(metadata.position(2), 2);

        // Two reporting tasks of three top tokens, packed compactly and
        // scored at a top width of four over a bucket of two tasks.
        assert_eq!(metadata.logprob_tasks, vec![0, 2]);
        assert_eq!(metadata.logprob_row(2), Some(1));
        assert_eq!(metadata.logprob_row(1), None);
        let layout = metadata
            .logprobs
            .as_ref()
            .expect("two tasks report logprobs");
        assert_eq!(layout.counts, vec![3, 3]);
        assert_eq!(metadata.logprob_bucket, (2, 4));
        assert_eq!(metadata.logprob_end(), 2 * (3 + 3 * 3));

        // The packed words bound the binding; a task without draws for
        // its sampler is refused.
        assert!(SamplingMetadata::build(&tasks, 50, 8, 23).is_err());
        let invalid = [task(&stochastic, 0, None)];
        assert!(SamplingMetadata::build(&invalid, 50, 8, 1000).is_err());
        Ok(())
    }

    /// A request's rows share one token set: its allowed tokens less the
    /// suppressed ones, or every token less the suppressed ones.
    #[test]
    fn rows_of_one_request_share_their_token_set() -> Result<()> {
        let mut stochastic = params();
        stochastic.temperature = 1.0;
        let mut allowed = task(&stochastic, 0, Some(vec![0.5]));
        allowed.controls.allowed = vec![Some(vec![1, 5, 9])];
        allowed.controls.suppress = vec![5];
        let mut suppressed = task(&stochastic, 1, Some(vec![0.5]));
        suppressed.controls.suppress = vec![0, 39];
        let tasks = [allowed, suppressed];
        let metadata = SamplingMetadata::build(&tasks, 40, 8, 1000)?;
        let words = bitmask_words(40);
        assert_eq!(metadata.mask_rows, vec![0, 1]);
        assert_eq!(bits(&metadata.masks[..words], 40), vec![1, 9]);
        let every = bits(&metadata.masks[words..], 40);
        assert_eq!(every, (1..39).collect::<Vec<_>>());
        // Bits past the vocabulary stay clear.
        assert_eq!(metadata.masks[2 * words - 1] as u32 >> 8, 0);

        // A verification forcing a token at each position has a set per
        // row; more sets than the binding's rows are refused.
        let mut forced = task(&stochastic, 0, Some(vec![0.5; 3]));
        forced.controls.allowed = vec![Some(vec![1]), Some(vec![2]), Some(vec![3])];
        forced.controls.draft_token_ids = vec![1, 2];
        let metadata = SamplingMetadata::build(std::slice::from_ref(&forced), 40, 3, 1000)?;
        assert_eq!(metadata.mask_rows, vec![0, 1, 2]);
        assert!(SamplingMetadata::build(&[forced], 40, 2, 1000).is_err());
        Ok(())
    }

    /// A verification's tasks hold its widest task's rows: a task's own
    /// rows, then copies of its first, and its drafts padded with -1.
    #[test]
    fn verification_tasks_pad_to_the_widest_task() -> Result<()> {
        let mut stochastic = params();
        stochastic.temperature = 1.0;
        let mut wide = task(&stochastic, 0, Some(vec![0.1, 0.2, 0.3]));
        wide.controls.allowed = vec![None; 3];
        wide.controls.draft_token_ids = vec![7, 8];
        wide.controls.finish_token_ids = vec![8];
        wide.controls.terminal_draft_prefix = Some(2);
        let mut narrow = task(&stochastic, 3, Some(vec![0.4, 0.5]));
        narrow.controls.allowed = vec![None; 2];
        narrow.controls.draft_token_ids = vec![6];
        let metadata = SamplingMetadata::build(&[wide, narrow], 20, 8, 1000)?;
        assert_eq!((metadata.tasks, metadata.width), (2, 3));
        assert_eq!(metadata.rows, vec![0, 1, 2, 3, 4, 3]);
        assert_eq!(metadata.draws, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.4]);
        assert_eq!(metadata.drafts, vec![7, 8, 6, -1]);
        assert_eq!(metadata.terminal, vec![2, 0]);
        Ok(())
    }

    #[test]
    fn a_top_bucket_holds_its_least_top_through_itself() {
        // Over 100 tokens: 0, 1, 2, 4, ..., 64, then 100.
        for top in 0..=100 {
            let bucket = top_bucket(top, 100);
            assert!(least_top(bucket) <= top && top <= bucket);
        }
        assert_eq!(least_top(100), 65);
        // A vocabulary of three tokens makes three a bucket of itself.
        assert_eq!(least_top(top_bucket(3, 3)), 3);
    }

    #[test]
    fn logprob_shapes_double_at_most() {
        assert_eq!(logprob_shape(3, 5, 100, 40), (4, 8));
        assert_eq!(logprob_shape(1, 0, 100, 40), (1, 0));
        assert_eq!(logprob_shape(5, 90, 100, 40), (8, 100));
        // A bucket never exceeds the binding's rows.
        assert_eq!(logprob_shape(33, 1, 100, 40), (40, 1));
        for rows in 1..=40 {
            for top in 0..=120 {
                let (stride, width) = logprob_shape(rows, top, 100, 40);
                assert!(stride >= rows && stride <= 40 && width >= top.min(100));
                assert!(logprob_words(stride, width) <= 4 * logprob_words(rows, top.min(100)));
            }
        }
    }
}
