//! One forward's sampling tasks, grouped by sampler.
//!
//! [`SamplingPlan::build`] validates every task on the host, groups tasks by
//! [`SamplingPath`] in first-appearance order and produces:
//!
//! - one [`SamplingGroup`] per sampler: what the runner's sampling method
//!   of that sampler takes besides the tensors, as flat integer lists;
//! - `draws` and `params`: per candidate row in group order, written into
//!   the runner's input tensors, with each task's slot and gate;
//! - each group's logprob layout, so readback decodes the packed words.
//!
//! Tasks are reordered into groups; [`SamplingPlan::position`] maps an input
//! task to its output position (its column in the selection outputs).
//!
//! A token set crosses in the form the sampler indexes with. A per-task set
//! (finish and transition tokens, reported logprob ids) is a row-major
//! matrix of at least one column, padded with -1. A set applied to logits
//! (allowed, suppressed and biased tokens, draft tokens counted by
//! penalties) is a list of scatter indices `row * vocab + token` over the
//! group's candidate rows, with tokens outside the vocabulary dropped and,
//! for masks, repeated tokens dropped.

use indexmap::IndexMap;
use uniserve_core::SamplingParams;

use super::{SamplingMetadata, SamplingPath, TOKEN_VALUE_MASK};
use crate::storage::LogprobLayout;
use crate::{Error, Result};

/// Parameters per candidate row: temperature, top-p, min-p, typical-p,
/// repetition, frequency and presence penalties, and one spare column.
pub(crate) const PARAMETER_COLUMNS: usize = 8;

/// One sampled call of an invocation.
pub(crate) struct SampleTask<'a> {
    pub(crate) params: &'a SamplingParams,
    pub(crate) metadata: SamplingMetadata,
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
        self.metadata.allowed.len()
    }
}

/// A group's logprob words: offset in the binding's logprob column, count,
/// and their layout.
pub(crate) struct GroupLogprobs {
    pub(crate) offset: usize,
    pub(crate) words: usize,
    pub(crate) layout: LogprobLayout,
}

/// The tasks of one sampler: the host lists of the runner's sampling
/// method. Rows are the group's candidate rows, task by task; a task has
/// one row, or one per draft token plus one in a verification.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SamplingGroup {
    pub(crate) path: SamplingPath,
    /// Output position of the group's first task; the others follow.
    pub(crate) first: usize,
    /// Each task's first candidate row in the forward's logits.
    pub(crate) rows: Vec<i64>,
    /// Each task's candidate rows.
    pub(crate) row_counts: Vec<i64>,
    /// The group's first row of `draws` and `params`.
    pub(crate) parameters: usize,
    /// Per task, whether it finishes whatever it selects.
    pub(crate) force_finish: Vec<i64>,
    /// `[tasks, width]` finish tokens.
    pub(crate) finish_ids: Vec<i64>,
    /// `[tasks, width]` tokens selecting a transition; an empty row for a
    /// task without a transition output.
    pub(crate) transition_ids: Vec<i64>,
    /// Per row, its top-k width inside the vocabulary, or zero.
    pub(crate) top_ks: Vec<i64>,
    /// Rows with a nucleus cutoff below one and no top-k width.
    pub(crate) top_p_rows: Vec<i64>,
    /// Rows with a minimum-probability cutoff.
    pub(crate) min_p_rows: Vec<i64>,
    /// Rows with a typical-probability cutoff below one.
    pub(crate) typical_p_rows: Vec<i64>,
    /// Rows restricted to their `allowed` scatter indices.
    pub(crate) allowed_rows: Vec<i64>,
    pub(crate) allowed: Vec<i64>,
    /// Scatter indices of tokens masked from selection.
    pub(crate) suppressed: Vec<i64>,
    /// Rows with penalties, and each one's request slot.
    pub(crate) penalty_rows: Vec<i64>,
    pub(crate) penalty_slots: Vec<i64>,
    /// Scatter indices `penalized row * vocab + token`: the draft tokens
    /// before each penalized row.
    pub(crate) penalty_drafts: Vec<i64>,
    /// Scatter indices of the additive offsets `bias_values`.
    pub(crate) bias: Vec<i64>,
    pub(crate) bias_values: Vec<f64>,
    /// The tasks' draft tokens, concatenated.
    pub(crate) drafts: Vec<i64>,
    /// Per task, the one-based length of its draft prefix ending at the
    /// first finish token, or zero.
    pub(crate) terminal_prefixes: Vec<i64>,
    /// The tasks reporting logprobs, each one's top count and its row of
    /// the `[tasks, width]` explicit token ids.
    pub(crate) logprob_tasks: Vec<i64>,
    pub(crate) logprob_counts: Vec<i64>,
    pub(crate) logprob_ids: Vec<i64>,
    /// Where the group's logprob words start in the logprob tensor.
    pub(crate) logprob_offset: usize,
}

/// Row-major `[sets.len(), width]` matrix of token sets padded with -1,
/// with at least one column.
pub(crate) fn padded(sets: &[&[u32]]) -> Vec<i64> {
    let width = sets.iter().map(|set| set.len()).max().unwrap_or(0).max(1);
    let mut matrix = vec![-1; sets.len() * width];
    for (row, set) in sets.iter().enumerate() {
        for (column, &token) in set.iter().enumerate() {
            matrix[row * width + column] = i64::from(token);
        }
    }
    matrix
}

/// Scatter indices of `tokens` in `row`, without tokens outside the
/// vocabulary or repeated.
fn scatter(row: usize, tokens: &[u32], vocab: usize) -> Vec<i64> {
    let mut seen = Vec::new();
    for &token in tokens {
        if (token as usize) < vocab && !seen.contains(&token) {
            seen.push(token);
        }
    }
    seen.into_iter()
        .map(|token| (row * vocab + token as usize) as i64)
        .collect()
}

impl SamplingGroup {
    /// An empty group of `path` whose first task has output position
    /// `first` and whose rows start at row `parameters` of the draws.
    fn new(path: SamplingPath, first: usize, parameters: usize) -> Self {
        Self {
            path,
            first,
            rows: Vec::new(),
            row_counts: Vec::new(),
            parameters,
            force_finish: Vec::new(),
            finish_ids: Vec::new(),
            transition_ids: Vec::new(),
            top_ks: Vec::new(),
            top_p_rows: Vec::new(),
            min_p_rows: Vec::new(),
            typical_p_rows: Vec::new(),
            allowed_rows: Vec::new(),
            allowed: Vec::new(),
            suppressed: Vec::new(),
            penalty_rows: Vec::new(),
            penalty_slots: Vec::new(),
            penalty_drafts: Vec::new(),
            bias: Vec::new(),
            bias_values: Vec::new(),
            drafts: Vec::new(),
            terminal_prefixes: Vec::new(),
            logprob_tasks: Vec::new(),
            logprob_counts: Vec::new(),
            logprob_ids: Vec::new(),
            logprob_offset: 0,
        }
    }

    /// Add `task`, whose candidate rows follow the group's so far.
    fn push(&mut self, task: &SampleTask<'_>, vocab: usize) {
        let params = task.params;
        let metadata = &task.metadata;
        let index = self.rows.len() as i64;
        let first_row = self.top_ks.len();
        let rows = task.rows();
        self.rows.push(task.row as i64);
        self.row_counts.push(rows as i64);
        self.force_finish.push(i64::from(metadata.force_finish));
        self.drafts
            .extend(metadata.draft_token_ids.iter().map(|&id| i64::from(id)));
        self.terminal_prefixes
            .push(metadata.terminal_draft_prefix.unwrap_or(0) as i64);
        if params.generated_logprobs_requested() {
            self.logprob_tasks.push(index);
        }

        let top_k = params.top_k as usize;
        let top_k = if top_k < vocab { top_k } else { 0 };
        let penalized = params.uses_penalties();
        for (offset, allowed) in metadata.allowed.iter().enumerate() {
            let row = first_row + offset;
            self.top_ks.push(top_k as i64);
            if top_k == 0 && params.top_p > 0.0 && params.top_p < 1.0 {
                self.top_p_rows.push(row as i64);
            }
            if params.min_p > 0.0 {
                self.min_p_rows.push(row as i64);
            }
            if params.typical_p < 1.0 {
                self.typical_p_rows.push(row as i64);
            }
            if let Some(tokens) = allowed {
                self.allowed_rows.push(row as i64);
                self.allowed.extend(scatter(row, tokens, vocab));
            }
            self.suppressed
                .extend(scatter(row, &metadata.suppress, vocab));
            if penalized {
                // A row's counts include the draft tokens before it.
                let counted = self.penalty_rows.len();
                self.penalty_rows.push(row as i64);
                self.penalty_slots.push(i64::from(task.slot));
                self.penalty_drafts.extend(
                    metadata.draft_token_ids[..offset]
                        .iter()
                        .map(|&token| (counted * vocab + token as usize) as i64),
                );
            }
            for &(token, bias) in &params.logit_bias {
                if (token as usize) < vocab {
                    self.bias.push((row * vocab + token as usize) as i64);
                    self.bias_values.push(f64::from(bias));
                }
            }
        }
    }
}

/// The sampling inputs of one forward.
pub(crate) struct SamplingPlan {
    pub(crate) groups: Vec<SamplingGroup>,
    pub(crate) draws: Vec<f32>,
    pub(crate) params: Vec<f32>,
    pub(crate) slots: Vec<i64>,
    pub(crate) gates: Vec<i64>,
    /// Output position of each input task.
    positions: Vec<usize>,
    /// The group and the group-local position of each input task.
    members: Vec<(usize, usize)>,
    pub(crate) logprobs: Vec<Option<GroupLogprobs>>,
}

impl SamplingPlan {
    /// Validate and group `tasks`. `vocab` is the global vocabulary,
    /// `logprob_base` the first free word of the logprob tensor and
    /// `logprob_capacity` its size.
    pub(crate) fn build(
        tasks: &[SampleTask<'_>],
        vocab: usize,
        cuda: bool,
        logprob_base: usize,
        logprob_capacity: usize,
    ) -> Result<Self> {
        if vocab > TOKEN_VALUE_MASK as usize {
            return Err(Error::Unsupported(
                "vocabulary exceeds the device token decision range".into(),
            ));
        }
        let mut paths: IndexMap<SamplingPath, Vec<usize>> = IndexMap::new();
        for (index, task) in tasks.iter().enumerate() {
            validate(task, vocab)?;
            paths
                .entry(task.metadata.path(task.params, vocab, cuda))
                .or_default()
                .push(index);
        }

        let mut groups = Vec::with_capacity(paths.len());
        let mut draws = Vec::new();
        let mut params = Vec::new();
        let mut slots = Vec::with_capacity(tasks.len());
        let mut gates = Vec::with_capacity(tasks.len());
        let mut positions = vec![0; tasks.len()];
        let mut members = vec![(0, 0); tasks.len()];
        let mut logprobs = Vec::with_capacity(paths.len());
        let mut logprob_offset = logprob_base;
        let mut first = 0;

        for (number, (path, indices)) in paths.iter().enumerate() {
            let mut group = SamplingGroup::new(*path, first, draws.len());
            let mut layout = LogprobLayout {
                rows: Vec::new(),
                counts: Vec::new(),
                requested_ids: Vec::new(),
                max_count: 0,
                max_requested: 0,
            };
            let mut finish: Vec<&[u32]> = Vec::with_capacity(indices.len());
            let mut transitions: Vec<&[u32]> = Vec::with_capacity(indices.len());
            for (local, &index) in indices.iter().enumerate() {
                let task = &tasks[index];
                positions[index] = first + local;
                members[index] = (number, local);
                slots.push(i64::from(task.slot));
                gates.push(i64::from(task.gated));
                group.push(task, vocab);
                finish.push(task.metadata.finish_token_ids.as_slice());
                transitions.push(if task.metadata.return_transition {
                    task.metadata.transition_token_ids.as_slice()
                } else {
                    &[]
                });
                let rows = task.rows();
                match &task.draws {
                    Some(values) => draws.extend_from_slice(values),
                    None => draws.extend(std::iter::repeat_n(0.0, rows)),
                }
                for _ in 0..rows {
                    params.extend_from_slice(&parameters(task.params));
                }
                if task.params.generated_logprobs_requested() {
                    let (count, ids) = logprob_request(task.params, vocab);
                    layout.rows.push(local);
                    layout.max_count = layout.max_count.max(count);
                    layout.max_requested = layout.max_requested.max(ids.len());
                    layout.counts.push(count);
                    layout.requested_ids.push(ids);
                }
            }
            group.finish_ids = padded(&finish);
            group.transition_ids = padded(&transitions);
            group.logprob_counts = layout.counts.iter().map(|&count| count as i64).collect();
            let requested: Vec<&[u32]> = layout.requested_ids.iter().map(Vec::as_slice).collect();
            group.logprob_ids = padded(&requested);

            let size = layout.rows.len() * (3 + 3 * layout.max_count + 2 * layout.max_requested);
            let entry = if size > 0 {
                if logprob_offset + size > logprob_capacity {
                    return Err(Error::Resource(
                        "logprob words exceed the binding's capacity",
                    ));
                }
                group.logprob_offset = logprob_offset;
                let entry = GroupLogprobs {
                    offset: logprob_offset,
                    words: size,
                    layout,
                };
                logprob_offset += size;
                Some(entry)
            } else {
                None
            };
            logprobs.push(entry);
            first += indices.len();
            groups.push(group);
        }
        Ok(Self {
            groups,
            draws,
            params,
            slots,
            gates,
            positions,
            members,
            logprobs,
        })
    }

    /// The output position of input task `index`.
    pub(crate) fn position(&self, index: usize) -> usize {
        self.positions[index]
    }

    /// Input task `index`'s logprob span `(offset, words, group-local row)`
    /// when it requested logprobs.
    pub(crate) fn logprob_span(&self, index: usize) -> Option<(usize, usize, usize)> {
        let (group, local) = self.members[index];
        let entry = self.logprobs[group].as_ref()?;
        entry
            .layout
            .rows
            .contains(&local)
            .then_some((entry.offset, entry.words, local))
    }

    /// Words of the logprob column this plan writes after its base.
    pub(crate) fn logprob_end(&self) -> Option<usize> {
        self.logprobs
            .iter()
            .flatten()
            .map(|entry| entry.offset + entry.words)
            .max()
    }
}

/// Top count and deduplicated in-range ids a call reports per position.
pub(crate) fn logprob_request(params: &SamplingParams, vocab: usize) -> (usize, Vec<u32>) {
    let count = (params.n_logprobs as usize).min(vocab);
    let mut ids = Vec::new();
    for &id in &params.logprob_token_ids {
        if (id as usize) < vocab && !ids.contains(&id) {
            ids.push(id);
        }
    }
    (count, ids)
}

/// Words of `rows` logprob rows of `count` top and `requested` ids each.
pub(crate) fn logprob_words(rows: usize, count: usize, requested: usize) -> usize {
    rows * (3 + 3 * count + 2 * requested)
}

fn parameters(params: &SamplingParams) -> [f32; PARAMETER_COLUMNS] {
    [
        params.temperature,
        params.top_p,
        params.min_p,
        params.typical_p,
        params.repetition_penalty,
        params.frequency_penalty,
        params.presence_penalty,
        0.0,
    ]
}

/// Host-only checks; no device value is read.
fn validate(task: &SampleTask<'_>, vocab: usize) -> Result<()> {
    let rows = task.rows();
    if rows == 0 || rows != task.metadata.draft_token_ids.len() + 1 {
        return Err(Error::Invalid(
            "sampling rows must cover the draft chain and bonus token".into(),
        ));
    }
    if task.metadata.device_greedy(task.params) {
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
        .metadata
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
            metadata: SamplingMetadata {
                allowed: vec![None],
                ..SamplingMetadata::default()
            },
            draws,
            slot: 3,
            row,
            gated: true,
        }
    }

    #[test]
    fn groups_keep_first_appearance_order_and_map_task_positions() -> Result<()> {
        let greedy = params();
        let mut stochastic = params();
        stochastic.temperature = 0.7;
        stochastic.n_logprobs = 2;
        stochastic.logprob_token_ids = vec![5, 5, 900];
        let tasks = [
            task(&stochastic, 0, Some(vec![0.25])),
            task(&greedy, 1, None),
            task(&stochastic, 2, Some(vec![0.5])),
        ];
        let plan = SamplingPlan::build(&tasks, 100, false, 10, 1000)?;
        // Categorical tasks first (first appearance), then greedy.
        assert_eq!(
            (0..3).map(|index| plan.position(index)).collect::<Vec<_>>(),
            vec![0, 2, 1]
        );
        // Categorical group: tasks 0 and 2 at output positions 0 and 1,
        // parameter rows from 0, logprobs at 10 for both tasks with the
        // explicit ids deduplicated and kept inside the vocabulary.
        let categorical = &plan.groups[0];
        assert_eq!(categorical.path, SamplingPath::Categorical);
        assert_eq!((categorical.first, categorical.parameters), (0, 0));
        assert_eq!(categorical.rows, vec![0, 2]);
        assert_eq!(categorical.logprob_tasks, vec![0, 1]);
        assert_eq!(categorical.logprob_counts, vec![2, 2]);
        assert_eq!(categorical.logprob_ids, vec![5, 5]);
        assert_eq!(categorical.logprob_offset, 10);
        // Greedy group: task 1 at output position 2, parameter row 2, with
        // one padded column of finish and transition tokens.
        let greedy_group = &plan.groups[1];
        assert_eq!(
            greedy_group.path,
            SamplingPath::Greedy { suppressed: false }
        );
        assert_eq!((greedy_group.first, greedy_group.parameters), (2, 2));
        assert_eq!(greedy_group.rows, vec![1]);
        assert_eq!(greedy_group.finish_ids, vec![-1]);
        assert_eq!(greedy_group.transition_ids, vec![-1]);
        assert_eq!(plan.draws, vec![0.25, 0.5, 0.0]);
        assert_eq!(plan.params.len(), 3 * PARAMETER_COLUMNS);
        assert_eq!(plan.logprob_span(2), Some((10, 22, 1)));
        assert_eq!(plan.logprob_span(1), None);
        assert_eq!(plan.logprob_end(), Some(32));
        assert_eq!(plan.slots, vec![3, 3, 3]);

        assert!(SamplingPlan::build(&tasks, 100, false, 990, 1000).is_err());
        let invalid = [task(&stochastic, 0, None)];
        assert!(SamplingPlan::build(&invalid, 100, false, 0, 1000).is_err());
        Ok(())
    }
}
