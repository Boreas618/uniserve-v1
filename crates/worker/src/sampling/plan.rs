//! One invocation's sampling tasks, grouped by selector and encoded for the
//! text step.
//!
//! [`SamplingPlan::build`] validates every task on the host, groups tasks by
//! [`SamplingPath`] in first-appearance order and produces:
//!
//! - `words`: the host descriptor the step decodes
//!   (`uniserve_models.interface.text.decode_tasks`): `[task count, group
//!   count]`, seven words per group (`path, top_k, first task, task count,
//!   logprob offset, logprob words, parameter row`), then one record per
//!   task in group order;
//! - `draws` and `params`: per candidate row in group order, staged into the
//!   binding's device columns;
//! - each group's logprob layout, so readback decodes the packed words.
//!
//! Tasks are reordered into groups; [`SamplingPlan::position`] maps an input
//! task to its output position (its column in the selection outputs).

use indexmap::IndexMap;
use uniserve_core::SamplingParams;

use super::{SamplingMetadata, SamplingPath, TOKEN_VALUE_MASK};
use crate::storage::LogprobLayout;
use crate::{Error, Result};

/// Parameters per candidate row: temperature, top-p, min-p, typical-p,
/// repetition, frequency and presence penalties, and one spare column.
pub(crate) const PARAMETER_COLUMNS: usize = 8;

const FORCE_FINISH: i64 = 1 << 0;
const RETURN_TRANSITION: i64 = 1 << 1;
const GATED: i64 = 1 << 2;
const PENALTIES: i64 = 1 << 3;
const LOGPROBS: i64 = 1 << 4;
const TOP_P: i64 = 1 << 5;
const MIN_P: i64 = 1 << 6;
const TYPICAL_P: i64 = 1 << 7;

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

/// The encoded sampling inputs of one invocation.
pub(crate) struct SamplingPlan {
    pub(crate) words: Vec<i64>,
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
    /// Validate and encode `tasks`. `vocab` is the global vocabulary,
    /// `logprob_base` the first free word of the logprob column and
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
        let mut groups: IndexMap<SamplingPath, Vec<usize>> = IndexMap::new();
        for (index, task) in tasks.iter().enumerate() {
            validate(task, vocab)?;
            groups
                .entry(task.metadata.path(task.params, vocab, cuda))
                .or_default()
                .push(index);
        }

        let mut words = vec![tasks.len() as i64, groups.len() as i64];
        let mut records = Vec::new();
        let mut draws = Vec::new();
        let mut params = Vec::new();
        let mut slots = Vec::with_capacity(tasks.len());
        let mut gates = Vec::with_capacity(tasks.len());
        let mut positions = vec![0; tasks.len()];
        let mut members = vec![(0, 0); tasks.len()];
        let mut logprobs = Vec::with_capacity(groups.len());
        let mut logprob_offset = logprob_base;
        let mut first = 0;

        for (group, (path, indices)) in groups.iter().enumerate() {
            let parameter_row = draws.len();
            let mut layout = LogprobLayout {
                rows: Vec::new(),
                counts: Vec::new(),
                requested_ids: Vec::new(),
                max_count: 0,
                max_requested: 0,
            };
            for (local, &index) in indices.iter().enumerate() {
                let task = &tasks[index];
                positions[index] = first + local;
                members[index] = (group, local);
                slots.push(i64::from(task.slot));
                gates.push(i64::from(task.gated));
                encode(task, vocab, &mut records);
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
            let size = layout.rows.len() * (3 + 3 * layout.max_count + 2 * layout.max_requested);
            let (top_k, path_code) = match *path {
                SamplingPath::Greedy { suppressed: false } => (0, 0),
                SamplingPath::Greedy { suppressed: true } => (0, 1),
                SamplingPath::TopK(width) => (i64::from(width), 2),
                SamplingPath::Categorical => (0, 3),
            };
            let entry = if size > 0 {
                if logprob_offset + size > logprob_capacity {
                    return Err(Error::Resource(
                        "logprob words exceed the binding's capacity",
                    ));
                }
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
            words.extend_from_slice(&[
                path_code,
                top_k,
                first as i64,
                indices.len() as i64,
                entry.as_ref().map_or(-1, |entry| entry.offset as i64),
                entry.as_ref().map_or(0, |entry| entry.words as i64),
                parameter_row as i64,
            ]);
            logprobs.push(entry);
            first += indices.len();
        }
        words.extend(records);
        Ok(Self {
            words,
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

/// The descriptor of tasks that only finish a decode graph's greedy
/// selections: every task in row order, no groups.
pub(crate) fn greedy_finish_words(tasks: &[SampleTask<'_>], vocab: usize) -> Vec<i64> {
    let mut words = vec![tasks.len() as i64, 0];
    for task in tasks {
        encode(task, vocab, &mut words);
    }
    words
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

fn encode(task: &SampleTask<'_>, vocab: usize, words: &mut Vec<i64>) {
    let params = task.params;
    let metadata = &task.metadata;
    let mut flags = 0;
    for (set, bit) in [
        (metadata.force_finish, FORCE_FINISH),
        (metadata.return_transition, RETURN_TRANSITION),
        (task.gated, GATED),
        (params.uses_penalties(), PENALTIES),
        (params.generated_logprobs_requested(), LOGPROBS),
        (params.top_p > 0.0 && params.top_p < 1.0, TOP_P),
        (params.min_p > 0.0, MIN_P),
        (params.typical_p < 1.0, TYPICAL_P),
    ] {
        if set {
            flags |= bit;
        }
    }
    let (count, ids) = logprob_request(params, vocab);
    let allowed = metadata.allowed.iter().any(Option::is_some);
    words.extend_from_slice(&[
        task.row as i64,
        task.rows() as i64,
        i64::from(task.slot),
        flags,
        i64::from(params.top_k),
        count as i64,
        metadata.terminal_draft_prefix.unwrap_or(0) as i64,
        metadata.finish_token_ids.len() as i64,
        metadata.transition_token_ids.len() as i64,
        metadata.suppress.len() as i64,
        params.logit_bias.len() as i64,
        ids.len() as i64,
        metadata.draft_token_ids.len() as i64,
        i64::from(allowed),
    ]);
    words.extend(metadata.finish_token_ids.iter().map(|&id| i64::from(id)));
    words.extend(
        metadata
            .transition_token_ids
            .iter()
            .map(|&id| i64::from(id)),
    );
    words.extend(metadata.suppress.iter().map(|&id| i64::from(id)));
    for &(token, bias) in &params.logit_bias {
        words.push(i64::from(token));
        words.push(i64::from(bias.to_bits()));
    }
    words.extend(ids.iter().map(|&id| i64::from(id)));
    words.extend(metadata.draft_token_ids.iter().map(|&id| i64::from(id)));
    if allowed {
        for row in &metadata.allowed {
            match row {
                Some(ids) => {
                    words.push(ids.len() as i64);
                    words.extend(ids.iter().map(|&id| i64::from(id)));
                }
                None => words.push(-1),
            }
        }
    }
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
        assert_eq!(&plan.words[..2], &[3, 2]);
        // Categorical group: path 3, tasks 0..2, logprobs at 10, two rows of
        // (3 + 3 * 2 + 2 * 1) words, parameter rows from 0.
        assert_eq!(&plan.words[2..9], &[3, 0, 0, 2, 10, 22, 0]);
        assert_eq!(&plan.words[9..16], &[0, 0, 2, 1, -1, 0, 2]);
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
