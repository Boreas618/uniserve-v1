//! The reference continuation a generation is compared against.
//!
//! A reference file records the ids a reference stack generated from
//! a fixed prompt and the margin of its choice at every step: the gap between the best logit and
//! the runner-up, in logit units. The margin is what makes a difference readable - a margin of
//! zero says the reference's own choice was a tie and a stack that picks the other token there is
//! not evidence of a defect, while a margin of seventeen says the two stacks disagree about a
//! token neither could have confused, which is a defect in one of them.
//!
//! This module is the comparison both a slice run and a runner-level harness use, so that the
//! expected values have one reader rather than one per harness.


use serde::Deserialize;

/// One reference generation: the ids it produced, and the margin of each choice.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct Reference {
    /// The ids the reference generated, in order, after the prompt.
    pub ids: Vec<u32>,
    /// The margin of the reference's choice at each step, in logit units.
    ///
    /// It may be shorter than `ids` when a file records fewer margins; a step past its end has no
    /// margin and is read as unknown rather than as a tie.
    #[serde(default)]
    pub margins: Vec<f64>,
}

/// One step where a generation and the reference disagree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Difference {
    /// The step, counted over the generated ids: `0` is the first token after the prompt.
    pub step: usize,
    /// The id this stack chose.
    pub ours: u32,
    /// The id the reference chose.
    pub theirs: u32,
    /// The reference's margin at that step, when the file records one.
    pub their_margin: Option<f64>,
}

impl Reference {

    /// Reads a reference from its JSON text.
    pub fn parse(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|error| format!("a reference file: {error}"))
    }

    /// Every step where `ours` and the reference disagree, in order.
    ///
    /// Only steps both sides have are compared: a generation that ended early is not a difference
    /// at the step it did not reach, and a caller that needs that fact compares lengths itself.
    pub fn differences(&self, ours: &[u32]) -> Vec<Difference> {
        ours.iter()
            .zip(self.ids.iter())
            .enumerate()
            .filter(|(_, (ours, theirs))| ours != theirs)
            .map(|(step, (ours, theirs))| Difference {
                step,
                ours: *ours,
                theirs: *theirs,
                their_margin: self.margins.get(step).copied(),
            })
            .collect()
    }

    /// The first step where `ours` and the reference disagree.
    pub fn first(&self, ours: &[u32]) -> Option<Difference> {
        self.differences(ours).into_iter().next()
    }

    /// The widest reference margin among the steps that disagree.
    ///
    /// This is the number that says how much a disagreement is worth: a run whose every difference
    /// sits on a tie is a run whose numerics are as good as the reference's within the reference's
    /// own resolution.
    pub fn widest(&self, ours: &[u32]) -> Option<f64> {
        self.differences(ours)
            .into_iter()
            .filter_map(|difference| difference.their_margin)
            .fold(None, |widest: Option<f64>, margin| {
                Some(widest.map_or(margin, |widest| widest.max(margin)))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str =
        r#"{"ids": [151667, 198, 279], "margins": [17.125, 0.0, 2.25], "note": "x"}"#;

    #[test]
    fn a_reference_reads_its_ids_and_margins_and_ignores_the_rest() {
        let reference = Reference::parse(FILE).expect("a reference");
        assert_eq!(reference.ids, vec![151667, 198, 279]);
        assert_eq!(reference.margins, vec![17.125, 0.0, 2.25]);
    }

    #[test]
    fn a_matching_generation_has_no_difference() {
        let reference = Reference::parse(FILE).expect("a reference");
        assert!(reference.differences(&[151667, 198, 279]).is_empty());
    }

    #[test]
    fn a_difference_carries_the_references_margin_at_that_step() {
        let reference = Reference::parse(FILE).expect("a reference");
        let differences = reference.differences(&[151667, 1414, 279]);
        assert_eq!(
            differences,
            vec![Difference {
                step: 1,
                ours: 1414,
                theirs: 198,
                their_margin: Some(0.0),
            }]
        );
        assert_eq!(reference.widest(&[151667, 1414, 279]), Some(0.0));
    }

    #[test]
    fn a_step_past_the_recorded_margins_has_none_rather_than_a_tie() {
        let reference = Reference::parse(r#"{"ids": [1, 2]}"#).expect("a reference");
        assert_eq!(reference.first(&[1, 9]).and_then(|d| d.their_margin), None);
    }

    #[test]
    fn a_generation_that_ended_early_is_not_a_difference_at_the_step_it_did_not_reach() {
        let reference = Reference::parse(FILE).expect("a reference");
        assert!(reference.differences(&[151667]).is_empty());
    }
}
