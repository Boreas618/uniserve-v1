//! The epoch's phases, as the snapshot states them.
//!
//! A policy reads a request's next phase as a [`PhaseId`]; what that phase
//! *is* — a prefill chunk, a decode step, a denoising step — and so how it is
//! ordered, charged and clipped, comes from [`PhaseView`], which the
//! mechanism states once per epoch in the snapshot (design §4.4: the phase is
//! the unit the scheduler addresses, design §0).

use scheduling::{Phase, PhaseId, PhaseView};

/// The epoch's phases, so a policy can classify a request's next phase
/// without asking the mechanism again.
#[derive(Clone, Debug, Default)]
pub(crate) struct Phases {
    /// The kind of each phase, by id; `None` for an id the snapshot names no phase by.
    kinds: Vec<Option<Phase>>,
}

impl Phases {
    /// The table a snapshot states.
    pub(crate) fn new(views: &[PhaseView]) -> Self {
        let mut kinds = Vec::new();
        for view in views {
            let at = view.id.0 as usize;
            if kinds.len() <= at {
                kinds.resize(at + 1, None);
            }
            kinds[at] = Some(view.kind);
        }
        Self { kinds }
    }

    /// The kind of a phase; `None` for one the snapshot does not state. The snapshot states
    /// every phase of the epoch, and a request's view names a phase of its epoch, so a request
    /// whose phase is not here names no call this policy can schedule.
    pub(crate) fn kind(&self, phase: PhaseId) -> Option<Phase> {
        self.kinds.get(phase.0 as usize).copied().flatten()
    }
}

#[cfg(test)]
mod tests {
    use scheduling::{PhaseClass, StageId};

    use super::*;

    #[test]
    fn a_phase_reads_as_the_snapshot_states_it() {
        let views = [
            PhaseView {
                id: PhaseId(0),
                kind: Phase::Prefill,
                block: StageId(0),
            },
            PhaseView {
                id: PhaseId(2),
                kind: Phase::Decode,
                block: StageId(0),
            },
        ];
        let phases = Phases::new(&views);
        assert_eq!(phases.kind(PhaseId(0)), Some(Phase::Prefill));
        assert_eq!(
            phases.kind(PhaseId(2)).map(Phase::class),
            Some(PhaseClass::Token)
        );
        assert_eq!(phases.kind(PhaseId(1)), None);
        assert_eq!(phases.kind(PhaseId(9)), None);
    }
}
