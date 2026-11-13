//! The epoch's phases, in the order the mechanism assigns them.
//!
//! A policy reads a request's next phase as a [`PhaseId`]; what that phase
//! *is* — a prefill chunk, a decode step, a denoising step — and so how it is
//! ordered, charged and clipped, comes from [`PhaseView`], which the
//! mechanism states once per epoch in the snapshot (design §4.4: the phase is
//! the unit the scheduler addresses, design §0).
//!
//! The ids are dense and assigned by the mechanism for an epoch, so this is a
//! table indexed by id.

use crate::ids::PhaseId;
use crate::view::{Phase, PhaseClass, PhaseView};

/// The epoch's phases, so a policy can classify a request's next phase
/// without asking the mechanism again.
#[derive(Clone, Debug, Default)]
pub struct Phases {
    /// The kind of each phase, by id; a churned table responds [`Phase::Other`].
    kinds: Vec<Phase>,
}

impl Phases {
    /// The table a snapshot states.
    pub fn new(views: &[PhaseView]) -> Self {
        let mut phases = Self::default();
        for view in views {
            phases.set(view.id, view.kind);
        }
        phases
    }

    /// Records the kind of one phase, which is how a snapshot states it.
    pub fn set(&mut self, phase: PhaseId, kind: Phase) {
        let at = phase.0 as usize;
        if self.kinds.len() <= at {
            self.kinds.resize(at + 1, Phase::Other);
        }
        self.kinds[at] = kind;
    }

    /// The kind of a phase; [`Phase::Other`] for one the table does not
    /// know, which a policy treats as background work.
    pub fn kind(&self, phase: PhaseId) -> Phase {
        self.kinds
            .get(phase.0 as usize)
            .copied()
            .unwrap_or(Phase::Other)
    }

    /// The assembly class of a phase (design §4.4, today's
    /// `assembly_priority`).
    pub fn class(&self, phase: PhaseId) -> PhaseClass {
        self.kind(phase).class()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::StageId;

    #[test]
    fn a_phase_reads_as_the_snapshot_states_it() {
        let views = [
            PhaseView {
                id: PhaseId(0),
                kind: Phase::Prefill,
                block: StageId(0),
            },
            PhaseView {
                id: PhaseId(1),
                kind: Phase::Decode,
                block: StageId(0),
            },
        ];
        let phases = Phases::new(&views);
        assert_eq!(phases.kind(PhaseId(0)), Phase::Prefill);
        assert_eq!(phases.class(PhaseId(1)), PhaseClass::Token);
        assert!(phases.kind(PhaseId(0)).is_prefill());
    }

    #[test]
    fn an_unknown_phase_is_background_work() {
        let phases = Phases::default();
        assert_eq!(phases.kind(PhaseId(9)), Phase::Other);
        assert_eq!(phases.class(PhaseId(9)), PhaseClass::Background);
    }
}
