//! The scheduler's name for one phase (`specs/worker.md` §4.5).
//!
//! A phase is one block's computation on one kind of call — prefill, decode,
//! one denoising step. The scheduling mechanism addresses the executor's work by
//! it: a batch names the phase it is, and a runner declares the phases it
//! serves. The engine owns the mapping from a model's `(BlockName, PhaseKind)`
//! to a [`PhaseId`]; nothing on the other side of the crossing needs to know a
//! block or a kernel program.
//!
//! The number is opaque on purpose. A scheduling policy runs as wasm and must
//! not link CUDA or `ops`, so the identifier it carries is a plain number; a
//! kernel-program selection key (`ops::PhaseKind`) is a different thing and
//! stays where programs are selected.

use std::fmt;

use serde::{Deserialize, Serialize};

/// One phase of one block, as the scheduler addresses it.
///
/// The engine assigns the numbers when it builds the epoch's runners, and every
/// main process of the epoch gets the same assignment from the same build
/// config. The number is unique within the epoch; it is not a stable identifier
/// across epochs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PhaseId(pub u16);

impl PhaseId {
    /// The phase's number.
    pub const fn get(self) -> u16 {
        self.0
    }

    /// The phase with `number` in the epoch's assignment.
    pub const fn new(number: u16) -> Self {
        Self(number)
    }
}

impl fmt::Display for PhaseId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "phase:{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_phase_id_carries_its_number() {
        let phase = PhaseId::new(3);
        assert_eq!(phase.get(), 3);
        assert_eq!(phase.to_string(), "phase:3");
        assert_eq!(PhaseId(3), phase);
    }

    #[test]
    fn a_phase_id_survives_the_peer_encoding() {
        let phase = PhaseId::new(9);
        let encoded = rmp_serde::to_vec(&phase).expect("a phase id encodes");
        let decoded: PhaseId = rmp_serde::from_slice(&encoded).expect("and decodes");
        assert_eq!(decoded, phase);
    }
}
