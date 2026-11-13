//! The scheduling policy's host (`specs/scheduler-design.md` §5): [`Policy`], a WASM module's
//! instance with the seven calls of the policy interface, and [`PolicyError`], the faults of a
//! policy. The scheduler holds one `Policy`; its fields, calls, switch and rollback are stated
//! here with the policy's host.

/// The serving scheduling policy and the one it replaced, kept for rollback.
#[derive(Debug)]
pub struct Policy {}

/// A fault of a policy: a trap, exhausted fuel, an undecodable or empty reply, or limits outside
/// the ceilings.
#[derive(Debug)]
pub enum PolicyError {}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {}
    }
}

impl std::error::Error for PolicyError {}
