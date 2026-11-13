//! The scheduling policy's host (`specs/scheduler-design.md` §5): [`Policy`], a WASM module's
//! instance with the seven calls of the policy interface, and [`PolicyError`], the faults of a
//! policy. The scheduler holds one `Policy`; its fields, calls, switch and rollback are stated
//! here with the policy's host.

/// The serving scheduling policy and the one it replaced, kept for rollback.
#[derive(Debug)]
pub struct Policy {
    /// The calls into the policy's instances since the scheduler started, over every generation
    /// it served.
    pub calls: u64,
    /// The fuel those calls consumed.
    pub fuel: u64,
    /// The state changes since the policy was last told, in the order they happened: what its
    /// next `observe` carries.
    pub events: Vec<scheduling_guest::Event>,
}

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
