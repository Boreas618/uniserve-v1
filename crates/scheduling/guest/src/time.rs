//! Monotonic time on the wire.
//!
//! The mechanism measures time with one monotonic clock per process; a policy
//! only compares instants and durations, never reads a clock of its own (a
//! policy that did would make the same decisions untestable and would let a
//! randomized policy diverge from its program's seed). Both are nanoseconds of
//! that clock since the process started.

use serde::{Deserialize, Serialize};

/// An instant on the mechanism's monotonic clock, in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Nanos(pub u64);

/// A span of nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Duration(pub u64);
