//! The leader's view of its engine processes: starting one (`control`), the
//! client of the request interface it serves (`client`), and the hold every
//! request passes, which waits while no engine is ready (`hold`).

pub mod client;
pub mod control;
pub mod hold;

/// Stand-in main processes over the daemon's command channel, for the tests
/// of the program layer: no engine runs, and every command is answered as
/// the protocol says.
#[cfg(test)]
pub(crate) mod stand_in;
