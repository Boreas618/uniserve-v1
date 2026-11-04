//! The leader's view of its processes: the application process that serves the
//! API (`application`), starting a main process (`control`), the client of the
//! request interface it serves (`client`), the launch file it reads (`launch`),
//! and the hold every request passes, which waits while no engine is ready
//! (`hold`).

pub(crate) mod application;
pub mod client;
pub mod control;
pub mod hold;
pub(crate) mod launch;

/// Stand-in main processes over the daemon's command channel, for the tests
/// of the program layer: no engine runs, and every command is answered as
/// the protocol says.
#[cfg(test)]
pub(crate) mod stand_in;
