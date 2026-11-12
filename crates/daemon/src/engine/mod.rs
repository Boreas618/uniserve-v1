//! The leader's view of its processes: starting a main process (`control`), the
//! client of the request interface it serves (`client`), the launch file it reads (`launch`),
//! and the hold every request passes, which waits while no engine is ready
//! (`hold`).

pub mod client;
pub mod control;
pub mod hold;
pub(crate) mod launch;
