//! The leader's view of its engine processes: starting one (`control`), the
//! client of the request interface it serves (`client`), and the hold every
//! request passes, which waits while no engine is ready (`hold`).

pub mod client;
pub mod control;
pub mod hold;
