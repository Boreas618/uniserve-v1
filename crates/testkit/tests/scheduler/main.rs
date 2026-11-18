//! The scheduler's integration suite (`specs/scheduler-design.md` §8, SD): the engine's
//! scheduler driven through its public surface (`Scheduler::new`, `apply`, `step`) over the fake
//! executor of its worker groups and the Storage Server's buffer pool behind real queue pairs,
//! one module per area of SD §8.1, and L6 of SD §10 as a module of its own. Every step of every
//! case checks the scheduler's side of A1 to A4 (`specs/buffer-pool.md` §9, §17; SD §10, L5).

#![cfg(test)]

mod columns;
mod drain;
mod failures;
mod groups;
mod handoff;
mod harness;
mod images;
mod keys;
mod l6;
mod limits;
mod memory;
mod pause_flush_resize;
mod policy;
mod preemption;
mod prefill;
mod queue_pair;
mod speculation;
mod statistics;
mod status;
mod video;
mod video_preemption;
