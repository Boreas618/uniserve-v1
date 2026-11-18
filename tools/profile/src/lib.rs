//! Reads profiling evidence (`docs/profiling.md`): a deployment's `span`
//! events through its admin socket, as one Perfetto trace; a session's
//! per-worker SQLite files, as a summary or one Perfetto trace; an Nsight
//! Systems export and a cProfile capture, as advisory execution hints. The
//! `uniserve-profile` executable is its command line.

pub mod admin;
pub mod calls;
pub mod nsys;
pub mod perfetto;
pub mod session;
pub mod timeline;
