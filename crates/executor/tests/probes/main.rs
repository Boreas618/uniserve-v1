//! The design's unverified facts, each measured by a small program
//! (`specs/worker.md` §12). The caller runs them; the main process rests on their
//! answers.
//!
//! A probe is a test so that its evidence is one command's output:
//!
//! ```text
//! cargo test -p executor --test probes -- --nocapture
//! ```
//!
//! `writers` answers §2.5: whether one main process can hold several rank
//! writer identities at the Storage Server. The other three of the design's open
//! questions - recursive node readback under capture (§4.5), two generations of
//! graphs over one graph pool (§7.3), and NCCL collectives captured in a graph
//! with one thread per GPU (§2.4) - were measured by the caller on a GB200 before
//! these probes were written, and their answers are recorded on the swarm's
//! board rather than here.

mod writers;
