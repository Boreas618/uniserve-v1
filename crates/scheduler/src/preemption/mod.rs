//! Preemption: what ends or suspends a running request so that others can
//! run, and resumes it later.
//!
//! - [`media`]: video requests and a token request's image, preempted and
//!   resumed at denoising-step boundaries through their stored trajectories.
//! - [`tokens`]: token requests that cannot get the KV cache units their
//!   next call needs, recomputed or ended at the length the cache holds.

mod media;
mod tokens;
