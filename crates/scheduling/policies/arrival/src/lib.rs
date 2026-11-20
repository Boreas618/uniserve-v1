//! The `arrival` scheduling policy: the deployment's default policy (design §4.4).
//!
//! Its waiting queue admits in the order its package's `order` argument names: the instant each
//! request reached the main process (`arrival`), or the request's own priority and then arrival
//! (`priority`). Each step assembles the running requests by path, phase class and admission
//! order; [`Arrival`] states every decision's rule.
//!
//! The package's manifest declares every parameter the policy requires with its default, and the
//! policy is constructed from the arguments its revision runs with (`TryFrom<Parameters>`).

mod arrival;
mod budget;
mod lru;
mod pages;
mod phases;
mod waiting;

pub use arrival::Arrival;

scheduling::export_policy!(Arrival);
