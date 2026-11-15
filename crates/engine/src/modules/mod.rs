//! The packages a main process installs: which kernel packages a generation takes from the
//! deployment and this host's Build Server store ([`select`]), and how their modules and the
//! scheduling policy's are resolved and loaded ([`load`]).
//!
//! [`load::select`] and [`load::install`] are the one way a generation's kernels are taken from a
//! Build Server: a rank's start and a generation update use them, and so do the device tests,
//! which hand the resulting kernels to `ops::Generation`.

pub mod load;
pub(crate) mod select;

pub use select::Routes;
