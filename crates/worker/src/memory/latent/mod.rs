//! The latent pool of denoised trajectories: its pages' ownership and
//! trajectory visibility ([`LatentPool`]) and their device storage
//! ([`LatentStorage`]).
//!
//! A trajectory's committed latents live in pages of one of two banks; a call
//! that steps it reads the committed bank and writes its successor into the
//! other bank's pages, so a failed batch never disturbs the committed version.
//! The executor reads and writes the banks itself; the computations' ledger
//! (`crate::executor::latents`) decides which pages and banks each call
//! addresses.

mod memory;
mod pages;

pub use memory::{LatentPlan, LatentStorage};
pub(crate) use pages::{LatentExport, LatentImport, LatentPool, LatentUpdate, TrajectoryVersion};
