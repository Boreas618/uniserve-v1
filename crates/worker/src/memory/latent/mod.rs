//! The latent pool of denoised trajectories: its pages' ownership and
//! trajectory visibility ([`LatentPool`]) and their device storage
//! ([`LatentStorage`]).
//!
//! A trajectory's committed latents live in pages of one of two banks; a call
//! that steps it reads the committed bank and writes its successor into the
//! other bank's pages, so a failed batch never disturbs the committed version.
//! The executor reads and writes the banks itself; the computations' ledger
//! (`crate::executor::latents`) decides which pages and banks each call
//! addresses. A video denoising rank also maps a host latent pool
//! ([`LatentHostPool`]), which a preempted request's trajectory is written
//! to; a resumed one's is restored from the pools that hold it, laid out in
//! the rank's own pages by a copy manifest ([`restore`]).

mod host;
mod memory;
mod pages;
pub(crate) mod restore;

pub(crate) use host::{LatentHostPool, SlotPages};
pub use memory::{LatentPlan, LatentStorage};
pub(crate) use pages::{LatentExport, LatentImport, LatentPool, LatentUpdate, TrajectoryVersion};
