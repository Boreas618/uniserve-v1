//! The host latent region of a denoising rank: where a preempted request's trajectory is saved and
//! restored from ([`host`]), and the copy manifest that lays a saved trajectory out in a resumed
//! request's own pages when the ranks that saved it held other parts of its samples ([`restore`]).
//!
//! A trajectory lives in its slot's row of the rank's trajectory rows
//! ([`BufferMemory::trajectory`](crate::memory::buffers::BufferMemory::trajectory)) from the
//! request's latent preparation to its last step; this module is where that row goes when the
//! request is preempted, and where it comes back from when the request resumes.

pub(crate) mod host;
pub(crate) mod restore;
