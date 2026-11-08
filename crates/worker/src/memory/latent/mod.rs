//! The host latent pool of a denoising rank: the Storage Server page pool a preempted request's
//! trajectory is written to ([`host::LatentHostPool`]), and the copy manifest that lays a stored
//! trajectory out in a resumed request's own pages ([`restore`]).
//!
//! A denoiser holds a request's trajectory in its own columns from the request's latent preparation
//! to its last step (`crate::runners::denoiser`); this module is where that trajectory goes when the
//! request is preempted, and where it comes back from when the request resumes.

pub mod host;
pub(crate) mod restore;
