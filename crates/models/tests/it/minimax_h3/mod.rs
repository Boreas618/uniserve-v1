//! MiniMax-H3's tests: the composition of a deployment's sources, the packing, the transformer's
//! modules, the VAEs and the denoiser on CPU tensors against the reference implementation, and
//! the stages' profile runs on the meta device.

mod composition;
mod denoiser;
mod denoiser_profile;
mod packing_reference;
mod transformer_reference;
mod vae_profile;
mod vae_reference;
