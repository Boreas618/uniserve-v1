//! Cross-rank coordination: host collectives and their bootstrap for the
//! executor's expert union, the process-group plan with each stream's
//! groups, and the storage and copy plans of expert and host weight
//! prefetch, whose schedule the library owns.

pub mod host;
pub(crate) mod topology;
mod weight_prefetch;
pub(crate) mod world;

pub(crate) use weight_prefetch::{
    DwdpManager, DwdpUpdate, ExpertWeights, HostRetention, HostWeights,
};
