//! Cross-rank coordination: host collectives, the process-group plan with
//! each stream's groups, and the storage and copy plans of expert and host
//! weight prefetch, whose schedule the library owns.
//!
//! The host group is the only channel rank coordination uses off the device:
//! startup agreement on block sizes, cache capacity and slot fits, plan
//! comparison, warmup broadcasts, and the bootstrap bytes of device
//! communicators. Device collectives never use it.

pub mod host;
pub(crate) mod topology;
