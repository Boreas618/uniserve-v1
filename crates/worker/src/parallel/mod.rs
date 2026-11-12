//! Cross-rank coordination off the device: the host collectives of a rank group.
//!
//! The host group is the only channel rank coordination uses off the device: startup agreement
//! on block sizes, cache capacity and slot fits, the comparison of the ranks' buckets, warm-up
//! broadcasts, the bootstrap bytes of device communicators, and an expert union's agreement on
//! each step. Device collectives never use it.

pub mod host;
