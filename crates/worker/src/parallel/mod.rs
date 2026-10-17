//! Cross-rank coordination: host collectives, the process-group plan,
//! stream-ordered collectives, and expert parallelism: step agreement, this
//! rank's member of its expert union, its joins and microbatches, and expert
//! weight prefetch.

mod collectives;
mod expert_union;
pub(crate) mod experts;
pub mod host;
pub(crate) mod joins;
mod microbatches;
pub(crate) mod topology;
mod weight_prefetch;
pub(crate) mod world;

pub(crate) use collectives::PeerGroup;
pub use collectives::{CollectiveFunctions, CollectiveGroup, StreamCollectives};
pub(crate) use expert_union::{ExpertStart, ExpertUnion, ExpertWarmup};
pub use experts::{Agreement, ExpertParallel, ExpertTransport};
pub(crate) use microbatches::MicrobatchForward;
pub use microbatches::{MicrobatchCall, MicrobatchLanes, yield_microbatch};
pub(crate) use weight_prefetch::DwdpManager;
