//! Cross-rank coordination: host collectives, the process-group plan,
//! stream-ordered collectives, and expert parallelism: step agreement, this
//! rank's member of its expert union, its joins and microbatches, and expert
//! weight prefetch.

mod collectives;
mod expert_union;
pub mod experts;
pub mod host;
pub mod joins;
mod microbatches;
pub mod topology;
mod weight_prefetch;
pub mod world;

pub use collectives::{CollectiveGroup, PeerGroup, StreamCollectives, StreamCommunicator};
pub(crate) use expert_union::{ExpertStart, ExpertUnion, ExpertWarmup};
pub use experts::{Agreement, ExpertExchange, ExpertParallel, ExpertPlan, ExpertTransport};
pub(crate) use microbatches::MicrobatchForward;
pub use microbatches::{MicrobatchCall, MicrobatchLanes, Microbatches, yield_microbatch};
pub(crate) use weight_prefetch::DwdpManager;
pub use weight_prefetch::{ExpertWeights, ShardField, WeightPrefetch, WeightStorage};
