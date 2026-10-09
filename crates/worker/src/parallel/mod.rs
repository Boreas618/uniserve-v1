//! Cross-rank coordination: host collectives, the process-group plan,
//! stream-ordered collectives, expert step agreement, microbatch rotation and
//! expert weight prefetch.

mod collectives;
pub mod experts;
pub mod host;
pub mod joins;
mod microbatches;
pub mod topology;
mod weight_prefetch;
pub mod world;

pub use collectives::{CollectiveGroup, StreamCollectives, StreamCommunicator};
pub use experts::{Agreement, ExpertExchange, ExpertParallel, ExpertPlan, ExpertTransport};
pub use microbatches::{MicrobatchCall, MicrobatchLanes, Microbatches, yield_microbatch};
pub use weight_prefetch::{ExpertWeights, ShardField, WeightPrefetch, WeightStorage};
