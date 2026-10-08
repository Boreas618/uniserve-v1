//! Cross-rank coordination: host collectives, the process-group plan,
//! stream-ordered collectives, expert exchange, microbatch rotation and
//! expert weight prefetch.

mod collectives;
mod experts;
pub mod host;
mod microbatches;
pub mod topology;
mod weight_prefetch;
pub mod world;

pub use collectives::StreamCommunicator;
pub use experts::ExpertExchange;
pub use microbatches::{Microbatches, yield_microbatch};
pub use weight_prefetch::WeightPrefetch;
