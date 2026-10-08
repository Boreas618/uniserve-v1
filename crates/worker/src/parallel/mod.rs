//! Cross-rank coordination: stream-ordered collectives, expert exchange,
//! microbatch rotation and expert weight prefetch.

mod collectives;
mod experts;
mod microbatches;
mod weight_prefetch;

pub use collectives::StreamCommunicator;
pub use experts::ExpertExchange;
pub use microbatches::{Microbatches, yield_microbatch};
pub use weight_prefetch::WeightPrefetch;
