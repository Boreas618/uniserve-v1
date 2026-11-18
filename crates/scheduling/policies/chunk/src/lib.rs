//! The `chunk` control program: the prefill chunk traded between time to first token (TTFT) and
//! time per output token (TPOT).
//!
//! Chunked prefill computes a long prompt one chunk per step, in the batch of the other
//! requests' decodes: a larger chunk shortens TTFT and lengthens the TPOT of the requests that
//! decode in the same steps. Every `period` the program reads each replica's p99 TTFT and TPOT
//! over the last `window` (`uniserve_ttft_seconds`, `uniserve_tpot_seconds` by `replica`):
//!
//! - while any replica's TPOT is over its target, it halves `scheduler_config.chunked_prefill_size`,
//!   down to `low`;
//! - while no replica's TPOT is and any replica's TTFT is over its target, it doubles the chunk,
//!   up to `high`;
//! - otherwise, and while nothing finished in the window, it leaves the configuration as it is.
//!
//! The chunk takes effect at each scheduler's next step. The program outputs a change only when
//! validation accepts it and serving goes on meanwhile (a chunk above the batch's token budget is
//! refused), and holds back for `pause` after a change of its own was rolled back. An unset chunk
//! is the step's whole token budget: the program reads it as `max_num_batched_tokens` when the
//! configuration states that, and as `high` otherwise.

use scheduling::prelude::*;

/// The option the program writes.
pub const CHUNK: &str = "scheduler_config.chunked_prefill_size";

/// The option an unset chunk is.
pub const BUDGET: &str = "max_num_batched_tokens";

/// The program's arguments, its package entry's `args`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Args {
    /// The p99 TPOT target.
    pub tpot: Duration,
    /// The p99 TTFT target.
    pub ttft: Duration,
    /// The smallest chunk, in tokens.
    pub low: u32,
    /// The largest chunk, in tokens.
    pub high: u32,
    /// The span the p99s are read over (10 s unless stated).
    #[serde(default = "window")]
    pub window: Duration,
    /// How long the program holds back after a change of its own was rolled back (1 h unless
    /// stated).
    #[serde(default = "pause")]
    pub pause: Duration,
}

fn window() -> Duration {
    secs(10)
}

fn pause() -> Duration {
    hours(1)
}

/// The program; its arguments are all its state.
#[derive(Deserialize, Serialize)]
pub struct Chunk {
    args: Args,
}

/// Whether any of `values` (seconds) is over `target`.
fn over(values: &std::collections::BTreeMap<String, f64>, target: Duration) -> bool {
    values.values().any(|&value| value > target.as_secs_f64())
}

impl Program for Chunk {
    type Args = Args;
    type Input = State;
    type Output = Option<Config>;

    fn new(args: Args) -> Self {
        Chunk { args }
    }

    fn step(&mut self, state: &State) -> Option<Config> {
        let Args {
            tpot,
            ttft,
            low,
            high,
            window,
            pause,
        } = self.args;

        // Each replica's p99 over the window, in seconds; a replica that finished nothing in it
        // is absent.
        let tpots = state.stats["uniserve_tpot_seconds"]
            .window(window)
            .by("replica")
            .p99();
        let ttfts = state.stats["uniserve_ttft_seconds"]
            .window(window)
            .by("replica")
            .p99();

        let chunk: u32 = state
            .config
            .get(CHUNK)
            .or_else(|| state.config.get(BUDGET))
            .unwrap_or(high);
        // Neither step leaves `[low, high]`, nor moves a chunk outside it further out.
        let next = if over(&tpots, tpot) {
            (chunk / 2).max(low.min(chunk))
        } else if over(&ttfts, ttft) {
            chunk.saturating_mul(2).min(high.max(chunk))
        } else {
            return None;
        };
        if next == chunk || state.rolled_back(pause) {
            return None;
        }

        let mut config = state.config.clone();
        config.set(CHUNK, next);
        if !state.admits(&config) {
            return None;
        }
        info!("chunk {chunk} -> {next} (p99 tpot by replica {tpots:?}, ttft {ttfts:?})");
        Some(config)
    }
}

export!(Chunk);
