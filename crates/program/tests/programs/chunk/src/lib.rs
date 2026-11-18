//! The control program of a deployment that trades time to first token against time per output
//! token through the prefill chunk: every `period` it halves the chunk while the last ten
//! seconds' p99 TPOT is over its target, doubles it while the p99 TTFT is over its target and
//! TPOT is not, within `[low, high]`, and otherwise leaves the configuration as it is. It holds
//! back for `cooldown` after a change of its own was rolled back, and never outputs a change
//! that would interrupt serving.

use scheduling::prelude::*;

/// The option the program writes.
const CHUNK: &str = "scheduler_config.chunked_prefill_size";

/// The writer its revisions are recorded under: its name in the package.
const WRITER: &str = "chunk";

#[derive(Deserialize, Serialize)]
pub struct Args {
    /// The p99 TPOT target.
    tpot: Duration,
    /// The p99 TTFT target.
    ttft: Duration,
    /// The smallest chunk, in tokens.
    low: u32,
    /// The largest chunk, in tokens.
    high: u32,
    /// How long to hold back after a change of its own was rolled back.
    cooldown: Duration,
}

#[derive(Deserialize, Serialize)]
pub struct Chunk {
    args: Args,
    /// The changes the program made, which survive restarts with its state.
    #[serde(default)]
    changes: u64,
}

impl Program for Chunk {
    type Args = Args;
    type Input = State;
    type Output = Option<Config>;

    fn new(args: Args) -> Self {
        Chunk { args, changes: 0 }
    }

    fn step(&mut self, state: &State) -> Option<Config> {
        // The last ten seconds' p99, in seconds; NaN when nothing finished, so no comparison
        // holds.
        let tpot = state.stats["uniserve_tpot_seconds"].window(secs(10)).p99();
        let ttft = state.stats["uniserve_ttft_seconds"].window(secs(10)).p99();
        let chunk: u32 = state.config.get(CHUNK)?;

        // Halve while TPOT is over its target; double while TTFT is over its target and TPOT is
        // not; otherwise no change.
        let next = if tpot > self.args.tpot.as_secs_f64() {
            (chunk / 2).max(self.args.low)
        } else if ttft > self.args.ttft.as_secs_f64() {
            chunk.saturating_mul(2).min(self.args.high)
        } else {
            return None;
        };
        if next == chunk {
            return None;
        }

        // The last change of its own was rolled back: hold back for the cooldown.
        if let Some(last) = state.events["revision"].writer(WRITER).last() {
            let rolled_back = last.get::<String>("result").as_deref() == Some("rolled_back");
            if rolled_back && state.time < last.time + self.args.cooldown {
                return None;
            }
        }

        // The chunk takes effect at the next step; a chunk above the batch's token limit is
        // rejected, and a change that would stop serving is not made.
        let mut config = state.config.clone();
        config.set(CHUNK, next);
        match state.validate(&config) {
            Ok(report) if !report.interrupts => {}
            Ok(_) => return None,
            Err(reason) => {
                warn!("chunk {next} is rejected: {reason}");
                return None;
            }
        }
        self.changes += 1;
        info!("chunk {chunk} -> {next} (p99 tpot {tpot:.3}s, ttft {ttft:.3}s)");
        Some(config)
    }
}

export!(Chunk);
