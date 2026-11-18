//! The `role` planning program: a replica's role switched in place between prefill and decode by
//! which side misses its latency target.
//!
//! Every replica is built mixed, with both phases' graphs captured, and `replicas.<id>.role`
//! decides only which requests it admits, so a change of a role takes effect at the replica's next
//! step: it stops taking requests of its former role, finishes those it holds, and takes the
//! other's, its process, weights and KV staying.
//!
//! Every `period` the program reads the deployment's p99 time to first token and time per output
//! token over the last `window` (`uniserve_ttft_seconds`, `uniserve_tpot_seconds`). When TTFT is
//! over `ttft` while TPOT has `room` below `tpot` (TPOT under `(1 - room) * tpot`), prompts wait
//! for prefill while decode has capacity to give: one decode replica becomes a prefill replica.
//! When TPOT is over `tpot` while TTFT has room below `ttft`, one prefill replica becomes a decode
//! replica. The replica that changes is the one of the giving role with the fewest running
//! requests (`uniserve_scheduler_running_requests` by `replica`), ties going to the first id. A
//! role keeps at least `keep` replicas, mixed replicas are never changed, and a change waits
//! `interval` after the program's last one, so that its effect is measured before the next.
//!
//! The program outputs a change only when validation accepts it and serving goes on meanwhile,
//! and holds back for `pause` after a change of its own was rolled back.

use std::collections::BTreeMap;

use scheduling::prelude::*;

/// The table of the deployment's replicas, by id.
pub const REPLICAS: &str = "replicas";

/// The program's arguments, its package entry's `args`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Args {
    /// The p99 time to first token target.
    pub ttft: Duration,
    /// The p99 time per output token target.
    pub tpot: Duration,
    /// The fraction below its target the other side's p99 must be for it to give a replica (0.2
    /// unless stated).
    #[serde(default = "room")]
    pub room: f64,
    /// The span the p99s and the running requests are read over (1 min unless stated).
    #[serde(default = "window")]
    pub window: Duration,
    /// The least time between two changes of the program's own (5 min unless stated).
    #[serde(default = "interval")]
    pub interval: Duration,
    /// The replicas each of the two roles keeps at least (1 unless stated).
    #[serde(default = "keep")]
    pub keep: usize,
    /// How long the program holds back after a change of its own was rolled back (1 h unless
    /// stated).
    #[serde(default = "pause")]
    pub pause: Duration,
}

fn room() -> f64 {
    0.2
}

fn window() -> Duration {
    mins(1)
}

fn interval() -> Duration {
    mins(5)
}

fn keep() -> usize {
    1
}

fn pause() -> Duration {
    hours(1)
}

/// The program and its state.
#[derive(Deserialize, Serialize)]
pub struct Planner {
    args: Args,
    /// When the program last changed a role.
    #[serde(default)]
    changed: Option<Instant>,
}

/// The role a replica's table states; `None` for a table that states none or another value.
fn role(replica: &Value) -> Option<Role> {
    Role::deserialize(replica.get("role")?).ok()
}

impl Program for Planner {
    type Args = Args;
    type Input = State;
    type Output = Option<Config>;

    fn new(args: Args) -> Self {
        Planner {
            args,
            changed: None,
        }
    }

    fn step(&mut self, state: &State) -> Option<Config> {
        let Args {
            ttft,
            tpot,
            room,
            window,
            interval,
            keep,
            pause,
        } = self.args;
        if self
            .changed
            .is_some_and(|changed| state.time < changed + interval)
        {
            return None;
        }

        // NaN, of a window in which nothing finished, is neither over nor under a target.
        let ttft_p99 = state.stats["uniserve_ttft_seconds"].window(window).p99();
        let tpot_p99 = state.stats["uniserve_tpot_seconds"].window(window).p99();
        let (ttft, tpot) = (ttft.as_secs_f64(), tpot.as_secs_f64());
        let (giver, taker) = if ttft_p99 > ttft && tpot_p99 < tpot * (1.0 - room) {
            (Role::Decode, Role::Prefill)
        } else if tpot_p99 > tpot && ttft_p99 < ttft * (1.0 - room) {
            (Role::Prefill, Role::Decode)
        } else {
            return None;
        };

        let replicas: BTreeMap<String, Value> = state.config.get(REPLICAS).unwrap_or_default();
        let givers: Vec<&String> = replicas
            .iter()
            .filter(|(_, replica)| role(replica) == Some(giver))
            .map(|(id, _)| id)
            .collect();
        if givers.len() <= keep {
            return None;
        }

        // The giving replica with the fewest running requests; one that reported none counts
        // as idle.
        let running = state.stats["uniserve_scheduler_running_requests"]
            .window(window)
            .by("replica")
            .last();
        let load = |id: &String| running.get(id).copied().unwrap_or(0.0);
        let id = givers
            .into_iter()
            .min_by(|a, b| load(a).total_cmp(&load(b)).then(a.cmp(b)))?
            .clone();
        if state.rolled_back(pause) {
            return None;
        }

        let mut config = state.config.clone();
        let name = match taker {
            Role::Prefill => "prefill",
            Role::Decode => "decode",
            Role::Mixed => "mixed",
        };
        config.set(&format!("{REPLICAS}.{id}.role"), name);
        if !state.admits(&config) {
            return None;
        }
        self.changed = Some(state.time);
        info!(
            "replica {id}: {giver:?} -> {taker:?} (p99 ttft {ttft_p99:.3}s, tpot {tpot_p99:.3}s)"
        );
        Some(config)
    }
}

export!(Planner);
