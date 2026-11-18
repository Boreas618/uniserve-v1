//! The `layout` control program: the deployment's parallel layout chosen by concurrency.
//!
//! The best parallel layout changes with concurrency: one wide tensor-parallel replica serves a
//! few requests fastest, several narrower replicas serve many with more throughput. The program
//! is given the layouts it chooses among (`layouts`), each by name the options that make it, by
//! their paths (`worker_groups`, `tensor_parallel_size`, `moe_config.expert_parallel`, ...), and
//! the bounds that divide the running requests into ranges (`bounds`).
//!
//! Every `period` it reads the mean of the running requests over the last `window`
//! (`uniserve_scheduler_running_requests`, the deployment's total) and the output token rate over
//! the last `rate` (`uniserve_generation_tokens_total`). The layout that serves is the first named
//! one whose every option the configuration states at its value; once it has served for a whole
//! `window`, the throughput is averaged into its record for the range the running requests fall
//! in. Its state keeps these records, the history of what each layout measured in each range,
//! across restarts.
//!
//! When another layout's record in the current range holds at least `samples` values and exceeds
//! the serving layout's by more than `margin`, the program writes that layout's options. It
//! outputs a change only when validation accepts it and serving goes on meanwhile (a layout
//! change is an epoch, whose engines start beside the serving ones), and holds back for `pause`
//! after a change of its own was rolled back. A configuration that matches no named layout is the
//! operator's, and the program leaves it as it is.

use std::collections::BTreeMap;

use scheduling::prelude::*;

/// The deployment's running requests.
pub const RUNNING: &str = "uniserve_scheduler_running_requests";

/// The deployment's generated tokens.
pub const TOKENS: &str = "uniserve_generation_tokens_total";

/// The program's arguments, its package entry's `args`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Args {
    /// The layouts the program chooses among: for each name, the options that make the layout,
    /// by their paths, at their values.
    pub layouts: BTreeMap<String, BTreeMap<String, Value>>,
    /// The ascending bounds of the ranges of running requests: `[16, 64]` divides them into
    /// below 16, 16 to below 64, and 64 or more.
    pub bounds: Vec<f64>,
    /// The span the running requests are averaged over, and that a layout must have served for
    /// before its throughput counts (15 min unless stated).
    #[serde(default = "window")]
    pub window: Duration,
    /// The span the throughput is measured over (1 min unless stated).
    #[serde(default = "rate")]
    pub rate: Duration,
    /// How much higher another layout's throughput must be, as a fraction of the serving
    /// layout's, for the program to switch to it (0.05 unless stated).
    #[serde(default = "margin")]
    pub margin: f64,
    /// The values a record must average before it is compared (3 unless stated).
    #[serde(default = "samples")]
    pub samples: u64,
    /// The weight of each new throughput in a record's moving average (0.2 unless stated).
    #[serde(default = "weight")]
    pub weight: f64,
    /// How long the program holds back after a change of its own was rolled back (1 h unless
    /// stated).
    #[serde(default = "pause")]
    pub pause: Duration,
}

fn window() -> Duration {
    mins(15)
}

fn rate() -> Duration {
    mins(1)
}

fn margin() -> f64 {
    0.05
}

fn samples() -> u64 {
    3
}

fn weight() -> f64 {
    0.2
}

fn pause() -> Duration {
    hours(1)
}

/// The layout that serves, and since when the program has seen it serve.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Serving {
    pub layout: String,
    pub since: Instant,
}

/// The program and its state.
#[derive(Deserialize, Serialize)]
pub struct Layout {
    args: Args,
    /// The output token rate each layout measured, by layout name and range index.
    #[serde(default)]
    throughput: BTreeMap<String, BTreeMap<usize, Ema>>,
    /// The layout that served at the last step.
    #[serde(default)]
    serving: Option<Serving>,
}

/// Whether `value` holds everything `stated` states: every key of a stated table at a value that
/// holds the stated one, every element of a stated array, and every other value equal. A
/// configuration states every option at its full value, a layout only what makes it.
fn holds(value: &Value, stated: &Value) -> bool {
    match (value, stated) {
        (Value::Object(value), Value::Object(stated)) => stated
            .iter()
            .all(|(key, stated)| value.get(key).is_some_and(|value| holds(value, stated))),
        (Value::Array(value), Value::Array(stated)) => {
            value.len() == stated.len() && value.iter().zip(stated).all(|(v, s)| holds(v, s))
        }
        (Value::Number(value), Value::Number(stated)) => value.as_f64() == stated.as_f64(),
        (value, stated) => value == stated,
    }
}

impl Layout {
    /// The first layout whose every option `config` states at its value.
    fn serving(&self, config: &Config) -> Option<String> {
        self.args
            .layouts
            .iter()
            .find(|(_, options)| {
                options.iter().all(|(path, stated)| {
                    config
                        .get::<Value>(path)
                        .is_some_and(|value| holds(&value, stated))
                })
            })
            .map(|(name, _)| name.clone())
    }

    /// The record of `layout` in `range`, when it averages enough values to compare.
    fn record(&self, layout: &str, range: usize) -> Option<f64> {
        let record = self.throughput.get(layout)?.get(&range)?;
        (record.samples >= self.args.samples).then_some(record.mean)
    }
}

impl Program for Layout {
    type Args = Args;
    type Input = State;
    type Output = Option<Config>;

    fn new(args: Args) -> Self {
        Layout {
            args,
            throughput: BTreeMap::new(),
            serving: None,
        }
    }

    fn step(&mut self, state: &State) -> Option<Config> {
        let current = self.serving(&state.config);
        if self.serving.as_ref().map(|serving| &serving.layout) != current.as_ref() {
            self.serving = current.clone().map(|layout| Serving {
                layout,
                since: state.time,
            });
        }
        let serving = self.serving.clone()?;

        // The range the running requests fall in; with nothing measured there is none.
        let running = state.stats[RUNNING].window(self.args.window).mean();
        if !running.is_finite() {
            return None;
        }
        let range = self.args.bounds.partition_point(|&bound| bound <= running);

        // The throughput counts once the whole window of running requests was this layout's.
        if state.time >= serving.since + self.args.window {
            let throughput = state.stats[TOKENS].window(self.args.rate).rate();
            self.throughput
                .entry(serving.layout.clone())
                .or_default()
                .entry(range)
                .or_default()
                .update(throughput, self.args.weight);
        }

        let own = self.record(&serving.layout, range)?;
        let (best, measured) = self
            .args
            .layouts
            .keys()
            .filter(|layout| **layout != serving.layout)
            .filter_map(|layout| Some((layout, self.record(layout, range)?)))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))?;
        if measured <= own * (1.0 + self.args.margin) || state.rolled_back(self.args.pause) {
            return None;
        }

        let mut config = state.config.clone();
        for (path, value) in &self.args.layouts[best] {
            config.set(path, value.clone());
        }
        if !state.admits(&config) {
            return None;
        }
        info!(
            "layout {} -> {best} at {running:.1} running requests ({own:.0} -> {measured:.0} \
             tokens/s)",
            serving.layout
        );
        Some(config)
    }
}

export!(Layout);
