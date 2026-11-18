//! The `backend` control program: deeper observation when the decode step slows, and the
//! attention kernels moved to the backend that measured faster.
//!
//! Every `period` the program compares the decode step's median device time over the last
//! `window` with its median over the last `baseline` (`uniserve_step_device_seconds` of phase
//! `decode`, module `step`). When it is more than `slowdown` slower, the program opens a probe:
//! it writes `profiling_config.fence_modules` as `modules` (every module class unless stated),
//! with `fence_probes` on and `fence_probes_off` empty, so that every module's device time is
//! measured. Once the configuration states the probe, it runs for `probe`; the step after that
//! writes the three options back to the values they had before the probe, and reads what the
//! probe measured:
//!
//! - each module class's median device time per decode step over the probe; a class slower than
//!   its record of the earlier probes by more than `slowdown` is a slowed module, which the
//!   program logs;
//! - the descriptor keys the operator `op` launched over the probe
//!   (`uniserve_kernel_launches_total` by `descriptor`), the attention kernels of the batch sizes
//!   that ran, and the backend each runs on: the backend `[kernels.<op>]` pins it to, or the first
//!   of `backends`, the one the Build Server selects by priority, for a key the table does not pin;
//! - the attention classes' (`attention` patterns) device time, which is averaged into the record
//!   of the backend that ran these keys.
//!
//! When the keys ran on one backend, the program moves them to the next of `backends` when that
//! backend's record on these keys is faster than this probe by more than `margin`, or, with no
//! record of it, when an attention class is among the slowed modules, so that the probe after
//! measures it. The move pins each key in `[kernels.<op>]` to the other backend, at the revision
//! the table already pins that backend at for the operator, or `latest`; it takes effect as a
//! kernel generation, whose new graphs are compared with the old ones before each bucket
//! switches. The move is written with the probe's end, in one change.
//!
//! A probe opens at most once per `interval`. The program outputs a change only when validation
//! accepts it and serving goes on meanwhile, and opens no probe and moves no kernel for `pause`
//! after a change of its own was rolled back; closing an open probe is never held back. A probe
//! whose change did not take effect ends without a reading.

use std::collections::{BTreeMap, BTreeSet};

use scheduling::prelude::*;

/// The module classes whose instances carry probes.
pub const FENCE_MODULES: &str = "profiling_config.fence_modules";

/// Whether the probes record.
pub const FENCE_PROBES: &str = "profiling_config.fence_probes";

/// The probed module classes whose probes are off.
pub const FENCE_PROBES_OFF: &str = "profiling_config.fence_probes_off";

/// The options a probe writes and its end writes back.
pub const PROBE_OPTIONS: [&str; 3] = [FENCE_MODULES, FENCE_PROBES, FENCE_PROBES_OFF];

/// The device time of each probed module class per step, by `phase` and `module`.
pub const DEVICE: &str = "uniserve_step_device_seconds";

/// The launches of each kernel, by `op` and `descriptor`.
pub const LAUNCHES: &str = "uniserve_kernel_launches_total";

/// The module label of a whole step's device time.
pub const STEP: &str = "step";

/// The program's arguments, its package entry's `args`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Args {
    /// How much slower than its baseline the decode step, and than its record a module class,
    /// must be to count as slowed (0.1 unless stated).
    #[serde(default = "slowdown")]
    pub slowdown: f64,
    /// The span of the step time compared with the baseline (10 s unless stated).
    #[serde(default = "window")]
    pub window: Duration,
    /// The span of the baseline (1 h unless stated).
    #[serde(default = "baseline")]
    pub baseline: Duration,
    /// How long a probe measures (30 s unless stated).
    #[serde(default = "probe")]
    pub probe: Duration,
    /// The module class patterns a probe places probes on (`["*"]` unless stated).
    #[serde(default = "modules")]
    pub modules: Vec<String>,
    /// The patterns of the attention module classes (`["*.Attention"]` unless stated).
    #[serde(default = "attention")]
    pub attention: Vec<String>,
    /// The attention operator (`paged_attention` unless stated).
    #[serde(default = "op")]
    pub op: String,
    /// The backends the operator's keys move between, the one the Build Server selects for a
    /// key the table does not pin first (`["trtllm", "triton"]` unless stated).
    #[serde(default = "backends")]
    pub backends: Vec<String>,
    /// How much faster, as a fraction of this probe's time, another backend's record must be for
    /// the keys to move to it (0.05 unless stated).
    #[serde(default = "margin")]
    pub margin: f64,
    /// The least time between two probes (10 min unless stated).
    #[serde(default = "interval")]
    pub interval: Duration,
    /// The weight of each probe's time in a record's moving average (0.3 unless stated).
    #[serde(default = "weight")]
    pub weight: f64,
    /// How long the program holds back after a change of its own was rolled back (1 h unless
    /// stated).
    #[serde(default = "pause")]
    pub pause: Duration,
}

fn slowdown() -> f64 {
    0.1
}

fn window() -> Duration {
    secs(10)
}

fn baseline() -> Duration {
    hours(1)
}

fn probe() -> Duration {
    secs(30)
}

fn modules() -> Vec<String> {
    vec!["*".to_owned()]
}

fn attention() -> Vec<String> {
    vec!["*.Attention".to_owned()]
}

fn op() -> String {
    "paged_attention".to_owned()
}

fn backends() -> Vec<String> {
    vec!["trtllm".to_owned(), "triton".to_owned()]
}

fn margin() -> f64 {
    0.05
}

fn interval() -> Duration {
    mins(10)
}

fn weight() -> f64 {
    0.3
}

fn pause() -> Duration {
    hours(1)
}

/// An open probe: when the configuration first stated it, and the values its options had
/// before it.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Probe {
    /// The step at which the configuration first stated the probe; `None` until it does.
    pub since: Option<Instant>,
    /// The probe's options before it, `null` for an option the configuration left unset.
    pub restore: BTreeMap<String, Value>,
}

/// The program and its state.
#[derive(Deserialize, Serialize)]
pub struct Backend {
    args: Args,
    /// The probe that is open.
    #[serde(default)]
    probe: Option<Probe>,
    /// When the last probe opened.
    #[serde(default)]
    opened: Option<Instant>,
    /// Each module class's median device time per decode step over the probes, in seconds.
    #[serde(default)]
    modules: BTreeMap<String, Ema>,
    /// The attention classes' device time per decode step over the probes, in seconds, by the
    /// descriptor keys that ran (joined with spaces) and the backend they ran on.
    #[serde(default)]
    attention: BTreeMap<String, BTreeMap<String, Ema>>,
}

/// Whether `name` matches `pattern`, in which `*` matches any run of characters.
fn matches(pattern: &str, name: &str) -> bool {
    let mut parts = pattern.split('*');
    let first = parts.next().unwrap_or_default();
    let Some(mut rest) = name.strip_prefix(first) else {
        return false;
    };
    let parts: Vec<&str> = parts.collect();
    let Some((last, middle)) = parts.split_last() else {
        return rest.is_empty();
    };
    for part in middle {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    rest.ends_with(last)
}

impl Backend {
    /// The probe's configuration: the probes on every module class `modules` names.
    fn open(&self, config: &Config) -> Config {
        let mut config = config.clone();
        config.set(FENCE_MODULES, self.args.modules.clone());
        config.set(FENCE_PROBES, true);
        config.set(FENCE_PROBES_OFF, Vec::<String>::new());
        config
    }

    /// Whether `config` states the probe.
    fn probing(&self, config: &Config) -> bool {
        config.get::<Vec<String>>(FENCE_MODULES).as_ref() == Some(&self.args.modules)
            && config.get::<bool>(FENCE_PROBES) == Some(true)
    }

    /// The step at an instant without an open probe: opens one when the decode step slowed.
    fn watch(&mut self, state: &State) -> Option<Config> {
        let Args {
            slowdown,
            window,
            baseline,
            interval,
            pause,
            ..
        } = self.args;
        if self
            .opened
            .is_some_and(|opened| state.time < opened + interval)
        {
            return None;
        }

        let step = |span| {
            state.stats[DEVICE]
                .window(span)
                .label("phase", "decode")
                .label("module", STEP)
                .p50()
        };
        let (recent, usual) = (step(window), step(baseline));
        // NaN, of a window that measured nothing, is no slowdown.
        let slowed = recent > usual * (1.0 + slowdown);
        if !slowed || state.rolled_back(pause) {
            return None;
        }

        let config = self.open(&state.config);
        if !state.admits(&config) {
            return None;
        }
        let restore = PROBE_OPTIONS
            .iter()
            .map(|&path| {
                let value = state.config.get::<Value>(path).unwrap_or(Value::Null);
                (path.to_owned(), value)
            })
            .collect();
        self.probe = Some(Probe {
            since: None,
            restore,
        });
        self.opened = Some(state.time);
        info!("decode step {recent:.6}s against {usual:.6}s: probing every module");
        Some(config)
    }

    /// What the probe that just ended measured: records it, and answers the backend the
    /// attention kernels should move to and their keys, when they should.
    fn read(&mut self, state: &State) -> Option<(String, Vec<String>)> {
        let span = self.args.probe;
        let medians = state.stats[DEVICE]
            .window(span)
            .label("phase", "decode")
            .by("module")
            .p50();

        // The slowed module classes, against the record of the earlier probes.
        let slowed: BTreeSet<&String> = medians
            .iter()
            .filter(|(module, median)| {
                *module != STEP
                    && self
                        .modules
                        .get(*module)
                        .is_some_and(|record| **median > record.mean * (1.0 + self.args.slowdown))
            })
            .map(|(module, _)| module)
            .collect();
        if !slowed.is_empty() {
            info!("slowed modules: {slowed:?}");
        }
        let attention: Vec<&String> = medians
            .keys()
            .filter(|module| {
                self.args
                    .attention
                    .iter()
                    .any(|pattern| matches(pattern, module))
            })
            .collect();
        let attention_slowed = attention.iter().any(|module| slowed.contains(module));
        let time: f64 = attention.iter().map(|module| medians[*module]).sum();
        for (module, &median) in &medians {
            let record = self.modules.entry(module.clone()).or_default();
            record.update(median, self.args.weight);
        }

        // The attention kernels that ran, and the backend they ran on.
        let keys = state.stats[LAUNCHES]
            .window(span)
            .label("op", &self.args.op)
            .values("descriptor");
        if keys.is_empty() || attention.is_empty() {
            return None;
        }
        let pins: BTreeMap<String, String> = state
            .config
            .get(&format!("kernels.{}", self.args.op))
            .unwrap_or_default();
        let backend_of = |key: &String| match pins.get(key) {
            Some(pin) => pin.split('@').next().unwrap_or_default().to_owned(),
            None => self.args.backends.first().cloned().unwrap_or_default(),
        };
        let running: BTreeSet<String> = keys.iter().map(backend_of).collect();
        let [current] = <[String; 1]>::try_from(running.into_iter().collect::<Vec<_>>()).ok()?;

        let records = self.attention.entry(keys.join(" ")).or_default();
        let other = self
            .args
            .backends
            .iter()
            .find(|backend| **backend != current)?
            .clone();
        let moves = match records.get(&other) {
            Some(record) => record.mean < time * (1.0 - self.args.margin),
            None => attention_slowed,
        };
        records
            .entry(current.clone())
            .or_default()
            .update(time, self.args.weight);

        if moves {
            info!("attention {time:.6}s on {current}: moving {keys:?} to {other}");
        }
        moves.then_some((other, keys))
    }

    /// `config` with `keys` of the operator pinned to `backend`: at the revision the table pins
    /// the backend at for another key of the operator, so the operator runs one revision of it,
    /// or at `latest`.
    fn pin(&self, config: &Config, backend: &str, keys: &[String]) -> Config {
        let path = format!("kernels.{}", self.args.op);
        let mut pins: BTreeMap<String, String> = config.get(&path).unwrap_or_default();
        let pin = pins
            .values()
            .find(|pin| pin.split('@').next() == Some(backend))
            .cloned()
            .unwrap_or_else(|| format!("{backend}@latest"));
        for key in keys {
            pins.insert(key.clone(), pin.clone());
        }

        let mut config = config.clone();
        let table = pins
            .into_iter()
            .map(|(key, pin)| (key, Value::String(pin)))
            .collect();
        config.set(&path, Value::Object(table));
        config
    }
}

impl Program for Backend {
    type Args = Args;
    type Input = State;
    type Output = Option<Config>;

    fn new(args: Args) -> Self {
        Backend {
            args,
            probe: None,
            opened: None,
            modules: BTreeMap::new(),
            attention: BTreeMap::new(),
        }
    }

    fn step(&mut self, state: &State) -> Option<Config> {
        let Some(probe) = self.probe.clone() else {
            return self.watch(state);
        };

        // A probe the deployment never took, or another writer replaced, has nothing to read.
        if !self.probing(&state.config) {
            info!("the probe's options are not the deployment's: it ends unread");
            self.probe = None;
            return None;
        }

        // The probe measures from the first step that reads it in the configuration, since its
        // change converged before that step.
        let since = match probe.since {
            Some(since) => since,
            None => {
                self.probe = Some(Probe {
                    since: Some(state.time),
                    ..probe.clone()
                });
                state.time
            }
        };
        if state.time < since + self.args.probe {
            return None;
        }

        // The probe ends: its options go back, with the kernels' move when there is one.
        let mut closed = state.config.clone();
        for (path, value) in &probe.restore {
            closed.set(path, value.clone());
        }
        let moved = self
            .read(state)
            .filter(|_| !state.rolled_back(self.args.pause))
            .map(|(backend, keys)| self.pin(&closed, &backend, &keys))
            .filter(|moved| state.admits(moved));
        let output = match moved {
            Some(moved) => moved,
            None if state.admits(&closed) => closed,
            // The deployment does not take the options back yet: the next step tries again.
            None => return None,
        };
        self.probe = None;
        Some(output)
    }
}

export!(Backend);

#[cfg(test)]
mod tests {
    use super::*;

    /// `*` matches any run, the rest literally.
    #[test]
    fn patterns_match_runs() {
        assert!(matches("*", "models.qwen3.transformer.Attention"));
        assert!(matches("*.Attention", "models.qwen3.transformer.Attention"));
        assert!(matches(
            "models.*.Attention",
            "models.qwen3.transformer.Attention"
        ));
        assert!(matches("step", "step"));
        assert!(!matches("*.Attention", "models.qwen3.transformer.MoE"));
        assert!(!matches("step", "steps"));
        assert!(!matches(
            "models.*.MoE",
            "models.qwen3.transformer.Attention"
        ));
    }
}
