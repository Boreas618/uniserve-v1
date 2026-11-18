//! A control program's input and what it reads through its host.
//!
//! A control program runs on the leader daemon, every `period` of its package entry. Its input
//! is [`State`]: the step's instant, the deployment's current configuration, and read-only views
//! of the observation library's two tables, `stats` (values over time) and `events` (what
//! happened). The views are lazy: a program names a series or a kind of event, narrows it, and
//! only the aggregate or the records it asks for are computed, by the host, when it asks
//! (`uniserve.stats`, `uniserve.events`). Its output is a new configuration, or `None` to leave
//! the deployment as it is; [`State::validate`] tells it beforehand what a configuration would
//! do.
//!
//! ```ignore
//! let tpot = state.stats["uniserve_tpot_seconds"].window(secs(10)).p99();
//! let ttft = state.stats["uniserve_ttft_seconds"].window(secs(10)).by("replica").p99();
//! let last = state.events["revision"].writer(&state.name).last();
//! ```

use std::collections::BTreeMap;
use std::ops::Index;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::program::guest;
use crate::time::{Duration, Instant};

/// The input of a control program's step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    /// The program's name in its package (`[programs.<name>]`): the writer the revisions of its
    /// outputs are recorded under, which its `events` select its own writes by.
    #[serde(default)]
    pub name: String,
    /// The step's instant.
    pub time: Instant,
    /// The deployment's current configuration.
    pub config: Config,
    /// The values over time, by name and labels; the host computes what the program reads.
    #[serde(skip)]
    pub stats: Stats,
    /// The records of what happened, by kind; the host selects what the program reads.
    #[serde(skip)]
    pub events: Events,
}

impl State {
    /// The input of program `name`'s step at `time` with the deployment's configuration
    /// `config`.
    pub fn new(name: impl Into<String>, time: Instant, config: Config) -> State {
        State {
            name: name.into(),
            time,
            config,
            stats: Stats::default(),
            events: Events::default(),
        }
    }

    /// What applying `config` would do, without applying it: the heaviest effect among the
    /// options it changes and whether serving would be interrupted; or why it would be rejected,
    /// by the same validation `apply` runs.
    pub fn validate(&self, config: &Config) -> Result<Report, String> {
        guest::validate(config)
    }

    /// Whether the latest revision this program wrote was rolled back less than `span` before
    /// the step's instant: what a program holds back for after a change of its own failed its
    /// trial or was refused, read from its `revision` events.
    pub fn rolled_back(&self, span: Duration) -> bool {
        let Some(last) = self.events.kind("revision").writer(&self.name).last() else {
            return false;
        };
        let result = last.get::<String>("result");
        result.as_deref() == Some("rolled_back") && self.time < last.time + span
    }

    /// Whether `config` may be output: validation accepts it and serving goes on while it takes
    /// effect. A refusal is logged with its reason, since the program's own arithmetic produced
    /// a configuration the deployment does not take.
    pub fn admits(&self, config: &Config) -> bool {
        match self.validate(config) {
            Ok(report) => !report.interrupts,
            Err(reason) => {
                log::warn!("the change is refused: {reason}");
                false
            }
        }
    }
}

/// A deployment's whole configuration, every option at its value, the defaults included: the
/// JSON of the leader's configuration type, which reads back as exactly that configuration, so a
/// program sees every option and an output it changes nothing else of changes nothing else.
/// Options are addressed by dotted paths such as `scheduler_config.chunked_prefill_size`; an
/// option the configuration leaves unset reads as `None`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Config(pub Value);

impl Config {
    /// The option at `path` as a `T`; `None` when the configuration states no such option, or
    /// states it as another type.
    pub fn get<T: DeserializeOwned>(&self, path: &str) -> Option<T> {
        let mut value = &self.0;
        for part in path.split('.') {
            value = value.get(part)?;
        }
        T::deserialize(value).ok()
    }

    /// Sets the option at `path` to `value`, creating the tables on the way; a table on the way
    /// that the configuration states as another value is replaced by one.
    pub fn set(&mut self, path: &str, value: impl Into<Value>) {
        let mut slot = &mut self.0;
        for part in path.split('.') {
            if !slot.is_object() {
                *slot = Value::Object(serde_json::Map::new());
            }
            let Value::Object(table) = slot else {
                unreachable!("the slot was made a table");
            };
            slot = table.entry(part).or_insert(Value::Null);
        }
        *slot = value.into();
    }
}

/// What applying a configuration would do ([`State::validate`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    /// The heaviest effect among the changed options, as configuration files name effects
    /// (`step`, `order`, `program`, `generation`, `storage`, `epoch`, `daemon`); `None` when
    /// nothing changes.
    pub effect: Option<String>,
    /// Whether serving would stop while the change takes effect: the new engine fits beside the
    /// old one neither whole nor by shrinking it, so the old one drains first.
    pub interrupts: bool,
    /// What a kernel generation the change starts takes on each GPU of the serving engines
    /// beside the graphs they serve, and the KV pages each device's cache demotes to host memory
    /// and gives back while both generations' graphs are held; empty when the change starts no
    /// generation or it replaces no kernel.
    #[serde(default)]
    pub rooms: Vec<Room>,
}

/// One GPU's share of a kernel generation ([`Report::rooms`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Room {
    /// The GPU's UUID, in lowercase hexadecimal.
    pub gpu: String,
    /// The bytes the generation's graphs take beside what serves until its commit.
    pub extra_bytes: u64,
    /// What the device holds of them now.
    pub room_bytes: u64,
    /// The KV pages the device's cache demotes and gives back when the room lacks them.
    pub kv_pages: u64,
}

/// The `stats` table: values over time, each a number or a histogram in time buckets, by name
/// and labels. Indexed by a series' name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    _view: (),
}

impl Stats {
    /// The series named `name`, for a name the program computes.
    pub fn series(&self, name: impl Into<String>) -> Window {
        Window::of(name.into())
    }
}

impl Index<&'static str> for Stats {
    type Output = Series;

    fn index(&self, name: &'static str) -> &Series {
        Series::of(name)
    }
}

/// A series of the `stats` table, by its name: the values of every label set under it.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq)]
pub struct Series(str);

impl Series {
    fn of(name: &str) -> &Series {
        // SAFETY: `Series` is a transparent wrapper of `str`, so the two references have one
        // layout and the same lifetime.
        unsafe { &*(name as *const str as *const Series) }
    }

    /// The last `span` of the series, ending at the step's instant.
    pub fn window(&self, span: Duration) -> Window {
        Window::of(self.0.to_owned()).window(span)
    }
}

/// A selection of a series: a window ending at the step's instant, the label values a value must
/// have, the configuration revision it must have been measured under. Its aggregates are what
/// the host computes; each is NaN when the selection holds no value, so a comparison with it
/// fails.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    query: Query,
}

impl Window {
    fn of(name: String) -> Window {
        Window {
            query: Query {
                name,
                labels: BTreeMap::new(),
                window: Duration(0),
                revision: None,
                function: Function::Last,
            },
        }
    }

    /// The last `span`, ending at the step's instant.
    pub fn window(mut self, span: Duration) -> Window {
        self.query.window = span;
        self
    }

    /// Only the values whose label `key` is `value`.
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Window {
        self.query.labels.insert(key.into(), value.into());
        self
    }

    /// Only the values measured under the configuration revision `revision`.
    pub fn revision(mut self, revision: u64) -> Window {
        self.query.revision = Some(revision);
        self
    }

    fn compute(&self, function: Function) -> f64 {
        let query = Query {
            function,
            ..self.query.clone()
        };
        guest::stats(&query)
    }

    /// The `q` quantile, `q` in `[0, 1]`.
    pub fn quantile(&self, q: f64) -> f64 {
        self.compute(Function::Quantile(q))
    }

    /// The median.
    pub fn p50(&self) -> f64 {
        self.quantile(0.5)
    }

    /// The 0.9 quantile.
    pub fn p90(&self) -> f64 {
        self.quantile(0.9)
    }

    /// The 0.99 quantile.
    pub fn p99(&self) -> f64 {
        self.quantile(0.99)
    }

    /// The mean.
    pub fn mean(&self) -> f64 {
        self.compute(Function::Mean)
    }

    /// The smallest value.
    pub fn min(&self) -> f64 {
        self.compute(Function::Min)
    }

    /// The largest value.
    pub fn max(&self) -> f64 {
        self.compute(Function::Max)
    }

    /// The sum of the values.
    pub fn sum(&self) -> f64 {
        self.compute(Function::Sum)
    }

    /// The number of values (of observations, for a histogram).
    pub fn count(&self) -> f64 {
        self.compute(Function::Count)
    }

    /// A counter's increase per second over the window.
    pub fn rate(&self) -> f64 {
        self.compute(Function::Rate)
    }

    /// The latest value.
    pub fn last(&self) -> f64 {
        self.compute(Function::Last)
    }

    /// The selection split by the values of its label `key`, as PromQL's `by (key)`: each
    /// aggregate of it is one value per label value, over the series that carry that value.
    pub fn by(&self, key: impl Into<String>) -> Groups {
        Groups {
            query: self.query.clone(),
            key: key.into(),
        }
    }

    /// The values the selection's series in the window carry for label `key`, in order.
    pub fn values(&self, key: impl Into<String>) -> Vec<String> {
        self.by(key).count().into_keys().collect()
    }
}

/// A selection of a series split by the values of one label ([`Window::by`]). Each aggregate is
/// a map from a label value to the aggregate over the series that carry it; a value whose series
/// hold nothing in the window is absent.
#[derive(Clone, Debug, PartialEq)]
pub struct Groups {
    query: Query,
    key: String,
}

impl Groups {
    fn compute(&self, function: Function) -> BTreeMap<String, f64> {
        let group = Group {
            query: Query {
                function,
                ..self.query.clone()
            },
            key: self.key.clone(),
        };
        guest::group(&group)
    }

    /// The `q` quantile of each group, `q` in `[0, 1]`.
    pub fn quantile(&self, q: f64) -> BTreeMap<String, f64> {
        self.compute(Function::Quantile(q))
    }

    /// The median of each group.
    pub fn p50(&self) -> BTreeMap<String, f64> {
        self.quantile(0.5)
    }

    /// The 0.9 quantile of each group.
    pub fn p90(&self) -> BTreeMap<String, f64> {
        self.quantile(0.9)
    }

    /// The 0.99 quantile of each group.
    pub fn p99(&self) -> BTreeMap<String, f64> {
        self.quantile(0.99)
    }

    /// The mean of each group.
    pub fn mean(&self) -> BTreeMap<String, f64> {
        self.compute(Function::Mean)
    }

    /// The smallest value of each group.
    pub fn min(&self) -> BTreeMap<String, f64> {
        self.compute(Function::Min)
    }

    /// The largest value of each group.
    pub fn max(&self) -> BTreeMap<String, f64> {
        self.compute(Function::Max)
    }

    /// The sum of each group's values.
    pub fn sum(&self) -> BTreeMap<String, f64> {
        self.compute(Function::Sum)
    }

    /// The number of each group's values (of observations, for a histogram).
    pub fn count(&self) -> BTreeMap<String, f64> {
        self.compute(Function::Count)
    }

    /// Each group's counter increase per second over the window.
    pub fn rate(&self) -> BTreeMap<String, f64> {
        self.compute(Function::Rate)
    }

    /// The latest value of each group.
    pub fn last(&self) -> BTreeMap<String, f64> {
        self.compute(Function::Last)
    }
}

/// One aggregate of the `stats` table: the message of `uniserve.stats`, which the host answers
/// with its value (NaN for an empty selection).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Query {
    /// The series' name.
    pub name: String,
    /// The label values a value must have.
    pub labels: BTreeMap<String, String>,
    /// The span of the window, which ends at the step's instant.
    pub window: Duration,
    /// The configuration revision a value must have been measured under.
    pub revision: Option<u64>,
    /// The aggregate.
    pub function: Function,
}

/// One aggregate of the `stats` table split by a label: the message of `uniserve.group`, which
/// the host answers with the aggregate of each value of the label among the selected series, a
/// value whose selection holds nothing left out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Group {
    /// The selection and the aggregate.
    pub query: Query,
    /// The label whose values split the selection.
    pub key: String,
}

/// An aggregate over a window.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Function {
    Quantile(f64),
    Mean,
    Min,
    Max,
    Sum,
    Count,
    Rate,
    Last,
}

/// The `events` table: what happened, by kind (`request`, `revision`, `process`, `span`, `log`).
/// Indexed by a kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Events {
    _view: (),
}

impl Events {
    /// The records of `kind`, for a kind the program computes.
    pub fn kind(&self, kind: impl Into<String>) -> Records {
        Records::of(kind.into())
    }
}

impl Index<&'static str> for Events {
    type Output = Kind;

    fn index(&self, kind: &'static str) -> &Kind {
        Kind::of(kind)
    }
}

/// A kind of the `events` table: every record of it.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq)]
pub struct Kind(str);

impl Kind {
    fn of(kind: &str) -> &Kind {
        // SAFETY: `Kind` is a transparent wrapper of `str`, so the two references have one
        // layout and the same lifetime.
        unsafe { &*(kind as *const str as *const Kind) }
    }

    fn records(&self) -> Records {
        Records::of(self.0.to_owned())
    }

    /// The records written by `writer` (the field `writer`).
    pub fn writer(&self, writer: &str) -> Records {
        self.records().writer(writer)
    }

    /// The records whose field `key` is `value`.
    pub fn field(&self, key: impl Into<String>, value: impl Into<Value>) -> Records {
        self.records().field(key, value)
    }

    /// The records whose label `key` is `value`.
    pub fn label(&self, key: impl Into<String>, value: impl Into<String>) -> Records {
        self.records().label(key, value)
    }

    /// The records of the last `span`, ending at the step's instant.
    pub fn window(&self, span: Duration) -> Records {
        self.records().window(span)
    }

    /// The latest record.
    pub fn last(&self) -> Option<Record> {
        self.records().last()
    }

    /// Every record, oldest first.
    pub fn all(&self) -> Vec<Record> {
        self.records().all()
    }
}

/// A selection of a kind's records, which the host reads when the program asks for them.
#[derive(Clone, Debug, PartialEq)]
pub struct Records {
    filter: Filter,
}

impl Records {
    fn of(kind: String) -> Records {
        Records {
            filter: Filter {
                kind,
                labels: BTreeMap::new(),
                fields: BTreeMap::new(),
                window: None,
                limit: None,
            },
        }
    }

    /// Only the records written by `writer` (the field `writer`).
    pub fn writer(self, writer: &str) -> Records {
        self.field("writer", writer)
    }

    /// Only the records whose field `key` is `value`.
    pub fn field(mut self, key: impl Into<String>, value: impl Into<Value>) -> Records {
        self.filter.fields.insert(key.into(), value.into());
        self
    }

    /// Only the records whose label `key` is `value`.
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Records {
        self.filter.labels.insert(key.into(), value.into());
        self
    }

    /// Only the records of the last `span`, ending at the step's instant.
    pub fn window(mut self, span: Duration) -> Records {
        self.filter.window = Some(span);
        self
    }

    /// The latest `n` records, oldest first.
    pub fn latest(&self, n: u32) -> Vec<Record> {
        let filter = Filter {
            limit: Some(n),
            ..self.filter.clone()
        };
        guest::events(&filter)
    }

    /// The latest record.
    pub fn last(&self) -> Option<Record> {
        self.latest(1).pop()
    }

    /// Every record, oldest first.
    pub fn all(&self) -> Vec<Record> {
        guest::events(&self.filter)
    }
}

/// A selection of the `events` table: the message of `uniserve.events`, which the host answers
/// with the selected records, oldest first.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Filter {
    /// The kind of the records.
    pub kind: String,
    /// The label values a record must have.
    pub labels: BTreeMap<String, String>,
    /// The field values a record must have.
    pub fields: BTreeMap<String, Value>,
    /// The span of the window, which ends at the step's instant; all of the table's history
    /// when `None`.
    pub window: Option<Duration>,
    /// The number of latest records; every selected record when `None`.
    pub limit: Option<u32>,
}

/// A row of the `events` table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// When it happened.
    pub time: Instant,
    /// What kind of thing happened.
    pub kind: String,
    /// Where in the deployment: replica, scheduler, group, rank, device, and the like.
    pub labels: BTreeMap<String, String>,
    /// What the kind records, such as a revision's `writer` and `result`.
    pub fields: BTreeMap<String, Value>,
    /// The configuration revision that was running.
    pub revision: u64,
}

impl Record {
    /// The field `key` as a `T`; `None` when the record has no such field or it is another type.
    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        T::deserialize(self.fields.get(key)?).ok()
    }
}

/// An exponentially weighted moving average: what a program keeps in its state of a measurement
/// it compares over a longer history than the observation store keeps, or under conditions the
/// store does not label, such as the layout or the backend a value was measured under.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Ema {
    /// The average; meaningless while `samples` is zero.
    pub mean: f64,
    /// The values averaged.
    pub samples: u64,
}

impl Ema {
    /// Averages in `value` with weight `weight` in `(0, 1]`: the first value is the average,
    /// each later one moves it by `weight` of the difference. A value that is not finite, which
    /// is what an aggregate of an empty selection is, is not averaged.
    pub fn update(&mut self, value: f64, weight: f64) {
        if !value.is_finite() {
            return;
        }
        if self.samples == 0 {
            self.mean = value;
        } else {
            self.mean += weight * (value - self.mean);
        }
        self.samples += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An option is read and written by its dotted path, and a path through a missing table
    /// creates it.
    #[test]
    fn options_are_read_and_written_by_path() {
        let mut config = Config(serde_json::json!({
            "scheduler_config": { "chunked_prefill_size": 2048 },
            "model": "Qwen/Qwen3-30B-A3B",
        }));

        assert_eq!(
            config.get::<u32>("scheduler_config.chunked_prefill_size"),
            Some(2048)
        );
        assert_eq!(config.get::<u32>("model"), None);
        assert_eq!(config.get::<u32>("scheduler_config.max_batch"), None);

        config.set("scheduler_config.chunked_prefill_size", 1024);
        config.set("moe_config.expert_parallel", true);
        config.set("model.revision", 3);

        assert_eq!(
            config.get::<u32>("scheduler_config.chunked_prefill_size"),
            Some(1024)
        );
        assert_eq!(config.get::<bool>("moe_config.expert_parallel"), Some(true));
        assert_eq!(config.get::<u64>("model.revision"), Some(3));
    }

    /// The step's input carries the program's name, the instant and the configuration; the
    /// views carry nothing, since the host computes what the program reads of them.
    #[test]
    fn the_state_carries_the_name_the_instant_and_the_configuration() {
        let state = State::new("chunk", Instant(7), Config(serde_json::json!({ "a": 1 })));
        let wire = serde_json::to_value(&state).unwrap();
        assert_eq!(
            wire,
            serde_json::json!({ "name": "chunk", "time": 7, "config": { "a": 1 } })
        );
        assert_eq!(serde_json::from_value::<State>(wire).unwrap(), state);
    }

    /// The first value is the average, later ones move it by their weight, and the NaN of an
    /// empty selection is not averaged.
    #[test]
    fn a_moving_average_weighs_each_later_value() {
        let mut average = Ema::default();
        average.update(f64::NAN, 0.5);
        assert_eq!(average.samples, 0);

        average.update(10.0, 0.5);
        average.update(20.0, 0.5);
        assert_eq!(
            average,
            Ema {
                mean: 15.0,
                samples: 2
            }
        );
    }
}
