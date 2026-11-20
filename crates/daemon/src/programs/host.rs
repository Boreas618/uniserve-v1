//! What answers a program's host functions on the leader daemon: its log, for every program
//! ([`Logs`]), and the views and the validation behind a control program's `State`
//! ([`Reader`]), all from the deployment's observation store and its current revision.

use std::sync::Arc;
use std::time::Duration;

use std::collections::BTreeMap;

use program::{Filter, Function, Group, Level, Query, Record, Report};
use scheduling::time::Instant;
use tokio::sync::watch;

use super::Current;
use crate::observation::{Event, Fields, Kind, Labels, Store, WRITER};
use crate::policy::Package;

/// The span a stats query reads when it states no window: the store's fine tier, so `last`
/// without a window is the series' latest value.
const LATEST: Duration = Duration::from_secs(3600);

/// The label of a program's log record that states its level.
const LEVEL: &str = "level";

/// Records what a program writes, and what happens to it, as `log` events of the observation
/// store under the program's name as their writer, at the revision that serves.
pub(crate) struct Logs {
    name: String,
    store: Store,
    current: watch::Receiver<Arc<Current>>,
}

impl Logs {
    /// The log of program `name`, written to `store` under the revision `current` names.
    pub(crate) fn new(name: String, store: Store, current: watch::Receiver<Arc<Current>>) -> Self {
        Self {
            name,
            store,
            current,
        }
    }

    /// Records `text` at `level`, with the record's other `fields`.
    pub(crate) fn record(&self, level: Level, text: &str, mut fields: Fields) {
        fields.insert("text".to_owned(), text.into());
        let labels = Labels::from([
            (WRITER.to_owned(), self.name.clone()),
            (
                LEVEL.to_owned(),
                serde_json::to_value(level)
                    .ok()
                    .and_then(|level| level.as_str().map(str::to_owned))
                    .unwrap_or_default(),
            ),
        ]);
        let revision = self.current.borrow().revision;
        self.store.append(Kind::Log, labels, fields, revision);
    }

    /// Records a failure of the program's host side: its instance ended, an output was refused
    /// or its state could not be kept. `outcome` names it (`ended`, `refused`, `state`).
    pub(crate) fn failure(&self, outcome: &str, text: &str) {
        let fields = Fields::from_iter([("outcome".to_owned(), outcome.into())]);
        self.record(Level::Warn, text, fields);
    }
}

impl program::Host for Logs {
    fn log(&mut self, level: Level, text: &str) {
        self.record(level, text, Fields::new());
    }
}

/// The embedder of a control program: its log, the `stats` and `events` views of the
/// observation store, and the validation of a configuration against the current revision.
pub(crate) struct Reader {
    pub(crate) logs: Logs,
    /// What a kernel generation to a target takes on the serving engines' devices.
    pub(crate) measure: Measure,
    /// The leader's epoch records, the epoch that serves among them, which a validation compares
    /// a configuration with the current revision's as the deployment runs them.
    pub(crate) epochs: Arc<crate::epoch::Epochs>,
}

/// Measures what a kernel generation to a target configuration takes on each GPU of the serving
/// engines beside their graphs (`Leader::rooms`): what a validation reports of a change that
/// starts one. A control program steps on a blocking thread, which the measure blocks while the
/// engines answer.
pub(crate) type Measure =
    Arc<dyn Fn(&crate::config::Config) -> Result<Vec<program::Room>, String> + Send + Sync>;

impl program::Host for Reader {
    fn log(&mut self, level: Level, text: &str) {
        self.logs.log(level, text);
    }
}

impl program::Control for Reader {
    fn stats(&mut self, query: &Query) -> f64 {
        stats(&self.logs.store, query)
    }

    fn group(&mut self, group: &Group) -> BTreeMap<String, f64> {
        self::group(&self.logs.store, group)
    }

    fn events(&mut self, filter: &Filter) -> Vec<Record> {
        events(&self.logs.store, filter)
    }

    fn validate(&mut self, config: &program::Config) -> Result<Report, String> {
        let current = Arc::clone(&self.logs.current.borrow());
        validate(&current, &self.epochs, config, Some(&self.measure))
    }
}

/// The value of one aggregate of the `stats` table; NaN when the selection holds no value.
pub(crate) fn stats(store: &Store, query: &Query) -> f64 {
    let (series, span) = select(store, query);
    measure(&series.window(span), query.function).unwrap_or(f64::NAN)
}

/// The aggregate of each value of label `group.key` among the series `group.query` selects; a
/// value whose selection holds no value is left out.
pub(crate) fn group(store: &Store, group: &Group) -> BTreeMap<String, f64> {
    let (series, span) = select(store, &group.query);
    series
        .by(span, &group.key)
        .into_iter()
        .filter_map(|(value, window)| {
            measure(&window, group.query.function).map(|measured| (value, measured))
        })
        .collect()
}

/// The series of `query` and the span of its window.
fn select(store: &Store, query: &Query) -> (crate::observation::Series, Duration) {
    let mut series = store.stats().series(&query.name);
    for (key, value) in &query.labels {
        series = series.label(key, value);
    }
    if let Some(revision) = query.revision {
        series = series.revision(revision);
    }
    let span = match query.window.0 {
        0 => LATEST,
        nanos => Duration::from_nanos(nanos),
    };
    (series, span)
}

/// The aggregate `function` of `window`; `None` when the window holds no value.
fn measure(window: &crate::observation::Window, function: Function) -> Option<f64> {
    match function {
        Function::Quantile(q) => window.quantile(q),
        Function::Mean => window.mean(),
        Function::Min => window.min(),
        Function::Max => window.max(),
        Function::Sum => window.sum(),
        Function::Count => window.count(),
        Function::Rate => window.rate(),
        Function::Last => window.last(),
    }
}

/// The records of the `events` table `filter` selects, oldest first: of its kind, with its
/// labels and fields, within its window, the latest `limit` of them. A kind the table does not
/// have selects nothing.
pub(crate) fn events(store: &Store, filter: &Filter) -> Vec<Record> {
    let Ok(kind) = filter.kind.parse::<Kind>() else {
        return Vec::new();
    };
    let span = filter
        .window
        .map_or(Duration::MAX, |window| Duration::from_nanos(window.0));
    let mut selected: Vec<Event> = store
        .events()
        .kind(kind)
        .window(span)
        .into_iter()
        .filter(|event| {
            filter
                .labels
                .iter()
                .all(|(key, value)| event.labels.get(key) == Some(value))
                && filter
                    .fields
                    .iter()
                    .all(|(key, value)| event.fields.get(key) == Some(value))
        })
        .collect();
    if let Some(limit) = filter.limit {
        let skip = selected.len().saturating_sub(limit as usize);
        selected.drain(..skip);
    }
    selected.into_iter().map(record).collect()
}

/// An event as a program reads it.
fn record(event: Event) -> Record {
    let time = event
        .time
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
        });
    Record {
        time: Instant(time),
        kind: event.kind.as_str().to_owned(),
        labels: event.labels,
        fields: event.fields.into_iter().collect(),
        revision: event.revision,
    }
}

/// What applying `config` would do to the deployment `current` serves, without applying it: the
/// heaviest effect among the options it changes, as an apply compares them with the serving epoch
/// `epochs` records (`Epochs::changes`), and whether serving stops meanwhile; or why the same
/// validation an apply runs refuses it.
///
/// No change interrupts serving: an update starts its new engines beside the serving ones and
/// moves the requests to them, and one whose engines do not fit is refused when it prepares,
/// leaving the serving deployment as it is. With `measure`, a change that starts a kernel
/// generation states what the generation takes of each GPU beside the serving graphs.
pub(crate) fn validate(
    current: &Current,
    epochs: &crate::epoch::Epochs,
    config: &program::Config,
    measure: Option<&Measure>,
) -> Result<Report, String> {
    let target: crate::config::Config = serde_json::from_value(config.0.clone())
        .map_err(|error| format!("the configuration does not read: {error}"))?;
    let package = Package::of(target, current.package.programs.clone())
        .map_err(|error| format!("{error:#}"))?;
    package
        .config
        .validate(&package.stated)
        .map_err(|error| format!("{error:#}"))?;
    package
        .config
        .engine_config()
        .map_err(|error| format!("{error:#}"))?;
    let changes = epochs.changes(Some(&current.package.config), &package.config);
    let effects = proto::reload::Effects::of(&changes);
    // A kernel generation holds both generations' graphs until it commits: what that takes of
    // each GPU, and the KV pages its cache yields when the room lacks it.
    let rooms = match measure {
        Some(measure) if effects.has(proto::reload::Effect::Generation) => {
            measure(&package.config)?
        }
        _ => Vec::new(),
    };
    Ok(Report {
        effect: effects.heaviest().map(|effect| effect.name().to_owned()),
        interrupts: false,
        rooms,
    })
}
