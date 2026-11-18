//! The read-only views of the two tables that control programs read
//! (`State::stats`, `State::events`), and that the program host's
//! `uniserve.stats` and `uniserve.events` functions answer from.
//!
//! The views are lazy: a `Series` or `Events` is a selection, and only
//! `Series::window`, `Events::last` and `Events::window` read the store, and
//! only the rows they select.

use std::collections::BTreeMap;
use std::time::Duration;

use super::events::{Event, Kind};
use super::stats::{Instrument, Value};
use super::{Store, WRITER, nanos};
use proto::histogram::Histogram;

/// The `stats` table, by metric name.
#[derive(Clone, Debug)]
pub struct Stats {
    store: Store,
}

impl Stats {
    pub(crate) fn new(store: Store) -> Self {
        Self { store }
    }

    /// Every series of the metric `name`, across label sets and revisions.
    pub fn series(&self, name: &str) -> Series {
        Series {
            store: self.store.clone(),
            name: name.to_owned(),
            labels: Vec::new(),
            revision: None,
        }
    }
}

/// The series of one metric that a selection of labels and a revision
/// admit.
#[derive(Clone, Debug)]
pub struct Series {
    store: Store,
    name: String,
    labels: Vec<(String, String)>,
    revision: Option<u64>,
}

impl Series {
    /// Keeps the series whose label `key` is `value`.
    pub fn label(mut self, key: &str, value: &str) -> Self {
        self.labels.push((key.to_owned(), value.to_owned()));
        self
    }

    /// Keeps the series recorded under configuration revision `revision`.
    pub fn revision(mut self, revision: u64) -> Self {
        self.revision = Some(revision);
        self
    }

    /// The selected series over the last `span`: the buckets of the `stats`
    /// timeline that start within it.
    ///
    /// The series combine bucket by bucket: counters and gauges add (a
    /// gauge over several label sets is their total, so a gauge that does
    /// not add up is selected down to one label set), histograms merge.
    pub fn window(&self, span: Duration) -> Window {
        let (instrument, mut windows) = self.fold(span, None);
        windows
            .remove("")
            .unwrap_or_else(|| Window::empty(instrument, span))
    }

    /// The selected series over the last `span` split by the values of
    /// their label `key`: one window per value, each combining the series
    /// that carry it as [`Self::window`] combines a selection. A series
    /// without the label is in none, and a value is present only when a
    /// series carrying it has a bucket in the window.
    pub fn by(&self, span: Duration, key: &str) -> BTreeMap<String, Window> {
        self.fold(span, Some(key)).1
    }

    /// One pass over the selected rows of the last `span`, combining each
    /// row into the window of its value of label `key`, or into the one
    /// window `""` when no key splits them.
    fn fold(
        &self,
        span: Duration,
        key: Option<&str>,
    ) -> (Option<Instrument>, BTreeMap<String, Window>) {
        let now = nanos(self.store.now());
        let start = now.saturating_sub(i64::try_from(span.as_nanos()).unwrap_or(i64::MAX));

        let mut instrument = None;
        let mut windows = BTreeMap::new();
        self.store.read_stats(now, |table| {
            instrument = table.instrument(&self.name);
            let Some(instrument) = instrument else {
                return;
            };

            let mut folds: BTreeMap<String, Fold<'_>> = BTreeMap::new();
            let names = [self.name.clone()];
            table.rows(Some(&names), now, |row| {
                if row.bucket.start < start {
                    return;
                }
                if self
                    .revision
                    .is_some_and(|revision| revision != row.revision)
                {
                    return;
                }
                if !self
                    .labels
                    .iter()
                    .all(|(key, value)| row.labels.get(key) == Some(value))
                {
                    return;
                }
                let group = match key {
                    None => String::new(),
                    Some(key) => match row.labels.get(key) {
                        Some(value) => value.clone(),
                        None => return,
                    },
                };

                folds
                    .entry(group)
                    .or_insert_with(|| Fold {
                        window: Window::empty(Some(instrument), span),
                        current: None,
                    })
                    .add(&row, instrument);
            });
            windows = folds
                .into_iter()
                .map(|(group, fold)| (group, fold.finish()))
                .collect();
        });

        (instrument, windows)
    }
}

/// One window being combined from rows, with the newest row of the series
/// being visited: rows of a series arrive together and in time order, so a
/// change of series ends the previous one's newest reading.
struct Fold<'a> {
    window: Window,
    current: Option<(&'a super::Labels, u64, f64)>,
}

impl<'a> Fold<'a> {
    fn add(&mut self, row: &super::stats::Row<'a>, instrument: Instrument) {
        match self.current {
            Some((labels, revision, _))
                if std::ptr::eq(labels, row.labels) && revision == row.revision => {}
            Some((_, _, last)) => self.window.latest += last,
            None => {}
        }
        self.current = Some((row.labels, row.revision, row.bucket.value.number()));

        let fold = match instrument {
            // Across series a gauge adds; within one, `Value::fold`
            // would keep the later reading.
            Instrument::Gauge => Instrument::Counter,
            instrument => instrument,
        };
        match self.window.buckets.get_mut(&row.bucket.start) {
            Some(value) => value.fold(&row.bucket.value, fold),
            None => {
                self.window
                    .buckets
                    .insert(row.bucket.start, row.bucket.value.clone());
            }
        }
    }

    fn finish(mut self) -> Window {
        if let Some((_, _, last)) = self.current {
            self.window.latest += last;
        }
        self.window
    }
}

/// The selected series over one window, combined bucket by bucket. Every
/// reading is `None` when the window holds no row.
#[derive(Clone, Debug)]
pub struct Window {
    instrument: Option<Instrument>,
    span: Duration,

    /// The combined value of each bucket, by start time.
    buckets: BTreeMap<i64, Value>,

    /// The sum over the series of each one's newest reading: the current
    /// total of a gauge.
    latest: f64,
}

impl Window {
    /// The window of no row over `span` of a metric of `instrument`.
    fn empty(instrument: Option<Instrument>, span: Duration) -> Self {
        Self {
            instrument,
            span,
            buckets: BTreeMap::new(),
            latest: 0.0,
        }
    }

    fn numbers(&self) -> impl Iterator<Item = f64> + '_ {
        self.buckets.values().map(Value::number)
    }

    /// The distribution of every observation in the window.
    fn histogram(&self) -> Histogram {
        let mut merged = Histogram::new();
        for histogram in self.buckets.values().filter_map(Value::histogram) {
            merged.merge(histogram);
        }

        merged
    }

    /// The sum: a counter's increase, a histogram's sum of observations, or
    /// a gauge's sum of bucket readings.
    pub fn sum(&self) -> Option<f64> {
        match self.instrument? {
            _ if self.buckets.is_empty() => None,
            Instrument::Histogram => Some(self.histogram().sum()),
            Instrument::Counter | Instrument::Gauge => Some(self.numbers().sum()),
        }
    }

    /// The mean: a histogram's mean observation, or the mean bucket value of
    /// a counter (increase per bucket) or gauge.
    pub fn mean(&self) -> Option<f64> {
        match self.instrument? {
            _ if self.buckets.is_empty() => None,
            Instrument::Histogram => self.histogram().mean(),
            Instrument::Counter | Instrument::Gauge => {
                Some(self.numbers().sum::<f64>() / self.buckets.len() as f64)
            }
        }
    }

    /// The `q` quantile (`q` in `[0, 1]`): of a histogram's observations,
    /// estimated from its buckets; of a counter's or gauge's bucket values,
    /// interpolated linearly between the nearest ranks as SQL's
    /// `percentile_cont` does.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        match self.instrument? {
            _ if self.buckets.is_empty() => None,
            Instrument::Histogram => self.histogram().quantile(q),
            Instrument::Counter | Instrument::Gauge => {
                let mut values: Vec<f64> = self.numbers().collect();
                values.sort_by(f64::total_cmp);

                let rank = q.clamp(0.0, 1.0) * (values.len() - 1) as f64;
                let (below, above) = (rank.floor() as usize, rank.ceil() as usize);
                Some(values[below] + (values[above] - values[below]) * (rank - rank.floor()))
            }
        }
    }

    /// The smallest value: a histogram's smallest observation, or the smallest bucket value of
    /// a counter or gauge.
    pub fn min(&self) -> Option<f64> {
        match self.instrument? {
            _ if self.buckets.is_empty() => None,
            Instrument::Histogram => self.histogram().min(),
            Instrument::Counter | Instrument::Gauge => self.numbers().reduce(f64::min),
        }
    }

    /// The largest value: a histogram's largest observation, or the largest bucket value of a
    /// counter or gauge.
    pub fn max(&self) -> Option<f64> {
        match self.instrument? {
            _ if self.buckets.is_empty() => None,
            Instrument::Histogram => self.histogram().max(),
            Instrument::Counter | Instrument::Gauge => self.numbers().reduce(f64::max),
        }
    }

    /// The number of values: a histogram's observations, or the buckets of a counter or gauge
    /// that hold a value.
    pub fn count(&self) -> Option<f64> {
        match self.instrument? {
            _ if self.buckets.is_empty() => None,
            Instrument::Histogram => Some(self.histogram().count() as f64),
            Instrument::Counter | Instrument::Gauge => Some(self.buckets.len() as f64),
        }
    }

    /// The median.
    pub fn p50(&self) -> Option<f64> {
        self.quantile(0.5)
    }

    /// The 99th percentile.
    pub fn p99(&self) -> Option<f64> {
        self.quantile(0.99)
    }

    /// The change per second: a counter's increase or a histogram's
    /// observation count over the window's span; a gauge's change from its
    /// first to its last bucket over the time between them (`None` with
    /// fewer than two buckets).
    pub fn rate(&self) -> Option<f64> {
        let seconds = self.span.as_secs_f64();

        match self.instrument? {
            _ if self.buckets.is_empty() => None,
            Instrument::Counter => Some(self.numbers().sum::<f64>() / seconds),
            Instrument::Histogram => Some(self.histogram().count() as f64 / seconds),
            Instrument::Gauge => {
                let (first, first_value) = self.buckets.first_key_value()?;
                let (last, last_value) = self.buckets.last_key_value()?;
                (last > first).then(|| {
                    (last_value.number() - first_value.number()) / ((last - first) as f64 / 1e9)
                })
            }
        }
    }

    /// The latest value: a gauge's current total over the series (each
    /// series' newest reading); the newest bucket's increase of a counter,
    /// or mean observation of a histogram.
    pub fn last(&self) -> Option<f64> {
        match self.instrument? {
            _ if self.buckets.is_empty() => None,
            Instrument::Gauge => Some(self.latest),
            Instrument::Counter | Instrument::Histogram => self
                .buckets
                .last_key_value()
                .map(|(_, value)| value.number()),
        }
    }
}

/// The events a selection of kind and writer admits.
#[derive(Clone, Debug)]
pub struct Events {
    store: Store,
    kind: Option<Kind>,
    writer: Option<String>,
}

impl Events {
    pub(crate) fn new(store: Store) -> Self {
        Self {
            store,
            kind: None,
            writer: None,
        }
    }

    /// Keeps the events of `kind`.
    pub fn kind(mut self, kind: Kind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Keeps the events whose writer (the `writer` label) is `writer`: the
    /// start, the operator, the Dynamo sidecar or a program.
    pub fn writer(mut self, writer: &str) -> Self {
        self.writer = Some(writer.to_owned());
        self
    }

    fn admits(&self, event: &Event) -> bool {
        self.kind.is_none_or(|kind| kind == event.kind)
            && self
                .writer
                .as_ref()
                .is_none_or(|writer| event.labels.get(WRITER) == Some(writer))
    }

    /// The newest selected event.
    pub fn last(&self) -> Option<Event> {
        self.store.read_events(|table| {
            table
                .after(0)
                .rev()
                .find(|event| self.admits(event))
                .cloned()
        })
    }

    /// The selected events recorded within the last `span`, oldest first.
    pub fn window(&self, span: Duration) -> Vec<Event> {
        let start = self.store.now().checked_sub(span);

        self.store.read_events(|table| {
            table
                .after(0)
                .filter(|event| start.is_none_or(|start| event.time >= start))
                .filter(|event| self.admits(event))
                .cloned()
                .collect()
        })
    }
}
