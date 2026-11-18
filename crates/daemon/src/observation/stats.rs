//! The storage of the `stats` table: every series' values in time buckets
//! of two widths, and its summary of what has aged out of both.
//!
//! A series is one metric name, one label set and one configuration
//! revision. Every write folds into the series' fine bucket and its coarse
//! bucket of the write's time, so each tier is complete on its own. A fine
//! bucket is dropped once older than the fine retention; a coarse bucket
//! older than the coarse retention folds into the series' `retired`
//! summary, which stays until its revision is forgotten.
//!
//! Rows are read as one timeline without overlap, so a sum over the table
//! counts every write once: the retired summary, then the coarse buckets
//! before the boundary `ceil(now - fine.keep, coarse.width)`, then the fine
//! buckets from it on. The fine tier holds every bucket after the boundary,
//! and the retention check of `Limits::new` keeps the retired summary
//! before it.

use std::collections::{HashMap, VecDeque};
use std::fmt;

use super::{Error, Labels, Retention};
use proto::histogram::Histogram;

/// How a metric's writes combine within a time bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Instrument {
    /// Increments: a bucket holds their sum.
    Counter,
    /// Readings of a level: a bucket holds the latest.
    Gauge,
    /// Observations: a bucket holds their distribution.
    Histogram,
}

impl fmt::Display for Instrument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Counter => "counter",
            Self::Gauge => "gauge",
            Self::Histogram => "histogram",
        })
    }
}

/// The value of a series in one bucket.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Value {
    Number(f64),
    Histogram(Histogram),
}

impl Value {
    /// Folds a later value of the same series into this one by the
    /// instrument's rule: counters add, gauges keep the later reading,
    /// histograms merge.
    pub(crate) fn fold(&mut self, later: &Value, instrument: Instrument) {
        match (self, later) {
            (Self::Number(value), Self::Number(later)) => match instrument {
                Instrument::Gauge => *value = *later,
                Instrument::Counter | Instrument::Histogram => *value += *later,
            },
            (Self::Histogram(histogram), Self::Histogram(later)) => histogram.merge(later),
            // A series holds one instrument's values (`Table::record`).
            (this, later) => *this = later.clone(),
        }
    }

    /// The row's `value` column: the number, or a histogram's mean
    /// observation.
    pub(crate) fn number(&self) -> f64 {
        match self {
            Self::Number(value) => *value,
            Self::Histogram(histogram) => histogram.mean().unwrap_or(0.0),
        }
    }

    pub(crate) fn histogram(&self) -> Option<&Histogram> {
        match self {
            Self::Number(_) => None,
            Self::Histogram(histogram) => Some(histogram),
        }
    }
}

/// The retention in nanoseconds, checked to make the timeline exact.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub(crate) fine_width: i64,
    pub(crate) fine_keep: i64,
    pub(crate) coarse_width: i64,
    pub(crate) coarse_keep: i64,
    pub(crate) events_keep: i64,
    pub(crate) capacity: usize,
}

impl Limits {
    pub(crate) fn new(retention: &Retention) -> Result<Self, Error> {
        let nanos =
            |duration: std::time::Duration| i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX);
        let limits = Self {
            fine_width: nanos(retention.fine.width),
            fine_keep: nanos(retention.fine.keep),
            coarse_width: nanos(retention.coarse.width),
            coarse_keep: nanos(retention.coarse.keep),
            events_keep: nanos(retention.events),
            capacity: retention.capacity,
        };

        if limits.fine_width <= 0 || limits.coarse_width <= 0 {
            return Err(Error::Retention("bucket widths must be positive"));
        }
        if limits.coarse_width % limits.fine_width != 0 {
            return Err(Error::Retention(
                "the coarse width must be a multiple of the fine width",
            ));
        }
        // A coarse bucket retires only after the fine tier has stopped
        // showing any of its time (see the module documentation).
        if limits.coarse_keep < limits.fine_keep.saturating_add(limits.coarse_width) {
            return Err(Error::Retention(
                "the coarse retention must cover the fine retention and one coarse bucket",
            ));
        }
        if limits.capacity == 0 {
            return Err(Error::Retention("the event capacity must be positive"));
        }

        Ok(limits)
    }

    /// The first time the fine tier shows; coarse buckets show before it.
    fn boundary(&self, now: i64) -> i64 {
        ceil(now.saturating_sub(self.fine_keep), self.coarse_width)
    }
}

/// The value of a series in the bucket starting at `start` (nanoseconds
/// since the Unix epoch).
#[derive(Clone, Debug)]
pub(crate) struct Bucket {
    pub(crate) start: i64,
    pub(crate) value: Value,
}

/// The buckets of one series.
#[derive(Debug, Default)]
struct Data {
    fine: VecDeque<Bucket>,
    coarse: VecDeque<Bucket>,

    /// The fold of every coarse bucket past its retention, starting at the
    /// first of them.
    retired: Option<Bucket>,
}

impl Data {
    fn record(&mut self, now: i64, value: &Value, instrument: Instrument, limits: &Limits) {
        insert(
            &mut self.fine,
            floor(now, limits.fine_width),
            value,
            instrument,
        );
        insert(
            &mut self.coarse,
            floor(now, limits.coarse_width),
            value,
            instrument,
        );
    }

    fn prune(&mut self, now: i64, instrument: Instrument, limits: &Limits) {
        let cutoff = now.saturating_sub(limits.fine_keep);
        while self
            .fine
            .front()
            .is_some_and(|bucket| bucket.start < cutoff)
        {
            self.fine.pop_front();
        }

        let cutoff = now.saturating_sub(limits.coarse_keep);
        while let Some(bucket) = self.coarse.pop_front_if(|bucket| bucket.start < cutoff) {
            match &mut self.retired {
                Some(retired) => retired.value.fold(&bucket.value, instrument),
                None => self.retired = Some(bucket),
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.fine.is_empty() && self.coarse.is_empty() && self.retired.is_none()
    }

    /// The series' rows in time order, without overlap.
    fn rows(&self, now: i64, limits: &Limits) -> impl Iterator<Item = &Bucket> {
        let boundary = limits.boundary(now);

        self.retired
            .iter()
            .chain(
                self.coarse
                    .iter()
                    .take_while(move |bucket| bucket.start < boundary),
            )
            .chain(
                self.fine
                    .iter()
                    .skip_while(move |bucket| bucket.start < boundary),
            )
    }
}

/// Folds `value` into the bucket starting at `start`, kept in time order. A
/// write normally lands in the newest bucket; an earlier time (the wall
/// clock stepped back) finds its bucket by search.
fn insert(buckets: &mut VecDeque<Bucket>, start: i64, value: &Value, instrument: Instrument) {
    if let Some(last) = buckets.back_mut() {
        if last.start == start {
            last.value.fold(value, instrument);
            return;
        }
        if last.start > start {
            match buckets.binary_search_by_key(&start, |bucket| bucket.start) {
                Ok(at) => buckets[at].value.fold(value, instrument),
                Err(at) => buckets.insert(
                    at,
                    Bucket {
                        start,
                        value: value.clone(),
                    },
                ),
            }
            return;
        }
    }

    buckets.push_back(Bucket {
        start,
        value: value.clone(),
    });
}

fn floor(time: i64, width: i64) -> i64 {
    time - time.rem_euclid(width)
}

fn ceil(time: i64, width: i64) -> i64 {
    let rem = time.rem_euclid(width);
    if rem == 0 { time } else { time - rem + width }
}

/// The series of one metric name.
#[derive(Debug)]
struct Metric {
    instrument: Instrument,
    series: HashMap<Labels, HashMap<u64, Data>>,
}

/// One fine bucket of one series, owned, as an export reads it ([`Table::fine`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Point {
    pub(crate) name: String,
    pub(crate) labels: Labels,
    pub(crate) revision: u64,
    pub(crate) instrument: Instrument,
    /// The bucket's start, in nanoseconds since the Unix epoch.
    pub(crate) start: i64,
    pub(crate) value: Value,
}

/// One row of the `stats` table, borrowed from the storage.
pub(crate) struct Row<'a> {
    pub(crate) name: &'a str,
    pub(crate) labels: &'a Labels,
    pub(crate) revision: u64,
    pub(crate) bucket: &'a Bucket,
}

/// Every series of the `stats` table, by name, label set and revision.
#[derive(Debug)]
pub(crate) struct Table {
    limits: Limits,
    metrics: HashMap<String, Metric>,
}

impl Table {
    pub(crate) fn new(limits: Limits) -> Self {
        Self {
            limits,
            metrics: HashMap::new(),
        }
    }

    /// Folds `value` into the series' buckets of time `now`.
    ///
    /// A name keeps the instrument of its first write; a write by another
    /// instrument is rejected.
    pub(crate) fn record(
        &mut self,
        name: &str,
        labels: &Labels,
        revision: u64,
        instrument: Instrument,
        value: &Value,
        now: i64,
    ) -> Result<(), Error> {
        let metric = match self.metrics.get_mut(name) {
            Some(metric) => metric,
            None => self.metrics.entry(name.to_owned()).or_insert(Metric {
                instrument,
                series: HashMap::new(),
            }),
        };
        if metric.instrument != instrument {
            return Err(Error::Instrument {
                name: name.to_owned(),
                existing: metric.instrument,
                requested: instrument,
            });
        }

        let revisions = match metric.series.get_mut(labels) {
            Some(revisions) => revisions,
            None => metric.series.entry(labels.clone()).or_default(),
        };
        let data = revisions.entry(revision).or_default();
        data.record(now, value, instrument, &self.limits);
        data.prune(now, instrument, &self.limits);

        Ok(())
    }

    /// Ages every series to time `now`.
    pub(crate) fn prune(&mut self, now: i64) {
        let limits = &self.limits;
        for metric in self.metrics.values_mut() {
            let instrument = metric.instrument;
            for revisions in metric.series.values_mut() {
                for data in revisions.values_mut() {
                    data.prune(now, instrument, limits);
                }
            }
        }
    }

    /// Drops the retired summaries of `revision`, and the series left
    /// without rows.
    pub(crate) fn forget(&mut self, revision: u64) {
        self.metrics.retain(|_, metric| {
            metric.series.retain(|_, revisions| {
                if let Some(data) = revisions.get_mut(&revision) {
                    data.retired = None;
                    if data.is_empty() {
                        revisions.remove(&revision);
                    }
                }
                !revisions.is_empty()
            });
            !metric.series.is_empty()
        });
    }

    /// The fine buckets that start at or after `from` and end at or before `to` (nanoseconds
    /// since the Unix epoch), of every series, owned: what an export of the window sends.
    pub(crate) fn fine(&self, from: i64, to: i64) -> Vec<Point> {
        let width = self.limits.fine_width;
        let mut points = Vec::new();
        for (name, metric) in &self.metrics {
            for (labels, revisions) in &metric.series {
                for (&revision, data) in revisions {
                    let window = data
                        .fine
                        .iter()
                        .filter(|bucket| bucket.start >= from && bucket.start + width <= to);
                    for bucket in window {
                        points.push(Point {
                            name: name.clone(),
                            labels: labels.clone(),
                            revision,
                            instrument: metric.instrument,
                            start: bucket.start,
                            value: bucket.value.clone(),
                        });
                    }
                }
            }
        }
        points
    }

    /// The instrument of a metric name, if it has been written.
    pub(crate) fn instrument(&self, name: &str) -> Option<Instrument> {
        self.metrics.get(name).map(|metric| metric.instrument)
    }

    /// Visits the rows of the metrics `names` selects (every metric when
    /// `None`) as of time `now`, each series' rows in time order. The
    /// caller prunes first.
    pub(crate) fn rows<'a>(
        &'a self,
        names: Option<&[String]>,
        now: i64,
        mut visit: impl FnMut(Row<'a>),
    ) {
        let limits = &self.limits;
        let mut each = |name: &'a str, metric: &'a Metric| {
            for (labels, revisions) in &metric.series {
                for (&revision, data) in revisions {
                    for bucket in data.rows(now, limits) {
                        visit(Row {
                            name,
                            labels,
                            revision,
                            bucket,
                        });
                    }
                }
            }
        };

        match names {
            Some(names) => {
                for name in names {
                    if let Some((name, entry)) = self.metrics.get_key_value(name.as_str()) {
                        each(name, entry);
                    }
                }
            }
            None => {
                for (name, entry) in &self.metrics {
                    each(name, entry);
                }
            }
        }
    }
}
