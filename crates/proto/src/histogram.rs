//! A distribution of observed values, as the observation store keeps a histogram stat and as an
//! engine reports one (the fence probes' step durations, `report::ForwardStats::fence_seconds`).
//!
//! Every histogram shares one bucket scheme, so any two merge without
//! agreeing on bounds first: a bucket is an interval between consecutive
//! powers of `2^(1/8)` (the exponential histogram of OpenTelemetry at scale
//! 3). A bucket is about 9% wide, so a quantile read from the buckets is
//! within about 9% of the exact one; the exact minimum and maximum bound
//! every estimate.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Buckets per power of two.
const SCALE: f64 = 8.0;

/// The distance of the positive bucket keys from zero. Bucket indices of
/// finite `f64` values lie within about ±8700, so every positive key is
/// above zero and every negative key below it, and keys order as the values
/// of their buckets do.
const OFFSET: i64 = 1 << 20;

/// A distribution of observed values: their count, sum, extremes and the
/// count of each nonempty bucket.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Histogram {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,

    /// Observation count by bucket key: `0` holds the zeros; key
    /// `OFFSET + i` holds positive values in `(2^(i/8), 2^((i+1)/8)]`, and
    /// key `-(OFFSET + i)` the negative values in `[-2^((i+1)/8), -2^(i/8))`.
    buckets: BTreeMap<i64, u64>,
}

impl Histogram {
    /// An empty histogram.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one observation. Non-finite values have no bucket and are
    /// not recorded; the store rejects them before they reach a histogram.
    pub fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }

        self.count += 1;
        self.sum += value;
        *self.buckets.entry(key(value)).or_insert(0) += 1;
    }

    /// Adds every observation of `other` to this histogram.
    pub fn merge(&mut self, other: &Histogram) {
        if other.count == 0 {
            return;
        }

        if self.count == 0 {
            self.min = other.min;
            self.max = other.max;
        } else {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }

        self.count += other.count;
        self.sum += other.sum;
        for (&key, &count) in &other.buckets {
            *self.buckets.entry(key).or_insert(0) += count;
        }
    }

    /// The number of observations.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The sum of the observations.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// The smallest observation; `None` when empty.
    pub fn min(&self) -> Option<f64> {
        (self.count > 0).then_some(self.min)
    }

    /// The largest observation; `None` when empty.
    pub fn max(&self) -> Option<f64> {
        (self.count > 0).then_some(self.max)
    }

    /// The mean observation; `None` when empty.
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum / self.count as f64)
    }

    /// The `q` quantile (`q` in `[0, 1]`, clamped) estimated from the
    /// buckets: the rank `q * count` is located in its bucket and the value
    /// interpolated linearly across the bucket's interval, then clamped to
    /// the observed extremes. `None` when empty.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }

        let q = if q.is_nan() { 0.5 } else { q.clamp(0.0, 1.0) };
        if q == 0.0 {
            return Some(self.min);
        }

        let rank = q * self.count as f64;
        let mut below = 0.0;
        for (&key, &count) in &self.buckets {
            let count = count as f64;
            if below + count >= rank {
                let (lower, upper) = bounds(key);
                let estimate = lower + (upper - lower) * (rank - below) / count;
                return Some(estimate.clamp(self.min, self.max));
            }
            below += count;
        }

        Some(self.max)
    }

    /// The nonempty buckets in ascending order of their values, each as
    /// `(lower, upper, count)`: the bucket's interval and its observations.
    pub fn buckets(&self) -> impl Iterator<Item = (f64, f64, u64)> + '_ {
        self.buckets.iter().map(|(&key, &count)| {
            let (lower, upper) = bounds(key);
            (lower, upper, count)
        })
    }
}

impl From<&Histogram> for crate::request::wire::Histogram {
    fn from(histogram: &Histogram) -> Self {
        Self {
            count: histogram.count,
            sum: histogram.sum,
            min: histogram.min,
            max: histogram.max,
            buckets: histogram.buckets.clone().into_iter().collect(),
        }
    }
}

impl From<crate::request::wire::Histogram> for Histogram {
    fn from(histogram: crate::request::wire::Histogram) -> Self {
        Self {
            count: histogram.count,
            sum: histogram.sum,
            min: histogram.min,
            max: histogram.max,
            buckets: histogram.buckets.into_iter().collect(),
        }
    }
}

/// The index `i` of the positive bucket `(2^(i/8), 2^((i+1)/8)]` holding
/// `magnitude` (> 0).
fn index(magnitude: f64) -> i64 {
    // `log2` of a finite positive `f64` lies in [-1075, 1024], so the
    // product fits an `i64` exactly.
    (magnitude.log2() * SCALE).ceil() as i64 - 1
}

/// The bucket key of a finite value.
fn key(value: f64) -> i64 {
    if value > 0.0 {
        OFFSET + index(value)
    } else if value < 0.0 {
        -(OFFSET + index(-value))
    } else {
        0
    }
}

/// The interval `(lower, upper)` of the bucket a key names.
fn bounds(key: i64) -> (f64, f64) {
    let power = |index: i64| (index as f64 / SCALE).exp2();

    if key > 0 {
        let index = key - OFFSET;
        (power(index), power(index + 1))
    } else if key < 0 {
        let index = -key - OFFSET;
        (-power(index + 1), -power(index))
    } else {
        (0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_fall_within_bucket_width_of_exact_values() {
        let mut histogram = Histogram::new();
        for millis in 1..=1000 {
            histogram.observe(f64::from(millis) / 1000.0);
        }

        for (q, exact) in [(0.5, 0.5), (0.9, 0.9), (0.99, 0.99)] {
            let estimate = histogram.quantile(q).unwrap();
            assert!((estimate - exact).abs() / exact < 0.09, "q={q}: {estimate}");
        }
        assert_eq!(histogram.quantile(0.0), Some(0.001));
        assert_eq!(histogram.quantile(1.0), Some(1.0));
        assert_eq!(histogram.count(), 1000);
    }

    #[test]
    fn merge_equals_observing_both_sets() {
        let mut left = Histogram::new();
        let mut right = Histogram::new();
        let mut both = Histogram::new();
        for value in [-3.0, 0.0, 0.25, 7.5] {
            left.observe(value);
            both.observe(value);
        }
        for value in [-0.5, 2.0, 1e6] {
            right.observe(value);
            both.observe(value);
        }

        left.merge(&right);
        assert_eq!(left, both);
        assert_eq!(left.min(), Some(-3.0));
        assert_eq!(left.max(), Some(1e6));

        // Buckets ascend with their values, negatives first.
        let lowers: Vec<f64> = left.buckets().map(|(lower, _, _)| lower).collect();
        assert!(lowers.is_sorted());
        assert!(left.buckets().all(|(lower, upper, _)| lower <= upper));
    }
}
