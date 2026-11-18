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

/// A histogram's buckets as OpenTelemetry's exponential histogram states them at scale 3
/// (base `2^(1/8)`): the zeros' count, and for each sign the index of its first bucket and the
/// counts of consecutive buckets from it, the bucket of index `i` holding magnitudes in
/// `(2^(i/8), 2^((i+1)/8)]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Exponential {
    pub zero_count: u64,
    pub positive: (i32, Vec<u64>),
    pub negative: (i32, Vec<u64>),
}

/// The OpenTelemetry exponential histogram scale of the shared bucket scheme: `SCALE` buckets
/// per power of two is `2^3`.
pub const EXPONENTIAL_SCALE: i32 = 3;

impl Histogram {
    /// The buckets as OpenTelemetry's exponential histogram at [`EXPONENTIAL_SCALE`] states them.
    pub fn exponential(&self) -> Exponential {
        let mut zero_count = 0;
        let mut positive = std::collections::BTreeMap::new();
        let mut negative = std::collections::BTreeMap::new();
        for (&key, &count) in &self.buckets {
            match key {
                0 => zero_count += count,
                key if key > 0 => {
                    positive.insert(key - OFFSET, count);
                }
                key => {
                    negative.insert(-key - OFFSET, count);
                }
            }
        }
        Exponential {
            zero_count,
            positive: dense(&positive),
            negative: dense(&negative),
        }
    }
}

/// Counts by bucket index as the index of the first and the counts of consecutive buckets from
/// it, empty buckets between them zero.
fn dense(counts: &BTreeMap<i64, u64>) -> (i32, Vec<u64>) {
    let (Some((&first, _)), Some((&last, _))) = (counts.first_key_value(), counts.last_key_value())
    else {
        return (0, Vec::new());
    };
    let mut dense = vec![0; usize::try_from(last - first + 1).unwrap_or(0)];
    for (&index, &count) in counts {
        dense[usize::try_from(index - first).unwrap_or(0)] = count;
    }
    // Indices of finite values lie within about ±8700 (`OFFSET`).
    (i32::try_from(first).unwrap_or(i32::MIN), dense)
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

    /// The exponential view states each bucket at OpenTelemetry's index for scale 3: 1.0 lies in
    /// `(2^(-1/8), 2^0]`, index -1, and 2.0 in `(2^(7/8), 2^1]`, index 7.
    #[test]
    fn the_exponential_view_is_opentelemetrys_scale_three() {
        let mut histogram = Histogram::new();
        for value in [1.0, 2.0, 2.0, 0.0, -1.0] {
            histogram.observe(value);
        }
        let view = histogram.exponential();
        assert_eq!(view.zero_count, 1);
        let (offset, counts) = &view.positive;
        assert_eq!(*offset, -1);
        assert_eq!(counts.len(), 9);
        assert_eq!((counts[0], counts[8]), (1, 2));
        assert_eq!(counts.iter().sum::<u64>(), 3);
        assert_eq!(view.negative, (-1, vec![1]));
        let base = 2f64.powf(2f64.powi(-EXPONENTIAL_SCALE));
        assert!((base.powi(8) - 2.0).abs() < 1e-12);
    }
}
