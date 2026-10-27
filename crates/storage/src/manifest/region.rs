//! Index regions of logical tensors: arithmetic progressions of one
//! dimension's indices and their intersections.
//!
//! A region is a box of [`Span`]s, one per dimension. Every distribution the
//! model library describes holds a union of such boxes: an even split is a
//! contiguous span, a position-mod-N share a span of step N, a compressed
//! index's share a span whose step follows from its stride and N
//! (`uniserve.distributed.distribution`). Intersections of spans are spans,
//! so the region two parts of one logical tensor share is again a box.

use serde::{Deserialize, Serialize};

/// The indices `start + step * k` for `k` in `0..count` of one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Span {
    pub start: u64,
    pub count: u64,
    pub step: u64,
}

impl Span {
    /// The contiguous indices `start..start + count`.
    pub fn range(start: u64, count: u64) -> Self {
        Self {
            start,
            count,
            step: 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The `k`-th index.
    pub fn index(&self, k: u64) -> u64 {
        self.start + self.step * k
    }

    /// The last index; `None` for an empty span.
    pub fn last(&self) -> Option<u64> {
        (self.count > 0).then(|| self.index(self.count - 1))
    }

    /// The position `k` of index `index` in the span, if it holds it.
    pub fn position(&self, index: u64) -> Option<u64> {
        if index < self.start || !(index - self.start).is_multiple_of(self.step) {
            return None;
        }
        let position = (index - self.start) / self.step;
        (position < self.count).then_some(position)
    }

    /// The span's positions `first..first + count`, as a span of indices.
    pub fn narrow(&self, first: u64, count: u64) -> Self {
        Self {
            start: self.index(first),
            count,
            step: self.step,
        }
    }

    /// The indices both spans hold: again a progression, of the least
    /// common multiple of their steps.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let (Some(last), Some(other_last)) = (self.last(), other.last()) else {
            return None;
        };
        let (first, period) = crt(self.start, self.step, other.start, other.step)?;
        let low = self.start.max(other.start);
        let high = last.min(other_last);
        if low > high {
            return None;
        }
        // The least index at or above `low` congruent to `first`.
        let start = if low <= first {
            first
        } else {
            low + (period - (low - first) % period) % period
        };
        if start > high {
            return None;
        }
        Some(Self {
            start,
            count: (high - start) / period + 1,
            step: period,
        })
    }

    /// Whether every index of `self` lies in `other`.
    pub fn within(&self, other: &Self) -> bool {
        self.intersect(other)
            .is_some_and(|shared| shared.count == self.count)
            || self.is_empty()
    }
}

/// The greatest common divisor.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The least common multiple; `None` on overflow.
pub fn lcm(a: u64, b: u64) -> Option<u64> {
    if a == 0 || b == 0 {
        return Some(0);
    }
    (a / gcd(a, b)).checked_mul(b)
}

/// The solutions of `x = a (mod m)` and `x = b (mod n)`, as the least
/// nonnegative solution and the period `lcm(m, n)`; `None` when the
/// congruences contradict each other.
fn crt(a: u64, m: u64, b: u64, n: u64) -> Option<(u64, u64)> {
    let divisor = gcd(m, n);
    let (a, b) = (i128::from(a), i128::from(b));
    let (m, n, divisor) = (i128::from(m), i128::from(n), i128::from(divisor));
    if (b - a) % divisor != 0 {
        return None;
    }
    let period = m / divisor * n;
    // x = a + m * t with m * t = b - a (mod n): t = (b - a) / d * inv(m / d)
    // (mod n / d).
    let reduced = n / divisor;
    let t = if reduced == 1 {
        0
    } else {
        ((b - a) / divisor).rem_euclid(reduced) * inverse(m / divisor, reduced)? % reduced
    };
    let solution = (a + m * t).rem_euclid(period);
    Some((u64::try_from(solution).ok()?, u64::try_from(period).ok()?))
}

/// The inverse of `a` modulo `m` (coprime), by the extended Euclid
/// algorithm.
fn inverse(a: i128, m: i128) -> Option<i128> {
    let (mut old_r, mut r) = (a.rem_euclid(m), m);
    let (mut old_s, mut s) = (1_i128, 0_i128);
    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }
    (old_r == 1).then(|| old_s.rem_euclid(m))
}

/// The coarsest division of `target` into progressions that each lie
/// wholly inside or wholly outside each of `spans`: within a progression of
/// a step every span's step divides, membership depends on the bounds
/// alone, so the residues of that step cut at every span's bounds divide
/// it so.
pub fn atoms(target: &Span, spans: &[Span]) -> Vec<Span> {
    if target.is_empty() {
        return Vec::new();
    }
    let relevant: Vec<Span> = spans
        .iter()
        .filter(|span| span.intersect(target).is_some())
        .copied()
        .collect();
    let period = relevant
        .iter()
        .try_fold(target.step, |period, span| lcm(period, span.step))
        .unwrap_or(target.step);
    // Positions of `target` repeat their residue every `classes` positions.
    let classes = (period / target.step).max(1);
    let mut atoms = Vec::new();
    for residue in 0..classes.min(target.count) {
        let class = Span {
            start: target.index(residue),
            count: (target.count - residue).div_ceil(classes),
            step: period,
        };
        // Cut the class at the first and past-the-last positions of every
        // span's share of it.
        let mut cuts = vec![0, class.count];
        for span in &relevant {
            if let Some(shared) = class.intersect(span) {
                let first = class.position(shared.start).unwrap_or(0);
                cuts.push(first);
                cuts.push(first + shared.count);
            }
        }
        cuts.sort_unstable();
        cuts.dedup();
        for pair in cuts.windows(2) {
            atoms.push(class.narrow(pair[0], pair[1] - pair[0]));
        }
    }
    atoms
}

/// The number of elements of a box.
pub fn volume(spans: &[Span]) -> u64 {
    spans.iter().map(|span| span.count).product()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indices(span: &Span) -> Vec<u64> {
        (0..span.count).map(|k| span.index(k)).collect()
    }

    #[test]
    fn intersections_hold_exactly_the_common_indices() {
        let spans = [
            Span::range(0, 100),
            Span::range(37, 20),
            Span {
                start: 3,
                count: 30,
                step: 4,
            },
            Span {
                start: 1,
                count: 25,
                step: 6,
            },
            Span {
                start: 10,
                count: 0,
                step: 1,
            },
        ];
        for a in &spans {
            for b in &spans {
                let expected: Vec<u64> = indices(a)
                    .into_iter()
                    .filter(|index| b.position(*index).is_some())
                    .collect();
                let got = a
                    .intersect(b)
                    .map(|span| indices(&span))
                    .unwrap_or_default();
                assert_eq!(got, expected, "{a:?} and {b:?}");
            }
        }
    }

    #[test]
    fn atoms_partition_the_target_by_membership() {
        let target = Span::range(0, 64);
        let spans = [
            Span::range(10, 20),
            Span {
                start: 1,
                count: 16,
                step: 4,
            },
        ];
        let atoms = atoms(&target, &spans);
        let mut covered: Vec<u64> = atoms.iter().flat_map(indices).collect();
        covered.sort_unstable();
        assert_eq!(covered, indices(&target));
        for atom in &atoms {
            for span in &spans {
                let inside = indices(atom)
                    .iter()
                    .filter(|index| span.position(**index).is_some())
                    .count() as u64;
                assert!(
                    inside == 0 || inside == atom.count,
                    "{atom:?} splits {span:?}"
                );
            }
        }
    }
}
