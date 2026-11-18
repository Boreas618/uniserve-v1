//! Time on the wire.
//!
//! The scheduling mechanism measures time with one monotonic clock per process; a scheduling
//! policy only compares instants and durations, never reads a clock of its own (a policy that did
//! would make the same decisions untestable and would let a randomized policy diverge from its
//! package's seed). [`Nanos`] and [`Duration`] are nanoseconds of that clock since the process
//! started.
//!
//! A control program reads the deployment's clock instead, [`Instant`]: its state and the
//! observations it reads outlive any one process, so their instants are nanoseconds since the
//! Unix epoch. A program never reads that clock either; the host hands it the step's instant.

use std::fmt;
use std::ops::{Add, Sub};

use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize};

/// An instant on the mechanism's monotonic clock, in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Nanos(pub u64);

/// A span of nanoseconds.
///
/// On the wire it is the count of nanoseconds. A package's arguments may also state it as text,
/// a decimal number and a unit (`ns`, `us`, `ms`, `s`, `m`, `h`), such as `"50ms"`, `"1.5s"` or
/// `"10m"`, as `config.toml` writes a program's `period` and its `args`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Duration(pub u64);

impl Duration {
    /// The span in seconds, as observed durations (`uniserve_tpot_seconds`) are stated.
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1e9
    }
}

/// `n` milliseconds.
pub const fn millis(n: u64) -> Duration {
    Duration(n.saturating_mul(1_000_000))
}

/// `n` seconds.
pub const fn secs(n: u64) -> Duration {
    Duration(n.saturating_mul(1_000_000_000))
}

/// `n` minutes.
pub const fn mins(n: u64) -> Duration {
    secs(n.saturating_mul(60))
}

/// `n` hours.
pub const fn hours(n: u64) -> Duration {
    mins(n.saturating_mul(60))
}

impl Add for Duration {
    type Output = Duration;

    fn add(self, other: Duration) -> Duration {
        Duration(self.0.saturating_add(other.0))
    }
}

/// Parses the text form of a duration: a non-negative decimal number followed by its unit.
fn parse(text: &str) -> Result<Duration, String> {
    let text = text.trim();
    let split = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .ok_or_else(|| format!("`{text}` names no unit (ns, us, ms, s, m, h)"))?;
    let (number, unit) = text.split_at(split);

    let scale: f64 = match unit.trim() {
        "ns" => 1.0,
        "us" => 1e3,
        "ms" => 1e6,
        "s" => 1e9,
        "m" => 60e9,
        "h" => 3600e9,
        other => {
            return Err(format!(
                "`{other}` is not a unit of time (ns, us, ms, s, m, h)"
            ));
        }
    };
    let number: f64 = number
        .parse()
        .map_err(|_| format!("`{number}` is not a number of {unit}"))?;

    // The product is finite and non-negative: the number has no sign and the scale is finite.
    let nanos = (number * scale).round();
    if nanos > u64::MAX as f64 {
        return Err(format!("`{text}` is longer than {} ns", u64::MAX));
    }
    Ok(Duration(nanos as u64))
}

impl<'de> Deserialize<'de> for Duration {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Span;

        impl Visitor<'_> for Span {
            type Value = Duration;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("nanoseconds, or a number and a unit such as \"50ms\"")
            }

            fn visit_u64<E: de::Error>(self, nanos: u64) -> Result<Duration, E> {
                Ok(Duration(nanos))
            }

            fn visit_i64<E: de::Error>(self, nanos: i64) -> Result<Duration, E> {
                u64::try_from(nanos)
                    .map(Duration)
                    .map_err(|_| E::custom(format!("a duration of {nanos} ns is negative")))
            }

            fn visit_str<E: de::Error>(self, text: &str) -> Result<Duration, E> {
                parse(text).map_err(E::custom)
            }
        }

        deserializer.deserialize_any(Span)
    }
}

/// An instant on the deployment's clock: nanoseconds since the Unix epoch.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Instant(pub u64);

impl Add<Duration> for Instant {
    type Output = Instant;

    fn add(self, span: Duration) -> Instant {
        Instant(self.0.saturating_add(span.0))
    }
}

impl Sub<Duration> for Instant {
    type Output = Instant;

    fn sub(self, span: Duration) -> Instant {
        Instant(self.0.saturating_sub(span.0))
    }
}

impl Sub for Instant {
    type Output = Duration;

    /// The span from `earlier` to `self`, zero when `earlier` is later.
    fn sub(self, earlier: Instant) -> Duration {
        Duration(self.0.saturating_sub(earlier.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A package states durations as text or nanoseconds, and both decode to the same span.
    #[test]
    fn durations_decode_from_text_and_nanoseconds() {
        let decode = |json: &str| serde_json::from_str::<Duration>(json);

        assert_eq!(decode("\"50ms\"").unwrap(), millis(50));
        assert_eq!(decode("\"1.5s\"").unwrap(), millis(1500));
        assert_eq!(decode("\"10m\"").unwrap(), mins(10));
        assert_eq!(decode("\"2h\"").unwrap(), hours(2));
        assert_eq!(decode("\"250us\"").unwrap(), Duration(250_000));
        assert_eq!(decode("\"7ns\"").unwrap(), Duration(7));
        assert_eq!(decode("9500000").unwrap(), Duration(9_500_000));

        assert!(decode("\"10\"").is_err());
        assert!(decode("\"10 parsecs\"").is_err());
        assert!(decode("\"-1s\"").is_err());
        assert!(decode("-1").is_err());
    }

    /// The wire form of a duration is its nanoseconds, so a saved state decodes to the same span.
    #[test]
    fn a_duration_encodes_as_nanoseconds() {
        assert_eq!(serde_json::to_string(&secs(2)).unwrap(), "2000000000");
    }
}
