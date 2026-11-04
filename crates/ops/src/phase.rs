//! Which phase of a request an operator call belongs to.
//!
//! A kernel program may be split by phase: prefill and decode attention are different programs, and
//! diffusion distinguishes encoding, denoising and decoding. The phase is part of a program's
//! selection key (specs/worker.md 5.2) and travels with the epoch record as its name, so it is a
//! plain value with no runtime state of its own; the epoch record carries it as data, so it derives
//! the wire encoding beside its `Display` form. `serde` is a compile-time dependency with no
//! runtime boundary in it, which is why the crate keeps `platform` as its only runtime dependency
//! (`specs/worker.md` 4.2).
//!
//! Variants are appended, never reordered: the record's encoding names them, and a program built
//! against an older list keeps its meaning.

use std::fmt;

/// The phase of a request a call belongs to.
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PhaseKind {
    /// One forward over the whole prompt.
    Prefill,
    /// One forward per generated token.
    Decode,
    /// The encoder of a multimodal or diffusion model.
    Encode,
    /// A denoising step of a diffusion model.
    Denoise,
}

impl PhaseKind {
    /// Every phase, in declaration order.
    pub const ALL: [Self; 4] = [Self::Prefill, Self::Decode, Self::Encode, Self::Denoise];

    /// The name the record and a program's selection key use.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Encode => "encode",
            Self::Denoise => "denoise",
        }
    }
}

impl fmt::Display for PhaseKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::PhaseKind;

    #[test]
    fn phases_are_named_once_and_in_order() {
        let names: Vec<&str> = PhaseKind::ALL.iter().map(|phase| phase.as_str()).collect();
        assert_eq!(names, ["prefill", "decode", "encode", "denoise"]);
        let mut printed: Vec<String> = PhaseKind::ALL.iter().map(ToString::to_string).collect();
        printed.dedup();
        assert_eq!(printed.len(), PhaseKind::ALL.len());
    }
}
