//! A program: the backend, revision and arguments that fill one operator
//! (design section 5.1).
//!
//! [`Program`] is defined here, in the Program Server's crate, because the
//! daemon and the main process both name programs with it: the daemon selects
//! them with `prepare` and records them, and the main process `resolve`s the
//! ones it is told to load. Its parameters travel as a TOML table, so a
//! program's arguments keep the types their configuration fields have.
//!
//! The revision fixes everything a module is built from (design section 5.2),
//! so a path in the module store ([`crate::store`]) needs nothing else to name
//! one module: the same revision of the same package is the same module on
//! every host.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Which phase of an operator a program fills.
///
/// A program usually fills every phase of its operator. Attention may be
/// filled per phase instead, by `prefill_attention_backend` and
/// `decode_attention_backend` (design section 5.2), and then the phase is
/// part of the descriptor: one generation still indexes its kernels by
/// (operator name, descriptor key) (design section 4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseKind {
    /// The phase before a token of the request exists: a prompt's tokens.
    Prefill,
    /// The phase that produces one token per request per step.
    Decode,
    /// The encoding phase of a diffusion computation: inputs become latents.
    Encode,
    /// The denoising phase of a diffusion computation: latents become outputs.
    Denoise,
}

impl PhaseKind {
    /// The phase's name, which the program manifests and the record use.
    pub fn as_str(self) -> &'static str {
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

/// One program: the backend that implements an operator, the backend's
/// revision, and the arguments given to it.
///
/// `backend` is the name the configuration field selects, such as
/// `"trtllm"` for attention or `"fcfs"` for the scheduling policy (design
/// section 5.1). `arguments` are the program's parameters, which come from
/// the configuration fields that are not part of the build config.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub backend: String,
    /// Which installation of the backend's package this is: one more than
    /// the revisions installed before it (design section 5.1).
    pub revision: u32,
    #[serde(default)]
    pub arguments: toml::Table,
}

/// The slot a program fills in an epoch: an operator, and the phase when the
/// operator is filled per phase (design section 5.1).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProgramSlot {
    pub op: String,
    /// `None` when one program fills every phase of the operator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<PhaseKind>,
}

impl ProgramSlot {
    /// The slot of an operator filled in all of its phases.
    pub fn all(op: impl Into<String>) -> Self {
        Self {
            op: op.into(),
            phase: None,
        }
    }

    /// The slot of an operator filled in one phase.
    pub fn phase(op: impl Into<String>, phase: PhaseKind) -> Self {
        Self {
            op: op.into(),
            phase: Some(phase),
        }
    }
}

/// The programs an epoch fills its operators with, by slot.
pub type Programs = BTreeMap<ProgramSlot, Program>;
