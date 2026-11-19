//! The module vocabulary: what fills an operator, which slot it fills, and the operators a build
//! has (design section 5.1).
//!
//! The daemon selects packages with the Build Server's `prepare` and records them in the epoch
//! record, the main process `resolve`s the ones it is told to load, and the kernel library keys its
//! generations by the same operator names and descriptor keys. The vocabulary is stated here, below all of
//! them, so a daemon or a scheduler names packages without linking the numerical library, and the
//! record key, the wire message and a package's descriptor cannot drift apart. Its parameters
//! are a TOML table, so a package's arguments keep the types their configuration fields have, and
//! the process interface carries each as a typed `Value` (`wire`).
//!
//! The revision fixes everything a module is built from (design section 5.2), so a path in the
//! Build Server's module store needs nothing else to name one module: the same revision of the
//! same package is the same module on every host.

mod wire;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use wire::{WireError, kernel_of, kernel_to_wire};

/// Which phase of a request an operator call belongs to: the scheduling interface's one phase
/// vocabulary ([`scheduling::view::Phase`]), which the scheduler, the policies and an operator's
/// descriptor name work by alike. A descriptor that depends on the phase states it as its `phase`
/// dimension, by [`Phase::name`], so the phase is part of the descriptor key a deployment pins a
/// package for ([`Pin`]).
pub use scheduling::view::Phase;

/// One package: the backend that implements an operator, the backend's revision, and the arguments
/// given to it.
///
/// `backend` is the name of an installed implementation, such as `"trtllm"` for attention or
/// `"arrival"` for the scheduling policy (design section 5.1). `arguments` are the package's
/// parameters, which come from the configuration fields that are not part of the build config.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PackageRevision {
    pub backend: String,
    /// Which installation of the backend's package this is: one more than the revisions installed
    /// before it (design section 5.1).
    pub revision: u32,
    #[serde(default)]
    pub arguments: toml::Table,
}

/// The slot a package fills in an epoch: an operator, and either the one descriptor key the
/// deployment pins the package for or the part of the operator's keys the Build Server's
/// selection gives it (design section 5.1).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModuleSlot {
    pub op: String,
    /// The descriptor key, in its canonical text, the deployment's `[kernels]` table pins this
    /// package for; `None` for a package the selection fills keys with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// The backend, when the operator's packages divide its descriptors between them: each holds a
    /// slot of its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
}

impl ModuleSlot {
    /// The slot of an operator one package fills whole.
    pub fn all(op: impl Into<String>) -> Self {
        Self {
            op: op.into(),
            key: None,
            part: None,
        }
    }

    /// The slot of the package a deployment pins for one descriptor key of an operator.
    pub fn key(op: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            op: op.into(),
            key: Some(key.into()),
            part: None,
        }
    }
}

/// The package a deployment's `[kernels]` table pins for one descriptor key: a backend and its
/// revision, written `backend@revision`, or `backend@latest` for the latest installed revision,
/// which an apply fixes to the revision it resolves (design section 5.1).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pin {
    /// The installed implementation, such as `trtllm`.
    pub backend: String,
    /// The installed revision; `None` for `latest`.
    pub revision: Option<u32>,
}

impl std::fmt::Display for Pin {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.revision {
            Some(revision) => write!(formatter, "{}@{revision}", self.backend),
            None => write!(formatter, "{}@latest", self.backend),
        }
    }
}

impl std::str::FromStr for Pin {
    type Err = String;

    /// Reads `backend@revision` or `backend@latest`, refusing an empty backend and a revision
    /// that is neither `latest` nor a positive integer: revisions count installations from one.
    fn from_str(text: &str) -> Result<Self, String> {
        let (backend, revision) = text.rsplit_once('@').ok_or_else(|| {
            format!("{text:?} names no revision: write `backend@revision` or `backend@latest`")
        })?;
        if backend.is_empty() {
            return Err(format!("{text:?} names no backend before its `@`"));
        }
        let revision = match revision {
            "latest" => None,
            number => Some(
                number
                    .parse::<u32>()
                    .ok()
                    .filter(|revision| *revision > 0)
                    .ok_or_else(|| {
                        format!("{text:?}: the revision is `latest` or a positive integer")
                    })?,
            ),
        };
        Ok(Self {
            backend: backend.to_owned(),
            revision,
        })
    }
}

impl Serialize for Pin {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Pin {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// A deployment's `[kernels]` table: for each operator, by its name, the package each descriptor
/// key it lists is pinned to, the key in the canonical text the operator's descriptor prints and
/// the kernel observations label it by (`op` and `descriptor`), matched exactly. A key the table
/// does not list is the Build Server's to select, by its packages' manifest priorities.
pub type Kernels = BTreeMap<String, BTreeMap<String, Pin>>;

/// The packages an epoch fills its operators with, by slot.
pub type Packages = BTreeMap<ModuleSlot, PackageRevision>;

/// The interface the engine package implements, as its packages are filed: the main process of an
/// epoch, an executable the daemon starts.
pub const ENGINE: &str = "engine";

/// The interface every program package implements, as its packages are filed: a wasm32 program
/// module of any kind (control, routing, KV, scheduling, processing), which states its kind in
/// its own section (`scheduling::program::SECTION`). The release's programs are installed under
/// their names; a policy package's own program crates under `programs.<dir>`.
pub const PROGRAM: &str = "program";

/// The key of the one module a package of the engine or a program builds: each fills a whole
/// interface, not a set of kernel shapes.
pub const SINGLE_KEY: &str = "default";

/// A module set: for every interface a deployment uses, the package revision selected for it and
/// its arguments.
///
/// The daemon fixes a module set once, when a configuration is applied: it asks the Build Server
/// which installed revision each named implementation and each installed kernel package is at, and
/// records what it returns. The set is part of the epoch's record and of every launch file made from it,
/// so a relaunch and a standby run exactly what their epoch runs, and a revision installed later is
/// run by nobody until a configuration is applied again.
#[serde_with::serde_as]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModuleSet {
    /// The engine package: the executable the epoch's main processes run as.
    pub engine: PackageRevision,
    /// The scheduling policy's package.
    pub policy: PackageRevision,
    /// The kernel packages. A slot of one descriptor key holds the package the deployment's
    /// `[kernels]` table pins for that key. A slot whose part names an implementation holds an
    /// installed package the Build Server selects among for every other key, by the manifests'
    /// priorities.
    ///
    /// Written as a list of pairs, because a launch file is JSON and a slot is no JSON key.
    #[serde_as(as = "Vec<(_, _)>")]
    pub kernels: BTreeMap<ModuleSlot, PackageRevision>,
    /// The policy package's programs, by their names in its `config.toml`: the program package
    /// revision each one's source fixed to and what its module states. The scheduling program
    /// among them is also [`Self::policy`], which the engines run; the others run on the leader
    /// daemon and the Storage Servers, so the engines' wire form of the set leaves them out.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub programs: BTreeMap<String, Program>,
    /// The processing program the leader daemon serves the epoch's requests through
    /// (`worker_config.processing`): the request side of the model families, fixed beside the
    /// engine so an epoch's request side is built from the same tree as its engine. The engines'
    /// wire form of the set leaves it out, so theirs states none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub processing: Option<PackageRevision>,
}

/// One program of a module set: where it came from, the program package revision its apply
/// fixed, and what its module states.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Program {
    /// Its source as the configuration writes it: `programs/<dir>` or `uniserve:<package>`.
    pub source: String,
    /// The installed program package that builds its module, at the fixed revision; a crate of
    /// the policy package is installed as the package `programs.<dir>`.
    pub backend: String,
    pub revision: u32,
    /// The kind of program its module states.
    pub kind: scheduling::program::Kind,
    /// The SHA-256 of its wasm32 module, in hex.
    pub digest: String,
}

/// The operators of this build, by the name their packages are filed under.
///
/// Each constant names the device code one package fills: the `NAME` of an operator of the kernel
/// library (`ops::Op::NAME`), which takes it from here, or the engine's range markers
/// ([`operator::RANGE_MARKER`]: a stamp of the device's clock at a module's entry and exit, and a
/// read of that clock for a calibration, `crates/engine/src/profiling/markers.rs`). The main
/// process installs a module under the same name without linking the kernel library to learn
/// them.
///
/// The constants and [`ALL`](operator::ALL) are compiled from the vocabulary
/// `src/module/operators.toml` (`build.rs`), the one statement of the names. That file is also
/// what an engine revision records at the Build Server as the operators it launches (`operators`
/// in `crates/engine/package.toml`): the daemon fixes an epoch's kernel packages over the
/// vocabulary of the epoch's engine revision, never over this build's, so a daemon built before
/// an operator existed fixes, pins and updates the packages of a later engine's operators.
pub mod operator {
    include!(concat!(env!("OUT_DIR"), "/operators.rs"));

    /// `name`, interned, when this build has an operator by that name.
    ///
    /// A package's operator name arrives as text; this gives back the static name a generation is
    /// keyed by, or `None` when this build has no operator named `name`.
    pub fn named(name: &str) -> Option<&'static str> {
        ALL.iter().copied().find(|candidate| *candidate == name)
    }

    /// The operators whose interface is past its first version, each with the version this build
    /// launches.
    ///
    /// An operator's interface is its descriptor and its argument layout. A change of either that
    /// is not an addition raises the operator's version here, and a package of the operator states
    /// the version it implements in its manifest (`interface_version`).
    const VERSIONS: &[(&str, u32)] = &[
        // The product reads an activation `fp8_quantize` encoded, and its factor from device
        // memory, where a measured scale is written by a launch before it.
        (FP8_GEMM, 2),
        // The router reads the batch's live count and adds each live route into the expert
        // load, two pointers after the map.
        (TOPK_SOFTMAX, 2),
    ];

    /// The version of `operator`'s interface in this build: 1 unless [`VERSIONS`] states another.
    ///
    /// The main process compares it with the version a module was built for when it loads the
    /// module, and refuses another.
    pub fn version(operator: &str) -> u32 {
        VERSIONS
            .iter()
            .find(|(name, _)| *name == operator)
            .map_or(1, |(_, version)| *version)
    }
}

/// The commit table a batch's producer writes and the `slot_commit` operator reads: one row of
/// [`commit::COMMIT_FIELDS`] columns per call, which states how the call's step advances its slot's
/// device state.
///
/// Stated here because the scheduler writes the rows and the kernel library launches the operator
/// over them, and neither links the other.
pub mod commit {
    /// The rows of the commit column table, in order.
    pub const COMMIT_FIELDS: usize = 8;

    /// The column table's row indices.
    pub mod column {
        /// The row's destination slot.
        pub const SLOT: usize = 0;
        /// The row's mode: decode, explicit, verify, cache or count.
        pub const MODE: usize = 1;
        /// The task the row sampled, or -1 when it sampled none.
        pub const TASK: usize = 2;
        /// The host cache length of an explicit row.
        pub const CACHE_LENGTH: usize = 3;
        /// The host logical position of an explicit row.
        pub const LOGICAL: usize = 4;
        /// The host sampling position of an explicit row.
        pub const SAMPLING: usize = 5;
        /// The verification base's visible length.
        pub const VISIBLE: usize = 6;
        /// Whether the row counts penalties.
        pub const PENALTIES: usize = 7;
    }

    /// The mode of a commit row.
    pub mod mode {
        /// One accepted token: the three lengths advance by one.
        pub const DECODE: i64 = 0;
        /// The host columns state the lengths.
        pub const EXPLICIT: i64 = 1;
        /// A verified span: the lengths move to `base + points`.
        pub const VERIFY: i64 = 2;
        /// The lengths are left as they are.
        pub const CACHE: i64 = 3;
        /// The row's token joins the slot's penalty history and nothing else moves: a
        /// verification's accepted drafts, which precede the token its `VERIFY` row installs.
        pub const COUNT: i64 = 4;
    }
}

/// The sampler's parameter columns of one row, which a batch's producer writes and the sampling
/// operators read.
///
/// Stated here because the scheduler writes the rows and the kernel library reads them, and neither
/// links the other.
pub mod sampler {
    /// The columns of one row of sampling parameters, in the order the worker writes them.
    ///
    /// The offsets are fixed (the temperature first, the repetition, frequency and presence penalties
    /// at 4, 5 and 6, and the top-k width, the nucleus threshold and the min-p threshold after them);
    /// the op reads only the parameters it names.
    pub const PARAMETER_COLUMNS: u32 = 8;

    /// The offset of the temperature among a row's parameters.
    pub const TEMPERATURE: u32 = 0;
    /// The offset of the top-k width (0 to keep every candidate).
    pub const TOP_K: u32 = 1;
    /// The offset of the nucleus threshold.
    pub const TOP_P: u32 = 2;
}

#[cfg(test)]
mod tests {
    use super::{ModuleSlot, Pin};

    /// A slot pinned for a key records the key, and a slot of the whole operator records none.
    #[test]
    fn a_slot_records_the_key_it_is_pinned_for() {
        let slot = ModuleSlot::key("paged_attention", "phase_decode_head_128");
        let recorded = serde_json::to_value(&slot).unwrap();
        assert_eq!(recorded["key"], "phase_decode_head_128");
        assert_eq!(
            serde_json::from_value::<ModuleSlot>(recorded).unwrap(),
            slot
        );
        let every = serde_json::to_value(ModuleSlot::all("rms_norm")).unwrap();
        assert!(every.get("key").is_none());
    }

    /// A pin reads and prints `backend@revision` and `backend@latest`, and refuses a text that
    /// names no backend or no revision of one.
    #[test]
    fn a_pin_is_a_backend_at_a_revision_or_the_latest() {
        let pinned: Pin = "trtllm@7".parse().unwrap();
        assert_eq!(
            pinned,
            Pin {
                backend: "trtllm".to_owned(),
                revision: Some(7),
            }
        );
        assert_eq!(pinned.to_string(), "trtllm@7");
        let latest: Pin = "triton@latest".parse().unwrap();
        assert_eq!(latest.revision, None);
        assert_eq!(latest.to_string(), "triton@latest");
        for refused in ["trtllm", "@7", "trtllm@0", "trtllm@seven", "trtllm@"] {
            assert!(refused.parse::<Pin>().is_err(), "{refused}");
        }
    }
}
