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
/// module of any kind (control, routing, KV, scheduling), which states its kind in its own
/// section (`scheduling::program::SECTION`). The release's programs are installed under their
/// names; a policy package's own program crates under `programs.<dir>`.
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
/// ([`operator::RANGE_MARKER`]). The daemon keys the epoch record by these names and filters the
/// Build Server's store by [`ALL`](operator::ALL), and the main process installs a module under
/// the same name, without either linking the kernel library to learn them.
pub mod operator {
    pub const ACT_AND_MUL: &str = "act_and_mul";
    pub const VALUE_FIRST_SWIGLU: &str = "value_first_swiglu";
    pub const SWIGLU: &str = "swiglu";
    pub const CACHE_BLOCK_SNAPSHOT: &str = "cache_block_snapshot";
    pub const CACHE_BLOCK_SCATTER: &str = "cache_block_scatter";
    pub const SLOT_COMMIT: &str = "slot_commit";
    pub const TRANSPOSE: &str = "transpose";
    pub const VSA_SCORES: &str = "vsa_scores";
    pub const CONV: &str = "conv";
    pub const EMBEDDING: &str = "embedding";
    pub const FRAME_PAD: &str = "frame_pad";
    pub const FRAME_MOMENTS: &str = "frame_moments";
    pub const FRAME_NORM_PAD: &str = "frame_norm_pad";
    pub const GEMM: &str = "gemm";
    pub const KV_WRITE: &str = "kv_write";
    pub const KV_WRITE_SCALED: &str = "kv_write_scaled";
    pub const LOGITS_GATHER: &str = "logits_gather";
    pub const VOCAB_UNPAD: &str = "vocab_unpad";
    pub const MODULATED_RMS_NORM: &str = "modulated_rms_norm";
    pub const GATED_RESIDUAL: &str = "gated_residual";
    pub const PHILOX_NOISE: &str = "philox_noise";
    pub const RMS_NORM: &str = "rms_norm";
    pub const ADD_RMS_NORM: &str = "add_rms_norm";
    pub const SIGMOID_GATED_ADD: &str = "sigmoid_gated_add";
    pub const GATED_DELTA_RULE: &str = "gated_delta_rule";
    pub const CAUSAL_CONV1D: &str = "causal_conv1d";
    pub const GATED_RMS_NORM: &str = "gated_rms_norm";
    pub const LAYER_NORM: &str = "layer_norm";
    pub const SOFTMAX: &str = "softmax";
    pub const WEIGHTED_RMS_NORM: &str = "weighted_rms_norm";
    pub const SCALED_RESIDUAL_RMS_NORM: &str = "scaled_residual_rms_norm";
    pub const SCALED_RESIDUAL_LAYER_NORM: &str = "scaled_residual_layer_norm";
    pub const SCALED_RESIDUAL: &str = "scaled_residual";
    pub const SCALED_ADD: &str = "scaled_add";
    pub const ROTARY_FACTORS: &str = "rotary_factors";
    pub const QK_NORM_ROPE: &str = "qk_norm_rope";
    pub const LOGITS_SHAPE: &str = "logits_shape";
    pub const TOP_K: &str = "top_k";
    pub const SAMPLING_FROM_LOGITS: &str = "sampling_from_logits";
    pub const ARGMAX: &str = "argmax";
    pub const LOG_SOFTMAX: &str = "log_softmax";
    pub const LOGPROB_GATHER: &str = "logprob_gather";
    pub const TOP_LOGPROBS: &str = "top_logprobs";
    pub const VSA_ZERO_PADDING: &str = "vsa_zero_padding";
    pub const VSA_ADD_GATED_TILES: &str = "vsa_add_gated_tiles";
    pub const VSA_COMPOSE_GATED_ROWS: &str = "vsa_compose_gated_rows";
    pub const VSA_SELECT_TILES: &str = "vsa_select_tiles";
    pub const VSA_PREPARE_TILES: &str = "vsa_prepare_tiles";
    pub const VSA_COMPRESS_POOLED: &str = "vsa_compress_pooled";
    pub const DENSE_ATTENTION: &str = "dense_attention";
    pub const PAGED_ATTENTION: &str = "paged_attention";
    pub const SPARSE_ATTENTION: &str = "sparse_attention";
    pub const INDEXED_ATTENTION: &str = "indexed_attention";
    pub const INDEX_SCORES: &str = "index_scores";
    pub const INDEX_SELECT: &str = "index_select";
    pub const LATENT_ATTENTION: &str = "latent_attention";
    pub const LATENT_COMPRESS: &str = "latent_compress";
    pub const RECORD_WRITE: &str = "record_write";
    pub const MOE_UNPERMUTE: &str = "moe_unpermute";
    pub const EXPERT_EXCHANGE: &str = "expert_exchange";
    pub const GATHER_GEMM: &str = "gather_gemm";
    pub const GATHER_GEMM_NVFP4: &str = "gather_gemm_nvfp4";
    pub const ROUTE_GEMM: &str = "route_gemm";
    pub const ROUTE_GEMM_NVFP4: &str = "route_gemm_nvfp4";
    pub const TOPK_SOFTMAX: &str = "topk_softmax";
    pub const MOE_SORT: &str = "moe_sort";
    pub const NVFP4_QUANTIZE: &str = "nvfp4_quantize";
    pub const NVFP4_GEMM: &str = "nvfp4_gemm";
    pub const FP8_GEMM: &str = "fp8_gemm";
    pub const FP8_QUANTIZE: &str = "fp8_quantize";
    pub const DYNAMIC_SCALE: &str = "dynamic_scale";
    pub const SNAKE: &str = "snake";
    pub const CHANNEL_AFFINE: &str = "channel_affine";
    pub const ADAPTIVE_POOL: &str = "adaptive_pool";
    pub const ROWS: &str = "rows";
    pub const POSITION_INTERPOLATE: &str = "position_interpolate";
    pub const ACTIVATION1D: &str = "activation1d";
    pub const PATCH: &str = "patch";
    pub const UNPATCHIFY_VIDEO: &str = "unpatchify_video";
    pub const COPY_BLOCK: &str = "copy_block";
    pub const CROSS_FADE: &str = "cross_fade";
    pub const TILE_STITCH: &str = "tile_stitch";
    pub const POSTERIOR: &str = "posterior";
    pub const ROPE_APPLY: &str = "rope_apply";
    pub const PCM_INTERLEAVE: &str = "pcm_interleave";
    pub const RGB_BYTES: &str = "rgb_bytes";
    pub const ELEMENTWISE_UNARY: &str = "elementwise_unary";
    pub const ELEMENTWISE_BINARY: &str = "elementwise_binary";
    pub const ELEMENTWISE_WHERE: &str = "elementwise_where";
    pub const CAST: &str = "cast";
    pub const COPY_STRIDED: &str = "copy_strided";
    pub const FILL: &str = "fill";
    pub const REDUCE: &str = "reduce";
    pub const INDEX: &str = "index";
    /// The engine's range markers: a stamp of the device's clock at a module's entry and exit,
    /// and a read of that clock for a calibration (`crates/engine/src/profiling/markers.rs`).
    pub const RANGE_MARKER: &str = "range_marker";

    /// Every operator this build has, in a stable order for a package's coverage check.
    ///
    /// A build without an operator's packages still names it here: the table lists the
    /// build's operators, not the loaded modules.
    pub const ALL: &[&str] = &[
        ACT_AND_MUL,
        VALUE_FIRST_SWIGLU,
        SWIGLU,
        CACHE_BLOCK_SNAPSHOT,
        CACHE_BLOCK_SCATTER,
        SLOT_COMMIT,
        TRANSPOSE,
        VSA_SCORES,
        CONV,
        EMBEDDING,
        FRAME_PAD,
        FRAME_MOMENTS,
        FRAME_NORM_PAD,
        GEMM,
        KV_WRITE,
        KV_WRITE_SCALED,
        LOGITS_GATHER,
        VOCAB_UNPAD,
        MODULATED_RMS_NORM,
        GATED_RESIDUAL,
        PHILOX_NOISE,
        RMS_NORM,
        ADD_RMS_NORM,
        SIGMOID_GATED_ADD,
        GATED_DELTA_RULE,
        CAUSAL_CONV1D,
        GATED_RMS_NORM,
        LAYER_NORM,
        SOFTMAX,
        WEIGHTED_RMS_NORM,
        SCALED_RESIDUAL_RMS_NORM,
        SCALED_RESIDUAL_LAYER_NORM,
        SCALED_RESIDUAL,
        SCALED_ADD,
        ROTARY_FACTORS,
        QK_NORM_ROPE,
        LOGITS_SHAPE,
        TOP_K,
        SAMPLING_FROM_LOGITS,
        ARGMAX,
        LOG_SOFTMAX,
        LOGPROB_GATHER,
        TOP_LOGPROBS,
        VSA_ZERO_PADDING,
        VSA_ADD_GATED_TILES,
        VSA_COMPOSE_GATED_ROWS,
        VSA_SELECT_TILES,
        VSA_PREPARE_TILES,
        VSA_COMPRESS_POOLED,
        DENSE_ATTENTION,
        PAGED_ATTENTION,
        SPARSE_ATTENTION,
        INDEXED_ATTENTION,
        INDEX_SCORES,
        INDEX_SELECT,
        LATENT_ATTENTION,
        LATENT_COMPRESS,
        RECORD_WRITE,
        MOE_UNPERMUTE,
        EXPERT_EXCHANGE,
        GATHER_GEMM,
        GATHER_GEMM_NVFP4,
        ROUTE_GEMM,
        ROUTE_GEMM_NVFP4,
        TOPK_SOFTMAX,
        MOE_SORT,
        NVFP4_QUANTIZE,
        NVFP4_GEMM,
        FP8_GEMM,
        FP8_QUANTIZE,
        DYNAMIC_SCALE,
        SNAKE,
        CHANNEL_AFFINE,
        ADAPTIVE_POOL,
        ROWS,
        POSITION_INTERPOLATE,
        ACTIVATION1D,
        PATCH,
        UNPATCHIFY_VIDEO,
        COPY_BLOCK,
        CROSS_FADE,
        TILE_STITCH,
        POSTERIOR,
        ROPE_APPLY,
        PCM_INTERLEAVE,
        RGB_BYTES,
        ELEMENTWISE_UNARY,
        ELEMENTWISE_BINARY,
        ELEMENTWISE_WHERE,
        CAST,
        COPY_STRIDED,
        FILL,
        REDUCE,
        INDEX,
        RANGE_MARKER,
    ];

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
