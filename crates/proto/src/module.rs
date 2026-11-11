//! The module vocabulary: what fills an operator, which slot it fills, and the operators a build
//! has (design section 5.1).
//!
//! The daemon selects packages with the Build Server's `prepare` and records them in the epoch
//! record, the main process `resolve`s the ones it is told to load, and the kernel library keys its
//! generations by the same operator names and phases. The vocabulary is stated here, below all of
//! them, so a daemon or a scheduler names packages without linking the numerical library, and the
//! record key, the wire message and a package's descriptor cannot drift apart. Its parameters
//! travel as a TOML table, so a package's arguments keep the types their configuration fields have.
//!
//! The revision fixes everything a module is built from (design section 5.2), so a path in the
//! Build Server's module store needs nothing else to name one module: the same revision of the
//! same package is the same module on every host.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Which phase of a request an operator call belongs to, and which phase of an operator a package
/// fills.
///
/// A kernel package may be split by phase: prefill and decode attention are different packages
/// (`prefill_attention_backend` and `decode_attention_backend`, design section 5.2), and diffusion
/// distinguishes encoding, denoising and decoding. The phase is part of an operator's descriptor and
/// of a package's selection key, so one generation still indexes its kernels by (operator name,
/// descriptor key) (design section 4.2).
///
/// Variants are appended, never reordered: the record's encoding names them, and a package built
/// against an older list keeps its meaning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
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

    /// The name the record and a package's selection key use.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Encode => "encode",
            Self::Denoise => "denoise",
        }
    }

    /// The phase a descriptor key states, when its operator's descriptor holds one.
    ///
    /// An operator split by phase writes the phase as its key's first field (`phase_decode_...`,
    /// the paged attention operator's keys); a key of an operator without one states no phase and
    /// answers `None`.
    pub fn of_key(key: &str) -> Option<Self> {
        let rest = key.strip_prefix("phase_")?;
        let name = rest.split('_').next()?;
        Self::ALL.into_iter().find(|phase| phase.as_str() == name)
    }
}

impl fmt::Display for PhaseKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One package: the backend that implements an operator, the backend's revision, and the arguments
/// given to it.
///
/// `backend` is the name the configuration field selects, such as `"trtllm"` for attention or
/// `"fcfs"` for the scheduling policy (design section 5.1). `arguments` are the package's
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

/// The slot a package fills in an epoch: an operator, and the phase when the operator is filled per
/// phase (design section 5.1).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModuleSlot {
    pub op: String,
    /// `None` when one package fills every phase of the operator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<PhaseKind>,
    /// The backend, when the operator's packages divide its descriptors between them with no phase
    /// to tell them apart: each holds a slot of its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
}

impl ModuleSlot {
    /// The slot of an operator filled in all of its phases.
    pub fn all(op: impl Into<String>) -> Self {
        Self {
            op: op.into(),
            phase: None,
            part: None,
        }
    }

    /// The slot of an operator filled in one phase.
    pub fn phase(op: impl Into<String>, phase: PhaseKind) -> Self {
        Self {
            op: op.into(),
            phase: Some(phase),
            part: None,
        }
    }
}

/// The packages an epoch fills its operators with, by slot.
pub type Packages = BTreeMap<ModuleSlot, PackageRevision>;

/// The operators of this build, by the name their packages are filed under.
///
/// Each constant is the `NAME` of one operator of the kernel library (`ops::Op::NAME`), which takes
/// it from here: the daemon keys the epoch record by these names and filters the Build Server's
/// store by [`ALL`], and the main process installs a module under the same name, without either
/// linking the kernel library to learn them.
pub mod operator {
    pub const ACT_AND_MUL: &str = "act_and_mul";
    pub const VALUE_FIRST_SWIGLU: &str = "value_first_swiglu";
    pub const SWIGLU: &str = "swiglu";
    pub const CACHE_BLOCK_SNAPSHOT: &str = "cache_block_snapshot";
    pub const CACHE_BLOCK_SCATTER: &str = "cache_block_scatter";
    pub const SLOT_COMMIT: &str = "slot_commit";
    pub const TRANSPOSE: &str = "transpose";
    pub const VSA_SCORES: &str = "vsa_scores";
    pub const CONV3D: &str = "conv3d";
    pub const CONV1D: &str = "conv1d";
    pub const CONV_TRANSPOSE1D: &str = "conv_transpose1d";
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
    pub const PAGE_CONVERSION: &str = "page_conversion";
    pub const CACHE_COPY_REGION: &str = "cache_copy_region";
    pub const WEIGHTED_RMS_NORM: &str = "weighted_rms_norm";
    pub const SCALED_RESIDUAL_RMS_NORM: &str = "scaled_residual_rms_norm";
    pub const SCALED_RESIDUAL_LAYER_NORM: &str = "scaled_residual_layer_norm";
    pub const SCALED_RESIDUAL: &str = "scaled_residual";
    pub const SCALED_ADD: &str = "scaled_add";
    pub const ROTARY_FACTORS: &str = "rotary_factors";
    pub const QK_NORM_ROPE: &str = "qk_norm_rope";
    pub const LOGITS_SHAPE: &str = "logits_shape";
    pub const TOP_K: &str = "top_k";
    pub const FILTER_CANDIDATES: &str = "filter_candidates";
    pub const CANDIDATE_ORDER: &str = "candidate_order";
    pub const INVERSE_CDF: &str = "inverse_cdf";
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
    pub const SNAKE: &str = "snake";
    pub const CHANNEL_AFFINE: &str = "channel_affine";
    pub const GELU: &str = "gelu";
    pub const SILU: &str = "silu";
    pub const ADAPTIVE_POOL: &str = "adaptive_pool";
    pub const ROWS: &str = "rows";
    pub const POSITION_INTERPOLATE: &str = "position_interpolate";
    pub const AUDIO_ACTIVATION: &str = "audio_activation";
    pub const PATCH: &str = "patch";
    pub const UNPATCHIFY_VIDEO: &str = "unpatchify_video";
    pub const COPY_BLOCK: &str = "copy_block";
    pub const CROSS_FADE: &str = "cross_fade";
    pub const TILE_STITCH: &str = "tile_stitch";
    pub const POSTERIOR: &str = "posterior";
    pub const ROPE_APPLY: &str = "rope_apply";
    pub const PCM_INTERLEAVE: &str = "pcm_interleave";
    pub const RGB_BYTES: &str = "rgb_bytes";

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
        CONV3D,
        CONV1D,
        CONV_TRANSPOSE1D,
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
        PAGE_CONVERSION,
        CACHE_COPY_REGION,
        WEIGHTED_RMS_NORM,
        SCALED_RESIDUAL_RMS_NORM,
        SCALED_RESIDUAL_LAYER_NORM,
        SCALED_RESIDUAL,
        SCALED_ADD,
        ROTARY_FACTORS,
        QK_NORM_ROPE,
        LOGITS_SHAPE,
        TOP_K,
        FILTER_CANDIDATES,
        CANDIDATE_ORDER,
        INVERSE_CDF,
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
        SNAKE,
        CHANNEL_AFFINE,
        GELU,
        SILU,
        ADAPTIVE_POOL,
        ROWS,
        POSITION_INTERPOLATE,
        AUDIO_ACTIVATION,
        PATCH,
        UNPATCHIFY_VIDEO,
        COPY_BLOCK,
        CROSS_FADE,
        TILE_STITCH,
        POSTERIOR,
        ROPE_APPLY,
        PCM_INTERLEAVE,
        RGB_BYTES,
    ];

    /// `name`, interned, when this build has an operator by that name.
    ///
    /// A package's operator name arrives as text; this gives back the static name a generation is
    /// keyed by, or `None` when no operator of this build answers to `name`.
    pub fn named(name: &str) -> Option<&'static str> {
        ALL.iter().copied().find(|candidate| *candidate == name)
    }

    /// The operators whose interface is past its first version, each with the version this build
    /// launches.
    ///
    /// An operator's interface is its descriptor and its argument layout. A change of either that
    /// is not an addition raises the operator's version here, and a package of the operator states
    /// the version it implements in its manifest (`interface_version`).
    const VERSIONS: &[(&str, u32)] = &[];

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
    use super::PhaseKind;

    #[test]
    fn phases_are_named_once_and_in_order() {
        let names: Vec<&str> = PhaseKind::ALL.iter().map(|phase| phase.as_str()).collect();
        assert_eq!(names, ["prefill", "decode", "encode", "denoise"]);
        let mut printed: Vec<String> = PhaseKind::ALL.iter().map(ToString::to_string).collect();
        printed.dedup();
        assert_eq!(printed.len(), PhaseKind::ALL.len());
    }

    #[test]
    fn a_key_states_its_phase_as_its_first_field() {
        assert_eq!(
            PhaseKind::of_key("phase_decode_dtype_bf16"),
            Some(PhaseKind::Decode)
        );
        assert_eq!(PhaseKind::of_key("phase_prefill"), Some(PhaseKind::Prefill));
        assert_eq!(PhaseKind::of_key("dtype_bf16_phase_decode"), None);
        assert_eq!(PhaseKind::of_key("phase_sideways_dtype_bf16"), None);
    }
}
