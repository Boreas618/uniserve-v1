//! The released checkpoints the profile runs read, each a Hugging Face repository at the commit
//! the runs were checked against ([`nn_testing::Release`]).
//!
//! A profile run reads a release's shard headers and configuration, so its kernel keys and peaks
//! are those of exactly this commit. The runs are ignored by default; with `--ignored` each
//! resolves its releases through the Hugging Face cache and fails, naming the repository and
//! the commit, when one cannot be resolved.

use nn_testing::Release;

/// Qwen3-30B-A3B in BF16 (`Qwen3MoeForCausalLM`).
pub(crate) const QWEN3_30B_A3B: Release = Release {
    repository: "Qwen/Qwen3-30B-A3B",
    revision: "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
};

/// NVIDIA's calibrated NVFP4 export of Qwen3-30B-A3B.
pub(crate) const QWEN3_30B_A3B_NVFP4: Release = Release {
    repository: "nvidia/Qwen3-30B-A3B-NVFP4",
    revision: "2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3",
};

/// Qwen3.6-35B-A3B in BF16 (`Qwen3_5MoeForConditionalGeneration`), with its drafter and its
/// vision tower.
pub(crate) const QWEN3_6_35B_A3B: Release = Release {
    repository: "Qwen/Qwen3.6-35B-A3B",
    revision: "345d467b117d1fd5eda9b879a5988de9ae5f47a7",
};

/// NVIDIA's calibrated mixed-precision export of Qwen3.6-35B-A3B.
pub(crate) const QWEN3_6_35B_A3B_NVFP4: Release = Release {
    repository: "nvidia/Qwen3.6-35B-A3B-NVFP4",
    revision: "1355db6a052410cfd62085d94b58866fd0f2c3c5",
};

/// NVIDIA's NVFP4 export of Qwen3.8 Flash Next (`Qwen4ExpForConditionalGeneration`).
pub(crate) const QWEN3_8_FLASH_NEXT_NVFP4: Release = Release {
    repository: "nvidia/Qwen3.8-Flash-Next-NVFP4",
    revision: "fc694b54fb0174e0913e6adf86691ef85a4ead47",
};

/// RadixArk's NVFP4 export of Qwen3.8 Flash Next.
pub(crate) const QWEN3_8_FLASH_NEXT_NVFP4_RADIXARK: Release = Release {
    repository: "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    revision: "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
};

/// The block-scaled FP8 export of Qwen3.8 Flash Next.
pub(crate) const QWEN3_8_FLASH_NEXT_FP8: Release = Release {
    repository: "Qwen/Qwen3.8-Flash-Next-FP8",
    revision: "236dfdf285828023ca3bcd3f37366c58a3469b13",
};

/// The MiniMax-H3 base release: the diffusers pipeline of both networks, the text encoder and
/// the VAEs.
pub(crate) const MINIMAX_H3: Release = Release {
    repository: "MiniMaxAI/MiniMax-H3",
    revision: "42ed227ee7df40d41602854ae760620d6eb651fe",
};

/// The eight-step FastH3 student: a DMD ladder under video sparse attention.
pub(crate) const FASTH3_8_STEP: Release = Release {
    repository: "FastVideo/FastVideo-FastH3-8-Step-V2",
    revision: "3da2ddfe1954d9cda4c05b643dc0f26007a655c5",
};

/// The four-step FastH3 student: a DMD ladder under video sparse attention.
pub(crate) const FASTH3_4_STEP: Release = Release {
    repository: "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
    revision: "5ea076f35b84da4c3c82217112fa733d8eea2ae1",
};

/// The four-step student's ModelOpt NVFP4 export, whose manifest calibrates the denoiser's
/// feed-forward projections and the video decoder's block projections.
pub(crate) const FASTH3_4_STEP_NVFP4: Release = Release {
    repository: "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4",
    revision: "3e7ccea5cb400e802cdb8aa1848707a6b0c82b06",
};
