//! Released checkpoints the engine's tests read, each a Hugging Face repository at a pinned commit.
//!
//! A release resolves through the standard Hugging Face cache as `nn_testing::Release` resolves
//! it ([`Release::snapshot`]). Tests that read one are ignored by default: the repository holds
//! none of them.

pub use nn_testing::Release;

/// The FastH3 export of four denoising steps with sparse video attention.
pub const FAST_H3_4_STEP: Release = Release {
    repository: "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
    revision: "5ea076f35b84da4c3c82217112fa733d8eea2ae1",
};

/// The FastH3 export of eight denoising steps.
pub const FAST_H3_8_STEP: Release = Release {
    repository: "FastVideo/FastVideo-FastH3-8-Step-V2",
    revision: "3da2ddfe1954d9cda4c05b643dc0f26007a655c5",
};

/// The base MiniMax-H3 release, whose dense denoiser any Ulysses degree splits.
pub const MINIMAX_H3: Release = Release {
    repository: "MiniMaxAI/MiniMax-H3",
    revision: "9bfb6693f2cf6de171db46d1aa586f67d773a1da",
};

/// Qwen3-30B-A3B, a text mixture-of-experts model.
pub const QWEN3_30B_A3B: Release = Release {
    repository: "Qwen/Qwen3-30B-A3B",
    revision: "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
};

/// Qwen3.6-35B-A3B, a hybrid mixture-of-experts model with a vision tower.
pub const QWEN3_6_35B_A3B: Release = Release {
    repository: "Qwen/Qwen3.6-35B-A3B",
    revision: "345d467b117d1fd5eda9b879a5988de9ae5f47a7",
};
