//! The model families UniServe serves, and how a checkpoint names its
//! family.
//!
//! A checkpoint names its family by the `model_type` of its root
//! architecture config or, for a diffusers pipeline, by the `_class_name` of
//! its root index; [`ModelDescription::from_model_type`] and
//! [`ModelDescription::from_pipeline_class`] read either back.

use serde::{Deserialize, Serialize};

/// The complete configured model-description set selected at server startup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelDescription {
    /// Qwen3 text-generation profile, serving dense, mixture-of-experts and
    /// Qwen3.5 (`qwen3_5`, whose language model the workers serve)
    /// checkpoints alike: they share tokenizer, chat template, output parsing
    /// and autoregressive runtime, and differ only in worker numerics. A
    /// checkpoint whose template asks for another tool-call format than the
    /// Qwen3 parser reads serves no tool calls.
    #[default]
    Qwen3,
    /// SenseNova multimodal-generation profile.
    #[serde(rename = "sensenova")]
    SenseNova,
    /// Bagel multimodal-generation profile.
    Bagel,
    /// MiniMax H3 video-generation profile.
    MiniMaxH3,
    /// DiffusionGemma block-diffusion profile: System One decision readout
    /// and block-diffusion chat generation.
    DiffusionGemma,
}

impl ModelDescription {
    /// Every served model family.
    ///
    /// `from_model_type` and `from_pipeline_class` search only this list, so a
    /// variant missing from it compiles but is never selected from checkpoint
    /// metadata.
    const ALL: [Self; 5] = [
        Self::Qwen3,
        Self::SenseNova,
        Self::Bagel,
        Self::MiniMaxH3,
        Self::DiffusionGemma,
    ];

    /// Returns the stable profile identifier.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::SenseNova => "sensenova",
            Self::Bagel => "bagel",
            Self::MiniMaxH3 => "minimax_h3",
            Self::DiffusionGemma => "diffusion_gemma",
        }
    }

    /// Returns every `model_type` a root `config.json` of this family may
    /// declare, empty for a family that ships as a diffusers pipeline.
    const fn model_types(self) -> &'static [&'static str] {
        match self {
            Self::Qwen3 => &["qwen3", "qwen3_moe", "qwen3_5"],
            Self::SenseNova => &["neo_chat"],
            Self::Bagel => &["bagel"],
            Self::DiffusionGemma => &["diffusion_gemma"],
            Self::MiniMaxH3 => &[],
        }
    }

    /// Returns the pipeline class a family's checkpoint declares in a root
    /// diffusers index.
    ///
    /// MiniMax H3 ships only as a pipeline. A DiffusionGemma checkpoint
    /// publishes a `DiffusionGemmaPipeline` index beside its root transformers
    /// `config.json`, which remains the source of its settings (see
    /// [`ModelDescription::is_pipeline`]).
    const fn pipeline_class(self) -> Option<&'static str> {
        match self {
            Self::MiniMaxH3 => Some("MiniMaxH3ModularPipeline"),
            Self::DiffusionGemma => Some("DiffusionGemmaPipeline"),
            Self::Qwen3 | Self::SenseNova | Self::Bagel => None,
        }
    }

    /// Returns whether the family ships only as a diffusers pipeline, whose
    /// root index rather than a root `config.json` describes the checkpoint.
    pub const fn is_pipeline(self) -> bool {
        self.model_types().is_empty()
    }

    /// Resolves the served profile from a checkpoint's `model_type` field.
    ///
    /// The checkpoint configuration is the only source of model identity.
    /// `None` means UniServe serves no family with this `model_type` from a
    /// root configuration; pipeline families never match here.
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|description| description.model_types().contains(&model_type))
    }

    /// Resolves the served profile from a checkpoint index's `_class_name`.
    ///
    /// The result may be a family described by its root configuration, which
    /// [`ModelDescription::is_pipeline`] distinguishes.
    pub fn from_pipeline_class(class_name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|description| description.pipeline_class() == Some(class_name))
    }
}
