//! The H3 checkpoint's identity: the DiT partitions a release holds, which tasks each serves, and
//! how a FastH3 export or adapter is recognized.
//!
//! A checkpoint is one of two things (`docs/minimax_h3/minimax_h3.md`):
//!
//! - the base release, a pipeline index (`model_index.json`) over the directories
//!   `transformer`, `transformer_ref`, `text_encoder`, `processor`, `vae` and `audio_vae`, of which
//!   the two DiT directories are optional and a deployment places one;
//! - a FastH3 export, whose `fastvideo_inference.json` names its task (so the base DiT partition it
//!   replaces), its schedule and its attention. A FastH3 adapter fused into the base release's
//!   `transformer` reads as the student it was extracted from.
//!
//! The typed architecture every layer reads is [`crate::config::Config`], which reads all of this
//! once at the loading boundary.

use std::collections::BTreeMap;

use nn::family::Stage;

use crate::VideoTask;

/// The diffusion transformers a checkpoint directory may hold, as the deployment stages that
/// place them, in the order a checkpoint's sidecars are read: the base network (`transformer`)
/// for `t2va` and `fl2va` and the reference-conditioned one (`transformer_ref`) for `ref2va`. A
/// FastH3 export holds the single partition it distils. A stage's name is its directory.
pub const DITS: [Stage; 2] = [Stage::Transformer, Stage::TransformerRef];

/// The diffusion transformer that serves `task`.
pub const fn dit(task: VideoTask) -> Stage {
    match task {
        VideoTask::TextToVideoAudio | VideoTask::FirstLastToVideoAudio => Stage::Transformer,
        VideoTask::ReferenceToVideoAudio => Stage::TransformerRef,
    }
}

/// The tasks the diffusion transformer `stage` serves, in canonical order.
pub fn dit_tasks(stage: Stage) -> Vec<VideoTask> {
    VideoTask::ALL
        .into_iter()
        .filter(|task| dit(*task) == stage)
        .collect()
}

/// The pipeline class a load of this crate takes (`model_index.json`'s `_class_name`).
///
/// A task export such as `MiniMax-H3/FL2VA` states `MiniMaxH3Pipeline` and names a `video_vae`
/// directory; it is not a pipeline root.
pub const PIPELINE_CLASS: &str = "MiniMaxH3ModularPipeline";

/// The base model a FastH3 adapter states it edits.
pub const FASTH3_BASE_MODEL: &str = "MiniMaxAI/MiniMax-H3";

/// Whether an adapter's header `metadata` identifies a FastH3 release.
///
/// FastVideo writes its container for ordinary H3 adapters too, so the container alone does not
/// claim one: the adapter must name a student of FastVideo's own namespace that is FastH3
/// (`finetuned_model`, such as `FastVideo/FastVideo-FastH3-4-step-v1`) over the base release
/// (`base_model`). Claiming another adapter would put it on a four-step schedule it was never
/// distilled for.
pub fn fasth3_adapter(metadata: &BTreeMap<String, String>) -> bool {
    let student = metadata
        .get("finetuned_model")
        .map(|name| name.to_lowercase())
        .unwrap_or_default();
    let base = metadata.get("base_model").map(String::as_str);
    metadata.get("format").map(String::as_str) == Some(nn::adapter::FASTVIDEO_FORMAT)
        && student.starts_with("fastvideo/")
        && student.contains("fasth3")
        && base.is_some_and(|base| base.eq_ignore_ascii_case(FASTH3_BASE_MODEL))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_task_runs_on_the_network_that_serves_it() {
        assert_eq!(
            dit_tasks(Stage::Transformer),
            [
                VideoTask::TextToVideoAudio,
                VideoTask::FirstLastToVideoAudio
            ]
        );
        assert_eq!(
            dit_tasks(Stage::TransformerRef),
            [VideoTask::ReferenceToVideoAudio]
        );
    }

    /// A FastVideo container names a FastH3 student of FastVideo's namespace over the base
    /// release; an ordinary H3 adapter in the same container is not one.
    #[test]
    fn a_fasth3_adapter_names_its_student_and_the_base_release() {
        let metadata = |student: &str, base: &str| {
            BTreeMap::from([
                (
                    "format".to_owned(),
                    nn::adapter::FASTVIDEO_FORMAT.to_owned(),
                ),
                ("finetuned_model".to_owned(), student.to_owned()),
                ("base_model".to_owned(), base.to_owned()),
            ])
        };
        assert!(fasth3_adapter(&metadata(
            "FastVideo/FastVideo-FastH3-4-step-v1",
            FASTH3_BASE_MODEL
        )));
        assert!(!fasth3_adapter(&metadata(
            "someone/h3-style",
            FASTH3_BASE_MODEL
        )));
        assert!(!fasth3_adapter(&metadata(
            "FastVideo/FastVideo-FastH3-4-step-v1",
            "other/base"
        )));
    }
}
