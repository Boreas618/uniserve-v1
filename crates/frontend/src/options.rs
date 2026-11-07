//! The frontend's options: the deployment-file values the application reads.
//!
//! The deployment file has one parser, `daemon::options` in the process that
//! owns the file and starts the other processes (`specs/worker.md` §2.3,
//! §7.7). The sections the frontend reads are the types that parser fills
//! (`proto::config`), so the root package's `src/frontend.rs` hands each over
//! whole from the daemon's resolved `ServeConfig` to [`crate::serve`].
//!
//! It carries the groups §7.7 gives the frontend (`server_config`,
//! `chat_config`, `image_fetch_config`, `media_config`, `video_job_config`,
//! `migration_config`, `mm_processor_kwargs`, the log flags) and the
//! deployment facts the application itself reads while serving (the model and
//! the name it serves, the prompt limit, the element type, the checkpoint
//! reader, the video plan, the readout, and this node's place in the
//! deployment). §7.7 assigns who reads the deployment file, not who consumes
//! a value: the daemon reads the file once and passes what the frontend
//! needs, so a value's update layer and the application's use of it are
//! independent.

pub use proto::config::{
    ChatConfig, ChatTemplateContentFormatOption, ImageFetchConfig, LoadConfig, LoggingConfig,
    MediaConfig, MigrationConfig, ReasoningParser, ServerConfig, VideoConfig, VideoJobConfig,
};
use std::num::NonZeroU32;

use proto::request::ModelDtype;
use serde_json::Value;

/// Everything the frontend reads from the deployment file.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// Local model directory or Hugging Face repository.
    pub model: String,
    /// The single model name the API serves; the model when unset.
    pub served_model_name: Option<String>,
    /// The configured prompt limit in tokens; the checkpoint's own limit
    /// applies when it is unset.
    pub max_model_len: Option<NonZeroU32>,
    /// Element type of parameters and activations.
    pub dtype: ModelDtype,
    /// Options of the model's multimodal processor.
    pub mm_processor_kwargs: serde_json::Map<String, Value>,
    /// This node's host name, which the frontend names itself to the main
    /// process with.
    pub host_identity: String,
    /// Nodes of the deployment.
    pub nnodes: NonZeroU32,
    /// This node's rank among `nnodes`; rank 0 is the leader, whose frontend
    /// serves the API.
    pub node_rank: u32,
    /// The checkpoint reader (`load_config`).
    pub load_config: LoadConfig,
    /// The HTTP server (`server_config`).
    pub server_config: ServerConfig,
    /// What the frontend logs about serving (`logging_config`).
    pub logging_config: LoggingConfig,
    /// ChatConfig models (`chat_config`).
    pub chat_config: ChatConfig,
    /// Fetching input images of chat requests (`image_fetch_config`).
    pub image_fetch_config: ImageFetchConfig,
    /// MiniMax-H3 video generation (`video_config`).
    pub video_config: VideoConfig,
    /// Condition media of video requests (`media_config`).
    pub media_config: MediaConfig,
    /// Finished video jobs (`video_job_config`).
    pub video_job_config: VideoJobConfig,
    /// Movement of token requests between epochs (`migration_config`).
    pub migration_config: MigrationConfig,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: String::new(),
            served_model_name: None,
            max_model_len: None,
            dtype: ModelDtype::BFloat16,
            mm_processor_kwargs: serde_json::Map::new(),
            host_identity: "localhost".to_owned(),
            nnodes: NonZeroU32::MIN,
            node_rank: 0,
            load_config: LoadConfig::default(),
            server_config: ServerConfig::default(),
            logging_config: LoggingConfig::default(),
            chat_config: ChatConfig::default(),
            image_fetch_config: ImageFetchConfig::default(),
            video_config: VideoConfig::default(),
            media_config: MediaConfig::default(),
            video_job_config: VideoJobConfig::default(),
            migration_config: MigrationConfig::default(),
        }
    }
}

impl Options {
    /// The single model name the API serves: `served_model_name`, else the
    /// model.
    pub fn served_model_name(&self) -> &str {
        self.served_model_name
            .as_deref()
            .unwrap_or_else(|| &self.model)
    }

    /// The longest sequence a request may reach, when the configuration
    /// states one; the checkpoint's own limit applies when it does not.
    pub fn prompt_limit(&self) -> Option<u32> {
        self.max_model_len.map(NonZeroU32::get)
    }

    /// Whether this node is the deployment's leader, whose frontend serves
    /// the API.
    pub fn leads(&self) -> bool {
        self.node_rank == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The deployment facts the application reads: the served name falls back
    /// to the model, and only rank 0 leads the deployment.
    #[test]
    fn the_served_name_defaults_to_the_model_and_only_rank_zero_leads() {
        let options = Options {
            model: "org/model".to_owned(),
            ..Options::default()
        };
        assert_eq!(options.served_model_name(), "org/model");
        let named = Options {
            served_model_name: Some("served".to_owned()),
            ..options
        };
        assert_eq!(named.served_model_name(), "served");
        assert!(named.leads());
        let follower = Options {
            node_rank: 1,
            ..named
        };
        assert!(!follower.leads());
    }
}
