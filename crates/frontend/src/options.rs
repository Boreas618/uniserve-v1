//! The frontend's options: the deployment-file values the application reads.
//!
//! The deployment file has one parser, `daemon::options` in the process that
//! owns the file and starts the other processes (`specs/worker.md` §2.3,
//! §7.7). This module is the frontend's side of that boundary: plain data,
//! no parser and no TOML, filled by the root package's `src/frontend.rs` from
//! the daemon's resolved `ServeConfig` and handed to [`crate::serve`].
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

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64};
use std::path::PathBuf;
use std::time::Duration;

use proto::request::{LoadFormat, ModelDtype};
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
    pub load: Load,
    /// The HTTP server (`server_config`).
    pub server: Server,
    /// What the frontend logs about serving (`logging_config`).
    pub logging: Logging,
    /// Chat models (`chat_config`).
    pub chat: Chat,
    /// Fetching input images of chat requests (`image_fetch_config`).
    pub images: Images,
    /// MiniMax-H3 video generation (`video_config`).
    pub video: Video,
    /// Condition media of video requests (`media_config`).
    pub media: Media,
    /// Finished video jobs (`video_job_config`).
    pub video_jobs: VideoJobs,
    /// System One readouts (`readout_config`). The design retires it with the
    /// model families that serve readouts; it is carried while that code
    /// still exists.
    pub readout: Readout,
    /// Movement of token requests between epochs (`migration_config`).
    pub migration: Migration,
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
            load: Load::default(),
            server: Server::default(),
            logging: Logging::default(),
            chat: Chat::default(),
            images: Images::default(),
            video: Video::default(),
            media: Media::default(),
            video_jobs: VideoJobs::default(),
            readout: Readout::default(),
            migration: Migration::default(),
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

/// The checkpoint reader (`load_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Load {
    /// Checkpoint reader: `auto`, `safetensors`, `pt`, `layered` or `dummy`.
    pub format: LoadFormat,
    /// Concurrent checkpoint file readers.
    pub threads: Option<NonZeroU32>,
    /// A plugin's weight source that streams the checkpoint's tensors in
    /// place of its files; the files when unset.
    pub weights: Option<String>,
}

impl Default for Load {
    fn default() -> Self {
        Self {
            format: LoadFormat::Auto,
            threads: None,
            weights: None,
        }
    }
}

/// The HTTP server (`server_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Server {
    /// TCP address the server listens on.
    pub host: String,
    /// TCP port the server listens on.
    pub port: u16,
    /// Unix domain socket to listen on instead of TCP; a stale socket file
    /// is replaced and the socket file is removed at shutdown.
    pub uds: Option<String>,
    /// Bearer token every route but `/health` and `/metrics` requires.
    pub api_key: Option<String>,
    /// Time until a response head is sent, answered with 504 when it
    /// expires; streamed bodies are not bounded.
    pub request_timeout: Option<Duration>,
    /// In-flight generation requests above which requests are shed with 503.
    pub max_concurrent_requests: Option<NonZeroU64>,
    /// Longest drain of in-flight requests at shutdown; zero stops at once.
    pub shutdown_timeout: Duration,
    /// Add an `X-Request-Id` response header with the ID the request ran
    /// under.
    pub request_id_headers: bool,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            uds: None,
            api_key: None,
            request_timeout: None,
            max_concurrent_requests: None,
            shutdown_timeout: Duration::from_secs(30),
            request_id_headers: false,
        }
    }
}

/// What the frontend logs about serving (`logging_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Logging {
    /// Log a summary line for each completed request.
    pub log_requests: bool,
    /// Log scheduler statistics periodically.
    pub log_stats: bool,
}

impl Default for Logging {
    fn default() -> Self {
        Self {
            log_requests: false,
            log_stats: true,
        }
    }
}

/// Chat models (`chat_config`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Chat {
    /// Chat template, as a file path or the template itself.
    pub template: Option<String>,
    /// Keyword arguments bound into every chat-template render.
    pub template_kwargs: Option<HashMap<String, Value>>,
    /// How message content renders.
    pub template_content_format: ChatTemplateContentFormat,
    /// How reasoning is split from content.
    pub reasoning_parser: ReasoningParser,
}

/// How a chat message's content renders into the template
/// (`chat_config.template_content_format`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChatTemplateContentFormat {
    /// Decide from the template.
    #[default]
    Auto,
    /// One string.
    String,
    /// OpenAI's list of typed parts.
    OpenAi,
}

/// Which reasoning parser splits model output (`chat_config.reasoning_parser`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReasoningParser {
    /// The model description's parser splits `reasoning_content` from
    /// `content`.
    #[default]
    Auto,
    /// Reasoning delimiter tokens stream verbatim as content.
    None,
}

/// Fetching input images of chat requests (`image_fetch_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Images {
    /// Time to fetch one `http(s)` image URL, redirects and body included.
    pub timeout: Duration,
    /// Largest input image, fetched or `data:`.
    pub max_bytes: u64,
    /// Allow image URLs resolving to loopback, private, link-local,
    /// unique-local or cloud metadata addresses.
    pub allow_private: bool,
}

impl Default for Images {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(20),
            // 20 MB.
            max_bytes: 20_000_000,
            allow_private: false,
        }
    }
}

/// MiniMax-H3 video generation (`video_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Video {
    /// Longest admitted clip, 4 to 15 seconds.
    pub max_seconds: f64,
    /// Resolution classes startup prepares, each at every aspect ratio.
    pub resolutions: Vec<String>,
    /// Aspect ratios startup prepares.
    pub aspect_ratios: Vec<String>,
    /// Prompt capacities below the prompt limit, strictly increasing.
    pub text_capacities: Option<Vec<NonZeroU32>>,
    /// Denoiser rows one request's conditions may take.
    pub max_condition_rows: Option<u32>,
}

impl Default for Video {
    fn default() -> Self {
        Self {
            max_seconds: models::video::plan::MAX_VIDEO_SECONDS,
            resolutions: vec!["768p".to_owned()],
            aspect_ratios: vec!["16:9".to_owned(), "9:16".to_owned()],
            text_capacities: None,
            max_condition_rows: None,
        }
    }
}

/// Condition media of video requests (`media_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Media {
    /// Directory `file://` condition media resolves under; `file://` is
    /// refused when unset.
    pub directory: Option<PathBuf>,
    /// Fetch `http(s)://` condition media.
    pub allow_remote: bool,
    /// Condition media one request may carry in total.
    pub max_request_bytes: u64,
    /// Executable that probes video and audio condition media.
    pub ffprobe: PathBuf,
    /// Executable the media reader decodes reference videos with.
    pub ffmpeg: PathBuf,
}

impl Default for Media {
    fn default() -> Self {
        Self {
            directory: None,
            allow_remote: true,
            // 256 MiB.
            max_request_bytes: 256 << 20,
            ffprobe: PathBuf::from("ffprobe"),
            ffmpeg: PathBuf::from("ffmpeg"),
        }
    }
}

/// Finished video jobs (`video_job_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct VideoJobs {
    /// Time a finished job stays retrievable.
    pub retention: Duration,
    /// Generated video the jobs keep in shared memory.
    pub max_bytes: usize,
}

impl Default for VideoJobs {
    fn default() -> Self {
        Self {
            retention: Duration::from_secs(3600),
            // 1 GiB.
            max_bytes: 1 << 30,
        }
    }
}

/// System One readouts (`readout_config`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Readout {
    /// `joint` packs questions into shared canvases; `independent` gives
    /// each its own prompt and canvas.
    pub layout: ReadoutLayout,
    /// Canvas length.
    pub canvas: CanvasMode,
    /// Which supported answer spellings are summed.
    pub candidates: CandidateTokens,
}

/// How a readout lays its questions out (`readout_config.layout`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReadoutLayout {
    /// Pack questions into shared canvases.
    #[default]
    Joint,
    /// Give each question its own prompt and canvas.
    Independent,
}

/// The canvas length of a readout (`readout_config.canvas`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CanvasMode {
    /// The canvas the questions fill.
    #[default]
    Full,
    /// The compact canvas.
    Compact,
    /// A fixed multiple of 16 tokens, no longer than the checkpoint's
    /// canvas.
    Fixed(u32),
}

/// Which answer spellings a readout sums (`readout_config.candidates`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CandidateTokens {
    /// Sum the variants of each answer.
    #[default]
    Variants,
    /// Read the space-prefixed one.
    Primary,
}

/// Movement of token requests between epochs (`migration_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Migration {
    /// Migrations caused by lost workers or a failed epoch that a request
    /// survives; it fails on the next.
    pub max_migrations: u32,
}

impl Default for Migration {
    fn default() -> Self {
        Self { max_migrations: 3 }
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
