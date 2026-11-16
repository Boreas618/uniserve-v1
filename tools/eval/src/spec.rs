//! The run specification: a profile the caller composes at use, in TOML.
//!
//! A profile holds an optional `root` (default `.`, relative to the file) and
//! `artifact_root` (where `run` writes without `--output-root`), and three
//! tables: `[servers.<name>]`, how to launch a deployment; `[benchmarks.<name>]`,
//! one measured point against a named server, with the changes a timeline load
//! applies as `[[benchmarks.<name>.changes]]`; and `[suites.<name>]`, benchmark
//! names in execution order. The repository holds no profile: each campaign
//! writes its own beside its results, and every bundle records the point it
//! measured (`run.json`).
//!
//! String values in server and benchmark tables may name the process
//! environment as `${NAME}` or `${NAME:-default}` ([`expand_environment`]). Every
//! path a benchmark states is relative to `root` unless absolute: its
//! `dataset_path`, `load.warmup_manifest`, `load.priming_manifest`,
//! `video.media_dir` and `video.reference_root`, and a `tokenizer` that names a
//! directory under `root`. Unknown keys are refused in every table.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::task::TaskName;

/// The endpoint a chat task posts to unless a point names another.
pub const CHAT_COMPLETIONS: &str = "/v1/chat/completions";
/// The image-generation endpoint of the OpenAI images API.
pub const IMAGES_GENERATIONS: &str = "/v1/images/generations";
/// UniServe's synchronous video endpoint, which answers with the MP4.
pub const VIDEOS_SYNC: &str = "/v1/videos/sync";

/// A validated profile.
#[derive(Debug, Clone)]
pub struct Profile {
    /// The directory the profile's relative paths resolve against.
    pub root: PathBuf,
    /// The default parent of the points' result directories.
    pub artifact_root: Option<PathBuf>,
    pub servers: IndexMap<String, Server>,
    pub benchmarks: IndexMap<String, Point>,
    pub suites: IndexMap<String, Vec<String>>,
}

impl Profile {
    /// The points a benchmark or suite name selects, in order; a benchmark
    /// name takes precedence over a suite of the same name.
    ///
    /// # Errors
    ///
    /// When `selection` names neither.
    pub fn selected(&self, selection: &str) -> Result<Vec<&Point>> {
        if let Some(point) = self.benchmarks.get(selection) {
            return Ok(vec![point]);
        }
        let Some(suite) = self.suites.get(selection) else {
            let mut known: Vec<&str> = self
                .benchmarks
                .keys()
                .chain(self.suites.keys())
                .map(String::as_str)
                .collect();
            known.sort_unstable();
            known.dedup();
            bail!(
                "unknown benchmark or suite {selection:?}; known: {}",
                known.join(", ")
            );
        };
        Ok(suite.iter().map(|name| &self.benchmarks[name]).collect())
    }
}

/// One deployment: a single server process, or independent replicas, each a
/// complete single-process server on its own devices, between which the
/// client routes requests.
///
/// A process is ready once its listener accepts a connection and, when
/// `ready_path` names one, that path answers 200. `admin_socket` is the Unix
/// socket of a single-process server's admin listener, which a timeline load's
/// changes call and whose event log it reads; a relative path is taken in the
/// deployment's runtime directory, which the harness creates for it.
#[derive(Debug, Clone, Serialize)]
pub struct Server {
    pub name: String,
    pub command: Vec<String>,
    pub host: String,
    pub port: u16,
    pub environment: BTreeMap<String, String>,
    pub replicas: Vec<Server>,
    pub ready_path: Option<String>,
    pub admin_socket: Option<String>,
}

impl Server {
    /// The single-process servers the deployment is made of.
    pub fn instances(&self) -> Vec<&Server> {
        if self.replicas.is_empty() {
            vec![self]
        } else {
            self.replicas.iter().collect()
        }
    }

    /// The HTTP origin of a single process.
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    /// Every origin requests may be routed to.
    pub fn base_urls(&self) -> Vec<String> {
        self.instances()
            .iter()
            .map(|instance| instance.base_url())
            .collect()
    }
}

/// A protected metric: a dotted path into the summary's `metrics` and the
/// direction in which it improves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Metric {
    pub path: Vec<String>,
    pub direction: Direction,
}

impl Metric {
    /// The dotted path.
    pub fn name(&self) -> String {
        self.path.join(".")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Higher,
    Lower,
}

/// Arrivals, concurrency, warm-up and the seed of a point's load.
///
/// `num_prompts` measured rows resolve exactly; a timeline load
/// (`duration_s`) cycles over them, its request `n` taking row
/// `n mod num_prompts`, issues requests from its start until `duration_s`
/// later (or `settle_s` after its last change ended, when later) and then waits
/// for every stream to end. `request_rate` is the mean Poisson rate in requests
/// per second; unbounded, every request is submitted at once, or a timeline load
/// runs a closed loop of `max_concurrency` streams. `warmup_requests` concurrent
/// requests of the first row complete before measurement, unless a
/// `warmup_manifest` (JSON Lines rows) replaces them. `seed` seeds the datasets
/// that shuffle, the arrival intervals and the generation seed of image or video
/// rows that state none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Load {
    pub num_prompts: usize,
    #[serde(deserialize_with = "rate", serialize_with = "rate_out")]
    pub request_rate: f64,
    pub max_concurrency: Option<usize>,
    pub warmup_requests: usize,
    pub seed: u64,
    pub warmup_manifest: Option<String>,
    pub priming_manifest: Option<String>,
    pub request_timeout_s: f64,
    pub duration_s: Option<f64>,
    pub settle_s: f64,
}

impl Default for Load {
    fn default() -> Self {
        Self {
            num_prompts: 1000,
            request_rate: f64::INFINITY,
            max_concurrency: None,
            warmup_requests: 1,
            seed: 42,
            warmup_manifest: None,
            priming_manifest: None,
            request_timeout_s: 6.0 * 60.0 * 60.0,
            duration_s: None,
            settle_s: 0.0,
        }
    }
}

impl Load {
    fn check(&self) -> Result<()> {
        ensure!(self.num_prompts >= 1, "num_prompts must be positive");
        ensure!(
            self.request_rate > 0.0 && !self.request_rate.is_nan(),
            "request_rate must be positive"
        );
        ensure!(
            self.max_concurrency != Some(0),
            "max_concurrency must be positive"
        );
        ensure!(
            self.request_timeout_s.is_finite() && self.request_timeout_s > 0.0,
            "request_timeout_s must be finite and positive"
        );
        if let Some(duration) = self.duration_s {
            ensure!(
                duration.is_finite() && duration > 0.0,
                "duration_s must be finite and positive"
            );
            // Unbounded arrivals over a duration would never stop arriving;
            // a closed loop's streams are its concurrency.
            ensure!(
                self.request_rate.is_finite() || self.max_concurrency.is_some(),
                "a timeline load at an unbounded request_rate is a closed loop and needs \
                 max_concurrency, its streams"
            );
        }
        ensure!(
            self.settle_s.is_finite() && self.settle_s >= 0.0,
            "settle_s must be finite and non-negative"
        );
        ensure!(
            self.settle_s == 0.0 || self.duration_s.is_some(),
            "settle_s applies only with duration_s"
        );
        Ok(())
    }

    /// Whether the load is a timeline load.
    pub fn timeline(&self) -> bool {
        self.duration_s.is_some()
    }
}

/// `request_rate` is a number, TOML's `inf`, or the string `"inf"`.
fn rate<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Rate {
        Number(f64),
        Text(String),
    }
    match Rate::deserialize(deserializer)? {
        Rate::Number(value) => Ok(value),
        Rate::Text(text) if text == "inf" => Ok(f64::INFINITY),
        Rate::Text(text) => Err(serde::de::Error::custom(format!(
            "request_rate must be a number or \"inf\", got {text:?}"
        ))),
    }
}

/// JSON has no infinity: an unbounded rate is written as `"inf"`.
fn rate_out<S: serde::Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if value.is_infinite() {
        serializer.serialize_str("inf")
    } else {
        serializer.serialize_f64(*value)
    }
}

/// One change a timeline load applies to the deployment as it runs.
///
/// A change acts once: `at_s` seconds after the load's start, or `delay_s`
/// seconds after the earlier change `after` ended; with `phase`, the delay
/// counts from the first admin event, recorded after `after` was issued, that
/// reports the update entering that phase (`trial` is the end of a switch). Its
/// action is exactly one of an admin call (`method`, `path`, an optional JSON
/// `body`), which succeeds on a 2xx answer; a `signal` to the one process whose
/// command line matches `process` (with `host`, found and signalled there over
/// ssh); a `command`, which succeeds on exit status 0; or `cancel`, that many of
/// a closed loop's streams disconnected by the client at once, the earliest
/// issued or, with `cancel_order = "newest"`, the latest. `configuration` names
/// the configuration the deployment serves once the change ended, which
/// `check-change` compares the requests after it with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Change {
    pub name: String,
    #[serde(default)]
    pub at_s: Option<f64>,
    #[serde(default)]
    pub after: Option<String>,
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(default)]
    pub delay_s: f64,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub body: Option<serde_json::Map<String, Value>>,
    #[serde(default)]
    pub signal: Option<String>,
    #[serde(default)]
    pub process: Option<String>,
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub cancel: Option<usize>,
    #[serde(default = "oldest")]
    pub cancel_order: String,
    #[serde(default)]
    pub configuration: Option<String>,
}

fn oldest() -> String {
    "oldest".to_owned()
}

/// The signals a change may send, by name.
pub const SIGNALS: &[(&str, i32)] = &[
    ("SIGSTOP", libc::SIGSTOP),
    ("SIGCONT", libc::SIGCONT),
    ("SIGTERM", libc::SIGTERM),
    ("SIGKILL", libc::SIGKILL),
    ("SIGINT", libc::SIGINT),
    ("SIGHUP", libc::SIGHUP),
    ("SIGUSR1", libc::SIGUSR1),
    ("SIGUSR2", libc::SIGUSR2),
    ("SIGQUIT", libc::SIGQUIT),
];

impl Change {
    fn check(&self) -> Result<()> {
        ensure!(!self.name.is_empty(), "a change needs a name");
        ensure!(
            self.at_s.is_some() != self.after.is_some(),
            "a change states exactly one of at_s and after"
        );
        if let Some(at) = self.at_s {
            ensure!(
                at.is_finite() && at >= 0.0,
                "at_s must be finite and non-negative"
            );
        }
        ensure!(
            self.delay_s.is_finite() && self.delay_s >= 0.0,
            "delay_s must be finite and non-negative"
        );
        ensure!(
            self.after.is_some() || (self.phase.is_none() && self.delay_s == 0.0),
            "phase and delay_s count from the change after"
        );
        let actions = [
            self.path.is_some(),
            self.signal.is_some(),
            !self.command.is_empty(),
            self.cancel.is_some(),
        ];
        ensure!(
            actions.iter().filter(|stated| **stated).count() == 1,
            "a change states exactly one of path, signal, command and cancel"
        );
        if let Some(path) = &self.path {
            ensure!(path.starts_with('/'), "path must be an absolute URL path");
        }
        ensure!(
            self.path.is_some() == self.method.is_some(),
            "method and path go together"
        );
        if let Some(method) = &self.method {
            ensure!(
                ["GET", "POST", "PUT", "PATCH", "DELETE"].contains(&method.as_str()),
                "unknown HTTP method {method:?}"
            );
        }
        ensure!(
            self.signal.is_some() == self.process.is_some(),
            "signal and process go together"
        );
        if let Some(signal) = &self.signal {
            ensure!(
                SIGNALS.iter().any(|(name, _)| name == signal),
                "unknown signal {signal:?}"
            );
        }
        if let Some(process) = &self.process {
            regex::Regex::new(process).context("process is not a pattern")?;
        }
        // A host is an ssh destination; one that starts with "-" would be read
        // as an option.
        if let Some(host) = &self.host {
            ensure!(
                self.signal.is_some() && !host.is_empty() && !host.starts_with('-'),
                "host names the host of a signal's process"
            );
        }
        ensure!(
            self.cancel != Some(0),
            "cancel must be a positive number of streams"
        );
        ensure!(
            ["oldest", "newest"].contains(&self.cancel_order.as_str()),
            "cancel_order must be \"oldest\" or \"newest\""
        );
        ensure!(
            self.cancel_order == "oldest" || self.cancel.is_some(),
            "cancel_order applies only with cancel"
        );
        ensure!(
            self.configuration.as_deref() != Some(""),
            "configuration must be a non-empty name"
        );
        Ok(())
    }

    /// Whether the change uses the admin listener: its call or its events.
    pub fn calls_admin(&self) -> bool {
        self.path.is_some() || self.phase.is_some()
    }
}

/// Model sampling and the request fields a task adds beside it.
///
/// Chat requests carry the sampling parameters, optional ones left out when
/// unset, and `extra_body` merged last, so its keys override any field the task
/// sets. Image-generation requests carry only `extra_body`, video requests none
/// of these. `max_tokens` is the output limit of rows without their own. With
/// `ignore_eos`, the text and i2t tasks also hold the output to its fixed length.
/// `server_sampling = true` in a profile states that the server's own schedule
/// governs every request: `temperature`, `top_p` and `ignore_eos` are then unset
/// and no token-sampling control may be named.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Sampling {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub sampling_seed: Option<i64>,
    pub ignore_eos: Option<bool>,
    pub max_tokens: Option<u64>,
    pub stream: bool,
    pub extra_body: serde_json::Map<String, Value>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            sampling_seed: None,
            ignore_eos: Some(true),
            max_tokens: None,
            stream: true,
            extra_body: serde_json::Map::new(),
        }
    }
}

/// The request-level controls a point with server-owned sampling may not set:
/// every token-sampling parameter and `ignore_eos`. A seed only selects the
/// random stream the server's schedule draws from.
const TOKEN_SAMPLING_CONTROLS: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "ignore_eos",
];

/// Image geometry, denoising, guidance and output count.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Image {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub image_count: Option<u32>,
    pub guidance_scale: Option<f64>,
    pub image_guidance_scale: Option<f64>,
    pub cfg_norm: Option<String>,
    pub cfg_interval: Option<[f64; 2]>,
    pub timestep_shift: Option<f64>,
}

impl Image {
    fn check(&self) -> Result<()> {
        ensure!(self.steps != Some(0), "steps must be positive");
        ensure!(self.image_count != Some(0), "image_count must be positive");
        Ok(())
    }

    /// Whether the point states any image setting.
    pub fn is_stated(&self) -> bool {
        *self != Self::default()
    }
}

/// The serving system a video point asks, by the API its requests take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VideoBackend {
    /// UniServe's `/v1/videos` body, sent as is.
    #[default]
    Uniserve,
    /// vLLM-Omni's multipart form with its `extra_params` field.
    VllmOmni,
    /// SGLang's native JSON request.
    Sglang,
    /// FastVideo's request.
    Fastvideo,
}

impl VideoBackend {
    /// The schedule fields the backend receives. A baseline runs whatever
    /// schedule it is sent, so a point measuring one states each of these;
    /// FastVideo takes the sigma point count alone, its shifts coming from the
    /// checkpoint; UniServe's checkpoint fixes its schedule.
    fn schedule_fields(self) -> &'static [&'static str] {
        match self {
            Self::Uniserve => &[],
            Self::Sglang | Self::VllmOmni => &VIDEO_SCHEDULE_FIELDS,
            Self::Fastvideo => &["num_inference_steps"],
        }
    }

    /// The requested-target fields the backend's native form takes beside or
    /// instead of the size and frame count.
    fn target_fields(self) -> &'static [&'static str] {
        match self {
            Self::Uniserve => &[],
            Self::Sglang | Self::VllmOmni => &["aspect_ratio", "duration_seconds"],
            Self::Fastvideo => &["duration_seconds"],
        }
    }

    /// The request fields that define the generated work, which the video task
    /// sets from each request and `extra_params` therefore may not; `None` for a
    /// backend that takes no serving options.
    pub fn work_fields(self) -> Option<&'static [&'static str]> {
        match self {
            Self::VllmOmni => Some(&["task", "duration", "audio_flow_shift"]),
            Self::Sglang => Some(&[
                "model",
                "prompt",
                "seed",
                "task",
                "conditions",
                "target",
                "num_inference_steps",
                "flow_shift",
                "audio_flow_shift",
            ]),
            Self::Uniserve | Self::Fastvideo => None,
        }
    }
}

/// The schedule fields a point may state: sigma points including the clean
/// endpoint, then the video and audio schedule shifts.
pub const VIDEO_SCHEDULE_FIELDS: [&str; 3] =
    ["num_inference_steps", "flow_shift", "audio_flow_shift"];

/// The generated video work and how a backend is asked for it.
///
/// A point states the output shape it measures: `size` (`"WxH"`, one of the
/// deployment's served sizes), `fps`, and the soundtrack's `audio_sample_rate`
/// and `audio_channels`; `num_frames` is the exact frame count of rows that
/// state none, sent verbatim. A baseline is asked for the same work in its own
/// form: `aspect_ratio` and `duration_seconds` are the requested target its form
/// names, stating the ratio and duration that resolve to the point's shape, and
/// `num_inference_steps`, `flow_shift` and `audio_flow_shift` the schedule it
/// runs; each backend states exactly the fields it takes. `reference_root` is
/// the directory the media paths of rows' `input_references` are relative to.
/// `parallel_decoding` marks a FastVideo point of a parallel-decoding student,
/// whose schedule FastVideo counts in fused-block forwards. `prompt_tokens` is
/// the length of prompts a fixed-length dataset synthesizes. `extra_params` adds
/// serving options a backend accepts to every request, never the fields that fix
/// the work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Video {
    pub size: Option<String>,
    pub num_frames: Option<u32>,
    pub fps: Option<u32>,
    pub audio_sample_rate: Option<u32>,
    pub audio_channels: Option<u32>,
    pub aspect_ratio: Option<String>,
    pub duration_seconds: Option<f64>,
    pub num_inference_steps: Option<u32>,
    pub flow_shift: Option<f64>,
    pub audio_flow_shift: Option<f64>,
    pub prompt_tokens: usize,
    pub backend: VideoBackend,
    pub poll_interval_s: f64,
    pub media_dir: Option<String>,
    pub reference_root: Option<String>,
    pub parallel_decoding: bool,
    pub extra_params: serde_json::Map<String, Value>,
}

impl Default for Video {
    fn default() -> Self {
        Self {
            size: None,
            num_frames: None,
            fps: None,
            audio_sample_rate: None,
            audio_channels: None,
            aspect_ratio: None,
            duration_seconds: None,
            num_inference_steps: None,
            flow_shift: None,
            audio_flow_shift: None,
            prompt_tokens: 1000,
            backend: VideoBackend::Uniserve,
            poll_interval_s: 0.1,
            media_dir: None,
            reference_root: None,
            parallel_decoding: false,
            extra_params: serde_json::Map::new(),
        }
    }
}

impl Video {
    fn check(&self) -> Result<()> {
        if let Some(size) = &self.size {
            parse_size(size)?;
        }
        for (name, value) in [
            ("num_frames", self.num_frames),
            ("fps", self.fps),
            ("audio_sample_rate", self.audio_sample_rate),
            ("audio_channels", self.audio_channels),
        ] {
            ensure!(value != Some(0), "video {name} must be a positive integer");
        }
        ensure!(
            self.prompt_tokens >= 1,
            "video prompt_tokens must be positive"
        );
        ensure!(
            self.poll_interval_s.is_finite() && self.poll_interval_s > 0.0,
            "poll_interval_s must be finite and positive"
        );
        // Only FastVideo's request counts steps by the checkpoint's kind.
        ensure!(
            !self.parallel_decoding || self.backend == VideoBackend::Fastvideo,
            "video parallel_decoding applies only to fastvideo"
        );

        // One interval between the noise and the clean endpoint is the
        // shortest schedule.
        if let Some(steps) = self.num_inference_steps {
            ensure!(
                steps >= 2,
                "video num_inference_steps counts sigma points including the clean endpoint \
                 and must be an integer of at least 2"
            );
        }
        for (name, shift) in [
            ("flow_shift", self.flow_shift),
            ("audio_flow_shift", self.audio_flow_shift),
        ] {
            if let Some(shift) = shift {
                ensure!(
                    shift.is_finite() && shift > 0.0,
                    "video {name} must be finite and positive"
                );
            }
        }
        let stated_schedule: Vec<&str> = VIDEO_SCHEDULE_FIELDS
            .into_iter()
            .filter(|name| self.schedule().contains_key(*name))
            .collect();
        stated_exactly(
            self.backend,
            &stated_schedule,
            self.backend.schedule_fields(),
            "runs the schedule it is sent and requires",
        )?;

        let mut stated_target = Vec::new();
        if self.aspect_ratio.is_some() {
            stated_target.push("aspect_ratio");
        }
        if let Some(seconds) = self.duration_seconds {
            ensure!(
                seconds.is_finite() && seconds > 0.0,
                "video duration_seconds must be finite and positive"
            );
            stated_target.push("duration_seconds");
        }
        stated_exactly(
            self.backend,
            &stated_target,
            self.backend.target_fields(),
            "names a requested target and requires",
        )?;

        match self.backend.work_fields() {
            None => ensure!(
                self.extra_params.is_empty(),
                "video extra_params apply only to sglang and vllm-omni"
            ),
            Some(fixed) => {
                let mut named: Vec<&str> = fixed
                    .iter()
                    .copied()
                    .filter(|field| self.extra_params.contains_key(*field))
                    .collect();
                named.sort_unstable();
                ensure!(named.is_empty(), "video extra_params may not set {named:?}");
            }
        }
        Ok(())
    }

    /// The stated schedule fields.
    pub fn schedule(&self) -> serde_json::Map<String, Value> {
        let mut schedule = serde_json::Map::new();
        if let Some(steps) = self.num_inference_steps {
            schedule.insert("num_inference_steps".into(), steps.into());
        }
        if let Some(shift) = self.flow_shift {
            schedule.insert("flow_shift".into(), shift.into());
        }
        if let Some(shift) = self.audio_flow_shift {
            schedule.insert("audio_flow_shift".into(), shift.into());
        }
        schedule
    }

    /// Whether the point states any video setting.
    pub fn is_stated(&self) -> bool {
        *self != Self::default()
    }
}

/// Refuses a backend that is missing one of `required` or is sent another.
fn stated_exactly(
    backend: VideoBackend,
    stated: &[&str],
    required: &[&str],
    missing_phrase: &str,
) -> Result<()> {
    let name = serde_json::to_value(backend)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_default();
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|field| !stated.contains(field))
        .collect();
    ensure!(
        missing.is_empty(),
        "video backend {name} {missing_phrase} {missing:?}"
    );
    let unsent: Vec<&str> = stated
        .iter()
        .copied()
        .filter(|field| !required.contains(field))
        .collect();
    ensure!(
        unsent.is_empty(),
        "video backend {name} cannot receive {unsent:?}"
    );
    Ok(())
}

/// The `(width, height)` of a `"WxH"` size in pixels.
///
/// # Errors
///
/// When the size is not two positive decimal integers joined by `x`.
pub fn parse_size(size: &str) -> Result<(u32, u32)> {
    let parsed = size.split_once('x').and_then(|(width, height)| {
        let digits =
            |text: &str| !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
        if !digits(width) || !digits(height) {
            return None;
        }
        Some((width.parse::<u32>().ok()?, height.parse::<u32>().ok()?))
    });
    match parsed {
        Some((width, height)) if width > 0 && height > 0 => Ok((width, height)),
        _ => bail!("video size must be WxH in pixels, got {size:?}"),
    }
}

/// One fully resolved measurement point: its workload, task, server and the
/// metrics it protects.
#[derive(Debug, Clone, Serialize)]
pub struct Point {
    pub name: String,
    pub server: String,
    pub task: TaskName,
    pub model: String,
    pub dataset: String,
    pub dataset_revision: Option<String>,
    pub dataset_path: Option<String>,
    pub tokenizer: Option<String>,
    pub endpoint: String,
    pub question: Option<String>,
    pub provenance: serde_json::Map<String, Value>,
    pub load: Load,
    pub sampling: Sampling,
    pub image: Image,
    pub video: Video,
    #[serde(serialize_with = "metrics_out")]
    pub metrics: Vec<Metric>,
    /// The changes a timeline load applies, in the profile's order; an `after`
    /// names a change before it.
    pub changes: Vec<Change>,
}

fn metrics_out<S: serde::Serializer>(metrics: &[Metric], serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let mut sequence = serializer.serialize_seq(Some(metrics.len()))?;
    for metric in metrics {
        sequence.serialize_element(&serde_json::json!({
            "path": metric.name(),
            "direction": metric.direction,
        }))?;
    }
    sequence.end()
}

impl Point {
    /// The point as a JSON value: what a bundle's `run.json` records of it.
    pub fn workload(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// Loads and validates a profile.
///
/// Environment references are expanded here; an unset `${NAME}` without a
/// default stays in the value, which `plan` shows and `run` refuses
/// ([`require_resolved`]).
///
/// # Errors
///
/// The first violation of the schema, naming its table.
pub fn load(path: &Path) -> Result<Profile> {
    let path = path
        .canonicalize()
        .with_context(|| format!("reading the profile {}", path.display()))?;
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading the profile {}", path.display()))?;
    let raw: toml::Table =
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let raw = to_json(toml::Value::Table(raw));
    let raw = object(&raw, &path.display().to_string())?;
    reject_unknown(
        raw,
        &["root", "artifact_root", "servers", "benchmarks", "suites"],
        &path.display().to_string(),
    )?;
    let parent = path.parent().unwrap_or(Path::new("/"));
    let root_value = raw.get("root").and_then(Value::as_str).unwrap_or(".");
    let root = parent
        .join(root_value)
        .canonicalize()
        .with_context(|| format!("the profile's root {root_value}"))?;

    let mut servers = IndexMap::new();
    for (name, value) in table(raw, "servers")? {
        servers.insert(name.clone(), server(name, value)?);
    }
    let mut benchmarks = IndexMap::new();
    for (name, value) in table(raw, "benchmarks")? {
        benchmarks.insert(name.clone(), point(name, value, &servers, &root)?);
    }
    own_media(&benchmarks)?;
    let mut suites = IndexMap::new();
    for (name, value) in table(raw, "suites")? {
        suites.insert(name.clone(), suite(name, value, &benchmarks)?);
    }
    let artifact_root = raw
        .get("artifact_root")
        .and_then(Value::as_str)
        .map(|artifact| root.join(artifact));
    Ok(Profile {
        root,
        artifact_root,
        servers,
        benchmarks,
        suites,
    })
}

/// TOML's values as JSON values: tables become objects in their order.
fn to_json(value: toml::Value) -> Value {
    match value {
        toml::Value::String(text) => Value::String(text),
        toml::Value::Integer(number) => Value::from(number),
        toml::Value::Float(number) => serde_json::Number::from_f64(number)
            .map(Value::Number)
            // JSON has no infinity: the one infinite value a profile states is
            // an unbounded request rate, which reads `"inf"` too.
            .unwrap_or_else(|| Value::String(if number > 0.0 { "inf" } else { "nan" }.into())),
        toml::Value::Boolean(flag) => Value::Bool(flag),
        toml::Value::Datetime(datetime) => Value::String(datetime.to_string()),
        toml::Value::Array(items) => Value::Array(items.into_iter().map(to_json).collect()),
        toml::Value::Table(entries) => Value::Object(
            entries
                .into_iter()
                .map(|(key, item)| (key, to_json(item)))
                .collect(),
        ),
    }
}

/// Expands `${NAME:-default}` to `NAME`'s value when it is set (empty
/// included) and `default` otherwise, and `${NAME}` to `NAME`'s value when it
/// is set, leaving it in place otherwise; recursively through arrays and object
/// values.
pub fn expand_environment(value: &Value) -> Value {
    match value {
        Value::String(text) => Value::String(expand_text(text)),
        Value::Array(items) => Value::Array(items.iter().map(expand_environment).collect()),
        Value::Object(entries) => Value::Object(
            entries
                .iter()
                .map(|(key, item)| (key.clone(), expand_environment(item)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// One string's references expanded. The default of `${NAME:-default}` ends at
/// the first `}`.
fn expand_text(text: &str) -> String {
    let with_default = environment_with_default();
    let expanded = with_default.replace_all(text, |captures: &regex::Captures<'_>| {
        std::env::var(&captures[1]).unwrap_or_else(|_| captures[2].to_owned())
    });
    environment_reference()
        .replace_all(&expanded, |captures: &regex::Captures<'_>| {
            std::env::var(&captures[1]).unwrap_or_else(|_| captures[0].to_owned())
        })
        .into_owned()
}

#[expect(
    clippy::expect_used,
    reason = "the pattern is a constant that compiles"
)]
fn environment_reference() -> regex::Regex {
    regex::Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}").expect("a valid pattern")
}

#[expect(
    clippy::expect_used,
    reason = "the pattern is a constant that compiles"
)]
fn environment_with_default() -> regex::Regex {
    regex::Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-(.*?)\}").expect("a valid pattern")
}

/// The environment names still referenced in a value, sorted.
pub fn unresolved(value: &Value) -> Vec<String> {
    let mut names = std::collections::BTreeSet::new();
    collect_unresolved(value, &mut names);
    names.into_iter().collect()
}

fn collect_unresolved(value: &Value, names: &mut std::collections::BTreeSet<String>) {
    match value {
        Value::String(text) => {
            for captures in environment_reference().captures_iter(text) {
                names.insert(captures[1].to_owned());
            }
        }
        Value::Array(items) => items
            .iter()
            .for_each(|item| collect_unresolved(item, names)),
        Value::Object(entries) => entries
            .values()
            .for_each(|item| collect_unresolved(item, names)),
        _ => {}
    }
}

/// Refuses a value that still names an unset environment variable.
///
/// # Errors
///
/// Naming `context` and every unresolved name.
pub fn require_resolved(value: &Value, context: &str) -> Result<()> {
    let names = unresolved(value);
    ensure!(
        names.is_empty(),
        "{context} has unresolved environment variables: {}",
        names.join(", ")
    );
    Ok(())
}

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a serde_json::Map<String, Value>> {
    value
        .as_object()
        .with_context(|| format!("{context} must be a TOML table"))
}

fn table<'a>(
    raw: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<Vec<(&'a String, &'a Value)>> {
    match raw.get(key) {
        None => Ok(Vec::new()),
        Some(value) => Ok(object(value, key)?.iter().collect()),
    }
}

fn reject_unknown(
    value: &serde_json::Map<String, Value>,
    allowed: &[&str],
    context: &str,
) -> Result<()> {
    let mut unknown: Vec<&str> = value
        .keys()
        .map(String::as_str)
        .filter(|key| !allowed.contains(key))
        .collect();
    unknown.sort_unstable();
    ensure!(
        unknown.is_empty(),
        "{context} has unknown fields: {}",
        unknown.join(", ")
    );
    Ok(())
}

/// One `[servers.<name>]` table. A table with `replicas` lists two or more
/// single-process tables, each inheriting the table's `host` and `ready_path`
/// unless it names its own and its `environment` beneath its own, on distinct
/// listeners.
fn server(name: &str, raw: &Value) -> Result<Server> {
    let context = format!("servers.{name}");
    let expanded = expand_environment(raw);
    let value = object(&expanded, &context)?;
    reject_unknown(
        value,
        &[
            "command",
            "host",
            "port",
            "environment",
            "ready_path",
            "replicas",
            "admin_socket",
        ],
        &context,
    )?;
    let Some(replicas) = value.get("replicas") else {
        return process(name, value, &context);
    };
    ensure!(
        !["command", "port", "admin_socket"]
            .iter()
            .any(|key| value.contains_key(*key)),
        "{context} declares replicas and cannot also declare a command, port or admin_socket"
    );
    let replicas = replicas
        .as_array()
        .filter(|items| items.len() >= 2)
        .with_context(|| format!("{context}.replicas must list two or more"))?;
    let host = host(value, &context)?;
    let shared = environment(value, &context)?;
    let ready_path = ready_path(value, &context)?;
    let mut instances = Vec::new();
    for (index, item) in replicas.iter().enumerate() {
        let where_ = format!("{context}.replicas[{index}]");
        let item = object(item, &where_)?;
        reject_unknown(
            item,
            &["command", "host", "port", "environment", "ready_path"],
            &where_,
        )?;
        let mut merged = serde_json::Map::new();
        merged.insert("host".into(), Value::String(host.clone()));
        if let Some(path) = &ready_path {
            merged.insert("ready_path".into(), Value::String(path.clone()));
        }
        for (key, item) in item {
            merged.insert(key.clone(), item.clone());
        }
        let mut environment = shared.clone();
        environment.extend(self::environment(item, &where_)?);
        merged.insert(
            "environment".into(),
            Value::Object(
                environment
                    .into_iter()
                    .map(|(key, item)| (key, Value::String(item)))
                    .collect(),
            ),
        );
        instances.push(process(
            &format!("{name}.replica-{index}"),
            &merged,
            &where_,
        )?);
    }
    let mut listeners: Vec<(String, u16)> = instances
        .iter()
        .map(|item| (item.host.clone(), item.port))
        .collect();
    listeners.sort();
    listeners.dedup();
    ensure!(
        listeners.len() == instances.len(),
        "{context}.replicas must use distinct ports"
    );
    Ok(Server {
        name: name.to_owned(),
        command: instances[0].command.clone(),
        host,
        port: instances[0].port,
        environment: shared,
        replicas: instances,
        ready_path,
        admin_socket: None,
    })
}

fn process(name: &str, value: &serde_json::Map<String, Value>, context: &str) -> Result<Server> {
    let command: Vec<String> = value
        .get("command")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .and_then(|items| {
            items
                .iter()
                .map(|item| item.as_str().map(str::to_owned))
                .collect()
        })
        .with_context(|| format!("{context}.command must be a non-empty string array"))?;
    let port = value
        .get("port")
        .and_then(Value::as_u64)
        .filter(|port| (1..65536).contains(port))
        .with_context(|| format!("{context}.port must be a valid TCP port"))?;
    let admin_socket = match value.get("admin_socket") {
        None => None,
        Some(Value::String(socket)) if !socket.is_empty() => Some(socket.clone()),
        Some(_) => bail!("{context}.admin_socket must be a non-empty string"),
    };
    Ok(Server {
        name: name.to_owned(),
        command,
        host: host(value, context)?,
        port: u16::try_from(port)?,
        environment: environment(value, context)?,
        replicas: Vec::new(),
        ready_path: ready_path(value, context)?,
        admin_socket,
    })
}

fn ready_path(value: &serde_json::Map<String, Value>, context: &str) -> Result<Option<String>> {
    match value.get("ready_path") {
        None => Ok(None),
        Some(Value::String(path)) if path.starts_with('/') => Ok(Some(path.clone())),
        Some(_) => bail!("{context}.ready_path must be an absolute URL path"),
    }
}

fn host(value: &serde_json::Map<String, Value>, context: &str) -> Result<String> {
    match value.get("host") {
        None => Ok("127.0.0.1".to_owned()),
        Some(Value::String(host)) if !host.is_empty() => Ok(host.clone()),
        Some(_) => bail!("{context}.host must be a non-empty string"),
    }
}

/// A table's launch environment. Each deployment runs under a runtime
/// directory the harness creates and removes, so its host services end with
/// it: a profile may not state `XDG_RUNTIME_DIR`.
fn environment(
    value: &serde_json::Map<String, Value>,
    context: &str,
) -> Result<BTreeMap<String, String>> {
    let Some(raw) = value.get("environment") else {
        return Ok(BTreeMap::new());
    };
    let entries = raw
        .as_object()
        .and_then(|entries| {
            entries
                .iter()
                .map(|(key, item)| item.as_str().map(|item| (key.clone(), item.to_owned())))
                .collect::<Option<BTreeMap<_, _>>>()
        })
        .with_context(|| format!("{context}.environment must contain string values"))?;
    ensure!(
        !entries.contains_key("XDG_RUNTIME_DIR"),
        "{context}.environment may not set XDG_RUNTIME_DIR: each deployment runs under a runtime \
         directory of its own"
    );
    Ok(entries)
}

/// The path-valued fields of a benchmark table, by the nested table that holds
/// them (`None` for the benchmark table itself).
const PATH_FIELDS: &[(Option<&str>, &[&str])] = &[
    (None, &["dataset_path"]),
    (Some("load"), &["warmup_manifest", "priming_manifest"]),
    (Some("video"), &["media_dir", "reference_root"]),
];

/// A benchmark table whose relative paths are joined to `root`. An empty path
/// stays empty (unset, as `${NAME:-}` leaves it) and an unresolved one stays as
/// written. A relative `tokenizer` becomes the directory under `root` when that
/// directory exists, and stays a repository id otherwise.
fn resolve_paths(value: &mut serde_json::Map<String, Value>, root: &Path) {
    let rooted = |path: &str| -> String {
        if path.is_empty()
            || !unresolved(&Value::String(path.to_owned())).is_empty()
            || Path::new(path).is_absolute()
        {
            path.to_owned()
        } else {
            root.join(path).display().to_string()
        }
    };
    for (nested, fields) in PATH_FIELDS {
        let target = match nested {
            None => Some(&mut *value),
            Some(key) => value.get_mut(*key).and_then(Value::as_object_mut),
        };
        let Some(target) = target else { continue };
        for field in *fields {
            if let Some(Value::String(path)) = target.get(*field) {
                let resolved = rooted(path);
                target.insert((*field).to_owned(), Value::String(resolved));
            }
        }
    }
    if let Some(Value::String(tokenizer)) = value.get("tokenizer")
        && !tokenizer.is_empty()
        && !Path::new(tokenizer).is_absolute()
        && root.join(tokenizer).is_dir()
    {
        let resolved = root.join(tokenizer).display().to_string();
        value.insert("tokenizer".into(), Value::String(resolved));
    }
}

fn settings<T: serde::de::DeserializeOwned + Default>(
    value: Option<&Value>,
    context: &str,
) -> Result<T> {
    match value {
        None => Ok(T::default()),
        Some(value) => {
            object(value, context)?;
            serde_json::from_value(value.clone())
                .map_err(|error| anyhow::anyhow!("{context}: {error}"))
        }
    }
}

/// One `[benchmarks.<name>]` table, which the task and dataset it names accept.
fn point(
    name: &str,
    raw: &Value,
    servers: &IndexMap<String, Server>,
    root: &Path,
) -> Result<Point> {
    let context = format!("benchmarks.{name}");
    let expanded = expand_environment(raw);
    let mut value = object(&expanded, &context)?.clone();
    reject_unknown(
        &value,
        &[
            "server",
            "task",
            "model",
            "dataset",
            "dataset_revision",
            "dataset_path",
            "tokenizer",
            "endpoint",
            "question",
            "load",
            "sampling",
            "image",
            "video",
            "metrics",
            "provenance",
            "changes",
        ],
        &context,
    )?;
    resolve_paths(&mut value, root);
    let server_name = value
        .get("server")
        .and_then(Value::as_str)
        .filter(|server| servers.contains_key(*server))
        .with_context(|| format!("{context}.server must name a declared server"))?;
    let task: TaskName = value
        .get("task")
        .and_then(Value::as_str)
        .with_context(|| format!("{context}.task must be a string"))?
        .parse()?;
    let dataset = value
        .get("dataset")
        .and_then(Value::as_str)
        .filter(|dataset| !dataset.is_empty())
        .with_context(|| format!("{context}.dataset must be a non-empty string"))?;
    crate::dataset::Kind::of(dataset)?;
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
        .with_context(|| format!("{context}.model must be a non-empty string"))?;

    let metrics = metrics(value.get("metrics"), &context)?;
    let load: Load = settings(value.get("load"), &format!("{context}.load"))?;
    load.check()
        .map_err(|error| anyhow::anyhow!("{context}.load: {error}"))?;
    let changes = changes(value.get("changes"), &servers[server_name], &load, &context)?;
    let sampling = sampling(value.get("sampling"), task, &format!("{context}.sampling"))?;
    let image: Image = settings(value.get("image"), &format!("{context}.image"))?;
    image
        .check()
        .map_err(|error| anyhow::anyhow!("{context}.image: {error}"))?;
    let video: Video = settings(value.get("video"), &format!("{context}.video"))?;
    video
        .check()
        .map_err(|error| anyhow::anyhow!("{context}.video: {error}"))?;

    let string = |key: &str| -> Result<Option<String>> {
        match value.get(key) {
            None => Ok(None),
            Some(Value::String(text)) => Ok(Some(text.clone())),
            Some(_) => bail!("{context}.{key} must be a string"),
        }
    };
    let endpoint = task.check_endpoint(string("endpoint")?, &context)?;
    let question = task.check_question(string("question")?, &context)?;
    let tokenizer = string("tokenizer")?;
    let dataset_path = string("dataset_path")?;
    crate::dataset::Kind::of(dataset)?.check_point(
        tokenizer.as_deref(),
        dataset_path.as_deref(),
        &context,
    )?;
    task.check_image(&image, &context)?;
    task.check_video(&video, &format!("{context}.video"))?;
    let provenance = match value.get("provenance") {
        None => serde_json::Map::new(),
        Some(provenance) => object(provenance, &format!("{context}.provenance"))?.clone(),
    };
    Ok(Point {
        name: name.to_owned(),
        server: server_name.to_owned(),
        task,
        model: model.to_owned(),
        dataset: dataset.to_owned(),
        dataset_revision: string("dataset_revision")?,
        dataset_path,
        tokenizer,
        endpoint,
        question,
        provenance,
        load,
        sampling,
        image,
        video,
        metrics,
        changes,
    })
}

/// A point's media directory belongs to its run, which requires it empty and
/// moves it aside on a rerun: two points may not share one.
fn own_media(benchmarks: &IndexMap<String, Point>) -> Result<()> {
    let mut owners: BTreeMap<&str, &str> = BTreeMap::new();
    for (name, point) in benchmarks {
        let Some(directory) = point
            .video
            .media_dir
            .as_deref()
            .filter(|directory| !directory.is_empty())
        else {
            continue;
        };
        if let Some(owner) = owners.insert(directory, name) {
            bail!(
                "benchmarks.{name}.video.media_dir is also the media directory of benchmarks.{owner}"
            );
        }
    }
    Ok(())
}

/// The protected metrics: dotted paths into the summary's `metrics`, each
/// `higher` or `lower`, the direction it improves in.
fn metrics(raw: Option<&Value>, context: &str) -> Result<Vec<Metric>> {
    let table = object(raw.unwrap_or(&Value::Null), &format!("{context}.metrics"))?;
    let mut metrics = Vec::new();
    for (path, direction) in table {
        let direction = match direction.as_str() {
            Some("higher") => Direction::Higher,
            Some("lower") => Direction::Lower,
            _ => bail!("{context}.metrics.{path} must be 'higher' or 'lower'"),
        };
        let parts: Vec<String> = path.split('.').map(str::to_owned).collect();
        ensure!(
            parts.iter().all(|part| !part.is_empty()),
            "{context} has an invalid metric path {path:?}"
        );
        metrics.push(Metric {
            path: parts,
            direction,
        });
    }
    ensure!(
        !metrics.is_empty(),
        "a benchmark point must protect at least one metric"
    );
    Ok(metrics)
}

/// A point's `[[changes]]`: unique names, each `after` naming an earlier
/// change; a change that calls the admin listener or waits for its events needs
/// the server's `admin_socket`; a `cancel` disconnects at most the streams of a
/// closed loop.
fn changes(
    raw: Option<&Value>,
    server: &Server,
    load: &Load,
    context: &str,
) -> Result<Vec<Change>> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    let items = raw
        .as_array()
        .with_context(|| format!("{context}.changes must be an array of tables"))?;
    ensure!(
        load.timeline(),
        "{context}.changes apply on a timeline load and require load.duration_s"
    );
    let mut changes: Vec<Change> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let where_ = format!("{context}.changes[{index}]");
        object(item, &where_)?;
        let change: Change = serde_json::from_value(item.clone())
            .map_err(|error| anyhow::anyhow!("{where_}: {error}"))?;
        change
            .check()
            .map_err(|error| anyhow::anyhow!("{where_}: {error}"))?;
        ensure!(
            !changes.iter().any(|earlier| earlier.name == change.name),
            "{where_} repeats the name {:?}",
            change.name
        );
        if let Some(after) = &change.after {
            ensure!(
                changes.iter().any(|earlier| &earlier.name == after),
                "{where_}.after must name an earlier change"
            );
        }
        ensure!(
            !change.calls_admin() || server.admin_socket.is_some(),
            "{where_} uses the admin listener, and servers.{} states no admin_socket",
            server.name
        );
        if let Some(cancel) = change.cancel {
            ensure!(
                load.request_rate.is_infinite() && cancel <= load.max_concurrency.unwrap_or(0),
                "{where_}.cancel disconnects at most the max_concurrency streams of a closed loop \
                 (request_rate = inf)"
            );
        }
        changes.push(change);
    }
    Ok(changes)
}

/// The sampling table, with the task's streaming default and the
/// `server_sampling` declaration.
fn sampling(raw: Option<&Value>, task: TaskName, context: &str) -> Result<Sampling> {
    let mut value = match raw {
        None => serde_json::Map::new(),
        Some(raw) => object(raw, context)?.clone(),
    };
    let server_sampling = match value.remove("server_sampling") {
        None => false,
        Some(Value::Bool(flag)) => flag,
        Some(_) => bail!("{context}.server_sampling must be a boolean"),
    };
    if server_sampling {
        let named: Vec<&str> = TOKEN_SAMPLING_CONTROLS
            .iter()
            .copied()
            .filter(|control| value.contains_key(*control))
            .collect();
        ensure!(
            named.is_empty(),
            "{context} declares server_sampling and sets token-sampling controls: {}",
            named.join(", ")
        );
    }
    if !value.contains_key("stream") {
        value.insert("stream".into(), Value::Bool(task.default_stream()));
    }
    let mut sampling: Sampling = serde_json::from_value(Value::Object(value))
        .map_err(|error| anyhow::anyhow!("{context}: {error}"))?;
    if server_sampling {
        sampling.temperature = None;
        sampling.top_p = None;
        sampling.ignore_eos = None;
    }
    Ok(sampling)
}

fn suite(name: &str, raw: &Value, benchmarks: &IndexMap<String, Point>) -> Result<Vec<String>> {
    let context = format!("suites.{name}");
    let value = object(raw, &context)?;
    reject_unknown(value, &["points"], &context)?;
    let points: Vec<String> = value
        .get("points")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .and_then(|items| {
            items
                .iter()
                .map(|item| item.as_str().map(str::to_owned))
                .collect()
        })
        .with_context(|| format!("{context}.points must be a non-empty string array"))?;
    let mut unique = points.clone();
    unique.sort();
    unique.dedup();
    ensure!(
        unique.len() == points.len(),
        "{context}.points must be unique"
    );
    let unknown: Vec<&str> = points
        .iter()
        .map(String::as_str)
        .filter(|point| !benchmarks.contains_key(*point))
        .collect();
    ensure!(
        unknown.is_empty(),
        "{context} names unknown points: {}",
        unknown.join(", ")
    );
    Ok(points)
}

/// The first executable file named `name` in a directory of `PATH`, as an
/// absolute path.
fn on_path(name: &Path) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt as _;

    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join(name))
        .find(|candidate| {
            std::fs::metadata(candidate)
                .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
        })
        .and_then(|found| std::path::absolute(found).ok())
}

/// A server's resolved launch: its command with its executable resolved, the
/// directory it runs in (the profile's root, so a relative path it was given
/// reads as in the file) and its environment.
#[derive(Debug, Clone, Serialize)]
pub struct Launch {
    pub command: Vec<String>,
    pub working_directory: PathBuf,
    pub environment: BTreeMap<String, String>,
}

/// The launch of one single-process server, its executable replaced by
/// `executable` when given. The executable resolves as a shell resolves a
/// command: a bare name through `PATH` ([`on_path`]), a relative path against
/// `root`; a bare name `PATH` does not hold stays as written, and starting it
/// fails naming it.
pub fn launch(server: &Server, executable: Option<&Path>, root: &Path) -> Launch {
    let mut command = server.command.clone();
    let selected = executable.map_or_else(|| PathBuf::from(&command[0]), Path::to_path_buf);
    let absolute = if selected.is_absolute() {
        selected
    } else if selected.components().count() == 1 {
        on_path(&selected).unwrap_or(selected)
    } else {
        root.join(selected)
    };
    command[0] = absolute.display().to_string();
    Launch {
        command,
        working_directory: root.to_path_buf(),
        environment: server.environment.clone(),
    }
}
