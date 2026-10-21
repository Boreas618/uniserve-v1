//! Process-local video jobs and immutable retained artifacts.
//!
//! `VideoJobs` is the job store behind the asynchronous video routes in
//! `openai::videos`. It holds each job's public record, its generated
//! artifact once complete, and the cancellation token of its detached
//! generation task, and it tracks those tasks so server shutdown can wait for
//! them. Nothing persists across a restart, so server shutdown cancels every
//! job. Synchronous video requests do not use it.
//!
//! The operator's `video_job_config` states the two policies of the store,
//! and the video `capabilities` route advertises both:
//!
//! - `max_bytes` of artifact bytes, counting artifacts retained by
//!   a job and artifacts held by an unfinished download. Retained artifacts
//!   are shared-memory mappings, so this is the memory the store may hold.
//! - `retention`, after which a completed or failed job's record is deleted.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use proto::request::ArtifactMedia;
use serde::Serialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::options::VideoJobOptions;
use crate::output::RequestOutput;
use crate::output::video::VideoStreamError;

/// Public job record, serialized in the responses of the video job routes.
///
/// Timestamps are Unix seconds from `timestamp`. `status` is `queued` when
/// the video routes create the record, `in_progress` after
/// `VideoJobs::progress`, and `completed` or `failed` after
/// `VideoJobs::finish`.
#[derive(Clone, Serialize)]
pub(crate) struct VideoJob {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    /// Set with `completed_at`, to the store's retention after it.
    pub expires_at: Option<u64>,
    /// Requested duration in seconds: the target's, or the length of the
    /// reference soundtrack that sets it.
    pub seconds: f64,
    /// Duration in seconds of the frames the model generates, `num_frames`
    /// at the model's frame rate. It may exceed `seconds`, since the model
    /// extends a duration to a complete VAE clip.
    pub actual_seconds: f64,
    /// Frames the model generates for the requested duration.
    pub num_frames: u32,
    /// The generated canvas, `WxH` in pixels.
    pub size: String,
    pub status: &'static str,
    /// `queued`, then the phases the generation task reports from runtime
    /// events, then `completed` or `failed`.
    pub phase: String,
    /// Completed inference steps; set to `total_steps` on success.
    pub completed_steps: u32,
    pub total_steps: u32,
    /// Seconds from scheduling until the runtime reported completion; set
    /// on a successful finish.
    pub inference_time_s: Option<f64>,
    /// Seconds each generation phase lasted; set with `inference_time_s`.
    pub stage_durations: Option<BTreeMap<String, f64>>,
    pub error: Option<VideoFailure>,
}

/// Server-observed wall time of one generation and of each of its phases.
///
/// A phase starts at its first report and ends where the next phase starts;
/// the last one ends at completion. `encoding` starts when the engine
/// schedules the request. Every later phase is the one the engine reports
/// after a media call completes: `preparing` after the first encoding call,
/// `denoising` once latent preparation completes, `decoding` once the last
/// denoising step completes (covering video and audio decoding, their
/// H.264/AAC encoding and muxing), and `finalizing` (artifact delivery)
/// once the muxer completes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GenerationTiming {
    /// Seconds from scheduling to completion.
    pub inference_time_s: f64,
    /// Seconds per phase, keyed by phase name.
    pub stage_durations: BTreeMap<String, f64>,
}

/// Times one generation's phases from its runtime events.
///
/// Scheduling starts the `encoding` phase; each `MediaProgress` report whose
/// phase differs from the current one starts that phase. `finish` closes the
/// current phase and yields [`GenerationTiming`], or `None` when the request
/// was never scheduled.
#[derive(Default)]
pub(crate) struct StageClock {
    scheduled: Option<Instant>,
    phases: Vec<(String, Instant)>,
}

impl StageClock {
    /// Records `event` and returns the progress it reports: the phase and
    /// completed steps of a scheduling (`encoding`, no step) or media-progress
    /// event.
    pub(crate) fn observe<'a>(&mut self, event: &'a RequestOutput) -> Option<(&'a str, u32)> {
        let now = Instant::now();
        match event {
            RequestOutput::Scheduled { .. } => {
                if self.scheduled.is_none() {
                    self.scheduled = Some(now);
                    self.phases.push(("encoding".to_owned(), now));
                }
                Some(("encoding", 0))
            }
            RequestOutput::MediaProgress {
                phase,
                completed_steps,
            } => {
                if self.scheduled.is_some()
                    && self
                        .phases
                        .last()
                        .is_some_and(|(current, _)| current != phase)
                {
                    self.phases.push((phase.clone(), now));
                }
                Some((phase, *completed_steps))
            }
            _ => None,
        }
    }

    /// Closes the current phase and returns the generation's timing.
    pub(crate) fn finish(&self) -> Option<GenerationTiming> {
        let scheduled = self.scheduled?;
        let end = Instant::now();
        let mut stage_durations = BTreeMap::new();
        for (index, (phase, start)) in self.phases.iter().enumerate() {
            let stop = self.phases.get(index + 1).map_or(end, |(_, next)| *next);
            *stage_durations.entry(phase.clone()).or_insert(0.0) +=
                stop.duration_since(*start).as_secs_f64();
        }
        Some(GenerationTiming {
            inference_time_s: end.duration_since(scheduled).as_secs_f64(),
            stage_durations,
        })
    }
}

/// Failure reported on a `failed` job.
///
/// The code is the video stream error's (`VideoStreamError::code`), or
/// `artifact_capacity_exceeded` from `VideoJobs::finish`.
#[derive(Clone, Serialize)]
pub(crate) struct VideoFailure {
    pub code: &'static str,
    pub message: String,
}

impl From<VideoStreamError> for VideoFailure {
    fn from(error: VideoStreamError) -> Self {
        Self {
            code: error.code(),
            message: error.to_string(),
        }
    }
}

/// One entry of the job table.
struct RetainedJob {
    record: VideoJob,
    /// The generated artifact; set only on successful completion.
    media: Option<Arc<RetainedMedia>>,
    /// Shared with the job's generation task and its expiry timer; a child
    /// of the store's shutdown token.
    cancellation: CancellationToken,
}

/// A completed artifact holding its share of the retained-byte budget.
///
/// The download route moves an `Arc` of it into the response body, so the
/// bytes stay counted against the store's byte budget until the download ends, even
/// if the job is deleted or expires meanwhile.
pub(crate) struct RetainedMedia {
    media: Arc<dyn ArtifactMedia>,
    _bytes: OwnedSemaphorePermit,
}

impl std::ops::Deref for RetainedMedia {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.media.bytes()
    }
}

impl AsRef<[u8]> for RetainedMedia {
    fn as_ref(&self) -> &[u8] {
        self.media.bytes()
    }
}

/// Process-local job table, its byte budget, and its generation tasks.
pub(crate) struct VideoJobs {
    /// Jobs keyed by public job ID.
    entries: Mutex<BTreeMap<String, RetainedJob>>,
    /// One permit per retained artifact byte, up to the configured budget.
    bytes: Arc<Semaphore>,
    /// The configured retention and byte budget.
    settings: VideoJobOptions,
    /// Cancelled at server shutdown; every job's token is its child.
    shutdown: CancellationToken,
    /// The generation tasks and expiry timers of the jobs.
    tasks: TaskTracker,
}

impl VideoJobs {
    /// Creates an empty store under `settings` whose jobs stop when
    /// `shutdown` is cancelled.
    pub(crate) fn new(settings: VideoJobOptions, shutdown: CancellationToken) -> Self {
        Self {
            entries: Mutex::default(),
            bytes: Arc::new(Semaphore::new(settings.max_bytes)),
            settings,
            shutdown,
            tasks: TaskTracker::new(),
        }
    }

    /// The retention and byte budget the store was created with.
    pub(crate) fn settings(&self) -> &VideoJobOptions {
        &self.settings
    }

    /// Locks the job table and recovers it after poisoning.
    fn entries(&self) -> MutexGuard<'_, BTreeMap<String, RetainedJob>> {
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Adds a record and runs its generation task, `generation(token)`.
    ///
    /// The token is cancelled when the job is deleted or the server shuts
    /// down; the task must then end. Record IDs are server-generated UUIDs,
    /// so an insert never replaces another job.
    pub(crate) fn start<F>(&self, record: VideoJob, generation: impl FnOnce(CancellationToken) -> F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let cancellation = self.shutdown.child_token();
        self.entries().insert(
            record.id.clone(),
            RetainedJob {
                record,
                media: None,
                cancellation: cancellation.clone(),
            },
        );
        self.tasks.spawn(generation(cancellation));
    }

    /// Waits until every generation task and expiry timer has ended, which
    /// they do promptly once the shutdown token is cancelled.
    pub(crate) async fn wait(&self) {
        self.tasks.close();
        self.tasks.wait().await;
    }

    pub(crate) fn get(&self, id: &str) -> Option<VideoJob> {
        self.entries().get(id).map(|job| job.record.clone())
    }

    /// Returns a snapshot of every retained record, newest first, with ties on
    /// `created_at` ordered by descending ID.
    pub(crate) fn list(&self) -> Vec<VideoJob> {
        let entries = self.entries();
        let mut records: Vec<_> = entries.values().map(|job| job.record.clone()).collect();
        records.sort_by(|a, b| (b.created_at, &b.id).cmp(&(a.created_at, &a.id)));
        records
    }

    /// Marks a job `in_progress` at `phase` with `steps` completed steps.
    ///
    /// An unknown or deleted ID is ignored.
    pub(crate) fn progress(&self, id: &str, phase: &str, steps: u32) {
        if let Some(job) = self.entries().get_mut(id) {
            job.record.status = "in_progress";
            job.record.phase = phase.to_owned();
            job.record.completed_steps = steps;
        }
    }

    /// Publish completion atomically with ownership of the one generated artifact.
    ///
    /// A successful result is retained only if its bytes fit the remaining
    /// retained-byte budget; otherwise the job fails with
    /// `artifact_capacity_exceeded`. Either way the record gets `completed_at`
    /// and `expires_at`, and a timer task deletes it after the retention
    /// unless its token is cancelled first. If the job was already deleted,
    /// the result and any budget it took are dropped and no timer starts.
    ///
    /// Must run inside a Tokio runtime, because it spawns the timer task.
    pub(crate) fn finish(
        self: &Arc<Self>,
        id: &str,
        result: Result<Arc<dyn ArtifactMedia>, VideoFailure>,
        timing: Option<GenerationTiming>,
    ) {
        let mut entries = self.entries();

        // `try_acquire_many_owned` takes a `u32` count and does not wait, so an
        // artifact larger than `u32::MAX` bytes and an exhausted budget both
        // fail the job.
        let result = result.and_then(|media| {
            let bytes = u32::try_from(media.bytes().len())
                .ok()
                .and_then(|bytes| Arc::clone(&self.bytes).try_acquire_many_owned(bytes).ok());
            match bytes {
                Some(permit) => Ok(Arc::new(RetainedMedia {
                    media,
                    _bytes: permit,
                })),
                None => Err(VideoFailure {
                    code: "artifact_capacity_exceeded",
                    message: format!(
                        "retained videos and active downloads exceed the {}-byte limit",
                        self.settings.max_bytes
                    ),
                }),
            }
        });
        let Some(job) = entries.get_mut(id) else {
            return;
        };

        let now = crate::openai::unix_time();
        job.record.completed_at = Some(now);
        job.record.expires_at = Some(now + self.settings.retention.as_secs());
        match result {
            Ok(media) => {
                job.record.status = "completed";
                job.record.completed_steps = job.record.total_steps;
                job.record.phase = "completed".to_owned();
                if let Some(timing) = timing {
                    job.record.inference_time_s = Some(timing.inference_time_s);
                    job.record.stage_durations = Some(timing.stage_durations);
                }
                job.media = Some(media);
            }
            Err(error) => {
                job.record.status = "failed";
                job.record.phase = "failed".to_owned();
                job.record.error = Some(error);
            }
        }

        // The expiry timer holds only a clone of the token and a `Weak` handle,
        // so a pending timer does not keep `VideoJobs` alive. `delete` and
        // server shutdown cancel the token, which ends it.
        let expiry_cancellation = job.cancellation.clone();
        let weak = Arc::downgrade(self);
        let id = id.to_owned();
        let retention = self.settings.retention;
        self.tasks.spawn(async move {
            tokio::select! {
                _ = tokio::time::sleep(retention) => {}
                _ = expiry_cancellation.cancelled() => return,
            }
            if let Some(jobs) = weak.upgrade() {
                jobs.delete(&id);
            }
        });
    }

    /// Returns the artifact of a successfully completed job.
    pub(crate) fn content(&self, id: &str) -> Option<Arc<RetainedMedia>> {
        self.entries().get(id).and_then(|job| job.media.clone())
    }

    /// Removes the record and cancels its token, which ends a running
    /// generation task (dropping its stream cancels its engine request) and
    /// a pending expiry timer. Removal wins over a racing completion. Returns
    /// `false` for an unknown ID.
    pub(crate) fn delete(&self, id: &str) -> bool {
        if let Some(job) = self.entries().remove(id) {
            job.cancellation.cancel();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn record(id: &str) -> VideoJob {
        VideoJob {
            id: id.to_owned(),
            object: "video",
            model: "FastH3".to_owned(),
            created_at: crate::openai::unix_time(),
            completed_at: None,
            expires_at: None,
            seconds: 5.0,
            actual_seconds: 124.0 / 24.0,
            num_frames: 124,
            size: "1344x768".to_owned(),
            status: "queued",
            phase: "queued".to_owned(),
            completed_steps: 0,
            total_steps: 4,
            inference_time_s: None,
            stage_durations: None,
            error: None,
        }
    }

    fn store() -> Arc<VideoJobs> {
        Arc::new(VideoJobs::new(
            VideoJobOptions::default(),
            CancellationToken::new(),
        ))
    }

    /// Starts a job whose task waits for its token and reports it.
    fn start(jobs: &VideoJobs, id: &str) -> tokio::sync::oneshot::Receiver<()> {
        let (cancelled, receiver) = tokio::sync::oneshot::channel();
        jobs.start(record(id), |token| async move {
            token.cancelled().await;
            let _ = cancelled.send(());
        });
        receiver
    }

    #[tokio::test]
    async fn deleted_work_is_cancelled_and_cannot_reappear() {
        let jobs = store();
        let cancelled = start(&jobs, "video-a");
        jobs.progress("video-a", "denoising", 2);
        let running = jobs.get("video-a").unwrap();
        assert_eq!(
            (running.status, running.completed_steps),
            ("in_progress", 2)
        );

        assert!(jobs.delete("video-a"));
        cancelled.await.unwrap();

        // A completion racing the deletion must not recreate the record.
        jobs.finish(
            "video-a",
            Err(VideoFailure {
                code: "cancelled",
                message: "cancelled".into(),
            }),
            None,
        );
        assert!(jobs.get("video-a").is_none());
        assert!(jobs.list().is_empty());
    }

    #[tokio::test]
    async fn a_failed_job_expires_after_the_configured_retention() {
        let jobs = Arc::new(VideoJobs::new(
            VideoJobOptions {
                retention: Duration::from_secs(90),
                ..VideoJobOptions::default()
            },
            CancellationToken::new(),
        ));
        let _cancelled = start(&jobs, "video-0");

        jobs.finish(
            "video-0",
            Err(VideoFailure {
                code: "generation_failed",
                message: "decoder failed".into(),
            }),
            None,
        );
        let failed = jobs.get("video-0").unwrap();
        assert_eq!(failed.status, "failed");
        assert_eq!(failed.error.unwrap().code, "generation_failed");
        assert_eq!(
            failed.expires_at.unwrap() - failed.completed_at.unwrap(),
            90
        );

        assert!(jobs.delete("video-0"));
        assert!(jobs.list().is_empty());
    }

    /// Server shutdown stops every job's task, and waiting returns once they
    /// have ended.
    #[tokio::test]
    async fn shutdown_stops_every_job() {
        let shutdown = CancellationToken::new();
        let jobs = VideoJobs::new(VideoJobOptions::default(), shutdown.clone());
        let first = start(&jobs, "video-a");
        let second = start(&jobs, "video-b");

        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(10), jobs.wait())
            .await
            .unwrap();

        first.await.unwrap();
        second.await.unwrap();
    }
}
