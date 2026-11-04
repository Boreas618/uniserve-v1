//! The runner side of media work: what the main process's executor calls.
//!
//! The executor runs every method of [`MediaRunner`] on its blocking thread
//! pool (`specs/worker.md` §2.4, §4.5): media work touches no stream and
//! schedules no job on a device. The executor owns the copies around it: it
//! stages device frames and decoded audio in host storage, hands the host
//! bytes here, and copies what the readers produce to the device products on
//! its own stream. The crate therefore knows nothing about executors, jobs,
//! leases or completion reports.
//!
//! One request's mux calls are serialized by the caller, as they are today:
//! the engine hands a request's units, audio and finalization to one muxer
//! rank one call at a time. Sessions of different requests proceed
//! concurrently.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::Result;
use crate::codec::{self, Rgb24Frames};
use crate::config::MediaConfig;
use crate::mux::MuxSession;
use crate::read::{ConditionProducts, ConditionRead};

/// The main process's media runner: encoders, muxers and readers.
///
/// It holds the request MP4 assemblies under assembly and the configured
/// reference-video executable. Every method blocks for its work's duration
/// and is safe to call from several blocking-pool threads at once; calls for
/// one request are the caller's to serialize.
pub struct MediaRunner {
    config: MediaConfig,
    sessions: Mutex<BTreeMap<u64, Arc<Mutex<MuxSession>>>>,
}

impl MediaRunner {
    /// A runner whose reference videos are decoded by `config.ffmpeg`.
    pub fn new(config: MediaConfig) -> Self {
        Self {
            config,
            sessions: Mutex::new(BTreeMap::new()),
        }
    }

    /// The configuration this runner was created with.
    pub fn config(&self) -> &MediaConfig {
        &self.config
    }

    /// Verifies that the linked FFmpeg runtime provides the serving encoders.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Codec`] naming the missing encoder.
    pub fn require_codecs(&self) -> Result<()> {
        codec::require_codecs()
    }

    /// Encodes one media unit into its framed row.
    ///
    /// The unit's MP4 is written behind its native-endian `u64` length into
    /// `row`; the framed length is returned and the rest of the row is left
    /// untouched, so only that prefix is published.
    ///
    /// # Errors
    ///
    /// The errors of [`codec::encode_video_unit`] and [`codec::frame_unit`].
    pub fn encode_unit(
        &self,
        frames: Rgb24Frames<'_>,
        frame_rate: u32,
        row: &mut [u8],
    ) -> Result<usize> {
        let unit = codec::encode_video_unit(frames, frame_rate)?;
        codec::frame_unit(&unit, row)
    }

    /// Encodes one request's `[samples, 2]` int16 soundtrack into an AAC MP4.
    ///
    /// # Errors
    ///
    /// The errors of [`codec::encode_audio_track`].
    pub fn encode_audio(&self, pcm: &[i16], samples: usize, rate: u32) -> Result<Vec<u8>> {
        codec::encode_audio_track(pcm, samples, rate)
    }

    /// Opens the MP4 assembly of `request`.
    ///
    /// `unit_frames` is the frame count of each unit in request order; it fixes
    /// the timeline the units cover and the soundtrack's sample count.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when the request already has assembly, the
    /// dimensions or rates are not positive, or the units cover no frames.
    pub fn open_mux(
        &self,
        request: u64,
        width: u32,
        height: u32,
        frame_rate: u32,
        audio_rate: u32,
        unit_frames: &[u32],
    ) -> Result<()> {
        let session = MuxSession::open(width, height, frame_rate, audio_rate, unit_frames)?;
        let mut sessions = self.lock_sessions();
        if sessions.contains_key(&request) {
            return Err(crate::Error::plan(format!(
                "request {request} already has a media assembly"
            )));
        }
        sessions.insert(request, Arc::new(Mutex::new(session)));
        Ok(())
    }

    /// Appends encoded units to `request`'s assembly, in request order.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when the request has no assembly or the units
    /// exceed it; the muxer's own errors otherwise.
    pub fn append_units(&self, request: u64, units: &[Vec<u8>]) -> Result<()> {
        let session = self.session(request)?;
        let mut session = session.lock().unwrap_or_else(|error| error.into_inner());
        session.append(units)
    }

    /// Keeps `request`'s encoded audio track until finalization.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when the request has no assembly or already has
    /// its audio.
    pub fn set_audio(&self, request: u64, audio: Vec<u8>) -> Result<()> {
        let session = self.session(request)?;
        let mut session = session.lock().unwrap_or_else(|error| error.into_inner());
        session.set_audio(audio)
    }

    /// Finalizes `request`'s MP4 and forgets its assembly.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when the request has no assembly, a unit or the
    /// audio is missing; the muxer's own errors otherwise.
    pub fn finish_mux(&self, request: u64) -> Result<Vec<u8>> {
        let mut sessions = self.lock_sessions();
        let session = sessions.remove(&request).ok_or_else(|| {
            crate::Error::plan(format!("request {request} has no media assembly"))
        })?;
        let mut session = session.lock().unwrap_or_else(|error| error.into_inner());
        session.finish()
    }

    /// Discards `request`'s unfinished assembly.
    pub fn close_mux(&self, request: u64) {
        let mut sessions = self.lock_sessions();
        if let Some(session) = sessions.remove(&request) {
            let mut session = session.lock().unwrap_or_else(|error| error.into_inner());
            session.close();
        }
    }

    /// Decodes a request's conditions into its products.
    ///
    /// # Errors
    ///
    /// The errors of [`crate::read::read_conditions`].
    pub fn read_conditions(
        &self,
        conditions: &[ConditionRead<'_>],
        products: &mut ConditionProducts<'_>,
    ) -> Result<()> {
        crate::read::read_conditions(Path::new(&self.config.ffmpeg), conditions, products)
    }

    /// The assembly of `request`, shared so that its own calls serialize while
    /// other requests proceed.
    fn session(&self, request: u64) -> Result<Arc<Mutex<MuxSession>>> {
        self.lock_sessions()
            .get(&request)
            .cloned()
            .ok_or_else(|| crate::Error::plan(format!("request {request} has no media assembly")))
    }

    /// The session map, surviving a panic of a previous holder: the map holds
    /// no invariant that a panic can break.
    fn lock_sessions(&self) -> std::sync::MutexGuard<'_, BTreeMap<u64, Arc<Mutex<MuxSession>>>> {
        self.sessions
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}
