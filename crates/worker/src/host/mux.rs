//! MP4 assembly of a video request's independently encoded media units.
//!
//! A video request's frames are encoded unit by unit on the CPU workers of
//! its video codec component: each unit becomes a self-contained MP4 that
//! starts at a keyframe, written into its row of the round's `encoded_units`
//! product behind a native-endian `u64` length ([`frame_unit`]). The muxer
//! rank appends the rows to the request's [`MuxSession`] in unit order as
//! their rounds complete, encodes the soundtrack once, and finalizes the
//! container when every unit has arrived.
//!
//! The session owns the assembly's state: how many units arrived, the decode
//! time the next unit starts at, the encoded audio track and completion.
//! PyAV performs the container operations (`media.codec.Container`), on a container
//! the session holds as an opaque handle and never calls concurrently.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use tvm_ffi::{Any, AnyView};

use crate::ffi::{Function, Handle, Interpreter};
use crate::{Error, Result, lock};

/// Bytes of an encoded unit's length prefix.
pub(crate) const LENGTH_BYTES: usize = 8;

/// Threads one serving encoder session uses; the encoded-size bound charges
/// one slice header per thread per frame (`uniserve.media.codec`).
const ENCODER_THREADS: u64 = 8;

/// Bound the MP4 bytes of one unit from the serving encoder (libx264,
/// yuv420p, `preset=ultrafast`, `tune=zerolatency`).
///
/// This preset uses baseline CAVLC. A padded 16x16 macroblock has at most 27
/// residual blocks (including the separate DC blocks), each with up to 16
/// coefficients. Baseline level codes occupy at most 28 bits; x264
/// re-encodes CAVLC overflows at a higher QP. Per-block token, zero and run
/// codes need at most 16 + 9 + 15 * 11 bits. A further 1536 bits covers all
/// macroblock headers and 16 pairs of motion-vector differences. NAL
/// escaping adds at most one byte per two source bytes. Each encoder thread
/// writes one slice header per frame of at most 1024 bytes; MP4 sample
/// tables need at most 64 bytes per packet, and parameter sets, encoder SEI
/// and fixed container boxes fit in 64 KiB. These are syntax bounds,
/// independent of image entropy (x264 `encoder/cavlc.c`, `encoder/encoder.c`
/// and FFmpeg `libavformat/movenc.c`).
pub(crate) fn encoded_video_bytes(frames: u64, height: u64, width: u64) -> u64 {
    let blocks = height.div_ceil(16) * width.div_ceil(16);
    let residual_bits: u64 = 27 * (16 + 16 * 28 + 9 + 15 * 11);
    let macroblock_bytes = (residual_bits + 1536).div_ceil(8);
    let escaped_bytes = (3 * macroblock_bytes).div_ceil(2);
    (1 << 16) + frames * (blocks * escaped_bytes + ENCODER_THREADS * 1024 + 64)
}

/// Bytes of one `encoded_units` row: the length prefix and the largest
/// encoding of a unit of `frames` frames at the raster.
pub(crate) fn encoded_unit_bytes(frames: u64, height: u64, width: u64) -> u64 {
    LENGTH_BYTES as u64 + encoded_video_bytes(frames, height, width)
}

/// Write `payload` behind its native-endian `u64` length into `row`.
/// Returns the framed length; bytes past it stay untouched, so only that
/// prefix of the row is published.
pub(crate) fn frame_unit(payload: &[u8], row: &mut [u8]) -> Result<usize> {
    let capacity = row.len().saturating_sub(LENGTH_BYTES);
    if row.len() < LENGTH_BYTES || payload.len() > capacity {
        return Err(Error::Invalid(format!(
            "encoded media unit of {} bytes exceeds the {capacity} bytes reserved for it",
            payload.len()
        )));
    }
    let framed = LENGTH_BYTES + payload.len();
    row[..LENGTH_BYTES].copy_from_slice(&(payload.len() as u64).to_ne_bytes());
    row[LENGTH_BYTES..framed].copy_from_slice(payload);
    Ok(framed)
}

/// The payload of one framed unit row.
pub(crate) fn read_unit(row: &[u8]) -> Result<&[u8]> {
    let Some((header, body)) = row.split_first_chunk::<LENGTH_BYTES>() else {
        return Err(Error::Invalid(
            "encoded media unit has no length prefix".into(),
        ));
    };
    let length = usize::try_from(u64::from_ne_bytes(*header)).unwrap_or(usize::MAX);
    body.get(..length)
        .ok_or_else(|| Error::Invalid("encoded media unit names an invalid length".into()))
}

/// The container settings of one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MuxConfig {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frame_rate: u32,
    pub(crate) audio_rate: u32,
    /// Frames of each media unit, in order; they sum to the video's frames.
    pub(crate) unit_frames: Arc<[u32]>,
}

impl MuxConfig {
    pub(crate) fn new(
        width: u32,
        height: u32,
        frame_rate: u32,
        audio_rate: u32,
        unit_frames: Arc<[u32]>,
    ) -> Result<Self> {
        if width == 0 || height == 0 || frame_rate == 0 || audio_rate == 0 {
            return Err(Error::Invalid(
                "media mux dimensions and rates must be positive".into(),
            ));
        }
        if unit_frames.is_empty() || unit_frames.contains(&0) {
            return Err(Error::Invalid(
                "media mux units must cover the output frames".into(),
            ));
        }
        Ok(Self {
            width,
            height,
            frame_rate,
            audio_rate,
            unit_frames,
        })
    }

    /// Frames of the whole video.
    pub(crate) fn frame_count(&self) -> u64 {
        self.unit_frames
            .iter()
            .map(|&frames| u64::from(frames))
            .sum()
    }

    /// Samples of the soundtrack at the audio rate: the video's duration,
    /// `frames * audio_rate / frame_rate` rounded half to even.
    pub(crate) fn audio_samples(&self) -> u64 {
        let numerator = self.frame_count() * u64::from(self.audio_rate);
        let denominator = u64::from(self.frame_rate);
        let (quotient, remainder) = (numerator / denominator, numerator % denominator);
        let twice = 2 * remainder;
        if twice > denominator || (twice == denominator && quotient % 2 == 1) {
            quotient + 1
        } else {
            quotient
        }
    }
}

const CONTAINER: &str = "uniserve.media.codec.Container";

/// The library's container operations.
#[derive(Clone)]
pub(crate) struct MuxFunctions {
    open: Function,
    append: Function,
    finish: Function,
    close: Function,
}

impl MuxFunctions {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            open: Interpreter::import(&format!("{CONTAINER}.open"))?,
            append: Interpreter::import(&format!("{CONTAINER}.append"))?,
            finish: Interpreter::import(&format!("{CONTAINER}.finish"))?,
            close: Interpreter::import(&format!("{CONTAINER}.close"))?,
        })
    }

    /// Container operations from explicit functions, which tests supply in
    /// place of the codec library.
    #[cfg(test)]
    fn from_functions(open: Function, append: Function, finish: Function, close: Function) -> Self {
        Self {
            open,
            append,
            finish,
            close,
        }
    }
}

#[derive(Default)]
struct MuxState {
    container: Option<Handle>,
    /// A call into the container is in progress; the container is out of
    /// the state for its duration.
    busy: bool,
    /// The encoded soundtrack (`bytes`), held as an opaque value.
    audio: Option<Handle>,
    /// Decode time, in the video stream's time base, where the next unit
    /// starts.
    offset: i64,
    units: usize,
    closed: bool,
}

/// One request's MP4 under assembly.
///
/// The request and its host tasks share the session. Calls never overlap:
/// the engine hands a request's units, audio and finalization to one muxer
/// rank one call at a time. No lock is held across a container call.
pub(crate) struct MuxSession {
    config: MuxConfig,
    functions: MuxFunctions,
    state: Mutex<MuxState>,
}

impl MuxSession {
    pub(crate) fn new(config: MuxConfig, functions: MuxFunctions) -> Self {
        Self {
            config,
            functions,
            state: Mutex::new(MuxState::default()),
        }
    }

    pub(crate) fn config(&self) -> &MuxConfig {
        &self.config
    }

    fn state(&self) -> MutexGuard<'_, MuxState> {
        lock(&self.state)
    }

    /// Take the container out of the state for one call.
    fn begin(&self, incoming: usize) -> Result<(Option<Handle>, i64)> {
        let mut state = self.state();
        if state.closed {
            return Err(Error::Invalid("artifact assembly is closed".into()));
        }
        if state.busy {
            return Err(Error::Invariant(
                "artifact assembly received overlapping calls".into(),
            ));
        }
        if state.units + incoming > self.config.unit_frames.len() {
            return Err(Error::Invalid(
                "artifact assembly received more media units than the request".into(),
            ));
        }
        state.busy = true;
        Ok((state.container.take(), state.offset))
    }

    /// Append encoded units in request order without decoding their frames.
    /// Excess units are rejected before any packet is written; a failed
    /// append closes the session.
    pub(crate) fn append(&self, units: &[&[u8]]) -> Result<()> {
        let (mut container, mut offset) = self.begin(units.len())?;
        let mut appended = 0;
        let result = (|| {
            for unit in units {
                let payload = Any::from(tvm_ffi::Bytes::from(*unit));
                let opened = match container.take() {
                    Some(container) => container,
                    None => Handle::new(self.functions.open.call_values(&[
                        payload.clone(),
                        Any::from(i64::from(self.config.frame_rate)),
                        Any::from(i64::from(self.config.audio_rate)),
                    ])?)?,
                };
                let opened = container.insert(opened);
                offset = self.functions.append.call_as::<i64>(&[
                    opened.view(),
                    AnyView::from(&payload),
                    AnyView::from(&Any::from(offset)),
                ])?;
                appended += 1;
            }
            Ok(())
        })();
        let mut state = self.state();
        state.busy = false;
        state.container = container;
        state.offset = offset;
        state.units += appended;
        if result.is_err() {
            state.closed = true;
        }
        result
    }

    /// Keep the encoded audio track until finalization.
    pub(crate) fn set_audio(&self, audio: tvm_ffi::Bytes) -> Result<()> {
        let mut state = self.state();
        if state.closed {
            return Err(Error::Invalid("artifact assembly is closed".into()));
        }
        if state.audio.is_some() {
            return Err(Error::Invalid(
                "artifact assembly received its audio twice".into(),
            ));
        }
        state.audio = Some(Handle::new(Any::from(audio))?);
        Ok(())
    }

    /// Add the audio track and finalize a complete request: every unit and
    /// the audio must have arrived. Returns the MP4 bytes.
    pub(crate) fn finish(&self) -> Result<tvm_ffi::Bytes> {
        let (container, audio) = {
            let mut state = self.state();
            if state.closed || state.busy {
                return Err(Error::Invalid("artifact assembly is closed".into()));
            }
            if state.units != self.config.unit_frames.len() {
                return Err(Error::Invalid(
                    "artifact assembly requires every media unit of the request".into(),
                ));
            }
            let audio = state
                .audio
                .take()
                .ok_or_else(|| Error::Invalid("artifact assembly has no encoded audio".into()))?;
            let container = state.container.take().ok_or_else(|| {
                Error::Invariant("a complete artifact assembly has no container".into())
            })?;
            state.closed = true;
            (container, audio)
        };
        let bytes: tvm_ffi::Bytes = self
            .functions
            .finish
            .call_as(&[container.view(), audio.view()])?;
        if bytes.is_empty() {
            return Err(Error::Invalid(
                "media mux produced an empty container".into(),
            ));
        }
        Ok(bytes)
    }
}

impl Drop for MuxSession {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        if let Some(container) = state.container.take()
            && let Err(error) = self.functions.close.invoke(&[container.view()])
        {
            tracing::warn!("closing an unfinished container failed: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[test]
    fn units_are_framed_with_their_length_and_read_back() {
        let mut row = vec![0xaa_u8; 20];
        assert_eq!(frame_unit(b"payload", &mut row).unwrap(), 15);
        assert_eq!(read_unit(&row).unwrap(), b"payload");
        // Bytes past the framed prefix are untouched.
        assert_eq!(row[15..], [0xaa; 5]);

        let error = frame_unit(&[0; 13], &mut row).unwrap_err();
        assert_eq!(
            error.to_string(),
            "encoded media unit of 13 bytes exceeds the 12 bytes reserved for it"
        );
        assert!(read_unit(&row[..4]).is_err());
        let mut short = row.clone();
        short[..8].copy_from_slice(&100u64.to_ne_bytes());
        assert!(read_unit(&short).is_err());
    }

    #[test]
    fn the_encoded_bound_covers_every_macroblock_and_slice_header() {
        // 1344x768 is 84x48 macroblocks; one frame adds its macroblocks'
        // escaped residual bound, eight slice headers and a sample entry.
        let one = encoded_video_bytes(1, 768, 1344);
        assert_eq!(one - (1 << 16), 84 * 48 * 3519 + 8 * 1024 + 64);
        assert_eq!(
            encoded_video_bytes(22, 768, 1344) - (1 << 16),
            22 * (one - (1 << 16))
        );
        // Partial macroblocks are padded.
        assert_eq!(
            encoded_video_bytes(1, 17, 17),
            encoded_video_bytes(1, 32, 32)
        );
        assert_eq!(
            encoded_unit_bytes(1, 16, 16),
            8 + encoded_video_bytes(1, 16, 16)
        );
    }

    #[test]
    fn the_soundtrack_spans_the_video_duration_rounded_half_to_even() {
        let config =
            |frames: &[u32], rate| MuxConfig::new(64, 64, 24, rate, frames.into()).unwrap();
        // 124 frames at 24 fps and 32 kHz.
        assert_eq!(
            config(&[17, 17, 17, 17, 17, 17, 22], 32_000).audio_samples(),
            165_333
        );
        // 1 * 36 / 24 = 1.5 rounds to 2; 1 * 60 / 24 = 2.5 rounds to 2.
        assert_eq!(config(&[1], 36).audio_samples(), 2);
        assert_eq!(config(&[1], 60).audio_samples(), 2);
        assert!(MuxConfig::new(64, 64, 24, 32_000, Arc::from([])).is_err());
    }

    /// Container operations recording what the session hands them: a
    /// unit's "end" is its offset plus its length.
    fn recorder() -> (MuxFunctions, Arc<Mutex<Vec<String>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let function = |name: &'static str, log: Arc<Mutex<Vec<String>>>| {
            Function::new(
                name,
                tvm_ffi::Function::from_packed(move |args: &[AnyView]| {
                    let mut entries = log.lock().unwrap();
                    match name {
                        "open" => {
                            entries.push("open".into());
                            Ok(Any::from(7_i64))
                        }
                        "append" => {
                            let unit = tvm_ffi::Bytes::try_from(args[1])?;
                            let offset = i64::try_from(args[2])?;
                            entries.push(format!("append {} at {offset}", unit.len()));
                            Ok(Any::from(offset + unit.len() as i64))
                        }
                        "finish" => {
                            let audio = tvm_ffi::Bytes::try_from(args[1])?;
                            entries.push(format!("finish {}", audio.len()));
                            Ok(Any::from(tvm_ffi::Bytes::from(b"mp4 bytes")))
                        }
                        _ => {
                            entries.push("close".into());
                            Ok(Any::from(()))
                        }
                    }
                }),
            )
        };
        let functions = MuxFunctions::from_functions(
            function("open", Arc::clone(&log)),
            function("append", Arc::clone(&log)),
            function("finish", Arc::clone(&log)),
            function("close", Arc::clone(&log)),
        );
        (functions, log)
    }

    #[test]
    fn a_session_offsets_units_in_order_and_finishes_once_complete() {
        let (functions, log) = recorder();
        let config = MuxConfig::new(64, 64, 24, 32_000, Arc::from([17, 17, 22])).unwrap();
        let session = MuxSession::new(config, functions);

        session.append(&[b"aaaa", b"bb"]).unwrap();
        // Excess units are refused before any is written.
        assert!(session.append(&[b"c", b"d"]).is_err());
        assert!(session.finish().is_err());
        session.set_audio(tvm_ffi::Bytes::from(b"aac")).unwrap();
        assert!(session.finish().is_err());

        session.append(&[b"ccc"]).unwrap();
        assert!(session.set_audio(tvm_ffi::Bytes::from(b"x")).is_err());
        let bytes = session.finish().unwrap();
        assert_eq!(bytes.as_slice(), b"mp4 bytes");
        assert!(session.finish().is_err());
        drop(session);
        assert_eq!(
            *log.lock().unwrap(),
            [
                "open",
                "append 4 at 0",
                "append 2 at 4",
                "append 3 at 6",
                "finish 3"
            ]
        );
    }

    #[test]
    fn an_unfinished_session_closes_its_container() {
        let (functions, log) = recorder();
        let config = MuxConfig::new(64, 64, 24, 32_000, Arc::from([17, 22])).unwrap();
        let session = MuxSession::new(config, functions);
        session.append(&[b"a"]).unwrap();
        drop(session);
        assert_eq!(*log.lock().unwrap(), ["open", "append 1 at 0", "close"]);
    }
}
