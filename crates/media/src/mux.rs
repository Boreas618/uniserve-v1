//! MP4 assembly from independently encoded units and one audio track.
//!
//! A request's MP4 is assembled without decoding: the video stream is
//! templated from the first unit's stream, the audio stream from a one-video-
//! frame silence track, and every unit's packets are remuxed at the decode
//! time the caller assigns. The caller orders the units and owns their
//! offsets; these types hold the codec work and the assembly's state only.

use crate::codec;
use crate::ffi;
use crate::{Error, Result};

/// An MP4 being assembled from encoded units and one audio track.
///
/// The container's video stream is templated from the first unit and its
/// audio stream from a one-sample silent AAC encoding, because both streams
/// must exist before the MP4 header is written. [`Container::append`] shifts
/// one unit's packets by the offset it is given and returns where the unit
/// ends, the offset of the unit that follows.
pub struct Container {
    muxer: ffi::OpenMuxer,
}

impl Container {
    /// Opens an empty container whose streams match the request's codecs.
    ///
    /// The audio template encodes one video frame's duration of silence.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when `first_unit` has no video stream, the frame
    /// rate is not positive, or the audio rate is not positive;
    /// [`crate::Error::Codec`] when the muxer fails.
    pub fn open(first_unit: &[u8], frame_rate: u32, audio_rate: u32) -> Result<Self> {
        if frame_rate < 1 || audio_rate < 1 {
            return Err(Error::plan(
                "media mux dimensions and rates must be positive",
            ));
        }
        // The template's duration is one video frame, as Python's `round`
        // computes it: half to even.
        let samples = (audio_rate as f64 / frame_rate as f64).round_ties_even() as usize;
        if samples < 1 {
            return Err(Error::plan("audio track length and rate must be positive"));
        }
        let silence = vec![0i16; samples * 2];
        let audio_template = codec::encode_audio_track(&silence, samples, audio_rate)?;
        let muxer = ffi::OpenMuxer::open(first_unit, &audio_template, frame_rate, audio_rate)?;
        Ok(Self { muxer })
    }

    /// Muxes one unit's packets at decode-time `offset`.
    ///
    /// Returns the decode time after the unit's last packet: its largest
    /// `dts + duration` (a zero duration counts one tick).
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when `unit` has no video stream;
    /// [`crate::Error::Codec`] when demuxing or muxing fails.
    pub fn append(&mut self, unit: &[u8], offset: i64) -> Result<i64> {
        self.muxer.append(unit, offset)
    }

    /// Muxes the encoded audio track and closes the container.
    ///
    /// Returns the complete MP4.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when the track has no audio or the container
    /// came out empty; [`crate::Error::Codec`] when demuxing or muxing fails.
    pub fn finish(self, audio: &[u8]) -> Result<Vec<u8>> {
        let bytes = self.muxer.finish(audio)?;
        if bytes.is_empty() {
            return Err(Error::plan("media mux produced an empty container"));
        }
        Ok(bytes)
    }

    /// Discards an unfinished container.
    pub fn close(self) {
        // Dropping the shim's assembly frees it without writing a trailer.
    }
}

/// One request's MP4 under assembly.
///
/// The engine hands a request's units, audio and finalization to one muxer
/// rank one call at a time; the caller runs them in order. The session rejects
/// more units than the request has, audio set twice, finalization before every
/// unit and the audio arrived, and any call after it closed.
pub struct MuxSession {
    frame_rate: u32,
    audio_rate: u32,
    /// Frames of each unit in request order; the timeline the units cover.
    unit_frames: Vec<u32>,
    container: Option<Container>,
    audio: Option<Vec<u8>>,
    /// Decode time, in the video stream's time base, where the next unit
    /// starts.
    offset: i64,
    units: usize,
    closed: bool,
}

impl MuxSession {
    /// Opens the assembly of a request with `unit_frames` frames in its units.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when a dimension or rate is not positive, or the
    /// units are empty or cover no frames.
    pub fn open(
        width: u32,
        height: u32,
        frame_rate: u32,
        audio_rate: u32,
        unit_frames: &[u32],
    ) -> Result<Self> {
        if width < 1 || height < 1 || frame_rate < 1 || audio_rate < 1 {
            return Err(Error::plan(
                "media mux dimensions and rates must be positive",
            ));
        }
        if unit_frames.is_empty() || unit_frames.contains(&0) {
            return Err(Error::plan("media mux units must cover the output frames"));
        }
        Ok(Self {
            frame_rate,
            audio_rate,
            unit_frames: unit_frames.to_vec(),
            container: None,
            audio: None,
            offset: 0,
            units: 0,
            closed: false,
        })
    }

    /// Samples of the request's soundtrack at the audio rate.
    pub fn samples(&self) -> usize {
        let frames = self.unit_frames.iter().map(|frames| *frames as usize).sum();
        codec::audio_samples(frames, self.frame_rate, self.audio_rate)
    }

    /// Appends encoded units in request order without decoding them.
    ///
    /// Excess units are rejected before any packet is written. A refused
    /// append leaves the session usable, as the reference container does: the
    /// caller learns that the request cannot be assembled and discards it, and
    /// the units of a request that may still be completed stay accepted.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when more units arrive than the request has, or
    /// the session is closed; [`crate::Error::Codec`] when muxing fails.
    pub fn append(&mut self, units: &[Vec<u8>]) -> Result<()> {
        if self.closed {
            return Err(Error::plan("artifact assembly is closed"));
        }
        if self.units + units.len() > self.unit_frames.len() {
            return Err(Error::plan(
                "artifact assembly received more media units than the request",
            ));
        }
        self.append_units(units)
    }

    /// Appends the admitted units, opening the container from the first one.
    fn append_units(&mut self, units: &[Vec<u8>]) -> Result<()> {
        for unit in units {
            if self.container.is_none() {
                self.container = Some(Container::open(unit, self.frame_rate, self.audio_rate)?);
            }
            let container = self
                .container
                .as_mut()
                .ok_or_else(|| Error::plan("a media assembly lost its container"))?;
            self.offset = container.append(unit, self.offset)?;
            self.units += 1;
        }
        Ok(())
    }

    /// Keeps the encoded audio track until finalization.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when the session is closed or received its audio
    /// twice.
    pub fn set_audio(&mut self, audio: Vec<u8>) -> Result<()> {
        if self.closed {
            return Err(Error::plan("artifact assembly is closed"));
        }
        if self.audio.is_some() {
            return Err(Error::plan("artifact assembly received its audio twice"));
        }
        self.audio = Some(audio);
        Ok(())
    }

    /// Adds the audio track and finalizes a complete request's MP4 bytes.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when a unit or the audio is missing, or the
    /// session is closed; [`crate::Error::Codec`] when muxing fails.
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        if self.closed {
            return Err(Error::plan("artifact assembly is closed"));
        }
        if self.units != self.unit_frames.len() {
            return Err(Error::plan(
                "artifact assembly requires every media unit of the request",
            ));
        }
        let audio = self
            .audio
            .take()
            .ok_or_else(|| Error::plan("artifact assembly has no encoded audio"))?;
        let container = self
            .container
            .take()
            .ok_or_else(|| Error::plan("a complete artifact assembly has no container"))?;
        self.closed = true;
        container.finish(&audio)
    }

    /// Discards an unfinished container.
    pub fn close(&mut self) {
        self.closed = true;
        self.container = None;
        self.audio = None;
    }
}

#[cfg(test)]
mod tests {
    use super::{Container, MuxSession};
    use crate::Error;

    #[test]
    fn a_session_requires_positive_dimensions_and_units() {
        assert!(matches!(
            MuxSession::open(0, 8, 24, 24_000, &[1]),
            Err(Error::Plan(_))
        ));
        assert!(matches!(
            MuxSession::open(8, 0, 24, 24_000, &[1]),
            Err(Error::Plan(_))
        ));
        assert!(matches!(
            MuxSession::open(8, 8, 0, 24_000, &[1]),
            Err(Error::Plan(_))
        ));
        assert!(matches!(
            MuxSession::open(8, 8, 24, 0, &[1]),
            Err(Error::Plan(_))
        ));
        assert!(matches!(
            MuxSession::open(8, 8, 24, 24_000, &[]),
            Err(Error::Plan(_))
        ));
        assert!(matches!(
            MuxSession::open(8, 8, 24, 24_000, &[10, 0, 10]),
            Err(Error::Plan(_))
        ));
    }

    #[test]
    fn a_sessions_soundtrack_covers_every_unit() {
        let session = MuxSession::open(64, 48, 24, 32_000, &[17, 17, 5]).expect("a valid session");
        // 39 frames at 24 fps last 1.625 s, 52 000 samples at 32 kHz.
        assert_eq!(session.samples(), 52_000);
        let session = MuxSession::open(64, 48, 2, 1, &[1]).expect("a valid session");
        // 1 * 1 / 2 = 0.5, rounded half to even, is zero samples.
        assert_eq!(session.samples(), 0);
    }

    #[test]
    fn excess_units_and_second_audio_and_repeated_calls_are_refused() {
        let mut session = MuxSession::open(8, 8, 24, 24_000, &[1]).expect("a valid session");
        // The excess is refused before the container is opened, so no encoder
        // runs in this test.
        let units = vec![vec![0u8; 4], vec![0u8; 4]];
        assert!(matches!(session.append(&units), Err(Error::Plan(_))));
        // A refused call leaves the session usable: the request may still be
        // assembled from the units that remain.
        assert!(session.append(&[]).is_ok());
        assert!(matches!(session.finish(), Err(Error::Plan(_))));
        assert!(session.set_audio(vec![0u8; 4]).is_ok());
        assert!(matches!(
            session.set_audio(vec![0u8; 4]),
            Err(Error::Plan(_))
        ));

        session.close();
        assert!(matches!(session.append(&[]), Err(Error::Plan(_))));
        assert!(matches!(
            session.set_audio(vec![0u8; 4]),
            Err(Error::Plan(_))
        ));
        assert!(matches!(session.finish(), Err(Error::Plan(_))));
    }

    #[test]
    fn a_container_needs_positive_rates() {
        assert!(matches!(
            Container::open(&[], 0, 24_000),
            Err(Error::Plan(_))
        ));
        assert!(matches!(Container::open(&[], 24, 0), Err(Error::Plan(_))));
    }
}
