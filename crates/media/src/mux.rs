//! MP4 assembly from independently encoded units and one audio track.
//!
//! A request's MP4 is assembled without decoding: the video stream is
//! templated from the first unit's stream, the audio stream from a one-video-
//! frame silence track, and every unit's packets are remuxed at the decode
//! time the caller assigns. The caller orders the units and owns their
//! offsets; these types hold the codec work and the assembly's state only.

use crate::Result;

/// An MP4 being assembled from encoded units and one audio track.
///
/// The container's video stream is templated from the first unit and its
/// audio stream from a one-sample silent AAC encoding, because both streams
/// must exist before the MP4 header is written. [`Container::append`] shifts
/// one unit's packets by the offset it is given and returns where the unit
/// ends, the offset of the unit that follows.
pub struct Container {
    _private: (),
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
        todo!("builder01: mux module")
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
        todo!("builder01: mux module")
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
        todo!("builder01: mux module")
    }

    /// Discards an unfinished container.
    pub fn close(self) {
        todo!("builder01: mux module")
    }
}

/// One request's MP4 under assembly.
///
/// The engine hands a request's units, audio and finalization to one muxer
/// rank one call at a time; the caller runs them in order. The session rejects
/// more units than the request has, audio set twice, finalization before every
/// unit and the audio arrived, and any call after it closed.
pub struct MuxSession {
    _private: (),
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
        todo!("builder01: mux module")
    }

    /// Samples of the request's soundtrack at the audio rate.
    pub fn samples(&self) -> usize {
        todo!("builder01: mux module")
    }

    /// Appends encoded units in request order without decoding them.
    ///
    /// Excess units are rejected before any packet is written; a failed
    /// append closes the session.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when more units arrive than the request has, or
    /// the session is closed; [`crate::Error::Codec`] when muxing fails.
    pub fn append(&mut self, units: &[Vec<u8>]) -> Result<()> {
        todo!("builder01: mux module")
    }

    /// Keeps the encoded audio track until finalization.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when the session is closed or received its audio
    /// twice.
    pub fn set_audio(&mut self, audio: Vec<u8>) -> Result<()> {
        todo!("builder01: mux module")
    }

    /// Adds the audio track and finalizes a complete request's MP4 bytes.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Plan`] when a unit or the audio is missing, or the
    /// session is closed; [`crate::Error::Codec`] when muxing fails.
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        todo!("builder01: mux module")
    }

    /// Discards an unfinished container.
    pub fn close(&mut self) {
        todo!("builder01: mux module")
    }
}
