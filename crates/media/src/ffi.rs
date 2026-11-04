//! The safe Rust side of the FFmpeg 8.1.2 shim.
//!
//! Every function here copies what the shim produced into owned Rust values
//! and frees the shim's allocation, so no libav* pointer survives a call. The
//! shim reports failures with a negative code and a message in a thread-local
//! buffer, which becomes [`Error::Codec`].

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;

use crate::{Error, Result};

/// One MP4 assembly owned by the shim.
#[repr(C)]
pub(crate) struct Muxer {
    _private: [u8; 0],
}

/// One audio track decoded to planar float at its stream's declared rate and
/// layout.
pub(crate) struct DecodedAudio {
    /// The stream's declared rate, which is also the converter's output rate.
    /// An HE-AAC track's frames decode at twice this rate and are converted
    /// back down, as the reference resampler does.
    pub(crate) rate: u32,
    /// Channel count; the planes are contiguous, one after another.
    pub(crate) channels: usize,
    /// Samples of each channel.
    pub(crate) samples: usize,
    /// `channels * samples` float samples, planar.
    pub(crate) planar: Vec<f32>,
}

unsafe extern "C" {
    fn usv_error() -> *const c_char;
    fn usv_free(pointer: *mut c_void);
    fn usv_require_codecs() -> c_int;
    fn usv_encode_video_unit(
        rgb24: *const u8,
        frames: c_int,
        width: c_int,
        height: c_int,
        frame_rate: c_int,
        data: *mut *mut u8,
        size: *mut usize,
    ) -> c_int;
    fn usv_encode_audio_track(
        pcm: *const i16,
        samples: c_int,
        rate: c_int,
        data: *mut *mut u8,
        size: *mut usize,
    ) -> c_int;
    fn usv_mux_open(
        first_unit: *const u8,
        first_size: usize,
        audio_template: *const u8,
        audio_size: usize,
        muxer: *mut *mut Muxer,
    ) -> c_int;
    fn usv_mux_append(
        muxer: *mut Muxer,
        unit: *const u8,
        size: usize,
        offset: i64,
        next: *mut i64,
    ) -> c_int;
    fn usv_mux_finish(
        muxer: *mut Muxer,
        audio: *const u8,
        size: usize,
        data: *mut *mut u8,
        out_size: *mut usize,
    ) -> c_int;
    fn usv_mux_close(muxer: *mut Muxer);
    fn usv_decode_audio(
        data: *const u8,
        size: usize,
        rate: *mut c_int,
        channels: *mut c_int,
        planar: *mut *mut f32,
        samples: *mut i64,
    ) -> c_int;
}

/// The shim's message for the failure it just reported on this thread.
fn last_error() -> String {
    // SAFETY: the shim returns a pointer to a thread-local buffer that is
    // valid until the next shim call on this thread, and we copy it here.
    let message = unsafe { CStr::from_ptr(usv_error()) };
    message.to_string_lossy().into_owned()
}

/// Turns a shim return code into a result, reading the message it left.
///
/// The shim reports a failure of the media itself (a unit without a video
/// stream, a track without audio) with its own code, which is a plan
/// violation here, as those failures were `ValueError` at the Python
/// boundary this crate replaces.
fn check(code: c_int) -> Result<()> {
    const PLAN: c_int = -2;
    if code == 0 {
        Ok(())
    } else if code == PLAN {
        Err(Error::Plan(last_error()))
    } else {
        Err(Error::Codec(last_error()))
    }
}

/// Copies the shim's allocation at `data`/`size` into a vector and frees it.
///
/// # Safety
///
/// `data` must be null or an allocation of `size` bytes from the shim, which
/// hands ownership of its buffers to the caller.
unsafe fn take_bytes(data: *mut u8, size: usize) -> Vec<u8> {
    if data.is_null() {
        return Vec::new();
    }
    // SAFETY: `data` is a shim allocation of `size` readable bytes; the shim
    // hands it to the caller, who frees it with `usv_free` after copying.
    let bytes = unsafe { std::slice::from_raw_parts(data, size) }.to_vec();
    // SAFETY: the allocation came from the shim and is not used again.
    unsafe { usv_free(data as *mut c_void) };
    bytes
}

/// Verifies that the FFmpeg runtime provides the serving encoders.
pub(crate) fn require_codecs() -> Result<()> {
    // SAFETY: the shim has no preconditions and reports its own failures.
    check(unsafe { usv_require_codecs() })
}

/// Encodes `frames` RGB24 frames into a self-contained MP4.
pub(crate) fn encode_video_unit(
    rgb24: &[u8],
    frames: u32,
    height: u32,
    width: u32,
    frame_rate: u32,
) -> Result<Vec<u8>> {
    let frame_bytes = rgb24.len() / frames.max(1) as usize;
    debug_assert_eq!(frame_bytes, height as usize * width as usize * 3);
    let mut data: *mut u8 = ptr::null_mut();
    let mut size: usize = 0;
    // SAFETY: `rgb24` holds `frames` frames of `width * height * 3` bytes,
    // checked by the caller; the shim writes its own allocation into
    // `data`/`size` on success.
    let code = unsafe {
        usv_encode_video_unit(
            rgb24.as_ptr(),
            frames as c_int,
            width as c_int,
            height as c_int,
            frame_rate as c_int,
            &mut data,
            &mut size,
        )
    };
    check(code)?;
    // SAFETY: on success the shim set `data` to an allocation of `size` bytes.
    Ok(unsafe { take_bytes(data, size) })
}

/// Encodes interleaved stereo int16 samples into a self-contained AAC MP4.
pub(crate) fn encode_audio_track(pcm: &[i16], samples: u32, rate: u32) -> Result<Vec<u8>> {
    let mut data: *mut u8 = ptr::null_mut();
    let mut size: usize = 0;
    // SAFETY: `pcm` holds `samples` interleaved stereo pairs; the shim writes
    // its own allocation into `data`/`size` on success.
    let code = unsafe {
        usv_encode_audio_track(
            pcm.as_ptr(),
            samples as c_int,
            rate as c_int,
            &mut data,
            &mut size,
        )
    };
    check(code)?;
    // SAFETY: on success the shim set `data` to an allocation of `size` bytes.
    Ok(unsafe { take_bytes(data, size) })
}

/// An MP4 under assembly, owned by the shim.
pub(crate) struct OpenMuxer {
    pointer: *mut Muxer,
}

// SAFETY: the handle owns an AVFormatContext and the streams inside it. FFmpeg contexts have no thread affinity, and the session holding the handle is guarded by a Mutex whose every use takes the lock, so moving the handle between threads transfers exclusive ownership.
unsafe impl Send for OpenMuxer {}

impl OpenMuxer {
    /// Opens an assembly whose streams match `first_unit`'s video stream and
    /// `audio_template`'s audio stream.
    pub(crate) fn open(
        first_unit: &[u8],
        audio_template: &[u8],
        _frame_rate: u32,
        _audio_rate: u32,
    ) -> Result<Self> {
        let mut pointer: *mut Muxer = ptr::null_mut();
        // SAFETY: both slices outlive the call; the shim copies the stream
        // parameters it needs and writes its own handle into `pointer`.
        let code = unsafe {
            usv_mux_open(
                first_unit.as_ptr(),
                first_unit.len(),
                audio_template.as_ptr(),
                audio_template.len(),
                &mut pointer,
            )
        };
        check(code)?;
        Ok(Self { pointer })
    }

    /// Muxes one unit's video packets at decode time `offset`, returning the
    /// decode time where the next unit starts.
    pub(crate) fn append(&mut self, unit: &[u8], offset: i64) -> Result<i64> {
        let mut next = offset;
        // SAFETY: `self.pointer` is an open assembly and `unit` outlives the
        // call; the shim writes the next decode time into `next`.
        let code =
            unsafe { usv_mux_append(self.pointer, unit.as_ptr(), unit.len(), offset, &mut next) };
        check(code)?;
        Ok(next)
    }

    /// Muxes the encoded audio track and finalizes the container.
    pub(crate) fn finish(mut self, audio: &[u8]) -> Result<Vec<u8>> {
        let mut data: *mut u8 = ptr::null_mut();
        let mut size: usize = 0;
        // SAFETY: `self.pointer` is an open assembly and `audio` outlives the
        // call; the shim writes its own allocation into `data`/`size`.
        let code = unsafe {
            usv_mux_finish(
                self.pointer,
                audio.as_ptr(),
                audio.len(),
                &mut data,
                &mut size,
            )
        };
        check(code)?;
        // SAFETY: on success the shim set `data` to an allocation of `size`
        // bytes and the assembly is emptied.
        let bytes = unsafe { take_bytes(data, size) };
        self.release();
        Ok(bytes)
    }

    /// Releases the shim's assembly without writing a trailer.
    fn release(&mut self) {
        if !self.pointer.is_null() {
            // SAFETY: `self.pointer` is a live assembly, closed exactly once
            // because `release` clears it.
            unsafe { usv_mux_close(self.pointer) };
            self.pointer = ptr::null_mut();
        }
    }
}

impl Drop for OpenMuxer {
    fn drop(&mut self) {
        self.release();
    }
}

/// Decodes the first audio stream of `data` to planar float.
pub(crate) fn decode_audio(data: &[u8]) -> Result<DecodedAudio> {
    let mut rate: c_int = 0;
    let mut channels: c_int = 0;
    let mut samples: i64 = 0;
    let mut planar: *mut f32 = ptr::null_mut();
    // SAFETY: `data` outlives the call; the shim writes its own allocation
    // into `planar` with `channels * samples` floats.
    let code = unsafe {
        usv_decode_audio(
            data.as_ptr(),
            data.len(),
            &mut rate,
            &mut channels,
            &mut planar,
            &mut samples,
        )
    };
    check(code)?;
    if rate <= 0 || channels <= 0 || samples < 0 {
        if !planar.is_null() {
            // SAFETY: the shim allocated `planar` and hands it to the caller.
            unsafe { usv_free(planar as *mut c_void) };
        }
        return Err(Error::Codec(
            "the audio decoder reported an invalid track extent".to_owned(),
        ));
    }
    let count = channels as usize * samples as usize;
    let mut values = Vec::with_capacity(count);
    if !planar.is_null() {
        // SAFETY: the shim allocated `count` floats at `planar` and hands
        // them to the caller.
        values.extend_from_slice(unsafe { std::slice::from_raw_parts(planar, count) });
        // SAFETY: the allocation is copied and not used again.
        unsafe { usv_free(planar as *mut c_void) };
    }
    Ok(DecodedAudio {
        rate: rate as u32,
        channels: channels as usize,
        samples: samples as usize,
        planar: values,
    })
}
