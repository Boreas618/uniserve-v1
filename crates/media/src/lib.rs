//! H.264/AAC output encoding and MP4 muxing, conditional media reading, and
//! the host resamplers a model conditions media on.
//!
//! The design places this crate under the executor: media work is blocking
//! host work, run on the main process's blocking thread pool beside module
//! loading and transfer reads (`specs/worker.md` §2.4, §4.5). The crate holds
//! no executor types: the executor copies device results into host buffers,
//! hands them here, and copies what the readers produce back to the device.
//!
//! A video request's output is assembled from independently encoded media
//! units: each unit is a self-contained MP4 of one H.264 stream starting at a
//! keyframe ([`codec::encode_video_unit`]), the soundtrack is one AAC MP4
//! ([`codec::encode_audio_track`]), and a [`mux::Container`] remuxes their
//! packets into the request's MP4 without decoding them. [`mux::MuxSession`]
//! holds one request's assembly: which units arrived, in which order, and the
//! decode time of the next one.
//!
//! Condition media is read exactly as its plan states: an image is decoded
//! with its EXIF orientation applied, resized with Pillow's LANCZOS filter and
//! cropped to its planned window ([`read::read_image`]); an audio track is
//! decoded to planar float at its native rate, trimmed, duplicated to stereo,
//! and brought to the model's rate with torchaudio's bandlimited sinc
//! interpolation ([`read::read_audio`]); a reference video is decoded by one
//! external `ffmpeg` process ([`read::read_video`]). Every reader checks that
//! it produced exactly the planned extent and fails with [`Error::Plan`]
//! otherwise, so media that decodes differently from what the server probed
//! fails the request instead of misaligning its rows.
//!
//! The three host resamplers reproduce the reference implementations'
//! mathematics: Pillow's LANCZOS resize ([`resample::lanczos_resize`]),
//! torchvision's antialiased bicubic resize ([`resample::bicubic_antialias_u8`]
//! and [`resample::bicubic_antialias_f32`]), and torchaudio's sinc resample
//! ([`resample::sinc_resample`]).
//!
//! Encoding, muxing and the libav* reading calls go through one C shim over
//! FFmpeg 8.1.2's headers (`src/ffi/shim.c`), built against a shared FFmpeg
//! runtime with libx264. The shim keeps FFmpeg's ABI out of the Rust side: the
//! crate declares only its own small C interface in [`ffi`]. The options are
//! the serving options of the implementation this replaces
//! (`uniserve/media/codec.py`): libx264 with `preset=ultrafast` and
//! `tune=zerolatency` at 8 threads, yuv420p through swscale's bilinear filter,
//! AAC with `aac_coder=fast` at 144 kb/s, and the MP4 muxer.
//!
//! [`MediaRunner`] is the side the main process calls: one value holding the
//! configured reference-video executable and the MP4 assemblies of the
//! requests it is assembling, with every operation safe to run on a blocking
//! pool thread.

pub mod codec;
pub mod config;
pub mod error;
mod ffi;
pub mod mux;
pub mod read;
pub mod resample;
pub mod runner;

pub use codec::{
    AUDIO_CODEC, AUDIO_FRAME_SAMPLES, ENCODER_THREADS, Rgb24Frames, VIDEO_CODEC, audio_samples,
    encode_audio_track, encode_video_unit, encoded_unit_bytes, encoded_video_bytes, frame_unit,
    read_unit, require_codecs,
};
pub use config::MediaConfig;
pub use error::{Error, Result};
pub use mux::{Container, MuxSession};
pub use read::{
    AudioPlan, ConditionProducts, ConditionRead, DecodedVideo, ImagePlan, VideoClip, read_audio,
    read_conditions, read_image, read_video,
};
pub use runner::MediaRunner;
