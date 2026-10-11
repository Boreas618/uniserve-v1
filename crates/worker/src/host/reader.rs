//! Reading a video request's condition media on a host rank.
//!
//! The server publishes every condition's fetched bytes to POSIX shared
//! memory on the head host and plans how each is prepared
//! (`uniserve_core::VideoCondition`). One media reading call per conditioned
//! request decodes them, in request order, into the request's condition
//! products:
//!
//! - `condition_pixels`: each visual condition's RGB24 frames the video
//!   encoder encodes, `[pixels, 3]` uint8, frame-major;
//! - `condition_samples`: each audio track's model-rate stereo PCM,
//!   `[samples, 2]` FP32;
//! - `vision_pixels`: the conditioner's packed patch rows of every condition
//!   it reads, from the decoded frames (an image's one frame, a video's
//!   sampled frames).
//!
//! Reference videos are decoded by one FFmpeg process per condition whose
//! standard input is the shared-memory object; images, audio and patch
//! packing are the model library's (`media.reader.read_image`,
//! `media.reader.read_audio`, `EncoderRunner.pack_pixels`). Every product
//! must be filled exactly.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::Arc;

use tvm_ffi::{Any, AnyView};
use uniserve_core::{
    AudioClip, ConditionMedia, MediaLocator, SharedMemory, VideoClip, VideoCondition,
};

use crate::ffi::{Backing, DType, Function, Handle, integer_list};
use crate::model::interface::Interface;
use crate::storage::memory::{Memory, StorageView};
use crate::{Error, Result};

/// The model functions condition reading calls.
#[derive(Clone)]
pub(crate) struct ReaderFunctions {
    read_image: Function,
    read_audio: Function,
    pack_vision: Function,
    /// The loaded model, whose vision encoder packs patches.
    loaded: Handle,
    /// The audio encoder's sample rate, which tracks are resampled to.
    pub(crate) sample_rate: u32,
}

impl ReaderFunctions {
    /// Resolve the reader functions of the model `loaded` describes: the
    /// library's condition decoders, and the patch packing and audio rate
    /// the Python `EncoderRunner` answers from the model's description.
    pub(crate) fn new(interface: &Interface, loaded: &Handle) -> Result<Self> {
        const ENCODER: &str = "uniserve_models.runner.encoder.EncoderRunner";
        let sample_rate: i64 = interface
            .import(&format!("{ENCODER}.sample_rate"))?
            .call_as(&[loaded.view()])?;
        Ok(Self {
            read_image: interface.import("uniserve_models.media.reader.read_image")?,
            read_audio: interface.import("uniserve_models.media.reader.read_audio")?,
            pack_vision: interface.import(&format!("{ENCODER}.pack_pixels"))?,
            loaded: loaded.clone(),
            sample_rate: u32::try_from(sample_rate)
                .map_err(|_| Error::Interface("the audio encoder's rate is invalid".into()))?,
        })
    }
}

/// The host views of one request's condition products; `None` where no
/// condition contributes to a product.
pub(crate) struct ConditionProducts {
    pub(crate) pixels: Option<StorageView>,
    pub(crate) samples: Option<StorageView>,
    pub(crate) patches: Option<StorageView>,
}

/// Rows of each product filled so far.
#[derive(Default)]
struct Cursor {
    pixels: i64,
    samples: i64,
    patches: i64,
}

/// The next `rows` rows of a product, advancing its cursor.
fn take(
    product: Option<&StorageView>,
    cursor: &mut i64,
    rows: u64,
    name: &str,
) -> Result<StorageView> {
    let product =
        product.ok_or_else(|| Error::Invalid(format!("a condition's {name} have no product")))?;
    let rows = i64::try_from(rows).map_err(|_| Error::Invalid(format!("{name} overflow")))?;
    if *cursor + rows > product.shape()[0] {
        return Err(Error::Invalid(format!(
            "the request's conditions exceed their {name}"
        )));
    }
    let view = product.narrow(0, *cursor, rows)?;
    *cursor += rows;
    Ok(view)
}

/// A host tensor over `bytes`.
fn host_bytes(bytes: Vec<u8>) -> Result<Any> {
    let length = bytes.len() as i64;
    let memory = Memory::host(bytes.len().max(1));
    if !bytes.is_empty() {
        // SAFETY: the allocation was just created with at least this length
        // and has no other reference.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), memory.address() as *mut u8, bytes.len());
        }
    }
    let view = StorageView::contiguous(memory as Arc<dyn Backing>, 0, DType::U8, &[length])?;
    Ok(Any::from(view.to_tensor()?))
}

/// The published bytes of a condition.
fn read_bytes(source: &MediaLocator) -> Result<Vec<u8>> {
    let file = SharedMemory::open(&source.name, false).map_err(|error| {
        Error::Invalid(format!(
            "condition media {} cannot be opened: {error}",
            source.name
        ))
    })?;
    let length = usize::try_from(source.bytes)
        .map_err(|_| Error::Invalid("condition media is too large".into()))?;
    let mut data = Vec::with_capacity(length);
    file.take(source.bytes)
        .read_to_end(&mut data)
        .map_err(|error| Error::Invalid(format!("condition media cannot be read: {error}")))?;
    if data.len() != length {
        return Err(Error::Invalid(
            "condition media is shorter than its locator".into(),
        ));
    }
    Ok(data)
}

/// Decodes one request's conditions into its products.
pub(crate) struct ConditionReader {
    functions: ReaderFunctions,
    /// The FFmpeg executable reference videos decode with.
    ffmpeg: String,
    /// The frame rate the model conditions on, which videos resample to.
    frame_rate: u32,
}

impl ConditionReader {
    pub(crate) fn new(functions: ReaderFunctions, ffmpeg: String, frame_rate: u32) -> Self {
        Self {
            functions,
            ffmpeg,
            frame_rate,
        }
    }

    /// Decode every condition in request order, filling each product
    /// exactly from its first row.
    pub(crate) fn read(
        &self,
        conditions: &[VideoCondition],
        products: &ConditionProducts,
    ) -> Result<()> {
        let mut cursor = Cursor::default();
        for condition in conditions {
            match &condition.media {
                ConditionMedia::Image(fit) => {
                    let (height, width) = (fit.size.height, fit.size.width);
                    let pixels = take(
                        products.pixels.as_ref(),
                        &mut cursor.pixels,
                        u64::from(height) * u64::from(width),
                        "pixels",
                    )?
                    .reshape(DType::U8, &[1, i64::from(height), i64::from(width), 3])?;
                    let data = host_bytes(read_bytes(&condition.source)?)?;
                    let resized =
                        integer_list([fit.resized.width, fit.resized.height].map(i64::from));
                    let window = integer_list([fit.left, fit.top, width, height].map(i64::from));
                    let target = Any::from(pixels.to_tensor()?);
                    self.functions.read_image.invoke(&[
                        AnyView::from(&data),
                        AnyView::from(&Any::from(resized)),
                        AnyView::from(&Any::from(window)),
                        AnyView::from(&target),
                    ])?;
                    self.pack(condition, &target, products, &mut cursor)?;
                }
                ConditionMedia::Video { clip, soundtrack } => {
                    let frames = self.read_video(&condition.source, clip)?;
                    let area = u64::from(clip.canvas.height) * u64::from(clip.canvas.width);
                    let pixels = take(
                        products.pixels.as_ref(),
                        &mut cursor.pixels,
                        u64::from(clip.vae_frames) * area,
                        "pixels",
                    )?;
                    // The video encoder encodes the leading kept frames.
                    let encoded = frames.narrow(0, 0, i64::from(clip.vae_frames))?;
                    copy_host(&encoded, &pixels)?;
                    self.pack(
                        condition,
                        &Any::from(frames.to_tensor()?),
                        products,
                        &mut cursor,
                    )?;
                    if let Some(track) = soundtrack {
                        self.read_track(&condition.source, track, products, &mut cursor)?;
                    }
                }
                ConditionMedia::Audio(track) => {
                    self.read_track(&condition.source, track, products, &mut cursor)?;
                }
            }
        }
        for (name, product, filled) in [
            ("pixels", &products.pixels, cursor.pixels),
            ("samples", &products.samples, cursor.samples),
            ("patches", &products.patches, cursor.patches),
        ] {
            if product
                .as_ref()
                .is_some_and(|view| view.shape()[0] != filled)
            {
                return Err(Error::Invalid(format!(
                    "the request's conditions leave {name} unfilled"
                )));
            }
        }
        Ok(())
    }

    /// Pack the conditioner's patch rows of a condition it reads.
    fn pack(
        &self,
        condition: &VideoCondition,
        frames: &Any,
        products: &ConditionProducts,
        cursor: &mut Cursor,
    ) -> Result<()> {
        let Some(vision) = &condition.vision else {
            return Ok(());
        };
        let patches = take(
            products.patches.as_ref(),
            &mut cursor.patches,
            vision.grid.patches(),
            "patches",
        )?;
        let indices = integer_list(vision.frame_indices.iter().map(|&index| i64::from(index)));
        let grid = integer_list([vision.grid.t, vision.grid.h, vision.grid.w].map(i64::from));
        self.functions.pack_vision.invoke(&[
            self.functions.loaded.view(),
            AnyView::from(frames),
            AnyView::from(&Any::from(indices)),
            AnyView::from(&Any::from(grid)),
            AnyView::from(&Any::from(patches.to_tensor()?)),
        ])
    }

    fn read_track(
        &self,
        source: &MediaLocator,
        track: &AudioClip,
        products: &ConditionProducts,
        cursor: &mut Cursor,
    ) -> Result<()> {
        let samples = take(
            products.samples.as_ref(),
            &mut cursor.samples,
            track.samples,
            "samples",
        )?;
        let data = host_bytes(read_bytes(source)?)?;
        self.functions.read_audio.invoke(&[
            AnyView::from(&data),
            AnyView::from(&Any::from(i64::from(track.sample_rate))),
            AnyView::from(&Any::from(track.start_sample as i64)),
            AnyView::from(&Any::from(track.source_samples as i64)),
            AnyView::from(&Any::from(track.samples as i64)),
            AnyView::from(&Any::from(i64::from(self.functions.sample_rate))),
            AnyView::from(&Any::from(samples.to_tensor()?)),
        ])
    }

    /// Decode a reference video's kept frames on its planned canvas with one
    /// FFmpeg pass: resampled to the model's frame rate, the start frames
    /// skipped, then scaled per frame with LANCZOS (the scale follows the
    /// trim, so it never sees a skipped frame). Returns host
    /// `[frames, height, width, 3]` uint8.
    fn read_video(&self, source: &MediaLocator, clip: &VideoClip) -> Result<StorageView> {
        let canvas = clip.canvas;
        let filters = format!(
            "fps={},trim=start_frame={},scale={}:{}:flags=lanczos,setsar=1",
            self.frame_rate, clip.start_frame, canvas.width, canvas.height
        );
        let frames_arg = clip.frames.to_string();
        // The shared-memory object is a seekable file FFmpeg reads directly.
        let input = SharedMemory::open(&source.name, false).map_err(|error| {
            Error::Invalid(format!(
                "condition media {} cannot be opened: {error}",
                source.name
            ))
        })?;
        let mut process = Command::new(&self.ffmpeg)
            .args([
                "-v",
                "error",
                "-i",
                "fd:",
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                &filters,
                "-frames:v",
                &frames_arg,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ])
            .stdin(Stdio::from(input))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| {
                Error::Invalid(format!(
                    "cannot start {} for a reference video: {error}",
                    self.ffmpeg
                ))
            })?;
        // Diagnostics drain on their own thread, so a full pipe never stalls
        // FFmpeg while frames stream into the result.
        let diagnostics = process.stderr.take().map(|mut stderr| {
            std::thread::spawn(move || {
                let mut text = Vec::new();
                let _ = stderr.read_to_end(&mut text);
                text
            })
        });

        let frame_bytes = u64::from(canvas.height) * u64::from(canvas.width) * 3;
        let length = frame_bytes * u64::from(clip.frames);
        let memory = Memory::host(length as usize);
        let mut filled = 0usize;
        let mut trailing = 0usize;
        if let Some(mut stdout) = process.stdout.take() {
            // SAFETY: the host allocation is owned here, has `length` bytes
            // and no other reference while FFmpeg's output is read into it.
            let target = unsafe {
                std::slice::from_raw_parts_mut(memory.address() as *mut u8, length as usize)
            };
            while filled < target.len() {
                match stdout.read(&mut target[filled..]) {
                    Ok(0) => break,
                    Ok(count) => filled += count,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(error) => {
                        return Err(Error::Invalid(format!(
                            "reading a decoded reference video failed: {error}"
                        )));
                    }
                }
            }
            // `-frames:v` stops the output at the planned count.
            let mut rest = Vec::new();
            let _ = stdout.read_to_end(&mut rest);
            trailing = rest.len();
        }
        let status = process
            .wait()
            .map_err(|error| Error::Invalid(format!("waiting for FFmpeg failed: {error}")))?;
        let message = diagnostics
            .and_then(|thread| thread.join().ok())
            .map(|text| String::from_utf8_lossy(&text).trim().to_owned())
            .unwrap_or_default();
        if !status.success() {
            return Err(Error::Invalid(format!(
                "decoding a reference video failed: {message}"
            )));
        }
        if filled as u64 != length || trailing != 0 {
            return Err(Error::Invalid(format!(
                "a reference video decoded to {} frames, {} were planned",
                (filled + trailing) as f64 / frame_bytes as f64,
                clip.frames
            )));
        }
        StorageView::contiguous(
            memory as Arc<dyn Backing>,
            0,
            DType::U8,
            &[
                i64::from(clip.frames),
                i64::from(canvas.height),
                i64::from(canvas.width),
                3,
            ],
        )
    }
}

/// Copy a contiguous host view into another of the same byte length.
fn copy_host(source: &StorageView, destination: &StorageView) -> Result<()> {
    let bytes = source.nbytes();
    if bytes != destination.nbytes() || !source.is_contiguous() || !destination.is_contiguous() {
        return Err(Error::Invariant(
            "condition pixels disagree with their product rows".into(),
        ));
    }
    // SAFETY: both views are contiguous host memory of `bytes` bytes owned by
    // their allocations; the destination product has no reader until the
    // reading call completes.
    unsafe {
        std::ptr::copy_nonoverlapping(
            source.address() as *const u8,
            destination.address() as *mut u8,
            bytes as usize,
        );
    }
    Ok(())
}
