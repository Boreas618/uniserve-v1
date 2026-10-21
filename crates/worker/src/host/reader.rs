//! Reading a video request's condition media on a CPU worker.
//!
//! The server publishes every condition's fetched bytes to POSIX shared
//! memory on the head host and plans how each is prepared
//! (`models::video::VideoCondition`). One media reading call per conditioned
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
//!
//! Each condition's rows of each product follow from the plan before
//! anything is decoded, so the conditions decode in any order: every
//! reference video's FFmpeg process starts first, on a thread of its own,
//! and the library's calls (images, tracks, then each video's patches once
//! its frames arrived) run on the Python thread meanwhile.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::Arc;

use models::video::{AudioClip, ConditionMedia, MediaLocator, VideoClip, VideoCondition};
use tvm_ffi::{Any, AnyView};
use uniserve_ipc::SharedMemory;

use crate::ffi::{Backing, DType, Function, Handle, Interpreter, integer_list};
use crate::storage::memory::{Storage, StorageView};
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
    pub(crate) fn new(loaded: &Handle) -> Result<Self> {
        const ENCODER: &str = "uniserve.runner.encoder.EncoderRunner";
        let sample_rate: i64 =
            Interpreter::import(&format!("{ENCODER}.sample_rate"))?.call_as(&[loaded.view()])?;
        Ok(Self {
            read_image: Interpreter::import("uniserve.media.reader.read_image")?,
            read_audio: Interpreter::import("uniserve.media.reader.read_audio")?,
            pack_vision: Interpreter::import(&format!("{ENCODER}.pack_pixels"))?,
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

/// Rows of each product taken so far.
#[derive(Default)]
struct Cursor {
    pixels: i64,
    samples: i64,
    patches: i64,
}

/// Where one condition's decoded media lands: its rows of each product.
#[derive(Default)]
struct Rows {
    /// The pixels the video encoder encodes.
    pixels: Option<StorageView>,
    /// The conditioner's patch rows, for a condition it reads.
    patches: Option<StorageView>,
    /// The condition's audio track, a video's soundtrack included.
    samples: Option<StorageView>,
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

/// A condition's media has no rows taken for it.
fn missing_rows() -> Error {
    Error::Invariant("a condition's product rows were not taken".into())
}

/// A host view over `bytes`.
fn host_bytes(bytes: Vec<u8>) -> Result<StorageView> {
    let length = bytes.len() as i64;
    let memory = Storage::host(bytes.len().max(1));
    if !bytes.is_empty() {
        // SAFETY: the allocation was just created with at least this length
        // and has no other reference.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), memory.address() as *mut u8, bytes.len());
        }
    }
    StorageView::contiguous(memory as Arc<dyn Backing>, 0, DType::U8, &[length])
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

    /// Decode every condition into its rows of the products, which together
    /// fill each product exactly from its first row, in request order.
    ///
    /// Every reference video's FFmpeg decode runs on a thread of its own
    /// from the start, beside the library's calls on the Python thread:
    /// each image and audio track in request order, then each video's
    /// patches once its frames arrived.
    pub(crate) fn read(
        &self,
        conditions: &[VideoCondition],
        products: &ConditionProducts,
    ) -> Result<()> {
        let rows = Self::rows(conditions, products)?;
        std::thread::scope(|scope| {
            let videos: Vec<_> = conditions
                .iter()
                .map(|condition| match &condition.media {
                    ConditionMedia::Video { clip, .. } => {
                        Some(scope.spawn(move || self.read_video(&condition.source, clip)))
                    }
                    _ => None,
                })
                .collect();

            for (condition, rows) in conditions.iter().zip(&rows) {
                if let ConditionMedia::Image(fit) = &condition.media {
                    let pixels = rows.pixels.as_ref().ok_or_else(missing_rows)?;
                    let data = host_bytes(read_bytes(&condition.source)?)?;
                    let resized =
                        integer_list([fit.resized.width, fit.resized.height].map(i64::from));
                    let window = integer_list(
                        [fit.left, fit.top, fit.size.width, fit.size.height].map(i64::from),
                    );
                    // The views' wrappings and the call take one handoff.
                    Interpreter::execute(|| {
                        self.functions.read_image.invoke(&[
                            data.handle()?.view(),
                            AnyView::from(&Any::from(resized)),
                            AnyView::from(&Any::from(window)),
                            pixels.handle()?.view(),
                        ])
                    })?;
                    self.pack(condition, pixels, rows)?;
                }
                if let Some(track) = condition.media.audio() {
                    let samples = rows.samples.as_ref().ok_or_else(missing_rows)?;
                    self.read_track(&condition.source, track, samples)?;
                }
            }

            for ((condition, rows), video) in conditions.iter().zip(&rows).zip(videos) {
                let (Some(video), ConditionMedia::Video { clip, .. }) = (video, &condition.media)
                else {
                    continue;
                };
                let frames = video.join().map_err(|_| {
                    Error::Invariant("a reference video's decode panicked".into())
                })??;
                // The video encoder encodes the leading kept frames.
                let encoded = frames.narrow(0, 0, i64::from(clip.vae_frames))?;
                copy_host(&encoded, rows.pixels.as_ref().ok_or_else(missing_rows)?)?;
                self.pack(condition, &frames, rows)?;
            }
            Ok(())
        })
    }

    /// Each condition's rows of each product, taken in request order, which
    /// must fill every product exactly.
    fn rows(conditions: &[VideoCondition], products: &ConditionProducts) -> Result<Vec<Rows>> {
        let mut cursor = Cursor::default();
        let mut taken = Vec::with_capacity(conditions.len());
        for condition in conditions {
            let mut rows = Rows::default();
            match &condition.media {
                ConditionMedia::Image(fit) => {
                    let (height, width) = (fit.size.height, fit.size.width);
                    rows.pixels = Some(
                        take(
                            products.pixels.as_ref(),
                            &mut cursor.pixels,
                            u64::from(height) * u64::from(width),
                            "pixels",
                        )?
                        .reshape(DType::U8, &[1, i64::from(height), i64::from(width), 3])?,
                    );
                }
                ConditionMedia::Video { clip, .. } => {
                    let area = u64::from(clip.canvas.height) * u64::from(clip.canvas.width);
                    rows.pixels = Some(take(
                        products.pixels.as_ref(),
                        &mut cursor.pixels,
                        u64::from(clip.vae_frames) * area,
                        "pixels",
                    )?);
                }
                ConditionMedia::Audio(_) => {}
            }
            if let Some(vision) = &condition.vision {
                rows.patches = Some(take(
                    products.patches.as_ref(),
                    &mut cursor.patches,
                    vision.grid.patches(),
                    "patches",
                )?);
            }
            if let Some(track) = condition.media.audio() {
                rows.samples = Some(take(
                    products.samples.as_ref(),
                    &mut cursor.samples,
                    track.samples,
                    "samples",
                )?);
            }
            taken.push(rows);
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
        Ok(taken)
    }

    /// Pack the conditioner's patch rows of a condition it reads from its
    /// decoded `frames` into its `rows`.
    fn pack(&self, condition: &VideoCondition, frames: &StorageView, rows: &Rows) -> Result<()> {
        let Some(vision) = &condition.vision else {
            return Ok(());
        };
        let patches = rows.patches.as_ref().ok_or_else(missing_rows)?;
        let indices = integer_list(vision.frame_indices.iter().map(|&index| i64::from(index)));
        let grid = integer_list([vision.grid.t, vision.grid.h, vision.grid.w].map(i64::from));
        Interpreter::execute(|| {
            self.functions.pack_vision.invoke(&[
                self.functions.loaded.view(),
                frames.handle()?.view(),
                AnyView::from(&Any::from(indices)),
                AnyView::from(&Any::from(grid)),
                patches.handle()?.view(),
            ])
        })
    }

    /// Decode an audio track into its `samples` rows.
    fn read_track(
        &self,
        source: &MediaLocator,
        track: &AudioClip,
        samples: &StorageView,
    ) -> Result<()> {
        let data = host_bytes(read_bytes(source)?)?;
        Interpreter::execute(|| {
            self.functions.read_audio.invoke(&[
                data.handle()?.view(),
                AnyView::from(&Any::from(i64::from(track.sample_rate))),
                AnyView::from(&Any::from(track.start_sample as i64)),
                AnyView::from(&Any::from(track.source_samples as i64)),
                AnyView::from(&Any::from(track.samples as i64)),
                AnyView::from(&Any::from(i64::from(self.functions.sample_rate))),
                samples.handle()?.view(),
            ])
        })
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
        let memory = Storage::host(length as usize);
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

#[cfg(test)]
mod tests {
    use models::video::{
        Canvas, ConditionRole, ConditionVision, ImageFit, MediaLocator, VisionGrid,
    };

    use super::*;

    fn product(rows: i64, width: i64, dtype: DType) -> StorageView {
        let bytes = (rows * width) as usize * dtype.bytes();
        StorageView::contiguous(
            Storage::host(bytes) as Arc<dyn Backing>,
            0,
            dtype,
            &[rows, width],
        )
        .unwrap()
    }

    fn condition(media: ConditionMedia, vision: Option<u32>) -> VideoCondition {
        VideoCondition {
            role: ConditionRole::Reference,
            source: MediaLocator {
                name: "uniserve-media-0".to_owned(),
                bytes: 1,
            },
            media,
            vision: vision.map(|patches| ConditionVision {
                grid: VisionGrid {
                    t: 1,
                    h: 1,
                    w: patches,
                },
                tokens: patches,
                frame_indices: Vec::new(),
            }),
            latent_units: Vec::new(),
            audio_rows: 0,
        }
    }

    /// Where `view` starts in `product`, in rows.
    fn row(view: Option<&StorageView>, product: &StorageView) -> Option<u64> {
        let bytes = product.nbytes() / product.shape()[0] as u64;
        view.map(|view| (view.address() - product.address()) / bytes)
    }

    /// Each condition's rows of each product follow the conditions before
    /// it in request order, whatever order the conditions decode in, and
    /// together fill every product; a product left short is refused.
    #[test]
    fn conditions_take_their_product_rows_in_request_order() {
        let canvas = Canvas {
            width: 2,
            height: 2,
        };
        let track = |samples| AudioClip {
            sample_rate: 32_000,
            start_sample: 0,
            source_samples: samples,
            samples,
        };
        let conditions = [
            condition(
                ConditionMedia::Video {
                    clip: models::video::VideoClip {
                        canvas,
                        start_frame: 0,
                        frames: 7,
                        vae_frames: 5,
                    },
                    soundtrack: Some(track(10)),
                },
                Some(6),
            ),
            condition(
                ConditionMedia::Image(ImageFit {
                    resized: canvas,
                    left: 0,
                    top: 0,
                    size: canvas,
                }),
                Some(2),
            ),
            condition(ConditionMedia::Audio(track(3)), None),
        ];
        let products = ConditionProducts {
            pixels: Some(product(5 * 4 + 4, 3, DType::U8)),
            samples: Some(product(13, 2, DType::F32)),
            patches: Some(product(8, 4, DType::F32)),
        };
        let rows = ConditionReader::rows(&conditions, &products).unwrap();

        let pixels = products.pixels.as_ref().unwrap();
        let samples = products.samples.as_ref().unwrap();
        let patches = products.patches.as_ref().unwrap();
        let at = |index: usize| {
            (
                row(rows[index].pixels.as_ref(), pixels),
                row(rows[index].samples.as_ref(), samples),
                row(rows[index].patches.as_ref(), patches),
            )
        };
        assert_eq!(at(0), (Some(0), Some(0), Some(0)));
        assert_eq!(at(1), (Some(20), None, Some(6)));
        assert_eq!(at(2), (None, Some(10), None));

        let short = ConditionProducts {
            samples: Some(product(14, 2, DType::F32)),
            ..products
        };
        assert!(ConditionReader::rows(&conditions, &short).is_err());
    }
}
