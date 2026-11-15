//! A request's condition media laid out as the model's encoders read them, on the host.
//!
//! The media reader decodes a condition's bytes into RGB frames and stereo PCM, the media's own
//! forms. What an encoder reads is stated by the model ([`MediaFormat`]): the normalization of its
//! pixels, the patch rows of its vision tower, the channels and padding of a track. The
//! conversions here execute
//! that statement on the host thread that reads the request's conditions (a thread of the media
//! host's pool, or the vision runner's image read), sharing a large image's work out over
//! [`share`]'s scoped threads, as many as the process's host-thread budget
//! (`worker_config.host_threads`), so no model holds a thread or a byte format.

use std::num::NonZeroUsize;

use nn::DType;
use nn::capability::{MediaFormat, PatchFormat, PixelNorm};

use crate::worker::{Error, Result};

/// Runs `work` over `items`, a run of them on each of `threads` scoped threads of this call.
pub(crate) fn share<T: Send>(threads: NonZeroUsize, items: &mut [T], work: impl Fn(&mut T) + Sync) {
    let run = items.len().div_ceil(threads.get()).max(1);
    let work = &work;
    std::thread::scope(|scope| {
        for items in items.chunks_mut(run) {
            scope.spawn(move || items.iter_mut().for_each(work));
        }
    });
}

/// How many bands a plane of `area` pixels is cut into so that `frames` frames' bands keep each of
/// `threads` threads busy: the band length.
pub(crate) fn band(threads: NonZeroUsize, area: usize, frames: usize) -> usize {
    area.div_ceil(threads.get().div_ceil(frames.max(1))).max(1)
}

/// The refusal of a conversion whose buffers do not hold what it writes.
fn mismatched(problem: String) -> Error {
    Error::Invalid(format!(
        "a condition's media does not fit its layout: {problem}"
    ))
}

/// Writes `[frames, height, width, 3]` RGB bytes into `out` as `[3, frames, height, width]` FP32,
/// each byte normalized as `(byte / 255 - mean) / std` in FP32, the work shared over `threads`
/// threads.
///
/// A byte has 256 values, so the normalization is a table of them, and a frame's pixels are shared
/// out in bands: a reference image at its encoding size is twenty million elements.
pub(crate) fn pixels(
    norm: &PixelNorm,
    rgb: &[u8],
    frames: u32,
    height: u32,
    width: u32,
    out: &mut [f32],
    threads: NonZeroUsize,
) -> Result<()> {
    let area = height as usize * width as usize;
    let frames = frames as usize;
    let channels = 3;
    if rgb.len() != frames * area * channels || out.len() != rgb.len() {
        return Err(mismatched(format!(
            "{} RGB bytes and {} elements for {frames} frames of {height}x{width}",
            rgb.len(),
            out.len()
        )));
    }
    let mut table = [[0.0_f32; 256]; 3];
    for byte in 0..=u8::MAX {
        for (channel, row) in table.iter_mut().enumerate() {
            row[usize::from(byte)] =
                (f32::from(byte) / 255.0 - norm.mean[channel]) / norm.std[channel];
        }
    }
    let table = &table;
    let band = band(threads, area, frames);
    let mut planes: Vec<_> = out
        .chunks_mut(area)
        .map(|plane| plane.chunks_mut(band))
        .collect();
    let mut bands = Vec::new();
    for frame in 0..frames {
        let mut first = frame * area;
        while let (Some(red), Some(green), Some(blue)) = (
            planes[frame].next(),
            planes[frames + frame].next(),
            planes[2 * frames + frame].next(),
        ) {
            let pixels = red.len();
            bands.push((first, red, green, blue));
            first += pixels;
        }
    }
    share(threads, &mut bands, |(first, red, green, blue)| {
        let source = &rgb[*first * channels..(*first + red.len()) * channels];
        for (index, pixel) in source.chunks_exact(channels).enumerate() {
            red[index] = table[0][usize::from(pixel[0])];
            green[index] = table[1][usize::from(pixel[1])];
            blue[index] = table[2][usize::from(pixel[2])];
        }
    });
    Ok(())
}

/// Writes `[frames, channels, height, width]` planar bytes, already at the grid's raster, into `out`
/// as the `grid[0] * grid[1] * grid[2]` patch rows a vision tower reads, in [`PatchFormat::dtype`].
///
/// The processor's arithmetic, op for op: `(byte - 255 · mean) / (255 · std)` in FP32, rounded once
/// to the tower's element type; the last frame repeated to a whole temporal patch; rows over each
/// temporal patch in merge-block order, each one channel-major tubelet. The work is shared over
/// `threads` threads.
pub(crate) fn patches(
    format: &PatchFormat,
    planes: &[u8],
    frames: u32,
    grid: [u32; 3],
    out: &mut [u8],
    threads: NonZeroUsize,
) -> Result<()> {
    let [time, rows, columns] = grid;
    let (patch, merge, temporal) = (format.side, format.merge, format.temporal);
    let channels = format.channels as usize;
    let (height, width) = ((rows * patch) as usize, (columns * patch) as usize);
    let area = height * width;
    let element = ops::abi::bytes(format.dtype).ok_or_else(|| {
        mismatched(format!(
            "{} patch rows have no whole-byte element",
            ops::abi::name(format.dtype)
        ))
    })? as usize;
    let width_bytes = format.width() as usize * element;
    let count = time as usize * rows as usize * columns as usize;
    if planes.len() != frames as usize * channels * area
        || out.len() != count * width_bytes
        || frames == 0
        || frames.div_ceil(temporal) != time
        || !rows.is_multiple_of(merge)
        || !columns.is_multiple_of(merge)
    {
        return Err(mismatched(format!(
            "{} plane bytes and {} row bytes for {frames} frames on a {grid:?} grid",
            planes.len(),
            out.len()
        )));
    }
    let mean = format.norm.mean.map(|value| value * 255.0);
    let std = format.norm.std.map(|value| value * 255.0);
    let (patch, merge, temporal) = (patch as usize, merge as usize, temporal as usize);
    // A byte has 256 values, so a channel's normalised element is a table of them, each the FP32
    // value rounded once to the tower's type.
    let mut values = Vec::with_capacity(channels * 256);
    for channel in 0..channels {
        for byte in 0..=u8::MAX {
            values.push((f32::from(byte) - mean[channel]) / std[channel]);
        }
    }
    let table = encode(format.dtype, &values)?;
    let table = &table;
    // One band is one temporal patch's row of merge blocks: its bytes follow its predecessor's, so
    // the bands are shared out as they lie in `out`.
    let block_rows = rows as usize / merge;
    let block_columns = columns as usize / merge;
    let band = count / (time as usize * block_rows) * width_bytes;
    let mut bands: Vec<(usize, &mut [u8])> = out.chunks_mut(band).enumerate().collect();
    share(threads, &mut bands, |(index, out)| {
        let (step, block_row) = (*index / block_rows, *index % block_rows);
        let mut at = 0_usize;
        for block_column in 0..block_columns {
            for merge_row in 0..merge {
                for merge_column in 0..merge {
                    let top = (block_row * merge + merge_row) * patch;
                    let left = (block_column * merge + merge_column) * patch;
                    for channel in 0..channels {
                        for offset in 0..temporal {
                            // A short temporal patch repeats the last frame: one image is one frame
                            // repeated.
                            let frame = (step * temporal + offset).min(frames as usize - 1);
                            let plane = (frame * channels + channel) * area;
                            for y in 0..patch {
                                let line = plane + (top + y) * width + left;
                                for x in 0..patch {
                                    let entry =
                                        (channel * 256 + usize::from(planes[line + x])) * element;
                                    out[at..at + element]
                                        .copy_from_slice(&table[entry..entry + element]);
                                    at += element;
                                }
                            }
                        }
                    }
                }
            }
        }
    });
    Ok(())
}

/// One vision block's frames, `[frames, height, width, 3]` RGB at `raster`, as its patch rows.
///
/// The tower reads a grid's own pixels, so each frame is resized to the raster the processor sized
/// the grid from ([`PatchFormat::raster`]) first and packed second, the work shared over
/// `threads` threads.
pub(crate) fn pack_block(
    format: &PatchFormat,
    rgb: &[u8],
    frames: u32,
    raster: (u32, u32),
    grid: [u32; 3],
    out: &mut [u8],
    threads: NonZeroUsize,
) -> Result<()> {
    let (height, width) = raster;
    let (rows, columns) = format.raster(grid);
    let planes = planes_of(
        rgb,
        frames as usize,
        (height as usize, width as usize),
        threads,
    );
    let planes = if (rows, columns) == (height, width) {
        planes
    } else {
        let frame = 3 * height as usize * width as usize;
        let mut resized = Vec::with_capacity(frames as usize * 3 * (rows * columns) as usize);
        for source in planes.chunks_exact(frame) {
            resized.extend(media::resample::bicubic_antialias_u8(
                source, 3, height, width, rows, columns,
            )?);
        }
        resized
    };
    patches(format, &planes, frames, grid, out, threads)
}

/// `[frames, height, width, 3]` interleaved RGB as `[frames, 3, height, width]` planes, the work
/// shared over `threads` threads.
fn planes_of(
    rgb: &[u8],
    frames: usize,
    (height, width): (usize, usize),
    threads: NonZeroUsize,
) -> Vec<u8> {
    let area = height * width;
    let mut planes = vec![0_u8; frames * 3 * area];

    // A reference image at its encoding size is seven million pixels on a request's critical path,
    // so a frame's pixels are shared out in bands of its three planes.
    let band = band(threads, area, frames);
    let mut bands = Vec::new();
    for (frame, block) in planes.chunks_mut((3 * area).max(1)).enumerate() {
        let (red, rest) = block.split_at_mut(area);
        let (green, blue) = rest.split_at_mut(area);
        let mut first = frame * area;
        for ((red, green), blue) in red
            .chunks_mut(band)
            .zip(green.chunks_mut(band))
            .zip(blue.chunks_mut(band))
        {
            let pixels = red.len();
            bands.push((first, red, green, blue));
            first += pixels;
        }
    }
    share(threads, &mut bands, |(first, red, green, blue)| {
        let pixels = &rgb[*first * 3..(*first + red.len()) * 3];
        for (index, pixel) in pixels.as_chunks::<3>().0.iter().enumerate() {
            red[index] = pixel[0];
            green[index] = pixel[1];
            blue[index] = pixel[2];
        }
    });
    planes
}

/// `values` in `dtype`'s little-endian bytes.
fn encode(dtype: DType, values: &[f32]) -> Result<Vec<u8>> {
    Ok(ops::values::encode(dtype, values)?)
}

/// The samples an encoder reads of a track of `samples` samples: the track right-padded with zeros
/// to a whole number of `format`'s latent hops, the reference's `unpack_samples`.
pub(crate) fn track_samples(format: &MediaFormat, samples: u64) -> Result<u64> {
    let hop = format.audio.hop;
    if samples == 0 || hop == 0 {
        return Err(mismatched(
            "an audio track of no samples encodes to no latent frames".to_owned(),
        ));
    }
    Ok(samples.div_ceil(hop) * hop)
}

/// The channels of the PCM the media reader decodes: interleaved stereo.
const READ_CHANNELS: u32 = 2;

/// Writes `[samples, 2]` interleaved stereo, as the media reader decodes it, into `out` as the
/// encoder's `[channels, 1, padded]` channel rows, each channel's tail zero:
/// `channels * track_samples(format, samples)` elements. An encoder that reads another number
/// of channels than the reader decodes is refused, since no channel is invented or mixed.
pub(crate) fn track(
    format: &MediaFormat,
    pcm: &[f32],
    samples: u64,
    out: &mut [f32],
) -> Result<()> {
    if format.audio.channels != READ_CHANNELS {
        return Err(mismatched(format!(
            "an audio encoder reads {} channels and the media reader decodes {READ_CHANNELS}",
            format.audio.channels
        )));
    }
    let padded = track_samples(format, samples)? as usize;
    let samples = samples as usize;
    if pcm.len() != 2 * samples || out.len() != 2 * padded {
        return Err(mismatched(format!(
            "{} PCM values and {} elements for a track of {samples} samples",
            pcm.len(),
            out.len()
        )));
    }
    out.fill(0.0);
    for sample in 0..samples {
        out[sample] = pcm[2 * sample];
        out[padded + sample] = pcm[2 * sample + 1];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use nn::capability::AudioFormat;

    /// The threads a test's conversion shares its work over: more than one, so the shares meet.
    const THREADS: NonZeroUsize = NonZeroUsize::new(2).unwrap();

    /// The format of a tower of `patch`-pixel patches, two-frame tubelets and `2 x 2` merges, its
    /// pixels and patches normalized about one half.
    fn format(patch: u32) -> MediaFormat {
        let half = PixelNorm {
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        MediaFormat {
            pixels: half,
            patches: PatchFormat {
                side: patch,
                temporal: 2,
                merge: 2,
                channels: 3,
                norm: half,
                dtype: DType::BF16,
            },
            audio: AudioFormat {
                channels: 2,
                hop: 800,
            },
        }
    }

    fn row_value(rows: &[u8], index: usize) -> f32 {
        let bits = u16::from_le_bytes([rows[2 * index], rows[2 * index + 1]]);
        bf16::from_bits(bits).to_f32()
    }

    /// Frames reach the resize as planes, one frame after another: an interleaved raster read as
    /// planes would resize a mixture of the channels, and every shape would agree.
    #[test]
    fn interleaved_frames_become_planes_one_frame_after_another() {
        // Two frames of one row of two pixels.
        let rgb = [1_u8, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16];
        assert_eq!(
            planes_of(&rgb, 2, (1, 2), THREADS),
            [1, 4, 2, 5, 3, 6, 11, 14, 12, 15, 13, 16]
        );
    }

    /// Patch rows run over merge blocks: the four patches of a `2 x 2` block are consecutive rows,
    /// and a block's rows come before the next block's. A row-major walk of the patches would give
    /// every merged token the wrong four, which no shape refuses.
    #[test]
    fn patch_rows_run_over_merge_blocks_and_a_row_is_one_channel_major_tubelet() {
        let format = format(1).patches;
        // One frame of 2 rows by 4 columns, channel c of pixel (y, x) holding 8 c + 4 y + x.
        let (height, width) = (2_usize, 4_usize);
        let mut planes = vec![0_u8; 3 * height * width];
        for channel in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    planes[(channel * height + y) * width + x] = (8 * channel + 4 * y + x) as u8;
                }
            }
        }
        let rows = 8;
        let row = format.width() as usize;
        assert_eq!(row, 3 * 2);
        let mut out = vec![0_u8; rows * row * 2];
        patches(&format, &planes, 1, [1, 2, 4], &mut out, THREADS)
            .expect("one frame fills one temporal patch");
        let pixel = |byte: u8| (f32::from(byte) - 127.5) / 127.5;
        let rounded = |value: f32| bf16::from_f32(value).to_f32();
        // The rows of the first block are pixels (0,0), (0,1), (1,0), (1,1); the second block's
        // first row is pixel (0,2). A row is channel-major with its two frames inside a channel, and
        // the one frame fills both.
        let order = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
        ];
        for (index, (y, x)) in order.iter().enumerate() {
            for channel in 0..3 {
                let expected = rounded(pixel((8 * channel + 4 * y + x) as u8));
                for frame in 0..2 {
                    assert_eq!(
                        row_value(&out, index * row + channel * 2 + frame),
                        expected,
                        "row {index} channel {channel} frame {frame}"
                    );
                }
            }
        }
    }

    /// An image's temporal patch repeats its one frame, and a video's short last patch repeats its
    /// last frame: a zero there would hand the tower a frame nobody decoded.
    #[test]
    fn a_short_temporal_patch_repeats_the_last_frame() {
        let format = format(2).patches;
        let frame = 3 * 4 * 4;
        // Three frames of 4 x 4, each one value: two temporal patches, the second one frame short.
        let mut planes = vec![0_u8; 3 * frame];
        for (index, value) in [10_u8, 20, 30].iter().enumerate() {
            planes[index * frame..(index + 1) * frame].fill(*value);
        }
        let row = format.width() as usize;
        let rows = 2 * 2 * 2;
        let mut out = vec![0_u8; rows * row * 2];
        patches(&format, &planes, 3, [2, 2, 2], &mut out, THREADS)
            .expect("three frames fill two temporal patches");
        let pixel = |byte: u8| bf16::from_f32((f32::from(byte) - 127.5) / 127.5).to_f32();
        // Inside a channel the tubelet holds its frames in order, four pixels each.
        assert_eq!(row_value(&out, 0), pixel(10));
        assert_eq!(row_value(&out, 4), pixel(20));
        let second = 4 * row;
        assert_eq!(row_value(&out, second), pixel(30));
        assert_eq!(row_value(&out, second + 4), pixel(30));
    }

    /// The patch rows are stored in the tower's own element type: a tower loaded in FP32 reads
    /// four-byte rows of the unrounded values.
    #[test]
    fn patch_rows_are_stored_in_the_towers_element_type() {
        let format = PatchFormat {
            dtype: DType::F32,
            ..format(1).patches
        };
        // One frame of 2 rows by 2 columns: one merge block, four patch rows.
        let planes: Vec<u8> = (0..3 * 4).map(|value| (10 * value) as u8).collect();
        let row = format.width() as usize;
        let mut out = vec![0_u8; 4 * row * 4];
        patches(&format, &planes, 1, [1, 2, 2], &mut out, THREADS).expect("FP32 rows");
        let value = |index: usize| {
            f32::from_le_bytes([
                out[4 * index],
                out[4 * index + 1],
                out[4 * index + 2],
                out[4 * index + 3],
            ])
        };
        // Row 0 is pixel (0, 0): channel 0 holds byte 0, channel 1 byte 40, each frame twice.
        assert_eq!(value(0), (0.0 - 127.5) / 127.5);
        assert_eq!(value(2), (40.0 - 127.5) / 127.5);
        // A buffer of BF16 width is two bytes per element short.
        let mut short = vec![0_u8; 4 * row * 2];
        assert!(patches(&format, &planes, 1, [1, 2, 2], &mut short, THREADS).is_err());
    }

    /// A condition's pixels are planar and normalised as the encoder reads them, and the track is
    /// two channel rows padded to whole latent frames.
    #[test]
    fn a_conditions_pixels_are_planar_and_a_track_is_padded_channel_rows() {
        let format = format(16);
        let rgb = [255_u8, 0, 128, 0, 255, 128];
        let mut out = [0.0_f32; 6];
        pixels(&format.pixels, &rgb, 1, 1, 2, &mut out, THREADS).expect("one frame of two pixels");
        let normalized = |byte: u8| (f32::from(byte) / 255.0 - 0.5) / 0.5;
        assert_eq!(
            out,
            [255, 0, 0, 255, 128, 128].map(normalized),
            "red plane, then green, then blue"
        );

        assert_eq!(track_samples(&format, 801).expect("a whole frame"), 1600);
        assert!(track_samples(&format, 0).is_err());
        let pcm = [1.0_f32, -1.0, 2.0, -2.0];
        let mut padded = vec![9.0_f32; 2 * 800];
        track(&format, &pcm, 2, &mut padded).expect("two samples");
        assert_eq!(&padded[..3], &[1.0, 2.0, 0.0]);
        assert_eq!(&padded[800..803], &[-1.0, -2.0, 0.0]);
    }
}
