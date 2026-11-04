//! The host-side position tables of the text encoder and its vision tower.
//!
//! Every table here is a function of the *request*, not of a step: the prompt's token ids and the
//! grids of its condition media decide them once, before the encode is captured or replayed
//! (specs/worker.md 4.3, 9). The executor's encoder runner calls these functions when it prepares a
//! request and uploads the results into the views [`crate::text_encoder`] reads; the model crate
//! itself owns no device storage and copies nothing (`AGENTS.md`, Model Library Boundary).
//!
//! Four kinds of table are derived:
//!
//! - the language layers' interleaved M-RoPE coordinates ([`rope_index`], `uniserve/models/`
//!   `qwen3_vl/positions.py`) and their FP32 factors ([`Mrope::factors`]);
//! - the vision tower's 2-D rotary coordinates of a patch grid in merge-block order
//!   ([`rotary_coordinates`]) and their factors ([`VisionRotary::factors`]);
//! - the learned position table's bilinear sample of each merged patch ([`position_samples`]), the
//!   indices and FP32 weights of the four table rows a merged row blends
//!   (`uniserve/nn/vision/position.py::PositionEmbedding.interpolate`);
//! - the vision tower's attention lengths ([`vision_lengths`]: one bidirectional sequence per
//!   temporal patch of every grid) and the row of the vision feature table each prompt placeholder
//!   takes ([`placeholder_rows`]).
//!
//! The arithmetic is the reference's, in its order and precision: frequencies and factors in FP32,
//! positions as integers, and the position table's sample coordinates on an FP32 `linspace` whose
//! step is computed in FP32 (`torch.linspace` over a float32 scalar type).

use crate::{Error, Result};

/// The largest number of position axes a sectioned rotation reads.
///
/// The text encoder uses three (temporal, height, width); a fourth is room for a model that adds a
/// modality axis, as `ops::rope::MAX_AXES` does.
pub const MAX_AXES: usize = 4;

/// The interleaved M-RoPE sections of the language layers.
///
/// The checkpoint states `mrope_section: [24, 20, 20]` and `mrope_interleaved: true`: the head's
/// compact frequency `j` rotates by the coordinate of axis `j % 3` while `j < 24 * 3`, and axis 0
/// takes every remaining frequency (`uniserve/nn/rope.py::_interleaved_axes`). Each axis therefore
/// receives exactly its section width and keeps frequencies from the whole spectrum.
/// The table rows one bilinear sample reads: four, the corners of one cell.
pub const POSITION_TAPS: u64 = 4;

/// The M-RoPE cut of a head's rotation: one width per axis, interleaved.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Mrope {
    widths: [u32; 3],
    head_dim: u32,
}

impl Mrope {
    /// The sections `widths` (frequencies per axis) of a `head_dim`-element head.
    ///
    /// The widths must be positive and split half the head, and the axes must be interleaved: H3's
    /// checkpoint states `mrope_interleaved: true`, and the contiguous cut
    /// (`ops::rope::FactorsDescriptor::sections`) is a different coordinate assignment.
    pub fn new(widths: [u32; 3], head_dim: u32) -> Result<Self> {
        if widths.contains(&0) {
            return Err(invalid(
                "Mrope",
                format!("the M-RoPE sections {widths:?} must be positive"),
            ));
        }
        if !head_dim.is_multiple_of(2) {
            return Err(invalid(
                "Mrope",
                format!("a {head_dim}-element head has no rotation pair"),
            ));
        }
        let covered: u32 = widths.iter().sum();
        if covered != head_dim / 2 {
            return Err(invalid(
                "Mrope",
                format!(
                    "the M-RoPE sections {widths:?} cover {covered} frequencies of the {} a \
                     {head_dim}-element head rotates",
                    head_dim / 2
                ),
            ));
        }
        Ok(Self { widths, head_dim })
    }

    /// The section widths, in frequency order.
    pub const fn widths(self) -> [u32; 3] {
        self.widths
    }

    /// The elements of one head the sections rotate: every element, for H3's text encoder.
    pub const fn rotated(self) -> u32 {
        self.head_dim
    }

    /// The position axis of every compact frequency, in frequency order.
    ///
    /// Interleaved M-RoPE cycles the axes: frequency `j` reads axis `j % 3` while
    /// `j < widths[axis] * 3`, and axis 0 otherwise. With H3's `[24, 20, 20]` the bound is 72 for
    /// all 64 frequencies, so every frequency is interleaved.
    ///
    /// The result is the per-column table the factor computation reads: `rope_index` gives one
    /// coordinate per axis per token and this table names the axis each column takes.
    pub fn frequency_axes(self) -> Vec<u32> {
        let count = 3;
        (0..self.head_dim / 2)
            .map(|frequency| {
                let axis = frequency % count;
                if frequency < self.widths[axis as usize] * count {
                    axis
                } else {
                    0
                }
            })
            .collect()
    }

    /// The FP32 factors of `positions`, written into `cosine` and `sine`.
    ///
    /// `positions` is one `[temporal, height, width]` coordinate per token; `frequencies` is the
    /// checkpoint's `inv_freq` table, one angle per compact frequency (`inverse_frequencies` of
    /// `head_dim`). Both tables are `[tokens, head_dim / 2]` row-major.
    ///
    /// This is the host evaluation the numeric diagnosis compares against; a served forward
    /// computes the same factors on the device from the uploaded positions
    /// (`ops::rope::RotaryFactors` with the interleaved section rule).
    pub fn factors(
        self,
        positions: &[[i64; 3]],
        frequencies: &[f32],
        cosine: &mut [f32],
        sine: &mut [f32],
    ) -> Result<()> {
        let columns = (self.head_dim / 2) as usize;
        check_factors("Mrope::factors", positions.len(), columns, frequencies, cosine, sine)?;
        let axes = self.frequency_axes();
        for (token, position) in positions.iter().enumerate() {
            for frequency in 0..columns {
                let angle = position[axes[frequency] as usize] as f32 * frequencies[frequency];
                cosine[token * columns + frequency] = angle.cos();
                sine[token * columns + frequency] = angle.sin();
            }
        }
        Ok(())
    }
}

/// The vision tower's 2-D rotary embedding: half of each head by the patch row, half by its column.
///
/// The reference builds one `RotaryEmbedding` of `head_dim / 2` **without sections** and calls it
/// with two coordinates per patch (`uniserve/models/qwen3_vl/vision.py`), so the embedding
/// evaluates the whole `head_dim / 4`-frequency spectrum at each coordinate:
/// `cos[p, a · (head_dim / 4) + f] = cos(coordinate(a) · inv_freq(f))`
/// (`uniserve/nn/rope.py::RotaryEmbedding._factors` over a `[patches, 2]` position). Each axis
/// therefore repeats the same frequency band rather than owning a distinct one, which is why a
/// sectioned `rotary_factors` call cannot produce these tables: its `sections` give every axis its
/// own band (the contiguous cut) or interleave one band per axis (`_interleaved_axes`), and neither
/// is this arrangement. The tables are therefore computed here, on the host, in the reference's
/// FP32 formulas, and reach the device as the encoder layout's `cosine`/`sine`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VisionRotary {
    head_dim: u32,
}

impl VisionRotary {
    /// The 2-D rotation of a `head_dim`-element head.
    pub fn new(head_dim: u32) -> Result<Self> {
        if !head_dim.is_multiple_of(4) {
            return Err(invalid(
                "VisionRotary",
                format!("a {head_dim}-element head does not split into two even rotary halves"),
            ));
        }
        Ok(Self { head_dim })
    }

    /// The elements of one head the rotation covers: every element.
    pub const fn rotated(self) -> u32 {
        self.head_dim
    }

    /// The compact frequencies one axis repeats: `head_dim / 4`.
    pub const fn per_axis(self) -> u32 {
        self.head_dim / 4
    }

    /// The FP32 factors of `coordinates`, written into `cosine` and `sine`.
    ///
    /// `coordinates` holds each patch's `(row, column)` in merge-block order, `frequencies` one
    /// angle per compact frequency (`inverse_frequencies` of `head_dim / 2`, 18 for a 72-element
    /// head), and both tables are `[patches, head_dim / 2]`.
    pub fn factors(
        self,
        coordinates: &[[i64; 2]],
        frequencies: &[f32],
        cosine: &mut [f32],
        sine: &mut [f32],
    ) -> Result<()> {
        let columns = (self.head_dim / 2) as usize;
        let per_axis = columns / 2;
        if frequencies.len() != per_axis {
            return Err(invalid(
                "VisionRotary::factors",
                format!(
                    "{} frequencies for the {per_axis} each rotary half of a {}-element head",
                    frequencies.len(),
                    self.head_dim
                ),
            ));
        }
        check_len("VisionRotary::factors", coordinates.len(), columns, cosine, sine)?;
        for (patch, coordinate) in coordinates.iter().enumerate() {
            for frequency in 0..columns {
                let axis = if frequency < per_axis { 0 } else { 1 };
                let angle = coordinate[axis] as f32 * frequencies[frequency % per_axis];
                cosine[patch * columns + frequency] = angle.cos();
                sine[patch * columns + frequency] = angle.sin();
            }
        }
        Ok(())
    }
}

/// The checkpoint's rotary frequencies: `theta^(-2j/dim)` for `j` in `0..dim/2`, in FP32.
///
/// The reference's `torch.arange(0, dim, 2, dtype=float32) / dim` and `theta ** ...`; the same
/// values keyed into retained storage as a device weight (`Rotary::frequencies`).
pub fn inverse_frequencies(dim: u32, theta: f32) -> Result<Vec<f32>> {
    if !dim.is_multiple_of(2) {
        return Err(invalid(
            "inverse_frequencies",
            format!("a rotary width of {dim} elements has no pair"),
        ));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(invalid(
            "inverse_frequencies",
            format!("the rotary base {theta} is not positive"),
        ));
    }
    Ok((0..dim / 2)
        .map(|index| 1.0 / theta.powf((2 * index) as f32 / dim as f32))
        .collect())
}

/// The `[temporal, height, width]` M-RoPE coordinates of one prompt's tokens.
///
/// Every maximal run of [`Mrope`] placeholders is one vision block and consumes the next grid of
/// its kind: `image_grids` for a run of `image_token_id`, `video_grids` for a run of
/// `video_token_id` (`uniserve/models/qwen3_vl/positions.py::rope_index`). A video grid
/// `(time, height, width)` covers `time` runs of one time step each, because Qwen3-VL presents
/// every temporal patch of a video behind its own text timestamp.
///
/// The reference is `uniserve/models/qwen3_vl/positions.py:21-149` (`rope_index`).
///
/// Text advances all three axes together from the running origin. A block of grid `(t, h, w)` has
/// `t * (h/merge) * (w/merge)` merged tokens; the token at merged coordinates `(i, j, k)` takes
/// `origin + (i, j, k)` in raster order, and the origin then advances by `max(h, w) / merge`.
pub fn rope_index(
    ids: &[u32],
    image_grids: &[[u32; 3]],
    video_grids: &[[u32; 3]],
    image_token_id: u32,
    video_token_id: u32,
    merge: u32,
) -> Result<Vec<[i64; 3]>> {
    if merge == 0 {
        return Err(invalid("rope_index", "the merge side must be positive"));
    }
    // A video contributes one single-step grid per temporal patch.
    let mut images = image_grids.iter().copied();
    let mut videos = video_grids
        .iter()
        .flat_map(|grid| std::iter::repeat_n([1, grid[1], grid[2]], grid[0] as usize));
    let mut positions = Vec::with_capacity(ids.len());
    let mut origin: i64 = 0;
    let mut index = 0;
    while index < ids.len() {
        let id = ids[index];
        let kind = if id == image_token_id {
            1
        } else if id == video_token_id {
            2
        } else {
            0
        };
        let mut stop = index + 1;
        while stop < ids.len() {
            let next = ids[stop];
            let next_kind = if next == image_token_id {
                1
            } else if next == video_token_id {
                2
            } else {
                0
            };
            if next_kind != kind {
                break;
            }
            stop += 1;
        }
        let run = (stop - index) as u32;
        if kind == 0 {
            for offset in 0..run {
                let position = origin + i64::from(offset);
                positions.push([position, position, position]);
            }
            origin += i64::from(run);
        } else {
            let grid = if kind == 1 {
                images.next()
            } else {
                videos.next()
            };
            let grid = grid.ok_or_else(|| {
                invalid(
                    "rope_index",
                    format!(
                        "the placeholder run at token {index} has no {}grid",
                        if kind == 1 { "image " } else { "video " }
                    ),
                )
            })?;
            let [time, height, width] = grid;
            if height % merge != 0 || width % merge != 0 {
                return Err(invalid(
                    "rope_index",
                    format!("the vision grid {grid:?} is not whole {merge}-patch blocks"),
                ));
            }
            let rows = height / merge;
            let columns = width / merge;
            if run != time * rows * columns {
                return Err(invalid(
                    "rope_index",
                    format!(
                        "the placeholder run at token {index} holds {run} tokens, the grid \
                         {grid:?} has {} merged tokens",
                        time * rows * columns
                    ),
                ));
            }
            for step in 0..time {
                for row in 0..rows {
                    for column in 0..columns {
                        let step = i64::from(step);
                        positions.push([
                            origin + step,
                            origin + i64::from(row),
                            origin + i64::from(column),
                        ]);
                    }
                }
            }
            origin += i64::from(height.max(width) / merge);
        }
        index = stop;
    }
    if images.next().is_some() || videos.next().is_some() {
        return Err(invalid(
            "rope_index",
            "a vision grid has no placeholder run".to_owned(),
        ));
    }
    Ok(positions)
}

/// The row of the vision feature table each prompt placeholder takes, `-1` for a text token.
///
/// The vision tower emits its merged tokens in grid order, one block after another, and the
/// prompt's placeholder run for a block holds exactly its merged tokens in that order
/// (`uniserve/models/qwen3_vl/encoder.py`): the `k`-th placeholder of the prompt takes row `k` of
/// the concatenated feature table.
pub fn placeholder_rows(ids: &[u32], image_token_id: u32, video_token_id: u32) -> Vec<i32> {
    let mut row: i32 = -1;
    ids.iter()
        .map(|id| {
            if *id == image_token_id || *id == video_token_id {
                row += 1;
                row
            } else {
                -1
            }
        })
        .collect()
}

/// The prompt position of every merged vision row, in prompt order.
///
/// The inverse of [`placeholder_rows`], and the table the row operator reads
/// (`ops::rows`: `destination[indices[j]] = source[j]`, `indices` `[count] i32`, `count` the number
/// of source rows). Entry `k` is the prompt row the `k`-th merged vision row takes, so the prompt's
/// placeholders are visited in the order the tower emits its tokens and a text row never appears.
pub fn vision_destinations(ids: &[u32], image_token_id: u32, video_token_id: u32) -> Vec<i32> {
    ids.iter()
        .enumerate()
        .filter(|(_, id)| **id == image_token_id || **id == video_token_id)
        .map(|(row, _)| row as i32)
        .collect()
}

/// The row and column of every patch of a grid, in merge-block order.
///
/// A merging encoder serializes a `height x width` patch grid block by block: the `merge x merge`
/// patches of each block, blocks in raster order, and the patches of a block in raster order, so a
/// merged token reads `merge**2` consecutive patch rows
/// (`uniserve/nn/vision/position.py::merged_grid_coordinates`).
pub fn merged_coordinates(height: u32, width: u32, merge: u32) -> Result<(Vec<u32>, Vec<u32>)> {
    if merge == 0 || height == 0 || width == 0 || height % merge != 0 || width % merge != 0 {
        return Err(invalid(
            "merged_coordinates",
            format!("a {height}x{width} patch grid is not whole {merge}x{merge} blocks"),
        ));
    }
    let rows = (height / merge) as usize;
    let columns = (width / merge) as usize;
    let side = merge as usize;
    let mut row = Vec::with_capacity(rows * columns * side * side);
    let mut column = Vec::new();
    column.reserve(rows * columns * side * side);
    for block_row in 0..rows {
        for block_column in 0..columns {
            for inner_row in 0..side {
                for inner_column in 0..side {
                    row.push((block_row * side + inner_row) as u32);
                    column.push((block_column * side + inner_column) as u32);
                }
            }
        }
    }
    Ok((row, column))
}

/// The rotary coordinates of a merged row's patches: `(row, column)` of each patch, merge order.
///
/// The reference rotates each patch by its own position and the merged rows keep the patches of
/// one token adjacent, so the coordinates are [`merged_coordinates`] as integer pairs.
pub fn rotary_coordinates(height: u32, width: u32, merge: u32) -> Result<Vec<[i64; 2]>> {
    let (rows, columns) = merged_coordinates(height, width, merge)?;
    Ok(rows
        .into_iter()
        .zip(columns)
        .map(|(row, column)| [i64::from(row), i64::from(column)])
        .collect())
}

/// The learned position table's bilinear sample of every merged row of a grid.
///
/// The table is a `side x side` grid of learned rows; a `height x width` patch grid samples it with
/// its corners aligned: patch row `i` reads table row `i * (side - 1) / (height - 1)` (an FP32
/// `linspace`), and likewise for columns. Each patch blends its four neighbouring table rows with
/// FP32 bilinear weights (`uniserve/nn/vision/position.py:167-200`,
/// `PositionEmbedding.interpolate`).
///
/// Returns, per patch row (in merge-block order), the four table row indices of the top-left,
/// top-right, bottom-left and bottom-right neighbours and their weights. A device kernel blends the
/// table's rows with these (`ops::position::Interpolate`); the tables are one image's, so they are
/// per image block, and a temporal grid repeats them once per time step - every time step samples
/// the same `height x width` grid (`PositionEmbedding.interpolate(...).repeat(time, 1)`).
pub fn position_samples(
    side: u32,
    height: u32,
    width: u32,
    merge: u32,
) -> Result<(Vec<[i32; 4]>, Vec<[f32; 4]>)> {
    if side == 0 {
        return Err(invalid(
            "position_samples",
            "the position table needs at least one row".to_owned(),
        ));
    }
    let (rows, columns) = merged_coordinates(height, width, merge)?;
    let vertical = grid_samples(side, height)?;
    let horizontal = grid_samples(side, width)?;
    let side = side as i32;
    let mut indices = Vec::with_capacity(rows.len());
    let mut weights = Vec::with_capacity(rows.len());
    for (row, column) in rows.into_iter().zip(columns) {
        let (top, bottom, down) = vertical[row as usize];
        let (left, right, across) = horizontal[column as usize];
        indices.push([
            top * side + left,
            top * side + right,
            bottom * side + left,
            bottom * side + right,
        ]);
        weights.push([
            (1.0 - down) * (1.0 - across),
            (1.0 - down) * across,
            down * (1.0 - across),
            down * across,
        ]);
    }
    Ok((indices, weights))
}

/// One axis's bilinear sample: `(lower, upper, fraction)` of every coordinate of `size`.
///
/// The reference's `torch.linspace(0, extent - 1, size)` computed in FP32, truncated to the lower
/// table row, with the upper neighbour clamped at the last row.
fn grid_samples(extent: u32, size: u32) -> Result<Vec<(i32, i32, f32)>> {
    if size == 0 || extent == 0 {
        return Err(invalid(
            "position_samples",
            format!("a {extent}-row table cannot sample a {size}-patch axis"),
        ));
    }
    // `torch.linspace(start, end, 1)` is `[start]`; otherwise the step is an FP32 division of the
    // span, as `at::linspace` computes it.
    let step = if size == 1 {
        0.0
    } else {
        (extent - 1) as f32 / (size - 1) as f32
    };
    let last = extent as i32 - 1;
    Ok((0..size)
        .map(|index| {
            let coordinate = index as f32 * step;
            let lower = (coordinate as i32).min(last);
            let upper = (lower + 1).min(last);
            (lower, upper, coordinate - lower as f32)
        })
        .collect())
}

/// The number of merged rows one grid's patches form: `time * h * w / merge**2`.
pub fn merged_rows(time: u32, height: u32, width: u32, merge: u32) -> Result<u32> {
    let patches = (time as u64)
        .checked_mul(height as u64)
        .and_then(|value| value.checked_mul(width as u64))
        .ok_or_else(|| invalid("merged_rows", "the patch count overflows".to_owned()))?;
    let block = (merge as u64) * (merge as u64);
    if merge == 0 || patches % block != 0 {
        return Err(invalid(
            "merged_rows",
            format!("{patches} patches of a {time}x{height}x{width} grid are not whole {merge}x{merge} blocks"),
        ));
    }
    u32::try_from(patches / block)
        .map_err(|_| invalid("merged_rows", "the merged row count overflows".to_owned()))
}

/// The attention lengths of one image's patch sequences: `time + 1` packed offsets.
///
/// The vision tower never attends across a time step: each temporal patch of a grid is one
/// bidirectional sequence of `height * width` patches
/// (`uniserve/models/qwen3_vl/vision.py::VisionTower`), so the offsets are
/// `[0, hw, 2 hw, …, time hw]`. They are the `cu_seq_lens_q` and `cu_seq_lens_k` of the tower's
/// dense attention, and the rows are the patches in grid order.
pub fn vision_lengths(time: u32, height: u32, width: u32) -> Result<Vec<i32>> {
    let patches = i64::from(height) * i64::from(width);
    let mut offsets = Vec::with_capacity(time as usize + 1);
    for step in 0..=time {
        let end = i64::from(step) * patches;
        let end = i32::try_from(end).map_err(|_| {
            invalid(
                "vision_lengths",
                format!("a {time}x{height}x{width} grid packs more patches than an i32 holds"),
            )
        })?;
        offsets.push(end);
    }
    Ok(offsets)
}

/// The error a table's inputs raise: a shape disagreement of `owner`.
pub(crate) fn invalid(owner: &str, problem: impl Into<String>) -> Error {
    Error::Nn(nn::Error::from(ops::Error::Shape(format!(
        "{owner}: {}",
        problem.into()
    ))))
}

/// Check two factor tables against `rows` tokens and `columns` frequencies.
fn check_factors(
    owner: &'static str,
    tokens: usize,
    columns: usize,
    frequencies: &[f32],
    cosine: &[f32],
    sine: &[f32],
) -> Result<()> {
    if frequencies.len() != columns {
        return Err(invalid(
            owner,
            format!("{} frequencies for {columns} columns", frequencies.len()),
        ));
    }
    check_len(owner, tokens, columns, cosine, sine)
}

/// Check that a factor table holds `rows * columns` values.
fn check_len(
    owner: &'static str,
    rows: usize,
    columns: usize,
    cosine: &[f32],
    sine: &[f32],
) -> Result<()> {
    let expected = rows * columns;
    if cosine.len() != expected || sine.len() != expected {
        return Err(invalid(
            owner,
            format!(
                "{} and {} factor values for {rows} rows of {columns} columns",
                cosine.len(),
                sine.len()
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: u32 = 128;
    const THETA: f32 = 5_000_000.0;
    const IMAGE_TOKEN: u32 = 151_655;
    const VIDEO_TOKEN: u32 = 151_656;

    fn mrope() -> Mrope {
        Mrope::new([24, 20, 20], HEAD_DIM).expect("H3's sections")
    }

    #[test]
    fn frequencies_are_the_checkpoints_rotary_base() {
        let frequencies = inverse_frequencies(HEAD_DIM, THETA).expect("frequencies");
        assert_eq!(frequencies.len(), 64);
        // `theta ** 0` is one, and the table decreases geometrically.
        assert_eq!(frequencies[0], 1.0);
        for pair in frequencies.windows(2) {
            assert!(pair[0] > pair[1]);
        }
        assert!(inverse_frequencies(HEAD_DIM + 1, THETA).is_err());
        assert!(inverse_frequencies(HEAD_DIM, 0.0).is_err());
    }

    #[test]
    fn interleaved_sections_cycle_their_axes() {
        let sections = mrope();
        let axes = sections.frequency_axes();
        assert_eq!(axes.len(), 64);
        // Every frequency is below `24 * 3`, so the cycle decides all of them.
        for (frequency, axis) in axes.iter().enumerate() {
            assert_eq!(*axis as usize, frequency % 3);
        }
        // A narrower first section pushes the frequencies it cannot hold onto axis 0.
        let narrow = Mrope::new([16, 24, 24], HEAD_DIM).expect("sections");
        let axes = narrow.frequency_axes();
        assert_eq!(axes[48], 1);
        assert_eq!(axes[49], 2);
        assert_eq!(axes[50], 0);
        assert_eq!(axes[3 * 16], 0);
    }

    #[test]
    fn sections_must_split_half_the_head() {
        assert!(Mrope::new([24, 20, 20], HEAD_DIM).is_ok());
        assert!(Mrope::new([24, 20, 19], HEAD_DIM).is_err());
        assert!(Mrope::new([0, 32, 32], HEAD_DIM).is_err());
        assert!(Mrope::new([24, 20, 20], HEAD_DIM + 1).is_err());
    }

    #[test]
    fn a_text_prompt_takes_consecutive_coordinates_on_every_axis() {
        let ids = [10, 11, 12];
        let positions = rope_index(&ids, &[], &[], IMAGE_TOKEN, VIDEO_TOKEN, 2).expect("positions");
        assert_eq!(positions, vec![[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
    }

    #[test]
    fn an_image_block_advances_by_its_wider_side() {
        // Two text tokens, a 4x6 patch grid merged 2x2 (6 merged tokens), one text token.
        let ids = [10, 11, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN,
            IMAGE_TOKEN, 12];
        let positions = rope_index(&ids, &[[1, 4, 6]], &[], IMAGE_TOKEN, VIDEO_TOKEN, 2)
            .expect("positions");
        assert_eq!(positions.len(), 9);
        // The block starts where the text clock is (2) and is stored in merge-block order.
        assert_eq!(positions[2], [2, 2, 2]);
        assert_eq!(positions[3], [2, 2, 4]);
        assert_eq!(positions[4], [2, 4, 2]);
        // The clock advances by `max(4, 6) / 2 = 3`.
        assert_eq!(positions[8], [5, 5, 5]);
    }

    #[test]
    fn a_video_grid_is_one_run_per_temporal_patch() {
        let ids = [VIDEO_TOKEN, VIDEO_TOKEN, 7, VIDEO_TOKEN, VIDEO_TOKEN];
        // A 2x2x4 grid (two temporal patches, 4 wide) and a 1x2x2 grid, merge 2.
        let positions = rope_index(
            &ids,
            &[],
            &[[2, 2, 4], [1, 2, 2]],
            IMAGE_TOKEN,
            VIDEO_TOKEN,
            2,
        )
        .expect("positions");
        // The first run is two temporal patches of a 1x2 patch row: 2 merged tokens each.
        assert_eq!(positions[0], [0, 0, 0]);
        assert_eq!(positions[1], [0, 0, 1]);
        // The text token continues the clock, which the first grid advanced by 4/2 = 2.
        assert_eq!(positions[2], [2, 2, 2]);
        // The second video advances by 2/2 = 1 after its own merged token.
        assert_eq!(positions[3], [3, 3, 3]);
        assert_eq!(positions[4], [3, 3, 4]);
    }

    #[test]
    fn a_grid_without_a_run_or_a_run_without_a_grid_is_refused() {
        let ids = [IMAGE_TOKEN, 3];
        assert!(rope_index(&ids, &[], &[], IMAGE_TOKEN, VIDEO_TOKEN, 2).is_err());
        let ids = [3];
        assert!(rope_index(&ids, &[[1, 2, 2]], &[], IMAGE_TOKEN, VIDEO_TOKEN, 2).is_err());
        let ids = [IMAGE_TOKEN];
        assert!(rope_index(&ids, &[[1, 2, 2]], &[], IMAGE_TOKEN, VIDEO_TOKEN, 2).is_err());
    }

    #[test]
    fn the_vision_destinations_are_the_placeholders_in_prompt_order() {
        let ids = [7, IMAGE_TOKEN, 8, VIDEO_TOKEN, VIDEO_TOKEN, 9, IMAGE_TOKEN];
        assert_eq!(
            vision_destinations(&ids, IMAGE_TOKEN, VIDEO_TOKEN),
            vec![1, 3, 4, 6]
        );
        // The two directions agree: the k-th placeholder takes row k, so inverting the destination
        // list through the placeholder map returns the rows in order.
        let rows = placeholder_rows(&ids, IMAGE_TOKEN, VIDEO_TOKEN);
        let destinations = vision_destinations(&ids, IMAGE_TOKEN, VIDEO_TOKEN);
        let mut next = 0;
        for (token, row) in rows.iter().enumerate() {
            if *row >= 0 {
                assert_eq!(destinations[*row as usize], token as i32);
                next += 1;
            }
        }
        assert_eq!(next, destinations.len());
        assert!(vision_destinations(&[1, 2, 3], IMAGE_TOKEN, VIDEO_TOKEN).is_empty());
    }

    #[test]
    fn placeholders_number_the_vision_rows() {
        let ids = [5, IMAGE_TOKEN, 6, VIDEO_TOKEN, VIDEO_TOKEN, 7];
        assert_eq!(
            placeholder_rows(&ids, IMAGE_TOKEN, VIDEO_TOKEN),
            vec![-1, 0, -1, 1, 2, -1]
        );
    }

    #[test]
    fn merged_order_is_block_then_patch() {
        let (rows, columns) = merged_coordinates(4, 6, 2).expect("coordinates");
        assert_eq!(rows.len(), 6);
        assert_eq!(columns, vec![0, 1, 4, 5, 0, 1, 4, 5, 0, 1, 4, 5]);
        assert_eq!(rows, vec![0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1]);
        assert!(merged_coordinates(3, 4, 2).is_err());
    }

    #[test]
    fn the_position_sample_blends_the_four_neighbours() {
        let (indices, weights) = position_samples(48, 4, 6, 2).expect("samples");
        // Six merged rows per grid; the first merged row is patch (0, 0), the table's corner.
        assert_eq!(indices.len(), 6);
        assert_eq!(indices[0], [0, 0, 48, 48]);
        assert_eq!(weights[0], [1.0, 0.0, 0.0, 0.0]);
        // The patch at column 2 samples table column `2 * 47 / 5 = 18.8`.
        let (indices, weights) = position_samples(48, 4, 6, 2).expect("samples");
        assert_eq!(indices[1][0], 18);
        assert_eq!(indices[1][1], 19);
        let fraction = 2.0f32 * (47.0 / 5.0) - 18.0;
        assert!((weights[1][1] - fraction).abs() < 1e-6);
        // A one-patch axis samples the table's first row.
        let (indices, _) = position_samples(48, 1, 1, 2).unwrap();
        assert_eq!(indices, vec![[0, 0, 48, 48]]);
        assert!(position_samples(0, 2, 2, 1).is_err());
    }

    #[test]
    fn vision_rotary_pairs_rows_then_columns() {
        let rotary = VisionRotary::new(72).expect("head width");
        assert_eq!(rotary.per_axis(), 18);
        let frequencies = inverse_frequencies(36, 10_000.0).expect("frequencies");
        // One patch at (0, 0) and one at (0, 2).
        let coordinates = [[0, 0], [0, 2]];
        let mut cosine = vec![0.0; 72];
        let mut sine = vec![0.0; 72];
        rotary
            .factors(&coordinates, &frequencies, &mut cosine, &mut sine)
            .expect("factors");
        // At the corner every factor is one and zero; the column coordinate only moves the second
        // half of the compact table.
        assert_eq!(cosine[0], 1.0);
        assert_eq!(sine[0], 0.0);
        assert_eq!(cosine[18], 1.0);
        let expected = (2.0 * frequencies[0]).cos();
        assert!((cosine[36 + 18] - expected).abs() < 1e-7);
    }

    #[test]
    fn mrope_factors_read_their_axis_coordinates() {
        let sections = mrope();
        let frequencies = inverse_frequencies(HEAD_DIM, THETA).expect("frequencies");
        let positions = [[0, 0, 0], [1, 2, 3]];
        let mut cosine = vec![0.0; 128];
        let mut sine = vec![0.0; 128];
        sections
            .factors(&positions, &frequencies, &mut cosine, &mut sine)
            .expect("factors");
        // Frequency 0 reads the temporal axis, frequency 1 the height, frequency 2 the width.
        let second = 64;
        assert!((cosine[second] - 1.0f32.cos()).abs() < 1e-7);
        assert!((cosine[second + 1] - (2.0f32 * frequencies[1]).cos()).abs() < 1e-7);
        assert!((sine[second + 2] - (3.0f32 * frequencies[2]).sin()).abs() < 1e-7);
        assert!(sections
            .factors(&positions, &frequencies[..63], &mut cosine, &mut sine)
            .is_err());
    }

    #[test]
    fn one_sequence_per_temporal_patch() {
        assert_eq!(vision_lengths(2, 4, 6).unwrap(), vec![0, 24, 48]);
        assert_eq!(merged_rows(2, 4, 6, 2).unwrap(), 12);
        assert!(merged_rows(1, 3, 6, 2).is_err());
    }
}
