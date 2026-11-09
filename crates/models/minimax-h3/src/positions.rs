//! The host-side position tables of the text encoder and its vision tower.
//!
//! Every table here is a function of the *request*, not of a step: the prompt's token ids and the
//! grids of its condition media decide them once, before the encode is captured or replayed. The
//! worker's encoder runner calls these functions when it prepares a request and uploads the results
//! into the views [`crate::text_encoder`] reads; the model crate itself owns no device storage and
//! copies nothing (`AGENTS.md`, Model Library Boundary).
//!
//! Four kinds of table are derived:
//!
//! - the language layers' interleaved M-RoPE coordinates ([`rope_index`]), whose factors the device
//!   computes from the uploaded positions under the text encoder's sectioned spectrum;
//! - the vision tower's 2-D rotary coordinates of a patch grid in merge-block order
//!   ([`rotary_coordinates`]) and their factors ([`VisionRotary::factors`]);
//! - the learned position table's bilinear sample of each merged patch ([`position_samples`]), the
//!   indices and FP32 weights of the four table rows a merged row blends;
//! - the vision tower's attention lengths ([`vision_lengths`]: one bidirectional sequence per
//!   temporal patch of every grid) and the prompt row each merged vision row takes
//!   ([`vision_destinations`]).
//!
//! The arithmetic is Qwen3-VL's, in its order and precision: factors in FP32 over the FP32 spectrum,
//! positions as integers, and the position table's sample coordinates on an FP32 `linspace` whose
//! step is computed in FP32 (`torch.linspace` over a float32 scalar type).

use crate::{Error, Result};

/// The table rows one bilinear sample reads: four, the corners of one cell.
pub const POSITION_TAPS: u64 = 4;

/// One merged row's bilinear sample: the four learned-table rows it blends, and their weights.
pub type PositionSamples = (Vec<[i32; 4]>, Vec<[f32; 4]>);

/// The vision tower's 2-D rotary embedding: half of each head by the patch row, half by its column.
///
/// The reference builds one `RotaryEmbedding` of `head_dim / 2` **without sections** and calls it
/// with two coordinates per patch, so the embedding
/// evaluates the whole `head_dim / 4`-frequency spectrum at each coordinate:
/// `cos[p, a · (head_dim / 4) + f] = cos(coordinate(a) · inv_freq(f))`
/// over a `[patches, 2]` position. Each axis
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

    /// The compact frequencies one axis repeats: `head_dim / 4`.
    pub const fn per_axis(self) -> u32 {
        self.head_dim / 4
    }

    /// The FP32 factors of `coordinates`, written into `cosine` and `sine`.
    ///
    /// `coordinates` holds each patch's `(row, column)` in merge-block order, `frequencies` the
    /// tower's repeated spectrum (`nn::layers::Spectrum::Repeated`, one block of `head_dim / 4`
    /// frequencies per axis, `head_dim / 2` in all), and both tables are `[patches, head_dim / 2]`.
    pub fn factors(
        self,
        coordinates: &[[i64; 2]],
        frequencies: &[f32],
        cosine: &mut [f32],
        sine: &mut [f32],
    ) -> Result<()> {
        let columns = (self.head_dim / 2) as usize;
        let per_axis = columns / 2;
        if frequencies.len() != columns {
            return Err(invalid(
                "VisionRotary::factors",
                format!(
                    "{} frequencies for the {columns} rotated pairs of a {}-element head",
                    frequencies.len(),
                    self.head_dim
                ),
            ));
        }
        check_len(
            "VisionRotary::factors",
            coordinates.len(),
            columns,
            cosine,
            sine,
        )?;
        for (patch, coordinate) in coordinates.iter().enumerate() {
            for frequency in 0..columns {
                let axis = if frequency < per_axis { 0 } else { 1 };
                let angle = coordinate[axis] as f32 * frequencies[frequency];
                cosine[patch * columns + frequency] = angle.cos();
                sine[patch * columns + frequency] = angle.sin();
            }
        }
        Ok(())
    }
}

/// The `[temporal, height, width]` M-RoPE coordinates of one prompt's tokens.
///
/// Every maximal run of vision placeholders is one vision block and consumes the next grid of
/// its kind: `image_grids` for a run of `image_token_id`, `video_grids` for a run of
/// `video_token_id`. A video grid
/// `(time, height, width)` covers `time` runs of one time step each, because Qwen3-VL presents
/// every temporal patch of a video behind its own text timestamp.
///
/// This is Qwen3-VL's `rope_index`.
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

/// The prompt position of every merged vision row, in prompt order.
///
/// The table the row operator reads
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
/// merged token reads `merge**2` consecutive patch rows.
pub fn merged_coordinates(height: u32, width: u32, merge: u32) -> Result<(Vec<u32>, Vec<u32>)> {
    if merge == 0
        || height == 0
        || width == 0
        || !height.is_multiple_of(merge)
        || !width.is_multiple_of(merge)
    {
        return Err(invalid(
            "merged_coordinates",
            format!("a {height}x{width} patch grid is not whole {merge}x{merge} blocks"),
        ));
    }
    let rows = (height / merge) as usize;
    let columns = (width / merge) as usize;
    let side = merge as usize;
    let mut row = Vec::with_capacity(rows * columns * side * side);
    let mut column = Vec::with_capacity(rows * columns * side * side);
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
/// FP32 bilinear weights, as Qwen3-VL's learned position embedding interpolates its table.
///
/// Returns, per patch row (in merge-block order), the four table row indices of the top-left,
/// top-right, bottom-left and bottom-right neighbours and their weights. A device kernel blends the
/// table's rows with these (`ops::position::Interpolate`); the tables are one image's, so they are
/// per image block, and a temporal grid repeats them once per time step - every time step samples
/// the same `height x width` grid (`PositionEmbedding.interpolate(...).repeat(time, 1)`).
pub fn position_samples(side: u32, height: u32, width: u32, merge: u32) -> Result<PositionSamples> {
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
    // span, as `at::linspace` computes it, and the last point is `end` exactly rather than
    // `start + (size - 1) * step`: the two differ by an ulp whenever the step is not
    // representable, and `47 / 5` is not, so `48 x 6` would otherwise truncate its last column to
    // 46 and sample the wrong two table rows.
    let step = if size == 1 {
        0.0
    } else {
        (extent - 1) as f32 / (size - 1) as f32
    };
    let last = extent as i32 - 1;
    Ok((0..size)
        .map(|index| {
            let coordinate = if index + 1 == size {
                last as f32
            } else {
                index as f32 * step
            };
            let lower = (coordinate as i32).min(last);
            let upper = (lower + 1).min(last);
            (lower, upper, coordinate - lower as f32)
        })
        .collect())
}

/// The attention lengths of one image's patch sequences: `time + 1` packed offsets.
///
/// The vision tower never attends across a time step: each temporal patch of a grid is one
/// bidirectional sequence of `height * width` patches,
/// so the offsets are
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

    const IMAGE_TOKEN: u32 = 151_655;
    const VIDEO_TOKEN: u32 = 151_656;

    #[test]
    fn interleaved_sections_cycle_their_axes() {
        // The device's rule takes the section widths in rotated elements, two per frequency.
        let widths: [u32; 3] = [24, 20, 20];
        let axes = ops::rope::interleaved_axes(&widths.map(|width| width * 2));
        assert_eq!(axes.len(), 64);
        // Each axis owns exactly its section's width of frequencies, counted from the whole
        // spectrum: every axis takes its stated count and no axis a contiguous block.
        let mut owned = [0_usize; 3];
        for axis in &axes {
            owned[*axis as usize] += 1;
        }
        assert_eq!(
            owned,
            [widths[0] as usize, widths[1] as usize, widths[2] as usize]
        );
        // A frequency below its cycle axis's bound takes that axis; `[24, 20, 20]` has two
        // frequencies (61 and 62) whose own axis is full, and they fall to axis 0.
        for (frequency, axis) in axes.iter().enumerate() {
            let cycle = frequency % 3;
            if frequency < widths[cycle] as usize * 3 {
                assert_eq!(*axis as usize, cycle, "frequency {frequency}");
            } else {
                assert_eq!(*axis, 0, "frequency {frequency} overflows its axis");
            }
        }
        // A narrower first section pushes the frequencies it cannot hold onto axis 0: with
        // `[16, 24, 24]` the bound of axis 0 is 48, so 48 itself falls back while 49 and 50 do not.
        let narrow: [u32; 3] = [16, 24, 24];
        let axes = ops::rope::interleaved_axes(&narrow.map(|width| width * 2));
        assert_eq!(axes[48], 0);
        assert_eq!(axes[49], 1);
        assert_eq!(axes[50], 2);
        assert_eq!(axes[51], 0);
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
        let ids = [
            10,
            11,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            12,
        ];
        let positions =
            rope_index(&ids, &[[1, 4, 6]], &[], IMAGE_TOKEN, VIDEO_TOKEN, 2).expect("positions");
        assert_eq!(positions.len(), 9);
        // The block starts where the text clock is (2) and is stored in merge-block order.
        assert_eq!(positions[2], [2, 2, 2]);
        assert_eq!(positions[3], [2, 2, 3]);
        assert_eq!(positions[4], [2, 2, 4]);
        assert_eq!(positions[5], [2, 3, 2]);
        // The clock advances by `max(4, 6) / 2 = 3`.
        assert_eq!(positions[8], [5, 5, 5]);
    }

    #[test]
    fn a_video_grid_is_one_run_per_temporal_patch() {
        // One video of two temporal patches, each a 2x4 patch grid merged 2x2: two runs of two
        // placeholders, with a timestamp between them.
        let ids = [VIDEO_TOKEN, VIDEO_TOKEN, 7, VIDEO_TOKEN, VIDEO_TOKEN];
        let positions =
            rope_index(&ids, &[], &[[2, 2, 4]], IMAGE_TOKEN, VIDEO_TOKEN, 2).expect("positions");
        assert_eq!(positions.len(), 5);
        // Each run is one time step: its merged grid is 1 row by 2 columns, and the clock is the
        // same one every axis reads.
        assert_eq!(positions[0], [0, 0, 0]);
        assert_eq!(positions[1], [0, 0, 1]);
        // The run advances the clock by `max(2, 4) / 2 = 2`, the timestamp takes 2 and leaves 3.
        assert_eq!(positions[2], [2, 2, 2]);
        assert_eq!(positions[3], [3, 3, 3]);
        assert_eq!(positions[4], [3, 3, 4]);
    }

    #[test]
    fn a_grid_without_a_run_or_a_run_without_a_grid_is_refused() {
        let ids = [IMAGE_TOKEN, 3];
        assert!(rope_index(&ids, &[], &[], IMAGE_TOKEN, VIDEO_TOKEN, 2).is_err());
        let ids = [3];
        assert!(rope_index(&ids, &[[1, 2, 2]], &[], IMAGE_TOKEN, VIDEO_TOKEN, 2).is_err());
        // A grid whose merged rows the prompt does not have is refused: a 4x4 grid merges to
        // four rows, and one placeholder cannot hold them.
        let ids = [IMAGE_TOKEN];
        assert!(rope_index(&ids, &[[1, 4, 4]], &[], IMAGE_TOKEN, VIDEO_TOKEN, 2).is_err());
        // A 2x2 grid does merge to one row, so the same prompt with it is well formed.
        assert!(rope_index(&ids, &[[1, 2, 2]], &[], IMAGE_TOKEN, VIDEO_TOKEN, 2).is_ok());
    }

    #[test]
    fn the_vision_destinations_are_the_placeholders_in_prompt_order() {
        let ids = [7, IMAGE_TOKEN, 8, VIDEO_TOKEN, VIDEO_TOKEN, 9, IMAGE_TOKEN];
        assert_eq!(
            vision_destinations(&ids, IMAGE_TOKEN, VIDEO_TOKEN),
            vec![1, 3, 4, 6]
        );
        assert!(vision_destinations(&[1, 2, 3], IMAGE_TOKEN, VIDEO_TOKEN).is_empty());
    }

    #[test]
    fn merged_order_is_block_then_patch() {
        // A 4x6 patch grid merges to six rows, and each merged row reads its own 2x2 patches:
        // twenty-four patch coordinates, blocks in raster order and patches within a block in
        // raster order. This is the order the reference's `merged_grid_coordinates` returns.
        let (rows, columns) = merged_coordinates(4, 6, 2).expect("coordinates");
        assert_eq!(rows.len(), 24);
        assert_eq!(
            rows,
            vec![
                0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3
            ]
        );
        assert_eq!(
            columns,
            vec![
                0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5
            ]
        );
        assert!(merged_coordinates(3, 4, 2).is_err());
    }

    #[test]
    fn the_position_sample_blends_the_four_neighbours() {
        // One sample per patch: a 4x6 grid has twenty-four, the table is 48x48, and patch
        // `(i, j)` reads table point `(i * 47 / 3, j * 47 / 5)` with the corners aligned, as the
        // reference's `PositionEmbedding.interpolate` does. Each tap list is the top-left,
        // top-right, bottom-left and bottom-right neighbour of that point.
        let (indices, weights) = position_samples(48, 4, 6, 2).expect("samples");
        assert_eq!(indices.len(), 24);
        assert_eq!(weights.len(), 24);
        // The grid's first patch is the table's top-left corner: the top-left tap carries the
        // whole blend and its three neighbours carry none of it.
        assert_eq!(indices[0], [0, 1, 48, 49]);
        assert_eq!(weights[0], [1.0, 0.0, 0.0, 0.0]);
        // Its right neighbour sits at column `47 / 5 = 9.4`: table columns 9 and 10 of the same
        // row, split 0.6 / 0.4.
        assert_eq!(indices[1], [9, 10, 57, 58]);
        assert!((weights[1][0] - 0.6).abs() < 1e-6);
        assert!((weights[1][1] - 0.4).abs() < 1e-6);
        assert_eq!(weights[1][2], 0.0);
        assert_eq!(weights[1][3], 0.0);
        // Every blend partitions one over four taps inside the table, for every patch.
        for (index, weight) in indices.iter().zip(&weights) {
            for tap in index {
                assert!((0..2304).contains(tap), "tap {tap} outside the table");
            }
            let total: f32 = weight.iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "{weight:?} does not sum to one");
        }
        // A one-block grid reaches the table's four corners through its four patches.
        let (indices, _) = position_samples(48, 2, 2, 2).expect("samples");
        assert_eq!(
            indices,
            vec![
                [0, 1, 48, 49],
                [47, 47, 95, 95],
                [2256, 2257, 2256, 2257],
                [2303, 2303, 2303, 2303],
            ]
        );
    }

    #[test]
    fn vision_rotary_pairs_rows_then_columns() {
        let rotary = VisionRotary::new(72).expect("head width");
        assert_eq!(rotary.per_axis(), 18);
        // The tower's spectrum: 18 frequencies of a 36-channel embedding, once per axis.
        let frequencies = nn::layers::Spectrum::Repeated {
            theta: 10_000.0,
            pairs: 18,
            axes: 2,
            precision: nn::layers::Precision::F32,
        }
        .inverse_frequencies()
        .expect("frequencies");
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
    fn one_sequence_per_temporal_patch() {
        assert_eq!(vision_lengths(2, 4, 6).unwrap(), vec![0, 24, 48]);
    }
}
