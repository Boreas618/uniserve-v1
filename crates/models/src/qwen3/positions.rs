//! The position tables of the Qwen3-VL language model and its vision tower, as CPU tensors.
//!
//! Every table here is a function of a request's shape: the prompt's runs of text and vision
//! placeholders and the grids of its media. A forward computes them on the CPU device and moves
//! them to the device of the tensor they meet; their extents follow from the layout's counts
//! alone. Each function returns one tensor:
//!
//! | Function | Tensor | Reference |
//! | --- | --- | --- |
//! | [`rope_index`] | `[3, tokens]` I32 M-RoPE coordinates of a prompt | `Qwen3VLModel.get_rope_index` |
//! | [`vision_block`] | `[3, rows]` I32 coordinates of one vision block | `get_vision_position_ids` |
//! | [`vision_rows`] | `[rows]` U32 prompt row of each merged vision row | the placeholder mask |
//! | [`patch_coordinates`] | `[2, patches]` I32 rotary coordinates of a grid's patches | `rot_pos_ids` |
//! | [`position_taps`] | `[4, patches]` U32 learned-table rows a patch blends | `get_vision_bilinear_indices_and_weights` |
//! | [`position_weights`] | `[4, patches]` F32 bilinear weights of those rows | the same |
//! | [`frame_offsets`] | `[time + 1]` U32 offsets of a grid's time steps | `get_vision_cu_seqlens` |
//!
//! The references are transformers' (`transformers/models/qwen3_vl/modeling_qwen3_vl.py`,
//! `transformers/vision_utils.py`) and vLLM's (`refs/vllm/vllm/model_executor/models/
//! qwen3_vl.py`); the expressions follow them operation for operation: positions as integers,
//! and the learned table's sample coordinates on an FP32 `torch.linspace`.

use nn::{DType, Device, Tensor, VisionGrid};
use processing::qwen_vl::position_span;

use crate::qwen3::{Error, Result};

/// The run of a prompt a token belongs to: text.
const TEXT: u8 = 0;

/// A run of image placeholders: one vision block under its whole grid.
const IMAGE: u8 = 1;

/// A run of video placeholders: one time step of a video grid.
const VIDEO: u8 = 2;

/// `value` as an I32 coordinate.
fn coordinate(value: u32) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::Config(format!("a position of {value} is wider than an i32")))
}

/// Refuses a grid that is not whole `merge × merge` blocks of at least one time step.
pub(crate) fn check_grid(grid: [u32; 3], merge: u32) -> Result<()> {
    let [time, height, width] = grid;

    if merge == 0
        || time == 0
        || height == 0
        || width == 0
        || !height.is_multiple_of(merge)
        || !width.is_multiple_of(merge)
    {
        return Err(Error::Config(format!(
            "the vision grid {grid:?} is not whole {merge}x{merge} merge blocks"
        )));
    }
    Ok(())
}

/// The M-RoPE coordinates of one prompt's tokens: `[3, tokens]` I32, axis-major (time, height,
/// width).
///
/// Every maximal run of vision placeholders is one vision block and consumes `grids` in prompt
/// order: a run of `image_token_id` takes the next grid whole, and a grid `(time, height,
/// width)` of a video covers `time` runs of `video_token_id`, one time step each, because
/// Qwen3-VL presents every temporal patch of a video behind its own text timestamp.
///
/// Text advances all three axes together from the running origin. A block's merged token at
/// `(i, j, k)` takes `origin + (i, j, k)` in raster order ([`vision_block`]), and the origin
/// then advances by the larger of the block's merged sides
/// (`modeling_qwen3_vl.py:997-1010`).
///
/// # Errors
///
/// A placeholder run without a grid, a grid without a run, a run that does not hold its
/// grid's merged tokens, or a grid that is not whole merge blocks.
pub fn rope_index(
    ids: &[u32],
    grids: &[[u32; 3]],
    image_token_id: u32,
    video_token_id: u32,
    merge: u32,
) -> Result<Tensor> {
    let kind = |id: u32| match id {
        id if id == image_token_id => IMAGE,
        id if id == video_token_id => VIDEO,
        _ => TEXT,
    };
    let mut grids = grids.iter().copied();

    // The time steps of the video grid being presented that have no run yet, and its frame.
    let mut steps = 0;
    let mut frame = (0, 0);

    let mut origin = 0_u32;
    let mut token = 0;
    let mut segments = Vec::new();
    for run in ids.chunk_by(|left, right| kind(*left) == kind(*right)) {
        let length = run.len() as u32;

        if kind(run[0]) == TEXT {
            let positions = Tensor::arange(
                coordinate(origin)?,
                coordinate(origin + length)?,
                &Device::Cpu,
            )?;

            segments.push(positions.unsqueeze(0)?.repeat((3, 1))?);
            origin += length;
        } else {
            let missing =
                || Error::Config(format!("the placeholder run at token {token} has no grid"));
            let grid = if kind(run[0]) == IMAGE {
                grids.next().ok_or_else(missing)?
            } else {
                if steps == 0 {
                    let [time, height, width] = grids.next().ok_or_else(missing)?;
                    (steps, frame) = (time, (height, width));
                }
                steps = steps.saturating_sub(1);

                [1, frame.0, frame.1]
            };

            let block = vision_block(grid, merge, origin)?;
            if block.dim(1)? != run.len() {
                return Err(Error::Config(format!(
                    "the placeholder run at token {token} holds {length} tokens, and the grid \
                     {grid:?} merges to {}",
                    block.dim(1)?
                )));
            }
            segments.push(block);

            let [t, h, w] = grid;
            origin += position_span(VisionGrid { t, h, w }, merge);
        }
        token += run.len();
    }

    if grids.next().is_some() || steps != 0 {
        return Err(Error::Config(
            "a vision grid has no placeholder run".to_owned(),
        ));
    }
    if segments.is_empty() {
        return Ok(Tensor::zeros((3, 0), DType::I32, &Device::Cpu)?);
    }
    Ok(Tensor::cat(&segments, 1)?)
}

/// The M-RoPE coordinates of one vision block of `grid` (patches, `[time, height, width]`)
/// whose first row takes position `origin`: `[3, rows]` I32, axis-major, the block's merged
/// tokens in raster order, time slowest.
///
/// The merged token at `(i, j, k)` takes `origin + (i, j, k)`
/// (`modeling_qwen3_vl.py:918-924`).
///
/// # Errors
///
/// A grid that is not whole merge blocks, or coordinates wider than an I32.
pub fn vision_block(grid: [u32; 3], merge: u32, origin: u32) -> Result<Tensor> {
    check_grid(grid, merge)?;
    let [time, height, width] = grid;
    let (rows, columns) = (height / merge, width / merge);
    coordinate(origin + time.max(rows).max(columns))?;

    // One axis's index along its own extent of the block, the same along the other two.
    let block = (time as usize, rows as usize, columns as usize);
    let axis = |extent: u32, along: (usize, usize, usize)| -> Result<Tensor> {
        Ok(Tensor::arange(0_i32, extent as i32, &Device::Cpu)?
            .reshape(along)?
            .broadcast_as(block)?
            .flatten_all()?)
    };
    let coordinates = Tensor::stack(
        &[
            axis(time, (block.0, 1, 1))?,
            axis(rows, (1, block.1, 1))?,
            axis(columns, (1, 1, block.2))?,
        ],
        0,
    )?;

    Ok(coordinates.broadcast_add(&Tensor::new(origin as i32, &Device::Cpu)?)?)
}

/// The prompt row of every merged vision row, in prompt order: `[rows]` U32.
///
/// Entry `k` is the prompt row the `k`-th merged vision row takes, so the prompt's
/// placeholders are visited in the order the tower emits its tokens and a text row never
/// appears.
///
/// # Errors
///
/// When the tensor cannot be made.
pub fn vision_rows(ids: &[u32], image_token_id: u32, video_token_id: u32) -> Result<Tensor> {
    let rows: Vec<u32> = (0_u32..)
        .zip(ids)
        .filter(|(_, id)| **id == image_token_id || **id == video_token_id)
        .map(|(row, _)| row)
        .collect();

    Ok(Tensor::new(rows.as_slice(), &Device::Cpu)?)
}

/// `grid`, a `[height, width]` table of one value per patch, flattened in merge-block order:
/// `[height · width]`.
///
/// A merging encoder serializes a patch grid block by block: the blocks of `merge × merge`
/// patches in raster order, and the patches of a block in raster order, so a merged token
/// reads `merge²` consecutive patch rows (`refs/vllm/vllm/model_executor/models/
/// qwen3_vl.py:695-715`).
fn merge_order(grid: &Tensor, merge: u32) -> Result<Tensor> {
    let (height, width) = grid.dims2()?;
    let merge = merge as usize;

    Ok(grid
        .reshape((height / merge, merge, width / merge, merge))?
        .permute((0, 2, 1, 3))?
        .flatten_all()?)
}

/// The rotary coordinates of every patch of `grid`: `[2, patches]` I32, axis-major (row,
/// column), the patches of each time step in merge-block order.
///
/// The tower rotates each patch by its own position in its frame, and every time step repeats
/// the frame's coordinates (`qwen3_vl.py:721-727`).
///
/// # Errors
///
/// A grid that is not whole merge blocks.
pub fn patch_coordinates(grid: [u32; 3], merge: u32) -> Result<Tensor> {
    check_grid(grid, merge)?;
    let [time, height, width] = grid;
    let frame = (height as usize, width as usize);

    let rows = Tensor::arange(0_i32, height as i32, &Device::Cpu)?
        .reshape((frame.0, 1))?
        .broadcast_as(frame)?;
    let columns = Tensor::arange(0_i32, width as i32, &Device::Cpu)?
        .reshape((1, frame.1))?
        .broadcast_as(frame)?;
    let coordinates = Tensor::stack(
        &[merge_order(&rows, merge)?, merge_order(&columns, merge)?],
        0,
    )?;

    Ok(coordinates.repeat((1, time as usize))?)
}

/// `torch.linspace(0, end, steps)` in FP32: `[steps]`.
///
/// The step is one FP32 division of the span, and the points are laid from both ends as
/// PyTorch's kernel lays them (`aten/src/ATen/native/RangeFactories.cpp`): `step · i` below
/// the halfway point and `end − step · (steps − 1 − i)` from it on, so the last point is `end`
/// exactly. The two halves differ by an ulp wherever the step is not representable, and a
/// sample that truncates to another table row reads two other rows.
fn linspace(end: u32, steps: u32) -> Result<Tensor> {
    let index = Tensor::arange(0_f32, steps as f32, &Device::Cpu)?;
    if steps == 1 {
        return Ok(index);
    }

    let end = end as f32;
    let step = f64::from(end / (steps - 1) as f32);
    let ascending = index.affine(step, 0.0)?;
    let descending = index
        .affine(-1.0, f64::from(steps - 1))?
        .affine(-step, 0.0)?
        .broadcast_add(&Tensor::new(end, &Device::Cpu)?)?;

    let halfway = Tensor::new((steps / 2) as f32, &Device::Cpu)?;
    Ok(index
        .broadcast_lt(&halfway)?
        .where_cond(&ascending, &descending)?)
}

/// The FP32 coordinate on a learned table of `side` rows of each of an axis's `size` patches,
/// corners aligned: `[size]`.
fn table_coordinates(side: u32, size: u32) -> Result<Tensor> {
    linspace(side - 1, size)
}

/// The table rows above each of an axis's `floor` rows, clamped at the table's last row.
fn ceil(floor: &Tensor, side: u32) -> Result<Tensor> {
    Ok(floor
        .broadcast_add(&Tensor::new(1_u32, &Device::Cpu)?)?
        .minimum(f64::from(side - 1))?)
}

/// Refuses a learned table of no row.
fn check_side(side: u32) -> Result<()> {
    if side == 0 {
        return Err(Error::Config(
            "the position table needs at least one row".to_owned(),
        ));
    }
    Ok(())
}

/// The rows of the learned `side × side` position table every patch of a `height × width`
/// frame blends: `[4, patches]` U32, the top-left, top-right, bottom-left and bottom-right
/// neighbours of the patch's point on the table, the patches in merge-block order.
///
/// The frame samples the table with its corners aligned: patch row `i` reads table row
/// `i · (side − 1) / (height − 1)`, an FP32 `torch.linspace`, truncated to the lower row, the
/// upper neighbour clamped at the last row (`transformers/vision_utils.py:196-215`).
///
/// # Errors
///
/// A table of no row, or a frame that is not whole merge blocks.
pub fn position_taps(side: u32, height: u32, width: u32, merge: u32) -> Result<Tensor> {
    check_side(side)?;
    check_grid([1, height, width], merge)?;

    let top = table_coordinates(side, height)?.to_dtype(DType::U32)?;
    let left = table_coordinates(side, width)?.to_dtype(DType::U32)?;
    let (bottom, right) = (ceil(&top, side)?, ceil(&left, side)?);

    // A corner's table row: its row of the table times the side, plus its column.
    let stride = Tensor::new(side, &Device::Cpu)?;
    let corner = |row: &Tensor, column: &Tensor| -> Result<Tensor> {
        let rows = row.broadcast_mul(&stride)?.unsqueeze(1)?;
        merge_order(&rows.broadcast_add(&column.unsqueeze(0)?)?, merge)
    };

    Ok(Tensor::stack(
        &[
            corner(&top, &left)?,
            corner(&top, &right)?,
            corner(&bottom, &left)?,
            corner(&bottom, &right)?,
        ],
        0,
    )?)
}

/// The FP32 bilinear weights of the rows [`position_taps`] names: `[4, patches]`, in the same
/// order.
///
/// A patch at fraction `(dh, dw)` of its table cell weighs its four neighbours by
/// `(1 − dh)(1 − dw)`, `(1 − dh) dw`, `dh (1 − dw)` and `dh dw`
/// (`transformers/vision_utils.py:216-221`).
///
/// # Errors
///
/// As [`position_taps`].
pub fn position_weights(side: u32, height: u32, width: u32, merge: u32) -> Result<Tensor> {
    check_side(side)?;
    check_grid([1, height, width], merge)?;

    let fraction = |size: u32| -> Result<Tensor> {
        let coordinates = table_coordinates(side, size)?;
        let floor = coordinates.to_dtype(DType::U32)?.to_dtype(DType::F32)?;

        Ok((coordinates - floor)?)
    };
    let (down, across) = (fraction(height)?, fraction(width)?);
    let (up, back) = (down.affine(-1.0, 1.0)?, across.affine(-1.0, 1.0)?);

    let corner = |vertical: &Tensor, horizontal: &Tensor| -> Result<Tensor> {
        let weights = vertical
            .unsqueeze(1)?
            .broadcast_mul(&horizontal.unsqueeze(0)?)?;
        merge_order(&weights, merge)
    };

    Ok(Tensor::stack(
        &[
            corner(&up, &back)?,
            corner(&up, &across)?,
            corner(&down, &back)?,
            corner(&down, &across)?,
        ],
        0,
    )?)
}

/// The packed offsets of the time steps of `grid`: `[time + 1]` U32, `[0, hw, 2hw, …]`.
///
/// The tower never attends across a time step: each temporal patch of a grid is one
/// bidirectional sequence of `height · width` patches, and these are the cumulative lengths
/// its attention reads for both the queries and the keys (`qwen3_vl.py:808-813`).
///
/// # Errors
///
/// A grid of more patches than a U32 counts.
pub fn frame_offsets(grid: [u32; 3]) -> Result<Tensor> {
    let [time, height, width] = grid;
    let patches = height
        .checked_mul(width)
        .filter(|frame| *frame > 0 && frame.checked_mul(time).is_some_and(|all| all < u32::MAX))
        .ok_or_else(|| Error::Config(format!("a {grid:?} grid packs no countable patches")))?;

    Ok(Tensor::arange_step(
        0_u32,
        time * patches + 1,
        patches,
        &Device::Cpu,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE_TOKEN: u32 = 151_655;
    const VIDEO_TOKEN: u32 = 151_656;

    /// The coordinates of `ids` over `grids`, token-major: one `[time, height, width]` per
    /// token.
    fn coordinates(ids: &[u32], grids: &[[u32; 3]]) -> Result<Vec<Vec<i32>>> {
        let index = rope_index(ids, grids, IMAGE_TOKEN, VIDEO_TOKEN, 2)?;
        Ok(index.t()?.to_vec2()?)
    }

    #[test]
    fn a_text_prompt_takes_consecutive_coordinates_on_every_axis() {
        let positions = coordinates(&[10, 11, 12], &[]).expect("positions");
        assert_eq!(positions, [[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
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
        let positions = coordinates(&ids, &[[1, 4, 6]]).expect("positions");
        assert_eq!(positions.len(), 9);

        // The block starts where the text clock is (2), its merged tokens in raster order.
        assert_eq!(positions[2], [2, 2, 2]);
        assert_eq!(positions[3], [2, 2, 3]);
        assert_eq!(positions[4], [2, 2, 4]);
        assert_eq!(positions[5], [2, 3, 2]);

        // The clock advances by `max(4, 6) / 2 = 3`.
        assert_eq!(positions[8], [5, 5, 5]);
    }

    #[test]
    fn a_video_grid_is_one_run_per_temporal_patch() {
        // One video of two temporal patches, each a 2x4 patch grid merged 2x2: two runs of
        // two placeholders, with a timestamp between them.
        let ids = [VIDEO_TOKEN, VIDEO_TOKEN, 7, VIDEO_TOKEN, VIDEO_TOKEN];
        let positions = coordinates(&ids, &[[2, 2, 4]]).expect("positions");
        assert_eq!(positions.len(), 5);

        // Each run is one time step: its merged grid is 1 row by 2 columns, and the clock is
        // the same one every axis reads.
        assert_eq!(positions[0], [0, 0, 0]);
        assert_eq!(positions[1], [0, 0, 1]);

        // The run advances the clock by `max(2, 4) / 2 = 2`, the timestamp takes 2 and leaves
        // 3.
        assert_eq!(positions[2], [2, 2, 2]);
        assert_eq!(positions[3], [3, 3, 3]);
        assert_eq!(positions[4], [3, 3, 4]);
    }

    #[test]
    fn a_grid_without_a_run_or_a_run_without_a_grid_is_refused() {
        assert!(coordinates(&[IMAGE_TOKEN, 3], &[]).is_err());
        assert!(coordinates(&[3], &[[1, 2, 2]]).is_err());

        // A grid whose merged rows the prompt does not have is refused: a 4x4 grid merges to
        // four rows, and one placeholder cannot hold them.
        assert!(coordinates(&[IMAGE_TOKEN], &[[1, 4, 4]]).is_err());

        // A 2x2 grid does merge to one row, so the same prompt with it is well formed.
        assert!(coordinates(&[IMAGE_TOKEN], &[[1, 2, 2]]).is_ok());
    }

    #[test]
    fn a_run_that_does_not_hold_its_grid_is_refused() {
        // A 4x6 image block merges to six rows; five placeholders are one short.
        let five = [
            7,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
        ];
        assert!(coordinates(&five, &[[1, 4, 6]]).is_err());

        // A grid with no run left, and a run with no grid, are both refused.
        assert!(coordinates(&[7, 8], &[[1, 4, 6]]).is_err());
        let six = [
            7,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
        ];
        assert!(coordinates(&six, &[]).is_err());
        assert!(coordinates(&six, &[[1, 4, 6]]).is_ok());

        // A video grid of two time steps needs two runs of one frame each.
        let one = six.map(|id| if id == IMAGE_TOKEN { VIDEO_TOKEN } else { id });
        assert!(coordinates(&one, &[[2, 4, 6]]).is_err());
        assert!(coordinates(&one, &[[1, 4, 6]]).is_ok());
    }

    #[test]
    fn a_block_alone_is_its_run_of_the_prompt() {
        // An image block placed at 7 holds the coordinates its run of a prompt would, which
        // is what a prefill states for an image's rows.
        let block = vision_block([1, 4, 6], 2, 7).expect("a block");
        let ids = [vec![3; 7], vec![IMAGE_TOKEN; 6]].concat();
        let prompt = rope_index(&ids, &[[1, 4, 6]], IMAGE_TOKEN, VIDEO_TOKEN, 2).expect("a prompt");

        assert_eq!(
            block.to_vec2::<i32>().expect("coordinates"),
            prompt
                .narrow(1, 7, 6)
                .expect("the block's run")
                .to_vec2::<i32>()
                .expect("coordinates")
        );
        assert!(vision_block([1, 3, 4], 2, 0).is_err());
    }

    #[test]
    fn the_vision_rows_are_the_placeholders_in_prompt_order() {
        let ids = [7, IMAGE_TOKEN, 8, VIDEO_TOKEN, VIDEO_TOKEN, 9, IMAGE_TOKEN];
        let rows = vision_rows(&ids, IMAGE_TOKEN, VIDEO_TOKEN).expect("rows");
        assert_eq!(rows.to_vec1::<u32>().expect("rows"), [1, 3, 4, 6]);

        let none = vision_rows(&[1, 2, 3], IMAGE_TOKEN, VIDEO_TOKEN).expect("rows");
        assert_eq!(none.elem_count(), 0);
    }

    #[test]
    fn merged_order_is_block_then_patch() {
        // A 4x6 patch grid merges to six rows, and each merged row reads its own 2x2 patches:
        // twenty-four patch coordinates, blocks in raster order and patches within a block in
        // raster order.
        let coordinates = patch_coordinates([1, 4, 6], 2).expect("coordinates");
        let axes: Vec<Vec<i32>> = coordinates.to_vec2().expect("two axes");

        assert_eq!(
            axes[0],
            [
                0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3
            ]
        );
        assert_eq!(
            axes[1],
            [
                0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5
            ]
        );
        assert!(patch_coordinates([1, 3, 4], 2).is_err());

        // Every time step repeats the frame's coordinates.
        let clip = patch_coordinates([2, 2, 2], 2).expect("coordinates");
        assert_eq!(
            clip.to_vec2::<i32>().expect("two axes"),
            [[0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]]
        );
    }

    #[test]
    fn a_linspace_is_laid_from_both_ends() {
        // `torch.linspace(0, 47, 6)`: the step 9.4 is not representable, the points below
        // the halfway point are multiples of it and the rest are 47 less multiples of it.
        let points: Vec<f32> = linspace(47, 6).expect("points").to_vec1().expect("points");
        let step = 47.0_f32 / 5.0;

        assert_eq!(
            points,
            [0.0, step, step * 2.0, 47.0 - step * 2.0, 47.0 - step, 47.0]
        );
        assert_eq!(
            linspace(47, 1)
                .expect("one point")
                .to_vec1::<f32>()
                .expect("one point"),
            [0.0]
        );
    }

    #[test]
    fn the_position_sample_blends_the_four_neighbours() {
        // One sample per patch: a 4x6 frame has twenty-four, the table is 48x48, and patch
        // `(i, j)` reads table point `(i · 47 / 3, j · 47 / 5)` with the corners aligned.
        // Patch-major: each tap list is the top-left, top-right, bottom-left and bottom-right
        // neighbour of that point.
        let patch_major = |tensor: Tensor| tensor.t().expect("patch-major");
        let taps: Vec<Vec<u32>> = patch_major(position_taps(48, 4, 6, 2).expect("taps"))
            .to_vec2()
            .expect("taps");
        let weights: Vec<Vec<f32>> = patch_major(position_weights(48, 4, 6, 2).expect("weights"))
            .to_vec2()
            .expect("weights");
        assert_eq!((taps.len(), weights.len()), (24, 24));

        // The frame's first patch is the table's top-left corner: the top-left tap carries
        // the whole blend and its three neighbours carry none of it.
        assert_eq!(taps[0], [0, 1, 48, 49]);
        assert_eq!(weights[0], [1.0, 0.0, 0.0, 0.0]);

        // Its right neighbour sits at column `47 / 5 = 9.4`: table columns 9 and 10 of the
        // same row, split 0.6 / 0.4.
        assert_eq!(taps[1], [9, 10, 57, 58]);
        assert!((weights[1][0] - 0.6).abs() < 1e-6);
        assert!((weights[1][1] - 0.4).abs() < 1e-6);
        assert_eq!(weights[1][2], 0.0);
        assert_eq!(weights[1][3], 0.0);

        // Every blend partitions one over four taps inside the table, for every patch.
        for (tap, weight) in taps.iter().zip(&weights) {
            assert!(
                tap.iter().all(|row| *row < 2304),
                "{tap:?} outside the table"
            );

            let total: f32 = weight.iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "{weight:?} does not sum to one");
        }

        // A one-block frame reaches the table's four corners through its four patches.
        let corners: Vec<Vec<u32>> = patch_major(position_taps(48, 2, 2, 2).expect("taps"))
            .to_vec2()
            .expect("taps");
        assert_eq!(
            corners,
            [
                [0, 1, 48, 49],
                [47, 47, 95, 95],
                [2256, 2257, 2256, 2257],
                [2303, 2303, 2303, 2303],
            ]
        );
    }

    #[test]
    fn one_sequence_per_temporal_patch() {
        let offsets = frame_offsets([2, 4, 6]).expect("offsets");
        assert_eq!(offsets.to_vec1::<u32>().expect("offsets"), [0, 24, 48]);
    }
}
