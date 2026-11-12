//! The packed sequence against the reference, on CPU.
//!
//! `fixtures/packing/` holds requests of every task and the tensors vLLM-Omni's packing builds
//! for each (`fixtures/generate_packing.py`, which runs the reference's own functions with
//! torch). Every tensor is compared exactly:
//!
//! - the document's coordinates, tags, timestep groups and conditioning rows, in the
//!   reference's row order. The coordinates are compared as the rotary embedding reads them,
//!   rounded to FP32 (`refs/vllm-omni/…/minimax_h3/minimax_h3_transformer.py:351`);
//! - the dense layout, which leads with the generated rows, and the rows a query attends;
//! - the tiled layout: the rows of every tile, each video's tiles and the tiles its queries
//!   keep.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use minimax_h3::config::{TransformerConfig, VsaConfig};
use minimax_h3::packing;
use nn::{ConditionRole, ConditionShape, DType, Device, Layout, Tensor, VideoRequest};
use serde_json::Value;

/// The directory of the committed fixtures.
fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/packing")
}

/// The requests the generator stated.
fn cases() -> Vec<Value> {
    let text = std::fs::read_to_string(fixtures().join("cases.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// The reference's tensors, by `<request>.<tensor>`.
fn expected() -> HashMap<String, Tensor> {
    uniserve_tensor::safetensors::load(fixtures().join("expected.safetensors"), &Device::Cpu)
        .unwrap()
}

/// The released network's widths: the packing reads its latent channels alone.
fn transformer() -> TransformerConfig {
    TransformerConfig {
        hidden_size: 5376,
        num_attention_heads: 56,
        head_dim: 128,
        num_hidden_layers: 50,
        num_refiner_layers: 2,
        intermediate_size: 14336,
        video_channels: 24,
        audio_channels: 32,
        text_dim: 5120,
        frequency_dim: 256,
        time_hidden_dim: 5376,
        time_dim: 2688,
        rope_frequency_dim: 16,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        qk_norm_eps: 1e-5,
        output_heads: 1,
    }
}

/// A field of a request as an unsigned integer.
fn count(case: &Value, field: &str) -> u32 {
    case[field].as_u64().unwrap() as u32
}

/// The request of `case` and the layout it is laid out in.
fn request_of(case: &Value) -> (VideoRequest, Layout) {
    let conditions: Vec<ConditionShape> = case["conditions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|condition| ConditionShape {
            role: match condition["role"].as_str().unwrap() {
                "first_frame" => ConditionRole::FirstFrame,
                "last_frame" => ConditionRole::LastFrame,
                _ => ConditionRole::Reference,
            },
            frames: count(condition, "frames"),
            height: count(condition, "height"),
            width: count(condition, "width"),
            audio_samples: count(condition, "audio_samples"),
        })
        .collect();
    let roles = |role: ConditionRole| {
        conditions
            .iter()
            .filter(|condition| condition.role == role)
            .count() as u32
    };

    let request = VideoRequest {
        width: count(case, "width"),
        height: count(case, "height"),
        frames: count(case, "frames"),
        fps: 24,
        first_frames: roles(ConditionRole::FirstFrame),
        last_frames: roles(ConditionRole::LastFrame),
        references: roles(ConditionRole::Reference),
        text_tokens: count(case, "text_tokens"),
        vision_spans: case["vision_spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|span| {
                (
                    span[0].as_u64().unwrap() as u32,
                    span[1].as_u64().unwrap() as u32,
                )
            })
            .collect(),
        condition_rows: Some(count(case, "condition_rows")),
        conditions,
    };

    let layout = packing::layout(
        &request,
        &transformer(),
        count(case, "text_rows"),
        count(case, "condition_rows"),
    )
    .unwrap();
    packing::check(&request, &layout).unwrap();

    (request, layout)
}

/// The elements of an integer tensor, in row-major order.
fn integers(tensor: &Tensor) -> Vec<i64> {
    tensor
        .to_dtype(DType::I64)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// The bits of a tensor's elements rounded to FP32, in row-major order.
fn bits(tensor: &Tensor) -> Vec<u32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

#[test]
fn the_document_is_the_references_packed_sequence() {
    let expected = expected();

    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (request, layout) = request_of(&case);
        let reference = |tensor: &str| &expected[&format!("{name}.{tensor}")];

        let positions = packing::positions(&request, &layout).unwrap();
        assert_eq!(positions.dtype(), DType::F64, "{name}");
        assert_eq!(positions.dims(), reference("positions").dims(), "{name}");
        assert_eq!(
            bits(&positions),
            bits(reference("positions")),
            "{name}: positions"
        );

        assert_eq!(
            integers(&packing::tags(&request, &layout).unwrap()),
            integers(reference("tags")),
            "{name}: tags"
        );
        assert_eq!(
            integers(&packing::groups(&request, &layout).unwrap()),
            integers(reference("groups")),
            "{name}: groups"
        );
        assert_eq!(
            integers(&packing::prefix_index(&request, &layout).unwrap()),
            integers(reference("prefix_index")),
            "{name}: prefix index"
        );
    }
}

#[test]
fn the_dense_layout_leads_with_the_generated_rows_and_bounds_the_keys() {
    let expected = expected();

    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (request, layout) = request_of(&case);
        let reference = integers(&expected[&format!("{name}.dense_rows")]);
        let used = integers(&expected[&format!("{name}.used")])[0] as usize;

        // The sequence boundary: the rows a query attends are the reference's used rows.
        assert_eq!(packing::dense_used(&request, &layout), used, "{name}");
        assert_eq!(packing::document_rows(&request, &layout), used, "{name}");

        for align in [1, 8] {
            let rows = integers(&packing::dense_rows(&request, &layout, align).unwrap());

            assert_eq!(rows.len(), packing::dense_len(&layout, align), "{name}");
            assert!(rows.len().is_multiple_of(align as usize), "{name}");
            assert_eq!(rows[..used], reference[..], "{name}: dense rows");
            assert!(
                rows[used..].iter().all(|row| *row == used as i64),
                "{name}: the rows after the used ones are padding"
            );
        }
    }
}

/// The tiles of a layout as the document rows each holds, by their first row.
fn tiles_by_first_row(rows: &[i64], sizes: &[i64], tile: usize) -> BTreeMap<i64, Vec<i64>> {
    sizes
        .iter()
        .enumerate()
        .filter(|(_, size)| **size > 0)
        .map(|(index, size)| {
            let held = rows[index * tile..][..*size as usize].to_vec();
            (held[0], held)
        })
        .collect()
}

#[test]
fn the_tiled_layout_holds_the_references_tiles() {
    let expected = expected();
    let mut tiled = 0;

    for case in cases() {
        let Some(tile) = case["tile_rows"].as_u64() else {
            continue;
        };
        tiled += 1;

        let name = case["name"].as_str().unwrap();
        let (request, layout) = request_of(&case);
        let reference = |tensor: &str| integers(&expected[&format!("{name}.{tensor}")]);
        let vsa = VsaConfig::new(
            tile as u32,
            case["sparsity"].as_f64().unwrap(),
            case["reference_keep"].as_f64(),
        )
        .unwrap();
        let tile = tile as usize;
        let document = packing::document_rows(&request, &layout) as i64;

        let layout_rows = packing::tiled_rows(&request, &layout, &vsa).unwrap();
        let rows = integers(&layout_rows);
        assert_eq!(rows.len(), packing::tiled_len(&layout, &vsa), "{name}");

        let table = packing::tile_table(&layout_rows, &request, &layout, &vsa).unwrap();
        assert_eq!(table.dims(), [3, rows.len() / tile], "{name}");
        let sizes = integers(&table.get(0).unwrap());
        let segments = integers(&table.get(1).unwrap());
        let keep = integers(&table.get(2).unwrap());

        // A tile's valid rows lead it and padding fills the rest.
        for (index, size) in sizes.iter().enumerate() {
            let held = &rows[index * tile..(index + 1) * tile];
            assert!(held[..*size as usize].iter().all(|row| *row < document));
            assert!(held[*size as usize..].iter().all(|row| *row == document));
        }

        // The reference's tiles: its partition cut by its sizes. The layout holds the same
        // tiles, each with the same rows in the same order; where a tile sits follows the
        // layout's capacities, so the tiles are matched by the row they start with.
        let partition = reference("tile_partition");
        let mut offset = 0;
        let reference_tiles: Vec<Vec<i64>> = reference("tile_sizes")
            .iter()
            .map(|size| {
                let held = partition[offset..offset + *size as usize].to_vec();
                offset += *size as usize;
                held
            })
            .collect();
        let by_first: BTreeMap<i64, Vec<i64>> = reference_tiles
            .iter()
            .map(|held| (held[0], held.clone()))
            .collect();
        assert_eq!(
            tiles_by_first_row(&rows, &sizes, tile),
            by_first,
            "{name}: tiles"
        );

        // Each video's tiles are one segment, in the reference's order of regions, and its
        // queries keep the reference's count of them.
        let regions = reference("tile_regions");
        for (segment, region) in regions.chunks(3).enumerate() {
            let (start, end, kept) = (region[0] as usize, region[1] as usize, region[2]);
            let mut region_tiles: Vec<i64> = reference_tiles[start..end]
                .iter()
                .map(|held| held[0])
                .collect();
            region_tiles.sort_unstable();

            let segment_tiles: Vec<i64> = segments
                .iter()
                .enumerate()
                .filter(|(_, held)| **held == segment as i64)
                .map(|(index, _)| rows[index * tile])
                .collect();

            assert_eq!(segment_tiles, region_tiles, "{name}: segment {segment}");
            assert_eq!(keep[segment], kept, "{name}: segment {segment} keeps");
        }
        let segmented = segments.iter().filter(|segment| **segment >= 0).count();
        let in_regions: i64 = regions.chunks(3).map(|region| region[1] - region[0]).sum();
        assert_eq!(
            segmented as i64, in_regions,
            "{name}: no other tile is in a segment"
        );
    }

    assert!(tiled >= 4, "the fixtures hold tiled requests");
}

#[test]
fn a_layout_gathers_the_documents_rows_and_fills_its_padding() {
    let case = cases()
        .into_iter()
        .find(|case| case["name"] == "ref2va_blocks")
        .unwrap();
    let (request, layout) = request_of(&case);
    let document = packing::document_rows(&request, &layout);

    let rows = packing::dense_rows(&request, &layout, 4).unwrap();
    let tags = packing::tags(&request, &layout).unwrap();
    let packed = integers(&packing::packed(&tags, &rows, 0.0).unwrap());
    let tags = integers(&tags);

    for (row, held) in integers(&rows).into_iter().enumerate() {
        let expected = if held as usize == document {
            i64::from(packing::TAG_VIDEO)
        } else {
            tags[held as usize]
        };
        assert_eq!(packed[row], expected, "row {row}");
    }

    // The layout rows of the generated video are where the layout placed its document rows.
    let video = (layout.video.frames * layout.video.height * layout.video.width / 4) as usize;
    let places = integers(&packing::layout_rows_of(&rows, document - video, video).unwrap());
    assert_eq!(places, (0..video as i64).collect::<Vec<_>>());
}

#[test]
fn a_request_the_layout_cannot_hold_is_refused() {
    let case = cases()
        .into_iter()
        .find(|case| case["name"] == "fl2va_first")
        .unwrap();
    let (request, layout) = request_of(&case);

    let mut long = request.clone();
    long.text_tokens = layout.text_rows + 1;
    assert!(packing::check(&long, &layout).is_err());

    let mut crowded = layout;
    crowded.condition_rows = 0;
    let error = packing::check(&request, &crowded).unwrap_err().to_string();
    assert!(error.contains("reserves 0"), "{error}");

    // A keyframe sits on the generated video's own grid.
    let mut resized = request.clone();
    resized.conditions[0].width += 32;
    assert!(packing::check(&resized, &layout).is_err());

    let mut outside = request;
    outside.vision_spans = vec![(48, 6)];
    assert!(packing::check(&outside, &layout).is_err());
}
