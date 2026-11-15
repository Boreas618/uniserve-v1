//! The engine's capture policy over a model's declarations: the condition ladder, one captured
//! request per canvas, kind, timeline, capacity and rung, the condition sets each layout opens
//! with, and the warm prompts, over a model that declares its mathematics and loads nothing.

use std::sync::Arc;

use nn::family::{ConditionBounds, ConditionMedia, Description, RequestKind, VideoModel};
use nn::{ConditionRole, ConditionShape, DType, Decoder, Denoiser, Encoder, MediaFormat, Splicer};

use super::{LEAST_CONDITION_RUNG, VideoServing, candidates, condition_ladder, warm_prompts};

/// The placeholder token a prompt presents a merged vision row by.
const PLACEHOLDER: u32 = 7;

/// A pipeline of 24 frames per second on the `17n + 5` frame rule, admitting four seconds and
/// more, at 1344x768 and 768x1344, whose condition row is a 32-pixel cell of one frame or a
/// thousand samples of a track, in 64-row capacity units, and whose prompt presents a frame as
/// its 16-pixel patches merged two by two.
///
/// The largest references a rung of `rows` rows holds are an image of exactly `rows` cells and a
/// track of half the rows, at most 500.
struct Declared {
    kinds: Vec<RequestKind>,
}

/// `t2va`, which carries no condition.
fn plain() -> RequestKind {
    RequestKind {
        roles: Vec::new(),
        media: ConditionMedia::None,
        vision_blocks: 0,
    }
}

/// `fl2va`: the first frame, the last or both, each presented in the prompt.
fn keyframes() -> RequestKind {
    RequestKind {
        roles: vec![ConditionRole::FirstFrame, ConditionRole::LastFrame],
        media: ConditionMedia::Canvas,
        vision_blocks: 2,
    }
}

/// `ref2va`: references of the client's media, up to twelve presented in a prompt.
fn references() -> RequestKind {
    RequestKind {
        roles: vec![ConditionRole::Reference],
        media: ConditionMedia::Client,
        vision_blocks: 12,
    }
}

/// A deployment serving 1344x768 outputs of up to five seconds, prompts of up to 2048 tokens and
/// 2048 condition rows.
fn serving() -> VideoServing {
    VideoServing {
        sizes: vec![(1344, 768)],
        max_seconds: 5.0,
        text_capacities: vec![1024, 2048],
        condition_rows: 2048,
    }
}

/// The rows one condition of [`Declared`] takes.
fn rows(condition: &ConditionShape) -> u32 {
    condition.frames.max(1) * condition.width * condition.height / (32 * 32)
        + condition.audio_samples / 1000
}

impl VideoModel for Declared {
    fn text_encoder(&self) -> Option<Arc<dyn Encoder + Send + Sync>> {
        None
    }

    fn text_width(&self) -> u32 {
        16
    }

    fn denoiser(&self) -> Option<Arc<dyn Denoiser + Send + Sync>> {
        None
    }

    fn denoiser_stage(&self) -> &'static str {
        "transformer"
    }

    fn sample_type(&self) -> DType {
        DType::F32
    }

    fn video_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>> {
        None
    }

    fn splicer(&self) -> Option<Arc<dyn Splicer + Send + Sync>> {
        None
    }

    fn audio_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>> {
        None
    }

    fn audio_rate(&self) -> u32 {
        32_000
    }

    fn frame_rate(&self) -> u32 {
        24
    }

    fn decode_window(&self) -> u32 {
        28
    }

    fn format(&self) -> nn::Result<MediaFormat> {
        Err(nn::Error::Unsupported("no media format".to_owned()))
    }

    fn timeline(&self, seconds: f64) -> nn::Result<u32> {
        let frames = (seconds * 24.0).round() as u32;
        Ok(frames.saturating_sub(5).div_ceil(17) * 17 + 5)
    }

    fn video_units(&self, _: u32) -> nn::Result<Vec<u32>> {
        Ok(Vec::new())
    }

    fn condition_bounds(&self) -> ConditionBounds {
        ConditionBounds {
            rows: 2048,
            pixels: 0,
            samples: 0,
            patches_per_token: 4,
        }
    }

    fn canvases(&self) -> nn::Result<Vec<(u32, u32)>> {
        Ok(vec![(1344, 768), (768, 1344)])
    }

    fn timelines(&self, seconds: f64) -> nn::Result<Vec<u32>> {
        let longest = self.timeline(seconds)?;
        Ok((self.timeline(4.0)?..=longest).step_by(17).collect())
    }

    fn requests(&self) -> Vec<RequestKind> {
        self.kinds.clone()
    }

    fn capacity_unit(&self) -> nn::Result<u32> {
        Ok(64)
    }

    fn condition_rows(&self, conditions: &[ConditionShape], _: (u32, u32)) -> nn::Result<u32> {
        Ok(conditions.iter().map(rows).sum())
    }

    fn largest_conditions(
        &self,
        _: (u32, u32),
        _: u32,
        rows: u32,
    ) -> nn::Result<Vec<ConditionShape>> {
        let reference = |frames, height, width, audio_samples| ConditionShape {
            role: ConditionRole::Reference,
            frames,
            height,
            width,
            audio_samples,
        };
        Ok(vec![
            reference(1, 32, 32 * rows, 0),
            reference(0, 0, 0, (rows / 2).min(500) * 1000),
        ])
    }

    fn vision_tokens(&self) -> nn::Result<nn::family::VisionTokens> {
        Ok(nn::family::VisionTokens {
            placeholder: PLACEHOLDER,
            merge: 2,
        })
    }

    fn vision_grid(&self, width: u32, height: u32) -> nn::Result<[u32; 3]> {
        Ok([1, height / 16, width / 16])
    }

    fn description(&self, _: &[(u32, u32)]) -> nn::Result<Description> {
        Err(nn::Error::Unsupported("no description".to_owned()))
    }
}

/// The ladder starts at one keyframe and doubles, with a rung halfway between two doublings from
/// four keyframes up; every rung is whole capacity units, and the last is the capacity itself,
/// also when it is below the least rung.
#[test]
fn a_ladder_doubles_from_one_keyframe_to_the_capacity() {
    assert_eq!(LEAST_CONDITION_RUNG, 1024);
    assert_eq!(condition_ladder(2048, 64), [1024, 2048]);
    assert_eq!(
        condition_ladder(49_152, 128),
        [
            1024, 2048, 4096, 6144, 8192, 12_288, 16_384, 24_576, 32_768, 49_152
        ]
    );
    // 96-row units: 1024 rows round up to 1056, 2048 to 2112 and 4096 to 4128.
    assert_eq!(condition_ladder(4800, 96), [1056, 2112, 4128, 4800]);
    assert_eq!(condition_ladder(64, 64), [64]);
}

/// A deployment captures one request per canvas, kind of request, timeline and prompt capacity,
/// a conditioned kind's once per rung and a plain one's with no condition row, each stating its
/// kind's roles and none of its conditions.
#[test]
fn a_deployment_captures_a_request_per_canvas_kind_timeline_capacity_and_rung() {
    let model = Declared {
        kinds: vec![plain(), keyframes()],
    };
    let captured = candidates(&model, &serving()).unwrap();

    let stated: Vec<(u32, u32, usize, Option<u32>)> = captured
        .iter()
        .map(|(request, _)| {
            (
                request.frames,
                request.text_tokens,
                request.roles.len(),
                request.condition_rows,
            )
        })
        .collect();
    // Four and five seconds are 107 and 124 frames.
    assert_eq!(
        stated,
        [
            (107, 1024, 0, None),
            (107, 2048, 0, None),
            (124, 1024, 0, None),
            (124, 2048, 0, None),
            (107, 1024, 2, Some(1024)),
            (107, 1024, 2, Some(2048)),
            (107, 2048, 2, Some(1024)),
            (107, 2048, 2, Some(2048)),
            (124, 1024, 2, Some(1024)),
            (124, 1024, 2, Some(2048)),
            (124, 2048, 2, Some(1024)),
            (124, 2048, 2, Some(2048)),
        ]
    );
    for (request, _) in &captured {
        assert_eq!(
            (request.width, request.height, request.fps),
            (1344, 768, 24)
        );
        assert!(request.conditions.is_empty() && request.vision_spans.is_empty());
    }
}

/// A keyframe set opens the layout of the smallest rung that holds its rows: either keyframe
/// alone (1008 rows) the 1024-row rung, both (2016 rows) the 2048-row rung, and a set no rung
/// holds none; a plain request opens with nothing.
#[test]
fn a_keyframe_set_opens_the_smallest_rung_that_holds_it() {
    let model = Declared {
        kinds: vec![plain(), keyframes()],
    };
    let keyframe = |role| ConditionShape {
        role,
        frames: 1,
        height: 768,
        width: 1344,
        audio_samples: 0,
    };
    let opened = |captured: &[super::Candidate], rung| {
        captured
            .iter()
            .find(|(request, _)| request.condition_rows == rung)
            .map(|(_, sets)| sets.clone())
            .unwrap()
    };

    let captured = candidates(&model, &serving()).unwrap();
    assert!(opened(&captured, None).is_empty());
    assert_eq!(
        opened(&captured, Some(1024)),
        [
            vec![keyframe(ConditionRole::FirstFrame)],
            vec![keyframe(ConditionRole::LastFrame)],
        ]
    );
    assert_eq!(
        opened(&captured, Some(2048)),
        [vec![
            keyframe(ConditionRole::FirstFrame),
            keyframe(ConditionRole::LastFrame),
        ]]
    );

    // A deployment of one 1024-row rung holds no pair of keyframes.
    let narrow = VideoServing {
        condition_rows: 1024,
        ..serving()
    };
    let captured = candidates(&model, &narrow).unwrap();
    assert_eq!(opened(&captured, Some(1024)).len(), 2);
}

/// Each of a rung's largest references opens the smallest rung that holds it, as a request of it
/// rides, and a reference two rungs bound alike opens once: the 2048-row rung's track of 500 rows
/// is the 1024-row rung's, and its image of 1024 cells is the 1024-row rung's own.
#[test]
fn a_reference_opens_the_smallest_rung_that_holds_it_once() {
    let model = Declared {
        kinds: vec![references()],
    };
    let captured = candidates(&model, &serving()).unwrap();

    let shapes = |rung| -> Vec<(u32, u32, u32)> {
        let (_, sets) = captured
            .iter()
            .find(|(request, _)| request.condition_rows == Some(rung))
            .unwrap();
        sets.iter()
            .map(|set| {
                let [reference] = set.as_slice() else {
                    panic!("one reference a set: {set:?}");
                };
                assert_eq!(reference.role, ConditionRole::Reference);
                (reference.frames, reference.width, reference.audio_samples)
            })
            .collect()
    };
    assert_eq!(shapes(1024), [(1, 32 * 1024, 0), (0, 0, 500_000)]);
    assert_eq!(shapes(2048), [(1, 32 * 2048, 0)]);
}

/// A size the model does not generate is refused by name, with the sizes it generates.
#[test]
fn a_size_the_model_does_not_generate_is_refused() {
    let model = Declared {
        kinds: vec![plain()],
    };
    let unserved = VideoServing {
        sizes: vec![(1000, 1000)],
        ..serving()
    };
    let message = candidates(&model, &unserved).unwrap_err().to_string();
    assert!(
        message.contains("1000x1000") && message.contains("1344x768"),
        "{message}"
    );
}

/// Keyframes warm, per served size, a prompt of one keyframe and one of both, each at the
/// model's grid of the size: 768x1344 pixels are 48x84 patches, which merge to 1008 tokens, each
/// block's run of placeholders between two markers. A plain kind warms none.
#[test]
fn keyframes_warm_a_prompt_per_count_at_each_size() {
    let model = Declared {
        kinds: vec![plain(), keyframes()],
    };
    let prompts = warm_prompts(&model, &serving()).unwrap();

    let grids: Vec<&Vec<[u32; 3]>> = prompts.iter().map(|(grids, _)| grids).collect();
    assert_eq!(grids, [&vec![[1, 48, 84]], &vec![[1, 48, 84], [1, 48, 84]]]);
    for (grids, ids) in &prompts {
        let placeholders = ids.iter().filter(|id| **id == PLACEHOLDER).count();
        assert_eq!(placeholders, 1008 * grids.len());
        assert_eq!(ids.len(), 1010 * grids.len());

        // Each block is one run of placeholders, apart from the next.
        let runs = ids
            .chunk_by(|left, right| (*left == PLACEHOLDER) == (*right == PLACEHOLDER))
            .filter(|run| run[0] == PLACEHOLDER)
            .count();
        assert_eq!(runs, grids.len());
    }

    let plain = Declared {
        kinds: vec![plain()],
    };
    assert!(warm_prompts(&plain, &serving()).unwrap().is_empty());
}

/// References warm, per prompt capacity, the capacity's tokens presenting vision rows in one
/// block and in the most blocks a prompt presents, every token beside the blocks' markers a
/// vision row, each block's grid merging to its run of placeholders.
#[test]
fn references_warm_the_most_vision_rows_each_capacity_presents() {
    let model = Declared {
        kinds: vec![references()],
    };
    let deployment = VideoServing {
        text_capacities: vec![1024, 8192],
        ..serving()
    };
    let prompts = warm_prompts(&model, &deployment).unwrap();

    let shape: Vec<(usize, usize)> = prompts
        .iter()
        .map(|(grids, ids)| (ids.len(), grids.len()))
        .collect();
    assert_eq!(shape, [(1024, 1), (1024, 12), (8192, 1), (8192, 12)]);
    for (grids, ids) in &prompts {
        // Two markers a block, and every other token a placeholder.
        let placeholders = ids.iter().filter(|id| **id == PLACEHOLDER).count();
        assert_eq!(placeholders, ids.len() - 2 * grids.len());

        // Each block is one run, of the merged rows its grid's patches make.
        let runs: Vec<usize> = ids
            .chunk_by(|left, right| (*left == PLACEHOLDER) == (*right == PLACEHOLDER))
            .filter(|run| run[0] == PLACEHOLDER)
            .map(<[u32]>::len)
            .collect();
        let merged: Vec<usize> = grids
            .iter()
            .map(|[time, rows, columns]| (time * rows * columns / 4) as usize)
            .collect();
        assert_eq!(runs, merged);
    }
}
