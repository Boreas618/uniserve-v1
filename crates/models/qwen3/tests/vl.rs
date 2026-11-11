//! The Qwen3-VL encoder against a released checkpoint (the MiniMax-H3 text encoder): it loads its
//! layers, tower and DeepStack mergers at the extents the configuration states, and an encode of a
//! prompt with an image composes the tower, the splice, the decoder layers and the DeepStack
//! additions, uploading the prompt's and the grid's host values.
//!
//! The load is a meta load and the forward a recording forward: it reads the checkpoint's headers
//! and launches nothing, so the test needs the checkpoint's files and no device.

use std::collections::BTreeSet;
use std::path::PathBuf;

use nn::capability::{
    Encoder, EncoderInputs, EncoderOutputs, EncoderPrompt, ImageInput, ImageShape, TokenBatch,
};
use nn::family::Split;
use nn::{Checkpoint, DType, PhaseKind, Shape, WeightLoader};
use processing::qwen_vl::ProcessorConfig;
use qwen3::vision::pixel_norm;
use qwen3::{Config, Qwen3VlModel};

/// The checkpoint root the test reads, from `UNISERVE_H3_MODEL` (a MiniMax-H3 checkpoint, whose
/// `text_encoder` is a Qwen3-VL model and whose `processor` holds its processor files).
fn root() -> PathBuf {
    std::env::var_os("UNISERVE_H3_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("UNISERVE_H3_MODEL names a MiniMax-H3 checkpoint"))
}

/// The programs a launch reported, as `(operator, key)` pairs.
#[derive(Default)]
struct Launches(BTreeSet<(String, String)>);

impl ops::capture::Sink for Launches {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.insert((operator.to_owned(), key.to_owned()));
    }
}

#[test]
#[ignore = "requires UNISERVE_H3_MODEL naming a MiniMax-H3 checkpoint"]
fn an_image_prompt_encodes_through_the_tower_the_layers_and_deepstack() {
    let root = root();
    let checkpoint = Checkpoint::open(&root.join("text_encoder"))
        .unwrap_or_else(|error| panic!("the shards: {error}"));
    let config = Config::open(&checkpoint).unwrap_or_else(|error| panic!("the config: {error}"));
    let read = |name: &str| {
        let text = std::fs::read_to_string(root.join("processor").join(name))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        serde_json::from_str::<serde_json::Value>(&text)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    };
    let [image_file, video_file] = ProcessorConfig::FILES;
    let (image, video) = (read(image_file), read(video_file));
    let processor = ProcessorConfig::from_values(&image, &video)
        .unwrap_or_else(|error| panic!("the processor: {error}"));
    let norm = pixel_norm(
        &nn::config::Fields::of(image_file, &image).unwrap_or_else(|error| panic!("{error}")),
    )
    .unwrap_or_else(|error| panic!("the pixel normalization: {error}"));

    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let layers = 8;
    let model = Qwen3VlModel::load(
        &mut loader,
        &config,
        (processor, norm),
        Split::WHOLE,
        DType::BF16,
        layers,
        false,
    )
    .unwrap_or_else(|error| panic!("the meta load: {error}"));
    let vision = config.vision.as_ref().unwrap_or_else(|| panic!("a tower"));

    // Three text tokens, one 24 × 36-patch image (216 merged rows), two text tokens.
    let grid = [1_u32, 24, 36];
    let merged = 24 * 36 / 4;
    let mut ids = vec![10_u32, 11, 12];
    ids.extend(std::iter::repeat_n(vision.image_token_id, merged));
    ids.extend([13, 14]);
    let tokens = ids.len() as u64;

    let arena = nn::Arena::counting();
    let staging = nn::forward::Staging::counting();
    let f = nn::Forward::recording(&arena).with_staging(&staging);
    let tensor = |dtype: DType, dims: &[u64]| {
        arena
            .tensor(dtype, Shape::new(dims))
            .unwrap_or_else(|error| panic!("a counting arena holds any reservation: {error}"))
    };
    let batch = TokenBatch {
        positions: tensor(DType::I32, &[tokens]),
        slot_mapping: tensor(DType::I32, &[tokens]),
        seq_lens: tensor(DType::I32, &[1]),
        query_offsets: tensor(DType::I32, &[2]),
        logits_rows: tensor(DType::I32, &[1]),
        phase: PhaseKind::Prefill,
        page_size: 64,
        num_tokens: tokens as u32,
        max_kv_len: tokens as u32,
        draft_lens: None,
        rope_positions: None,
    };
    let layout = model.layout(&batch, &[ImageShape { grid }]);
    let patches = u64::from(grid[1] * grid[2]);
    let images = [ImageInput {
        pixels: tensor(DType::BF16, &[patches, u64::from(vision.tubelet())]),
    }];
    let hidden = tensor(DType::BF16, &[tokens, u64::from(config.text.hidden_size)]);

    let mut launches = Launches::default();
    ops::capture::set_sink(Some(&mut launches));
    let walked = model.encode(
        &f,
        &EncoderPrompt { ids: &ids },
        EncoderInputs {
            ids: tensor(DType::I32, &[tokens]),
            batch,
            images: &images,
            layout: &layout,
        },
        EncoderOutputs { hidden },
    );
    ops::capture::set_sink(None);
    walked.unwrap_or_else(|error| panic!("the recorded encode: {error}"));

    let operators: BTreeSet<&str> = launches
        .0
        .iter()
        .map(|(operator, _)| operator.as_str())
        .collect();
    for operator in [
        "position_interpolate",
        "rows",
        "rotary_factors",
        "copy_block",
    ] {
        assert!(
            operators.contains(operator),
            "{operator} launches: {operators:?}"
        );
    }
    // The tower's repeated 2-D spectrum and the language model's sectioned M-RoPE are two programs.
    let factors = launches
        .0
        .iter()
        .filter(|(operator, _)| operator == "rotary_factors")
        .count();
    assert_eq!(factors, 2, "{launches:?}", launches = launches.0);
    // The prompt's M-RoPE coordinates and placeholder rows, and the grid's taps, weights, rotary
    // coordinates and offsets, are uploaded.
    let uploaded = 4 * (3 * tokens + merged as u64) + patches * (4 * 4 + 4 * 4 + 2 * 4) + 8;
    assert!(
        staging.peak() >= uploaded,
        "{} staged of at least {uploaded}",
        staging.peak()
    );

    // A prompt whose placeholders do not hold the image's merged rows is refused.
    let short = &ids[..ids.len() - 3];
    let refused = model.encode(
        &f,
        &EncoderPrompt { ids: short },
        EncoderInputs {
            ids: tensor(DType::I32, &[tokens]),
            batch,
            images: &images,
            layout: &layout,
        },
        EncoderOutputs { hidden },
    );
    assert!(refused.is_err());
}
