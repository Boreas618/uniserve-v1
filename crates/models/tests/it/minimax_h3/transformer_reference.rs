//! The transformer's modules against the reference, on CPU.
//!
//! `fixtures/transformer/` is a tiny checkpoint in the released checkpoint's names and
//! `fixtures/transformer.expected.safetensors` the inputs and the outputs of vLLM-Omni's
//! modules over it (`fixtures/generate_transformer.py`: `MiniMaxH3DiTBlock`,
//! `MiniMaxH3TimeEmbedder`, `MiniMaxH3AdalnProj`, `MiniMaxH3Rope`, `MiniMaxH3TokenRefiner` and
//! `MiniMaxH3FinalLayer`, in FP32 on CPU). The crate's modules load the same checkpoint
//! through `nn::CpuWeights` and compute on CPU tensors.
//!
//! ## Tolerances
//!
//! Both sides evaluate the same formulas in FP32; they differ in the order of each reduction
//! and in the last place of the elementary functions. A reduction of `K` terms differs by at
//! most `K · 2⁻²⁴` relative between two orders, and a module chains `L` such stages, so each
//! comparison bounds the difference by `L · K · 2⁻²⁴` of a value's magnitude, with
//!
//! - `K = 48`, the longest reduction: the feed-forward's width (the attention's 40 keys, the
//!   hidden width 32 and the head width 16 are shorter);
//! - `L` the stages of the module under test, counted in each test.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
use super::support;

use std::collections::HashMap;
use std::path::PathBuf;

use models::minimax_h3::VideoTask;
use models::minimax_h3::config::{DenoiserConfig, ScheduleConfig, TransformerConfig};
use models::minimax_h3::modulation::{self, Modulation};
use models::minimax_h3::refiner::TokenRefiner;
use models::minimax_h3::transformer::{Block, Transformer, rotary_config};
use nn::family::DenoiserSplit;
use nn::modules::{Rotary, TimestepTable, TimestepTableConfig};
use nn::{Checkpoint, CpuWeights, DType, Device, Tensor, WeightLoader};
use support::{assert_close, values};

/// The FP32 unit roundoff.
const ROUNDOFF: f32 = 1.0 / 16_777_216.0;

/// The longest reduction of the tiny network: the feed-forward's width.
const LONGEST_REDUCTION: f32 = 48.0;

/// The rows of the packed sequence the fixture's block runs over.
const ROWS: u32 = 40;

/// The directory of the committed fixtures.
fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/it/minimax_h3/fixtures")
}

/// The inputs and the reference's outputs, by name.
fn expected() -> HashMap<String, Tensor> {
    let path = fixtures().join("transformer.expected.safetensors");
    crate::fixture::tensors(&path)
}

/// The tiny network the generator states: two blocks of two heads of sixteen channels, twelve
/// of them rotated, and one refiner block.
fn config() -> DenoiserConfig {
    let transformer = TransformerConfig {
        hidden_size: 32,
        num_attention_heads: 2,
        head_dim: 16,
        num_hidden_layers: 2,
        num_refiner_layers: 1,
        intermediate_size: 48,
        video_channels: 4,
        audio_channels: 8,
        text_dim: 24,
        frequency_dim: 16,
        time_hidden_dim: 32,
        time_dim: 16,
        rope_frequency_dim: 2,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        qk_norm_eps: 1e-5,
        output_heads: 1,
    };

    DenoiserConfig::new(
        transformer,
        ScheduleConfig::uniform(50, 12.0, 3.0).unwrap(),
        None,
        vec![VideoTask::TextToVideoAudio],
    )
    .unwrap()
}

/// The modulation table of the fixture's four timesteps, filled by the checkpoint's
/// projections: one step whose four groups read the four levels.
fn modulation_table(loader: &WeightLoader<'_>, timesteps: &Tensor) -> TimestepTable {
    let transformer = config().transformer;
    let modulation = Modulation::load(&transformer, loader.root()).unwrap();

    let table = TimestepTable::load(
        &TimestepTableConfig {
            frequency_dim: transformer.frequency_dim,
            layers: transformer.num_hidden_layers,
            layer_columns: modulation::BLOCK_COLUMNS * transformer.hidden_size,
            output_columns: modulation::OUTPUT_COLUMNS * transformer.hidden_size,
            dtype: modulation.dtype(),
        },
        &timesteps.unsqueeze(0).unwrap(),
        loader.root().pp("adaln"),
    )
    .unwrap();
    modulation.fill(&table).unwrap();

    table
}

/// Each row's modulation row, `3 · group + tag`, as the blocks index their tables.
fn modulation_rows(expected: &HashMap<String, Tensor>) -> Tensor {
    let groups = expected["groups"].to_dtype(DType::U32).unwrap();
    let tags = expected["tags"].to_dtype(DType::U32).unwrap();

    (groups.affine(3.0, 0.0).unwrap() + tags).unwrap()
}

/// The one sequence of the fixture: every row a query and a key.
fn offsets() -> Vec<Tensor> {
    let table = || Tensor::new(&[0_u32, ROWS], &Device::Cpu).unwrap();
    vec![table(), table()]
}

#[test]
fn the_modulation_is_the_references_adaln_projection_of_each_level() {
    let expected = expected();
    let checkpoint = Checkpoint::open(&fixtures().join("transformer")).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let table = modulation_table(&loader, &expected["timesteps"]);

    // The sinusoidal embedding, two projections of the embedder and the modulation projection.
    let tolerance = 4.0 * LONGEST_REDUCTION * ROUNDOFF;
    for layer in 0..2 {
        assert_close(
            &values(&table.layer_table(layer).unwrap()),
            &values(&expected[&format!("block{layer}.modulation")]),
            tolerance,
        );
    }
    assert_close(
        &values(table.output_table()),
        &values(&expected["final_modulation"]),
        tolerance,
    );
}

#[test]
fn the_rotary_factors_are_the_references_table() {
    let expected = expected();
    let checkpoint = Checkpoint::open(&fixtures().join("transformer")).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    let rotary = Rotary::load(
        &rotary_config(&config().transformer),
        loader.root().pp("rotary_emb"),
    )
    .unwrap();
    let positions = expected["positions"].t().unwrap().contiguous().unwrap();
    let (cos, sin) = rotary.forward(&positions).unwrap();

    // The reference's table is `[cos | sin]` of the same FP32 angles: the two sides differ by
    // the last place of the cosine and of the sine.
    let table = Tensor::cat(&[&cos, &sin], 1).unwrap();
    assert_eq!(table.dims(), expected["rope_table"].dims());
    assert_close(
        &values(&table),
        &values(&expected["rope_table"]),
        2.0 * ROUNDOFF,
    );
}

#[test]
fn a_block_is_the_references_block() {
    let expected = expected();
    let checkpoint = Checkpoint::open(&fixtures().join("transformer")).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let config = config();

    let block = Block::load(
        &config,
        DenoiserSplit::WHOLE,
        ROWS,
        loader.root().pp("transformer_blocks").pp(0),
    )
    .unwrap();

    // The block is given the reference's own modulation and rotary table, so the comparison
    // is of the block alone: `[cos | sin]` halves of the table are its two factor tables.
    let table = &expected["rope_table"];
    let half = table.dims()[1] / 2;
    let cos = table.narrow(1, 0, half).unwrap().contiguous().unwrap();
    let sin = table.narrow(1, half, half).unwrap().contiguous().unwrap();

    let output = block
        .forward(
            &expected["block0.modulation"],
            &modulation_rows(&expected),
            &expected["hidden"],
            (&cos, &sin),
            &offsets(),
        )
        .unwrap();

    // Two modulated norms, three projections with the per-head norm and the rotation, the
    // attention, the output projection, the feed-forward's two products and its gate, and
    // the two gated sums: twelve stages.
    assert_close(
        &values(&output),
        &values(&expected["block0.out"]),
        12.0 * LONGEST_REDUCTION * ROUNDOFF,
    );
}

#[test]
fn the_stack_and_its_heads_are_the_references() {
    let expected = expected();
    let checkpoint = Checkpoint::open(&fixtures().join("transformer")).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let config = config();

    let transformer =
        Transformer::load(&config, DenoiserSplit::WHOLE, ROWS, loader.root()).unwrap();
    transformer.initialize().unwrap();
    let table = modulation_table(&loader, &expected["timesteps"]);
    let entries = table.entries_of(0).unwrap();

    let positions = expected["positions"].t().unwrap().contiguous().unwrap();
    let (cos, sin) = transformer.rotary_factors(&positions).unwrap();
    let hidden = transformer
        .forward(
            &table,
            &entries,
            &expected["hidden"],
            &modulation_rows(&expected),
            (&cos, &sin),
            &offsets(),
        )
        .unwrap();

    // The modulation (four stages) and two blocks of twelve.
    let stack = 28.0 * LONGEST_REDUCTION * ROUNDOFF;
    assert_close(&values(&hidden), &values(&expected["block1.out"]), stack);

    // The final norm and a head: three more stages.
    let groups = expected["groups"].to_dtype(DType::U32).unwrap();
    let (video, audio) = transformer
        .velocities(&table, &entries, &hidden, &groups, 0)
        .unwrap();
    let heads = 31.0 * LONGEST_REDUCTION * ROUNDOFF;
    assert_close(&values(&video), &values(&expected["video_velocity"]), heads);
    assert_close(&values(&audio), &values(&expected["audio_velocity"]), heads);
}

#[test]
fn the_refined_prompt_is_the_references() {
    let expected = expected();
    let checkpoint = Checkpoint::open(&fixtures().join("transformer")).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    let features = &expected["features"];
    let rows = features.dims()[0] as u32;
    let refiner = TokenRefiner::load(&config().transformer, rows, loader.root()).unwrap();
    let refined = refiner.forward(features, rows).unwrap();

    // The input projection, a block (two norms, three projections with the per-head norms,
    // the attention, the output projection, the feed-forward) and the final norm.
    assert_close(
        &values(&refined),
        &values(&expected["refined"]),
        12.0 * LONGEST_REDUCTION * ROUNDOFF,
    );
}
