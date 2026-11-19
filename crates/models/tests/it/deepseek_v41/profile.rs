//! Profile runs of DeepSeek-V4.1-Flash on the meta device over the release's headers.
//!
//! A meta load reads the checkpoint's shard headers and no tensor data, and a forward over meta
//! tensors under a meta context launches nothing: it checks every operand, states the kernel each
//! launch would run and counts what the forward allocates. Each run loads the release through its
//! family, as one rank holding the whole model and as rank one of an expert union of four (its
//! window of 96 experts of every layer and its quarter of the n-gram tables' hash heads), and runs
//! a decode bucket of one row, one of eight rows and a prefill bucket of two rows of 256 tokens
//! (the embedding, the decoder stack and the logits), and, on the union's rank, a join of a step
//! it holds no tokens of. Each run asserts the extents of the outputs and the model's
//! declarations, that every launch names an operator, that no kernel key names a model family,
//! that every key is one the tree's kernel packages declare (`crates/ops/kernels`: a package of
//! the launch's operator whose fields read the key), and that a second run of a bucket states
//! the same keys and peak. It prints the keys, the peaks, the host bytes and one layer's launches.
//!
//! The release resolves through the Hugging Face cache ([`crate::releases::DEEPSEEK_V41_FLASH`]);
//! the run needs no device and is ignored by default:
//!
//! ```text
//! cargo test -p models --test it deepseek_v41::profile -- --ignored --nocapture
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use build_server::manifest::Manifest;
use models::deepseek_v41::cache::{COMPRESSED, WINDOW};
use nn::family::{CacheField, Family, History, INDEX_KEY, Layout, TextModel, TextPlacement};
use nn::{DType, ExpertExchange, KvCache, MetaWeights, Phase, RecurrentState, Split, WeightLoader};
use nn_testing::text::{self, WORKSPACE_BYTES};
use nn_testing::{Launch, PAGE_TOKENS, Recording, assert_generic, meta, record, report};

use crate::releases::DEEPSEEK_V41_FLASH;

/// The family the release declares.
const FAMILIES: [Family; 1] = [models::deepseek_v41::family::FAMILY];

/// Request slots of the recurrent state's buffers.
const SLOTS: usize = 64;

/// Members of the profiled expert union.
const UNION: u32 = 4;

/// The most tokens a rank's exchange carries in a step.
const RANK_ROWS: u32 = 512;

/// The kernels a forward stated: `(operator, key)` pairs.
type Keys = BTreeSet<(&'static str, String)>;

/// The model of the release, meta-loaded for `placement`.
fn load(placement: &TextPlacement) -> Arc<dyn TextModel> {
    let checkpoint = DEEPSEEK_V41_FLASH.checkpoint();
    let store = MetaWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    text::load(&FAMILIES, &checkpoint, &mut loader, placement)
        .unwrap_or_else(|error| panic!("{DEEPSEEK_V41_FLASH}: {error}"))
}

/// The meta group of the fields of `history` among `fields`, over `rows` rows of `row_pages`
/// pages each, lent to a forward of `tokens` tokens; a window's states its rows' starts.
fn group(
    fields: &[CacheField],
    history: History,
    (rows, row_pages): (usize, usize),
    tokens: usize,
) -> KvCache {
    let fields: Vec<CacheField> = fields
        .iter()
        .filter(|field| field.history == history)
        .cloned()
        .collect();
    let mut group = text::cache(
        &fields,
        DType::BF16,
        (rows * row_pages, PAGE_TOKENS),
        (
            meta((rows, row_pages), DType::U32),
            meta(tokens, DType::U32),
            meta(WORKSPACE_BYTES, DType::U8),
        ),
        |shape, dtype| Ok(meta(shape, dtype)),
    )
    .expect("a meta cache group");
    if history != History::Full {
        group.starts = Some(meta(rows, DType::U32));
    }
    group
}

/// The rows of an exchange over the union, `step_rows` rows a rank, as meta tensors.
fn exchange(step_rows: u32) -> ExpertExchange {
    let rows = (UNION * RANK_ROWS) as usize;
    let hidden = 5120;

    ExpertExchange {
        ranks: UNION,
        attention: 0,
        rank_rows: RANK_ROWS,
        step_rows,
        multiprocessors: 148,
        states: meta((rows, hidden), DType::BF16),
        ids: meta((rows, 6), DType::U32),
        weights: meta((rows, 6), DType::F32),
        combined: meta((rows, hidden), DType::BF16),
        live: meta(1, DType::U32),
        packed: meta((RANK_ROWS as usize, hidden / 2), DType::U8),
        packed_scales: meta((RANK_ROWS as usize, hidden / 16), DType::F8E4M3),
        fused: false,
        source_only: false,
    }
}

/// The axes a rank's forwards address: its worker group's, unpartitioned, and the union's
/// (`nn::family::EXPERT_AXIS`) on a rank of one.
fn axes(union: Split) -> Vec<Split> {
    if union.size() > 1 {
        vec![Split::WHOLE, union]
    } else {
        Vec::new()
    }
}

/// One step of `model` over a bucket of `rows` rows of `tokens` tokens in all: what it recorded,
/// each launch checked ([`assert_generic`]).
fn step(
    model: &dyn TextModel,
    phase: Phase,
    (rows, tokens): (usize, usize),
    union: Split,
) -> Recording {
    let row_pages = (tokens / rows).div_ceil(PAGE_TOKENS as usize).max(4);
    let lent = (union.size() > 1).then(|| exchange(tokens.next_power_of_two() as u32));

    let (stepped, recording) = record(&axes(union), || {
        let batch = text::batch(
            phase,
            [
                meta(tokens, DType::I32),
                meta(rows, DType::U32),
                meta(rows + 1, DType::U32),
                meta(rows, DType::U32),
            ],
            Some((meta(tokens, DType::U32), meta((rows, 3), DType::U32))),
            PAGE_TOKENS,
            PAGE_TOKENS * row_pages as u32,
        );
        let fields = model.cache();
        let cache = [
            group(&fields, History::Full, (rows, row_pages), tokens),
            group(
                &fields,
                History::Window { tokens: 128 },
                (rows, row_pages),
                tokens,
            ),
        ];
        let shapes = model.state_layers();
        let buffers =
            RecurrentState::buffers(&shapes, |_| SLOTS, |shape, dtype| Ok(meta(shape, dtype)))
                .expect("the state buffers");
        let state = text::state(
            [
                meta(rows + 1, DType::U32),
                meta(rows, DType::U32),
                meta(rows, DType::U32),
                meta(rows, DType::U32),
            ],
            &buffers,
            &buffers,
            None,
        );

        let embedded = model
            .embed_input_ids(&meta(tokens, DType::U32))
            .expect("the embeddings");
        let rows_out = model
            .forward(&batch, &cache, Some(&state), lent.as_ref(), &embedded)
            .unwrap_or_else(|error| panic!("the {phase:?} forward: {error}"));
        assert_eq!(rows_out.dims(), [tokens, model.stream_width() as usize]);
        assert_eq!(rows_out.dtype(), DType::BF16);

        let logits = model
            .compute_logits(&rows_out.narrow(0, 0, rows).expect("one read-out row a row"))
            .expect("the logits");
        assert_eq!(logits.dims(), [rows, model.vocabulary() as usize]);
        assert_eq!(logits.dtype(), DType::F32);
        Ok(())
    });
    stepped.unwrap_or_else(|error| panic!("the {phase:?} step: {error}"));

    assert_generic(&format!("a {phase:?} step"), &recording.launches);
    recording
}

/// The kernel packages of the tree, by operator.
fn packages() -> BTreeMap<String, Vec<(PathBuf, Manifest)>> {
    let kernels = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ops/kernels");
    let mut packages: BTreeMap<String, Vec<(PathBuf, Manifest)>> = BTreeMap::new();
    for operator in std::fs::read_dir(&kernels).expect("the kernel packages") {
        let operator = operator.expect("a package directory").path();
        for backend in std::fs::read_dir(&operator).expect("an operator's packages") {
            let backend = backend.expect("a package directory").path();
            if backend.join(Manifest::FILE).is_file() {
                let manifest = Manifest::load(&backend)
                    .unwrap_or_else(|error| panic!("{}: {error}", backend.display()));
                packages
                    .entry(manifest.op.clone())
                    .or_default()
                    .push((backend, manifest));
            }
        }
    }
    packages
}

/// Asserts that a package of the tree declares every key of `keys`: one of the launch's operator
/// whose fields read the key.
fn assert_declared(run: &str, keys: &Keys) {
    let packages = packages();
    let undeclared: Vec<String> = keys
        .iter()
        .filter(|(operator, key)| {
            !packages.get(*operator).is_some_and(|manifests| {
                manifests
                    .iter()
                    .any(|(_, manifest)| manifest.typed_fields(key).is_some())
            })
        })
        .map(|(operator, key)| format!("{operator} {key}"))
        .collect();
    assert!(
        undeclared.is_empty(),
        "{run}: keys no package of the tree declares:\n{}",
        undeclared.join("\n")
    );
}

/// The launches of decoder layer `layer` of `launches`, by operator: from the attention's read
/// of the layer's streams to the next layer's (four `mhc_boundary` launches a layer before the
/// read-out, the first of a layer opening its attention).
fn layer_launches(launches: &[Launch], layer: usize) -> BTreeMap<&'static str, usize> {
    let boundaries: Vec<usize> = launches
        .iter()
        .enumerate()
        .filter(|(_, (operator, _))| *operator == "mhc_boundary")
        .map(|(index, _)| index)
        .collect();
    let mut operators = BTreeMap::new();
    for (operator, _) in &launches[boundaries[2 * layer]..boundaries[2 * layer + 2]] {
        *operators.entry(*operator).or_default() += 1;
    }
    operators
}

/// Runs the buckets of `model` at `union`'s place twice each, asserts both runs of a bucket state
/// the same keys and peak and that the tree declares every key, and reports them under `name`.
fn buckets(name: &str, model: &dyn TextModel, union: Split) -> Keys {
    let mut every = Keys::new();
    for (phase, bucket) in [
        (Phase::Decode, (1, 1)),
        (Phase::Decode, (8, 8)),
        (Phase::Prefill, (2, 512)),
    ] {
        let recording = step(model, phase, bucket, union);
        let again = step(model, phase, bucket, union);
        let keys: Keys = recording.launches.iter().cloned().collect();
        assert_eq!(
            (&keys, recording.peak),
            (&again.launches.iter().cloned().collect(), again.peak),
            "two runs of one {phase:?} bucket differ"
        );

        let run = format!("{name} {phase:?} {} rows, {} tokens", bucket.0, bucket.1);
        report(&run, &recording);
        assert_declared(&run, &keys);
        // Layer 20 compresses every token, indexes, ranks the candidate blocks and attends.
        println!(
            "{run} layer 20: {:?}",
            layer_launches(&recording.launches, 20)
        );
        assert!(recording.peak > 0, "a forward allocates its activations");
        every.extend(keys);
    }
    every
}

/// What the model declares of its cache, state and experts: the window latent of every layer,
/// the compressed latents and page-laid index keys of the four sources and their readers, the
/// rings of the three ratio-2 sources, and 384 routed experts of 2304, six a token.
fn assert_declarations(model: &dyn TextModel) {
    let fields = model.cache();
    let names: Vec<&str> = fields.iter().map(|field| field.name).collect();
    assert_eq!(names, [WINDOW, COMPRESSED, INDEX_KEY]);
    assert_eq!(fields[0].owners.len(), 40);
    assert_eq!(fields[0].history, History::Window { tokens: 128 });
    let owners: Vec<(u32, u32)> = fields[1]
        .owners
        .iter()
        .map(|owner| (owner.layer, owner.ratio))
        .collect();
    assert_eq!(owners, [(2, 2), (8, 2), (14, 2), (20, 1)]);
    assert_eq!(fields[1].readers.len(), 34);
    let index_readers: Vec<(u32, u32)> = fields[2]
        .readers
        .iter()
        .map(|reader| (reader.layer, reader.owner))
        .collect();
    assert_eq!(index_readers, [(24, 20), (28, 20), (32, 20), (36, 20)]);
    assert_eq!(fields[2].record.layout, Layout::Page);
    assert_eq!(
        [&fields[0], &fields[1], &fields[2]].map(|field| field.record.bytes(DType::BF16).unwrap()),
        [528, 288, 68]
    );

    let [ring] = model.state_layers().try_into().expect("one kind of state");
    assert_eq!(ring.layers, 3);
    assert_eq!(ring.fields[0].dims, [8, 1024]);

    let experts = model.experts().expect("routed experts");
    assert_eq!(
        (
            experts.experts,
            experts.top,
            experts.intermediate,
            experts.encoded
        ),
        (384, 6, 2304, false)
    );
    assert_eq!(model.token_history().map(|history| history.width), Some(3));
}

#[test]
#[ignore = "reads the release's headers through the Hugging Face cache"]
fn a_whole_rank_and_a_rank_of_an_expert_union_state_declared_kernels() {
    let whole = load(&TextPlacement::WHOLE);
    assert_declarations(&*whole);
    assert!(!whole.expert_collectives());
    let alone = buckets("whole", &*whole, Split::WHOLE);
    drop(whole);

    let union = Split::new(UNION, 1).unwrap();
    let member = load(&TextPlacement {
        experts: union,
        ..TextPlacement::WHOLE
    });
    assert_declarations(&*member);
    assert!(member.expert_collectives());
    let shared = buckets("union rank 1 of 4", &*member, union);

    // The union's rank runs the exchange the whole rank does not, and the same products.
    let operators = |keys: &Keys| -> BTreeSet<&'static str> {
        keys.iter().map(|(operator, _)| *operator).collect()
    };
    let added: Vec<&str> = operators(&shared)
        .difference(&operators(&alone))
        .copied()
        .collect();
    assert_eq!(added, ["expert_exchange"]);

    // A rank with no tokens of its own joins a step: every layer's experts, and the n-gram
    // lookups of layers 1 and 14 over the union.
    let lent = exchange(8);
    let (joined, recording) = record(&axes(union), || member.join_expert_step(&lent));
    joined.expect("the join");
    let reads = recording
        .launches
        .iter()
        .filter(|(operator, _)| *operator == "embedding_mxfp8")
        .count();
    assert_eq!(reads, 2);
    let exchanges = recording
        .launches
        .iter()
        .filter(|(operator, _)| *operator == "expert_exchange")
        .count();
    assert_eq!(exchanges, 2 * 40);
    assert_declared("the join", &recording.launches.iter().cloned().collect());
}
