//! What a recording forward answers: the shapes and programs the launching forward of the same
//! rank would, with collectives that check their views and issue nothing.
//!
//! A recording forward has no device, so every view here addresses a counting arena that no memory
//! backs; nothing reads it.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use nn::family::Projection;
use nn::layers::{ParallelLMHead, UlyssesAttention, UlyssesSpec};
use nn::{Arena, Checkpoint, DType, Forward, Shape, Split, WeightLoader};
use ops::Op;
use ops::logits::VocabUnpad;

/// The programs a walk reported, as `(operator, key)` pairs.
#[derive(Default)]
struct Launches(Vec<(String, String)>);

impl ops::capture::Sink for Launches {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator.to_owned(), key.to_owned()));
    }
}

/// Runs `walk` with a capture sink installed and answers what it reported.
fn recorded<T>(walk: impl FnOnce() -> T) -> (T, Vec<(String, String)>) {
    let mut launches = Launches::default();
    ops::capture::set_sink(Some(&mut launches));
    let answer = walk();
    ops::capture::set_sink(None);
    (answer, launches.0)
}

/// A checkpoint whose `lm_head.weight` is `[vocabulary, 8]` BF16, in `directory`.
fn checkpoint(directory: &std::path::Path, vocabulary: u64) -> Checkpoint {
    let header = format!(
        r#"{{"lm_head.weight":{{"dtype":"BF16","shape":[{vocabulary},8],"data_offsets":[0,{}]}}}}"#,
        vocabulary * 16
    );
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.resize(file.len() + (vocabulary * 16) as usize, 0);
    std::fs::write(directory.join("model.safetensors"), file).unwrap();
    Checkpoint::open(directory).unwrap()
}

/// Rank `rank` of 2 of a head over `vocabulary` rows, read by a meta loader.
fn head(checkpoint: &Checkpoint, vocabulary: u32, rank: u32) -> nn::Result<ParallelLMHead> {
    let mut loader = WeightLoader::meta(checkpoint, None, None);
    ParallelLMHead::load(
        &mut loader.scope("lm_head"),
        vocabulary,
        8,
        nn::family::Split { size: 2, rank },
    )
}

/// A vocabulary-parallel head over two ranks gathers both shards and unpads the gathered width:
/// a recording forward split as rank 1 of 2 reserves the same gather and states the same unpad
/// program as that rank's launching forward.
#[test]
fn a_split_recording_answers_the_gathered_width_of_its_rank() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), 32);
    let arena = Arena::counting();
    let f = Forward::recording(&arena).with_split(1, 2).unwrap();
    assert_eq!(f.split(), Some(Split { rank: 1, ranks: 2 }));

    // Rank 1 holds rows 16..32, a whole number of the head's eight-row tiles.
    let head = head(&checkpoint, 32, 1).unwrap();
    assert_eq!(head.padded(), 16);
    let hidden = arena.tensor(DType::BF16, Shape::new(&[3, 8])).unwrap();

    let before = arena.used();
    let (answer, launches) = recorded(|| head.forward(&f, hidden));
    let logits = answer.expect("two shards of 16 gather into 32 columns");
    assert_eq!(
        (logits.shape.dims(), logits.dtype),
        (&[3, 32][..], DType::F32)
    );
    // The head reserved its shard's projection and the two shards' gather, BF16 each, and the
    // FP32 logits.
    assert!(arena.used() - before >= 3 * 16 * 2 + 3 * 32 * 2 + 3 * 32 * 4);
    // The unpad reads the gathered BF16 logits two shards wide and writes the vocabulary in FP32.
    assert!(
        launches
            .iter()
            .any(|(operator, key)| operator == <VocabUnpad as Op>::NAME
                && key.starts_with("dtype_bf16_output_f32_vocabulary_32_padded_32_")),
        "{launches:?}"
    );

    // Unsplit, the shard's 16 columns are the whole gathered width and cannot hold a vocabulary
    // of 32: the unpad refuses the call, as the device would.
    let unsplit = Forward::recording(&arena);
    assert!(head.forward(&unsplit, hidden).is_err());
}

/// A shard that is not a whole number of tiles has no padded rows in the checkpoint to read: the
/// next rank's rows follow it, so the head is refused at load rather than projected over them.
#[test]
fn a_shard_of_a_partial_tile_is_refused_at_load() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), 30);
    let refused = head(&checkpoint, 30, 1).err().expect("fifteen rows");
    assert!(refused.to_string().contains("15"), "{refused}");
}

/// A checkpoint whose `attn.to_q.weight` is `[64, 32]` and `attn.to_out.weight` `[32, 64]` BF16.
fn attention_checkpoint(directory: &std::path::Path) -> Checkpoint {
    let header = r#"{"attn.to_q.weight":{"dtype":"BF16","shape":[64,32],"data_offsets":[0,4096]},"attn.to_out.weight":{"dtype":"BF16","shape":[32,64],"data_offsets":[4096,8192]}}"#;
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.resize(file.len() + 8192, 0);
    std::fs::write(directory.join("model.safetensors"), file).unwrap();
    Checkpoint::open(directory).unwrap()
}

/// Member 1 of 2 of a four-head, sixteen-wide attention reaching its heads by `projection`.
fn ulysses(checkpoint: &Checkpoint, projection: Projection) -> UlyssesAttention<()> {
    let mut loader = WeightLoader::meta(checkpoint, None, None);
    UlyssesAttention::load(
        &mut loader.scope("attn"),
        UlyssesSpec {
            hidden: 32,
            heads: 4,
            head_dim: 16,
            inputs: &["to_q"],
            output: "to_out",
        },
        nn::family::Split { size: 2, rank: 1 },
        projection,
        |_| Ok(()),
    )
    .unwrap()
}

/// In the exchange mode a member projects its rows for each member's heads and exchanges the
/// blocks; in the gather mode it gathers every member's rows and projects its own heads. Either
/// way the return sets the members' blocks side by side before the output projection, and a
/// forward split over another member count is refused before any collective.
#[test]
fn a_sequence_split_projects_heads_by_its_mode_and_returns_rows() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = attention_checkpoint(directory.path());
    let arena = Arena::counting();
    let f = Forward::recording(&arena).with_split(1, 2).unwrap();
    let tensor = |dims: &[u64]| arena.tensor(DType::BF16, Shape::new(dims)).unwrap();
    // Four own rows of a sequence of eight; this member's heads are two of four, 32 channels.
    let normalized = tensor(&[4, 32]);

    let exchange = ulysses(&checkpoint, Projection::Exchange);
    let (projected, launches) = recorded(|| {
        let rows = exchange.rows(&f, normalized)?;
        exchange.project(&f, 0, rows)
    });
    let heads = projected.unwrap();
    assert_eq!(
        heads.shape.dims(),
        [8, 32],
        "every row of this member's heads"
    );
    let operators: Vec<&str> = launches.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(operators, ["gemm", "gemm"], "one product per member");

    let gather = ulysses(&checkpoint, Projection::Gather);
    let (projected, launches) = recorded(|| {
        let rows = gather.rows(&f, normalized)?;
        gather.project(&f, 0, rows)
    });
    assert_eq!(projected.unwrap().shape.dims(), [8, 32]);
    assert_eq!(
        launches.len(),
        1,
        "one product of this member's heads: {launches:?}"
    );

    let (returned, launches) = recorded(|| exchange.return_rows(&f, heads));
    assert_eq!(
        returned.unwrap().shape.dims(),
        [4, 32],
        "this member's rows of every head"
    );
    let operators: Vec<&str> = launches.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(operators[operators.len() - 1], "gemm");
    assert!(
        operators.len() == 2,
        "the merge and the output projection: {operators:?}"
    );

    let four = Forward::recording(&arena).with_split(1, 4).unwrap();
    let refused = exchange.project(&four, 0, normalized).unwrap_err();
    assert!(
        format!("{refused}").contains("built for 2 members"),
        "{refused}"
    );
}

/// Collectives are in place or shape-preserving, so a recording forward checks their views and
/// issues nothing: no program is reported and every view keeps its shape.
#[test]
fn collectives_on_a_recording_forward_report_nothing() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena).with_split(1, 2).unwrap();
    let rows = arena.tensor(DType::BF16, Shape::new(&[4, 64])).unwrap();
    let gathered = arena.tensor(DType::BF16, Shape::new(&[8, 64])).unwrap();
    let blocks = arena.tensor(DType::BF16, Shape::new(&[8, 64])).unwrap();

    let (answers, launches) = recorded(|| {
        [
            f.all_reduce(rows),
            f.all_gather(rows, gathered),
            f.all_to_all(gathered, blocks),
            f.start_all_to_all(gathered, blocks)
                .and_then(|transfer| f.finish(transfer)),
            f.start_gather_rows(gathered, 4, 0, 4)
                .and_then(|transfer| f.finish(transfer)),
        ]
    });
    for answer in answers {
        answer.unwrap();
    }
    assert!(launches.is_empty(), "{launches:?}");

    // The views a launching group would refuse are refused here too.
    assert!(
        f.all_gather(rows, rows).is_err(),
        "one rank's rows are not two ranks'"
    );
    let odd = arena.tensor(DType::BF16, Shape::new(&[3])).unwrap();
    assert!(
        f.all_to_all(odd, odd).is_err(),
        "three elements are not two equal blocks"
    );
}

/// A forward without a group or a split has no one to collect with, and says so.
#[test]
fn a_collective_needs_a_split() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    assert_eq!(f.split(), None);
    let rows = arena.tensor(DType::BF16, Shape::new(&[4, 64])).unwrap();
    assert!(f.all_reduce(rows).is_err());
    assert!(Forward::recording(&arena).with_split(2, 2).is_err());
    assert!(Forward::recording(&arena).with_split(0, 0).is_err());
}

/// A recording forward checks every call as the device would before it reports the program: a
/// call a launch would refuse is refused here with the same error, and no key is recorded for it.
#[test]
fn a_recording_forward_refuses_a_call_the_device_would_refuse() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let descriptor = ops::activation::SiluDescriptor {
        dtype: DType::BF16,
        columns: 1024,
        block: 1024,
        pdl: false,
    };
    let x = arena.tensor(DType::BF16, Shape::new(&[4, 1024])).unwrap();
    let short = arena.tensor(DType::BF16, Shape::new(&[3, 1024])).unwrap();

    let (answer, launches) = recorded(|| {
        f.operate::<ops::activation::Silu>(&descriptor, ops::activation::SiluArgs { x, out: short })
    });
    let refused = answer.unwrap_err().to_string();
    assert!(refused.contains("silu"), "{refused}");
    assert!(launches.is_empty(), "{launches:?}");

    let (answer, launches) = recorded(|| {
        f.operate::<ops::activation::Silu>(&descriptor, ops::activation::SiluArgs { x, out: x })
    });
    answer.unwrap();
    assert_eq!(launches.len(), 1);
}
