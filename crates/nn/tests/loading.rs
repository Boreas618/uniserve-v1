//! The loading boundary's behaviour: a checkpoint read through its index, its file identities,
//! the loader's conversion, fusion and key reuse, the packed fields of a calibrated export, and
//! a meta load against a load that reads the payload.
//!
//! These tests write their own shards, header and payload, so they need no checkpoint and no
//! device: weights are published to the CPU store, whose tensors hold what was published.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use std::collections::BTreeMap;
use std::path::Path;

use nn::quantization::{ModeloptManifest, Nvfp4Names, UNIFIED_EXPORT, ValueLayout};
use nn::weights::PartBlock;
use nn::{
    Checkpoint, CpuWeights, DType, Device, Library, MetaWeights, PipelineIndex, Shape, Source,
    Tensor, WeightKey, WeightLayout, WeightLoader, WeightSource,
};
use nn_testing::{Stored, encode, stored, values, write_checkpoint, write_file};
use ops::fp4::ScaleLayout;

/// The local directory `directory`.
fn local(directory: &Path) -> Source {
    Source::Local(directory.to_path_buf())
}

/// The BF16 values of the tensor `name` of `checkpoint`, as a read returns them.
fn read_bf16(checkpoint: &Checkpoint, name: &str) -> Vec<f32> {
    ops::values::decode(DType::BF16, &checkpoint.read(name).unwrap()).unwrap()
}

/// The whole of `tensor`, `shape` elements from its first.
fn region(tensor: &str, shape: impl Into<Shape>) -> WeightSource {
    WeightSource::Region {
        tensor: tensor.to_owned(),
        start: 0,
        shape: shape.into(),
    }
}

/// A plain load of `source` as `shape`, row-major, unpinned and not asked to hold a value.
fn load(loader: &WeightLoader<'_>, source: &WeightSource, shape: impl Into<Shape>) -> Tensor {
    loader
        .load(
            "weight",
            source,
            WeightLayout::ROW_MAJOR,
            &shape.into(),
            None,
            false,
        )
        .unwrap()
}

/// The bytes of `tensor`, a U8 tensor.
fn bytes(tensor: &Tensor) -> Vec<u8> {
    tensor.flatten_all().unwrap().to_vec1().unwrap()
}

/// The key among `keys` whose text holds `piece`.
fn key_of<'k>(keys: &'k [WeightKey], piece: &str) -> &'k WeightKey {
    keys.iter()
        .find(|key| key.to_string().contains(piece))
        .unwrap_or_else(|| panic!("no key holds `{piece}`: {keys:#?}"))
}

// ---- Checkpoints ------------------------------------------------------------------------------

#[test]
fn a_checkpoint_is_read_through_its_index() {
    let directory = tempfile::tempdir().unwrap();
    let numbers = [1.0_f32, 2.0, 3.0, 4.0];
    let first: &[Stored<&str>] = &[(
        "model.embed_tokens.weight",
        DType::BF16,
        vec![2, 2],
        encode(DType::BF16, &numbers),
    )];
    let second: &[Stored<&str>] = &[(
        "lm_head.weight",
        DType::BF16,
        vec![2, 2],
        encode(DType::BF16, &numbers),
    )];
    write_checkpoint(directory.path(), &[first, second]);

    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    assert_eq!(checkpoint.len(), 2);
    assert_eq!(checkpoint.files().len(), 2);

    let located = checkpoint.locate("model.embed_tokens.weight").unwrap();
    assert_eq!(
        (located.dtype, located.shape.dims()),
        (DType::BF16, &[2, 2][..])
    );
    assert_eq!(
        checkpoint.read("model.embed_tokens.weight").unwrap(),
        encode(DType::BF16, &numbers)
    );
    assert!(checkpoint.tensor("lm_head.weight").is_some());
    assert!(checkpoint.tensor("absent").is_none());

    // Two files of one checkpoint have different identities.
    let first_identity = checkpoint.identity("model.embed_tokens.weight").unwrap();
    let second_identity = checkpoint.identity("lm_head.weight").unwrap();
    assert_ne!(first_identity, second_identity);
}

/// A file is identified by where its caller states it comes from: a snapshot's file by the
/// repository, the commit and its path in the snapshot, whatever directory holds it; a local file
/// by its status, so two copies are two files and one file read twice is one.
#[test]
fn a_file_is_identified_by_the_source_its_caller_states() {
    let (first, second) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    for directory in [&first, &second] {
        write_checkpoint(
            directory.path(),
            &[&stored(DType::BF16, &[("weight", vec![2], vec![1.0, 2.0])])],
        );
    }
    let identity = |source: Source| {
        Checkpoint::shards(&source, Library::Transformers)
            .unwrap()
            .identity("weight")
            .unwrap()
            .clone()
    };
    let snapshot = |revision: &str, directory: &Path| {
        Source::snapshot("owner/model", revision, directory.to_path_buf())
    };

    // Two copies of one commit's snapshot hold the same files; another commit holds others, the
    // same bytes notwithstanding.
    assert_eq!(
        identity(snapshot("a1b2", first.path())),
        identity(snapshot("a1b2", second.path()))
    );
    assert_ne!(
        identity(snapshot("a1b2", first.path())),
        identity(snapshot("c3d4", first.path()))
    );

    // A local file just written is one file however often it is read, and its copy another.
    assert_eq!(identity(local(first.path())), identity(local(first.path())));
    assert_ne!(
        identity(local(first.path())),
        identity(local(second.path()))
    );
}

#[test]
fn a_reading_names_the_tensors_a_load_never_asked_for() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(
        directory.path(),
        &[&stored(
            DType::BF16,
            &[
                ("a", vec![2], vec![1.0, 2.0]),
                ("b", vec![2], vec![3.0, 4.0]),
                ("c", vec![2], vec![5.0, 6.0]),
            ],
        )],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    checkpoint.read("a").unwrap();
    assert_eq!(
        checkpoint.reading(),
        (vec!["a".to_owned()], vec!["b".to_owned(), "c".to_owned()])
    );

    // Reading the rest completes it.
    checkpoint.read("b").unwrap();
    checkpoint.read("c").unwrap();
    assert!(checkpoint.reading().1.is_empty());

    // A pipeline lists its stages' names qualified, and a stage that was not read whole.
    let root = tempfile::tempdir().unwrap();
    let (transformer, vae) = (root.path().join("transformer"), root.path().join("vae"));
    std::fs::create_dir_all(&transformer).unwrap();
    std::fs::create_dir_all(&vae).unwrap();
    write_checkpoint(
        &transformer,
        &[&stored(DType::BF16, &[("weight", vec![2], vec![1.0, 2.0])])],
    );
    write_checkpoint(
        &vae,
        &[&stored(DType::BF16, &[("weight", vec![2], vec![3.0, 4.0])])],
    );

    let stages = BTreeMap::from([
        (
            "transformer".to_owned(),
            Checkpoint::open(&transformer).unwrap(),
        ),
        ("vae".to_owned(), Checkpoint::open(&vae).unwrap()),
    ]);
    let pipeline = Checkpoint::compose(local(root.path()), stages, BTreeMap::new()).unwrap();
    assert!(pipeline.is_pipeline());
    pipeline
        .stage("transformer")
        .unwrap()
        .read("weight")
        .unwrap();
    assert_eq!(
        pipeline.reading(),
        (
            vec!["transformer/weight".to_owned()],
            vec!["vae/weight".to_owned()]
        )
    );
}

/// A shard set is read through the names its library declares, one shard or an index of
/// several; another file is no shard set, and a directory that states both is refused.
#[test]
fn a_shard_set_is_read_only_through_its_librarys_names() {
    let directory = tempfile::tempdir().unwrap();
    let tensors = [(
        "weight",
        DType::BF16,
        vec![2],
        encode(DType::BF16, &[1.0, 2.0]),
    )];

    write_file(&directory.path().join("weights.safetensors"), &[], &tensors);
    let error = Checkpoint::open(directory.path()).err().expect("refused");
    assert!(error.to_string().contains("model.safetensors"), "{error}");

    // A diffusers shard set is read by the diffusers names alone.
    write_file(
        &directory.path().join("diffusion_pytorch_model.safetensors"),
        &[],
        &tensors,
    );
    assert!(Checkpoint::open(directory.path()).is_err());
    let diffusers = Checkpoint::shards(&local(directory.path()), Library::Diffusers).unwrap();
    assert!(diffusers.tensor("weight").is_some());

    // One shard and an index of several are two statements of one set.
    write_checkpoint(directory.path(), &[&tensors]);
    std::fs::write(
        directory.path().join("model.safetensors.index.json"),
        r#"{"weight_map": {"weight": "model.safetensors"}}"#,
    )
    .unwrap();
    let error = Checkpoint::open(directory.path()).err().expect("refused");
    assert!(error.to_string().contains("twice"), "{error}");
}

/// A pipeline's stages are the components its index declares, each in the directory the index
/// names and read by its library's names; a directory the index does not declare is not one.
#[test]
fn a_pipelines_stages_are_the_components_its_index_declares() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("modular_model_index.json"),
        r#"{
            "_class_name": "Pipeline",
            "_diffusers_version": "0.36.0",
            "transformer": ["diffusers", "Network", {"subfolder": "network"}],
            "text_encoder": ["transformers", "Encoder"],
            "scheduler": ["diffusers", "Scheduler"],
            "tokenizer": [null, null]
        }"#,
    )
    .unwrap();
    for directory in ["network", "text_encoder", "scheduler", "extra"] {
        std::fs::create_dir_all(root.path().join(directory)).unwrap();
    }
    let tensors = |value: f32| {
        [(
            "weight",
            DType::BF16,
            vec![1],
            encode(DType::BF16, &[value]),
        )]
    };
    write_file(
        &root
            .path()
            .join("network")
            .join("diffusion_pytorch_model.safetensors"),
        &[],
        &tensors(1.0),
    );
    write_checkpoint(&root.path().join("text_encoder"), &[&tensors(2.0)]);
    write_checkpoint(&root.path().join("extra"), &[&tensors(3.0)]);

    let source = local(root.path());
    let index = PipelineIndex::read(&source).unwrap();
    assert_eq!(index.class(), "Pipeline");
    assert!(index.declares("transformer") && index.declares("scheduler"));
    assert!(!index.declares("tokenizer"), "declared absent");
    assert!(!index.declares("extra"), "a directory is not a declaration");

    let network = index.stage("transformer", &source).unwrap();
    assert_eq!(network.root(), root.path().join("network"));
    assert_eq!(read_bf16(&network, "weight"), [1.0]);
    assert!(index.stage("extra", &source).is_err());
    assert!(
        index.stage("scheduler", &source).is_err(),
        "it holds no shard set"
    );
    assert!(index.directory("scheduler", &source).is_ok());

    // A declared directory its source does not hold is refused, not skipped.
    let empty = tempfile::tempdir().unwrap();
    assert!(index.directory("scheduler", &local(empty.path())).is_err());
    let error = PipelineIndex::read(&local(empty.path())).unwrap_err();
    assert!(error.to_string().contains("model_index.json"), "{error}");
}

/// A composed pipeline reads each component's files from the one source it was composed from,
/// and its own statements from its root, whatever another directory holds.
#[test]
fn a_composed_pipeline_reads_each_component_from_its_source() {
    let (export, base) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    std::fs::write(
        export.path().join("model_index.json"),
        r#"{"_class_name": "Pipeline", "transformer": ["diffusers", "Network"],
            "text_encoder": ["transformers", "Encoder"], "scheduler": ["diffusers", "Scheduler"]}"#,
    )
    .unwrap();
    std::fs::write(export.path().join("manifest.json"), r#"{"from": "export"}"#).unwrap();
    for (root, directory) in [
        (export.path(), "transformer"),
        (export.path(), "text_encoder"),
        (base.path(), "text_encoder"),
        (base.path(), "scheduler"),
    ] {
        let directory = root.join(directory);
        std::fs::create_dir_all(&directory).unwrap();
        let from = if root == export.path() {
            "export"
        } else {
            "base"
        };
        std::fs::write(
            directory.join("config.json"),
            format!(r#"{{"from": "{from}"}}"#),
        )
        .unwrap();
    }
    let tensors = [("weight", DType::BF16, vec![1], encode(DType::BF16, &[1.0]))];
    write_file(
        &export
            .path()
            .join("transformer")
            .join("diffusion_pytorch_model.safetensors"),
        &[],
        &tensors,
    );
    write_checkpoint(&base.path().join("text_encoder"), &[&tensors]);

    let (export_source, base_source) = (local(export.path()), local(base.path()));
    let index = PipelineIndex::read(&export_source).unwrap();
    let stages = BTreeMap::from([
        (
            "transformer".to_owned(),
            index.stage("transformer", &export_source).unwrap(),
        ),
        (
            "text_encoder".to_owned(),
            index.stage("text_encoder", &base_source).unwrap(),
        ),
    ]);
    let directories = BTreeMap::from([(
        "scheduler".to_owned(),
        index.directory("scheduler", &base_source).unwrap(),
    )]);
    let pipeline = Checkpoint::compose(export_source.clone(), stages, directories).unwrap();

    let from = |relative: &str| pipeline.sidecar(relative).unwrap().unwrap()["from"].clone();
    assert_eq!(from("manifest.json"), "export");
    assert_eq!(from("transformer/config.json"), "export");
    assert_eq!(
        from("text_encoder/config.json"),
        "base",
        "not the export's stray copy"
    );
    assert_eq!(from("scheduler/config.json"), "base");

    // A component is a stage or a directory, not both.
    let twice = Checkpoint::compose(
        export_source.clone(),
        BTreeMap::from([(
            "scheduler".to_owned(),
            index.stage("transformer", &export_source).unwrap(),
        )]),
        BTreeMap::from([("scheduler".to_owned(), base_source.clone())]),
    );
    assert!(twice.is_err());
}

// ---- Dense loads ------------------------------------------------------------------------------

#[test]
fn a_region_is_converted_once_and_reused_by_its_key() {
    let directory = tempfile::tempdir().unwrap();
    let numbers: Vec<f32> = (0..8).map(|index| index as f32).collect();
    write_checkpoint(
        directory.path(),
        &[&stored(
            DType::BF16,
            &[("weight", vec![2, 4], numbers.clone())],
        )],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();

    // The launch states FP32, so the BF16 tensor is converted, once.
    let loader = WeightLoader::new(&checkpoint, &store, Some(DType::F32));
    let first = load(&loader, &region("weight", (2, 4)), (2, 4));
    assert_eq!((first.dtype(), first.dims()), (DType::F32, &[2, 4][..]));
    assert_eq!(values(&first), numbers);
    assert_eq!(store.keys().len(), 1);

    // The same weight is served from retained storage by another loader: no second publish.
    let loader = WeightLoader::new(&checkpoint, &store, Some(DType::F32));
    let second = load(&loader, &region("weight", (2, 4)), (2, 4));
    assert_eq!(values(&second), numbers);
    assert_eq!(store.keys().len(), 1);

    // The key names the file, the retained element type, the layout and the region.
    let file = checkpoint.identity("weight").unwrap();
    assert_eq!(
        store.keys()[0].to_string(),
        format!("{file}|f32|row_major|weight[0..8][2, 4]")
    );

    // A window of its rows is another region, and a pin another element type.
    let rows = loader.root().get_rows((1, 4), "weight", 1).unwrap();
    assert_eq!(values(&rows), numbers[4..]);
    let pinned = loader
        .root()
        .to_dtype(DType::BF16)
        .get((2, 4), "weight")
        .unwrap();
    assert_eq!(pinned.dtype(), DType::BF16);
    assert_eq!(store.keys().len(), 3);
}

#[test]
fn fused_parts_are_joined_in_the_order_the_operator_reads() {
    let directory = tempfile::tempdir().unwrap();
    let gate: Vec<f32> = (0..4).map(|index| 10.0 + index as f32).collect();
    let up: Vec<f32> = (0..4).map(|index| 20.0 + index as f32).collect();
    write_checkpoint(
        directory.path(),
        &[&stored(
            DType::BF16,
            &[
                ("gate", vec![2, 2], gate.clone()),
                ("up", vec![2, 2], up.clone()),
            ],
        )],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, Some(DType::F32));

    // `up` before `gate`, as `WeightLayout::UP_GATE` declares.
    let parts = vec!["up".to_owned(), "gate".to_owned()];
    let fused = WeightSource::Parts {
        parts: parts.clone(),
        shape: Shape::from((4, 2)),
        block: None,
    };
    let weight = loader
        .load(
            "experts.up_gate",
            &fused,
            WeightLayout::UP_GATE,
            &Shape::from((4, 2)),
            None,
            false,
        )
        .unwrap();
    assert_eq!(weight.dims(), [4, 2]);
    assert_eq!(values(&weight), [up.clone(), gate.clone()].concat());
    let key = store.keys()[0].to_string();
    assert!(key.contains("|up_gate|up+gate[0..8][4, 2]"), "{key}");

    // A rank's block of each part: the second column of both, under a key of its own.
    let block = WeightSource::Parts {
        parts,
        shape: Shape::from((4, 1)),
        block: Some(PartBlock {
            start: 1,
            rows: 2,
            columns: 1,
            row_stride: 2,
        }),
    };
    let shard = loader
        .load(
            "experts.up_gate",
            &block,
            WeightLayout::UP_GATE,
            &Shape::from((4, 1)),
            None,
            false,
        )
        .unwrap();
    assert_eq!(values(&shard), [21.0, 23.0, 11.0, 13.0]);
    assert!(
        store
            .keys()
            .iter()
            .any(|key| key.to_string().ends_with("up+gate[1..5][4, 1]|stride_2"))
    );

    // A layout that fuses projections cannot be produced from one region.
    let single = loader.load(
        "gate",
        &region("gate", (2, 2)),
        WeightLayout::UP_GATE,
        &Shape::from((2, 2)),
        None,
        false,
    );
    assert!(single.is_err());
    assert_eq!(store.keys().len(), 2);
}

#[test]
fn a_strided_row_parallel_shard_gathers_its_columns() {
    let directory = tempfile::tempdir().unwrap();
    // Four rows of six elements: row r holds 6r .. 6r+5.
    let numbers: Vec<f32> = (0..24).map(|value| value as f32).collect();
    write_checkpoint(
        directory.path(),
        &[&stored(
            DType::BF16,
            &[("o_proj.weight", vec![4, 6], numbers)],
        )],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    let columns = |first: u64| WeightSource::Strided {
        tensor: "o_proj.weight".to_owned(),
        start: first,
        shape: Shape::from((4, 3)),
        row_stride: 6,
    };
    let weight = load(&loader, &columns(0), (4, 3));
    assert_eq!(
        values(&weight),
        [
            0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 12.0, 13.0, 14.0, 18.0, 19.0, 20.0
        ],
        "columns 0..2 of every row, rows contiguous"
    );

    // The same source reuses the retained weight; a second column block of the same extents is
    // another key.
    load(&loader, &columns(0), (4, 3));
    assert_eq!(store.keys().len(), 1);
    load(&loader, &columns(3), (4, 3));
    assert_eq!(store.keys().len(), 2);
}

/// A tensor whose every stored byte is zero is a missing export, not a value, when the load
/// asks for a value: the check reads the region the caller asked for, so a rank whose own block
/// is zero is refused even when the rest of the matrix holds values.
#[test]
fn a_region_of_zeros_is_refused_when_the_load_asks_for_a_value() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(
        directory.path(),
        &[&stored(
            DType::BF16,
            &[
                ("v_proj.weight", vec![2, 2], vec![0.0; 4]),
                ("k_proj.weight", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            ],
        )],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let asked = |loader: &WeightLoader<'_>, tensor: &str, nonzero: bool| {
        loader.load(
            tensor,
            &region(tensor, (2, 2)),
            WeightLayout::ROW_MAJOR,
            &Shape::from((2, 2)),
            None,
            nonzero,
        )
    };

    let text = asked(&loader, "v_proj.weight", true)
        .unwrap_err()
        .to_string();
    assert!(text.contains("v_proj.weight"), "{text}");
    assert!(
        text.contains("every stored byte of the region is zero"),
        "{text}"
    );

    // Without the question the same bytes load: a zero norm's scale and a zero bias are values.
    asked(&loader, "v_proj.weight", false).unwrap();
    asked(&loader, "k_proj.weight", true).unwrap();

    // A meta loader reads no payload, so the zeros are not there for it to refuse.
    let store = MetaWeights::new();
    let meta = WeightLoader::new(&checkpoint, &store, None);
    asked(&meta, "v_proj.weight", true).unwrap();
}

#[test]
fn a_weight_normalized_convolution_is_computed_once_at_load() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(
        directory.path(),
        &[&stored(
            DType::BF16,
            &[
                // Two output channels of two elements: v = [[3, 4], [0, 1]].
                ("conv.weight_v", vec![2, 2], vec![3.0, 4.0, 0.0, 1.0]),
                // PyTorch keeps the gain as [channels, 1, 1].
                ("conv.weight_g", vec![2, 1, 1], vec![5.0, 10.0]),
                ("short.weight_v", vec![2, 2], vec![3.0, 4.0, 0.0, 1.0]),
                ("short.weight_g", vec![3, 1, 1], vec![5.0, 10.0, 1.0]),
            ],
        )],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    // v · g / ‖v‖ per channel: [3, 4] · 5/5 and [0, 1] · 10/1.
    let weight = loader.root().pp("conv").weight_norm("weight").unwrap();
    assert_eq!((weight.dtype(), weight.dims()), (DType::BF16, &[2, 2][..]));
    assert_eq!(values(&weight), [3.0, 4.0, 0.0, 10.0]);

    // A second load reuses the retained weight, under a key that names both sources.
    loader.root().pp("conv").weight_norm("weight").unwrap();
    assert_eq!(store.keys().len(), 1);
    let key = store.keys()[0].to_string();
    assert!(
        key.contains("conv.weight__weight_norm+conv.weight_v+conv.weight_g"),
        "{key}"
    );

    // A gain that is not one magnitude per output channel is refused before anything is read.
    assert!(loader.root().pp("short").weight_norm("weight").is_err());
    assert_eq!(store.keys().len(), 1);
}

// ---- Computed constants -----------------------------------------------------------------------

/// A computed weight is named by what computed it, not by the label its model chose: two
/// deployments that both call their fused output heads one name and compute different numbers
/// are two weights.
#[test]
fn a_computed_weight_is_named_by_its_provenance_and_not_by_its_label_alone() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(
        directory.path(),
        &[&stored(DType::BF16, &[("weight", vec![1], vec![1.0])])],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    let (name, shape) = ("denoiser.video.student_heads.weight", Shape::from(2));
    let (base, reference) = (
        encode(DType::F32, &[0.5, 0.25]),
        encode(DType::F32, &[1.0, 2.0]),
    );
    let constant = |bytes: &[u8], producer: &'static str| {
        loader
            .constant(name, DType::F32, &shape, bytes, producer)
            .unwrap()
    };

    let first = constant(&base, "transformer");
    let second = constant(&reference, "transformer_ref");
    assert_eq!(values(&first), [0.5, 0.25]);
    assert_eq!(values(&second), [1.0, 2.0]);
    assert_eq!(store.keys().len(), 2);
    assert!(
        store.keys()[0]
            .to_string()
            .contains("|computed:transformer")
    );
    assert!(store.keys()[0].files.is_empty());

    // The same numbers by the same class are one weight: a later load finds what the first
    // published instead of computing it again.
    constant(&base, "transformer");
    assert_eq!(store.keys().len(), 2);

    // The same numbers by another class are another weight: the producer is part of the
    // provenance even when the bytes agree.
    constant(&base, "transformer_ref");
    assert_eq!(store.keys().len(), 3);

    // A byte count that does not match the extents is refused before anything is published.
    let short = loader.constant(name, DType::F32, &shape, &base[..4], "transformer");
    assert!(short.is_err());
    assert_eq!(store.keys().len(), 3);
}

#[test]
fn a_builder_names_its_constants_by_its_module_path_and_encodes_their_type() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(
        directory.path(),
        &[&stored(DType::BF16, &[("weight", vec![1], vec![1.0])])],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let vb = loader.root().pp("model").pp("rotary_emb");
    assert_eq!(vb.path(), "model.rotary_emb");
    assert_eq!(vb.name("inv_freq"), "model.rotary_emb.inv_freq");
    assert!(!vb.contains("inv_freq") && loader.root().contains("weight"));

    let table = vb
        .constant("inv_freq", DType::BF16, 2, &[0.5, 0.25])
        .unwrap();
    assert_eq!(
        (table.dtype(), values(&table)),
        (DType::BF16, vec![0.5, 0.25])
    );

    // An index row and a table of zeros are CPU tensors published as they are.
    let rows = vb
        .constant_from("rows", &Tensor::arange(3_u32, 5, &Device::Cpu).unwrap())
        .unwrap();
    assert_eq!(rows.dtype(), DType::U32);
    assert_eq!(rows.to_vec1::<u32>().unwrap(), [3, 4]);
    let pad = vb
        .constant_from(
            "pad",
            &Tensor::zeros((1, 3), DType::F32, &Device::Cpu).unwrap(),
        )
        .unwrap();
    assert_eq!(pad.dims(), [1, 3]);

    // A value count that is not the shape's is refused.
    assert!(vb.constant("short", DType::F32, 3, &[1.0]).is_err());

    // A writable table is this load's own: it is published under no retained key.
    let zeros = vb.writable_zeros("layers", DType::BF16, (2, 2)).unwrap();
    assert_eq!(values(&zeros), [0.0; 4]);

    let keys = store.keys();
    assert_eq!(keys.len(), 3);
    for (name, dtype) in [("inv_freq", "bf16"), ("rows", "u32"), ("pad", "f32")] {
        let key = key_of(&keys, &format!("model.rotary_emb.{name}"));
        // The producer of a module's constants is part of every stored key of one.
        assert!(
            key.to_string()
                .starts_with(&format!("|{dtype}|row_major|model.rotary_emb.{name}[")),
            "{key}"
        );
        assert!(
            key.to_string().contains("|computed:nn::weights::Scope:"),
            "{key}"
        );
    }
}

// ---- Calibrated NVFP4 fields ------------------------------------------------------------------

/// A calibrated NVFP4 matrix is its two byte fields and its tensor scale, published under keys
/// that name the scale layout; nothing is quantized at load.
#[test]
fn a_calibrated_nvfp4_matrix_is_loaded_as_its_fields() {
    let directory = tempfile::tempdir().unwrap();
    // `[4, 64]` encoded: 32 bytes a row of packed pairs, four block scales a row.
    let packed: Vec<u8> = (0..128).map(|byte| byte as u8).collect();
    let scales: Vec<u8> = (0..16).map(|byte| 0x30 + byte).collect();
    write_checkpoint(
        directory.path(),
        &[&[
            ("w", DType::U8, vec![4, 32], packed.clone()),
            ("w_scale", DType::U8, vec![4, 4], scales.clone()),
            ("w_input", DType::F32, vec![1], encode(DType::F32, &[0.5])),
        ]],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let fields = |layouts| {
        loader.nvfp4(
            &region("w", (4, 32)),
            &region("w_scale", (4, 4)),
            Some("w_input"),
            UNIFIED_EXPORT,
            layouts,
            0.25,
        )
    };

    let linear = fields((ScaleLayout::Linear, ScaleLayout::Linear)).unwrap();
    assert_eq!(
        (linear.packed.dims(), linear.packed.dtype()),
        (&[4, 32][..], DType::U8)
    );
    assert_eq!(bytes(&linear.packed), packed);
    assert_eq!(bytes(&linear.scales), scales);
    assert_eq!(
        (linear.global_scale, linear.layout),
        (0.25, ScaleLayout::Linear)
    );

    // The encoder's operand is the reciprocal of the calibrated activation scale.
    assert_eq!(values(linear.input_scale.as_ref().unwrap()), [2.0]);

    // The layout names the arrangement, so the fields never share a key with a plain region.
    let keys = store.keys();
    assert_eq!(key_of(&keys, "|w[").layout, WeightLayout::NVFP4_VALUES);
    assert_eq!(
        key_of(&keys, "|w_scale[").layout,
        WeightLayout::NVFP4_SCALES_LINEAR
    );
    key_of(&keys, "w_input__nvfp4_input_scale");

    // Asked in the product's 128x4 table, the same scales are permuted once: row `r`, column
    // `c` of the linear table lands at `c + 16 r`, and the padding is zero.
    let swizzled = fields((ScaleLayout::Linear, ScaleLayout::Swizzled128X4)).unwrap();
    assert_eq!(swizzled.scales.dims(), [128, 4]);
    let table = bytes(&swizzled.scales);
    for row in 0..4 {
        for column in 0..4 {
            assert_eq!(table[column + 16 * row], scales[row * 4 + column]);
        }
    }
    assert_eq!(table.iter().filter(|byte| **byte != 0).count(), 16);
    key_of(
        &store.keys(),
        "nvfp4_scales_swizzled_128x4|w_scale__nvfp4_scales",
    );

    // Block scales are the shape their file's layout names; another layout's shape is refused,
    // because the two arrangements index the same bytes differently.
    let error = fields((ScaleLayout::Swizzled128X4, ScaleLayout::Swizzled128X4)).unwrap_err();
    assert!(error.to_string().contains("swizzled_128x4"), "{error}");
}

/// A checkpoint opened in one export's spelling and written in the other's is refused by the
/// spelling, not as a missing tensor.
#[test]
fn a_calibrated_projection_is_read_in_the_spelling_its_export_wrote() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(
        directory.path(),
        &[&[
            ("proj.weight_packed", DType::U8, vec![2, 8], vec![1; 16]),
            ("proj.weight_scale", DType::U8, vec![2, 1], vec![0x38; 2]),
        ]],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let fields = |names: Nvfp4Names| {
        loader.nvfp4(
            &region(&format!("proj.{}", names.values), (2, 8)),
            &region(&format!("proj.{}", names.block_scale), (2, 1)),
            None,
            names,
            (ScaleLayout::Linear, ScaleLayout::Linear),
            1.0,
        )
    };

    fields(ModeloptManifest::NAMES).expect("the packed export's spelling names its packed values");
    let error = fields(UNIFIED_EXPORT).unwrap_err().to_string();
    assert!(
        error.contains(UNIFIED_EXPORT.export) && error.contains("packed E2M1 values"),
        "{error}"
    );
}

/// The shard of a stack of `experts` calibrated experts, `[2, 64]` each, with an `up` and a
/// `gate` projection, block scales of `scale` elements and a tensor scale of `0.5 · (e + 1)`;
/// returns the packed bytes in the order the stack reads them.
fn stack_checkpoint(directory: &Path, experts: u8, scale: DType) -> (Checkpoint, Vec<u8>) {
    let mut tensors: Vec<(String, DType, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut expected = Vec::new();

    for expert in 0..experts {
        for (index, projection) in ["up", "gate"].iter().enumerate() {
            let stem = format!("experts.{expert}.{projection}_proj");
            let packed: Vec<u8> = (0..64)
                .map(|byte| byte as u8 + expert * 64 + index as u8)
                .collect();
            let scales: Vec<u8> = (0..8)
                .map(|byte| 0x20 + byte as u8 + expert * 8 + index as u8)
                .collect();

            expected.extend_from_slice(&packed);
            tensors.push((format!("{stem}.weight"), DType::U8, vec![2, 32], packed));
            tensors.push((format!("{stem}.weight_scale"), scale, vec![2, 4], scales));
            tensors.push((
                format!("{stem}.weight_scale_2"),
                DType::F32,
                vec![1],
                encode(DType::F32, &[0.5 * f32::from(expert + 1)]),
            ));
        }
    }

    let named: Vec<(&str, DType, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| (name.as_str(), *dtype, shape.clone(), bytes.clone()))
        .collect();
    write_checkpoint(directory, &[&named]);
    (Checkpoint::open(directory).unwrap(), expected)
}

/// An expert stack is its experts' encoded tensors concatenated in the operator's reading
/// order, expert-major, up before gate, and its tensor scales one per expert **and
/// projection**, published the other way round: one packed table per projection, which is what
/// the batched-GEMM entry reads.
#[test]
fn a_calibrated_expert_stack_is_joined_expert_major_and_projection_by_projection() {
    let directory = tempfile::tempdir().unwrap();
    let (checkpoint, expected) = stack_checkpoint(directory.path(), 2, DType::U8);
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let stack = |loader: &WeightLoader<'_>| {
        loader
            .nvfp4_stack(
                "layer.0.mlp.experts.up_gate",
                "experts",
                &["up_proj", "gate_proj"],
                &[0, 1],
                2,
                UNIFIED_EXPORT,
                (ValueLayout::Plain, ScaleLayout::Linear),
                false,
            )
            .unwrap()
    };

    let (loaded, scales) = stack(&loader);
    assert_eq!(
        (loaded.values.dims(), loaded.values.dtype()),
        (&[2, 4, 32][..], DType::U8)
    );
    assert_eq!(bytes(&loaded.values), expected);
    assert_eq!(loaded.scales.dims(), [2, 4, 4]);

    // The scales as the loader read them are expert-major; the published table is one packed
    // row per projection: up's two experts, then gate's.
    assert_eq!(scales, [0.5, 0.5, 1.0, 1.0]);
    assert_eq!(loaded.global_scale.dims(), [2, 2]);
    assert_eq!(values(&loaded.global_scale), [0.5, 1.0, 0.5, 1.0]);

    // A stack that encodes nothing multiplies by one, published so the encoder reads a tensor.
    assert_eq!(values(&loaded.encode_scale), [1.0]);
    assert!(loaded.input_scale.is_none());

    let keys = store.keys();
    assert_eq!(keys.len(), 4);
    assert_eq!(
        key_of(&keys, "layer.0.mlp.experts.up_gate__nvfp4_values").layout,
        WeightLayout::NVFP4_VALUES
    );
    key_of(&keys, "layer.0.mlp.experts.up_gate__nvfp4_scales");
    key_of(&keys, "layer.0.mlp.experts.up_gate__nvfp4_tensor_scales");
    assert!(
        key_of(&keys, "layer.0.mlp.experts.up_gate__nvfp4_encode_scale")
            .to_string()
            .contains("|computed:nn:")
    );

    // A second load of the same checkpoint publishes nothing: the four keys are held.
    stack(&loader);
    assert_eq!(store.keys().len(), 4);

    // A redistribution's order is a load of the same checkpoint: slot 0 holds expert 1.
    let (swapped, _) = loader
        .nvfp4_stack(
            "layer.0.mlp.experts.up_gate",
            "experts",
            &["up_proj", "gate_proj"],
            &[1, 0],
            2,
            UNIFIED_EXPORT,
            (ValueLayout::Plain, ScaleLayout::Linear),
            false,
        )
        .unwrap();
    assert_eq!(bytes(&swapped.values)[..128], expected[128..]);
}

/// A stack keeps the checkpoint's own block-scale type, and a shuffled stack permutes its
/// values' rows inside its shuffle blocks at load, under a layout name of its own.
#[test]
fn a_stack_keeps_its_scale_type_and_arranges_its_rows_at_load() {
    let directory = tempfile::tempdir().unwrap();
    let (checkpoint, _) = stack_checkpoint(directory.path(), 1, DType::F8E4M3);
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let stack = |values: ValueLayout| {
        loader.nvfp4_stack(
            "experts.up_gate",
            "experts",
            &["up_proj", "gate_proj"],
            &[0],
            1,
            UNIFIED_EXPORT,
            (values, ScaleLayout::Swizzled128X4),
            false,
        )
    };

    let (plain, _) = stack(ValueLayout::Plain).unwrap();
    assert_eq!(plain.scales.dtype(), DType::F8E4M3);
    assert_eq!(
        plain.scales.dims(),
        [1, 128, 4],
        "padded to the swizzled table"
    );

    // Four rows are not a whole sixteen-row shuffle block: the arrangement does not exist.
    let error = stack(ValueLayout::Shuffled {
        epilogue_tile_m: 64,
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("16-row blocks"), "{error}");
}

/// A stack whose experts disagree on their extents is refused, not read as one stack.
#[test]
fn a_stack_needs_every_expert_to_agree_on_its_extents() {
    let directory = tempfile::tempdir().unwrap();
    let zeros = |count: usize| vec![0_u8; count];
    write_checkpoint(
        directory.path(),
        &[&[
            (
                "experts.0.up_proj.weight",
                DType::U8,
                vec![2, 32],
                zeros(64),
            ),
            (
                "experts.0.up_proj.weight_scale",
                DType::U8,
                vec![2, 4],
                zeros(8),
            ),
            (
                "experts.0.up_proj.weight_scale_2",
                DType::F32,
                vec![1],
                encode(DType::F32, &[1.0]),
            ),
            (
                "experts.1.up_proj.weight",
                DType::U8,
                vec![4, 32],
                zeros(128),
            ),
            (
                "experts.1.up_proj.weight_scale",
                DType::U8,
                vec![4, 4],
                zeros(16),
            ),
            (
                "experts.1.up_proj.weight_scale_2",
                DType::F32,
                vec![1],
                encode(DType::F32, &[1.0]),
            ),
        ]],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);

    let error = loader
        .nvfp4_stack(
            "layer.0.mlp.experts.up",
            "experts",
            &["up_proj"],
            &[0, 1],
            2,
            UNIFIED_EXPORT,
            (ValueLayout::Plain, ScaleLayout::Linear),
            false,
        )
        .expect_err("two experts of different shapes are not one stack");
    assert!(error.to_string().contains("against the stack's"), "{error}");
    assert!(store.keys().is_empty());
}

// ---- Stages and policies ----------------------------------------------------------------------

/// A pipeline's stages keep their indexes apart; one loader reborrows another checkpoint and
/// keeps the store, so the two stages share the conversion path and its retained weights.
#[test]
fn a_loader_borrows_another_stage_s_checkpoint_and_keeps_the_store() {
    let root = tempfile::tempdir().unwrap();
    let (transformer, vae) = (root.path().join("transformer"), root.path().join("vae"));
    std::fs::create_dir_all(&transformer).unwrap();
    std::fs::create_dir_all(&vae).unwrap();
    write_checkpoint(
        &transformer,
        &[&stored(DType::BF16, &[("weight", vec![2], vec![1.0, 2.0])])],
    );
    write_checkpoint(
        &vae,
        &[&stored(DType::BF16, &[("weight", vec![2], vec![1.0, 2.0])])],
    );

    let transformer = Checkpoint::open(&transformer).unwrap();
    let vae = Checkpoint::open(&vae).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&transformer, &store, Some(DType::F32));

    let first = load(&loader, &region("weight", 2), 2);
    let second = load(&loader.reborrow(&vae).unwrap(), &region("weight", 2), 2);
    assert_eq!((first.dtype(), second.dtype()), (DType::F32, DType::F32));

    // Read again through the first loader: the store holds it, nothing is published a third
    // time.
    load(&loader, &region("weight", 2), 2);

    // Same tensor name, same bytes, two stages: the keys differ by the file that holds them.
    let keys = store.keys();
    assert_eq!(keys.len(), 2);
    assert_ne!(keys[0].files, keys[1].files);

    // A policy that does not describe a stage is refused when the loader is borrowed for it:
    // an encoded stage is not read at a stated floating-point type.
    let encoded = root.path().join("encoded");
    std::fs::create_dir_all(&encoded).unwrap();
    write_checkpoint(
        &encoded,
        &[&[("weight", DType::F8E4M3, vec![2], vec![0x38; 2])]],
    );
    let encoded = Checkpoint::open(&encoded).unwrap();
    let refused = loader.reborrow(&encoded).err().unwrap();
    assert!(matches!(refused, nn::Error::Policy { .. }), "{refused}");
}

// ---- Meta loads -------------------------------------------------------------------------------

/// A dense checkpoint: a projection, a row-parallel matrix, two fusable parts and a
/// weight-normalized pair, all BF16 with values in every byte.
fn dense_checkpoint(directory: &Path) -> Checkpoint {
    let numbers = |count: usize| (1..=count).map(|value| value as f32).collect::<Vec<_>>();
    write_checkpoint(
        directory,
        &[&stored(
            DType::BF16,
            &[
                ("proj.weight", vec![4, 8], numbers(32)),
                ("rows.weight", vec![4, 8], numbers(32)),
                ("experts.0.up", vec![2, 8], numbers(16)),
                ("experts.0.gate", vec![2, 8], numbers(16)),
                ("conv.weight_v", vec![2, 3, 1], numbers(6)),
                ("conv.weight_g", vec![2, 1, 1], numbers(2)),
            ],
        )],
    );
    Checkpoint::open(directory).unwrap()
}

/// Every kind of dense load, in one order: a pinned region, a strided column block, fused
/// parts, a tensor-parallel block of each part, a weight-normalized weight and a computed
/// constant; each as its element type and extents.
fn dense_loads(loader: &WeightLoader<'_>) -> Vec<(DType, Vec<usize>)> {
    let parts = vec!["experts.0.up".to_owned(), "experts.0.gate".to_owned()];
    let fused = |shape: (usize, usize), block| {
        loader
            .load(
                "experts.0.up_gate",
                &WeightSource::Parts {
                    parts: parts.clone(),
                    shape: Shape::from(shape),
                    block,
                },
                WeightLayout::UP_GATE,
                &Shape::from(shape),
                None,
                false,
            )
            .unwrap()
    };

    let loaded = [
        loader
            .load(
                "proj.weight",
                &region("proj.weight", (4, 8)),
                WeightLayout::ROW_MAJOR,
                &Shape::from((4, 8)),
                Some(DType::F32),
                true,
            )
            .unwrap(),
        load(
            loader,
            &WeightSource::Strided {
                tensor: "rows.weight".to_owned(),
                start: 4,
                shape: Shape::from((4, 4)),
                row_stride: 8,
            },
            (4, 4),
        ),
        fused((4, 8), None),
        fused(
            (4, 4),
            Some(PartBlock {
                start: 4,
                rows: 2,
                columns: 4,
                row_stride: 8,
            }),
        ),
        loader
            .weight_norm("conv.weight", "conv.weight_v", "conv.weight_g", None)
            .unwrap(),
        loader
            .constant(
                "table",
                DType::I32,
                &Shape::from(2),
                &[1, 0, 0, 0, 2, 0, 0, 0],
                "test",
            )
            .unwrap(),
    ];
    loaded
        .iter()
        .map(|tensor| (tensor.dtype(), tensor.dims().to_vec()))
        .collect()
}

/// A meta load of a dense checkpoint returns the element type and extents a load that reads
/// the payload returns for every kind of source, as meta tensors, and reads no tensor of the
/// checkpoint.
#[test]
fn a_meta_load_returns_what_a_load_of_the_payload_returns() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = dense_checkpoint(directory.path());

    let store = MetaWeights::new();

    let meta = WeightLoader::new(&checkpoint, &store, None);
    let from_headers = dense_loads(&meta);
    assert_eq!(
        checkpoint.reading().0,
        Vec::<String>::new(),
        "a meta load reads no tensor"
    );

    let store = CpuWeights::new();
    let from_payload = dense_loads(&WeightLoader::new(&checkpoint, &store, None));
    assert_eq!(from_headers, from_payload);
    assert_eq!(
        from_headers,
        [
            (DType::F32, vec![4, 8]),
            (DType::BF16, vec![4, 4]),
            (DType::BF16, vec![4, 8]),
            (DType::BF16, vec![4, 4]),
            (DType::BF16, vec![2, 3, 1]),
            (DType::I32, vec![2]),
        ]
    );

    // A meta weight is a tensor of the meta device with no data to read, and a key asked for
    // twice is the same tensor, as a retained store returns a key it holds.
    let weight = load(&meta, &region("proj.weight", (4, 8)), (4, 8));
    assert!(!matches!(weight.device(), Device::Cpu));
    assert!(
        weight.to_vec2::<f32>().is_err(),
        "a meta tensor has no data"
    );

    let again = load(&meta, &region("proj.weight", (4, 8)), (4, 8));
    let address = |tensor: &Tensor| ops::operand(tensor).unwrap().ptr.address();
    assert_eq!(address(&weight), address(&again));
    let other = load(&meta, &region("rows.weight", (4, 8)), (4, 8));
    assert_ne!(address(&weight), address(&other));

    // A loader borrowed for another checkpoint shares the meta tensors.
    let borrowed = meta.reborrow(&checkpoint).unwrap();
    let shared = load(&borrowed, &region("proj.weight", (4, 8)), (4, 8));
    assert_eq!(address(&weight), address(&shared));
}

/// What the header contradicts is refused by a meta load with the other load's own refusal: a
/// region past its tensor, parts of different tails, a gain of the wrong length and an expert
/// stack whose block scales are not the linear table of its values.
#[test]
fn a_meta_load_refuses_what_the_header_contradicts() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = dense_checkpoint(directory.path());
    let store = CpuWeights::new();
    let payload = WeightLoader::new(&checkpoint, &store, None);
    let store = MetaWeights::new();
    let meta = WeightLoader::new(&checkpoint, &store, None);

    let past = region("proj.weight", (8, 8));
    let tails = WeightSource::Parts {
        parts: vec!["experts.0.up".to_owned(), "conv.weight_v".to_owned()],
        shape: Shape::from((4, 8)),
        block: None,
    };
    for (source, shape, layout) in [
        (&past, (8, 8), WeightLayout::ROW_MAJOR),
        (&tails, (4, 8), WeightLayout::UP_GATE),
    ] {
        let refusal = |loader: &WeightLoader<'_>| {
            loader
                .load("weight", source, layout, &Shape::from(shape), None, false)
                .unwrap_err()
                .to_string()
        };
        assert_eq!(refusal(&meta), refusal(&payload));
    }

    let gain = |loader: &WeightLoader<'_>| {
        loader
            .weight_norm("conv.weight", "conv.weight_v", "proj.weight", None)
            .unwrap_err()
            .to_string()
    };
    assert_eq!(gain(&meta), gain(&payload));

    // A stack whose block scales are stated as wide as the packed values: the published table
    // is the loader's to build, so the header alone refuses it.
    let encoded = tempfile::tempdir().unwrap();
    write_checkpoint(
        encoded.path(),
        &[&[
            (
                "experts.0.up_proj.weight",
                DType::U8,
                vec![2, 32],
                vec![1; 64],
            ),
            (
                "experts.0.up_proj.weight_scale",
                DType::U8,
                vec![2, 32],
                vec![1; 64],
            ),
            (
                "experts.0.up_proj.weight_scale_2",
                DType::F32,
                vec![1],
                encode(DType::F32, &[1.0]),
            ),
        ]],
    );
    let checkpoint = Checkpoint::open(encoded.path()).unwrap();
    let stack = |loader: &WeightLoader<'_>| {
        loader
            .nvfp4_stack(
                "experts.up",
                "experts",
                &["up_proj"],
                &[0],
                1,
                UNIFIED_EXPORT,
                (ValueLayout::Plain, ScaleLayout::Linear),
                false,
            )
            .expect_err("block scales of another table")
            .to_string()
    };
    let (store, meta) = (CpuWeights::new(), MetaWeights::new());
    let refused = stack(&WeightLoader::new(&checkpoint, &meta, None));
    assert_eq!(
        refused,
        stack(&WeightLoader::new(&checkpoint, &store, None))
    );
    assert!(refused.contains("linear"), "{refused}");
}

/// A tensor scale is read and refused by its value over every store: a zero scale encodes no
/// product, whether the load publishes its weights or states meta tensors.
#[test]
fn a_tensor_scale_is_refused_by_its_value_over_every_store() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(
        directory.path(),
        &[&[(
            "proj.weight_scale_2",
            DType::F32,
            vec![1],
            encode(DType::F32, &[0.0]),
        )]],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let (store, meta) = (CpuWeights::new(), MetaWeights::new());

    for loader in [
        WeightLoader::new(&checkpoint, &store, None),
        WeightLoader::new(&checkpoint, &meta, None),
    ] {
        let refused = loader.tensor_scale("proj.weight_scale_2").unwrap_err();
        assert!(
            refused.to_string().contains("not a positive number"),
            "{refused}"
        );
    }
}
