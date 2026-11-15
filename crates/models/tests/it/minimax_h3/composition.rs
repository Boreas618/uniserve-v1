//! A deployment's sources compose to one MiniMax-H3 checkpoint by what the checkpoint declares:
//! the base release by its pipeline index, a student export by its contract, the base release an
//! export continues for the components the export omits, and an adapter by the contract beside
//! it. Each component comes from exactly one source.
//!
//! The directories are written here (an index, a contract, and per component one tiny shard set
//! or one scheduler file, each beside a `config.json` that names the directory it lies in), so no
//! checkpoint and no device is needed.

use std::path::Path;

use models::minimax_h3::config::{Contract, Networks};
use models::minimax_h3::family::compose;
use nn::family::Stage;
use nn::{Checkpoint, DType, Source, Sources};

use super::support::{f32_bytes, f32_values, shard, shard_with_header};

/// Every component of a MiniMax-H3 pipeline index, with its library and whether it holds a
/// shard set.
const COMPONENTS: [(&str, &str, bool); 9] = [
    ("text_encoder", "transformers", true),
    ("tokenizer", "transformers", false),
    ("processor", "transformers", false),
    ("vae", "diffusers", true),
    ("audio_vae", "diffusers", true),
    ("transformer", "diffusers", true),
    ("transformer_ref", "diffusers", true),
    ("scheduler", "diffusers", false),
    ("audio_scheduler", "diffusers", false),
];

/// The components a reference student export holds itself.
const REFERENCE_EXPORT: [&str; 3] = ["transformer_ref", "scheduler", "audio_scheduler"];

/// The components a reference student export takes from the base release it continues.
const REFERENCE_BASE: [&str; 4] = ["text_encoder", "processor", "vae", "audio_vae"];

/// The base release this module's reference student continues.
const BASE_REVISION: &str = "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da";

/// Writes the pipeline index of `class` over every component, as the released checkpoints
/// publish it.
fn index(root: &Path, class: &str) {
    let components: Vec<String> = COMPONENTS
        .iter()
        .map(|(name, library, _)| {
            format!(r#""{name}": ["{library}", "Class", {{"subfolder": "{name}"}}]"#)
        })
        .collect();
    std::fs::write(
        root.join("modular_model_index.json"),
        format!(r#"{{"_class_name": "{class}", {}}}"#, components.join(", ")),
    )
    .unwrap();
}

/// Writes the components `names` under `root`, each labelled `label`: a stage's shard set under
/// its library's name, holding `proj.weight` (`[2, 3]`, the values 1 to 6), and every component's
/// `config.json` naming `label`.
fn components(root: &Path, names: &[&str], label: &str) {
    for name in names {
        let (_, library, weights) = COMPONENTS
            .iter()
            .find(|(component, ..)| component == name)
            .unwrap();
        let directory = root.join(name);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            format!(r#"{{"from": "{label}"}}"#),
        )
        .unwrap();

        if *weights {
            let file = match *library {
                "transformers" => "model.safetensors",
                _ => "diffusion_pytorch_model.safetensors",
            };
            shard(
                &directory,
                file,
                &[(
                    "proj.weight",
                    DType::F32,
                    vec![2, 3],
                    f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                )],
            );
        }
    }
}

/// Writes a student's contract into `directory`: its task under the key its form states it by,
/// its base when it continues one, and a four-rung ladder.
fn contract(directory: &Path, task: (&str, &str), base: Option<&str>) {
    let mut contract = serde_json::json!({
        "schema_version": Contract::SCHEMA,
        task.0: task.1,
        "dmd_denoising_steps": [999, 749, 500, 250],
        "attention_backend": "DENSE",
    });
    if let Some(base) = base {
        contract["base_model_revision"] = serde_json::json!(base);
    }
    std::fs::write(directory.join(Contract::FILE), contract.to_string()).unwrap();
}

/// A base release at `root`: its index and every component.
fn release(root: &Path) {
    index(root, "MiniMaxH3ModularPipeline");
    let all: Vec<&str> = COMPONENTS.iter().map(|(name, ..)| *name).collect();
    components(root, &all, "release");
}

/// A reference student export at `root` that continues [`BASE_REVISION`], and the components
/// it omits at `base`, as the base's partial snapshot holds them.
fn reference_export(root: &Path, base: &Path) {
    index(root, "MiniMaxH3ModularPipeline");
    components(root, &REFERENCE_EXPORT, "export");
    contract(root, ("model_type", "ref2va"), Some(BASE_REVISION));
    components(base, &REFERENCE_BASE, "base");
}

/// The local directory `directory`.
fn local(directory: &Path) -> Source {
    Source::Local(directory.to_path_buf())
}

/// The sources of a deployment of `checkpoint`, over `base` with `adapter`.
fn sources(checkpoint: &Path, base: Option<&Path>, adapter: Option<&Path>) -> Sources {
    Sources {
        checkpoint: local(checkpoint),
        base: base.map(local),
        adapter: adapter.map(local),
    }
}

/// The label of the directory the composed checkpoint read the component `name`'s files from.
fn origin(checkpoint: &Checkpoint, name: &str) -> String {
    let config = checkpoint
        .sidecar(&format!("{name}/config.json"))
        .unwrap()
        .unwrap();
    config["from"].as_str().unwrap().to_owned()
}

/// The networks `networks` serves.
fn served(networks: &Networks) -> Vec<Stage> {
    match networks {
        Networks::Released(stages) => stages.clone(),
        Networks::Student(contract) => vec![contract.network],
    }
}

#[test]
fn a_base_release_serves_every_network_its_index_declares() {
    let root = tempfile::tempdir().unwrap();
    release(root.path());

    let (checkpoint, networks) = compose(&sources(root.path(), None, None)).unwrap();
    assert!(matches!(networks, Networks::Released(_)));
    assert_eq!(
        served(&networks),
        [Stage::Transformer, Stage::TransformerRef]
    );
    for name in [
        "text_encoder",
        "vae",
        "audio_vae",
        "transformer",
        "transformer_ref",
    ] {
        assert_eq!(checkpoint.stage(name).unwrap().len(), 1, "{name}");
    }
    assert_eq!(origin(&checkpoint, "processor"), "release");

    // A base release continues no base.
    let other = tempfile::tempdir().unwrap();
    let error = compose(&sources(root.path(), Some(other.path()), None))
        .err()
        .expect("refused");
    assert!(error.to_string().contains("continues none"), "{error}");
}

#[test]
fn a_checkpoint_is_refused_by_what_it_declares_and_lacks() {
    // A pipeline of another class is not this family's, whatever its directories hold.
    let task = tempfile::tempdir().unwrap();
    release(task.path());
    index(task.path(), "MiniMaxH3Pipeline");
    let error = compose(&sources(task.path(), None, None))
        .err()
        .expect("refused");
    assert!(error.to_string().contains("MiniMaxH3Pipeline"), "{error}");

    // A network the index declares and the release does not hold is refused, not skipped.
    let partial = tempfile::tempdir().unwrap();
    release(partial.path());
    std::fs::remove_dir_all(partial.path().join("transformer_ref")).unwrap();
    let error = compose(&sources(partial.path(), None, None))
        .err()
        .expect("refused");
    assert!(error.to_string().contains("transformer_ref"), "{error}");
}

#[test]
fn a_reference_export_takes_what_it_omits_from_the_base_it_continues() {
    let (export, base) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    reference_export(export.path(), base.path());

    let (checkpoint, networks) = compose(&sources(export.path(), Some(base.path()), None)).unwrap();
    assert_eq!(served(&networks), [Stage::TransformerRef]);
    for name in REFERENCE_EXPORT {
        assert_eq!(origin(&checkpoint, name), "export", "{name}");
    }
    for name in REFERENCE_BASE {
        assert_eq!(origin(&checkpoint, name), "base", "{name}");
    }
    assert!(
        checkpoint.stage("transformer").is_err(),
        "not the student's"
    );

    // The base it continues is the deployment's to state.
    let error = compose(&sources(export.path(), None, None))
        .err()
        .expect("refused");
    assert!(error.to_string().contains(BASE_REVISION), "{error}");

    // A snapshot of another commit is not the base it continues; the one it pins is.
    let snapshot = |revision: &str| Sources {
        base: Some(Source::snapshot(
            "MiniMaxAI/MiniMax-H3",
            revision,
            base.path().to_path_buf(),
        )),
        ..sources(export.path(), None, None)
    };
    assert!(compose(&snapshot("42ed227ee7df40d41602854ae760620d6eb651fe")).is_err());
    assert!(compose(&snapshot("9bfb6693f2cf6de171db46d1aa586f67d773a1da")).is_ok());

    // An export that also holds a component it takes from the base is refused.
    components(export.path(), &["vae"], "export");
    let error = compose(&sources(export.path(), Some(base.path()), None))
        .err()
        .expect("refused");
    assert!(error.to_string().contains("`vae`"), "{error}");
}

#[test]
fn a_whole_student_export_continues_no_base() {
    let export = tempfile::tempdir().unwrap();
    release(export.path());
    contract(export.path(), ("task", "t2av"), None);

    let (checkpoint, networks) = compose(&sources(export.path(), None, None)).unwrap();
    assert_eq!(served(&networks), [Stage::Transformer]);
    assert_eq!(origin(&checkpoint, "vae"), "release");
    assert!(checkpoint.stage("transformer_ref").is_err());

    let base = tempfile::tempdir().unwrap();
    let error = compose(&sources(export.path(), Some(base.path()), None))
        .err()
        .expect("refused");
    assert!(error.to_string().contains("continues none"), "{error}");
}

/// Writes a FastVideo adapter into `directory`: a rank-one edit of `proj.weight`.
fn adapter(directory: &Path) {
    let header = r#"{"__metadata__": {"format": "fastvideo-lora-v2", "rank": "1",
        "low_rank_tensors": "2", "diff_tensors": "0", "set_weight_tensors": "0"},
        "proj.lora_A.weight": {"dtype": "F32", "shape": [1, 3], "data_offsets": [0, 12]},
        "proj.lora_B.weight": {"dtype": "F32", "shape": [2, 1], "data_offsets": [12, 20]}}"#;
    shard_with_header(
        directory,
        "adapter_model.safetensors",
        header,
        &f32_bytes(&[1.0, 2.0, 3.0, 0.5, -1.0]),
    );
}

#[test]
fn an_adapter_is_fused_into_the_network_its_contract_names() {
    let (root, directory) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    release(root.path());
    adapter(directory.path());

    // An adapter states the student it is in its contract, and is refused without one.
    let error = compose(&sources(root.path(), None, Some(directory.path())))
        .err()
        .expect("refused");
    assert!(error.to_string().contains(Contract::FILE), "{error}");

    contract(directory.path(), ("task", "t2av"), None);
    let (checkpoint, networks) =
        compose(&sources(root.path(), None, Some(directory.path()))).unwrap();
    assert_eq!(served(&networks), [Stage::Transformer]);
    assert!(checkpoint.stage("transformer_ref").is_err());

    // `W + lora_B @ lora_A`: rows of [1, 2, 3] scaled by 0.5 and -1, added to the base.
    let transformer = checkpoint.stage("transformer").unwrap();
    let values = f32_values(&transformer.read("proj.weight").unwrap());
    assert_eq!(values, [1.5, 3.0, 4.5, 3.0, 3.0, 3.0]);

    // An adapter is fused into the base release, not into a student.
    contract(root.path(), ("task", "t2av"), None);
    let error = compose(&sources(root.path(), None, Some(directory.path())))
        .err()
        .expect("refused");
    assert!(error.to_string().contains("student export"), "{error}");
}
