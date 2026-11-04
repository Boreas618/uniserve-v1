//! The programs this Qwen3-MoE deployment compiles, printed as fixture lines.
//!
//!     cargo run -p qwen3-moe --example keys -- [--checkpoint /opt/uv/models/Qwen3-30B-A3B] \
//!         [--cache bf16]
//!
//! Each line is `<operator>\t<key>`, the form `tools/build_store.py` consumes and
//! `notes/programs/qwen3-moe-keys.txt` holds, so the note can be regenerated from the model rather
//! than kept by hand:
//!
//!     cargo run -q -p qwen3-moe --example keys > notes/programs/qwen3-moe-keys.txt
//!
//! **The addresses an `nn::HostStore` answers are host addresses.** They are the right shape and
//! element type, so every descriptor built over them is the one a device run would build, and this
//! example enumerates and audits only: a deployment publishes its weights to device memory and
//! passes what the model then asks for. Nothing here launches a kernel.
//!
//! The checkpoint is read through its headers and converted on the host, so the run needs the
//! shards' bytes but no CUDA context; a checkpoint whose shards are missing cannot be loaded this way.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nn::{Checkpoint, DType, HostStore, WeightLoader};
use qwen3_moe::{Config, LoadOptions, Partition};

/// The deployment a fixture is generated for, as the walk needs to hear it.
///
/// Every value here changes the answer: the two degrees change which programs one rank launches and
/// at what widths, the cache's element type is a plane of its own, and the capacity is the exchange
/// transport's rows per rank. A note that does not state them cannot be checked against a store.
struct Deployment {
    tensor_parallel: usize,
    tensor_rank: usize,
    expert_parallel: usize,
    expert_rank: usize,
    cache: DType,
    capacity: u32,
}

fn main() -> ExitCode {
    let mut checkpoint = PathBuf::from("/opt/uv/models/Qwen3-30B-A3B");
    let mut deployment = Deployment {
        tensor_parallel: 1,
        tensor_rank: 0,
        expert_parallel: 1,
        expert_rank: 0,
        cache: DType::BF16,
        capacity: 0,
    };
    let number = |value: Option<String>, name: &str| -> Result<usize, String> {
        value
            .ok_or_else(|| format!("{name} takes a number"))?
            .parse()
            .map_err(|_| format!("{name} takes a number"))
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--checkpoint" => match arguments.next() {
                Some(value) => checkpoint = PathBuf::from(value),
                None => return usage("--checkpoint takes a directory"),
            },
            "--cache" => match arguments.next().as_deref() {
                Some("bf16") => deployment.cache = DType::BF16,
                Some("f16") => deployment.cache = DType::F16,
                Some("f8e4m3") => deployment.cache = DType::F8E4M3,
                other => {
                    return usage(&format!("--cache takes bf16, f16 or f8e4m3, not {other:?}"));
                }
            },
            "--tensor-parallel" => match number(arguments.next(), "--tensor-parallel") {
                Ok(value) => deployment.tensor_parallel = value,
                Err(message) => return usage(&message),
            },
            "--tensor-rank" => match number(arguments.next(), "--tensor-rank") {
                Ok(value) => deployment.tensor_rank = value,
                Err(message) => return usage(&message),
            },
            "--expert-parallel" => match number(arguments.next(), "--expert-parallel") {
                Ok(value) => deployment.expert_parallel = value,
                Err(message) => return usage(&message),
            },
            "--expert-rank" => match number(arguments.next(), "--expert-rank") {
                Ok(value) => deployment.expert_rank = value,
                Err(message) => return usage(&message),
            },
            "--capacity" => match number(arguments.next(), "--capacity") {
                Ok(value) => match u32::try_from(value) {
                    Ok(value) => deployment.capacity = value,
                    Err(_) => return usage("--capacity takes a row count that fits u32"),
                },
                Err(message) => return usage(&message),
            },
            other => return usage(&format!("unknown argument {other}")),
        }
    }

    match run(&checkpoint, &deployment) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
    }
}

fn run(root: &Path, deployment: &Deployment) -> Result<(), String> {
    // The configuration is read on its own so a metadata defect names itself before the load.
    let config = Config::read(root).map_err(|error| format!("{}: {error}", root.display()))?;
    let partition = Partition::new(
        &config,
        deployment.tensor_parallel,
        deployment.tensor_rank,
        deployment.expert_parallel,
        deployment.expert_rank,
    )
    .map_err(|error| format!("{}: {error}", root.display()))?;
    let file = Checkpoint::open_roots(&[root])
        .map_err(|error| format!("opening {}: {error}", root.display()))?;
    let mut store = HostStore::new();
    let model = {
        let mut loader = WeightLoader::new(&file, &mut store, None, None);
        qwen3_moe::load(root, partition, &mut loader, LoadOptions::default())
            .map_err(|error| format!("loading {}: {error}", root.display()))?
    };
    let keys = model
        .keys(deployment.cache, deployment.capacity, &[])
        .map_err(|error| format!("enumerating keys: {error}"))?;
    if keys.is_empty() {
        return Err("the model enumerated no programs".to_owned());
    }
    // The deployment comes first, so a reader can tell which rank and capacity this fixture is for
    // without inferring it from the keys. The comment is skipped by every fixture reader.
    println!(
        "# deployment: tensor_parallel {} tensor_rank {} expert_parallel {} expert_rank {} \
         cache {} capacity {}",
        deployment.tensor_parallel,
        deployment.tensor_rank,
        deployment.expert_parallel,
        deployment.expert_rank,
        match deployment.cache {
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::F8E4M3 => "f8e4m3",
            other => return Err(format!("the cache's {other} has no fixture name")),
        },
        deployment.capacity,
    );
    for (operator, key) in keys {
        println!("{operator}\t{key}");
    }
    Ok(())
}

fn usage(message: &str) -> ExitCode {
    eprintln!("{message}");
    eprintln!(
        "usage: qwen3-moe-keys [--checkpoint <dir>] [--cache bf16|f16|f8e4m3]\n\
         \x20   [--tensor-parallel N] [--tensor-rank N] [--expert-parallel N] [--expert-rank N]\n\
         \x20   [--capacity rows]\n\n\
         Prints the deployment, then one `<operator>\\t<key>` line per program the model launches, \
         sorted and unique. --capacity is the exchange's rows per rank and is read only when \
         --expert-parallel is above 1."
    );
    ExitCode::from(2)
}
