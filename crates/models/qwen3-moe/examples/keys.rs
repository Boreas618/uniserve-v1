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
//! **Nothing here opens a shard or a device.** The walk reads the checkpoint's `config.json` and its
//! declaration and states every program from geometry and element types alone, so a machine holding
//! only a checkpoint's metadata prints the same lines a GPU host does — which is what makes the
//! note's diff checkable on any machine.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nn::DType;
use qwen3_moe::{Config, Partition, Representation};

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
    /// Tokens one prefix-cache page holds, and the dtype the activations have.
    page: u32,
    dtype: DType,
    /// The row counts the executor captures, which the dense packed product is compiled per entry of.
    buckets: Vec<u32>,
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
        page: 64,
        dtype: DType::BF16,
        buckets: Vec::new(),
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
            "--page" => match number(arguments.next(), "--page") {
                Ok(value) => match u32::try_from(value) {
                    Ok(value) => deployment.page = value,
                    Err(_) => return usage("--page takes a token count that fits u32"),
                },
                Err(message) => return usage(&message),
            },
            "--buckets" => match arguments.next() {
                None => return usage("--buckets takes a comma-separated row list"),
                Some(value) => {
                    let mut rows: Vec<u32> = Vec::new();
                    for entry in value.split(',') {
                        match entry.trim().parse::<u32>() {
                            Ok(row) => rows.push(row),
                            Err(_) => {
                                return usage(&format!("--buckets takes numbers, not {entry:?}"));
                            }
                        }
                    }
                    rows.sort_unstable();
                    rows.dedup();
                    deployment.buckets = rows;
                }
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
    // The checkpoint's own declaration says which representation its weights are in, and that is
    // what decides whether the encoded programs are named at all — read from metadata, like the
    // configuration.
    let representation = Representation::read(root)
        .map_err(|error| format!("{}: {error}", root.display()))?;
    let packed = representation.is_encoded();
    let keys = qwen3_moe::keys::keys(
        &config,
        &partition,
        deployment.dtype,
        deployment.page,
        deployment.cache,
        deployment.capacity,
        &deployment.buckets,
        packed,
        packed,
    )
    .map_err(|error| format!("enumerating keys: {error}"))?;
    if keys.is_empty() {
        return Err("the model enumerated no programs".to_owned());
    }
    // The deployment comes first, so a reader can tell which rank and capacity this fixture is for
    // without inferring it from the keys. The comment is skipped by every fixture reader.
    println!(
        "# deployment: tensor_parallel {} tensor_rank {} expert_parallel {} expert_rank {} \
         cache {} capacity {} page {} buckets {}",
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
        deployment.page,
        if deployment.buckets.is_empty() {
            "none".to_owned()
        } else {
            deployment
                .buckets
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        },
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
         \x20   [--capacity rows] [--page tokens] [--buckets N,N,…]\n\n\
         Prints the deployment, then one `<operator>\\t<key>` line per program the model launches, \
         sorted and unique. --capacity is the exchange's rows per rank and is read only when \
         --expert-parallel is above 1; --buckets is the executor's captured row counts, and an \
         empty list is legitimate only when no packed projection is named."
    );
    ExitCode::from(2)
}
