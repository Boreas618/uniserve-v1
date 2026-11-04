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

fn main() -> ExitCode {
    let mut checkpoint = PathBuf::from("/opt/uv/models/Qwen3-30B-A3B");
    let mut cache = DType::BF16;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--checkpoint" => match arguments.next() {
                Some(value) => checkpoint = PathBuf::from(value),
                None => return usage("--checkpoint takes a directory"),
            },
            "--cache" => match arguments.next().as_deref() {
                Some("bf16") => cache = DType::BF16,
                Some("f16") => cache = DType::F16,
                Some("f8e4m3") => cache = DType::F8E4M3,
                other => {
                    return usage(&format!("--cache takes bf16, f16 or f8e4m3, not {other:?}"));
                }
            },
            other => return usage(&format!("unknown argument {other}")),
        }
    }

    match run(&checkpoint, cache) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
    }
}

fn run(root: &Path, cache: DType) -> Result<(), String> {
    // The configuration is read on its own so a metadata defect names itself before the load.
    let _ = Config::read(root).map_err(|error| format!("{}: {error}", root.display()))?;
    let file = Checkpoint::open_roots(&[root])
        .map_err(|error| format!("opening {}: {error}", root.display()))?;
    let mut store = HostStore::new();
    let model = {
        let mut loader = WeightLoader::new(&file, &mut store, None, None);
        qwen3_moe::load(
            root,
            Partition::single(),
            &mut loader,
            LoadOptions::default(),
        )
        .map_err(|error| format!("loading {}: {error}", root.display()))?
    };
    let keys = model
        .keys(cache)
        .map_err(|error| format!("enumerating keys: {error}"))?;
    if keys.is_empty() {
        return Err("the model enumerated no programs".to_owned());
    }
    for (operator, key) in keys {
        println!("{operator}\t{key}");
    }
    Ok(())
}

fn usage(message: &str) -> ExitCode {
    eprintln!("{message}");
    eprintln!(
        "usage: qwen3-moe-keys [--checkpoint <dir>] [--cache bf16|f16|f8e4m3]\n\n\
         Prints one `<operator>\\t<key>` line per program the model launches, sorted and unique."
    );
    ExitCode::from(2)
}
