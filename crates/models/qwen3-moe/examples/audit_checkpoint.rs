//!
//!     cargo run -p qwen3-moe --example audit_checkpoint -- [--root /opt/uv/models/Qwen3-30B-A3B]
//!
//! The slice loads this checkpoint against these rules, so a single renamed tensor in one of 48
//! layers surfaces as a missing-tensor refusal deep in a GPU run. This reads only what the checkout
//! holds — `config.json` and `model.safetensors.index.json` — and answers three questions before any
//! device is involved:
//!
//! - **resolved**: a name the model claims and the checkpoint holds.
//! - **present but unclaimed**: a tensor the checkpoint holds that no rule names. Harmless only where
//!   the model genuinely computes the value itself (the rotary table), or where the checkpoint keeps a
//!   tensor for a path this model does not serve; the audit prints them so the judgement is made
//!   rather than assumed.
//! - **claimed but absent**: a rule naming a tensor the checkpoint does not hold. Always a defect.
//!
//! The expected names come from [`qwen3_moe::weights::parameter_sources`] and
//! [`qwen3_moe::weights::expert_sources`] — the same functions `load` builds its requests from — never
//! from a list written here. The one fallback the loader applies is also applied here: a released
//! checkpoint may serialize the stack under `layers.<i>…` instead of `model.layers.<i>…`, so a name
//! resolves when the index holds it with or without the `model.` prefix. How many names resolve *only*
//! through that fallback is printed, because a rule that accepts anything is not coverage.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use qwen3_moe::weights::{expert_sources, parameter_sources};
use qwen3_moe::Config;

fn main() -> ExitCode {
    let root = match root_argument() {
        Ok(root) => root,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    let config = match Config::read(&root) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{}: {error}", root.display());
            return ExitCode::from(2);
        }
    };
    let present = match index_names(&root) {
        Ok(names) => names,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    // What the model claims: the named parameters, and every expert tensor of every sparse layer.
    let mut claimed: BTreeMap<String, String> = BTreeMap::new();
    for source in parameter_sources(&config) {
        claimed.insert(source.checkpoint, source.parameter);
    }
    let mut expert_tensors = 0;
    for source in expert_sources(&config) {
        for expert in 0..config.num_experts {
            let tensors = source.tensors(expert);
            for (projection, name) in [
                ("down", tensors.down),
                ("up", tensors.up),
                ("gate", tensors.gate),
            ] {
                claimed.insert(
                    name,
                    format!("{}.{expert}.{projection}_proj.weight", source.module),
                );
                expert_tensors += 1;
            }
        }
    }

    let mut resolved = 0;
    let mut by_fallback = 0;
    let mut absent: Vec<(String, String)> = Vec::new();
    for (checkpoint, parameter) in &claimed {
        if present.contains(checkpoint) {
            resolved += 1;
        } else if let Some(stripped) = checkpoint.strip_prefix("model.") {
            if present.contains(stripped) {
                // The loader's documented fallback: the stack serialized without the prefix.
                resolved += 1;
                by_fallback += 1;
            } else {
                absent.push((checkpoint.clone(), parameter.clone()));
            }
        } else {
            absent.push((checkpoint.clone(), parameter.clone()));
        }
    }

    // Present but unclaimed: everything the checkpoint holds that no rule names. A checkpoint that
    // omits the prefix lists names the rules' `model.`-prefixed form does not equal, so both forms of
    // each claimed name are removed before the remainder is reported.
    let mut unclaimed: Vec<&String> = present
        .iter()
        .filter(|name| {
            !claimed.contains_key(*name)
                && !claimed.contains_key(&format!("model.{name}"))
                && !(name.strip_prefix("model.").is_some_and(|stripped| claimed.contains_key(stripped)))
        })
        .collect();
    unclaimed.sort();

    println!(
        "# {} names claimed by the model ({} parameters + {expert_tensors} expert tensors over {} \
         sparse layers), {} tensors held by the checkpoint, {} resolved",
        claimed.len(),
        claimed.len() - expert_tensors,
        expert_sources(&config).len(),
        present.len(),
        resolved,
    );
    println!("# {by_fallback} resolved only through the loader's `model.`-prefix fallback");
    println!("# {} present but unclaimed", unclaimed.len());
    for name in unclaimed.iter().take(5) {
        println!("    {name}");
    }
    println!("# {} claimed but absent", absent.len());
    for (checkpoint, parameter) in absent.iter().take(5) {
        println!("    {checkpoint}  ←  {parameter}");
    }

    if absent.is_empty() {
        println!("# every claimed tensor is present");
        ExitCode::SUCCESS
    } else {
        println!("# {} rules name a tensor the checkpoint does not hold", absent.len());
        ExitCode::FAILURE
    }
}

/// The checkout root: `--root <path>`, or the local Qwen3-30B-A3B metadata checkout.
fn root_argument() -> Result<PathBuf, String> {
    let mut arguments = std::env::args().skip(1);
    let mut root = PathBuf::from("/opt/uv/models/Qwen3-30B-A3B");
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--root" => {
                root = arguments
                    .next()
                    .map(PathBuf::from)
                    .ok_or_else(|| "--root takes a path".to_owned())?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(root)
}

/// Every tensor name the checkpoint's index holds.
fn index_names(root: &Path) -> Result<BTreeSet<String>, String> {
    let path = root.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let index: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let map = index
        .get("weight_map")
        .and_then(|map| map.as_object())
        .ok_or_else(|| format!("{}: no weight_map object", path.display()))?;
    Ok(map.keys().cloned().collect())
}
