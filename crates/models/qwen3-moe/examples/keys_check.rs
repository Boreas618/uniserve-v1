//! The model's own key walk against the committed fixture, both page types, and the gap named.
//!
//! ```text
//! cargo run -p qwen3-moe --example keys_check -- [--checkpoint /opt/uv/models/Qwen3-30B-A3B]
//!     [--fixture notes/programs/qwen3-moe-keys.txt] [--capacity 0] [--page-tokens 64]
//! ```
//!
//! `keys()` is a pure function of the configuration, the partition and the deployment's values, so
//! this runs where no shard does: the checkpoint's `config.json` is enough. That matters, because the
//! loaded form wants every shard and this sandbox's copy of the checkpoint is metadata only.
//!
//! It answers three questions and refuses on the first two:
//!
//! - **a row the walk names and the fixture lacks** — the store cannot hold a program the model
//!   launches, so this is a failure;
//! - **a row the fixture holds that is neither the walk's nor the executor's** — an invented row, or
//!   one whose operator moved ownership; a failure, because the fixture claims to be a statement of
//!   what is launched;
//! - **the executor's rows**, which are listed and counted but not refused: their geometry is the
//!   executor's request and schedule state, and the walk never names them.
//!
//! The pages are asked twice, because a deployment's page type decides the cache write's operator
//! (`kv_write` for a full-precision page, `kv_write_scaled` for E4M3) and the attention's key: the
//! fixture states the full-precision deployment, and the E4M3 deployment differs only in those rows.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use qwen3_moe::keys::keys;
use qwen3_moe::{Config, Partition};

/// The operators whose rows the executor's request and schedule state decide, so the model's walk
/// never names them: the sampler's chain and the logits gathers.
const EXECUTOR_OWNED: [&str; 10] = [
    "argmax",
    "candidate_order",
    "filter_candidates",
    "inverse_cdf",
    "log_softmax",
    "logits_gather",
    "logits_shape",
    "logprob_gather",
    "softmax",
    "top_k",
];

fn main() -> ExitCode {
    let mut checkpoint = PathBuf::from("/opt/uv/models/Qwen3-30B-A3B");
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../notes/programs/qwen3-moe-keys.txt");
    let mut capacity = 0u32;
    let mut page_tokens = 64u32;

    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || arguments.next();
        match argument.as_str() {
            "--checkpoint" => checkpoint = PathBuf::from(value().unwrap_or_default()),
            "--fixture" => fixture = PathBuf::from(value().unwrap_or_default()),
            "--capacity" => match value().and_then(|text| text.parse().ok()) {
                Some(parsed) => capacity = parsed,
                None => return usage("--capacity takes a number"),
            },
            "--page-tokens" => match value().and_then(|text| text.parse().ok()) {
                Some(parsed) => page_tokens = parsed,
                None => return usage("--page-tokens takes a number"),
            },
            other => return usage(&format!("unknown argument {other}")),
        }
    }

    let config = match Config::read(&checkpoint) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{}: {error}", checkpoint.display());
            return ExitCode::from(2);
        }
    };
    let present = match fixture_operators(&fixture) {
        Ok(present) => present,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    let mut failures = Vec::new();
    for (label, cache) in [("bf16", ops::DType::BF16), ("E4M3", ops::DType::F8E4M3)] {
        let walked = match keys(
            &config,
            &Partition::single(),
            ops::DType::BF16,
            page_tokens,
            cache,
            capacity,
            &[],
            false,
            false,
        ) {
            Ok(walked) => walked,
            Err(error) => {
                eprintln!("{label}: {error}");
                return ExitCode::from(2);
            }
        };
        let named: BTreeSet<String> = walked.iter().map(|(operator, _)| operator.clone()).collect();
        println!("# {label} pages: {} rows over {} operators", walked.len(), named.len());

        // Only the full-precision deployment is the fixture's; the E4M3 rows are reported as what
        // that file would hold instead, so the header's claim is checked rather than repeated.
        if label != "bf16" {
            let full = keys(
                &config,
                &Partition::single(),
                ops::DType::BF16,
                page_tokens,
                ops::DType::BF16,
                capacity,
                &[],
                false,
                false,
            )
            .unwrap_or_default();
            let mut differs: Vec<String> = Vec::new();
            for (operator, key) in &walked {
                if !full.iter().any(|(other, full_key)| other == operator && full_key == key) {
                    differs.push(format!("{operator}\t{key}"));
                }
            }
            let mut gained: Vec<String> = Vec::new();
            for (operator, key) in &full {
                if !walked.iter().any(|(other, walk_key)| other == operator && walk_key == key) {
                    gained.push(format!("{operator}\t{key}"));
                }
            }
            println!("    in place of the bf16 deployment's rows:");
            for row in &gained {
                println!("    - {row}");
            }
            for row in &differs {
                println!("    + {row}");
            }
            continue;
        }

        for operator in &named {
            if !present.contains(operator) {
                failures.push(format!("the walk names `{operator}` and the fixture lacks it"));
            }
        }
        let unowned: Vec<&String> = present
            .iter()
            .filter(|operator| {
                !named.contains(*operator) && !EXECUTOR_OWNED.contains(&operator.as_str())
            })
            .collect();
        for operator in &unowned {
            failures.push(format!(
                "the fixture holds `{operator}`, which is neither the walk's nor the executor's"
            ));
        }
        let executor: Vec<&String> = present
            .iter()
            .filter(|operator| EXECUTOR_OWNED.contains(&operator.as_str()))
            .collect();
        println!(
            "    {} the walk names, {} the executor's, {} unaccounted",
            named.intersection(&present).count(),
            executor.len(),
            unowned.len()
        );
    }

    if failures.is_empty() {
        println!("# the fixture is the walk's rows plus the executor's, and nothing else");
        ExitCode::SUCCESS
    } else {
        for failure in &failures {
            println!("# {failure}");
        }
        ExitCode::FAILURE
    }
}

/// The operators the fixture's rows name, comments and blank lines skipped.
fn fixture_operators(path: &Path) -> Result<BTreeSet<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(text
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| line.split_once('\t').map(|(operator, _)| operator.trim().to_owned()))
        .collect())
}

fn usage(problem: &str) -> ExitCode {
    eprintln!("{problem}");
    eprintln!("usage: keys_check [--checkpoint PATH] [--fixture PATH] [--capacity N] [--page-tokens N]");
    ExitCode::from(2)
}
