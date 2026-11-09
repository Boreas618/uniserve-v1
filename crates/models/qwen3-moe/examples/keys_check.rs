//! The model's own key walk against the committed fixture, both page types, and the gap named.
//!
//! ```text
//! cargo run -p qwen3-moe --example keys_check -- [--checkpoint $UNISERVE_QWEN3_MODEL]
//!     --fixture <keys file> [--capacity 0] [--page-tokens 64]
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
//! - **a row the fixture holds that is neither the walk's nor the worker's** — an invented row, or
//!   one whose operator moved ownership; a failure, because the fixture claims to be a statement of
//!   what is launched;
//! - **the worker's rows**, which are listed and counted but not refused: their geometry is the
//!   worker's request and schedule state, and the walk never names them.
//!
//! The first comparison is by **operator and row class**: the packed product's key carries the class
//! its module was compiled for, that class is the deployment's bucket bound, and a fixture generated
//! for one bound states a class no launch of another states. Comparing operator names alone is what
//! let a fixture built for `--buckets 1,15` read as complete for a deployment whose bound is 512.
//!
//! The rest of a key is *not* compared, because this file holds three deployments (the bf16 one, the
//! calibrated one and the E4M3-page one) and only the first is the walk's for a given run: their rows
//! differ in fields that belong to each deployment's own declaration.
//!
//! The pages are asked twice, because a deployment's page type decides the cache write's operator
//! (`kv_write` for a full-precision page, `kv_write_scaled` for E4M3) and the attention's key: the
//! fixture states the full-precision deployment, and the E4M3 deployment differs only in those rows.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use qwen3_moe::keys::keys;
use qwen3_moe::{Config, Partition};

/// The operators whose rows the worker's request and schedule state decide, so the model's walk
/// never names them: the sampler's chain and the logits gathers.
const WORKER_OWNED: [&str; 10] = [
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
    let mut checkpoint = PathBuf::new();
    let mut fixture = PathBuf::new();
    let mut capacity = 0u32;
    let mut page_tokens = 64u32;
    let mut buckets: Vec<u32> = Vec::new();
    let mut experts_encoded = false;
    let mut projections_packed = false;
    let mut print = false;

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
            "--buckets" => match value() {
                Some(list) => {
                    buckets = list
                        .split(',')
                        .filter_map(|entry| entry.trim().parse().ok())
                        .collect();
                }
                None => return usage("--buckets takes a comma-separated row list"),
            },
            // A checkpoint whose expert stacks are calibrated NVFP4, and whose projections are too:
            // the walk states those rows itself, and the store needs them per captured bucket.
            "--experts-encoded" => experts_encoded = true,
            "--projections-packed" => projections_packed = true,
            // Print the walk's rows for the full-precision deployment, the form the fixture merges.
            "--print" => print = true,
            other => return usage(&format!("unknown argument {other}")),
        }
    }
    if checkpoint.as_os_str().is_empty() {
        checkpoint = std::env::var_os("UNISERVE_QWEN3_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                eprintln!("name the checkpoint with --checkpoint or UNISERVE_QWEN3_MODEL");
                std::process::exit(2)
            });
    }

    let config = match Config::read(&checkpoint) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{}: {error}", checkpoint.display());
            return ExitCode::from(2);
        }
    };
    if fixture.as_os_str().is_empty() {
        return usage("--fixture names the key file to check");
    }
    let present_rows = match fixture_rows(&fixture) {
        Ok(rows) => rows,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    let present: BTreeSet<String> = present_rows
        .iter()
        .map(|(operator, _)| operator.clone())
        .collect();

    if print {
        match keys(
            &config,
            &Partition::single(),
            ops::DType::BF16,
            page_tokens,
            ops::DType::BF16,
            capacity,
            &buckets,
            experts_encoded,
            projections_packed,
        ) {
            Ok(walked) => {
                println!(
                    "# deployment: tensor_parallel 1 tensor_rank 0 expert_parallel 1 expert_rank 0 \
                     cache bf16 capacity {capacity} page {page_tokens} buckets {}",
                    if buckets.is_empty() {
                        "none".to_owned()
                    } else {
                        buckets
                            .iter()
                            .map(|bucket| bucket.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    }
                );
                for (operator, key) in walked {
                    println!("{operator}\t{key}");
                }
                return ExitCode::SUCCESS;
            }
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::from(2);
            }
        }
    }

    let mut failures = Vec::new();
    for (label, cache) in [("bf16", ops::DType::BF16), ("E4M3", ops::DType::F8E4M3)] {
        let walked = match keys(
            &config,
            &Partition::single(),
            ops::DType::BF16,
            page_tokens,
            cache,
            capacity,
            &buckets,
            experts_encoded,
            projections_packed,
        ) {
            Ok(walked) => walked,
            Err(error) => {
                eprintln!("{label}: {error}");
                return ExitCode::from(2);
            }
        };
        let named: BTreeSet<String> = walked
            .iter()
            .map(|(operator, _)| operator.clone())
            .collect();
        println!(
            "# {label} pages: {} rows over {} operators",
            walked.len(),
            named.len()
        );

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
                &buckets,
                experts_encoded,
                projections_packed,
            )
            .unwrap_or_default();
            let mut differs: Vec<String> = Vec::new();
            for (operator, key) in &walked {
                if !full
                    .iter()
                    .any(|(other, full_key)| other == operator && full_key == key)
                {
                    differs.push(format!("{operator}\t{key}"));
                }
            }
            let mut gained: Vec<String> = Vec::new();
            for (operator, key) in &full {
                if !walked
                    .iter()
                    .any(|(other, walk_key)| other == operator && walk_key == key)
                {
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
                failures.push(format!(
                    "the walk names `{operator}` and the fixture lacks it"
                ));
            }
        }
        let unowned: Vec<&String> = present
            .iter()
            .filter(|operator| {
                !named.contains(*operator) && !WORKER_OWNED.contains(&operator.as_str())
            })
            .collect();
        for operator in &unowned {
            failures.push(format!(
                "the fixture holds `{operator}`, which is neither the walk's nor the worker's"
            ));
        }
        // An operator's presence is not its row set. The packed product's key carries the row class
        // its module was compiled for, and the class is the deployment's bucket bound, so a store
        // built for one bound serves no launch that states another - which is how a fixture
        // generated for one bucket set read as complete for a deployment whose bound is larger.
        let bound = buckets
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        for (operator, class) in class_gaps(&walked, &present_rows) {
            failures.push(format!(
                "the walk names `{operator}` at row class `m_{class}`, and the fixture states no \
                 row for it: a module compiled for another class serves none of these launches, \
                 whose bucket rows are {bound}"
            ));
        }
        let worker: Vec<&String> = present
            .iter()
            .filter(|operator| WORKER_OWNED.contains(&operator.as_str()))
            .collect();
        println!(
            "    {} the walk names, {} the worker's, {} unaccounted",
            named.intersection(&present).count(),
            worker.len(),
            unowned.len()
        );
    }

    if failures.is_empty() {
        println!("# the fixture is the walk's rows plus the worker's, and nothing else");
        ExitCode::SUCCESS
    } else {
        for failure in &failures {
            println!("# {failure}");
        }
        ExitCode::FAILURE
    }
}

/// The operators the fixture's rows name, comments and blank lines skipped.
fn fixture_rows(path: &Path) -> Result<Vec<(String, String)>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(text
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| {
            line.split_once('\t')
                .map(|(operator, key)| (operator.trim().to_owned(), key.trim().to_owned()))
        })
        .collect())
}

/// Every `(operator, row class)` the walk names that no row of the fixture holds.
///
/// The walk states one class per packed operator per deployment, so this is what a store must hold
/// for the deployment's bucket bound: a fixture generated for another bound states a class nothing
/// launches, and an operator-name comparison calls that complete.
fn class_gaps(walked: &[(String, String)], held: &[(String, String)]) -> Vec<(String, String)> {
    let classes = |rows: &[(String, String)]| -> BTreeSet<(String, String)> {
        rows.iter()
            .filter_map(|(operator, key)| Some((operator.clone(), row_class(key)?.to_owned())))
            .collect()
    };
    let held = classes(held);
    classes(walked)
        .into_iter()
        .filter(|entry| !held.contains(entry))
        .collect()
}

/// The packed row class a key states, when it states one.
///
/// The class is a compile-time constant of the packed product's module, so two keys that differ
/// only in it name two programs, and a store holding one of them serves no launch that states the
/// other.
fn row_class(key: &str) -> Option<&str> {
    let rest = &key[key.find("_m_")? + 3..];
    let end = rest.find('_')?;
    Some(&rest[..end])
}

fn usage(problem: &str) -> ExitCode {
    eprintln!("{problem}");
    eprintln!(
        "usage: keys_check [--checkpoint PATH] [--fixture PATH] [--capacity N] [--page-tokens N]"
    );
    ExitCode::from(2)
}

#[cfg(test)]
mod tests {
    use super::class_gaps;

    fn rows(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(operator, key)| ((*operator).to_owned(), (*key).to_owned()))
            .collect()
    }

    const PACKED_128: &str = "packed_u8_scale_f8e4m3_layout_swizzled_128x4_m_128_n_2048_k_4096";
    const PACKED_1024: &str = "packed_u8_scale_f8e4m3_layout_swizzled_128x4_m_1024_n_2048_k_4096";

    #[test]
    fn a_class_the_fixture_does_not_state_is_a_gap() {
        let walked = rows(&[("nvfp4_gemm", PACKED_1024)]);
        let held = rows(&[("nvfp4_gemm", PACKED_128)]);
        assert_eq!(
            class_gaps(&walked, &held),
            vec![("nvfp4_gemm".to_owned(), "1024".to_owned())]
        );
    }

    #[test]
    fn the_class_the_fixture_states_is_not_a_gap() {
        let walked = rows(&[
            ("nvfp4_gemm", PACKED_1024),
            ("gemm", "input_bf16_weight_bf16"),
        ]);
        let held = rows(&[("nvfp4_gemm", PACKED_1024)]);
        assert!(class_gaps(&walked, &held).is_empty());
    }

    #[test]
    fn a_key_that_states_no_class_is_outside_the_rule() {
        let walked = rows(&[("gemm", "input_bf16_weight_bf16_accumulate_f32_order_nxk")]);
        assert!(class_gaps(&walked, &[]).is_empty());
    }
}
