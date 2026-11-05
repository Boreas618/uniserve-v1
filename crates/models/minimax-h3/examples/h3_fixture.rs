//! One deployment's fixture, copied from that deployment's walk print.
//!
//! ```text
//! cargo run -q -p minimax-h3 --example programs -- /opt/uv/models/MiniMax-H3 --dtype f16 ... > /tmp/walk.txt
//! cargo run -q -p minimax-h3 --example h3_fixture -- --walk /tmp/walk.txt --commit <rev> \
//!     [--blocked /tmp/blocked.txt] > notes/programs/minimax-h3-keys.txt
//! ```
//!
//! ## Why copying and not restating
//!
//! A fixture is the rows **one deployment** launches. The only implementation of "which rows" is the
//! model's own walk: a tool that restates them — recomputing each geometry from the checkpoint's
//! config — is a second implementation, it drifts from the model silently, and when it unions the
//! element types a deployment may be stored in it produces a row set no single store build can serve.
//! A tree that held exactly that wrote a 248-row fixture of which 105 rows were in no walk at all, and
//! `keys_delta` refused it while `key_coverage` called it complete, because coverage compares the store
//! against the fixture and only the delta compares the fixture against the walk.
//!
//! So this tool copies: the rows come from the walk byte for byte, the deployment line comes from the
//! walk's own header, and the file states its count, its commit and the exact command that produced it.
//! Regenerating it is one command, and `keys_delta` on the result is the check that it is still the
//! walk's rows.
//!
//! ## The blocked block
//!
//! `--blocked` takes the lines `key_coverage --declared-blocked` prints: rows this deployment asks for
//! and no package fills yet, each with the coverage tool's own sentence. Pasted rather than written, so
//! a declaration that lands removes its own line on the next regeneration.

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(problem) => {
            eprintln!("error: {problem}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let mut walk: Option<PathBuf> = None;
    let mut commit: Option<String> = None;
    let mut blocked: Option<PathBuf> = None;
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} needs a value"))
        };
        match argument.as_str() {
            "--walk" => walk = Some(PathBuf::from(value()?)),
            "--blocked" => blocked = Some(PathBuf::from(value()?)),
            "--commit" => {
                let stated = value()?;
                if stated.is_empty() {
                    return Err(
                        "--commit is the tree the walk came from, and it cannot be empty".into(),
                    );
                }
                commit = Some(stated);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let walk = walk.ok_or("--walk is the walk's own print")?;
    let commit = commit.ok_or(
        "the fixture names the commit the walk came from: pass --commit <revision>, as a row set is \
         evidence only of the tree that produced it",
    )?;
    let print =
        std::fs::read_to_string(&walk).map_err(|error| format!("{}: {error}", walk.display()))?;

    // The walk's rows, in the order it printed them, and its own header lines unread: the deployment
    // line is the header line that names the deployment, and the row count is printed once every row
    // is read, so no two readings of the file can disagree about it.
    let mut rows: Vec<&str> = Vec::new();
    let mut deployment: Option<&str> = None;
    for line in print.lines() {
        if let Some(header) = line.strip_prefix("# ") {
            // `# MiniMax-H3 programs for these clip lengths: …` is the walk's statement of what it
            // walked; every other header line is the tool's own and is not copied.
            if header.starts_with("MiniMax-H3 programs for") {
                deployment = Some(line);
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        if !line.contains('\t') {
            return Err(format!(
                "{:?} is not a `operator<TAB>key` row: {}",
                walk.display(),
                line
            ));
        }
        rows.push(line);
    }
    if rows.is_empty() {
        return Err(format!(
            "{} names no rows: an empty fixture is a walk that printed nothing",
            walk.display()
        ));
    }
    let deployment = deployment.ok_or_else(|| {
        format!(
            "{} states no deployment line: the walk prints one and the fixture carries it verbatim",
            walk.display()
        )
    })?;

    // One row per line, the same key once: a variant two layers launch is packaged once.
    rows.sort_unstable();
    rows.dedup();

    let mut operators: Vec<&str> = rows
        .iter()
        .map(|row| row.split('\t').next().unwrap_or_default())
        .collect();
    operators.sort_unstable();
    operators.dedup();

    if let Some(blocked) = blocked {
        let lines = std::fs::read_to_string(&blocked)
            .map_err(|error| format!("{}: {error}", blocked.display()))?;
        // The coverage tool's own sentence per row, pasted: the count of them is not printed here
        // because the coverage run states it, and two numbers for one set is how they come to differ.
        for line in lines
            .lines()
            .filter(|line| line.starts_with("# blocked:\t"))
        {
            println!("{line}");
        }
    }

    println!("# {} data rows, {} operators", rows.len(), operators.len());
    println!("{deployment}");
    println!("# walk commit: {commit}");
    println!(
        "# generated by `cargo run -p minimax-h3 --example h3_fixture -- --walk <walk print> --commit <revision>`"
    );
    for row in &rows {
        println!("{row}");
    }

    eprintln!("{} row(s) from {}", rows.len(), walk.display());
    Ok(())
}
