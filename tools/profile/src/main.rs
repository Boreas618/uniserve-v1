//! `uniserve-profile`: reads a profiling session's files and the captures of
//! external profilers (`docs/profiling.md`).

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};
use profile::{calls, nsys, perfetto, session};

#[derive(Parser)]
#[command(
    name = "uniserve-profile",
    about = "Reads profiling sessions' files and external profilers' captures"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// What each file of a session holds: its session, worker, collectors and
    /// rows by table.
    Summary { files: Vec<PathBuf> },
    /// One trace of a session's files in the Trace Event Format.
    Perfetto {
        files: Vec<PathBuf>,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Advisory hints from an Nsight Systems SQLite export.
    Trace {
        capture: PathBuf,
        /// The NVTX name prefix of the ranges whose calls are inspected.
        #[arg(long, default_value = "uniserve.")]
        scope: String,
        /// The rules whose hints are kept (repeat to keep more); all by default.
        #[arg(long, value_parser = nsys::RULES)]
        rule: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// Repeated call edges between Python and native functions, from a cProfile
    /// capture.
    Calls {
        capture: PathBuf,
        /// A native function name's substring (repeat to select more).
        #[arg(long, required = true)]
        native: Vec<String>,
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(2)
        }
    }
}

fn run(command: Command) -> Result<()> {
    match command {
        Command::Summary { files } => {
            let summaries: Vec<serde_json::Value> = session::open_all(&files)?
                .iter()
                .map(session::ProfileFile::summary)
                .collect::<Result<_>>()?;
            println!("{}", serde_json::to_string_pretty(&summaries)?);
        }
        Command::Perfetto { files, output } => {
            let trace = perfetto::convert(&files)?;
            std::fs::write(&output, serde_json::to_string(&trace)?)?;
            let count = trace["traceEvents"].as_array().map_or(0, Vec::len);
            println!("wrote {count} events to {}", output.display());
        }
        Command::Trace {
            capture,
            scope,
            rule,
            json,
        } => {
            let mut report = nsys::analyze(&capture, &scope)?;
            if !rule.is_empty() {
                report.hints.retain(|hint| rule.contains(&hint.rule));
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
                return Ok(());
            }
            let coverage: Vec<String> = report
                .coverage
                .iter()
                .map(|(name, count)| format!("{name}={count}"))
                .collect();
            println!("Coverage: {}", coverage.join(", "));
            for hint in &report.hints {
                println!("\n[{}] {}", hint.rule, hint.observation);
                println!("  {}", hint.suggestion);
                for event in &hint.evidence {
                    println!(
                        "  {}..{} ns ({:.3} us) tid={} batch={} correlation={} {}",
                        event.start_ns,
                        event.end_ns,
                        (event.end_ns - event.start_ns) as f64 / 1000.0,
                        event.global_tid,
                        event
                            .scope
                            .batch_id
                            .map_or_else(|| "None".to_owned(), |batch| batch.to_string()),
                        event.correlation_id,
                        event.name
                    );
                    println!("    scope: {}", event.scope.name);
                    for state in &event.gil {
                        println!(
                            "    {}: {}..{} ns tid={}",
                            state.name, state.start_ns, state.end_ns, state.global_tid
                        );
                    }
                    for copy in &event.copies {
                        println!(
                            "    GPU {}..{} ns: {} {} B device={} stream={} {}->{}",
                            copy.start_ns,
                            copy.end_ns,
                            copy.direction,
                            copy.bytes,
                            copy.device,
                            copy.stream,
                            copy.source_storage,
                            copy.destination_storage
                        );
                    }
                    if event.stack.is_empty() {
                        println!("    stack: not captured");
                    }
                    for frame in &event.stack {
                        println!("    {frame}");
                    }
                }
            }
            println!();
            for note in &report.notes {
                println!("{note}");
            }
        }
        Command::Calls {
            capture,
            native,
            json,
        } => {
            let graph = calls::analyze(&capture, &native)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&graph)?);
                return Ok(());
            }
            println!(
                "Recorded native functions: {}",
                graph.native_functions.len()
            );
            for edge in &graph.hints {
                println!(
                    "\n[{}] {} calls, {:.6} s inclusive\n  {}\n  -> {}",
                    edge.direction, edge.calls, edge.total_seconds, edge.caller, edge.callee
                );
            }
            println!();
            for note in &graph.notes {
                println!("{note}");
            }
        }
    }
    Ok(())
}
