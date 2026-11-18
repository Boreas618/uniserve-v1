//! `uniserve-profile`: reads a deployment's timeline, a profiling session's
//! files and the captures of external profilers (`docs/profiling.md`).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use profile::admin::Admin;
use profile::{calls, nsys, perfetto, session, timeline};

#[derive(Parser)]
#[command(
    name = "uniserve-profile",
    about = "Reads a deployment's timeline, profiling sessions' files and external profilers' captures"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// One trace of the `span` events a deployment's admin socket serves, in
    /// the Trace Event Format: one request's, one batch's, or those of a time
    /// window.
    Timeline {
        /// The daemon's admin address: `unix:<path>` of its admin socket.
        #[arg(long)]
        admin: String,
        /// A request's id: its spans and those of the batches it was in.
        #[arg(long, conflicts_with = "batch")]
        request: Option<String>,
        /// A batch id: its spans and its requests' membership records.
        #[arg(long)]
        batch: Option<String>,
        /// The spans ending at or after this time, in nanoseconds since the
        /// Unix epoch.
        #[arg(long)]
        from: Option<i64>,
        /// The spans starting at or before this time, in nanoseconds since
        /// the Unix epoch.
        #[arg(long)]
        to: Option<i64>,
        /// The spans ending within this many seconds before the daemon's
        /// clock.
        #[arg(long, conflicts_with_all = ["from", "to"], value_parser = seconds)]
        last: Option<Duration>,
        #[arg(short, long)]
        output: PathBuf,
    },
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

/// A positive number of seconds.
fn seconds(text: &str) -> Result<Duration, String> {
    let value: f64 = text.parse().map_err(|error| format!("{error}"))?;
    Duration::try_from_secs_f64(value)
        .ok()
        .filter(|span| !span.is_zero())
        .ok_or_else(|| format!("`{text}` is not a positive number of seconds"))
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
        Command::Timeline {
            admin,
            request,
            batch,
            from,
            to,
            last,
            output,
        } => {
            let filter = timeline::Filter {
                request,
                batch,
                from,
                to,
                last,
            };
            let spans = timeline::read(&Admin::new(&admin)?, &filter)?;
            let trace = timeline::convert(&spans)?;
            std::fs::write(&output, serde_json::to_string(&trace)?)?;
            println!("wrote {} spans to {}", spans.len(), output.display());
        }
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
