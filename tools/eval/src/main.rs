//! `uniserve-eval`: measures a serving system through its public HTTP
//! interface (`specs/eval-harness.md`).
//!
//! `run` measures the points a profile (`--config`, which the caller composes at
//! use) selects, each into a bundle; `plan` and `list` show what a profile
//! states; `compare` holds a GSM8K candidate to its reference; `check-change`,
//! `check-faults` and `check-stop` compute the heavy cases' passes from bundles.

use eval::{bundle, check, compare, load, pipeline, server, spec, stops};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};

#[derive(Parser)]
#[command(
    name = "uniserve-eval",
    about = "Measures a serving system through its public HTTP interface"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Prints a profile's servers, benchmark points and suites.
    List {
        #[arg(long)]
        config: PathBuf,
        #[arg(value_parser = ["all", "servers", "benchmarks", "suites"], default_value = "all")]
        section: String,
    },
    /// Prints the resolved plan of a benchmark or suite: an unset `${NAME}`
    /// appears as written.
    Plan {
        #[arg(long)]
        config: PathBuf,
        selection: String,
        /// Replaces the server binary, the first word of each command.
        #[arg(long)]
        executable: Option<PathBuf>,
    },
    /// Runs the selected points one after another, each into
    /// `<output root>/<point>`. Exits 2 when a point fails its validation.
    Run {
        #[arg(long)]
        config: PathBuf,
        selection: String,
        #[arg(long)]
        executable: Option<PathBuf>,
        #[arg(long)]
        output_root: Option<PathBuf>,
        #[arg(long, default_value_t = 1800.0)]
        launch_timeout_s: f64,
        #[arg(long)]
        request_timeout_s: Option<f64>,
        /// Consecutive points of one server share one deployment.
        #[arg(long)]
        reuse_deployment: bool,
        /// Measures the remaining points after an invalid one.
        #[arg(long)]
        keep_going: bool,
    },
    /// Compares a GSM8K candidate bundle with its reference. Exits 2 when the
    /// candidate is rejected, 1 when the runs are not comparable.
    Compare {
        reference: PathBuf,
        candidate: PathBuf,
        #[arg(long, default_value_t = 0.95)]
        confidence: f64,
    },
    /// The pass under change of a timeline run against its control.
    CheckChange {
        run: PathBuf,
        #[arg(long)]
        control: PathBuf,
        /// The control of a configuration a change moves to, as `NAME=PATH`.
        #[arg(long = "control-of", value_parser = named_path)]
        control_of: Vec<(String, PathBuf)>,
        #[arg(long)]
        switch: Option<String>,
        #[arg(long)]
        block_size: Option<u64>,
        #[arg(long)]
        check_metric: String,
        #[arg(long)]
        replay: PathBuf,
        #[arg(long)]
        result: Option<PathBuf>,
    },
    /// Case E6's pass of a timeline run of faults against its rows served alone.
    CheckFaults {
        run: PathBuf,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        check_metric: String,
        #[arg(long)]
        replay: PathBuf,
        #[arg(long)]
        result: Option<PathBuf>,
    },
    /// Case E2's pass of a timeline run that stopped a process.
    CheckStop {
        run: PathBuf,
        #[arg(long)]
        control: PathBuf,
        #[arg(long)]
        headroom_tokens: usize,
        #[arg(long, default_value = stops::COUNTER)]
        counter: String,
        #[arg(long)]
        listing: Option<PathBuf>,
        #[arg(long)]
        replay: Option<PathBuf>,
        #[arg(long)]
        result: Option<PathBuf>,
    },
    /// Signals the one process of this user whose command line matches the
    /// pattern on standard input (`{"process": ..., "signal": ...}`), as a
    /// change's signal on another host asks over ssh; prints the matches.
    #[command(hide = true)]
    SignalProcess,
}

fn named_path(value: &str) -> Result<(String, PathBuf), String> {
    match value.split_once('=') {
        Some((name, path)) if !name.is_empty() && !path.is_empty() => {
            Ok((name.to_owned(), PathBuf::from(path)))
        }
        _ => Err(format!("{value:?} is not NAME=PATH")),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("uniserve-eval: {error}");
            return ExitCode::from(1);
        }
    };
    match runtime.block_on(dispatch(cli.command)) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("uniserve-eval: {error:#}");
            ExitCode::from(1)
        }
    }
}

async fn dispatch(command: Command) -> Result<ExitCode> {
    match command {
        Command::List { config, section } => {
            let profile = spec::load(&config)?;
            if ["all", "servers"].contains(&section.as_str()) {
                println!("[servers]");
                for (name, server) in &profile.servers {
                    println!("{name}\t{}", server.base_urls().join(","));
                }
            }
            if ["all", "benchmarks"].contains(&section.as_str()) {
                println!("[benchmarks]");
                for (name, point) in &profile.benchmarks {
                    println!("{name}\t{}\tserver={}", point.task.as_str(), point.server);
                }
            }
            if ["all", "suites"].contains(&section.as_str()) {
                println!("[suites]");
                for (name, points) in &profile.suites {
                    println!("{name}\t{}", points.join(","));
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Plan {
            config,
            selection,
            executable,
        } => {
            let profile = spec::load(&config)?;
            let mut plan = Vec::new();
            for point in profile.selected(&selection)? {
                let server = &profile.servers[&point.server];
                let launches: Vec<spec::Launch> = server
                    .instances()
                    .iter()
                    .map(|instance| spec::launch(instance, executable.as_deref(), &profile.root))
                    .collect();
                plan.push(json!({
                    "benchmark": point.name,
                    "task": point.task.as_str(),
                    "server": server.name,
                    "server_commands": launches.iter().map(|launch| &launch.command).collect::<Vec<_>>(),
                    "server_working_directory": launches[0].working_directory,
                    "base_urls": server.base_urls(),
                    "dataset": point.dataset,
                    "num_prompts": point.load.num_prompts,
                    "request_rate": point.workload()["load"]["request_rate"],
                    "max_concurrency": point.load.max_concurrency,
                    "workload": point.workload(),
                    "metrics": point.workload()["metrics"],
                }));
            }
            println!("{}", serde_json::to_string_pretty(&plan)?);
            Ok(ExitCode::SUCCESS)
        }
        Command::Run {
            config,
            selection,
            executable,
            output_root,
            launch_timeout_s,
            request_timeout_s,
            reuse_deployment,
            keep_going,
        } => {
            run(
                &config,
                &selection,
                executable.as_deref(),
                output_root,
                launch_timeout_s,
                request_timeout_s,
                reuse_deployment,
                keep_going,
            )
            .await
        }
        Command::Compare {
            reference,
            candidate,
            confidence,
        } => match compare::compare(&reference, &candidate, confidence) {
            Ok(report) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&bundle::sorted(&report))?
                );
                Ok(if report["verdict"] == "accepted" {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(2)
                })
            }
            Err(error) => {
                eprintln!("uniserve-eval compare: {error:#}");
                Ok(ExitCode::from(1))
            }
        },
        Command::CheckChange {
            run,
            control,
            control_of,
            switch,
            block_size,
            check_metric,
            replay,
            result,
        } => {
            let configurations: BTreeMap<String, PathBuf> = control_of.iter().cloned().collect();
            anyhow::ensure!(
                configurations.len() == control_of.len(),
                "a configuration has one control"
            );
            anyhow::ensure!(
                switch.is_some() == block_size.is_some(),
                "item 6 takes both the switch and the block size"
            );
            let switch = switch.as_deref().zip(block_size);
            report(
                check::change_report(
                    &run,
                    &control,
                    &check_metric,
                    &replay,
                    &configurations,
                    switch,
                ),
                &run,
                result,
                check::render_change,
                "check-change",
            )
        }
        Command::CheckFaults {
            run,
            reference,
            check_metric,
            replay,
            result,
        } => report(
            check::fault_report(&run, &reference, &check_metric, &replay),
            &run,
            result,
            check::render_faults,
            "check-faults",
        ),
        Command::CheckStop {
            run,
            control,
            headroom_tokens,
            counter,
            listing,
            replay,
            result,
        } => report(
            stops::stop_report(
                &run,
                &control,
                headroom_tokens,
                &counter,
                listing.as_deref(),
                replay.as_deref(),
            ),
            &run,
            result,
            stops::render,
            "check-stop",
        ),
        Command::SignalProcess => {
            load::signal_process()?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Writes a check's `result.md` (into the run's bundle unless `result` names
/// another path) and prints the report as one JSON line: 0 when it passes, 2
/// when an item fails, 1 when a file cannot be read.
fn report(
    computed: Result<Value>,
    run: &Path,
    result: Option<PathBuf>,
    render: fn(&Value) -> String,
    name: &str,
) -> Result<ExitCode> {
    let report = match computed {
        Ok(report) => report,
        Err(error) => {
            eprintln!("uniserve-eval {name}: {error:#}");
            return Ok(ExitCode::from(1));
        }
    };
    let path = result.unwrap_or_else(|| run.join("result.md"));
    std::fs::write(&path, render(&report))
        .with_context(|| format!("writing {}", path.display()))?;
    println!("{}", serde_json::to_string(&bundle::sorted(&report))?);
    Ok(if report["pass"] == true {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}

/// Holds the host-wide lock that keeps one harness's servers on a host at a
/// time: an advisory lock on a file in the system's temporary directory, which
/// excludes only other harness processes.
fn host_lock() -> Result<std::fs::File> {
    let path = std::env::temp_dir().join("uniserve-eval.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    // SAFETY: flock(2) on a descriptor this function owns.
    let status = unsafe {
        libc::flock(
            std::os::fd::AsRawFd::as_raw_fd(&file),
            libc::LOCK_EX | libc::LOCK_NB,
        )
    };
    anyhow::ensure!(
        status == 0,
        "another uniserve-eval process holds the host lock"
    );
    Ok(file)
}

/// A read-only provenance command's output, empty when it fails.
fn capture(command: &[&str], directory: Option<&Path>) -> String {
    let mut process = std::process::Command::new(command[0]);
    process
        .args(&command[1..])
        .stderr(std::process::Stdio::null());
    if let Some(directory) = directory {
        process.current_dir(directory);
    }
    match process.output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }
        _ => String::new(),
    }
}

/// One process's provenance: its command, directory and environment; the
/// source revision and whether the work tree is dirty, where Git reads one; the
/// selected GPU's name and driver (one selected device, else the first listed).
fn describe(launch: &spec::Launch) -> Value {
    let mut record = json!({
        "command": launch.command,
        "working_directory": launch.working_directory,
        "environment": launch.environment,
    });
    let head = capture(
        &["git", "rev-parse", "HEAD"],
        Some(&launch.working_directory),
    );
    if !head.is_empty() {
        record["git_head"] = head.into();
        record["dirty"] = (!capture(
            &["git", "status", "--short"],
            Some(&launch.working_directory),
        )
        .is_empty())
        .into();
    }
    let mut query = vec!["nvidia-smi".to_owned()];
    if let Some(selector) = launch
        .environment
        .get("CUDA_VISIBLE_DEVICES")
        .filter(|selector| !selector.contains(',') && !selector.is_empty())
    {
        query.push(format!("--id={selector}"));
    }
    query.extend([
        "--query-gpu=name,driver_version".to_owned(),
        "--format=csv,noheader".to_owned(),
    ]);
    let words: Vec<&str> = query.iter().map(String::as_str).collect();
    if let Some((name, driver)) = capture(&words, None)
        .lines()
        .next()
        .and_then(|line| line.split_once(','))
    {
        record["gpu"] = json!({"name": name.trim(), "driver_version": driver.trim()});
    }
    record
}

#[expect(clippy::too_many_arguments, reason = "the run command's options")]
async fn run(
    config: &Path,
    selection: &str,
    executable: Option<&Path>,
    output_root: Option<PathBuf>,
    launch_timeout_s: f64,
    request_timeout_s: Option<f64>,
    reuse: bool,
    keep_going: bool,
) -> Result<ExitCode> {
    let profile = spec::load(config)?;
    let output_root = output_root
        .or_else(|| profile.artifact_root.clone())
        .context(
            "name the output directory with --output-root, or state artifact_root in the profile",
        )?;
    let points = profile.selected(selection)?;
    // Consecutive points of one server form one deployment group when asked.
    let mut groups: Vec<Vec<&spec::Point>> = Vec::new();
    for point in points {
        match groups.last_mut() {
            Some(group) if reuse && group[0].server == point.server => group.push(point),
            _ => groups.push(vec![point]),
        }
    }
    // Held for the whole selection, so another harness cannot start a server
    // between points.
    let _lock = host_lock()?;
    let mut invalid = 0;
    for group in groups {
        let first = group[0];
        let server = &profile.servers[&first.server];
        let instances = server.instances();
        let mut processes = Vec::new();
        for (index, instance) in instances.iter().enumerate() {
            let launch = spec::launch(instance, executable, &profile.root);
            spec::require_resolved(&json!(launch.command), &format!("server {}", instance.name))?;
            let suffix = if instances.len() > 1 {
                format!(".replica-{index}")
            } else {
                String::new()
            };
            let log = output_root
                .join("server-logs")
                .join(format!("{}{suffix}.log", first.name));
            processes.push(((*instance).clone(), launch, log));
        }
        for point in &group {
            spec::require_resolved(&point.workload(), &format!("benchmark {}", point.name))?;
        }
        let mut launch_record = describe(&processes[0].1);
        if processes.len() > 1 {
            launch_record["replicas"] = processes
                .iter()
                .map(|(_, launch, _)| describe(launch))
                .collect::<Vec<_>>()
                .into();
        }
        launch_record["deployment"] = json!({
            "started_for": first.name,
            "logs": processes.iter().map(|(_, _, log)| log.display().to_string()).collect::<Vec<_>>(),
        });
        // An earlier run's evidence moves out of the canonical paths first.
        for point in &group {
            let mut evidence = vec![("bundle".to_owned(), output_root.join(&point.name))];
            if let Some(media) = point
                .video
                .media_dir
                .as_deref()
                .filter(|media| !media.is_empty())
            {
                evidence.push(("media".to_owned(), PathBuf::from(media)));
            }
            if point.name == first.name
                && let Ok(entries) = std::fs::read_dir(output_root.join("server-logs"))
            {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.starts_with(&format!("{}.", first.name)) {
                        evidence.push((format!("server-logs/{name}"), entry.path()));
                    }
                }
            }
            if let Some(archive) = bundle::supersede(&output_root, &point.name, &evidence)? {
                println!("{}: earlier run moved to {}", point.name, archive.display());
            }
        }
        let mut deployment =
            server::Deployment::start(processes, Duration::from_secs_f64(launch_timeout_s)).await?;
        launch_record["deployment"]["startup_s"] = deployment.startup_s().into();
        let admin_sockets = deployment.admin_sockets();
        let acting = Arc::new(load::Deployment {
            admin_socket: admin_sockets.first().cloned().flatten(),
            pids: Arc::new(deployment.pid_reader()),
            working_directory: deployment
                .launch()
                .map(|launch| launch.working_directory.clone())
                .unwrap_or_default(),
            environment: deployment
                .launch()
                .map(|launch| launch.environment.clone())
                .unwrap_or_default(),
        });
        for point in group {
            let exited = deployment.exited();
            anyhow::ensure!(
                exited.is_empty(),
                "a deployment process exited before {}; see {}",
                point.name,
                exited[0].display()
            );
            let summary = pipeline::run_point(
                point,
                pipeline::Target {
                    base_urls: server.base_urls(),
                    admin_sockets: admin_sockets.clone(),
                    launch: launch_record.clone(),
                    timeout: request_timeout_s.map(Duration::from_secs_f64),
                    deployment: Some(Arc::clone(&acting)),
                    output: &output_root.join(&point.name),
                },
            )
            .await?;
            let valid = summary["validation"]["valid"] == true;
            println!(
                "{}: {} ({}/{} requests)",
                point.name,
                if valid { "pass" } else { "fail" },
                summary["ok_count"],
                summary["request_count"]
            );
            if !valid {
                invalid += 1;
                if !keep_going {
                    deployment.stop()?;
                    return Ok(ExitCode::from(2));
                }
            }
        }
        deployment.stop()?;
    }
    Ok(if invalid > 0 {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    })
}
