//! The processes of a deployment, from their launch until they and everything
//! they started have ended.
//!
//! Each process runs as the leader of its own session, so its group (which its
//! workers inherit) is signalled as a whole when it stops. Every process of a
//! deployment runs with `XDG_RUNTIME_DIR` set to a directory the deployment
//! creates, so the host services its processes start belong to it alone; they
//! are stopped after the processes, their logs kept beside the first process's,
//! and the directory removed.
//!
//! A process whose profile states `ready_fd` is passed the write end of a pipe
//! as `--ready-fd N` and reports its readiness records on it
//! ([`service::Readiness`]): it is ready once a record states that it serves,
//! and that record's admin endpoint is its admin socket.

use std::fs::File;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::spec::{Launch, Server};

/// How long a stopped group, or a host service, may take to exit after SIGTERM
/// before it is sent SIGKILL.
const STOP_GRACE: Duration = Duration::from_secs(60);

/// One server process.
struct Process {
    profile: Server,
    launch: Launch,
    log: PathBuf,
    child: Option<Child>,
    started: Option<Instant>,
    /// Seconds from the launch until the process was first found ready.
    startup_s: Option<f64>,
    /// The readiness records of a process that reports them, as they arrive.
    records: Option<mpsc::Receiver<Record>>,
    /// The admin socket the record that states it serves names.
    admin: Option<PathBuf>,
}

/// A readiness record, its settings kept as the server wrote them.
type Record = service::Readiness<serde_json::Value>;

/// A pipe for readiness records: its read end and its write end, both
/// close-on-exec.
fn pipe() -> Result<(OwnedFd, OwnedFd)> {
    let mut ends = [0; 2];
    // SAFETY: pipe2(2) writes two descriptors into the array it is given.
    if unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: both descriptors were just created and nothing else owns them.
    Ok(unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) })
}

/// Reads records from `read` on a thread of its own until the server closes
/// its end, which it does once it serves or when it ends.
fn read_records(read: OwnedFd) -> mpsc::Receiver<Record> {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = service::ReadyReader::new(read);
        while let Ok(Some(record)) = reader.next::<serde_json::Value>(Duration::MAX) {
            if sender.send(record).is_err() {
                break;
            }
        }
    });
    receiver
}

impl Process {
    fn start(&mut self) -> Result<()> {
        if let Some(parent) = self.log.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let log =
            File::create(&self.log).with_context(|| format!("creating {}", self.log.display()))?;
        let mut command = Command::new(&self.launch.command[0]);
        command
            .args(&self.launch.command[1..])
            .current_dir(&self.launch.working_directory)
            .envs(&self.launch.environment)
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log);
        let ready = if self.profile.ready_fd {
            let (read, write) = pipe()?;
            command.args(["--ready-fd", &write.as_raw_fd().to_string()]);
            Some((read, write))
        } else {
            None
        };
        // The descriptor the child inherits, its close-on-exec flag cleared
        // in the child alone.
        let inherited = ready.as_ref().map(|(_, write)| write.as_raw_fd());
        // A new session makes the process the leader of its own group.
        // SAFETY: setsid(2) and fcntl(2) are async-signal-safe, the only calls
        // made between fork and exec.
        unsafe {
            std::os::unix::process::CommandExt::pre_exec(&mut command, move || {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if let Some(fd) = inherited
                    && libc::fcntl(fd, libc::F_SETFD, 0) != 0
                {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        self.started = Some(Instant::now());
        self.child = Some(
            command
                .spawn()
                .with_context(|| format!("launching {}", self.launch.command[0]))?,
        );
        // The write end is the child's alone now, so its exit ends the reader.
        if let Some((read, write)) = ready {
            drop(write);
            self.records = Some(read_records(read));
        }
        Ok(())
    }

    /// Ready: a process that reports readiness records has reported that it
    /// serves; any other's listener accepts a connection and, when the profile
    /// names a readiness path, that path answers 200 (a server that listens
    /// before it can serve answers an error status there until it can).
    async fn ready(&mut self) -> bool {
        if let Some(records) = &self.records {
            while let Ok(record) = records.try_recv() {
                if record.state == service::State::Serving {
                    self.admin = record
                        .endpoints
                        .admin
                        .as_ref()
                        .and_then(|endpoint| endpoint.path())
                        .map(Path::to_path_buf);
                    self.records = None;
                    return true;
                }
            }
            return false;
        }
        if self.profile.ready_fd {
            // Its serving record was read before.
            return true;
        }
        let address = format!("{}:{}", self.profile.host, self.profile.port);
        let connected = tokio::time::timeout(
            Duration::from_secs(1),
            tokio::net::TcpStream::connect(&address),
        )
        .await;
        if !matches!(connected, Ok(Ok(_))) {
            return false;
        }
        let Some(path) = &self.profile.ready_path else {
            return true;
        };
        let client = reqwest::Client::new();
        matches!(
            client.get(format!("http://{address}{path}")).timeout(Duration::from_secs(5)).send().await,
            Ok(response) if response.status().as_u16() == 200
        )
    }

    fn exited(&mut self) -> bool {
        match &mut self.child {
            None => true,
            Some(child) => child.try_wait().ok().flatten().is_some(),
        }
    }

    /// Stops the process's group: SIGTERM whether or not the process still
    /// runs (its workers inherit the group, and after an abrupt exit nothing
    /// else stops them), SIGKILL to what remains after the grace period.
    fn stop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        // The process's pid names its group; the kernel reuses it only once the
        // group is empty.
        let group = child.id() as i32;
        signal_group(group, libc::SIGTERM);
        let deadline = Instant::now() + STOP_GRACE;
        while (child.try_wait().ok().flatten().is_none() || group_exists(group))
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(100));
        }
        if child.try_wait().ok().flatten().is_none() || group_exists(group) {
            signal_group(group, libc::SIGKILL);
        }
        let _ = child.wait();
    }
}

fn signal_group(group: i32, signal: i32) {
    // SAFETY: killpg(2) takes a group and a signal and touches no memory.
    unsafe {
        libc::killpg(group, signal);
    }
}

fn group_exists(group: i32) -> bool {
    // SAFETY: as above; signal 0 only tests for members.
    unsafe { libc::killpg(group, 0) == 0 }
}

/// Every process of one deployment: replicas launch together so their loading
/// overlaps, and the deployment is ready once all are. A failure to launch or to
/// become ready stops all of them, as does dropping the deployment.
pub struct Deployment {
    processes: Vec<Process>,
    runtime: Option<tempfile::TempDir>,
}

impl Deployment {
    /// Launches every process and waits until all are ready, within `timeout`.
    ///
    /// # Errors
    ///
    /// When a process exits before it is ready, or one is not ready in time;
    /// every process is stopped first.
    pub async fn start(
        processes: Vec<(Server, Launch, PathBuf)>,
        timeout: Duration,
    ) -> Result<Self> {
        // A short directory under the system's temporary directory keeps the
        // services' Unix socket paths within the kernel's limit.
        let runtime = tempfile::Builder::new()
            .prefix("uniserve-eval-")
            .tempdir()?;
        let mut deployment = Self {
            processes: processes
                .into_iter()
                .map(|(profile, mut launch, log)| {
                    launch.environment.insert(
                        "XDG_RUNTIME_DIR".into(),
                        runtime.path().display().to_string(),
                    );
                    Process {
                        profile,
                        launch,
                        log,
                        child: None,
                        started: None,
                        startup_s: None,
                        records: None,
                        admin: None,
                    }
                })
                .collect(),
            runtime: Some(runtime),
        };
        if let Err(error) = deployment.ready(timeout).await {
            deployment.stop()?;
            return Err(error);
        }
        Ok(deployment)
    }

    async fn ready(&mut self, timeout: Duration) -> Result<()> {
        for process in &mut self.processes {
            process.start()?;
        }
        let deadline = Instant::now() + timeout;
        for process in &mut self.processes {
            loop {
                if process.exited() {
                    let code = process
                        .child
                        .as_mut()
                        .and_then(|child| child.try_wait().ok().flatten());
                    bail!(
                        "server exited with {}; see {}",
                        code.map_or("an unknown status".into(), |status| status.to_string()),
                        process.log.display()
                    );
                }
                if process.ready().await {
                    process.startup_s = process
                        .started
                        .map(|started| started.elapsed().as_secs_f64());
                    break;
                }
                if Instant::now() >= deadline {
                    bail!(
                        "server was not ready on {}:{} within {}s",
                        process.profile.host,
                        process.profile.port,
                        timeout.as_secs_f64()
                    );
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
        Ok(())
    }

    /// Seconds from the launch until every process was ready: the slowest's.
    pub fn startup_s(&self) -> Option<f64> {
        self.processes
            .iter()
            .map(|process| process.startup_s)
            .collect::<Option<Vec<f64>>>()
            .map(|times| times.into_iter().fold(0.0, f64::max))
    }

    /// Each process's admin socket, as its readiness record named it; `None`
    /// for a process that reports no records.
    pub fn admin_sockets(&self) -> Vec<Option<PathBuf>> {
        self.processes
            .iter()
            .map(|process| process.admin.clone())
            .collect()
    }

    /// The first process's working directory and environment, which a change's
    /// command runs with.
    pub fn launch(&self) -> Option<&Launch> {
        self.processes.first().map(|process| &process.launch)
    }

    /// The logs of the processes that no longer run.
    pub fn exited(&mut self) -> Vec<PathBuf> {
        self.processes
            .iter_mut()
            .filter_map(|process| process.exited().then(|| process.log.clone()))
            .collect()
    }

    /// The deployment's processes in increasing order: the members of each
    /// server's group, every descendant of a server (which includes the
    /// processes it starts in groups of their own), and the host services serving
    /// inside its runtime directory, which run in sessions of their own.
    pub fn pids(&self) -> Vec<u32> {
        let roots: Vec<u32> = self
            .processes
            .iter()
            .filter_map(|process| process.child.as_ref().map(Child::id))
            .collect();
        let runtime = self
            .runtime
            .as_ref()
            .map(|runtime| runtime.path().join("uniserve"));
        deployment_pids(&roots, runtime.as_deref())
    }

    /// A reader of [`Deployment::pids`] that a timeline's changes call at their
    /// time, when the deployment's processes are the ones running then.
    pub fn pid_reader(&self) -> impl Fn() -> Vec<u32> + Send + Sync + 'static {
        let roots: Vec<u32> = self
            .processes
            .iter()
            .filter_map(|process| process.child.as_ref().map(Child::id))
            .collect();
        let runtime = self
            .runtime
            .as_ref()
            .map(|runtime| runtime.path().join("uniserve"));
        move || deployment_pids(&roots, runtime.as_deref())
    }

    /// Stops every process, then the host services they started, keeping each
    /// service's log beside the first process's as `<log stem>.<service>.log`.
    ///
    /// # Errors
    ///
    /// When a host service still runs after SIGKILL: it holds the deployment's
    /// storage, so no later point may start on this host.
    pub fn stop(&mut self) -> Result<()> {
        for process in &mut self.processes {
            process.stop();
        }
        let Some(runtime) = self.runtime.take() else {
            return Ok(());
        };
        let services = runtime.path().join("uniserve");
        let stopped = stop_services(&services);
        if let Some(first) = self.processes.first()
            && let Ok(entries) = std::fs::read_dir(&services)
        {
            let stem = first
                .log
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default();
            let mut logs: Vec<PathBuf> = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|extension| extension == "log"))
                .collect();
            logs.sort();
            for log in logs {
                let name = log
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let _ = std::fs::copy(&log, first.log.with_file_name(format!("{stem}.{name}")));
            }
        }
        drop(runtime);
        stopped
    }
}

impl Drop for Deployment {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// One `/proc/<pid>/stat`'s parent and group.
fn stat(pid: u32) -> Option<(u32, u32, char)> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // After the parenthesized command name: the state, the parent, the group.
    let fields: Vec<&str> = text.rsplit_once(')')?.1.split_whitespace().collect();
    Some((
        fields.get(1)?.parse().ok()?,
        fields.get(2)?.parse().ok()?,
        fields.first()?.chars().next()?,
    ))
}

fn processes() -> Vec<u32> {
    std::fs::read_dir("/proc")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse().ok())
        .collect()
}

fn deployment_pids(roots: &[u32], services: Option<&Path>) -> Vec<u32> {
    let mut parents = std::collections::BTreeMap::new();
    let mut members = std::collections::BTreeSet::new();
    for pid in processes() {
        let Some((parent, group, _)) = stat(pid) else {
            continue;
        };
        parents.insert(pid, parent);
        if roots.contains(&group) {
            members.insert(pid);
        }
    }
    // A process belongs to the deployment when a server is among its ancestors;
    // the walk stops at the first known answer.
    let mut known: std::collections::BTreeMap<u32, bool> =
        roots.iter().map(|root| (*root, true)).collect();
    for pid in parents.keys().copied().collect::<Vec<_>>() {
        let mut chain = Vec::new();
        let mut current = pid;
        while !known.contains_key(&current) {
            let Some(parent) = parents.get(&current) else {
                break;
            };
            chain.push(current);
            current = *parent;
        }
        let belongs = known.get(&current).copied().unwrap_or(false);
        for member in chain {
            known.insert(member, belongs);
        }
        if belongs {
            members.insert(pid);
        }
    }
    if let Some(services) = services {
        members.extend(service_pids(services));
    }
    members.into_iter().collect()
}

/// The processes serving a socket inside `services`: a host service runs as
/// `<executable> <service> serve --socket <path>`.
fn service_pids(services: &Path) -> Vec<u32> {
    let prefix = format!("{}/", services.display());
    processes()
        .into_iter()
        .filter(|pid| {
            let Ok(command) = std::fs::read(format!("/proc/{pid}/cmdline")) else {
                return false;
            };
            let arguments: Vec<&[u8]> = command.split(|byte| *byte == 0).collect();
            arguments.contains(&b"serve".as_slice())
                && arguments
                    .iter()
                    .any(|argument| argument.starts_with(prefix.as_bytes()))
        })
        .collect()
}

fn alive(pid: u32) -> bool {
    stat(pid).is_some_and(|(_, _, state)| state != 'Z')
}

/// Stops every host service serving inside `services` and waits for it: SIGTERM,
/// then SIGKILL to one still running after the grace period. The services are
/// not children of this process, so their exit is their disappearance from
/// `/proc`.
fn stop_services(services: &Path) -> Result<()> {
    let mut pids = service_pids(services);
    for signal in [libc::SIGTERM, libc::SIGKILL] {
        for pid in &pids {
            // SAFETY: kill(2) touches no memory of this process.
            unsafe {
                libc::kill(*pid as i32, signal);
            }
        }
        let deadline = Instant::now() + STOP_GRACE;
        while !pids.is_empty() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(100));
            pids.retain(|pid| alive(*pid));
        }
        if pids.is_empty() {
            return Ok(());
        }
    }
    bail!(
        "host services {pids:?} of {} did not exit after SIGKILL",
        services.display()
    )
}
