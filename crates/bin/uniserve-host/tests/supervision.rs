//! Host supervision through its connection to the head and real child
//! processes.
//!
//! Each test binds a listener that stands in for the head, starts the built
//! `uniserve-host` binary against it, and exchanges the launcher's frames:
//! its registration, instructions to the launcher, and reservation replies
//! and rank exit reports back.

#![cfg(test)]

use std::collections::HashMap;
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use uniserve_ipc::launcher::{Instruction, RankExit, RankSpawn, Report, Reservation};
use uniserve_ipc::{Channel, Listener, Registration, VERSION};

/// Kills and reaps the launcher process when a test ends, including when it
/// panics.
struct Launcher(Child);

impl Drop for Launcher {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Waits for the launcher's next report, panicking if none arrives within
/// thirty seconds or the launcher closes the connection.
fn receive(head: &mut Channel) -> Report {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(report) = head.try_recv().unwrap() {
            return report;
        }
        assert!(Instant::now() < deadline, "the launcher sent no report");
        head.wait(Duration::from_millis(50)).unwrap();
    }
}

/// Waits for the launcher's answer to a reservation.
fn reservation(head: &mut Channel) -> Reservation {
    match receive(head) {
        Report::Reservation(reservation) => reservation,
        other => panic!("expected a reservation, received {other:?}"),
    }
}

/// Waits for the launcher's report of a rank's exit.
fn exit(head: &mut Channel) -> RankExit {
    match receive(head) {
        Report::Exit(exit) => exit,
        other => panic!("expected an exit, received {other:?}"),
    }
}

/// The launch of one rank of `worker_id`, run by `python`.
fn spawn(worker_id: &str, rank: u32, generation: u64, python: &str) -> RankSpawn {
    RankSpawn {
        rank,
        worker_id: worker_id.into(),
        generation,
        python: python.into(),
        head: "127.0.0.1:1".into(),
        environment: HashMap::new(),
    }
}

/// Starts a launcher against a head bound here and takes its registration.
///
/// Returns the launcher guard and the head's end of the connection.
fn start_launcher() -> (Launcher, Channel) {
    start_launcher_with(Stdio::null(), Stdio::inherit())
}

/// `start_launcher` with the launcher's log output sent to `stdout` and
/// `stderr`.
fn start_launcher_with(stdout: Stdio, stderr: Stdio) -> (Launcher, Channel) {
    let listener = Listener::bind(&["127.0.0.1:0"]).unwrap();
    let launcher = Launcher(
        Command::new(env!("CARGO_BIN_EXE_uniserve-host"))
            .args([
                "--head",
                &listener.addresses()[0],
                "--host-identity",
                "host",
            ])
            .stdout(stdout)
            .stderr(stderr)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut head = loop {
        if let Some(head) = listener.try_accept().unwrap() {
            break head;
        }
        assert!(Instant::now() < deadline, "the launcher did not connect");
        std::thread::sleep(Duration::from_millis(5));
    };

    // The launcher's first frame says which host it owns and what it was
    // built as.
    let deadline = Instant::now() + Duration::from_secs(30);
    let registration: Registration = loop {
        if let Some(registration) = head.try_recv().unwrap() {
            break registration;
        }
        assert!(Instant::now() < deadline, "the launcher did not register");
        head.wait(Duration::from_millis(50)).unwrap();
    };
    assert_eq!(
        registration,
        Registration::Launcher {
            host: "host".into(),
            version: VERSION.into(),
        }
    );
    (launcher, head)
}

#[test]
fn groups_reserve_distinct_ports_and_exits_arrive_without_another_instruction() {
    let (_launcher, mut head) = start_launcher();

    // The head reserves for one group at a time and reads the answer before
    // it sends anything else.
    let mut ports = Vec::new();
    for worker in ["first", "second"] {
        head.send(&Instruction::Reserve {
            worker_id: worker.into(),
        })
        .unwrap();
        ports.push(reservation(&mut head).port);
    }
    assert_ne!(ports[0], ports[1]);

    // /bin/false is a real process with a known exit status; numerical worker
    // initialization is irrelevant to the host's independent exit reporting.
    head.send(&Instruction::Spawn(Box::new(spawn(
        "first",
        0,
        3,
        "/bin/false",
    ))))
    .unwrap();

    // No further instruction is sent: the report must come from the
    // launcher's own polling of its ranks. It names the generation the
    // rank's spawn carried, which is how the head tells it from an exit of
    // the same rank before its group was relaunched.
    assert_eq!(
        exit(&mut head),
        RankExit {
            worker_id: "first".into(),
            generation: 3,
            rank: 0,
            status: "exit status: 1".into(),
        }
    );
}

#[test]
fn the_first_rank_serves_its_store_on_the_reserved_port() {
    let (_launcher, mut head) = start_launcher();
    head.send(&Instruction::Reserve {
        worker_id: "group".into(),
    })
    .unwrap();
    let reserved = reservation(&mut head);

    // The rank's module accepts one peer on the socket it inherits at the
    // number the reservation reported, which is how a collective store serves
    // on an inherited socket, and answers it with the launch generation its
    // command line names, which a rank registers under. The head names the
    // socket's number in the rank's launch descriptor; this stand-in is given
    // it in its environment. The package is regular so it shadows any
    // installed worker package.
    let modules = tempfile::tempdir().unwrap();
    let package = modules.path().join("uniserve");
    std::fs::create_dir(&package).unwrap();
    std::fs::write(package.join("__init__.py"), "").unwrap();
    std::fs::write(
        package.join("worker.py"),
        "import os, socket, sys\n\
         store = socket.socket(fileno=int(os.environ['STORE_FD']))\n\
         peer, _ = store.accept()\n\
         generation = sys.argv[sys.argv.index('--generation') + 1]\n\
         peer.sendall(f'served {generation}\\n'.encode())\n",
    )
    .unwrap();
    let mut rank = spawn("group", 0, 4, "python3");
    rank.environment = HashMap::from([
        (
            "PYTHONPATH".to_owned(),
            modules.path().to_str().unwrap().to_owned(),
        ),
        ("STORE_FD".to_owned(), reserved.listen_fd.to_string()),
    ]);
    head.send(&Instruction::Spawn(Box::new(rank))).unwrap();

    // The reserved socket is already listening, so this connection waits in
    // its backlog until the rank starts and accepts it; the generous read
    // timeout covers the interpreter's startup.
    let peer = TcpStream::connect(("127.0.0.1", reserved.port)).unwrap();
    peer.set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let mut served = String::new();
    std::io::BufRead::read_line(&mut std::io::BufReader::new(peer), &mut served).unwrap();
    assert_eq!(served, "served 4\n");

    assert_eq!(
        exit(&mut head),
        RankExit {
            worker_id: "group".into(),
            generation: 4,
            rank: 0,
            status: "exit status: 0".into(),
        }
    );
    // The launcher kept no copy of the socket it handed over, so the port
    // closes with the rank that served it.
    assert_eq!(
        TcpStream::connect(("127.0.0.1", reserved.port))
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::ConnectionRefused
    );
}

/// A long-running stand-in rank started by a launcher, killed when a test
/// ends so a failing test leaves no process behind.
#[cfg(target_os = "linux")]
struct StandIn {
    pid: libc::pid_t,
    _directory: tempfile::TempDir,
}

#[cfg(target_os = "linux")]
impl StandIn {
    /// Has the launcher behind `head` spawn rank 1 of group `group` as a
    /// shell script that records its PID and then sleeps under that PID, and
    /// waits for it to start.
    fn spawn(head: &mut Channel, group: &str) -> Self {
        use std::os::unix::fs::PermissionsExt;

        // The launcher runs `<python> -m uniserve.worker ...`; the
        // script ignores those arguments.
        let directory = tempfile::tempdir().unwrap();
        let script = directory.path().join("rank");
        let pid_file = directory.path().join("rank.pid");
        std::fs::write(
            &script,
            "#!/bin/sh\necho $$ > \"$RANK_PID_FILE\"\nexec sleep 600\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut rank = spawn(group, 1, 0, script.to_str().unwrap());
        rank.environment = HashMap::from([(
            "RANK_PID_FILE".to_owned(),
            pid_file.to_str().unwrap().to_owned(),
        )]);
        head.send(&Instruction::Spawn(Box::new(rank))).unwrap();

        let mut pid = None;
        assert!(
            eventually(|| {
                pid = std::fs::read_to_string(&pid_file)
                    .ok()
                    .and_then(|text| text.trim().parse().ok());
                pid.is_some()
            }),
            "the launcher did not start the rank"
        );
        Self {
            pid: pid.unwrap(),
            _directory: directory,
        }
    }

    /// Whether the rank exists and is not a zombie awaiting its reaper.
    fn alive(&self) -> bool {
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{}/stat", self.pid)) else {
            return false;
        };
        // The state letter follows the parenthesized command name.
        let state = stat.rsplit(')').next().unwrap_or("").trim_start();
        !state.starts_with('Z')
    }
}

#[cfg(target_os = "linux")]
impl Drop for StandIn {
    fn drop(&mut self) {
        if self.alive() {
            // SAFETY: signal delivery to the stand-in this test started.
            unsafe { libc::kill(self.pid, libc::SIGKILL) };
        }
    }
}

/// Polls `condition` for up to 20 s, the bound for a launcher or rank to
/// react, and returns its last value.
#[cfg(target_os = "linux")]
fn eventually(mut condition: impl FnMut() -> bool) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(20) {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    condition()
}

#[cfg(target_os = "linux")]
#[test]
fn a_killed_launcher_takes_its_ranks_with_it() {
    let (mut launcher, mut head) = start_launcher();
    let rank = StandIn::spawn(&mut head, "group");
    assert!(rank.alive());

    // SIGKILL skips every exit path of the launcher's own.
    launcher.0.kill().unwrap();
    launcher.0.wait().unwrap();

    assert!(
        eventually(|| !rank.alive()),
        "the rank outlived its launcher"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn a_closed_log_pipe_does_not_end_supervision() {
    let (mut launcher, mut head) = start_launcher_with(Stdio::piped(), Stdio::piped());
    let rank = StandIn::spawn(&mut head, "group");

    // Closing the read ends, as a dropped SSH session does, makes every
    // later log write fail.
    drop(launcher.0.stdout.take());
    drop(launcher.0.stderr.take());

    // The launcher still follows the head: a stop ends the group's rank, and
    // the head closing the connection ends the launcher cleanly.
    head.send(&Instruction::Stop {
        worker_id: "group".into(),
    })
    .unwrap();
    assert!(
        eventually(|| !rank.alive()),
        "the stop did not end the rank"
    );
    assert_eq!(
        launcher.0.try_wait().unwrap(),
        None,
        "the launcher ended with its log pipe"
    );

    drop(head);
    let mut status = None;
    assert!(eventually(|| {
        status = launcher.0.try_wait().unwrap();
        status.is_some()
    }));
    assert!(
        status.unwrap().success(),
        "the launcher failed: {:?}",
        status.unwrap()
    );
}

/// The stand-in rank module of `a_reaped_rank_leaves_no_event_memory`: it
/// creates an interprocess event, as a rank does when it exports a device
/// product, which makes the CUDA driver create shared memory named for this
/// process, records its process id and waits to be killed.
#[cfg(target_os = "linux")]
const EVENT_HOLDER: &str = "\
import ctypes, os, time
cuda = ctypes.CDLL('libcuda.so.1')
device, context, event = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_void_p()
assert cuda.cuInit(0) == 0
assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0
assert cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) == 0
assert cuda.cuCtxPushCurrent_v2(context) == 0
# CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS
assert cuda.cuEventCreate(ctypes.byref(event), 2 | 4) == 0
with open(os.environ['RANK_PID_FILE'] + '.tmp', 'w') as file:
    file.write(str(os.getpid()))
os.rename(os.environ['RANK_PID_FILE'] + '.tmp', os.environ['RANK_PID_FILE'])
time.sleep(600)
";

/// The interprocess-event memory the driver created for process `pid`, by
/// its name in `/dev/shm` (`cuda.shm.<uid in hex>.<pid in hex>.<n>`).
#[cfg(target_os = "linux")]
fn event_memory(pid: libc::pid_t) -> Vec<std::path::PathBuf> {
    // SAFETY: getuid has no preconditions.
    let prefix = format!("cuda.shm.{:x}.{pid:x}.", unsafe { libc::getuid() });
    std::fs::read_dir("/dev/shm")
        .unwrap()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
        .map(|entry| entry.path())
        .collect()
}

/// Has the launcher behind `head` spawn rank 0 of `group` as the
/// [`EVENT_HOLDER`] module, and returns its process id once its event memory
/// exists, with that memory.
#[cfg(target_os = "linux")]
fn hold_event(
    head: &mut Channel,
    group: &str,
    modules: &std::path::Path,
) -> (libc::pid_t, Vec<std::path::PathBuf>) {
    let pid_file = modules.join(format!("{group}.pid"));
    let mut rank = spawn(group, 0, 7, "python3");
    rank.environment = HashMap::from([
        (
            "PYTHONPATH".to_owned(),
            modules.to_str().unwrap().to_owned(),
        ),
        (
            "RANK_PID_FILE".to_owned(),
            pid_file.to_str().unwrap().to_owned(),
        ),
    ]);
    head.send(&Instruction::Spawn(Box::new(rank))).unwrap();

    let mut held = None;
    assert!(
        eventually(|| {
            held = std::fs::read_to_string(&pid_file)
                .ok()
                .and_then(|text| text.trim().parse().ok())
                .map(|pid| (pid, event_memory(pid)))
                .filter(|(_, memory)| !memory.is_empty());
            held.is_some()
        }),
        "the rank of {group} created no event memory"
    );
    held.unwrap()
}

/// A rank killed outside the launcher cannot destroy its CUDA contexts, so
/// the driver's interprocess-event memory named for it stays in `/dev/shm`.
/// The launcher removes exactly that memory when it reaps the rank, before
/// it reports the exit, and leaves a live rank's; a stop that kills a rank
/// removes that rank's memory too.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a CUDA device"]
fn a_reaped_rank_leaves_no_event_memory() {
    let (_launcher, mut head) = start_launcher();
    // The package is regular so it shadows any installed worker package.
    let modules = tempfile::tempdir().unwrap();
    let package = modules.path().join("uniserve");
    std::fs::create_dir(&package).unwrap();
    std::fs::write(package.join("__init__.py"), "").unwrap();
    std::fs::write(package.join("worker.py"), EVENT_HOLDER).unwrap();
    let (killed, killed_memory) = hold_event(&mut head, "killed", modules.path());
    // Should the test fail, dropping the launcher kills both stand-ins.
    let (_, live_memory) = hold_event(&mut head, "live", modules.path());

    // SAFETY: signal delivery to a stand-in this test started.
    assert_eq!(unsafe { libc::kill(killed, libc::SIGKILL) }, 0);
    let report = exit(&mut head);
    assert_eq!(
        (report.worker_id.as_str(), report.generation, report.rank),
        ("killed", 7, 0)
    );
    assert!(
        report.status.contains("SIGKILL"),
        "unexpected exit status {}",
        report.status
    );
    assert!(
        killed_memory.iter().all(|path| !path.exists()),
        "{killed_memory:?} outlived its rank"
    );
    assert!(
        live_memory.iter().all(|path| path.exists()),
        "a live rank lost {live_memory:?}"
    );

    head.send(&Instruction::Stop {
        worker_id: "live".into(),
    })
    .unwrap();
    assert!(
        eventually(|| live_memory.iter().all(|path| !path.exists())),
        "{live_memory:?} outlived its stopped rank"
    );
}
