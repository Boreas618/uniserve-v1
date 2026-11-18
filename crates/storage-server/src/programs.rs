//! The deployments' KV programs on this server (`specs/kv-cache.md` §5): one instance of each
//! deployment's program, run by the program host every component of a deployment shares
//! (`program`), at its two hooks.
//!
//! - **Handoff**, on the decode side's server, for every prefill-to-decode handoff
//!   ([`Request::Handoff`](::storage::protocol::Request::Handoff)): the program chooses how the
//!   full pages reach the decode replica. The answer is local, a call into the instance on the
//!   server's own thread, so the handoff's critical path takes no network hop.
//! - **Tiering**, every period on the leader node's server
//!   ([`KvProgram::tiering`]): the program reads the cluster directory, which that server holds
//!   (`crate::directory`: every node's device and host pages, and Mooncake Store's), and gives
//!   the moves to make. The server carries out what it can itself: a copy of a host page of its
//!   own node into Mooncake Store, through its connector. A page on a device moves through the
//!   engine that maps it, a page read into host memory takes a block only a scheduler's prefetch
//!   allocates, and a page of another node is that node's server's to move, so every other move
//!   is skipped with its reason.
//!
//! The server checks every output and takes the default rule on one it cannot use: a handoff
//! pushes over the first transport both sides have. A program that fails (a panic, past its fuel
//! or memory, a trap) ends its instance and the default rules take over until its head sets a
//! program again; what it decided before stays. What the program logs, its end and every output
//! the server refused or skipped are kept as records its head drains
//! ([`Request::ProgramRecords`](::storage::protocol::Request::ProgramRecords)) into the
//! deployment's `events` table.
//!
//! The program's fields are its state. The server saves it with the program's arguments as
//! `<name>.state` in the directory its head names (the node's `program-state/`, the file the
//! daemon keeps its own programs' states in, `{"args": ..., "state": ...}`) after every period
//! (the tiering step's on the leader's node), before a new instance replaces it, and when the
//! server ends or hands over to a successor. A new instance is restored from it when it was saved
//! under the same arguments and decodes as the program's type: a new module of the program, or a
//! new server, goes on from it, while new arguments start the program anew, since a program keeps
//! the arguments it was created with in its fields.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::os::fd::{AsFd, BorrowedFd};
use std::path::{Path as FilePath, PathBuf};
use std::time::{Duration, SystemTime};

use ::storage::protocol::{KvProgram, ProgramEvent, ProgramRecord, ProgramStart};
use program::{Budget, Error, Host, Instance, Kind, Level, Module};
use scheduling::kv::{
    Directory, Handoff, Hook, Key, Location, Move, Path, Tier, Tiering, Transport,
};

use crate::timer::Timer;

/// What each call into a KV program runs within. A tiering step decodes the directory, a
/// thousand pages of keys in a few hundred kilobytes, within the fuel each byte adds.
const BUDGET: Budget = Budget {
    fuel: 200_000_000,
    per_byte: 200,
    memory: 256 << 20,
};

/// The most records a deployment's program keeps for its head; past it the oldest go.
const RECORDS: usize = 4096;

/// The embedder of a KV program: what it logs, kept as records of its program.
pub(crate) struct Logs {
    program: String,
    records: Vec<ProgramRecord>,
}

impl Host for Logs {
    fn log(&mut self, level: Level, text: &str) {
        let event = ProgramEvent::Log {
            level,
            text: text.to_owned(),
        };
        self.records.push(record(&self.program, event));
    }
}

/// A record of `program` now.
fn record(program: &str, event: ProgramEvent) -> ProgramRecord {
    ProgramRecord {
        time: now(),
        program: program.to_owned(),
        event,
    }
}

/// Nanoseconds since the Unix epoch: the deployment's clock.
fn now() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
        })
}

/// One deployment's program on this server.
struct Running {
    /// What its head set.
    setup: KvProgram,
    /// The instance; `None` once it ended, until the head sets a program again.
    instance: Option<Instance<Logs>>,
    /// Fires every period: the tiering step, and the save of the state.
    timer: Timer,
    /// What the head has not drained yet, oldest first.
    records: VecDeque<ProgramRecord>,
}

impl Running {
    /// Keeps `records`, the oldest going past [`RECORDS`].
    fn keep(&mut self, records: impl IntoIterator<Item = ProgramRecord>) {
        self.records.extend(records);
        let excess = self.records.len().saturating_sub(RECORDS);
        self.records.drain(..excess);
    }

    /// Moves what the instance logged into the records.
    fn collect(&mut self) {
        let logged = self
            .instance
            .as_mut()
            .map(|instance| std::mem::take(&mut instance.host_mut().records))
            .unwrap_or_default();
        self.keep(logged);
    }

    /// Records `event` of the program.
    fn note(&mut self, event: ProgramEvent) {
        let noted = record(&self.setup.name, event);
        self.keep([noted]);
    }

    /// Ends the instance after `error`, recording why.
    fn end(&mut self, error: &Error) {
        self.collect();
        self.instance = None;
        let reason = error.to_string();
        tracing::warn!(program = %self.setup.name, %reason, "the KV program ended");
        self.note(ProgramEvent::Ended { reason });
    }

    /// One step of the instance at `hook`; `None` when there is none or it ended in the step.
    fn step<O: serde::de::DeserializeOwned>(
        &mut self,
        hook: &Hook<&Handoff, &Tiering>,
    ) -> Option<O> {
        let stepped = self.instance.as_mut()?.step::<_, O>(hook);
        match stepped {
            Ok(output) => {
                self.collect();
                Some(output)
            }
            Err(error) => {
                self.end(&error);
                None
            }
        }
    }

    /// Saves the instance's state under the program's name, atomically: a crash leaves the
    /// previous state or this one. An instance that fails to save ends.
    fn save(&mut self) {
        let Some(instance) = self.instance.as_mut() else {
            return;
        };
        let state = match instance.save() {
            Ok(state) => state,
            Err(error) => {
                self.end(&error);
                return;
            }
        };
        let written = write_state(
            &self.setup.state,
            &self.setup.name,
            &self.setup.args,
            &state,
        );
        if let Err(error) = written {
            tracing::warn!(program = %self.setup.name, %error, "the KV program's state was not saved");
        }
    }
}

/// The file the state of the program `name` is saved in.
fn state_file(directory: &FilePath, name: &str) -> PathBuf {
    directory.join(format!("{name}.state"))
}

/// A saved state as its file holds it: the arguments the program ran with, and the JSON of its
/// fields, as its `save` replied them.
#[derive(serde::Serialize, serde::Deserialize)]
struct Saved {
    args: serde_json::Value,
    state: serde_json::Value,
}

/// Writes `state`, the reply of the program's `save`, as the saved state of `name`, created
/// from `args`, in `directory`: a temporary file renamed over it once synced.
fn write_state(
    directory: &FilePath,
    name: &str,
    args: &serde_json::Value,
    state: &[u8],
) -> io::Result<()> {
    let saved = Saved {
        args: args.clone(),
        state: serde_json::from_slice(state)?,
    };
    std::fs::create_dir_all(directory)?;
    let staging = directory.join(format!(".{name}.state.new"));
    std::fs::write(&staging, serde_json::to_vec(&saved)?)?;
    std::fs::File::open(&staging)?.sync_all()?;
    std::fs::rename(&staging, state_file(directory, name))
}

/// The state of `setup`'s program saved under the arguments it now has, as its `restore` takes
/// it; `None` when none was saved, or it was saved under other arguments.
///
/// # Errors
///
/// The file cannot be read, or holds no saved state.
fn read_state(setup: &KvProgram) -> Result<Option<Vec<u8>>, String> {
    let file = state_file(&setup.state, &setup.name);
    let text = match std::fs::read(&file) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("reading {}: {error}", file.display())),
    };
    let saved: Saved = serde_json::from_slice(&text)
        .map_err(|error| format!("{} holds no saved state: {error}", file.display()))?;
    if saved.args != setup.args {
        return Ok(None);
    }
    serde_json::to_vec(&saved.state)
        .map(Some)
        .map_err(|error| error.to_string())
}

/// The deployments' KV programs.
#[derive(Default)]
pub(crate) struct Programs {
    running: HashMap<String, Running>,
    /// The holder number the pool knows the last move by: each move holds its copies under a
    /// number of its own.
    next_holder: u64,
    /// Each move under way: its deployment, the connector tasks still running and the keys
    /// whose pages did not all arrive.
    moves: HashMap<u64, (String, usize, Vec<Key>)>,
}

/// A move of the tiering hook that this server carries out: copies of host pages into
/// Mooncake Store.
pub(crate) struct Upload {
    pub(crate) keys: Vec<::storage::protocol::Key>,
}

impl Programs {
    /// Runs `setup` as `deployment`'s program, or none (see
    /// [`Request::Program`](::storage::protocol::Request::Program)).
    ///
    /// # Errors
    ///
    /// The reason the module does not load or the arguments are not the program's; nothing
    /// changed then.
    pub(crate) fn set(
        &mut self,
        deployment: &str,
        setup: Option<KvProgram>,
    ) -> Result<Option<ProgramStart>, String> {
        let Some(setup) = setup else {
            if let Some(mut running) = self.running.remove(deployment) {
                running.save();
            }
            return Ok(None);
        };
        let period = Duration::from_millis(setup.period_ms.max(1));

        // The same module, name and arguments: the instance goes on with the new period and
        // hooks. An instance that ended is started again.
        if let Some(running) = self.running.get_mut(deployment)
            && running.instance.is_some()
            && running.setup.module == setup.module
            && running.setup.name == setup.name
            && running.setup.args == setup.args
        {
            running.setup = setup;
            running
                .timer
                .arm(period)
                .map_err(|error| error.to_string())?;
            return Ok(Some(ProgramStart::Kept));
        }

        let module = Module::<Logs>::new(&setup.module).map_err(|error| error.to_string())?;
        if module.kind() != Kind::Kv {
            return Err(format!(
                "the module is a {} program, not a KV program",
                module.kind().name()
            ));
        }
        // The instance this one replaces leaves its state for it.
        if let Some(running) = self.running.get_mut(deployment) {
            running.save();
        }
        let (instance, start) = instantiate(&module, &setup)?;
        let timer = Timer::new().map_err(|error| error.to_string())?;
        timer.arm(period).map_err(|error| error.to_string())?;

        let records = self
            .running
            .remove(deployment)
            .map(|running| running.records)
            .unwrap_or_default();
        tracing::info!(deployment, program = %setup.name, ?start, "the KV program runs");
        self.running.insert(
            deployment.to_owned(),
            Running {
                setup,
                instance: Some(instance),
                timer,
                records,
            },
        );
        Ok(Some(start))
    }

    /// How the full pages of `handoff` reach the decode replica: `deployment`'s program's
    /// path when the handoff can take it, and otherwise a push over the first transport both
    /// sides have.
    ///
    /// # Errors
    ///
    /// The sides have no transport in common: the tail is pushed or read whatever the full
    /// pages do.
    pub(crate) fn handoff(&mut self, deployment: &str, handoff: &Handoff) -> Result<Path, String> {
        let Some(&transport) = handoff.transports.first() else {
            return Err("the two sides of the handoff have no transport in common".to_owned());
        };
        let default = Path::Push(transport);
        let Some(running) = self.running.get_mut(deployment) else {
            return Ok(default);
        };
        let Some(path) = running.step::<Path>(&Hook::Handoff(handoff)) else {
            return Ok(default);
        };

        match refusal(handoff, path) {
            None => Ok(path),
            Some(reason) => {
                running.note(ProgramEvent::Refused {
                    hook: "handoff".to_owned(),
                    reason,
                });
                Ok(default)
            }
        }
    }

    /// What every program logged and how its outputs fared, of `deployment`, since the last
    /// call.
    pub(crate) fn drain(&mut self, deployment: &str) -> Vec<ProgramRecord> {
        self.running
            .get_mut(deployment)
            .map(|running| running.records.drain(..).collect())
            .unwrap_or_default()
    }

    /// The timers of the programs, which the server's loop waits on.
    pub(crate) fn timers(&self) -> impl Iterator<Item = BorrowedFd<'_>> {
        self.running.values().map(|running| running.timer.as_fd())
    }

    /// The deployments whose period came, each armed for its next: each program's state is
    /// saved, and the deployments returned step their tiering hook.
    pub(crate) fn due(&mut self) -> Vec<String> {
        let mut tiering = Vec::new();
        for (deployment, running) in &mut self.running {
            if !running.timer.fired() {
                continue;
            }
            let period = Duration::from_millis(running.setup.period_ms.max(1));
            if let Err(error) = running.timer.arm(period) {
                tracing::warn!(deployment, %error, "the KV program's period is not armed again");
            }
            if running.setup.tiering && running.instance.is_some() {
                tiering.push(deployment.clone());
            } else {
                running.save();
            }
        }
        tiering
    }

    /// One tiering step of `deployment`'s program over `directory`, the cluster directory with
    /// this node first, and its state saved after it: the moves this server carries out. Every other move is
    /// recorded, refused when the directory has no such place, skipped when it is one this
    /// server cannot move between.
    pub(crate) fn tiering(&mut self, deployment: &str, directory: Directory) -> Upload {
        let mut upload = Upload { keys: Vec::new() };
        let Some(running) = self.running.get_mut(deployment) else {
            return upload;
        };
        let tiering = Tiering {
            time: scheduling::time::Instant(now()),
            directory,
        };
        let moves = running.step::<Vec<Move>>(&Hook::Tiering(&tiering));
        running.save();

        for found in moves.unwrap_or_default() {
            match carried(&tiering.directory, found) {
                Carried::Upload(key) => upload.keys.push(::storage::protocol::Key(key.0)),
                Carried::Refused(reason) => running.note(ProgramEvent::Refused {
                    hook: "tiering".to_owned(),
                    reason,
                }),
                Carried::Skipped(reason) => running.note(ProgramEvent::Skipped { reason }),
            }
        }
        upload
    }

    /// Records, of `deployment`'s program, that the move of `key` was skipped, and why.
    pub(crate) fn skipped(
        &mut self,
        deployment: &str,
        key: &::storage::protocol::Key,
        reason: &str,
    ) {
        if let Some(running) = self.running.get_mut(deployment) {
            let reason = format!("{}: {reason}", key.hex());
            running.note(ProgramEvent::Skipped { reason });
        }
    }

    /// A holder number for a move of `deployment`'s, whose pool holds the copies it writes.
    pub(crate) fn holder(&mut self) -> u64 {
        self.next_holder += 1;
        self.next_holder
    }

    /// A move holding its copies under `holder` started `tasks` connector tasks.
    pub(crate) fn moving(&mut self, deployment: &str, holder: u64, tasks: usize) {
        self.moves
            .insert(holder, (deployment.to_owned(), tasks, Vec::new()));
    }

    /// A connector task of the move under `holder` ended, each page of `keys` moved as
    /// `results` states. Returns the deployment whose pool holds the move's copies once its last
    /// task ended, when the holds end.
    pub(crate) fn moved(
        &mut self,
        holder: u64,
        keys: &[::storage::protocol::Key],
        results: &[bool],
    ) -> Option<String> {
        let (_, tasks, failed) = self.moves.get_mut(&holder)?;
        *tasks = tasks.saturating_sub(1);
        let missed = keys
            .iter()
            .zip(results.iter().chain(std::iter::repeat(&false)))
            .filter(|(_, moved)| !**moved)
            .map(|(key, _)| Key(key.0));
        failed.extend(missed);
        if *tasks > 0 {
            return None;
        }

        let (deployment, _, failed) = self.moves.remove(&holder)?;
        if !failed.is_empty() {
            let reason = format!(
                "{} page(s) of the move did not reach Mooncake Store",
                failed.len()
            );
            if let Some(running) = self.running.get_mut(&deployment) {
                running.note(ProgramEvent::Skipped { reason });
            }
        }
        Some(deployment)
    }

    /// Saves every program's state: the server ends or hands over.
    pub(crate) fn save_all(&mut self) {
        for running in self.running.values_mut() {
            running.save();
        }
    }

    /// Every deployment's program as its head set it, which a successor runs again.
    pub(crate) fn setups(&self) -> Vec<(String, KvProgram)> {
        self.running
            .iter()
            .map(|(deployment, running)| (deployment.clone(), running.setup.clone()))
            .collect()
    }

    /// The programs a predecessor ran, each restored from the state it saved before it handed
    /// over; one that no longer loads is left out, logged.
    pub(crate) fn resume(setups: Vec<(String, KvProgram)>) -> Self {
        let mut programs = Self::default();
        for (deployment, setup) in setups {
            if let Err(error) = programs.set(&deployment, Some(setup)) {
                tracing::warn!(deployment, %error, "the predecessor's KV program does not run again");
            }
        }
        programs
    }
}

/// A new instance of `module` for `setup`: restored from the program's state saved under its
/// arguments when that state decodes as its type, and created from its arguments otherwise.
fn instantiate(
    module: &Module<Logs>,
    setup: &KvProgram,
) -> Result<(Instance<Logs>, ProgramStart), String> {
    let logs = || Logs {
        program: setup.name.clone(),
        records: Vec::new(),
    };
    if let Some(saved) = read_state(setup)? {
        match Instance::restore(module, logs(), &BUDGET, &saved) {
            Ok(instance) => return Ok((instance, ProgramStart::Restored)),
            // A state of another type is the program's previous version's: it starts anew.
            Err(Error::State(reason)) => {
                tracing::info!(program = %setup.name, %reason, "the saved state is not the program's type");
            }
            Err(error) => return Err(error.to_string()),
        }
    }
    let instance =
        Instance::new(module, logs(), &BUDGET, &setup.args).map_err(|error| error.to_string())?;
    Ok((instance, ProgramStart::Created))
}

/// Why `handoff` cannot take `path`; `None` when it can.
fn refusal(handoff: &Handoff, path: Path) -> Option<String> {
    match path {
        Path::Push(transport) | Path::Read(transport)
            if !handoff.transports.contains(&transport) =>
        {
            Some(format!(
                "{} is not a transport both sides have",
                transport_name(transport)
            ))
        }
        Path::Push(_) | Path::Read(_) => None,
        Path::Pull(Tier::Device) => {
            Some("a pull reads the node's host memory or Mooncake Store, never a device".to_owned())
        }
        Path::Pull(tier) => {
            let held = handoff
                .found
                .iter()
                .filter(|found| found.tier == tier)
                .map(|found| found.pages)
                .max()
                .unwrap_or(0);
            (held < handoff.pages).then(|| {
                format!(
                    "the {tier:?} tier holds {held} of the {} full pages",
                    handoff.pages
                )
            })
        }
    }
}

fn transport_name(transport: Transport) -> &'static str {
    match transport {
        Transport::Nixl => "NIXL",
        Transport::Mooncake => "Mooncake Transfer Engine",
    }
}

/// What the server does with one move of the tiering hook.
enum Carried {
    /// A copy of a host page of this node into Mooncake Store.
    Upload(Key),
    /// A move between places the directory does not have.
    Refused(String),
    /// A move this server cannot make.
    Skipped(String),
}

/// What the server does with `found`, a move over `directory`.
fn carried(directory: &Directory, found: Move) -> Carried {
    let Move {
        key,
        from,
        to,
        copy,
    } = found;
    let named = |location: Location| directory.nodes.get(location.node).map(|node| &node.name);
    let (Some(_), Some(_)) = (named(from), named(to)) else {
        return Carried::Refused(format!(
            "the move of {key:?} names a node the directory of {} node(s) does not",
            directory.nodes.len()
        ));
    };
    if from == to {
        return Carried::Refused(format!("the move of {key:?} goes nowhere"));
    }
    let local = |location: Location| location.node == 0;
    match (from.tier, to.tier) {
        (Tier::Host, Tier::Store) if local(from) => {
            if !copy {
                tracing::debug!(
                    ?key,
                    "the host copy stays until the pool evicts it: a move to Mooncake Store copies"
                );
            }
            Carried::Upload(key)
        }
        (Tier::Device, _) | (_, Tier::Device) => Carried::Skipped(format!(
            "the move of {key:?} touches device memory, which only its engine moves"
        )),
        (_, Tier::Host) => Carried::Skipped(format!(
            "the move of {key:?} reads into host memory, whose blocks only a scheduler's prefetch \
             allocates"
        )),
        (Tier::Host, Tier::Store) => Carried::Skipped(format!(
            "the move of {key:?} copies a host page of node {}, whose own server holds it",
            from.node
        )),
        _ => Carried::Skipped(format!(
            "the move of {key:?} goes between nodes or out of Mooncake Store, which no server \
             carries out"
        )),
    }
}

impl Drop for Programs {
    /// The server ends: every program leaves its state for the next instance.
    fn drop(&mut self) {
        self.save_all();
    }
}
