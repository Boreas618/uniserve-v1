//! The Storage Server's integration tests' helper executable.
//!
//! A test's server runs its children and a restart's successor from this executable, which
//! serves the `storage-server` command itself ([`storage_server::command`]): the processes are
//! the real ones. A test also starts the client processes its server follows from it, each a
//! process of its own because the server follows clients by their processes:
//!
//! - `writer`: a worker that puts an exclusive tensor and holds write access to it until it is
//!   killed;
//! - `engine`: an engine that opens a session, attaches an owner, closes its connection and stays
//!   until its standard input reaches its end.
//!
//! A client says it is ready by writing one byte to its standard output, a pipe its test reads.

use std::io::{Read as _, Write as _};
use std::os::fd::AsFd as _;
use std::path::PathBuf;
use std::time::Duration;

use ::storage::protocol::{Access, Claim, Key, Location, Role, Scope, StorageRecord, TensorRecord};
use ::storage::runtime::{Completion, GroupId, Open, OwnerId, Request, RequestId};
use ::storage::{Client, Interest, QueuePair};
use clap::Parser as _;

/// The helper's command line: the `storage-server` command, or a client.
#[derive(Debug, clap::Parser)]
struct Helper {
    /// The level every target logs at.
    #[arg(long, global = true)]
    log_level: Option<tracing::level_filters::LevelFilter>,
    #[command(subcommand)]
    role: HelperRole,
}

#[derive(Debug, clap::Subcommand)]
enum HelperRole {
    #[command(flatten)]
    Server(storage_server::command::Command),
    /// A worker of `scope` (deployment `model`) under `claim` that puts the exclusive tensor
    /// `key`, `bytes` bytes each `fill`, and holds write access until it is killed.
    Writer {
        #[arg(long)]
        socket: PathBuf,
        #[arg(long)]
        group: String,
        #[arg(long)]
        rank: u32,
        /// The claim, as hexadecimal.
        #[arg(long)]
        claim: String,
        /// The tensor's key, as hexadecimal.
        #[arg(long)]
        key: String,
        #[arg(long)]
        bytes: u64,
        #[arg(long)]
        fill: u8,
    },
    /// An engine of epoch `epoch` of deployment `model` that opens the session `open` states
    /// (its MessagePack, as hexadecimal), attaches owner 1 to group 0 through its first
    /// scheduler's queue pair, closes its connection and stays until its standard input ends.
    Engine {
        #[arg(long)]
        socket: PathBuf,
        #[arg(long)]
        epoch: u64,
        #[arg(long)]
        open: String,
    },
}

/// How long a client waits for the server's completion before it fails.
const PATIENCE: Duration = Duration::from_secs(30);

fn main() -> anyhow::Result<()> {
    let helper = Helper::parse();
    match helper.role {
        HelperRole::Server(command) => storage_server::command::run(
            storage_server::command::Args { command },
            &service::logging::Levels::new(helper.log_level),
        ),
        HelperRole::Writer {
            socket,
            group,
            rank,
            claim,
            key,
            bytes,
            fill,
        } => writer(
            &socket,
            Scope {
                deployment: "model".into(),
                group,
                rank,
            },
            &claim,
            &key,
            (bytes, fill),
        ),
        HelperRole::Engine {
            socket,
            epoch,
            open,
        } => engine(&socket, epoch, &open),
    }
}

/// The bytes of `text`, hexadecimal.
fn unhex(text: &str) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        text.len().is_multiple_of(2),
        "{text} is no hexadecimal byte string"
    );
    (0..text.len())
        .step_by(2)
        .map(|at| Ok(u8::from_str_radix(&text[at..at + 2], 16)?))
        .collect()
}

/// Says the client is ready: one byte on standard output.
fn ready() -> anyhow::Result<()> {
    let mut out = std::io::stdout().lock();
    out.write_all(&[1])?;
    out.flush()?;
    Ok(())
}

fn writer(
    socket: &std::path::Path,
    scope: Scope,
    claim: &str,
    key: &str,
    (bytes, fill): (u64, u8),
) -> anyhow::Result<()> {
    let claim = Claim(
        unhex(claim)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("a claim is 16 bytes"))?,
    );
    let key = Key(unhex(key)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("a key is 32 bytes"))?);
    let mut client = Client::connect(
        socket,
        Role::Worker {
            scope,
            generation: 1,
            claim,
            epoch: 0,
        },
    )?;
    let storage = rustix::fs::memfd_create("storage-writer", rustix::fs::MemfdFlags::CLOEXEC)?;
    let mut file = std::fs::File::from(storage);
    file.write_all(&vec![fill; usize::try_from(bytes)?])?;
    client.put(
        vec![TensorRecord {
            key,
            name: "latents".into(),
            storage: 0,
            offset: 0,
            dtype: "uint8".into(),
            shape: vec![bytes],
            strides: vec![1],
            access: Access::Exclusive,
        }],
        &[(
            StorageRecord {
                location: Location::Host { numa_node: None },
                bytes,
                segment: None,
                fabric: None,
            },
            file.as_fd(),
        )],
    )?;
    ready()?;
    // Holds the tensor until the server's shoot-down kills the process.
    loop {
        std::thread::park();
    }
}

fn engine(socket: &std::path::Path, epoch: u64, open: &str) -> anyhow::Result<()> {
    let open: Open = ::storage::protocol::decode(&unhex(open)?)?;
    let mut engine = Client::connect(
        socket,
        Role::Head {
            deployment: "model".into(),
            epoch,
            generation: 0,
        },
    )?;
    let (_, descriptors) = engine.open_pool(open)?;
    let descriptor = descriptors
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("the session has no scheduler"))?;
    let mut pair = QueuePair::open(descriptor, true)?;
    let mut record = Vec::new();
    Request::Attach {
        owner: OwnerId(1),
        group: GroupId(0),
    }
    .encode(RequestId(1), &mut record);
    anyhow::ensure!(pair.submit(&record)?, "the submission queue is full");
    let completion = loop {
        let seen = pair.signals();
        if pair.next_completion(&mut record)? {
            break Completion::decode(&record)?;
        }
        anyhow::ensure!(
            pair.wait(Interest::RECORDS, seen, Some(PATIENCE))?,
            "the attach was not completed"
        );
    };
    anyhow::ensure!(
        matches!(completion, (RequestId(1), Completion::Attached { .. })),
        "the attach completed as {completion:?}"
    );
    // The server has acted on the close once this returns.
    engine.close()?;
    ready()?;
    // Stays, with the queue pair mapped, until the test closes standard input.
    let mut rest = Vec::new();
    std::io::stdin().read_to_end(&mut rest)?;
    drop(pair);
    Ok(())
}
