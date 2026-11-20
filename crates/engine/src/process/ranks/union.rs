//! How the members of an expert union ask one another to join a step, whichever process of the
//! epoch holds each of them.
//!
//! The members of an expert union run every step together (`crate::runner::causal_lm::experts`):
//! a member that enters a step with tokens of its own first asks every other member to join it,
//! and an idle member's worker takes the join batch (`Variant::ExpertJoin`) as a job. The members
//! are the deployment's ranks, one each, placed by their deployment rank
//! (`crate::process::ranks::rank::ModelRunner::ranks`); each process of the epoch holds the
//! members its host's devices serve.
//!
//! - A member this process holds is asked through its worker: the join batch is submitted to its
//!   queue.
//! - A member another process holds is asked over the union's **join link**. The formation that
//!   forms the deployment's and the union's host groups across the epoch's processes forms a
//!   third group of the same members (`crate::parallel::HostGroup::world`, its channels), whose
//!   connections carry join batches instead of collectives. An asking member writes the batch on
//!   its connection to each member of another process, and one relay thread per process reads
//!   the connections of the members it holds and submits each batch it reads to the worker of the
//!   member it arrived for.
//!
//! A join asks for a numbered step and for nothing back. A member that already took part in the
//! step finds it closed (`Member::join`), so a join that arrives late does no harm, and an asking
//! member never waits for its joins to be read: it goes on into the step's agreement over the
//! union's host group, which every member reaches once its worker took the join.
//!
//! **Failure.** A process that exits closes its connections: a member's next write to one of its
//! members fails, which fails the asking member's step, and the relay stops reading them. A join
//! the relay reads but cannot hand to its member's worker (the worker closed) is logged; the
//! asker then waits in the step's agreement, which its worker's watchdog reports, as it reports a
//! member that never enters.

use std::os::fd::AsRawFd as _;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::thread::JoinHandle;

use service::{Connection, Wake, Waker, lock};

use crate::batch::Batch;
use crate::parallel::HostGroup;
use crate::runner::causal_lm::experts::JoinHook;
use crate::worker::{Error, Result, Worker};

/// This process's members of an expert union and how each asks the others to join its steps.
pub(crate) struct Joins {
    /// The workers of the members this process holds, by place, `None` at a place another
    /// process holds; set once the workers are spawned ([`Joins::serve`]).
    workers: Arc<OnceLock<Vec<Option<Weak<Worker>>>>>,
    /// The link to the members other processes hold; `None` when this process holds every
    /// member.
    link: Option<Link>,
}

/// The union's join link of this process (module documentation).
struct Link {
    /// By place: the place's write ends of its connections to the members other processes hold,
    /// with the member each reaches. Only the worker of the place writes them; a place another
    /// process holds has none.
    posts: Vec<Mutex<Vec<(usize, Connection)>>>,
    /// The read ends, by the place held here and the member at the other end, until the relay
    /// takes them.
    readers: Mutex<Vec<(usize, usize, Connection)>>,
    /// The relay thread once it started, with the wake that ends it.
    relay: Mutex<Option<(Wake, JoinHandle<()>)>>,
}

impl Joins {
    /// The joins of a union whose every member this process holds: each asks the others through
    /// their workers alone.
    pub(crate) fn local() -> Self {
        Self {
            workers: Arc::new(OnceLock::new()),
            link: None,
        }
    }

    /// The joins of a union whose members other processes hold too, over `groups`: this process's
    /// members of the formation's group of the join link, each numbered by its place. The
    /// connections between two members of this process stay unused, since those members ask each
    /// other through their workers.
    ///
    /// # Errors
    ///
    /// A member numbered outside the union, or a connection that cannot be duplicated for its
    /// writer.
    pub(crate) fn linked(members: usize, groups: Vec<HostGroup>) -> Result<Self> {
        let held: Vec<usize> = groups.iter().map(HostGroup::rank).collect();
        let mut posts: Vec<Vec<(usize, Connection)>> = (0..members).map(|_| Vec::new()).collect();
        let mut readers = Vec::new();
        for group in groups {
            let place = group.rank();
            let written = posts.get_mut(place).ok_or_else(|| {
                Error::Invalid(format!(
                    "place {place} of an expert union of {members} members"
                ))
            })?;
            for (member, connection) in group.into_connections().into_iter().enumerate() {
                let Some(connection) = connection.filter(|_| !held.contains(&member)) else {
                    continue;
                };
                let writer = connection.writer().map_err(|error| {
                    Error::transport_source("a connection of the expert union's join link", error)
                })?;
                written.push((member, writer));
                readers.push((place, member, connection));
            }
        }

        Ok(Self {
            workers: Arc::new(OnceLock::new()),
            link: Some(Link {
                posts: posts.into_iter().map(Mutex::new).collect(),
                readers: Mutex::new(readers),
                relay: Mutex::new(None),
            }),
        })
    }

    /// The hook through which the member at `place` asks every other member to join its step:
    /// a join batch to each one's worker here, of the group that worker serves, and over the
    /// join link to each member another process holds.
    pub(crate) fn hook(self: &Arc<Self>, place: usize) -> JoinHook {
        let joins = Arc::clone(self);
        Arc::new(move |batch: Batch| joins.ask(place, &batch))
    }

    /// Sets `workers`, the workers of this process's members by place (`None` at a place another
    /// process holds), which the members ask to join their steps from then on, and starts the
    /// relay that hands them the joins of the members other processes hold.
    ///
    /// # Errors
    ///
    /// Workers set already, or a relay that cannot start.
    pub(crate) fn serve(&self, workers: Vec<Option<Weak<Worker>>>) -> Result<()> {
        self.workers
            .set(workers)
            .map_err(|_| Error::State("the expert union's workers were set already"))?;
        let Some(link) = &self.link else {
            return Ok(());
        };

        let readers = std::mem::take(&mut *lock(&link.readers));
        let waker = Waker::new()
            .map_err(|error| Error::transport_source("the expert union's relay", error))?;
        let stop = waker.wake();
        let workers = Arc::clone(&self.workers);
        let thread = std::thread::Builder::new()
            .name("union-relay".into())
            .spawn(move || relay(readers, &workers, waker))
            .map_err(|error| Error::transport_source("starting the expert union's relay", error))?;
        *lock(&link.relay) = Some((stop, thread));
        Ok(())
    }

    /// Asks every member but the one at `place` to join the step of `batch`.
    fn ask(&self, place: usize, batch: &Batch) -> Result<()> {
        let workers = self
            .workers
            .get()
            .ok_or(Error::State("the expert union's workers are not spawned"))?;
        for (member, worker) in workers.iter().enumerate() {
            let Some(worker) = worker.as_ref().filter(|_| member != place) else {
                continue;
            };
            let worker = worker
                .upgrade()
                .ok_or(Error::State("a worker of the expert union is closed"))?;
            let mut join = batch.clone();
            join.group = worker.group();
            worker.join(join)?;
        }

        match &self.link {
            Some(link) => link.post(place, batch),
            None => Ok(()),
        }
    }
}

impl Link {
    /// Writes `batch` on each connection of the member at `place` to a member another process
    /// holds, each write whole before the next: the asking member then enters the step's
    /// agreement, which waits for those members.
    fn post(&self, place: usize, batch: &Batch) -> Result<()> {
        let posts = self
            .posts
            .get(place)
            .ok_or_else(|| Error::Invariant(format!("no expert union member at place {place}")))?;
        let mut posts = lock(posts);
        if posts.is_empty() {
            return Ok(());
        }

        let frame = Asked::of(batch)?;
        for (member, connection) in posts.iter_mut() {
            connection.deliver(&frame).map_err(|error| {
                Error::transport_source(
                    format!("asking expert union member {member} to join a step"),
                    error,
                )
            })?;
        }
        Ok(())
    }
}

impl Drop for Link {
    /// Ends the relay: the workers it hands joins to are closing with the epoch.
    fn drop(&mut self) {
        if let Some((stop, thread)) = lock(&self.relay).take() {
            stop.wake();
            if thread.join().is_err() {
                tracing::warn!("the expert union's relay panicked");
            }
        }
    }
}

/// A join batch as the link carries it: the batch in the peer link's encoding (positional
/// MessagePack, `crate::executor::peers::link`), whose group the receiving member's worker sets.
struct Asked(Vec<u8>);

impl Asked {
    fn of(batch: &Batch) -> Result<Self> {
        rmp_serde::to_vec(batch).map(Self).map_err(|error| {
            Error::Invariant(format!("encoding an expert union's join batch: {error}"))
        })
    }

    fn batch(&self) -> std::io::Result<Batch> {
        rmp_serde::from_slice(&self.0).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("decoding an expert union's join batch: {error}"),
            )
        })
    }
}

impl service::Message for Asked {
    fn encode(&self) -> std::io::Result<Vec<u8>> {
        Ok(self.0.clone())
    }

    fn decode(payload: &[u8]) -> std::io::Result<Self> {
        Ok(Self(payload.to_vec()))
    }
}

/// The relay of this process's join link: reads every join batch that arrives on `readers` (by
/// the place held here, the member that wrote it and the connection) and submits it to the
/// worker of its place in `workers`, until `waker` fires. A connection whose peer closed it or
/// that carried no join batch is no longer read.
fn relay(
    mut readers: Vec<(usize, usize, Connection)>,
    workers: &OnceLock<Vec<Option<Weak<Worker>>>>,
    mut waker: Waker,
) {
    loop {
        // Every join that arrived whole goes to its member's worker.
        let mut index = 0;
        while index < readers.len() {
            let (place, from, connection) = &mut readers[index];
            match connection
                .try_recv::<Asked>()
                .and_then(|frame| frame.map(|frame| frame.batch()).transpose())
            {
                Ok(Some(batch)) => deliver(workers, *place, batch),
                Ok(None) => index += 1,
                Err(error) => {
                    tracing::info!(
                        place = *place,
                        from = *from,
                        %error,
                        "the expert union's join link to a member of another process ended"
                    );
                    readers.swap_remove(index);
                }
            }
        }

        // Sleep until a join can be read or the relay is ended.
        let mut descriptors: Vec<libc::pollfd> = readers
            .iter()
            .map(|(_, _, connection)| libc::pollfd {
                fd: connection.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            })
            .chain(std::iter::once(waker.poll_fd()))
            .collect();
        if let Err(error) = service::wait_ready(&mut descriptors, None) {
            tracing::error!(%error, "the expert union's relay cannot wait for its joins");
            return;
        }
        if descriptors.last().is_some_and(|stop| stop.revents != 0) {
            waker.drain();
            return;
        }
    }
}

/// Submits `batch`, a join another process's member asked for, to the worker of the member at
/// `place`, as a join of the group that worker serves.
fn deliver(workers: &OnceLock<Vec<Option<Weak<Worker>>>>, place: usize, mut batch: Batch) {
    let worker = workers
        .get()
        .and_then(|workers| workers.get(place))
        .and_then(|worker| worker.as_ref()?.upgrade());
    let Some(worker) = worker else {
        tracing::warn!(
            place,
            "a join reached an expert union member whose worker is closed"
        );
        return;
    };

    batch.group = worker.group();
    let step = batch.id.0;
    if let Err(error) = worker.join(batch) {
        tracing::warn!(place, step, %error, "an expert union member's worker refused a join");
    }
}
