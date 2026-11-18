//! The server's side of a handoff's import (`storage::runtime::Request::Import`): the spans
//! from a lender's units into the importing owner's, which reshard the pages' heads between the
//! two replicas' layouts ([`crate::handoff`]), run as one pool task.
//!
//! - **A lender of this host** lends in this host's pool: its ranks' places are this server's,
//!   and the pool holds the blocks the task reads. A read and a push are the same copy here.
//! - **A lender of another host** is borrowed from its Storage Server
//!   ([`storage::protocol::Request::Borrow`]) by a helper, which keeps the connection, and with
//!   it the borrow, until the task ended, then returns the lend
//!   ([`storage::protocol::Request::Return`]). A read runs on the Transfer Server of each
//!   destination's host, which READs the lender's units; a push runs on the lender's host,
//!   whose Transfer Server WRITEs them into this host's units, which it reaches at the address
//!   the connection leaves this host from.
//!
//! Every import ends its lend once its task ended, whatever its outcome: a lend of this host by
//! the pool ([`crate::buffer_pool::BufferPool::give_back`]), one of another host by the
//! helper's return.

use std::collections::BTreeMap;
use std::io;
use std::sync::mpsc;

use storage::protocol::{Lent, LentRank, Region, Span, UnitRun};
use storage::runtime::OwnerId;
use storage::{Client, Result};

use super::hosts::Here;
use super::polling::Notifier;
use super::{PoolTask, Running, Session};
use crate::Server;
use crate::buffer_pool::{ImportWork, NamedRun};
use crate::handoff::{self, Page, Placed, Rank};

/// The spans of an import, by the host whose Transfer Server runs them (`None` for this one).
type Tasks = BTreeMap<Option<String>, Vec<(Span, Here)>>;

/// One rank of a side of an import as the server knows it: its host (`None` for this one), its
/// runs and its layout.
type Known = (
    Option<String>,
    Vec<UnitRun>,
    Option<storage::protocol::UnitLayout>,
);

impl Server {
    /// An import of the pool task `task` of `session` starts ([`ImportWork`]).
    pub(crate) fn start_import(&mut self, session: Session, task: PoolTask, work: ImportWork) {
        let number = self.sessions.next;
        self.sessions.next += 1;
        let mut running = Running::new(session.clone(), task, Vec::new());
        let lender = work.lender.clone();
        if lender.endpoint.is_none() {
            running.lend = Some((lender.session, lender.owner));
        }
        self.sessions.running.insert(number, running);

        let started = match &lender.endpoint {
            None => self.import_local(&session, number, &work),
            Some(endpoint) => self.borrow_remote(&session, number, endpoint, work),
        };
        match started {
            Ok(true) => {}
            Ok(false) => self.end_pool_task(number, true),
            Err(reason) => {
                tracing::warn!(deployment = session.0, %reason, "a handoff's import cannot run");
                self.end_pool_task(number, false);
            }
        }
    }

    /// The import from a lender of this host: its places are this server's.
    fn import_local(
        &mut self,
        session: &Session,
        number: u64,
        work: &ImportWork,
    ) -> std::result::Result<bool, String> {
        if work.pages.is_empty() {
            return Ok(false);
        }
        let memory = work
            .lender_memory
            .as_deref()
            .ok_or("the lender's owner is gone")?;
        let region = region_of(work.pages.iter().map(|(.., source)| source))?;
        let source: BTreeMap<u32, Known> = self
            .ranks(&session.0, &region, memory)
            .into_iter()
            .map(|(part, placed)| (part, placed.clone()))
            .collect();
        let pages: Vec<(u8, Vec<NamedRun>, Vec<NamedRun>)> = work
            .pages
            .iter()
            .map(|(class, _, destination, source)| (*class, source.clone(), destination.clone()))
            .collect();
        let tasks = self.import_tasks(session, work, &source, &pages, None)?;
        self.run_import(session, number, tasks)
    }

    /// The import from a lender of another host: a helper borrows the lend there
    /// ([`Self::borrowed`]) and returns it once the task ended.
    fn borrow_remote(
        &mut self,
        session: &Session,
        number: u64,
        endpoint: &str,
        work: ImportWork,
    ) -> std::result::Result<bool, String> {
        let (release, released) = mpsc::channel();
        borrow_at(
            self.sessions.thread.notifier(),
            number,
            (endpoint.to_owned(), session.0.clone()),
            work,
            released,
        )
        .map_err(|error| format!("the borrow of a handoff's lend cannot start: {error}"))?;
        if let Some(running) = self.sessions.running.get_mut(&number) {
            running.release = Some(release);
        }
        Ok(true)
    }

    /// The helper of the import `number` borrowed its lend at the Storage Server at
    /// `endpoint` (`lent`), or could not: the import's spans run, or the task ends as failed.
    pub(crate) fn borrowed(
        &mut self,
        number: u64,
        endpoint: &str,
        work: &ImportWork,
        lent: std::result::Result<Lent, String>,
    ) {
        let Some(session) = self
            .sessions
            .running
            .get(&number)
            .map(|running| running.session.clone())
        else {
            return;
        };
        let started = lent.and_then(|lent| {
            if work.pages.is_empty() {
                return Ok(false);
            }
            let wanted = |class: u8, position: u32| {
                lent.blocks
                    .iter()
                    .find(|(of, at, _)| *of == class && *at == position)
                    .map(|(.., runs)| runs.clone())
            };
            let mut pages = Vec::with_capacity(work.pages.len());
            for (class, position, destination, _) in &work.pages {
                let source = wanted(*class, *position)
                    .ok_or_else(|| format!("the lend holds no page {position} of class {class}"))?;
                pages.push((*class, source, destination.clone()));
            }
            let region = region_of(pages.iter().map(|(_, source, _)| source))?;
            // A rank the lender's server states without a host is that server's.
            let source: BTreeMap<u32, Known> = lent
                .ranks
                .iter()
                .filter(|rank| rank.region == region)
                .map(
                    |LentRank {
                         part,
                         endpoint: at,
                         runs,
                         layout,
                         ..
                     }| {
                        let host = at.clone().or_else(|| Some(endpoint.to_owned()));
                        (*part, (host, runs.clone(), layout.clone()))
                    },
                )
                .collect();
            let tasks = self.import_tasks(&session, work, &source, &pages, Some(endpoint))?;
            self.run_import(&session, number, tasks)
        });
        match started {
            Ok(true) => {}
            Ok(false) => self.end_pool_task(number, true),
            Err(reason) => {
                tracing::warn!(deployment = session.0, %reason, "a handoff's import cannot run");
                self.end_pool_task(number, false);
            }
        }
    }

    /// The spans of an import, by the host that runs them: a read on each destination's host, a
    /// push on each source's. `pages` are, per page, its class, the lender's runs and the
    /// importing owner's; `source` the lender's ranks by part; `lender` the host of the
    /// lender's server, `None` for this one.
    fn import_tasks(
        &self,
        session: &Session,
        work: &ImportWork,
        source: &BTreeMap<u32, Known>,
        pages: &[(u8, Vec<NamedRun>, Vec<NamedRun>)],
        lender: Option<&str>,
    ) -> std::result::Result<Tasks, String> {
        let region = region_of(pages.iter().map(|(.., destination)| destination))?;
        let destination: BTreeMap<u32, Known> = self
            .ranks(&session.0, &region, &work.memory)
            .into_iter()
            .map(|(part, placed)| (part, placed.clone()))
            .collect();
        let (from, to) = (sides(source, "lender")?, sides(&destination, "importer")?);
        if from.is_empty() || to.is_empty() {
            return Err("a side of the handoff placed no unit".into());
        }
        let pages: Vec<Page> = pages
            .iter()
            .map(|(class, source, destination)| Page {
                class: *class,
                source: units(source),
                destination: units(destination),
            })
            .collect();
        let placed = handoff::spans(&from, &to, &pages)?;

        let mut tasks = Tasks::new();
        for Placed {
            source,
            destination,
            mut span,
        } in placed
        {
            // A push runs where the lender's units are; a read where the importer's are.
            let runs_on = if work.push {
                source.clone().or_else(|| lender.map(str::to_owned))
            } else {
                destination.clone()
            };
            let mut here = Here::Neither;
            let transport = work.transport;
            let mut name = |end: Option<String>, region: Region, destination: bool| match end {
                end if end == runs_on => region,
                Some(endpoint) => Region::Remote {
                    endpoint,
                    region: Box::new(region),
                    transport,
                },
                // This host's storage, which the host that runs the span reaches by address.
                None => {
                    here = if destination {
                        Here::Destination(transport)
                    } else {
                        Here::Source(transport)
                    };
                    region
                }
            };
            span.source = name(source, span.source, false);
            span.destination = name(destination, span.destination, true);
            tasks.entry(runs_on).or_default().push((span, here));
        }
        Ok(tasks)
    }

    /// Runs an import's spans on the Transfer Servers of their hosts. Returns whether any run.
    fn run_import(
        &mut self,
        session: &Session,
        number: u64,
        tasks: Tasks,
    ) -> std::result::Result<bool, String> {
        if tasks.is_empty() {
            return Ok(false);
        }
        self.spread(session, number, tasks)?;
        Ok(true)
    }

    /// The helper of the import `number` returned its lend: the task ends.
    pub(crate) fn import_returned(&mut self, number: u64) {
        self.end_pool_parts(number, 0, true);
    }

    /// See [`storage::protocol::Request::Borrow`], from the peer `id`.
    pub(crate) fn borrow_lend(
        &mut self,
        id: crate::ClientId,
        deployment: &str,
        (session, owner): (u32, u64),
    ) -> std::result::Result<crate::Handling, (storage::protocol::ErrorKind, String)> {
        let named = deployment.to_owned();
        let borrowed = self
            .sessions
            .thread
            .call(move |pools| pools.borrow(&named, id, session, OwnerId(owner)))
            .map_err(crate::resource)?;
        let Some((memory, blocks)) = borrowed else {
            return Err(crate::invalid(&format!(
                "no lend of owner {owner} of session {session} stands"
            )));
        };
        let regions: std::collections::BTreeSet<String> = blocks
            .iter()
            .flat_map(|(.., runs)| runs.iter().map(|(region, ..)| region.clone()))
            .collect();
        let ranks = regions
            .iter()
            .flat_map(|region| {
                self.ranks(deployment, region, &memory)
                    .into_iter()
                    .map(|(part, (endpoint, runs, layout))| LentRank {
                        region: region.clone(),
                        part,
                        endpoint: endpoint.clone(),
                        runs: runs.clone(),
                        layout: layout.clone(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok(crate::Handling::Response(
            storage::protocol::Response::Borrowed {
                lent: Lent { blocks, ranks },
            },
            Vec::new(),
        ))
    }

    /// See [`storage::protocol::Request::Return`], from the peer `id`.
    pub(crate) fn return_lend(
        &mut self,
        id: crate::ClientId,
        deployment: &str,
        (session, owner): (u32, u64),
    ) -> std::result::Result<crate::Handling, (storage::protocol::ErrorKind, String)> {
        let named = deployment.to_owned();
        self.sessions.thread.send(move |pools| {
            pools.release_borrows(&named, id, Some((session, OwnerId(owner))));
            pools.give_back(&named, session, OwnerId(owner));
        });
        Ok(crate::Handling::Response(
            storage::protocol::Response::Returned,
            Vec::new(),
        ))
    }
}

/// The ranks of one side of an import as [`handoff::spans`] takes them.
fn sides<'a>(
    ranks: &'a BTreeMap<u32, Known>,
    side: &str,
) -> std::result::Result<BTreeMap<u32, Rank<'a>>, String> {
    ranks
        .iter()
        .map(|(part, (host, runs, layout))| {
            let layout = layout.as_ref().ok_or_else(|| {
                format!("rank {part} of the {side} placed no layout of its units")
            })?;
            Ok((
                *part,
                Rank {
                    endpoint: host.as_deref(),
                    runs,
                    layout,
                },
            ))
        })
        .collect()
}

/// The one region the runs of `blocks` lie in.
fn region_of<'a>(
    mut blocks: impl Iterator<Item = &'a Vec<NamedRun>>,
) -> std::result::Result<String, String> {
    let mut region: Option<&str> = None;
    blocks.try_for_each(|runs| {
        for (name, ..) in runs {
            match region {
                Some(held) if held != name => {
                    return Err(format!("a handoff's pages lie in {held} and {name}"));
                }
                _ => region = Some(name),
            }
        }
        Ok(())
    })?;
    region
        .map(str::to_owned)
        .ok_or_else(|| "a handoff's pages lie in no region".to_owned())
}

/// The units of runs, in order.
fn units(runs: &[NamedRun]) -> Vec<u32> {
    runs.iter()
        .flat_map(|(_, first, count)| *first..first + count)
        .collect()
}

/// A helper that borrows the lend of `work.lender` at the Storage Server at `endpoint` for the
/// import `number` of `deployment` ([`Server::borrowed`]), keeps the borrow until `release`
/// fires or is dropped, then returns the lend and ends the task ([`Server::import_returned`]).
fn borrow_at(
    notifier: Notifier,
    number: u64,
    (endpoint, deployment): (String, String),
    work: ImportWork,
    release: mpsc::Receiver<()>,
) -> io::Result<()> {
    std::thread::Builder::new()
        .name("lend-borrow".into())
        .spawn(move || {
            let (session, owner) = (work.lender.session, work.lender.owner.0);
            let borrowed: Result<(Client, Lent)> =
                Client::connect_peer(&endpoint).and_then(|mut client| {
                    let lent = client.borrow(deployment.clone(), session, owner)?;
                    Ok((client, lent))
                });
            let (client, lent) = match borrowed {
                Ok((client, lent)) => (Some(client), Ok(lent)),
                Err(error) => (
                    None,
                    Err(format!("borrowing the lend at {endpoint}: {error}")),
                ),
            };
            let notified = endpoint.clone();
            notifier.send(Box::new(move |server: &mut Server| {
                server.borrowed(number, &notified, &work, lent);
            }));

            // The task's end drops the sender; the lend ends then.
            let _ = release.recv();
            if let Some(mut client) = client
                && let Err(error) = client.give_back(deployment, session, owner)
            {
                tracing::warn!(endpoint, %error, "returning a handoff's lend failed");
            }
            notifier.send(Box::new(move |server: &mut Server| {
                server.import_returned(number);
            }));
        })?;
    Ok(())
}
