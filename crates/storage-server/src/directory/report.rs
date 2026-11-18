//! The helper that sends this node's reports to the cluster directory on the leader node's
//! Storage Server.
//!
//! The server's loop never waits on another host: it hands each period's reports to a thread of
//! its own, which keeps one connection to the leader's server as a peer and sends each report
//! there ([`Request::Holdings`](::storage::protocol::Request::Holdings)). A report that cannot be
//! sent (the leader's server is not reachable, ended or restarted) drops the connection, and the
//! next period's report connects again; the leader's directory forgot the node when the
//! connection closed, and that report states it again. A thread behind on its reports sends only
//! the latest of each deployment: each replaces the one before at the leader anyway.

use std::collections::BTreeMap;
use std::io;
use std::sync::mpsc;

use ::storage::Client;
use ::storage::protocol::Holding;

/// One node's report of one deployment.
pub(crate) struct Report {
    pub(crate) deployment: String,
    pub(crate) node: String,
    pub(crate) tiers: Vec<Holding>,
}

/// The reports for the leader's server at an endpoint.
type Batch = (String, Vec<Report>);

/// The handle of the reporting thread, which ends once the handle drops.
pub(crate) struct Reporter {
    sender: mpsc::Sender<Batch>,
}

impl Reporter {
    /// Starts the thread.
    ///
    /// # Errors
    ///
    /// Fails when the thread cannot be made.
    pub(crate) fn start() -> io::Result<Self> {
        let (sender, batches) = mpsc::channel::<Batch>();
        std::thread::Builder::new()
            .name("uniserve-directory-reports".into())
            .spawn(move || send_reports(&batches))?;
        Ok(Self { sender })
    }

    /// Sends `reports` to the server at `leader`.
    pub(crate) fn send(&self, leader: String, reports: Vec<Report>) {
        let _ = self.sender.send((leader, reports));
    }
}

/// The thread's loop: every batch that arrived, the latest report of each deployment, on the
/// connection to its leader.
fn send_reports(batches: &mpsc::Receiver<Batch>) {
    let mut connected: Option<(String, Client)> = None;
    while let Ok(first) = batches.recv() {
        // The latest report of each deployment for the latest leader.
        let mut leader = first.0;
        let mut latest: BTreeMap<String, Report> = BTreeMap::new();
        for (to, reports) in std::iter::once((leader.clone(), first.1)).chain(batches.try_iter()) {
            if to != leader {
                latest.clear();
                leader = to;
            }
            for report in reports {
                latest.insert(report.deployment.clone(), report);
            }
        }

        for report in latest.into_values() {
            if connected.as_ref().is_some_and(|(at, _)| *at != leader) {
                connected = None;
            }
            let client = match &mut connected {
                Some((_, client)) => client,
                None => match Client::connect_peer(&leader) {
                    Ok(client) => &mut connected.insert((leader.clone(), client)).1,
                    Err(error) => {
                        tracing::debug!(%error, leader, "the directory's leader is not reached");
                        break;
                    }
                },
            };
            let sent = client.report_holdings(report.deployment, report.node, report.tiers);
            if let Err(error) = sent {
                tracing::debug!(%error, leader, "a report to the directory's leader failed");
                connected = None;
                break;
            }
        }
    }
}
