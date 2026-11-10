//! The prefill/decode hand-off: a request prefilled on a group that only
//! prefills (`WorkerGroupRole::Prefill`) continues on a decode group, whose
//! device cache receives the request's pages straight from the prefill
//! group's.
//!
//! Every group of a deployment indexes one unit space: a unit the scheduler
//! allocated to a request is that request's in every group's pool. A
//! hand-off therefore copies the request's units from each prefill rank's
//! planes into the same units of the decode rank of the same rank index, and
//! the decode group reads them where the prefill group wrote them. Each rank
//! reports where the Storage Server retains its planes, run by run
//! (`ipc::CachePlaneRun`); the copy is one transfer task per decode rank,
//! submitted to the decode rank's host, of one span per field and run of
//! consecutive units, from the prefill rank's retained tensor
//! (`Region::Tensor`, or a `Region::Remote` of it when the prefill rank
//! lives on another host, which that host's Transfer Server exposes through
//! NIXL) into the decode rank's. The task copies every byte of the units,
//! the prompt's last partial page and the flags and FP8 scales included, so
//! the decode group starts with the request's next token.
//!
//! The scheduler holds the transfer contexts of the GPUs of its groups on
//! each host for as long as it serves (`Request::OpenTransfer`), so a
//! hand-off pays for no context creation. Units are copied only once the
//! call that wrote them was accepted, so their device work completed.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use ipc::{CachePlaneRun, PrefixCacheInfo, RequestId};
use storage::protocol::{Event, Priority, Region, Reply, Request, Span, TransferEnd};

use crate::host_pages::Servers;
use crate::worker::WorkerGroupId;
use crate::{CallKind, ForwardMode, RequestKey, Scheduler};

/// One rank's planes, as a hand-off addresses them.
struct RankPlanes {
    /// The host the rank runs on.
    host: String,
    /// The connection to the rank's host's Storage Server.
    client: usize,
    /// Where other hosts reach that server.
    endpoint: Option<String>,
    runs: Vec<CachePlaneRun>,
}

/// What a request to a server was for.
enum Purpose {
    /// The transfer task of `request`'s hand-off, submitted on connection
    /// `client`.
    Transfer { request: RequestId, client: usize },
    /// A session holding the transfer contexts of a host's GPUs.
    Session,
}

/// One hand-off in flight.
struct Copy {
    /// The tasks submitted, and those that ended.
    tasks: usize,
    ended: usize,
    failed: Option<String>,
    started: Instant,
    bytes: u64,
    spans: usize,
}

/// How a hand-off ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HandOffEnd {
    /// Every unit landed in the decode group's planes: `bytes` in `spans`
    /// spans, in `micros` microseconds from the submission to the last task's
    /// end.
    Complete {
        bytes: u64,
        spans: usize,
        micros: u64,
    },
    /// A copy could not be made; the request cannot continue on the decode
    /// group.
    Failed(String),
}

/// The scheduler's hand-offs between its prefill and decode groups.
pub(crate) struct HandOffs {
    servers: Servers<Purpose>,
    ranks: HashMap<WorkerGroupId, Vec<RankPlanes>>,
    copies: HashMap<RequestId, Copy>,
    /// The request of each task in flight, by connection and task.
    tasks: HashMap<(usize, u64), RequestId>,
    ended: Vec<(RequestId, HandOffEnd)>,
}

impl HandOffs {
    /// Connect to this host's Storage Server, at `socket`, as a head of
    /// `epoch` of `deployment`, for the hand-offs between the groups of
    /// `ranks` (each rank's host and cache report, by group); the engine runs
    /// on `host`. Every group's ranks must report the runs of their planes.
    ///
    /// # Errors
    ///
    /// Fails when a server cannot be reached, or a rank reports no plane runs.
    pub(crate) fn connect(
        socket: &std::path::Path,
        deployment: &str,
        epoch: u64,
        host: &str,
        ranks: &BTreeMap<WorkerGroupId, Vec<(String, Option<PrefixCacheInfo>)>>,
    ) -> anyhow::Result<Self> {
        let mut servers =
            Servers::connect(socket, deployment, epoch, "the prefill/decode hand-off")?;
        let mut planes = HashMap::new();
        let mut gpus: BTreeMap<usize, BTreeSet<[u8; 16]>> = BTreeMap::new();
        for (worker, reports) in ranks {
            let mut group = Vec::with_capacity(reports.len());
            for (rank_host, cache) in reports {
                let Some(cache) = cache else {
                    continue;
                };
                anyhow::ensure!(
                    !cache.plane_runs.is_empty(),
                    "rank {} of worker group {} reports no retained cache planes to hand off from",
                    group.len(),
                    worker.0
                );
                let endpoint = cache.storage_endpoint.clone();
                let client = servers
                    .client(rank_host == host, endpoint.as_deref())
                    .ok_or_else(|| {
                        anyhow::anyhow!("the Storage Server of host {rank_host} is unreachable")
                    })?;
                gpus.entry(client)
                    .or_default()
                    .extend(cache.plane_runs.iter().map(|run| run.gpu));
                group.push(RankPlanes {
                    host: rank_host.clone(),
                    client,
                    endpoint,
                    runs: cache.plane_runs.clone(),
                });
            }
            planes.insert(worker.clone(), group);
        }
        // Each host's Transfer Server keeps the contexts of its groups' GPUs
        // while the scheduler serves; the connection closing ends the session.
        for (client, gpus) in gpus {
            let open = Request::OpenTransfer {
                gpus: gpus.into_iter().collect(),
            };
            servers.submit(client, &open, Purpose::Session);
        }
        Ok(Self {
            servers,
            ranks: planes,
            copies: HashMap::new(),
            tasks: HashMap::new(),
            ended: Vec::new(),
        })
    }

    /// Copy `request`'s `units` from `source`'s planes into `destination`'s,
    /// rank by rank. Returns the reason when the hand-off cannot start: the
    /// groups' ranks or planes differ, or no task could be submitted.
    pub(crate) fn start(
        &mut self,
        request: RequestId,
        source: &WorkerGroupId,
        destination: &WorkerGroupId,
        units: &[u32],
    ) -> Result<(), String> {
        let (Some(sources), Some(destinations)) =
            (self.ranks.get(source), self.ranks.get(destination))
        else {
            return Err(format!(
                "worker groups {} and {} report no planes to hand off between",
                source.0, destination.0
            ));
        };
        if sources.len() != destinations.len() {
            return Err(format!(
                "worker group {} has {} ranks and {} has {}: a hand-off copies between equal ranks",
                source.0,
                sources.len(),
                destination.0,
                destinations.len()
            ));
        }
        let mut units: Vec<u32> = units.iter().copied().filter(|unit| *unit > 0).collect();
        units.sort_unstable();
        units.dedup();

        let mut tasks = Vec::with_capacity(destinations.len());
        for (from, to) in sources.iter().zip(destinations) {
            tasks.push((to.client, spans(from, to, &units)?));
        }
        let mut copy = Copy {
            tasks: 0,
            ended: 0,
            failed: None,
            started: Instant::now(),
            bytes: 0,
            spans: 0,
        };
        for (client, spans) in tasks {
            if spans.is_empty() {
                continue;
            }
            copy.bytes += spans.iter().map(Span::copied).sum::<u64>();
            copy.spans += spans.len();
            let transfer = Request::Transfer {
                spans,
                priority: Priority::Request,
                update: 0,
            };
            if !self
                .servers
                .submit(client, &transfer, Purpose::Transfer { request, client })
            {
                copy.failed = Some("a hand-off's transfer could not be submitted".to_owned());
                break;
            }
            copy.tasks += 1;
        }
        if copy.tasks == 0 {
            // Nothing to copy (no unit), or nothing submitted.
            let end = match copy.failed {
                Some(reason) => HandOffEnd::Failed(reason),
                None => HandOffEnd::Complete {
                    bytes: 0,
                    spans: 0,
                    micros: 0,
                },
            };
            self.ended.push((request, end));
            return Ok(());
        }
        self.copies.insert(request, copy);
        Ok(())
    }

    /// Take the servers' answers and events, and the hand-offs that ended.
    pub(crate) fn poll(&mut self) -> Vec<(RequestId, HandOffEnd)> {
        let (events, answers) = self.servers.take();
        for (purpose, reply) in answers {
            match (purpose, reply) {
                (Purpose::Transfer { request, client }, Reply::Transferring { task }) => {
                    self.tasks.insert((client, task), request);
                }
                (Purpose::Transfer { request, .. }, other) => {
                    self.task_ended(
                        request,
                        Some(format!("the hand-off's transfer was refused: {other:?}")),
                    );
                }
                (Purpose::Session, Reply::TransferOpened { .. }) => {}
                (Purpose::Session, other) => {
                    tracing::warn!(?other, "a host refused to hold its GPUs' transfer contexts");
                }
            }
        }
        for (client, event) in events {
            let Event::Transferred { task, end, .. } = event else {
                continue;
            };
            let Some(request) = self.tasks.remove(&(client, task)) else {
                continue;
            };
            let failure = match end {
                TransferEnd::Complete => None,
                TransferEnd::Cancelled => Some("the hand-off's transfer was cancelled".to_owned()),
                TransferEnd::Failed { message } => Some(message),
            };
            self.task_ended(request, failure);
        }
        std::mem::take(&mut self.ended)
    }

    /// One task of `request`'s hand-off ended, failed with the reason when
    /// one is given.
    fn task_ended(&mut self, request: RequestId, failure: Option<String>) {
        let Some(copy) = self.copies.get_mut(&request) else {
            return;
        };
        copy.ended += 1;
        if let Some(failure) = failure {
            copy.failed.get_or_insert(failure);
        }
        if copy.ended < copy.tasks {
            return;
        }
        let Some(copy) = self.copies.remove(&request) else {
            return;
        };
        let end = match copy.failed {
            Some(reason) => HandOffEnd::Failed(reason),
            None => HandOffEnd::Complete {
                bytes: copy.bytes,
                spans: copy.spans,
                micros: copy.started.elapsed().as_micros() as u64,
            },
        };
        self.ended.push((request, end));
    }
}

/// The spans copying `units` from the prefill rank `from` into the decode
/// rank `to`: for each field, one span per run of consecutive units that one
/// run of each rank holds.
fn spans(from: &RankPlanes, to: &RankPlanes, units: &[u32]) -> Result<Vec<Span>, String> {
    let mut spans = Vec::new();
    let fields: BTreeSet<&str> = to.runs.iter().map(|run| run.field.as_str()).collect();
    for field in fields {
        let holding = |runs: &[CachePlaneRun], unit: u32| {
            runs.iter()
                .find(|run| {
                    run.field == field && (run.first..run.first + run.count).contains(&unit)
                })
                .cloned()
        };
        let mut pending: Option<(CachePlaneRun, CachePlaneRun, u32, u32)> = None;
        for &unit in units {
            let (Some(source), Some(destination)) =
                (holding(&from.runs, unit), holding(&to.runs, unit))
            else {
                return Err(format!(
                    "unit {unit}'s {field} lies in no retained plane of the prefill or the decode rank"
                ));
            };
            if source.bytes != destination.bytes {
                return Err(format!(
                    "the prefill and decode ranks hold {field} at {} and {} bytes a unit",
                    source.bytes, destination.bytes
                ));
            }
            match &mut pending {
                Some((held_source, held_destination, _, end))
                    if *end == unit
                        && *held_source == source
                        && *held_destination == destination =>
                {
                    *end = unit + 1;
                }
                _ => {
                    if let Some(run) = pending.take() {
                        spans.push(span(from, to, run));
                    }
                    pending = Some((source, destination, unit, unit + 1));
                }
            }
        }
        if let Some(run) = pending {
            spans.push(span(from, to, run));
        }
    }
    Ok(spans)
}

/// The span copying units `start..end` from `source` (a run of `from`) into
/// `destination` (a run of `to`).
fn span(
    from: &RankPlanes,
    to: &RankPlanes,
    (source, destination, start, end): (CachePlaneRun, CachePlaneRun, u32, u32),
) -> Span {
    let tensor = Region::Tensor {
        key: source.key,
        at: None,
    };
    let source_region = match (&from.endpoint, from.host == to.host) {
        (Some(endpoint), false) => Region::Remote {
            endpoint: endpoint.clone(),
            region: Box::new(tensor),
        },
        _ => tensor,
    };
    Span {
        source: source_region,
        source_offset: source.offset + u64::from(start - source.first) * source.bytes,
        destination: Region::Tensor {
            key: destination.key,
            at: None,
        },
        destination_offset: destination.offset
            + u64::from(start - destination.first) * destination.bytes,
        bytes: u64::from(end - start) * source.bytes,
        rows: 1,
        source_pitch: 0,
        destination_pitch: 0,
    }
}

impl Scheduler {
    /// Start the hand-off of every request whose prompt its prefill group
    /// completed and whose calls there were all accepted, and move on those
    /// whose hand-off ended: a request whose units landed decodes on its
    /// decode group; one whose copy failed fails. Returns whether a request
    /// moved on.
    pub(crate) fn poll_hand_offs(&mut self) -> bool {
        let ended = match self.hand_offs.as_mut() {
            Some(hand_offs) => hand_offs.poll(),
            None if self.dispatch.prefill_groups.is_empty() => return false,
            // Prefill groups without a hand-off: a request that completed its
            // prompt there cannot continue (`Scheduler::enable_hand_off`).
            None => Vec::new(),
        };
        let ready: Vec<RequestId> = self
            .running
            .iter()
            .filter(|(id, state)| {
                state.phase == crate::generation::GenerationPhase::HandOff
                    && state.hand_off == crate::generation::HandOff::None
                    && state.failure.is_none()
                    && self
                        .inflight
                        .pending_calls
                        .get(id)
                        .is_none_or(|calls| calls.is_empty())
            })
            .map(|(id, _)| *id)
            .collect();
        let mut progressed = !ended.is_empty();
        for id in ready {
            progressed |= self.start_hand_off(id);
        }
        for (id, end) in ended {
            let Some(state) = self.running.get_mut(&id) else {
                continue;
            };
            match end {
                HandOffEnd::Complete {
                    bytes,
                    spans,
                    micros,
                } => {
                    tracing::info!(
                        request = id.0,
                        bytes,
                        spans,
                        micros,
                        "handed a request's KV cache to its decode group"
                    );
                    state.hand_off = crate::generation::HandOff::Copied;
                    state.phase = crate::generation::GenerationPhase::DecodeUnd;
                }
                HandOffEnd::Failed(reason) => {
                    state.hand_off = crate::generation::HandOff::None;
                    self.fail_after_inflight(
                        id,
                        format!("the KV cache hand-off to the decode group failed: {reason}"),
                    );
                }
            }
        }
        progressed
    }

    /// Place `id`'s decode on a decode group and copy its units there from
    /// its prefill group. Returns whether the request moved on: its copy
    /// started, or it failed.
    fn start_hand_off(&mut self, id: RequestId) -> bool {
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        let key = RequestKey::new(self.engine_id, id, state.request_epoch);
        let units: Vec<u32> = state
            .block_tables()
            .unwrap_or_default()
            .iter()
            .flat_map(|table| table.unit_ids().into_iter().map(|unit| unit.0))
            .collect();
        let Some(source) = self
            .dispatch
            .affinity
            .get(&(key, crate::DEFAULT_BLOCK.to_owned()))
            .cloned()
        else {
            return false;
        };
        // The decode group is chosen now, and the request's decodes stay there.
        let Some(destination) = self.dispatch.place(
            &self.worker,
            &self.info,
            key,
            CallKind::Forward(ForwardMode::Decode),
        ) else {
            return false;
        };
        let started = match self.hand_offs.as_mut() {
            Some(hand_offs) => hand_offs.start(id, &source, &destination, &units),
            None => Err("the scheduler reaches no Storage Server to copy it with".to_owned()),
        };
        match started {
            Ok(()) => {
                if let Some(state) = self.running.get_mut(&id) {
                    state.hand_off = crate::generation::HandOff::Copying;
                }
            }
            Err(reason) => self.fail_after_inflight(
                id,
                format!("the KV cache cannot be handed to the decode group: {reason}"),
            ),
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(field: &str, key: u8, first: u32, count: u32, offset: u64, bytes: u64) -> CachePlaneRun {
        CachePlaneRun {
            field: field.to_owned(),
            key: storage::Key([key; 32]),
            gpu: [key; 16],
            first,
            count,
            offset,
            bytes,
        }
    }

    fn rank(host: &str, runs: Vec<CachePlaneRun>) -> RankPlanes {
        RankPlanes {
            host: host.to_owned(),
            client: 0,
            endpoint: Some(format!("{host}:7000")),
            runs,
        }
    }

    /// A hand-off copies every unit's bytes of each field once, in one span
    /// per run of consecutive units both ranks hold in one run, at the
    /// offsets each rank's run states; a prefill rank of another host is read
    /// through its Storage Server.
    #[test]
    fn units_copy_field_by_field_in_runs_of_consecutive_units() {
        // The prefill rank holds its key values in one storage of 16 units;
        // the decode rank's are two segments of 8 units each.
        let from = rank(
            "a",
            vec![
                run("key.values/0", 1, 0, 16, 4096, 64),
                run("key.initialized/0", 2, 0, 16, 0, 1),
            ],
        );
        let to = rank(
            "b",
            vec![
                run("key.values/0", 3, 0, 8, 0, 64),
                run("key.values/0", 4, 8, 8, 0, 64),
                run("key.initialized/0", 5, 0, 16, 16, 1),
            ],
        );
        let spans = spans(&from, &to, &[6, 7, 8, 9, 12]).unwrap();
        let summary: Vec<(u8, u64, u8, u64, u64)> = spans
            .iter()
            .map(|span| {
                let Region::Remote { region, .. } = &span.source else {
                    panic!(
                        "a source of another host is read remotely: {:?}",
                        span.source
                    );
                };
                let (
                    Region::Tensor { key: source, .. },
                    Region::Tensor {
                        key: destination, ..
                    },
                ) = (region.as_ref(), &span.destination)
                else {
                    panic!("spans copy between retained tensors");
                };
                (
                    source.0[0],
                    span.source_offset,
                    destination.0[0],
                    span.destination_offset,
                    span.copied(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                // Flags: units 6..10 in one run each side, then unit 12.
                (2, 6, 5, 16 + 6, 4),
                (2, 12, 5, 16 + 12, 1),
                // Values: units 6..8 end the first segment, 8..10 and 12
                // lie in the second.
                (1, 4096 + 6 * 64, 3, 6 * 64, 2 * 64),
                (1, 4096 + 8 * 64, 4, 0, 2 * 64),
                (1, 4096 + 12 * 64, 4, 4 * 64, 64),
            ]
        );
    }

    /// Ranks that hold a field at other bytes per unit, or not at all, are
    /// refused rather than copied partly; on one host a source is read where
    /// it lies.
    #[test]
    fn ranks_of_other_planes_are_refused() {
        let from = rank("a", vec![run("key.values/0", 1, 0, 4, 0, 64)]);
        let wider = rank("a", vec![run("key.values/0", 2, 0, 4, 0, 128)]);
        assert!(spans(&from, &wider, &[1]).is_err());
        let other = rank("a", vec![run("value.values/0", 2, 0, 4, 0, 64)]);
        assert!(spans(&from, &other, &[1]).is_err());
        let local = rank("a", vec![run("key.values/0", 3, 0, 4, 0, 64)]);
        let spans = spans(&from, &local, &[1]).unwrap();
        assert!(matches!(spans[0].source, Region::Tensor { .. }));
    }
}
