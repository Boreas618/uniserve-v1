//! A session's file: one SQLite database per worker, in UniServe's own
//! schema.
//!
//! The schema is versioned ([`SCHEMA_VERSION`]) and changes only by
//! additions: a later version adds tables or columns, never removes or
//! redefines one, so a reader of an earlier version reads every file of a
//! later one. The file states its version and its own schema
//! (`schema_version`, and SQLite's `sqlite_schema`).
//!
//! Every raw record kind has a table of its own, beside the tables that
//! describe them: `metadata` (the session, the world, the worker, its device,
//! the CUPTI version, the request, the clocks and what the identifiers
//! mean), `strings` (kernel, API, NVTX and function names, by id), `clocks`
//! (the alignment of every clock the records use) and `collectors` (what ran,
//! and what the quality-of-service guard turned off). Times are nanoseconds
//! in the clock `metadata` names for each table.
//!
//! The writer is a thread of its own: collectors hand it rows over a
//! channel ([`Writer`]) and never wait for the file. It commits a
//! transaction per [`COMMIT_PERIOD`] and checkpoints the write-ahead log into
//! the file when the session ends, leaving one file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use cuda::cupti::pc_sampling::Taken;
use cuda::cupti::pm_sampling::Sample;
use cuda::cupti::range_profiler::Range;
use cuda::cupti::records::Record;
use rusqlite::{Connection, params};

use crate::profiling::fences::FenceRecord;
use crate::{Error, Result};

/// The schema's version.
pub(crate) const SCHEMA_VERSION: i64 = 1;

/// How often the writer commits what it received.
const COMMIT_PERIOD: Duration = Duration::from_millis(200);

/// The schema of version 1.
const SCHEMA: &str = "
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE strings (id INTEGER PRIMARY KEY, text TEXT NOT NULL);
CREATE TABLE collectors (name TEXT NOT NULL, state TEXT NOT NULL, time INTEGER NOT NULL,
    detail TEXT NOT NULL);
CREATE TABLE clocks (source TEXT NOT NULL, device INTEGER, gpu INTEGER, realtime INTEGER NOT NULL,
    cupti INTEGER, monotonic INTEGER, uncertainty INTEGER NOT NULL);
CREATE TABLE steps (step INTEGER NOT NULL, stage TEXT NOT NULL, selected INTEGER NOT NULL,
    begin INTEGER NOT NULL, end INTEGER);
CREATE TABLE kernel (start INTEGER NOT NULL, end INTEGER NOT NULL, queued INTEGER,
    submitted INTEGER, device INTEGER NOT NULL, context INTEGER NOT NULL, stream INTEGER NOT NULL,
    correlation INTEGER NOT NULL, grid_x INTEGER, grid_y INTEGER, grid_z INTEGER, block_x INTEGER,
    block_y INTEGER, block_z INTEGER, registers INTEGER, static_shared INTEGER,
    dynamic_shared INTEGER, name INTEGER NOT NULL, graph INTEGER, graph_node INTEGER,
    launch_type INTEGER, channel INTEGER);
CREATE TABLE copy (start INTEGER NOT NULL, end INTEGER NOT NULL, device INTEGER NOT NULL,
    context INTEGER NOT NULL, stream INTEGER NOT NULL, correlation INTEGER NOT NULL,
    copy_kind INTEGER NOT NULL, source_kind INTEGER, destination_kind INTEGER,
    bytes INTEGER NOT NULL, graph INTEGER, graph_node INTEGER, source_device INTEGER,
    destination_device INTEGER);
CREATE TABLE fill (start INTEGER NOT NULL, end INTEGER NOT NULL, device INTEGER NOT NULL,
    context INTEGER NOT NULL, stream INTEGER NOT NULL, correlation INTEGER NOT NULL,
    value INTEGER NOT NULL, bytes INTEGER NOT NULL, memory_kind INTEGER, graph INTEGER,
    graph_node INTEGER);
CREATE TABLE storage (time INTEGER NOT NULL, operation INTEGER NOT NULL,
    memory_kind INTEGER NOT NULL, correlation INTEGER NOT NULL, address INTEGER NOT NULL,
    bytes INTEGER NOT NULL, process INTEGER, device INTEGER, context INTEGER, stream INTEGER,
    asynchronous INTEGER NOT NULL);
CREATE TABLE marker (time INTEGER NOT NULL, flags INTEGER NOT NULL, id INTEGER NOT NULL,
    object_kind INTEGER NOT NULL, object_0 INTEGER, object_1 INTEGER, object_2 INTEGER,
    name INTEGER NOT NULL, domain INTEGER NOT NULL);
CREATE TABLE marker_data (id INTEGER NOT NULL, flags INTEGER NOT NULL,
    payload_kind INTEGER NOT NULL, payload INTEGER, color INTEGER, category INTEGER);
CREATE TABLE graph (start INTEGER NOT NULL, end INTEGER NOT NULL, device INTEGER NOT NULL,
    context INTEGER NOT NULL, stream INTEGER NOT NULL, correlation INTEGER NOT NULL,
    graph INTEGER NOT NULL);
CREATE TABLE nvlink (version INTEGER NOT NULL, type_0 INTEGER NOT NULL, type_1 INTEGER NOT NULL,
    end_0 BLOB NOT NULL, end_1 BLOB NOT NULL, flag INTEGER NOT NULL, links INTEGER NOT NULL,
    bandwidth INTEGER NOT NULL, nvswitch INTEGER NOT NULL);
CREATE TABLE pcie (device_type INTEGER NOT NULL, id INTEGER NOT NULL, domain INTEGER NOT NULL,
    generation INTEGER NOT NULL, link_rate INTEGER NOT NULL, link_width INTEGER NOT NULL,
    upstream_bus INTEGER NOT NULL);
CREATE TABLE api (start INTEGER NOT NULL, end INTEGER NOT NULL, domain INTEGER NOT NULL,
    cbid INTEGER NOT NULL, name INTEGER NOT NULL, thread INTEGER NOT NULL,
    correlation INTEGER NOT NULL, result INTEGER NOT NULL);
CREATE TABLE fence_probe (probe INTEGER PRIMARY KEY, instance INTEGER NOT NULL,
    site TEXT NOT NULL, path TEXT NOT NULL, component TEXT NOT NULL);
CREATE TABLE fence (device INTEGER NOT NULL, time INTEGER NOT NULL, probe INTEGER NOT NULL,
    step INTEGER NOT NULL);
CREATE TABLE pm_sample (sample INTEGER NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
    metric TEXT NOT NULL, value REAL NOT NULL);
CREATE TABLE range_metric (pass INTEGER NOT NULL, range INTEGER NOT NULL, name TEXT NOT NULL,
    metric TEXT NOT NULL, value REAL NOT NULL);
CREATE TABLE pc_stall_reason (reason INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE pc_sample (take INTEGER NOT NULL, function TEXT NOT NULL,
    function_index INTEGER NOT NULL, cubin_crc INTEGER NOT NULL, offset INTEGER NOT NULL,
    correlation INTEGER NOT NULL, reason INTEGER NOT NULL, samples INTEGER NOT NULL);
CREATE TABLE pc_take (take INTEGER PRIMARY KEY, time INTEGER NOT NULL,
    total_samples INTEGER NOT NULL, dropped_samples INTEGER NOT NULL,
    non_user_samples INTEGER NOT NULL, hardware_buffer_full INTEGER NOT NULL);
";

/// What the meaning of each table's identifiers and clocks is, written into
/// `metadata` under `doc.<table>`.
const DOCS: &[(&str, &str)] = &[
    (
        "doc.clocks",
        "kernel, copy, fill, storage, marker, graph, api, pm_sample and steps use CUPTI's \
         clock (clocks.cupti); fence uses the device's %globaltimer (clocks.gpu of its \
         device). Each clocks row is one \
         sample of the clocks it names at one instant, to within uncertainty nanoseconds.",
    ),
    (
        "doc.correlation",
        "kernel, copy, fill, storage and graph rows carry the correlation id of the api row \
         (CUPTI's runtime or driver call) that issued them; the nodes of a CUDA graph carry \
         its launch's. api rows of domain 2 (runtime) and 1 (driver) of one call share it.",
    ),
    (
        "doc.strings",
        "kernel.name, api.name, marker.name and marker.domain are ids of strings.",
    ),
    (
        "doc.fence",
        "fence.probe is 2 * instance at a component instance's entry and 2 * instance + 1 at \
         its exit (fence_probe), 1048574 at a step's begin and 1048575 at its end; \
         fence.step is the low 32 bits of the step's batch id (steps.step).",
    ),
    (
        "doc.steps",
        "A step is one batch's forward on the worker: its batch id, its stage (call kind) and \
         whether the session's window recorded it; begin and end are when the worker \
         submitted the forward's first work and saw its last complete.",
    ),
    (
        "doc.pc_sample",
        "One row per program counter and stall reason of one take (pc_take); reason names \
         pc_stall_reason.",
    ),
];

/// A message to the writer.
pub(crate) enum Message {
    Metadata(Vec<(String, String)>),
    Collector {
        name: String,
        state: String,
        detail: String,
    },
    Clock(ClockRow),
    Step(StepRow),
    Timeline {
        records: Vec<Record>,
        names: Vec<(u32, String)>,
    },
    FenceProbes(Vec<(u32, String, String)>),
    Fences {
        device: i32,
        records: Vec<FenceRecord>,
    },
    Pm {
        metrics: Vec<String>,
        samples: Vec<Sample>,
    },
    Ranges {
        metrics: Vec<String>,
        ranges: Vec<Range>,
    },
    PcReasons(Vec<(u32, String)>),
    Pc(Taken),
}

/// One sample of clocks at one instant.
#[derive(Clone, Debug)]
pub(crate) struct ClockRow {
    pub(crate) source: &'static str,
    pub(crate) device: Option<i32>,
    pub(crate) gpu: Option<u64>,
    pub(crate) realtime: u64,
    pub(crate) cupti: Option<u64>,
    pub(crate) monotonic: Option<u64>,
    pub(crate) uncertainty: u64,
}

/// One step.
#[derive(Clone, Debug)]
pub(crate) struct StepRow {
    pub(crate) step: u64,
    pub(crate) stage: String,
    pub(crate) selected: bool,
    pub(crate) begin: u64,
    pub(crate) end: Option<u64>,
}

/// What a finished file holds.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub(crate) struct Summary {
    pub(crate) path: PathBuf,
    pub(crate) bytes: u64,
    /// Rows by table.
    pub(crate) rows: BTreeMap<String, u64>,
}

/// The writer of one session's file.
pub(crate) struct Writer {
    sender: Sender<Message>,
    thread: Option<JoinHandle<Result<Summary>>>,
    bytes: Arc<AtomicU64>,
}

/// Where collectors send rows; cloned into each.
#[derive(Clone)]
pub(crate) struct Rows(Sender<Message>);

impl Rows {
    /// Send `message`; a writer that ended (its file failed) drops it.
    pub(crate) fn send(&self, message: Message) {
        let _ = self.0.send(message);
    }
}

impl Writer {
    /// Create the file `path` with the schema and `metadata`, and start the
    /// writer.
    ///
    /// # Errors
    ///
    /// Fails when the file exists or cannot be created.
    pub(crate) fn create(path: PathBuf, metadata: Vec<(String, String)>) -> Result<Self> {
        if path.exists() {
            return Err(Error::Invalid(format!("{} exists", path.display())));
        }
        let connection = Connection::open(&path).map_err(|error| sqlite(&path, &error))?;
        connection
            .execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .map_err(|error| sqlite(&path, &error))?;
        connection
            .execute_batch(SCHEMA)
            .map_err(|error| sqlite(&path, &error))?;
        connection
            .execute(
                "INSERT INTO schema_version VALUES (?1)",
                params![SCHEMA_VERSION],
            )
            .map_err(|error| sqlite(&path, &error))?;
        let mut file = File {
            connection,
            path,
            rows: BTreeMap::new(),
            pm_samples: 0,
            range_passes: 0,
            pc_takes: 0,
        };
        let docs = DOCS
            .iter()
            .map(|&(key, value)| (key.to_owned(), value.to_owned()));
        file.write(Message::Metadata(
            metadata.into_iter().chain(docs).collect(),
        ))?;

        let (sender, receiver) = crossbeam_channel::unbounded();
        let bytes = Arc::new(AtomicU64::new(size(&file.path)));
        let thread = std::thread::Builder::new()
            .name("profile-writer".into())
            .spawn({
                let bytes = Arc::clone(&bytes);
                move || file.serve(&receiver, &bytes)
            })
            .map_err(|error| Error::transport_source("starting the profile writer", error))?;
        Ok(Self {
            sender,
            thread: Some(thread),
            bytes,
        })
    }

    /// Where collectors send rows.
    pub(crate) fn rows(&self) -> Rows {
        Rows(self.sender.clone())
    }

    /// The bytes the file holds so far, its write-ahead log included.
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Acquire)
    }

    /// Write what remains, checkpoint the file and close it, once every
    /// [`Rows`] clone is dropped.
    ///
    /// # Errors
    ///
    /// Fails with the first error the file met.
    pub(crate) fn finish(mut self) -> Result<Summary> {
        drop(self.sender);
        match self.thread.take().map(JoinHandle::join) {
            Some(Ok(summary)) => summary,
            Some(Err(_)) => Err(Error::State("the profile writer panicked")),
            None => Err(Error::State("the profile writer finished already")),
        }
    }
}

/// The open file, on the writer's thread.
struct File {
    connection: Connection,
    path: PathBuf,
    rows: BTreeMap<String, u64>,
    pm_samples: u64,
    range_passes: u64,
    pc_takes: u64,
}

fn sqlite(path: &Path, error: &rusqlite::Error) -> Error {
    Error::Invalid(format!("profile file {}: {error}", path.display()))
}

/// The bytes of the file and its write-ahead log.
fn size(path: &Path) -> u64 {
    let mut wal = path.as_os_str().to_owned();
    wal.push("-wal");
    [path.to_path_buf(), PathBuf::from(wal)]
        .iter()
        .filter_map(|file| std::fs::metadata(file).ok())
        .map(|metadata| metadata.len())
        .sum()
}

impl File {
    /// Write what arrives, a transaction per period, until every sender is
    /// dropped; then checkpoint and close.
    fn serve(mut self, receiver: &Receiver<Message>, bytes: &AtomicU64) -> Result<Summary> {
        let mut failed: Option<Error> = None;
        loop {
            let first = match receiver.recv_timeout(COMMIT_PERIOD) {
                Ok(message) => message,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            };
            if failed.is_some() {
                continue;
            }
            let deadline = Instant::now() + COMMIT_PERIOD;
            let written = self.transaction(|file| {
                file.write(first)?;
                while Instant::now() < deadline {
                    match receiver.try_recv() {
                        Ok(message) => file.write(message)?,
                        Err(_) => break,
                    }
                }
                Ok(())
            });
            if let Err(error) = written {
                tracing::error!("the profile file stops being written: {error}");
                failed = Some(error);
            }
            bytes.store(size(&self.path), Ordering::Release);
        }
        if let Some(error) = failed {
            return Err(error);
        }
        self.connection
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;")
            .map_err(|error| sqlite(&self.path, &error))?;
        let path = self.path.clone();
        self.connection
            .close()
            .map_err(|(_, error)| sqlite(&path, &error))?;
        let bytes_now = size(&path);
        bytes.store(bytes_now, Ordering::Release);
        Ok(Summary {
            path,
            bytes: bytes_now,
            rows: self.rows,
        })
    }

    fn transaction(&mut self, write: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        self.connection
            .execute_batch("BEGIN")
            .map_err(|error| sqlite(&self.path, &error))?;
        let written = write(self);
        let end = if written.is_ok() {
            "COMMIT"
        } else {
            "ROLLBACK"
        };
        self.connection
            .execute_batch(end)
            .map_err(|error| sqlite(&self.path, &error))?;
        written
    }

    fn count(&mut self, table: &str, rows: u64) {
        *self.rows.entry(table.to_owned()).or_default() += rows;
    }

    /// Insert `rows` into `table` with the statement `sql`.
    fn insert<P: rusqlite::Params>(
        &mut self,
        table: &str,
        sql: &str,
        rows: impl IntoIterator<Item = P>,
    ) -> Result<()> {
        let path = self.path.clone();
        let mut count = 0;
        {
            let mut statement = self
                .connection
                .prepare_cached(sql)
                .map_err(|error| sqlite(&path, &error))?;
            for row in rows {
                statement
                    .execute(row)
                    .map_err(|error| sqlite(&path, &error))?;
                count += 1;
            }
        }
        self.count(table, count);
        Ok(())
    }

    fn write(&mut self, message: Message) -> Result<()> {
        match message {
            Message::Metadata(entries) => self.insert(
                "metadata",
                "INSERT OR REPLACE INTO metadata VALUES (?1, ?2)",
                entries,
            ),
            Message::Collector {
                name,
                state,
                detail,
            } => self.insert(
                "collectors",
                "INSERT INTO collectors VALUES (?1, ?2, ?3, ?4)",
                [(
                    name,
                    state,
                    cuda::cupti::timestamp().unwrap_or(0) as i64,
                    detail,
                )],
            ),
            Message::Clock(clock) => self.insert(
                "clocks",
                "INSERT INTO clocks VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                [(
                    clock.source,
                    clock.device,
                    clock.gpu.map(|gpu| gpu as i64),
                    clock.realtime as i64,
                    clock.cupti.map(|cupti| cupti as i64),
                    clock.monotonic.map(|monotonic| monotonic as i64),
                    clock.uncertainty as i64,
                )],
            ),
            Message::Step(step) => self.insert(
                "steps",
                "INSERT INTO steps VALUES (?1, ?2, ?3, ?4, ?5)",
                [(
                    step.step as i64,
                    step.stage,
                    step.selected,
                    step.begin as i64,
                    step.end.map(|end| end as i64),
                )],
            ),
            Message::Timeline { records, names } => {
                self.insert(
                    "strings",
                    "INSERT OR REPLACE INTO strings VALUES (?1, ?2)",
                    names,
                )?;
                for record in records {
                    self.timeline(record)?;
                }
                Ok(())
            }
            Message::FenceProbes(instances) => self.insert(
                "fence_probe",
                "INSERT INTO fence_probe VALUES (?1, ?2, ?3, ?4, ?5)",
                instances
                    .into_iter()
                    .flat_map(|(instance, path, component)| {
                        [("entry", 0), ("exit", 1)].map(|(site, offset)| {
                            (
                                i64::from(2 * instance + offset),
                                i64::from(instance),
                                site,
                                path.clone(),
                                component.clone(),
                            )
                        })
                    })
                    .collect::<Vec<_>>(),
            ),
            Message::Fences { device, records } => self.insert(
                "fence",
                "INSERT INTO fence VALUES (?1, ?2, ?3, ?4)",
                records.into_iter().map(|record| {
                    (
                        device,
                        record.time as i64,
                        i64::from(record.probe),
                        i64::from(record.step),
                    )
                }),
            ),
            Message::Pm { metrics, samples } => {
                let first = self.pm_samples;
                self.pm_samples += samples.len() as u64;
                let rows: Vec<_> = samples
                    .into_iter()
                    .enumerate()
                    .flat_map(|(index, sample)| {
                        let sample_id = (first + index as u64) as i64;
                        metrics
                            .iter()
                            .zip(sample.values)
                            .map(move |(metric, value)| {
                                (
                                    sample_id,
                                    sample.start as i64,
                                    sample.end as i64,
                                    metric.clone(),
                                    value,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                self.insert(
                    "pm_sample",
                    "INSERT INTO pm_sample VALUES (?1, ?2, ?3, ?4, ?5)",
                    rows,
                )
            }
            Message::Ranges { metrics, ranges } => {
                let pass = self.range_passes as i64;
                self.range_passes += 1;
                let rows: Vec<_> = ranges
                    .into_iter()
                    .enumerate()
                    .flat_map(|(index, range)| {
                        metrics
                            .iter()
                            .zip(range.values)
                            .map(|(metric, value)| {
                                (
                                    pass,
                                    index as i64,
                                    range.name.clone(),
                                    metric.clone(),
                                    value,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                self.insert(
                    "range_metric",
                    "INSERT INTO range_metric VALUES (?1, ?2, ?3, ?4, ?5)",
                    rows,
                )
            }
            Message::PcReasons(reasons) => self.insert(
                "pc_stall_reason",
                "INSERT OR REPLACE INTO pc_stall_reason VALUES (?1, ?2)",
                reasons
                    .into_iter()
                    .map(|(reason, name)| (i64::from(reason), name)),
            ),
            Message::Pc(taken) => {
                let take = self.pc_takes as i64;
                self.pc_takes += 1;
                self.insert(
                    "pc_take",
                    "INSERT INTO pc_take VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    [(
                        take,
                        cuda::cupti::timestamp().unwrap_or(0) as i64,
                        taken.total_samples as i64,
                        taken.dropped_samples as i64,
                        taken.non_user_samples as i64,
                        taken.hardware_buffer_full,
                    )],
                )?;
                let rows: Vec<_> = taken
                    .pcs
                    .into_iter()
                    .flat_map(|pc| {
                        pc.stalls
                            .iter()
                            .map(|&(reason, samples)| {
                                (
                                    take,
                                    pc.function.clone(),
                                    i64::from(pc.function_index),
                                    pc.cubin_crc as i64,
                                    pc.offset as i64,
                                    i64::from(pc.correlation),
                                    i64::from(reason),
                                    i64::from(samples),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                self.insert(
                    "pc_sample",
                    "INSERT INTO pc_sample VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    rows,
                )
            }
        }
    }

    /// Insert one CUPTI record into its kind's table.
    fn timeline(&mut self, record: Record) -> Result<()> {
        match record {
            Record::Kernel(kernel) => self.insert(
                "kernel",
                "INSERT INTO kernel VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, \
                 ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22)",
                [rusqlite::params_from_iter([
                    rusqlite::types::Value::from(kernel.start as i64),
                    (kernel.end as i64).into(),
                    (kernel.queued as i64).into(),
                    (kernel.submitted as i64).into(),
                    i64::from(kernel.device).into(),
                    i64::from(kernel.context).into(),
                    i64::from(kernel.stream).into(),
                    i64::from(kernel.correlation).into(),
                    i64::from(kernel.grid[0]).into(),
                    i64::from(kernel.grid[1]).into(),
                    i64::from(kernel.grid[2]).into(),
                    i64::from(kernel.block[0]).into(),
                    i64::from(kernel.block[1]).into(),
                    i64::from(kernel.block[2]).into(),
                    i64::from(kernel.registers).into(),
                    i64::from(kernel.static_shared).into(),
                    i64::from(kernel.dynamic_shared).into(),
                    i64::from(kernel.name).into(),
                    i64::from(kernel.graph).into(),
                    (kernel.graph_node as i64).into(),
                    i64::from(kernel.launch_type).into(),
                    i64::from(kernel.channel).into(),
                ])],
            ),
            Record::Copy(copy) => self.insert(
                "copy",
                "INSERT INTO copy VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, \
                 ?13, ?14)",
                [rusqlite::params_from_iter([
                    rusqlite::types::Value::from(copy.start as i64),
                    (copy.end as i64).into(),
                    i64::from(copy.device).into(),
                    i64::from(copy.context).into(),
                    i64::from(copy.stream).into(),
                    i64::from(copy.correlation).into(),
                    i64::from(copy.copy_kind).into(),
                    i64::from(copy.source_kind).into(),
                    i64::from(copy.destination_kind).into(),
                    (copy.bytes as i64).into(),
                    i64::from(copy.graph).into(),
                    (copy.graph_node as i64).into(),
                    copy.peer
                        .map_or(rusqlite::types::Value::Null, |(source, _)| {
                            i64::from(source).into()
                        }),
                    copy.peer
                        .map_or(rusqlite::types::Value::Null, |(_, destination)| {
                            i64::from(destination).into()
                        }),
                ])],
            ),
            Record::Fill(fill) => self.insert(
                "fill",
                "INSERT INTO fill VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                [rusqlite::params_from_iter([
                    rusqlite::types::Value::from(fill.start as i64),
                    (fill.end as i64).into(),
                    i64::from(fill.device).into(),
                    i64::from(fill.context).into(),
                    i64::from(fill.stream).into(),
                    i64::from(fill.correlation).into(),
                    i64::from(fill.value).into(),
                    (fill.bytes as i64).into(),
                    i64::from(fill.memory_kind).into(),
                    i64::from(fill.graph).into(),
                    (fill.graph_node as i64).into(),
                ])],
            ),
            Record::Memory(storage) => self.insert(
                "storage",
                "INSERT INTO storage VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                [rusqlite::params_from_iter([
                    rusqlite::types::Value::from(storage.timestamp as i64),
                    i64::from(storage.operation).into(),
                    i64::from(storage.memory_kind).into(),
                    i64::from(storage.correlation).into(),
                    (storage.address as i64).into(),
                    (storage.bytes as i64).into(),
                    i64::from(storage.process).into(),
                    i64::from(storage.device).into(),
                    i64::from(storage.context).into(),
                    i64::from(storage.stream).into(),
                    i64::from(storage.asynchronous).into(),
                ])],
            ),
            Record::Marker(marker) => self.insert(
                "marker",
                "INSERT INTO marker VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                [(
                    marker.timestamp as i64,
                    marker.flags,
                    marker.id,
                    marker.object_kind,
                    marker.object[0],
                    marker.object[1],
                    marker.object[2],
                    marker.name,
                    marker.domain,
                )],
            ),
            Record::MarkerData(data) => self.insert(
                "marker_data",
                "INSERT INTO marker_data VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                [(
                    data.id,
                    data.flags,
                    data.payload_kind,
                    data.payload as i64,
                    data.color,
                    data.category,
                )],
            ),
            Record::Graph(graph) => self.insert(
                "graph",
                "INSERT INTO graph VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                [(
                    graph.start as i64,
                    graph.end as i64,
                    graph.device,
                    graph.context,
                    graph.stream,
                    graph.correlation,
                    graph.graph,
                )],
            ),
            Record::NvLink(link) => self.insert(
                "nvlink",
                "INSERT INTO nvlink VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                [(
                    link.version,
                    link.types[0],
                    link.types[1],
                    link.ends[0].to_vec(),
                    link.ends[1].to_vec(),
                    link.flag,
                    link.links,
                    link.bandwidth as i64,
                    link.nvswitch,
                )],
            ),
            Record::Pcie(pcie) => self.insert(
                "pcie",
                "INSERT INTO pcie VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                [(
                    pcie.device_type,
                    pcie.id,
                    pcie.domain,
                    pcie.generation,
                    pcie.link_rate,
                    pcie.link_width,
                    pcie.upstream_bus,
                )],
            ),
            Record::Api(api) => self.insert(
                "api",
                "INSERT INTO api VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                [(
                    api.start as i64,
                    api.end as i64,
                    api.domain,
                    api.cbid,
                    api.name,
                    api.thread,
                    api.correlation,
                    api.result,
                )],
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use cuda::cupti::records::{Api, Kernel};

    use super::*;

    #[test]
    fn a_file_states_its_schema_and_holds_what_it_was_sent() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("rank0.sqlite");
        let writer =
            Writer::create(path.clone(), vec![("session".to_owned(), "7".to_owned())]).unwrap();
        let rows = writer.rows();
        rows.send(Message::Timeline {
            records: vec![
                Record::Kernel(Kernel {
                    start: 10,
                    end: 20,
                    queued: 0,
                    submitted: 0,
                    device: 0,
                    context: 1,
                    stream: 7,
                    correlation: 3,
                    grid: [1, 1, 1],
                    block: [128, 1, 1],
                    registers: 32,
                    static_shared: 0,
                    dynamic_shared: 0,
                    name: 1,
                    graph: 0,
                    graph_node: 0,
                    launch_type: 0,
                    channel: 0,
                }),
                Record::Api(Api {
                    domain: 1,
                    cbid: 307,
                    name: 2,
                    start: 5,
                    end: 8,
                    thread: 99,
                    correlation: 3,
                    result: 0,
                }),
            ],
            names: vec![(1, "gemm".to_owned()), (2, "cuLaunchKernel".to_owned())],
        });
        rows.send(Message::Fences {
            device: 0,
            records: vec![FenceRecord {
                time: 100,
                probe: 4,
                step: 9,
            }],
        });
        drop(rows);
        let summary = writer.finish().unwrap();
        assert_eq!(summary.rows.get("kernel"), Some(&1));
        assert!(summary.bytes > 0);
        // One file: the write-ahead log was checkpointed into it.
        assert!(!directory.path().join("rank0.sqlite-wal").exists());

        let connection = Connection::open(&path).unwrap();
        let version: i64 = connection
            .query_row("SELECT version FROM schema_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
        let joined: (String, String) = connection
            .query_row(
                "SELECT k.text, a.text FROM kernel JOIN api USING (correlation) \
                 JOIN strings k ON k.id = kernel.name JOIN strings a ON a.id = api.name",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(joined, ("gemm".to_owned(), "cuLaunchKernel".to_owned()));
        let session: String = connection
            .query_row(
                "SELECT value FROM metadata WHERE key = 'session'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(session, "7");
        let probe: i64 = connection
            .query_row("SELECT probe FROM fence", [], |row| row.get(0))
            .unwrap();
        assert_eq!(probe, 4);
    }

    #[test]
    fn an_existing_file_is_not_overwritten() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("taken.sqlite");
        std::fs::write(&path, b"").unwrap();
        assert!(Writer::create(path, Vec::new()).is_err());
    }
}
