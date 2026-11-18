//! The parts of a serving engine that the jobs act on, over a Storage Server in the test's
//! process and workers that hold no device: one scheduler on its thread, over a session whose
//! region of the KV planes is made of four segments, with the engine's worker of its one worker
//! group, whose rank serves generation 1 of no kernel, and a weight binding of no rank; and the
//! same parts of an engine of a later epoch of the same layout beside it ([`Node::successor`]).
//!
//! The tests read what the jobs leave: the scheduler's replies and statistics, the generation
//! each worker serves, and the segment table the Storage Server holds.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use anyhow::bail;
use crossbeam_channel::{Receiver, Sender};
use proto::config::{SchedulerConfig, WorkerGroupConfig};
use proto::report::SchedulerStats;
use proto::request::GenerationLimits;
use service::lock;
use storage::runtime::{self, Class, Open, Opened, Region, Segments, Tier};

use super::at_boundary;
use crate::executor::WorkerExecutor;
use crate::executor::tests::{GROUP, node};
use crate::memory::client::StorageClient;
use crate::process::binding::{Target, WeightBinding};
use crate::scheduler::tests::{capacity, idle_policy};
use crate::scheduler::{Message, Output, Scheduler};
use testkit::LONG;
use testkit::serving::{DEPLOYMENT, StorageServer};

/// The units of the region of the KV planes, the first of them reserved, in four segments.
pub(in crate::process) const KV_UNITS: u32 = 64;
pub(in crate::process) const SEGMENT_UNITS: u32 = 16;

/// The bytes of one unit of the KV planes on one rank.
pub(in crate::process) const UNIT_BYTES: u64 = 4096;

/// Segment `ordinal` of the KV planes, which epoch 1's ranks created, as a segment table states
/// it.
pub(in crate::process) fn segment(ordinal: u32) -> (storage::Segment, u32, u32) {
    let stated = storage::Segment { epoch: 1, ordinal };
    (stated, ordinal * SEGMENT_UNITS, SEGMENT_UNITS)
}

/// The layout of epoch `epoch`'s session: the region of the KV planes, `kv`, of one block class,
/// and the slot rows of group [`GROUP`]; the class of stored trajectories takes from no region.
/// Epoch 1's ranks created the four segments of `kv` before it opened; a later epoch of the same
/// layout states none, and maps those the Storage Server holds.
fn layout(epoch: u64) -> Open {
    let segments = if epoch == 1 {
        (0..KV_UNITS / SEGMENT_UNITS).map(segment).collect()
    } else {
        Vec::new()
    };
    let slots = format!("slots {epoch}");
    let class = |units| Class {
        units,
        host: None,
        state_in_slot: false,
        tensors: [0; 32],
        encoding: [0; 32],
        source: None,
    };

    Open {
        version: runtime::VERSION,
        epoch,
        trial: false,
        regions: vec![
            Region {
                name: "kv".to_owned(),
                tier: Tier::Device,
                space: "kv".to_owned(),
                unit_bytes: UNIT_BYTES,
                part_bytes: UNIT_BYTES,
                reserved: 1,
                units: KV_UNITS - 1,
                limit: KV_UNITS,
                segments,
                memory: String::new(),
            },
            Region {
                name: slots.clone(),
                tier: Tier::Device,
                space: slots,
                unit_bytes: 256,
                part_bytes: 256,
                reserved: 1,
                units: 4,
                limit: 5,
                segments: Vec::new(),
                memory: String::new(),
            },
        ],
        classes: vec![class(1), class(0)],
        trajectory: 1,
        groups: vec![runtime::Group {
            id: GROUP,
            memory: "memory".to_owned(),
            slots: 1,
            regions: vec![vec![0], Vec::new()],
            receives: Vec::new(),
            parts: Vec::new(),
        }],
        schedulers: vec![vec![GROUP]],
        peers: Vec::new(),
    }
}

/// The parts of a serving engine of epoch 1 on the meta device (module documentation).
pub(in crate::process) struct Node {
    /// The scheduler's inbox.
    pub(in crate::process) schedulers: Vec<Sender<Message>>,
    /// What the scheduler reports.
    pub(in crate::process) outputs: Receiver<Output>,
    /// The worker group's one rank.
    pub(in crate::process) workers: Vec<Arc<crate::worker::Worker>>,
    /// The node's route over its rank.
    pub(in crate::process) route: Arc<crate::executor::Route>,
    pub(in crate::process) binding: Arc<WeightBinding>,
    pub(in crate::process) head: Mutex<StorageClient>,
    /// The generation the scheduler's batches carry.
    pub(in crate::process) served: Arc<crate::batch::Generations>,
    threads: Vec<JoinHandle<()>>,
    /// The Storage Server's socket.
    socket: PathBuf,
    /// The Storage Server, which the node of epoch 1 runs and a successor shares.
    server: Option<StorageServer>,
}

impl Node {
    pub(in crate::process) fn start() -> Self {
        let server = StorageServer::start();
        let mut node = Self::open(&server.socket, layout(1));
        node.server = Some(server);
        node
    }

    /// An engine of epoch `epoch` beside this one, of the same layout, whose division numbers its
    /// KV planes `units` units, the padding unit included, as one stating another
    /// `kv_cache_config.max_tokens` numbers them: its session names the region of the KV planes
    /// this one's names and, as a start's does, asks for no unit of it at `Open` and states no
    /// segment of it; its one scheduler runs over a worker of its own.
    pub(in crate::process) fn successor(&self, epoch: u64, units: u32) -> Self {
        let mut layout = layout(epoch);
        layout.regions[0].units = 0;
        layout.regions[0].limit = units;
        Self::open(&self.socket, layout)
    }

    /// The parts of an engine of `layout`'s epoch whose session the Storage Server at `socket`
    /// answers, its scheduler running over the session as the head keeps it.
    fn open(socket: &Path, layout: Open) -> Self {
        let epoch = layout.epoch;
        let mut head = StorageClient::connect_head(socket, DEPLOYMENT, epoch, epoch).unwrap();
        let mut pairs = head.open(layout).unwrap();
        let (layout, opened) = head
            .session()
            .map(|(layout, opened)| (layout.clone(), opened.clone()))
            .unwrap();

        let (route, workers) = node(1, &Arc::default());
        let (inbox, messages) = crossbeam_channel::unbounded();
        let (output, outputs) = crossbeam_channel::unbounded();
        let served = Arc::new(crate::batch::Generations::new(1, 1));

        let groups =
            WorkerExecutor::new(Arc::clone(&route), &[GROUP], inbox.clone(), Arc::default())
                .unwrap();
        let config = WorkerGroupConfig::data_parallel(
            &["localhost".to_owned()],
            proto::config::DeviceKind::Cpu,
            1,
            1,
            2,
            BTreeMap::new(),
        )
        .remove(0);
        let scheduler = Scheduler::new(
            SchedulerConfig::default(),
            GenerationLimits::default(),
            &layout,
            &opened,
            0,
            pairs.remove(0),
            vec![(config, capacity())],
            idle_policy(),
            scheduling::Parameters::default(),
            Box::new(groups),
            output,
            Arc::clone(&served),
        )
        .unwrap();
        let thread = std::thread::spawn(move || scheduler.run(messages));

        Self {
            schedulers: vec![inbox],
            outputs,
            workers,
            route,
            binding: Arc::new(WeightBinding::new(
                Vec::new(),
                Target::default(),
                Vec::new(),
            )),
            head: Mutex::new(head),
            served,
            threads: vec![thread],
            socket: socket.to_owned(),
            server: None,
        }
    }

    /// The number the buffer pool gave the region of the KV planes.
    pub(in crate::process) fn kv(&self) -> u8 {
        lock(&self.head).session().unwrap().1.regions[0].0
    }

    /// The segment table of the KV planes' memory space as the Storage Server holds it: what a
    /// session of the same layout, of epoch `epoch`, opened now is answered.
    pub(in crate::process) fn segments_seen(&self, epoch: u64) -> Segments {
        self.seen(epoch).segments[0].clone()
    }

    /// The units in the region of the KV planes, the most its blocks can take: what a session of
    /// the same layout, of epoch `epoch`, opened now is answered.
    pub(in crate::process) fn units_seen(&self, epoch: u64) -> u32 {
        self.seen(epoch).regions[0].1
    }

    /// The buffer pool's answer to a session of the same layout, of epoch `epoch`, opened now
    /// and closed again.
    fn seen(&self, epoch: u64) -> Opened {
        let mut head = StorageClient::connect_head(&self.socket, DEPLOYMENT, epoch, epoch).unwrap();
        head.open(layout(epoch)).unwrap();
        let opened = head.session().unwrap().1.clone();
        head.close().unwrap();
        opened
    }

    /// The next statistics the scheduler reports.
    pub(in crate::process) fn next_stats(&self) -> SchedulerStats {
        loop {
            match self.outputs.recv_timeout(LONG).unwrap() {
                Output::Stats(stats) => return stats,
                Output::Failed(error) => panic!("the scheduler failed: {error}"),
                _ => {}
            }
        }
    }

    /// The steps the scheduler took from one of its reports to the next, both sent after this
    /// call: a paused scheduler takes none.
    pub(in crate::process) fn steps_in_a_second(&self) -> u64 {
        // A report already sent may have been counted before the scheduler's last message.
        while self.outputs.try_recv().is_ok() {}
        let first = self.next_stats();
        let second = self.next_stats();
        second.steps - first.steps
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        for inbox in &self.schedulers {
            let _ = inbox.send(Message::Shutdown);
        }
        for thread in self.threads.drain(..) {
            thread.join().unwrap();
        }
        let _ = lock(&self.head).close();
    }
}

#[test]
fn a_boundary_resumes_every_scheduler_after_its_work_failed() {
    let node = Node::start();

    let failed: anyhow::Result<()> = at_boundary(&node.schedulers, || {
        // The work runs at the boundary: the scheduler takes no step meanwhile.
        assert_eq!(node.steps_in_a_second(), 0);
        bail!("the work fails")
    });
    assert_eq!(failed.unwrap_err().to_string(), "the work fails");

    // The scheduler resumed: it steps again.
    assert!(node.steps_in_a_second() > 0);
}
