//! The graph lifecycles a runner owns: capturing a bucket, preparing one generation's bucket from
//! another's walk, switching over, reverting, discarding and purging.
//!
//! Nothing here names a token, a cache column or a sampler. The forward a bucket runs is the
//! caller's: it hands in [`Walked`], the capture's template with its per-launch node table, and the
//! kernels its own generation resolved. That is what makes the causal-LM runner and the media
//! runners one implementation of 4.5's capture-and-rebind rule and 7.3's probation and purge,
//! rather than two or four that must be kept in step.

use std::sync::Arc;

use crate::batch::{Job, Jobs};
use crate::arena::StreamId;
use crate::captured::{Captured, Nodes};
use crate::error::{Error, Result};
use crate::arena::Pools;
use crate::graph::Graphs;
use crate::plan::Bucket;
use crate::runner::RunnerId;

/// The generations one runner holds: the one that serves, the one being prepared, and the one kept
/// for a revert (7.3).
///
/// The graphs, their capture order and their lifetimes are all keyed by these three numbers, so they
/// live beside them rather than in each runner.
pub(crate) struct Lifecycle {
    /// The runner this lifecycle belongs to, which every job it builds names.
    pub(crate) id: RunnerId,
    /// The graphs captured for every generation this runner prepared.
    pub(crate) graphs: Graphs,
    /// The generation that serves: a batch the mechanism hands over is planned in its graphs.
    pub(crate) generation: Arc<ops::Generation>,
    /// The generation prepared and waiting for its switchover (7.3 step 3).
    pub(crate) next: Option<Arc<ops::Generation>>,
    /// The generation kept for a revert while the prepared one is on probation (7.3 step 4).
    pub(crate) previous: Option<Arc<ops::Generation>>,
    /// The stream every graph of every generation is captured and replayed on.
    pub(crate) stream: platform::Stream,
    /// The device's pools, through which a released generation's arena regions go back (4.1).
    ///
    /// Attached once the device has opened them: a runner is built before the pool set is complete,
    /// and a purge before that attachment would have nothing to release.
    pub(crate) pools: Option<Arc<std::sync::Mutex<Pools>>>,
}

impl Lifecycle {
    /// A lifecycle serving `generation` with no other generation held.
    pub(crate) fn new(id: RunnerId, generation: Arc<ops::Generation>, stream: platform::Stream) -> Self {
        Self {
            id,
            graphs: Graphs::default(),
            generation,
            next: None,
            previous: None,
            stream,
            pools: None,
        }
    }

    /// The device's pools, or a refusal naming what is missing.
    ///
    /// A release without them would leave a generation's regions mapped and unaccounted for, which
    /// is why the paths that release ask for them rather than skipping.
    pub(crate) fn pool(&self) -> Result<std::sync::MutexGuard<'_, Pools>> {
        self.pools
            .as_ref()
            .map(|pools| {
                pools
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            })
            .ok_or_else(|| Error::State("the runner has no graph pools yet"))
    }

    /// Hands the lifecycle the device's pools, once the device has opened them.
    pub(crate) fn attach_pools(&mut self, pools: Arc<std::sync::Mutex<Pools>>) {
        self.pools = Some(pools);
    }

    /// The pool stream this runner's graphs are planned and replayed on.
    ///
    /// A graph pool is per stream (4.5), and two runners on one device fork their own streams, so the
    /// identity is the stream's own handle rather than a position in a list the executor keeps
    /// elsewhere.
    pub(crate) fn stream_id(&self) -> StreamId {
        StreamId(self.stream.raw())
    }

    /// The generation numbered `number`, from any this runner holds.
    pub(crate) fn generation_named(&self, number: u64) -> Result<Arc<ops::Generation>> {
        for candidate in [Some(&self.generation), self.next.as_ref(), self.previous.as_ref()]
            .into_iter()
            .flatten()
        {
            if candidate.number() == number {
                return Ok(Arc::clone(candidate));
            }
        }
        Err(Error::Invariant(format!(
            "the runner holds no generation {number}"
        )))
    }    pub(crate) fn switchover(&mut self, generation: u64) -> Result<()> {
        // From the next batch on, the prepared generation serves; the old one stays for a revert
        // (7.3 step 4).
        let prepared = self.next.take().filter(|next| next.number() == generation);
        let Some(prepared) = prepared else {
            // The daemon asks for the cut-over only after `precopy` built the generation. Taking
            // the number anyway would serve a generation with no kernels and no graphs, which
            // answers every later batch with "no graph for this bucket" - a disagreement between
            // the daemon and this process stated as a defect rather than hidden as a switch.
            return Err(Error::Invariant(format!(
                "switchover to generation {generation} found no preparation: this runner serves \
                 generation {} and prepared none other",
                self.generation.number()
            )));
        };
        self.previous = Some(std::mem::replace(&mut self.generation, prepared));
        Ok(())
    }
    pub(crate) fn revert(&mut self, generation: u64) {
        // A revert after the switchover is the switchover reversed (7.3 step 4): the generation
        // kept for a revert serves again and the failed one takes its place as the kept
        // generation, so its graphs and kernels live until `purge` releases them after the last
        // batch that used them.
        if self
            .previous
            .as_ref()
            .is_some_and(|previous| previous.number() == generation)
        {
            let serving = self.previous.take();
            if let Some(serving) = serving {
                let failed = std::mem::replace(&mut self.generation, serving);
                self.previous = Some(failed);
                tracing::info!(
                    generation,
                    serving = self.generation.number(),
                    "reverted the switched-over generation; the kept one serves again"
                );
                return;
            }
        }
        // A prepared generation that never took over: its graphs go, and with them the kernels
        // they held. The serving generation's graphs, arena plans and sampler regions stay
        // exactly as they were, so an abandoned update cannot disturb a batch in flight. This is
        // `discard`, not `purge`: an abandoned generation is one nobody serves from, so its whole
        // set goes whatever its number is beside the serving one.
        let dropped = self.graphs.discard(generation);
        let dropped_count = dropped.len();
        if self.next.as_ref().map(|next| next.number()) == Some(generation) {
            self.next = None;
        }
        tracing::info!(
            dropped = dropped_count,
            generation,
            "reverted the prepared generation; the serving generation is unchanged"
        );
    }

    pub(crate) fn discard(&mut self, generation: u64) -> Result<()> {
        // A generation in service is not a prepared one: discarding it would take the graphs that
        // serve batches away and leave the device able to answer nothing. The engine discards the
        // update it abandoned on a drain, which is always newer than the serving generation, so a
        // number that names either of the two live ones is a fault of the caller and is refused by
        // name rather than obeyed.
        let serving = self.generation.number();
        let probation = self.previous.as_ref().map(|previous| previous.number());
        if generation == serving || probation == Some(generation) {
            return Err(Error::Invariant(format!(
                "generation {generation} is one this runner serves (serving {serving}, probation {}) \
                 and cannot be discarded",
                probation.map(|number| number.to_string()).unwrap_or_else(|| "none".into())
            )));
        }
        // A prepared generation that never served: its graphs go, then their arena regions where
        // no other generation holds the bucket, in `Captured`'s order. The serving generation and
        // probation's old one are keyed separately and are not touched. What the capture recorded
        // about that generation's annotation goes with it.
        let dropped = self.graphs.discard(generation);
        let dropped_count = dropped.len();
        for bucket in dropped
            .iter()
            .filter(|bucket| !self.graphs.holds(**bucket))
        {
            if let Err(error) = self
                .pool()
                .and_then(|mut pools| pools.release_bucket(self.stream_id(), *bucket))
            {
                tracing::warn!(%error, %bucket, "releasing a discarded bucket's arena region");
            }
        }
        // The generation object itself: a prepared one is held as `next` until switchover takes
        // it, so an abandoned precopy drops the last reference that keeps its kernels loaded.
        if self
            .next
            .as_ref()
            .is_some_and(|next| next.number() == generation)
        {
            self.next = None;
        }
        tracing::debug!(
            dropped = dropped_count,
            generation,
            "discarded a prepared generation that never served"
        );
        Ok(())
    }

    pub(crate) fn purge(&mut self, generation: u64) {
        // The graphs go first, then their arena regions: `Captured` guarantees that order, and a
        // region released under a live executable would be reused by the next generation's
        // capture while the old graph still addressed it. Exactly this generation's graphs go - a
        // released generation may be numbered either side of the serving one.
        let dropped = self.graphs.purge(generation, self.generation.number());
        let dropped_count = dropped.len();
        let mut released = 0u64;
        // Both generations' graphs can address one bucket's plan: the pool records a plan per
        // bucket, not per generation, so a purged generation's bucket keeps its region while the
        // serving (or probation's) generation still has a graph for it.
        for bucket in dropped
            .iter()
            .filter(|bucket| !self.graphs.holds(**bucket))
        {
            let released_bucket = self
                .pool()
                .and_then(|mut pools| pools.release_bucket(self.stream_id(), *bucket));
            match released_bucket {
                Ok(bytes) => released += bytes,
                // A pool that cannot shrink keeps the bytes mapped until the executor closes;
                // the purge itself still holds, so this is not a reason to fail the command.
                Err(error) => {
                    tracing::warn!(%error, %bucket, "releasing a purged bucket's arena region");
                }
            }
        }
        tracing::debug!(
            dropped = dropped_count,
            released,
            generation,
            "purged graphs of a replaced generation"
        );
        if self
            .previous
            .as_ref()
            .is_some_and(|previous| previous.number() <= generation)
        {
            // The last `Arc` of the purged generation drops here; every kernel whose graph was
            // rebound or destroyed unloads its library with it (7.3's purge). The generation a
            // revert kept is named by the purge too - it is the *newer* of the two then - so the
            // test is "this generation or an older one", not "older than this".
            self.previous = None;
        }
    }

    /// Captures one bucket for one generation from a walk the caller performed (4.5).
    ///
    /// The walk ran the bucket's forward eagerly once under a stream capture and reported every
    /// operator launch's nodes; what is left is to hold the kernels the graph references and to
    /// instantiate the executable from the template.
    pub(crate) fn capture(
        &mut self,
        bucket: Bucket,
        generation: u64,
        walked: Walked,
        kernels: &Arc<ops::Generation>,
    ) -> Result<()> {
        let held = hold_kernels(kernels, &walked.nodes)?;
        let captured = Captured::new(walked.template, walked.nodes, held)?;
        self.graphs.insert(bucket, generation, captured);
        Ok(())
    }

    pub(crate) fn rebind(
        &mut self,
        bucket: Bucket,
        generation: u64,
        walked: Walked,
        kernels: &Arc<ops::Generation>,
    ) -> Result<()> {
        let current = self.generation.number();
        // Preparing a bucket for the generation it already serves from would replace a live graph
        // under a batch that is running on it. `precopy` names the generation it prepares, and an
        // update that changes nothing prepares no bucket at all, so this is a disagreement.
        if generation == current {
            return Err(Error::Invariant(format!(
                "a rebind names bucket {bucket}'s serving generation {current} as the one it \
                 prepares"
            )));
        }
        let served = self.graphs.captured(bucket, current).ok_or_else(|| {
            Error::Invariant(format!(
                "no graph of bucket {bucket} in generation {current} to prepare the next one from"
            ))
        })?;
        // A walk with no node table means the operator launches were not annotated, so there is no
        // node to set the prepared generation's parameters on and the bucket must be recaptured
        // rather than prepared from a structure nobody can address.
        let unannotated = walked.nodes.is_empty();
        // Which fact declined the rebind, when one did: a count, a node's kind, or an edge whose
        // kind changed. A recapture that says only "the structure changed" leaves the reader to
        // guess whether a program was built differently or the same program is ordered differently.
        let mismatch = served.structure_mismatch(&walked.nodes)?;
        if unannotated || mismatch.is_some() {
            // The launch structure changed: the bucket is recaptured from the walk's template,
            // which already carries the new kernels and their arguments. The generation being
            // replaced keeps its own graph and executable, which a revert serves from.
            let held = hold_kernels(kernels, &walked.nodes)?;
            let recaptured = Captured::new(walked.template, walked.nodes, held)?;
            self.graphs.insert(bucket, generation, recaptured);
            if unannotated {
                tracing::error!(
                    bucket = %bucket,
                    generation,
                    "the walk reported no node table; recaptured the bucket rather than silently \
                     rebinding nothing"
                );
            } else {
                tracing::info!(
                    bucket = %bucket,
                    generation,
                    mismatch = %mismatch.as_deref().unwrap_or("the walk's structure differs"),
                    "the launch structure changed; recaptured the bucket"
                );
            }
            return Ok(());
        }
        // 4.5's rebind, on the serving graph's own template: the prepared generation gets an
        // executable instantiated from the structure this bucket already has, and the walk's
        // parameters are then set on it. Nothing is captured a second time, and the generation
        // being replaced keeps the executable it serves from - probation and a revert need it
        // (7.3 step 4), so a rebind that rewrote the live executable in place would leave a
        // revert with no graph to serve.
        let template = served.shared_template();
        let mut launches: Vec<(&'static str, usize, Vec<platform::OwnedKernelNodeParams>)> =
            Vec::with_capacity(walked.nodes.len());
        let mut ordinal_of: std::collections::HashMap<&'static str, usize> =
            std::collections::HashMap::new();
        for (operator, _, launch) in &walked.nodes {
            let ordinal = ordinal_of.entry(*operator).or_insert(0);
            let mut params = Vec::with_capacity(launch.len());
            for node in launch {
                if node.node.kind()? != platform::GraphNodeKind::Kernel {
                    continue;
                }
                params.push(walked.template.kernel_node_params(&node.node)?);
            }
            launches.push((*operator, *ordinal, params));
            *ordinal += 1;
        }
        let held = hold_kernels(kernels, &walked.nodes)?;
        let prepared = Captured::from_template(template, walked.nodes, held)?;
        let mut applied = 0usize;
        for (operator, ordinal, params) in launches {
            applied += prepared.rebind_launch(operator, ordinal, &params)?;
        }
        self.graphs.insert(bucket, generation, prepared);
        // At `info`, because a run's log is where the delivery test's two mechanisms are told apart:
        // a recapture states itself at `info` a page below, and a rebind that stated itself at
        // `debug` would be indistinguishable from a bucket the update never prepared at all. `applied`
        // is the number of node tables set on the executable the serving graph's own template built,
        // which is the strongest statement this layer can make about what happened (7.3).
        tracing::info!(bucket = %bucket, generation, applied, "rebound a bucket for a new generation");
        Ok(())
    }

    pub(crate) fn precopy(&mut self, next: &Arc<ops::Generation>) -> Result<Jobs> {
        // One update at a time: a second generation prepared while the first is still waiting for
        // its switchover would leave the first one's graphs unreachable and unaccounted for. The
        // daemon discards an update before starting the next, so this is a disagreement, not a
        // sequence to support.
        if let Some(prepared) = &self.next
            && prepared.number() != next.number()
        {
            return Err(Error::Invariant(format!(
                "precopy of generation {} while generation {} is prepared and has not been \
                 switched over or discarded",
                next.number(),
                prepared.number()
            )));
        }
        // Which (operator, key) pairs changed between the serving generation and `next` (7.3
        // step 1). A kernel is unchanged when it is the same `Arc`, so an untouched program keeps
        // its bucket as it is.
        let mut changed: Vec<(&'static str, String)> = Vec::new();
        for (operator, key, kernel) in self.generation.iter() {
            let same = next
                .kernel(operator, key)
                .map(|candidate| Arc::ptr_eq(candidate, kernel))
                .unwrap_or(false);
            if !same {
                changed.push((operator, key.to_owned()));
            }
        }
        // A launch `next` adds is a change too: its bucket has nothing to compare it against, and
        // only a recapture can serve it (4.5).
        for (operator, key, _) in next.iter() {
            if !self.generation.contains(operator, key) {
                changed.push((operator, key.to_owned()));
            }
        }
        // Every changed operator's buckets are prepared once. Whether a bucket is rebound or
        // recaptured is not decided here: it is a fact about the walk's launch structure against
        // the captured graph, and only the walk can establish it (4.5's rebind condition).
        let mut jobs = Vec::new();
        let mut buckets: std::collections::BTreeSet<Bucket> = std::collections::BTreeSet::new();
        for (operator, key) in changed {
            // The buckets that launch this operator, whichever key they use: a program under a new
            // key of the same operator changes the launch structure of the buckets that launch the
            // operator at all, and only a walk over them can tell rebind from recapture (4.5).
            let mut launching = self.graphs.using(operator, self.generation.number());
            // Whether the graphs could name the buckets that launch the operator. When they cannot,
            // the walk is the only thing that knows, and the paired forward is not attempted: it
            // needs a bucket that really launches the operator, and the graphs are exactly what
            // cannot say which those are. The mechanism's own shadow batches (`Batches::shadow`)
            // still compare the two kernels per operator, which is the check 7.3 step 2 asks for.
            let named = !launching.is_empty();
            if launching.is_empty() {
                // No captured graph names the operator, which means the launch annotation was not
                // reported when the bucket was captured. The graph cannot say which buckets launch
                // it, so every serving bucket is prepared - and each walk recaptures, because
                // there is no node to rebind - rather than the update quietly doing nothing.
                launching = self.graphs.buckets(self.generation.number());
                // An operator the serving generation already holds a kernel for is one its graphs
                // launch, so a graph that names none of them was captured without the annotation.
                // A genuinely new operator names no graph for the honest reason, and says nothing.
                let existed = self
                    .generation
                    .iter()
                    .any(|(name, _, _)| name == operator);
                if existed && !launching.is_empty() {
                    tracing::error!(
                        %operator,
                        "no captured graph names {operator}: the operator launches are not \
                         annotated, so every serving bucket is recaptured rather than rebound"
                    );
                }
            }
            // The paired forward runs in the **smallest** bucket that launches the operator: what it
            // establishes is a fact about the operator's program under this descriptor key, not about
            // the bucket, so the cheapest forward the model has establishes it as well as every other
            // one would, and the mechanism's own shadow batches compare the buckets its resident
            // requests occupy (7.3 step 2). Comparing every bucket here would spend `buckets x
            // operators` full forwards of the precopy's budget re-establishing one fact.
            let compared = named
                .then(|| launching.iter().copied().min())
                .flatten()
                // A kernel cannot be compared with one of another descriptor (`Forward::operate`
                // pairs on operator and key), so a key one generation does not hold - a new launch,
                // or one whose descriptor changed - is left to the walk, which recaptures.
                .filter(|_| next.kernel(operator, &key).is_ok())
                .filter(|_| self.generation.kernel(operator, &key).is_ok());
            for bucket in launching {
                if compared == Some(bucket) {
                    jobs.push(Job::Shadow {
                        runner: self.id,
                        operator,
                        key: key.clone(),
                        bucket,
                        generation: next.number(),
                    });
                }
                buckets.insert(bucket);
            }
        }
        for bucket in buckets {
            jobs.push(Job::Rebind {
                runner: self.id,
                bucket,
                generation: next.number(),
            });
        }
        self.next = Some(Arc::clone(next));
        Ok(Jobs::new(jobs))
    }
}

pub(crate) fn walk_capture(
    stream: &platform::Stream,
    walk: impl FnOnce() -> Result<()>,
) -> Result<Walked> {
        // One thread records and takes the capture back: an `Arc`/`Mutex` here would say the
        // handle is shared across threads, which it is not, and the allocator is right to warn
        // about a non-`Send` handle in an `Arc`.
        let capture = std::rc::Rc::new(std::cell::RefCell::new(platform::Capture::begin(
            stream,
            platform::CaptureMode::Relaxed,
        )?));
        let mut recording = Recording::new(std::rc::Rc::clone(&capture));
        // The forward runs with the generation being walked and the bucket's arena, so every
        // operator launch becomes a node of the template and is reported to `recording`.
        ops::capture::set_sink(Some(&mut recording));
        let walked = walk();
        ops::capture::set_sink(None);
        let nodes = recording.nodes;
        if let Some(failure) = recording.failure {
            return Err(failure.into());
        }
        walked?;
        let capture = std::rc::Rc::try_unwrap(capture).map_err(|_| {
            Error::Invariant("the capture is still shared by its recorder".into())
        })?;
        let capture = capture.into_inner();
        let template = capture.end()?;
        // The node table comes from the capture sink, and the template is the record of what was
        // actually launched: the two can only be compared here. A graph that launched kernels and
        // reported no node table serves batches correctly but cannot be rebound, because there is
        // no node to set the new kernel's parameters on. It is recorded rather than refused - a
        // programme update is what needs it, and `precopy` refuses there by name instead of letting
        // the update silently do nothing.
        Ok(Walked { template, nodes })
    }

/// Every kernel a walk's launches reference, so a graph built from it keeps them loaded (4.5).
pub(crate) fn hold_kernels(
    kernels: &Arc<ops::Generation>,
    nodes: &crate::captured::Nodes,
) -> Result<Vec<Arc<ops::Kernel>>> {
    let mut held: Vec<Arc<ops::Kernel>> = Vec::new();
    for (operator, key, _) in nodes {
        let kernel = kernels.kernel(operator, key)?;
        if !held.iter().any(|existing| Arc::ptr_eq(existing, kernel)) {
            held.push(Arc::clone(kernel));
        }
    }
    Ok(held)
}

impl ops::capture::Sink for Recording {
    fn launched(&mut self, operator: &'static str, key: &str) {
        // The walk's edges, not only its nodes: a programmatic edge orders less than a
        // completion edge, and 4.5's rebind condition compares the structure those kinds make.
        let recorded = self.capture.borrow_mut().record_walk();
        match recorded {
            Ok(nodes) => self.nodes.push((operator, key.to_owned(), nodes)),
            Err(error) => {
                if self.failure.is_none() {
                    self.failure = Some(error);
                }
            }
        }
    }
}

/// One walk's result: the throwaway template the capture produced, and the nodes each operator
/// launch added to it.
pub(crate) struct Walked {
    pub(crate) template: platform::Graph,
    pub(crate) nodes: Nodes,
}

/// A capture sink: which operator launch added which graph nodes, in capture order.
///
/// Public because a device test of the rebind mechanism must record with the sink the runner
/// captures with: a copy of it in a test would measure the copy.
pub struct Recording {
    capture: std::rc::Rc<std::cell::RefCell<platform::Capture>>,
    nodes: Nodes,
    failure: Option<platform::Error>,
}

impl Recording {
    /// Records into `capture`, which the caller installs as the sink and takes back afterwards.
    pub fn new(capture: std::rc::Rc<std::cell::RefCell<platform::Capture>>) -> Self {
        Self {
            capture,
            nodes: Vec::new(),
            failure: None,
        }
    }

    /// The per-launch node table, in capture order: `Captured.index_nodes`'s own input.
    pub fn nodes(&self) -> &Nodes {
        &self.nodes
    }

    /// The first failure a record reported, if any: a capture that failed part-way through.
    pub fn failure(&self) -> Option<&platform::Error> {
        self.failure.as_ref()
    }

    /// The node table, the recorder consumed.
    pub fn into_nodes(self) -> Nodes {
        self.nodes
    }
}
