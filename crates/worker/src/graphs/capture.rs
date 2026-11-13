//! The device side of a captured graph: capturing a forward into segments, the executable a
//! replay launches, and rebinding it to another generation's kernels.
//!
//! A capture runs a bucket's forward once on a capturing stream. Every launch of the forward
//! passes through the context's route ([`Recorder`]), which launches it and then reads from the
//! driver the graph nodes that launch added, so the nodes of each launch are known in launch
//! order: the node table a rebind sets new kernel parameters through. A graph break
//! (`ops::graph_break`) ends the segment being captured and begins the next; the host work
//! it states is kept with the segment before it.
//!
//! The capture in progress is this thread's ([`SESSION`]): a capture is begun, recorded and
//! ended on the executor thread of its device, and the driver's capture object is bound to that
//! thread.

use std::cell::RefCell;
use std::sync::Arc;

use ops::breaks::HostWork;
use ops::{Generation, Kernel};
use platform::{
    Capture, CaptureMode, Graph, GraphExec, GraphNode, GraphNodeKind, OwnedKernelNodeParams,
    Stream, WalkedNode,
};

use crate::{Error, Result};

thread_local! {
    /// The capture in progress on this thread, or none.
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

/// One capture in progress: the segment being captured and the segments already ended.
struct Session {
    stream: Stream,
    /// The driver's capture of the current segment; `None` once a failure ended it.
    capture: Option<Capture>,
    /// The nodes each launch of the current segment added, in launch order.
    nodes: Vec<Vec<WalkedNode>>,
    /// The segments a graph break ended, each with its node table and the break's host work.
    ended: Vec<(Graph, Vec<Vec<WalkedNode>>, HostWork)>,
}

impl Session {
    /// Reads the nodes the launch just issued added to the current segment.
    fn launched(&mut self) -> platform::Result<()> {
        let Some(capture) = self.capture.as_mut() else {
            return Ok(());
        };

        self.nodes.push(capture.record_walk()?);
        Ok(())
    }

    /// Ends the current segment at a graph break stating `work`, and begins the next.
    fn broken(&mut self, work: HostWork) -> platform::Result<()> {
        let Some(capture) = self.capture.take() else {
            return Ok(());
        };

        let graph = capture.end()?;
        self.ended
            .push((graph, std::mem::take(&mut self.nodes), work));
        self.capture = Some(Capture::begin(&self.stream, CaptureMode::Relaxed)?);
        Ok(())
    }
}

/// The route of a capturing context: launches each call with the generation's kernel and
/// records the graph nodes the launch added.
pub(crate) struct Recorder;

impl ops::Route for Recorder {
    fn launch(
        &self,
        call: &dyn ops::Call,
        current: &Generation,
        stream: &Stream,
    ) -> ops::Result<bool> {
        let kernel = current.kernel(call.operator(), &call.key())?;
        call.launch(kernel, stream)?;

        SESSION.with(|session| match session.borrow_mut().as_mut() {
            Some(session) => session.launched().map_err(ops::Error::Platform),
            None => Ok(()),
        })?;
        Ok(true)
    }
}

/// The break taker of a capturing context: ends the segment being captured on this thread and
/// begins the next, keeping the break's host work.
pub(crate) fn take_break(work: HostWork) -> ops::Result<()> {
    SESSION.with(|session| match session.borrow_mut().as_mut() {
        Some(session) => session.broken(work).map_err(ops::Error::Platform),
        None => Err(ops::Error::Break(
            "a graph break was stated on a thread that captures nothing".into(),
        )),
    })
}

/// One captured segment: the kernels and copies between two graph breaks.
///
/// The fields are declared in the order they are released: the executable first, then the
/// template its nodes belong to.
struct Segment {
    /// The executable a replay launches. A rebind's executable is instantiated from the serving
    /// graph's template and carries the parameters in `rebound`.
    exec: GraphExec,
    /// The graph the executable was instantiated from; its nodes are the handles a rebind sets.
    template: Arc<Graph>,
    /// The nodes each launch of the segment added, in launch order.
    nodes: Vec<Vec<WalkedNode>>,
    /// The kernel parameters a rebind set on the executable, which a new executable of the
    /// template needs again ([`Executable::renew`]).
    rebound: Vec<(GraphNode, OwnedKernelNodeParams)>,
    /// The host work of the graph break that ends the segment; `None` for the last segment.
    then: Option<HostWork>,
}

/// One bucket's captured forward on the device: its segments and the kernels they call.
///
/// The segments are released before the kernels, so a kernel stays loaded while a graph that
/// calls it can still be launched.
pub(crate) struct Executable {
    segments: Vec<Segment>,
    /// Every kernel the segments call.
    kernels: Vec<Arc<Kernel>>,
    /// The kernel nodes one replay launches, over every segment: what a replay adds to the
    /// device's count of kernel launches. A rebind's executable has its template's.
    kernel_nodes: u64,
}

impl Executable {
    /// Captures `forward` on `stream`: its launches become the nodes of one segment per stretch
    /// between graph breaks.
    ///
    /// The caller installed a context on this thread whose route is [`Recorder`] and whose break
    /// taker is [`take_break`]. Returns the executable, which holds no kernel yet
    /// ([`Executable::hold`]), and what `forward` returned.
    ///
    /// # Errors
    ///
    /// When another capture is in progress on this thread, `forward` fails, or the driver
    /// refuses the capture or the instantiation.
    pub(crate) fn capture<T>(
        stream: &Stream,
        forward: impl FnOnce() -> Result<T>,
    ) -> Result<(Self, T)> {
        let capture = Capture::begin(stream, CaptureMode::Relaxed)?;
        let began = SESSION.with(|session| {
            let mut session = session.borrow_mut();
            if session.is_some() {
                return false;
            }

            *session = Some(Session {
                stream: stream.clone(),
                capture: Some(capture),
                nodes: Vec::new(),
                ended: Vec::new(),
            });
            true
        });
        if !began {
            return Err(Error::State(
                "a capture is already in progress on this thread",
            ));
        }

        let output = forward();

        // The session is taken back whatever the forward returned: dropping it ends the
        // driver's capture, so a failed forward leaves the stream usable.
        let Some(mut session) = SESSION.with(|session| session.borrow_mut().take()) else {
            return Err(Error::Invariant(
                "the capture's session was taken during its forward".into(),
            ));
        };
        let output = output?;

        let last = session
            .capture
            .take()
            .ok_or_else(|| Error::Invariant("the capture ended before its forward did".into()))?
            .end()?;

        let mut segments = Vec::with_capacity(session.ended.len() + 1);
        for (graph, nodes, work) in session.ended {
            segments.push(Segment::new(graph, nodes, Some(work))?);
        }
        segments.push(Segment::new(last, session.nodes, None)?);

        let kernel_nodes = Self::count_kernel_nodes(&segments)?;
        let executable = Self {
            segments,
            kernels: Vec::new(),
            kernel_nodes,
        };
        Ok((executable, output))
    }

    /// Holds `kernels`, every kernel the segments call, for as long as the executable exists.
    pub(crate) fn hold(&mut self, kernels: Vec<Arc<Kernel>>) {
        self.kernels = kernels;
    }

    /// Replays the executable on `stream`, after the work already queued on it: each segment's
    /// launch, and between two segments the host work of the break that separates them.
    ///
    /// # Errors
    ///
    /// A launch the driver refuses, or a break's host work that fails.
    pub(crate) fn launch(&self, stream: &Stream) -> Result<()> {
        for segment in &self.segments {
            segment.exec.launch(stream)?;

            if let Some(work) = &segment.then {
                work().map_err(|source| Error::from(ops::Error::Break(source)))?;
            }
        }
        Ok(())
    }

    /// Instantiates every segment's executable again from its template, with the parameters a
    /// rebind set: what a replay needs after the process's CUPTI recording changed how graphs
    /// are traced, or the storage the graph addresses was mapped again.
    ///
    /// # Errors
    ///
    /// When the driver refuses an instantiation or a node's parameters.
    pub(crate) fn renew(&mut self) -> Result<()> {
        for segment in &mut self.segments {
            let exec = segment.template.instantiate()?;
            for (node, params) in &segment.rebound {
                exec.set_kernel_node_params_owned(node.clone(), params)?;
            }

            // An executable still in flight is released by the driver when it completes.
            segment.exec = exec;
        }
        Ok(())
    }

    /// The kernel nodes one replay launches.
    pub(crate) fn kernel_nodes(&self) -> u64 {
        self.kernel_nodes
    }

    /// The kernel nodes of `segments`' templates, read from the driver once per executable.
    fn count_kernel_nodes(segments: &[Segment]) -> Result<u64> {
        let mut kernels = 0;
        for segment in segments {
            for node in segment.template.nodes()? {
                if node.kind()? == GraphNodeKind::Kernel {
                    kernels += 1;
                }
            }
        }
        Ok(kernels)
    }

    /// The executable of the same graphs with the kernels of the launches in `changed` taken
    /// from `fresh`, a capture of the same forward under another generation; `None` when the
    /// two captures differ in structure, so that `fresh` itself is what serves the generation.
    ///
    /// `changed` holds indices into the forward's launches in launch order, over all segments.
    /// The serving executable is left as it is: the generation it serves keeps it for a
    /// rollback. A rebound node keeps the arguments the serving graph captured and takes the
    /// function, grid, block and shared memory `fresh` launched with.
    ///
    /// # Errors
    ///
    /// When a changed launch's kernel takes parameters of other sizes than the serving one's, or
    /// the driver refuses an instantiation or a node's parameters.
    pub(crate) fn rebound(
        &self,
        fresh: &Self,
        changed: &[usize],
        kernels: Vec<Arc<Kernel>>,
    ) -> Result<Option<Self>> {
        if let Some(difference) = self.difference(fresh)? {
            tracing::info!(%difference, "the launch structure changed; the bucket is recaptured");
            return Ok(None);
        }

        let mut segments = Vec::with_capacity(self.segments.len());
        let mut first = 0;

        for (serving, captured) in self.segments.iter().zip(&fresh.segments) {
            let exec = serving.template.instantiate()?;
            let mut rebound = Vec::new();

            for (index, (held, walked)) in serving.nodes.iter().zip(&captured.nodes).enumerate() {
                if !changed.contains(&(first + index)) {
                    continue;
                }

                for (held, walked) in held.iter().zip(walked) {
                    if held.node.kind()? != GraphNodeKind::Kernel {
                        continue;
                    }

                    let serving_params = serving.template.kernel_node_params(&held.node)?;
                    let fresh_params = captured.template.kernel_node_params(&walked.node)?;
                    if widths(&serving_params) != widths(&fresh_params) {
                        return Err(Error::Invariant(format!(
                            "launch {} changed its kernel's parameters under the same launch \
                             structure, so its bucket cannot be rebound",
                            first + index
                        )));
                    }

                    let params = OwnedKernelNodeParams {
                        parameters: serving_params.parameters,
                        ..fresh_params
                    };
                    exec.set_kernel_node_params_owned(held.node.clone(), &params)?;
                    rebound.push((held.node.clone(), params));
                }
            }

            first += serving.nodes.len();
            segments.push(Segment {
                exec,
                template: Arc::clone(&serving.template),
                nodes: serving.nodes.clone(),
                rebound,
                then: serving.then.clone(),
            });
        }

        Ok(Some(Self {
            segments,
            kernels,
            kernel_nodes: self.kernel_nodes,
        }))
    }

    /// How `fresh`'s launch structure differs from this executable's, or `None` when a rebind
    /// may carry `fresh`'s kernels over: the same segments, each launch adding the same number
    /// of nodes of the same kinds, ordered by the same kinds of edges, and every launch of this
    /// executable naming at least one node to set.
    fn difference(&self, fresh: &Self) -> Result<Option<String>> {
        if self.segments.len() != fresh.segments.len() {
            return Ok(Some(format!(
                "{} segments were captured where the serving graph holds {}",
                fresh.segments.len(),
                self.segments.len()
            )));
        }

        let mut launch = 0;
        for (serving, captured) in self.segments.iter().zip(&fresh.segments) {
            if serving.nodes.len() != captured.nodes.len() {
                return Ok(Some(format!(
                    "a segment launched {} times where the serving graph's launched {}",
                    captured.nodes.len(),
                    serving.nodes.len()
                )));
            }

            for (held, walked) in serving.nodes.iter().zip(&captured.nodes) {
                if held.is_empty() {
                    return Ok(Some(format!(
                        "launch {launch} of the serving graph names no node to set"
                    )));
                }
                if held.len() != walked.len() {
                    return Ok(Some(format!(
                        "launch {launch} added {} node(s) where the serving graph's added {}",
                        walked.len(),
                        held.len()
                    )));
                }

                // A rebind carries parameters over by position and keeps the template's edges,
                // so the kinds of the nodes and of the edges that order them must agree: a
                // programmatic edge orders less than a completion edge.
                for (held, walked) in held.iter().zip(walked) {
                    let (serving_kind, fresh_kind) = (held.node.kind()?, walked.node.kind()?);
                    if serving_kind != fresh_kind {
                        return Ok(Some(format!(
                            "launch {launch} added a {fresh_kind:?} node where the serving \
                             graph's added a {serving_kind:?} node"
                        )));
                    }
                    if held.incoming != walked.incoming {
                        return Ok(Some(format!(
                            "launch {launch} is ordered by {:?} edges where the serving graph's \
                             is ordered by {:?}",
                            walked.incoming, held.incoming
                        )));
                    }
                }

                launch += 1;
            }
        }
        Ok(None)
    }

    /// The launches the executable's node table names, over all segments.
    pub(crate) fn launches(&self) -> usize {
        self.segments
            .iter()
            .map(|segment| segment.nodes.len())
            .sum()
    }
}

impl Segment {
    /// The segment of `graph`, instantiated, ending at the break stating `then`.
    fn new(graph: Graph, nodes: Vec<Vec<WalkedNode>>, then: Option<HostWork>) -> Result<Self> {
        let exec = graph.instantiate()?;

        Ok(Self {
            exec,
            template: Arc::new(graph),
            nodes,
            rebound: Vec::new(),
            then,
        })
    }
}

/// The byte sizes of a kernel node's parameters, in declaration order.
fn widths(params: &OwnedKernelNodeParams) -> Vec<usize> {
    params.parameters.iter().map(Vec::len).collect()
}

/// The kernels of `generation` that the launches of `table` call, each once.
///
/// A graph holds them so that a kernel stays loaded for as long as a graph that calls it can be
/// replayed.
///
/// # Errors
///
/// When the generation holds no kernel for a launch of the table.
pub(crate) fn kernels_of(
    generation: &Generation,
    table: &[(&'static str, String, String)],
) -> Result<Vec<Arc<Kernel>>> {
    let mut held: Vec<Arc<Kernel>> = Vec::new();

    for (operator, key, _) in table {
        if *operator == ops::breaks::BREAK {
            continue;
        }

        let kernel = generation.kernel(operator, key)?;
        if !held.iter().any(|existing| Arc::ptr_eq(existing, kernel)) {
            held.push(Arc::clone(kernel));
        }
    }
    Ok(held)
}
