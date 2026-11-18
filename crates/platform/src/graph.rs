//! CUDA graphs: stream capture, the template graph, executable graphs, and
//! their launch and update.
//!
//! Capture is the driver's stream capture: operations issued on the capturing
//! stream are recorded as nodes instead of executing, and the template graph
//! is returned when the capture ends. An executable graph is instantiated
//! from the template ([`Graph::instantiate`]), or an executable of the same
//! topology takes a new template's nodes and parameters in place
//! ([`GraphExec::update`]). The template stays alive as long as the executable
//! or any node handle does.
//!
//! A graph is never changed node by node: the work of other kernels is a new
//! capture of the same forward under them, which records whatever launches,
//! launch attributes and library kernels they issue.

use std::sync::Arc;

use crate::driver::{self, Handle};
use crate::sys::cuda as sys;
use crate::{DeviceGuard, DeviceId, Error, PrimaryContext, Result, Stream, reclaim};

/// What a template graph, its executable and its nodes share: the `CUgraph`,
/// the context it belongs to, and the device.
struct Core {
    raw: Handle,
    context: PrimaryContext,
    device: DeviceId,
}

// SAFETY: a graph handle is a driver object usable from any thread, and the
// retained context is thread-safe.
unsafe impl Send for Core {}
// SAFETY: as for Send above; shared references only pass the immutable handle
// to driver calls that are themselves thread-safe.
unsafe impl Sync for Core {}

impl std::fmt::Debug for Core {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Core")
            .field("raw", &(self.raw as usize))
            .field("device", &self.device)
            .finish()
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        let (raw, context) = (self.raw as usize, self.context.clone());
        // The executable graphs' own releases are queued before this one, so
        // the template is destroyed after every executable of it.
        reclaim::defer(move || {
            driver::in_context(context.raw(), |driver| {
                driver.check(
                    // SAFETY: the graph is owned and destroyed once.
                    unsafe { (driver.graph_destroy)(raw as Handle) },
                    "cuGraphDestroy",
                )
            })
        });
    }
}

/// A captured graph before instantiation (`CUgraph`).
///
/// It owns the graph: the last [`GraphExec`] or [`GraphNode`] taken from it
/// and the graph itself share its destruction, so node handles stay valid for
/// as long as anything refers to them. Dropping it destroys the graph once
/// nothing does.
#[derive(Debug)]
pub struct Graph {
    core: Arc<Core>,
}

// SAFETY: a graph is only shared driver state; every method passes immutable
// handles to thread-safe driver calls.
unsafe impl Send for Graph {}
// SAFETY: as for Send above.
unsafe impl Sync for Graph {}

impl Graph {
    /// Instantiates an executable graph from this template. The template has
    /// to stay alive for the executable's node parameters to be updatable,
    /// which the returned value guarantees by holding it.
    pub fn instantiate(&self) -> Result<GraphExec> {
        let driver = driver::driver()?;
        let _context = self.core.context.enter()?;
        let mut raw = std::ptr::null_mut();
        driver.check(
            // SAFETY: the template graph is live; the driver writes one
            // executable handle to the local `raw`.
            unsafe { (driver.graph_instantiate)(&mut raw, self.core.raw, 0) },
            "cuGraphInstantiateWithFlags",
        )?;
        Ok(GraphExec {
            core: Arc::clone(&self.core),
            raw,
        })
    }

    /// The number of nodes in the graph.
    pub fn node_count(&self) -> Result<usize> {
        let nodes = self.nodes()?;
        Ok(nodes.len())
    }

    /// Every node of the graph.
    ///
    /// **The order is the driver's, and `cuGraphGetNodes` does not specify
    /// one.** Do not map an operation onto a range of indices here: a graph
    /// captured from two launches may list them in either order, and the order
    /// can differ between the capture that made it and a later read.
    pub fn nodes(&self) -> Result<Vec<GraphNode>> {
        let handles = self.node_handles()?;
        Ok(handles
            .into_iter()
            .map(|raw| GraphNode {
                core: Arc::clone(&self.core),
                raw,
            })
            .collect())
    }

    /// The graph's node handles, in the driver's order.
    fn node_handles(&self) -> Result<Vec<Handle>> {
        let driver = driver::driver()?;
        let _context = self.core.context.enter()?;
        let mut count = 0usize;
        driver.check(
            // SAFETY: a null node array asks only for the count.
            unsafe { (driver.graph_get_nodes)(self.core.raw, std::ptr::null_mut(), &mut count) },
            "cuGraphGetNodes",
        )?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut nodes = vec![std::ptr::null_mut(); count];
        driver.check(
            // SAFETY: the array holds `count` handles, which the driver fills.
            unsafe { (driver.graph_get_nodes)(self.core.raw, nodes.as_mut_ptr(), &mut count) },
            "cuGraphGetNodes",
        )?;
        nodes.truncate(count);
        Ok(nodes)
    }
}

/// One node of a template graph (`CUgraphNode`).
///
/// The handle is valid while the graph is; the node keeps its graph alive.
/// Clone it to refer to the same node elsewhere.
#[derive(Clone, Debug)]
pub struct GraphNode {
    core: Arc<Core>,
    raw: Handle,
}

// SAFETY: the node is an immutable handle into a shared graph.
unsafe impl Send for GraphNode {}
// SAFETY: as for Send above.
unsafe impl Sync for GraphNode {}

impl GraphNode {
    /// What work the node does (`cuGraphNodeGetType`): the capture of an
    /// operation may add other nodes than kernels, such as a copy of a constant
    /// or the empty node that joins dependencies.
    pub fn kind(&self) -> Result<GraphNodeKind> {
        let driver = driver::driver()?;
        let _context = self.core.context.enter()?;
        let mut kind = 0;
        driver.check(
            // SAFETY: the node belongs to a graph that is live.
            unsafe { (driver.graph_node_get_type)(self.raw, &mut kind) },
            "cuGraphNodeGetType",
        )?;
        Ok(match kind {
            sys::CU_GRAPH_NODE_TYPE_KERNEL => GraphNodeKind::Kernel,
            sys::CU_GRAPH_NODE_TYPE_MEMCPY => GraphNodeKind::Memcpy,
            sys::CU_GRAPH_NODE_TYPE_MEMSET => GraphNodeKind::Memset,
            sys::CU_GRAPH_NODE_TYPE_HOST => GraphNodeKind::Host,
            sys::CU_GRAPH_NODE_TYPE_GRAPH => GraphNodeKind::Graph,
            sys::CU_GRAPH_NODE_TYPE_EMPTY => GraphNodeKind::Empty,
            other => GraphNodeKind::Other(other),
        })
    }
}

/// The kind of work a graph node does (`CUgraphNodeType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphNodeKind {
    /// A kernel launch.
    Kernel,
    /// A memory copy.
    Memcpy,
    /// A memory set.
    Memset,
    /// Host code.
    Host,
    /// A nested graph.
    Graph,
    /// A node that only orders its dependencies.
    Empty,
    /// A kind this binding does not name, as the driver's number.
    Other(u32),
}

/// A captured graph instantiated for replay (`CUgraphExec`).
///
/// Each launch is ordered after the earlier launches of the same executable,
/// and after the work already queued on the stream it is launched on.
#[derive(Debug)]
pub struct GraphExec {
    core: Arc<Core>,
    raw: Handle,
}

// SAFETY: the driver accepts `cuGraphLaunch` from any thread and orders each
// launch behind the earlier launches of the same executable; the handle is
// never dereferenced by this crate.
unsafe impl Send for GraphExec {}
// SAFETY: as for Send above; a shared reference only passes the immutable
// handle to thread-safe driver calls.
unsafe impl Sync for GraphExec {}

impl GraphExec {
    /// Submits one execution of the graph to `stream` and returns without
    /// waiting for it. The graph may have been captured on another stream of
    /// the same device.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        let driver = driver::driver()?;
        // The launch resolves the executable in the stream's context; the
        // calling thread may have none.
        let _context = stream.enter()?;
        driver.check(
            // SAFETY: the executable is live while this value is, and the
            // stream is a live handle.
            unsafe { (driver.graph_launch)(self.raw, stream.handle()) },
            "cuGraphLaunch",
        )
    }

    /// Writes `graph` into this executable in place (`cuGraphExecUpdate`): from the next launch
    /// on it runs `graph`'s nodes and parameters, without an instantiation. A launch already
    /// enqueued runs what it was enqueued with. The update holds only for a graph of this
    /// executable's topology, node kinds and kernel contexts; the driver refuses any other and
    /// the executable is left as it was.
    ///
    /// # Errors
    ///
    /// The driver's refusal, with the update's result code when it states one.
    pub fn update(&self, graph: &Graph) -> Result<()> {
        let driver = driver::driver()?;
        let _context = self.core.context.enter()?;
        let mut info = driver::ExecUpdateInfo {
            result: 0,
            error_node: std::ptr::null_mut(),
            error_from_node: std::ptr::null_mut(),
        };
        let checked = driver.check(
            // SAFETY: both handles are live while their values are; the driver writes the
            // result into the local `info`.
            unsafe { (driver.graph_exec_update)(self.raw, graph.core.raw, &mut info) },
            "cuGraphExecUpdate_v2",
        );
        if checked.is_err() && info.result != 0 {
            return Err(Error::InvalidArgument(format!(
                "cuGraphExecUpdate refused the graph with update result {}",
                info.result
            )));
        }
        checked
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        let (core, raw) = (Arc::clone(&self.core), self.raw as usize);
        // Queued before the template's own release, which the `core` clone
        // keeps from happening first.
        reclaim::defer(move || {
            driver::in_context(core.context.raw(), |driver| {
                driver.check(
                    // SAFETY: the executable is owned and destroyed once.
                    unsafe { (driver.graph_exec_destroy)(raw as Handle) },
                    "cuGraphExecDestroy",
                )
            })
        });
    }
}

/// The number of captures open in this process, as of the last begin or end.
///
/// A hang inside a capture and a hang inside a launch look identical from outside and are different
/// defects, and the thread that hangs cannot say which: the counter is written where a capture opens
/// and closes and read by whoever reports the stall (2.4).
static OPEN_CAPTURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The captures open in this process on any stream.
pub fn open_captures() -> usize {
    OPEN_CAPTURES.load(std::sync::atomic::Ordering::Relaxed)
}

/// A capture in progress on one stream.
///
/// It begins the driver's stream capture and keeps the device's context
/// current on this thread until it ends, so it is not `Send`: a capture is
/// begun, recorded and ended on one thread, as the worker that runs it is
/// one thread per device.
pub struct Capture {
    stream: Stream,
    core: Arc<Core>,
    ended: bool,
    // Keeps the context current on this thread for the capture's lifetime,
    // and makes the capture thread-bound.
    _context: DeviceGuard,
}

/// `CUstreamCaptureMode`: how a capture interacts with CUDA calls from other
/// threads of the process, such as the other devices' workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureMode {
    /// `CU_STREAM_CAPTURE_MODE_GLOBAL`: an unsafe call from any thread
    /// invalidates the capture. The strictest mode, which catches a thread
    /// that issues CUDA work while this one captures.
    Global,
    /// `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`: only the capturing thread's
    /// unsafe calls invalidate the capture.
    ThreadLocal,
    /// `CU_STREAM_CAPTURE_MODE_RELAXED`: other threads' calls are neither
    /// captured nor invalidating. The worker captures in this mode: its
    /// device's stream is its own, and the process's other workers keep
    /// working meanwhile.
    Relaxed,
}

impl CaptureMode {
    fn raw(self) -> sys::CUstreamCaptureMode {
        match self {
            Self::Global => sys::CU_STREAM_CAPTURE_MODE_GLOBAL,
            Self::ThreadLocal => sys::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            Self::Relaxed => sys::CU_STREAM_CAPTURE_MODE_RELAXED,
        }
    }
}

impl Capture {
    /// Begins capturing `stream` in `mode`.
    pub fn begin(stream: &Stream, mode: CaptureMode) -> Result<Self> {
        let context = PrimaryContext::retain(stream.device())?;
        let guard = context.enter()?;
        let driver = driver::driver()?;
        driver.check(
            // SAFETY: the stream is live and not capturing, by this type's
            // invariant.
            unsafe { (driver.stream_begin_capture)(stream.handle(), mode.raw()) },
            "cuStreamBeginCapture",
        )?;

        let (status, graph) = capture_info(stream)?;
        if status != sys::CU_STREAM_CAPTURE_STATUS_ACTIVE {
            return Err(Error::invalid(format!(
                "the stream's capture is in status {status} right after it began"
            )));
        }
        OPEN_CAPTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Self {
            stream: stream.clone(),
            core: Arc::new(Core {
                raw: graph,
                context,
                device: stream.device(),
            }),
            ended: false,
            _context: guard,
        })
    }

    /// Ends the capture and returns the template graph. `stream` is usable
    /// again afterwards.
    pub fn end(mut self) -> Result<Graph> {
        let driver = driver::driver()?;
        let mut raw = std::ptr::null_mut();
        driver.check(
            // SAFETY: the capture is active on this stream, on this thread.
            unsafe { (driver.stream_end_capture)(self.stream.handle(), &mut raw) },
            "cuStreamEndCapture",
        )?;
        self.ended = true;
        OPEN_CAPTURES.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // The driver promises the same graph at the end of the capture; a
        // different handle would leave the template's node handles dangling.
        debug_assert_eq!(raw, self.core.raw);
        Ok(Graph {
            core: Arc::clone(&self.core),
        })
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        OPEN_CAPTURES.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        // Ends the capture so the stream can be used again. The captured graph
        // is discarded: its owner drops it along with the capture.
        if let Ok(driver) = driver::driver() {
            let mut raw = std::ptr::null_mut();
            // SAFETY: the capture is active on this stream, on this thread;
            // the returned graph is the one this capture owns, which is
            // destroyed when the capture's values drop.
            let _ = unsafe { (driver.stream_end_capture)(self.stream.handle(), &mut raw) };
        }
    }
}

/// The stream's capture status and the graph being captured
/// (`cuStreamGetCaptureInfo_v3`).
///
/// The dependency and edge-data arrays are asked for, though unread: the
/// driver refuses a query that asks for a capture's dependencies without their
/// edge data when one of them carries non-default data
/// (`CUDA_ERROR_LOSSY_QUERY`), as a programmatic dependent launch's does.
fn capture_info(stream: &Stream) -> Result<(sys::CUstreamCaptureStatus, Handle)> {
    let driver = driver::driver()?;
    let mut status = 0;
    let mut id = 0u64;
    let mut graph: Handle = std::ptr::null_mut();
    let mut dependencies: *const Handle = std::ptr::null();
    let mut edge_data: *const sys::CUgraphEdgeData = std::ptr::null();
    let mut count = 0usize;
    driver.check(
        // SAFETY: the stream is live; each pointer addresses a local the
        // driver writes, and the dependency array is copied below while it is
        // valid.
        unsafe {
            (driver.stream_get_capture_info)(
                stream.handle(),
                &mut status,
                &mut id,
                &mut graph,
                &mut dependencies,
                &mut edge_data,
                &mut count,
            )
        },
        "cuStreamGetCaptureInfo_v3",
    )?;
    Ok((status, graph))
}
