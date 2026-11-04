//! CUDA graphs: stream capture, the template graph, executable graphs,
//! kernel-node rebinding and launch.
//!
//! Capture is the driver's stream capture: operations issued on the capturing
//! stream are recorded as nodes instead of executing, and the template graph
//! is returned when the capture ends. An executable graph is instantiated
//! from the template; the template stays alive as long as the executable or
//! any node handle does, because `cuGraphExecKernelNodeSetParams` addresses a
//! node of the graph the executable was instantiated from.
//!
//! Two kernel updates exist (design 4.5):
//!
//! - rebind, [`GraphExec::set_kernel_node_params`], changes a kernel node's
//!   function, grid, block, dynamic shared memory and arguments in place;
//! - recapture, a new [`Capture`] of the same work, is what a program whose
//!   launch *attributes* (cluster, programmatic dependent launch, cooperative
//!   launch) change needs, because those are fixed in a graph node when it is
//!   captured and no driver call updates them afterwards.

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Arc;

use crate::driver::{self, Handle, KernelNodeParams as RawKernelNodeParams};
use crate::launch::Function;
use crate::{DeviceGuard, DeviceId, Error, PrimaryContext, Result, Stream, reclaim};

/// `CU_STREAM_CAPTURE_STATUS_ACTIVE`.
const CU_STREAM_CAPTURE_STATUS_ACTIVE: i32 = 1;

/// `CU_GRAPH_NODE_TYPE_KERNEL`.
const CU_GRAPH_NODE_TYPE_KERNEL: i32 = 0;

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

/// The template graph's root nodes: those with no dependency.
fn root_handles(core: &Core) -> Result<Vec<Handle>> {
    let driver = driver::driver()?;
    let _context = core.context.enter()?;
    let mut count = 0usize;
    driver.check(
        // SAFETY: a null array asks only for the count.
        unsafe { (driver.graph_get_root_nodes)(core.raw, std::ptr::null_mut(), &mut count) },
        "cuGraphGetRootNodes",
    )?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut nodes = vec![std::ptr::null_mut(); count];
    driver.check(
        // SAFETY: the array holds `count` handles, which the driver fills.
        unsafe { (driver.graph_get_root_nodes)(core.raw, nodes.as_mut_ptr(), &mut count) },
        "cuGraphGetRootNodes",
    )?;
    nodes.truncate(count);
    Ok(nodes)
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

/// The parameters a rebind may change in one kernel node.
///
/// The launch attributes — the cluster, programmatic dependent launch and
/// cooperative launch — are not part of a node's parameters: they are fixed
/// when the node is captured. A program that changes any of them is
/// recaptured rather than rebound, which the runner decides by comparing the
/// two programs' metadata.
pub struct KernelNodeParams<'a> {
    /// The kernel to run.
    pub function: &'a Function,
    /// Blocks per grid dimension.
    pub grid: [u32; 3],
    /// Threads per block dimension.
    pub block: [u32; 3],
    /// Dynamic shared memory per block, in bytes.
    pub shared_memory: u32,
    /// The address of each kernel parameter's value, in declaration order.
    pub parameters: &'a [*mut c_void],
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
    /// The `CUgraphExec` handle.
    pub fn raw(&self) -> usize {
        self.raw as usize
    }

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

    /// Replaces the work of `node` in this executable with `params`: the
    /// function, the grid and block dimensions, the dynamic shared-memory
    /// size, and the argument values. It affects only future launches; one
    /// already enqueued or running is unaffected, and the template node is not
    /// modified.
    ///
    /// `node` must be a kernel node of the graph this executable was
    /// instantiated from, and `params.function` must belong to the same
    /// context; the node's launch attributes stay as they were captured.
    pub fn set_kernel_node_params(
        &self,
        node: GraphNode,
        params: &KernelNodeParams<'_>,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.core, &node.core) {
            return Err(Error::invalid(
                "the node does not belong to the graph this executable was instantiated from",
            ));
        }
        if params.function.device().id() != self.core.device {
            return Err(Error::invalid(
                "the node's function belongs to another device",
            ));
        }

        let driver = driver::driver()?;
        let _context = self.core.context.enter()?;
        let mut kind = 0;
        driver.check(
            // SAFETY: the node belongs to a graph that is live.
            unsafe { (driver.graph_node_get_type)(node.raw, &mut kind) },
            "cuGraphNodeGetType",
        )?;
        if kind != CU_GRAPH_NODE_TYPE_KERNEL {
            return Err(Error::invalid(format!(
                "the node is of type {kind}, not a kernel node"
            )));
        }

        let raw = RawKernelNodeParams {
            func: params.function.raw(),
            grid_dim: params.grid,
            block_dim: params.block,
            shared_mem_bytes: params.shared_memory,
            kernel_params: params.parameters.as_ptr().cast_mut(),
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
        };
        driver.check(
            // SAFETY: the executable and node are live and belong together;
            // the parameter array outlives the call, and the driver copies
            // the values it points at.
            unsafe { (driver.graph_exec_kernel_node_set_params)(self.raw, node.raw, &raw) },
            "cuGraphExecKernelNodeSetParams",
        )
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

/// A capture in progress on one stream.
///
/// It begins the driver's stream capture and keeps the device's context
/// current on this thread until it ends, so it is not `Send`: a capture is
/// begun, recorded and ended on one thread, as the executor that runs it is
/// one thread per device.
///
/// [`Capture::record`] is called after each operation the owner issues: it
/// returns the nodes the operation added, in the order they can execute, which
/// is what the owner records to rebind a graph later.
pub struct Capture {
    stream: Stream,
    core: Arc<Core>,
    /// The dependency set of the next node: the nodes added by the previous
    /// operation end here, and the next operation's nodes are reachable from
    /// them.
    frontier: Vec<Handle>,
    /// Every node recorded so far, so an operation's nodes are exactly the new
    /// ones.
    seen: HashSet<usize>,
    ended: bool,
    // Keeps the context current on this thread for the capture's lifetime,
    // and makes the capture thread-bound.
    _context: DeviceGuard,
}

/// `CUstreamCaptureMode`: how a capture interacts with CUDA calls from other
/// threads of the process, such as the other devices' executors.
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
    /// captured nor invalidating. The executor captures in this mode: its
    /// device's stream is its own, and the process's other executors keep
    /// working meanwhile.
    Relaxed,
}

impl CaptureMode {
    fn raw(self) -> i32 {
        match self {
            Self::Global => 0,
            Self::ThreadLocal => 1,
            Self::Relaxed => 2,
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
            // contract.
            unsafe { (driver.stream_begin_capture)(stream.handle(), mode.raw()) },
            "cuStreamBeginCapture",
        )?;

        let (status, graph, frontier) = capture_info(stream)?;
        if status != CU_STREAM_CAPTURE_STATUS_ACTIVE {
            return Err(Error::invalid(format!(
                "the stream's capture is in status {status} right after it began"
            )));
        }
        Ok(Self {
            stream: stream.clone(),
            core: Arc::new(Core {
                raw: graph,
                context,
                device: stream.device(),
            }),
            frontier,
            seen: HashSet::new(),
            ended: false,
            _context: guard,
        })
    }

    /// The nodes the work issued since the previous call added, in an order
    /// that respects their dependencies; before the first operation, none.
    ///
    /// Call it after each operation the owner issues on the capturing stream:
    /// the nodes are that operation's, and the owner records which operation
    /// they belong to for a later rebind.
    pub fn record(&mut self) -> Result<Vec<GraphNode>> {
        let (status, _, frontier) = capture_info(&self.stream)?;
        if status != CU_STREAM_CAPTURE_STATUS_ACTIVE {
            return Err(Error::invalid(format!(
                "the capture is in status {status}; no node can be recorded"
            )));
        }

        let mut found: Vec<Handle> = Vec::new();
        let mut queue: std::collections::VecDeque<Handle> = self.frontier.iter().copied().collect();
        // The next node depends on the frontier. When the dependency set is
        // empty (the first operation, or an operation that clears it), the
        // new nodes are the graph's root nodes instead.
        for root in root_handles(&self.core)? {
            if self.seen.insert(root as usize) {
                found.push(root);
                queue.push_back(root);
            }
        }
        while let Some(node) = queue.pop_front() {
            for dependent in dependent_handles(node)? {
                if self.seen.insert(dependent as usize) {
                    found.push(dependent);
                    queue.push_back(dependent);
                }
            }
        }
        self.frontier = frontier;

        Ok(found
            .into_iter()
            .map(|raw| GraphNode {
                core: Arc::clone(&self.core),
                raw,
            })
            .collect())
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
        // The driver promises the same graph at the end of the capture; a
        // different handle would leave the node handles of `record` dangling.
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

/// The stream's capture status, the graph being captured, and the dependency
/// set of the next node (`cuStreamGetCaptureInfo`).
fn capture_info(stream: &Stream) -> Result<(i32, Handle, Vec<Handle>)> {
    let driver = driver::driver()?;
    let mut status = 0;
    let mut id = 0u64;
    let mut graph: Handle = std::ptr::null_mut();
    let mut dependencies: *const Handle = std::ptr::null();
    let mut edge_data: *const c_void = std::ptr::null();
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
    let frontier = if dependencies.is_null() || count == 0 {
        Vec::new()
    } else {
        // SAFETY: the driver returns `count` node handles, valid until the
        // next call that operates on the stream.
        unsafe { std::slice::from_raw_parts(dependencies, count) }.to_vec()
    };
    Ok((status, graph, frontier))
}

/// The nodes that depend directly on `node` (`cuGraphNodeGetDependentNodes`).
fn dependent_handles(node: Handle) -> Result<Vec<Handle>> {
    let driver = driver::driver()?;
    let mut count = 0usize;
    driver.check(
        // SAFETY: the node is live with its graph; a null array asks only for
        // the count.
        unsafe {
            (driver.graph_node_get_dependent_nodes)(
                node,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut count,
            )
        },
        "cuGraphNodeGetDependentNodes",
    )?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut nodes = vec![std::ptr::null_mut(); count];
    driver.check(
        // SAFETY: the array holds `count` handles, which the driver fills.
        unsafe {
            (driver.graph_node_get_dependent_nodes)(
                node,
                nodes.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut count,
            )
        },
        "cuGraphNodeGetDependentNodes",
    )?;
    nodes.truncate(count);
    Ok(nodes)
}
