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

use crate::driver::{
    self, CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC, GraphEdgeData, Handle,
    KernelNodeParams as RawKernelNodeParams,
};
use crate::launch::Function;
use crate::{Device, DeviceGuard, DeviceId, Error, PrimaryContext, Result, Stream, reclaim};

/// `CU_STREAM_CAPTURE_STATUS_ACTIVE`.
const CU_STREAM_CAPTURE_STATUS_ACTIVE: i32 = 1;

/// `CU_GRAPH_NODE_TYPE_KERNEL`.
const CU_GRAPH_NODE_TYPE_KERNEL: i32 = 0;
/// `CU_GRAPH_NODE_TYPE_MEMCPY`.
const CU_GRAPH_NODE_TYPE_MEMCPY: i32 = 1;
/// `CU_GRAPH_NODE_TYPE_MEMSET`.
const CU_GRAPH_NODE_TYPE_MEMSET: i32 = 2;
/// `CU_GRAPH_NODE_TYPE_HOST`.
const CU_GRAPH_NODE_TYPE_HOST: i32 = 3;
/// `CU_GRAPH_NODE_TYPE_GRAPH`.
const CU_GRAPH_NODE_TYPE_GRAPH: i32 = 4;
/// `CU_GRAPH_NODE_TYPE_EMPTY`.
const CU_GRAPH_NODE_TYPE_EMPTY: i32 = 5;

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
    ///
    /// An owner that needs the nodes *one operation* added, in the order they
    /// can execute, calls [`Capture::record`] after issuing that operation:
    /// the capture knows which nodes are new, and returns them in order. That
    /// is the record a rebind is issued against.
    /// Every copy the capture recorded, with the operands the driver states for it.
    ///
    /// A copy node names **where it reads and writes**, and a capture bakes both addresses. A
    /// recorded copy from a host buffer that the caller fills per batch is therefore a copy of the
    /// values that buffer held when the recording happened, replayed at every launch: the batch's
    /// own values never reach the device, and nothing about the graph's kernels or parameters says
    /// so.
    pub fn copies(&self) -> Result<Vec<Copy>> {
        let _context = self.core.context.enter()?;
        let driver = driver::driver()?;
        let mut copies = Vec::new();
        for node in self.nodes()? {
            if node.kind()? != GraphNodeKind::Memcpy {
                continue;
            }
            let get_params = match driver.graph_memcpy_node_get_params {
                Ok(get_params) => get_params,
                Err(_) => return Ok(copies),
            };
            let mut params = driver::Memcpy3D::default();
            driver.check(
                // SAFETY: the node belongs to this live graph, and the driver fills the struct.
                unsafe { get_params(node.raw, &mut params) },
                "cuGraphMemcpyNodeGetParams",
            )?;
            copies.push(Copy {
                source: Space::from(
                    params.src_memory_type,
                    params.src_host as u64,
                    params.src_device,
                ),
                destination: Space::from(
                    params.dst_memory_type,
                    params.dst_host as u64,
                    params.dst_device,
                ),
                bytes: params.width * params.height.max(1) * params.depth.max(1),
            });
        }
        Ok(copies)
    }

    /// The nodes that depend directly on `node`, in the driver's order
    /// (`cuGraphNodeGetDependentNodes`).
    ///
    /// A captured graph holds only the edges the recording saw, and a launch that another stream
    /// issued leaves none: within a graph the work is ordered by its edges, so an edge nobody
    /// recorded is work that runs in whatever order the driver chooses. This is the only reading
    /// that answers "does the node after this one actually wait for it".
    pub fn dependents(&self, node: &GraphNode) -> Result<Vec<GraphNode>> {
        Ok(self
            .dependents_with_edges(node)?
            .into_iter()
            .map(|(dependent, _)| dependent)
            .collect())
    }

    /// The nodes that depend directly on `node`, each with what its edge says.
    ///
    /// An edge is normally "after the producer completed", and a **programmatic** edge says the
    /// consumer may start early, from a point inside the producer - which only holds if the consumer
    /// waits for the producer's data itself. A graph that recorded such an edge for a launch that
    /// did not ask for one is a consumer reading its input early, and the scalars of both launches
    /// still agree.
    pub fn dependents_with_edges(&self, node: &GraphNode) -> Result<Vec<(GraphNode, Edge)>> {
        if !Arc::ptr_eq(&self.core, &node.core) {
            return Err(Error::invalid("the node belongs to another graph"));
        }
        let _context = self.core.context.enter()?;
        Ok(dependent_edges(node.raw)?
            .into_iter()
            .map(|(raw, data)| {
                (
                    GraphNode {
                        core: Arc::clone(&self.core),
                        raw,
                    },
                    Edge {
                        port: data.from_port,
                        kind: data.kind,
                    },
                )
            })
            .collect())
    }

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

    /// Reads back the parameters of kernel node `node`: the kernel it
    /// captured, its grid, block, dynamic shared memory and the values of its
    /// arguments.
    ///
    /// This is the read half of a rebind walk (design 4.5): a capture with the
    /// new kernels on a scratch stream gives a throwaway template, each of its
    /// changed launches is read with this, and the values are set on the live
    /// executable. `node` must be a kernel node of this graph, and the graph
    /// may be destroyed once the returned values have been read.
    pub fn kernel_node_params(&self, node: &GraphNode) -> Result<OwnedKernelNodeParams> {
        if !Arc::ptr_eq(&self.core, &node.core) {
            return Err(Error::invalid("the node belongs to another graph"));
        }
        let kind = node.kind()?;
        if kind != GraphNodeKind::Kernel {
            return Err(Error::invalid(format!(
                "the node is {kind:?}, not a kernel node"
            )));
        }

        let driver = driver::driver()?;
        let _context = self.core.context.enter()?;
        let mut raw = RawKernelNodeParams {
            func: std::ptr::null_mut(),
            grid_dim: [0; 3],
            block_dim: [0; 3],
            shared_mem_bytes: 0,
            kernel_params: std::ptr::null_mut(),
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
        };
        driver.check(
            // SAFETY: the node is a live kernel node of this graph; the driver
            // fills `raw` with the node's own parameter storage.
            unsafe { (driver.graph_kernel_node_get_params)(node.raw, &mut raw) },
            "cuGraphKernelNodeGetParams",
        )?;

        // Each parameter's size comes from the kernel's signature: the node's
        // array carries no count, so the walk ends where the driver does.
        let mut sizes = Vec::new();
        while let Some(parameter) = crate::launch::param_at(raw.func, sizes.len())? {
            sizes.push((parameter.offset(), parameter.size()));
        }

        let mut parameters = Vec::with_capacity(sizes.len());
        for (index, (offset, size)) in sizes.into_iter().enumerate() {
            // SAFETY: the node's array has one entry per parameter of the
            // signature, and its packed form lays the values out at the
            // offsets just read; each value is `size` bytes there.
            let source: *const c_void = unsafe {
                if !raw.kernel_params.is_null() {
                    *raw.kernel_params.add(index)
                } else {
                    (raw.extra as *const u8).add(offset).cast()
                }
            };
            let mut value = vec![0u8; size];
            if size > 0 {
                // SAFETY: `source` addresses `size` readable bytes, as the
                // signature says, and `value` holds exactly that many.
                unsafe {
                    std::ptr::copy_nonoverlapping(source.cast::<u8>(), value.as_mut_ptr(), size)
                };
            }
            parameters.push(value);
        }

        Ok(OwnedKernelNodeParams {
            function: Function::new(raw.func, Device::new(self.core.device)?),
            grid: raw.grid_dim,
            block: raw.block_dim,
            shared_memory: raw.shared_mem_bytes,
            parameters,
        })
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

impl GraphNode {
    /// What work the node does (`cuGraphNodeGetType`).
    ///
    /// A rebind touches only [`GraphNodeKind::Kernel`] nodes; the capture of
    /// an operation may add others (a copy of a constant, or the empty node
    /// that joins dependencies), and setting kernel parameters on one of them
    /// is the driver's error.
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
            CU_GRAPH_NODE_TYPE_KERNEL => GraphNodeKind::Kernel,
            CU_GRAPH_NODE_TYPE_MEMCPY => GraphNodeKind::Memcpy,
            CU_GRAPH_NODE_TYPE_MEMSET => GraphNodeKind::Memset,
            CU_GRAPH_NODE_TYPE_HOST => GraphNodeKind::Host,
            CU_GRAPH_NODE_TYPE_GRAPH => GraphNodeKind::Graph,
            CU_GRAPH_NODE_TYPE_EMPTY => GraphNodeKind::Empty,
            other => GraphNodeKind::Other(other),
        })
    }
}

/// Where one recorded copy reads from or writes to, as the driver states it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// Host memory, at the address the driver gave.
    Host(u64),
    /// Device memory, at the address the driver gave.
    Device(u64),
    /// An array object, which a capture of this worker's work has not produced.
    Array,
    /// A spelling this binding does not name, as the driver's number.
    Other(i32, u64),
}

impl Space {
    /// The address, for the two spellings that name one.
    pub const fn address(&self) -> Option<u64> {
        match self {
            Self::Host(address) | Self::Device(address) => Some(*address),
            Self::Array | Self::Other(..) => None,
        }
    }

    fn from(memory_type: i32, host: u64, device: u64) -> Self {
        // `CU_MEMORYTYPE_*`: 1 is host, 2 is device, 3 is an array, 4 is a unified address.
        match memory_type {
            1 => Self::Host(host),
            2 => Self::Device(device),
            4 => Self::Device(device),
            3 => Self::Array,
            other => Self::Other(other, device),
        }
    }
}

/// One copy a capture recorded: what it reads, what it writes, and how many bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Copy {
    pub source: Space,
    pub destination: Space,
    pub bytes: usize,
}

/// What one recorded edge says about when its consumer may start (`CUgraphEdgeData`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Edge {
    /// When, inside the producer, the dependency triggers.
    pub port: u8,
    /// `CUgraphDependencyType`: zero is "after the producer completed", one is programmatic.
    pub kind: u8,
}

impl Edge {
    /// Whether the consumer may start before its producer completed (a programmatic dependent
    /// launch), which is only correct if the consumer synchronises with the producer itself.
    pub const fn is_programmatic(&self) -> bool {
        self.kind == 1
    }
}

/// The kind of work a graph node does (`CUgraphNodeType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphNodeKind {
    /// A kernel launch, the only kind a rebind changes.
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
    Other(i32),
}

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

/// A kernel node's parameters read back from a template graph, owned by the
/// caller.
///
/// [`Graph::kernel_node_params`] fills one from a node of a throwaway capture;
/// a rebind walk then sets the changed ones on the live executable with
/// [`GraphExec::set_kernel_node_params_owned`], which replaces the function of
/// the kernel that answers for the node.
///
/// The values are copied out of the node — the driver keeps each parameter's
/// size in the kernel's signature — so nothing dangles when the template is
/// destroyed.
#[derive(Clone)]
pub struct OwnedKernelNodeParams {
    /// The kernel the node captured. Valid while the library it was loaded
    /// from is loaded, the same obligation a captured graph has.
    pub function: Function,
    /// Blocks per grid dimension.
    pub grid: [u32; 3],
    /// Threads per block dimension.
    pub block: [u32; 3],
    /// Dynamic shared memory per block, in bytes.
    pub shared_memory: u32,
    /// Each parameter's value, copied, in declaration order.
    pub parameters: Vec<Vec<u8>>,
}

impl std::fmt::Debug for OwnedKernelNodeParams {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedKernelNodeParams")
            .field("function", &(self.function.raw() as usize))
            .field("grid", &self.grid)
            .field("block", &self.block)
            .field("shared_memory", &self.shared_memory)
            .field("parameters", &self.parameters)
            .finish()
    }
}

impl OwnedKernelNodeParams {
    /// Pointers to the copied values, as [`KernelNodeParams`] takes them. The
    /// pointers are valid while this value is alive and not mutated.
    pub fn pointers(&self) -> Vec<*mut c_void> {
        self.parameters
            .iter()
            .map(|value| value.as_ptr().cast_mut().cast())
            .collect()
    }

    /// The same parameters for another kernel: what a rebind sets, which
    /// changes the function and keeps every argument the node captured.
    pub fn with_function(mut self, function: Function) -> Self {
        self.function = function;
        self
    }
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
    /// Sets parameters read back from a template's node on this executable:
    /// the values of [`Graph::kernel_node_params`], with the function a rebind
    /// chooses ([`OwnedKernelNodeParams::with_function`]).
    pub fn set_kernel_node_params_owned(
        &self,
        node: GraphNode,
        params: &OwnedKernelNodeParams,
    ) -> Result<()> {
        let pointers = params.pointers();
        self.set_kernel_node_params(
            node,
            &KernelNodeParams {
                function: &params.function,
                grid: params.grid,
                block: params.block,
                shared_memory: params.shared_memory,
                parameters: &pointers,
            },
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
        let frontier: Vec<Handle> = frontier.into_iter().map(|(node, _)| node).collect();
        if status != CU_STREAM_CAPTURE_STATUS_ACTIVE {
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
        Ok(self
            .record_walk()?
            .into_iter()
            .map(|walked| walked.node)
            .collect())
    }

    /// [`Self::record`], with the kind of every dependency that reaches each node.
    ///
    /// This is the walk a structure comparison reads: two captures of the same work
    /// are compared per operation and per position, so the kinds belong to the
    /// positions rather than to the handles. A rebind sets node parameters and keeps
    /// the template's edges, so a launch whose edge became programmatic is a
    /// structure change and its bucket is recaptured.
    pub fn record_walk(&mut self) -> Result<Vec<WalkedNode>> {
        let (status, _, frontier) = capture_info(&self.stream)?;
        if status != CU_STREAM_CAPTURE_STATUS_ACTIVE {
            return Err(Error::invalid(format!(
                "the capture is in status {status}; no node can be recorded"
            )));
        }

        let mut found: Vec<Handle> = Vec::new();
        let mut fresh: HashSet<usize> = HashSet::new();
        let mut kinds: std::collections::HashMap<usize, Vec<EdgeKind>> =
            std::collections::HashMap::new();
        // The frontier's own edge data is read and dropped: the driver refuses a
        // frontier query that asks for its nodes without it, and what it describes is
        // the edge the *next* operation will add - which the walk reads from that
        // edge itself once it exists. Attributing it here would state a kind for an
        // edge this operation did not create.
        for (_, data) in &frontier {
            EdgeKind::of(data)?;
        }
        // The frontier names the nodes the *next* operation will depend on, so after an operation
        // has been issued they are the nodes that operation added - the very nodes this walk is
        // asked for. Collecting only the frontier's dependents finds nothing for a launch, because
        // the node the launch just added has no dependents yet: the first walk of a capture found
        // its node only because that node was also a root, and every launch after it was invisible.
        for (node, _) in &frontier {
            if self.seen.insert(*node as usize) {
                found.push(*node);
                fresh.insert(*node as usize);
            }
        }
        // The walk starts at the *previous* frontier as well as this one. An operation may add more
        // than one node - a native module that launches two kernels, a reduction beside its main
        // pass - and only the last of them is the frontier: the earlier ones are reached by walking
        // downstream from where the operation began, and `seen` keeps a node from being collected
        // twice.
        let mut queue: std::collections::VecDeque<Handle> = self
            .frontier
            .iter()
            .copied()
            .chain(frontier.iter().map(|(node, _)| *node))
            .collect();
        // The next node depends on the frontier. When the dependency set is
        // empty (the first operation, or an operation that clears it), the
        // new nodes are the graph's root nodes instead.
        for root in root_handles(&self.core)? {
            if self.seen.insert(root as usize) {
                found.push(root);
                fresh.insert(root as usize);
                queue.push_back(root);
            }
        }
        while let Some(node) = queue.pop_front() {
            // Both edge kinds are dependencies: a programmatic edge lets the
            // downstream node start early, not become unrelated to it.
            for (dependent, data) in dependent_edges(node)? {
                if fresh.contains(&(dependent as usize)) {
                    // Every edge between the nodes this operation added is stated,
                    // in the order the walk reaches their upstream ends, so the list
                    // is the same for two captures of the same work.
                    let entry = kinds.entry(dependent as usize).or_default();
                    entry.push(EdgeKind::of(&data)?);
                }
                if self.seen.insert(dependent as usize) {
                    found.push(dependent);
                    fresh.insert(dependent as usize);
                    queue.push_back(dependent);
                }
            }
        }
        self.frontier = frontier.into_iter().map(|(node, _)| node).collect();

        Ok(found
            .into_iter()
            .map(|raw| WalkedNode {
                node: GraphNode {
                    core: Arc::clone(&self.core),
                    raw,
                },
                incoming: kinds.remove(&(raw as usize)).unwrap_or_default(),
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
        OPEN_CAPTURES.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
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

/// The nodes the next operation will depend on, each with what its edge says.
type Frontier = Vec<(Handle, GraphEdgeData)>;

/// The stream's capture status, the graph being captured, and the dependency
/// set of the next node, with what each of those edges says
/// (`cuStreamGetCaptureInfo_v3`).
///
/// **The edge-data array is read for the same reason the walk reads it:** the
/// driver refuses a query that asks for a capture's dependencies without their
/// edge data when one of them carries non-default data
/// (`CUDA_ERROR_LOSSY_QUERY`), and the frontier edge of a programmatic
/// dependent launch is exactly such an edge.
fn capture_info(stream: &Stream) -> Result<(i32, Handle, Frontier)> {
    let driver = driver::driver()?;
    let mut status = 0;
    let mut id = 0u64;
    let mut graph: Handle = std::ptr::null_mut();
    let mut dependencies: *const Handle = std::ptr::null();
    let mut edge_data: *const GraphEdgeData = std::ptr::null();
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
        // SAFETY: the driver returns `count` node handles, and `count` edge
        // data entries when it was asked for them, valid until the next call
        // that operates on the stream.
        let nodes = unsafe { std::slice::from_raw_parts(dependencies, count) };
        let data = if edge_data.is_null() {
            Vec::new()
        } else {
            // SAFETY: as above; the array is `count` entries and was returned
            // with the node array.
            unsafe { std::slice::from_raw_parts(edge_data, count) }.to_vec()
        };
        nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (*node, data.get(index).copied().unwrap_or_default()))
            .collect()
    };
    Ok((status, graph, frontier))
}

/// The nodes that depend directly on `node`, with what each edge says
/// (`cuGraphNodeGetDependentNodes`).
///
/// # Why both arrays are always passed
///
/// The driver refuses a query that asks for dependent nodes without their edge
/// data as soon as one edge carries non-default data: `CUDA_ERROR_LOSSY_QUERY`
/// (`cuda.h`, `cuGraphNodeGetDependentNodes`: "if an edge has non-zero
/// (non-default) edge data and `edgeData` is NULL, this API will return
/// CUDA_ERROR_LOSSY_QUERY"). A programmatic dependent launch is exactly such an
/// edge, so a captured kernel that triggers its successor early made this walk
/// fail on a null array - and `edgeData` cannot be passed alone either, because
/// the driver requires `dependentNodes` with it.
///
/// The array sizes are therefore guessed and grown rather than counted first:
/// there is no way to ask for the count of a graph whose edges carry data. The
/// driver fills what fits and answers how many it wrote, so one call with room
/// for eight is most nodes and the loop is bounded by the real out-degree.
fn dependent_edges(node: Handle) -> Result<Vec<(Handle, GraphEdgeData)>> {
    let driver = driver::driver()?;
    let mut capacity = 8usize;
    loop {
        let mut nodes = vec![std::ptr::null_mut(); capacity];
        let mut data = vec![GraphEdgeData::default(); capacity];
        let mut count = capacity;
        driver.check(
            // SAFETY: both arrays hold `capacity` entries and the driver writes
            // at most that many, answering the number it wrote.
            unsafe {
                (driver.graph_node_get_dependent_nodes)(
                    node,
                    nodes.as_mut_ptr(),
                    data.as_mut_ptr(),
                    &mut count,
                )
            },
            "cuGraphNodeGetDependentNodes",
        )?;
        if count <= capacity {
            nodes.truncate(count);
            data.truncate(count);
            return Ok(nodes.into_iter().zip(data).collect());
        }
        // The driver filled the array whole: more dependents than it held.
        capacity = count;
    }
}

/// What one edge of a captured graph means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// The downstream node starts when the upstream node has finished
    /// executing, with the upstream node's writes visible to it.
    Completion,
    /// **A programmatic dependent launch.** The downstream node may start
    /// before the upstream node finishes: it is launched at the upstream node's
    /// programmatic trigger point and synchronizes itself with
    /// `cudaGridDependencySynchronize`.
    ///
    /// It is still a dependency - the walk follows it, and the node it reaches
    /// belongs to the operation that issued it - but it orders **less** than a
    /// completion edge: the upstream node has not finished when the downstream
    /// one begins, so nothing may treat the two as interchangeable. Anything
    /// that reorders, rebinds or compares captured graphs reads the kind, and a
    /// comparison of two graph structures that ignores it compares two different
    /// orders as one.
    Programmatic,
}

impl EdgeKind {
    /// The kind `data` states, or an error for a value this walk does not know.
    ///
    /// A kind it does not know is refused rather than read as a completion
    /// edge: the driver may add kinds, and a walk that guessed would put a
    /// weaker order where a stronger one was meant.
    ///
    /// The driver states a programmatic edge in either of two ways - the
    /// dependency kind itself, or the default kind with the upstream node's
    /// programmatic port - so both are read, and a comparison of two captures
    /// does not depend on which encoding the driver used in each.
    fn of(data: &GraphEdgeData) -> Result<Self> {
        let programmatic_port = data.from_port == CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC;
        match data.kind {
            driver::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT if !programmatic_port => Ok(Self::Completion),
            driver::CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
            | driver::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT => Ok(Self::Programmatic),
            other => Err(Error::invalid(format!(
                "a graph edge states dependency kind {other}, which this walk does not know"
            ))),
        }
    }
}

/// One node an operation added, with the kinds of the dependencies that reach it.
///
/// The edges are stated **per node, in the walk's order**, rather than as pairs of
/// driver handles. Two captures of the same work are compared position by position
/// (design 4.5's rebind condition), and a handle belongs to one graph while a
/// position belongs to the order both walks agree on.
///
/// `incoming` holds a kind for every edge that reaches this node from another node
/// of the same walk, in the order their upstream ends are walked - so two captures of
/// the same work state the same vectors. An edge into a node added by an earlier
/// operation is not listed: it is that operation's, and it was stated then.
#[derive(Clone, Debug)]
pub struct WalkedNode {
    /// The node the operation added.
    pub node: GraphNode,
    /// The kinds of the edges from the nodes already walked into this one, in the
    /// driver's order. A node an operation added after another operation's nodes has
    /// one entry per dependency that crosses the boundary.
    pub incoming: Vec<EdgeKind>,
}
