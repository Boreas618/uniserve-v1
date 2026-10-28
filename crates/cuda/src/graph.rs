//! CUDA graph executables this process instantiates, launches and destroys.

use crate::driver::{Handle, driver, in_context};
use crate::{DeviceId, PrimaryContext, Result, Stream, reclaim};

/// A CUDA graph executable (`CUgraphExec`) owned by this value: it is
/// instantiated from a captured graph and destroyed when the value drops.
///
/// The executable holds the addresses of the storage its nodes read and
/// write and calls the kernels of the modules it was captured with; its
/// owner keeps both until the executable is dropped, and drains every
/// stream it was launched on first.
pub struct GraphExec {
    context: PrimaryContext,
    raw: Handle,
}

// SAFETY: the driver accepts `cuGraphLaunch` from any host thread and orders
// each launch behind the earlier launches of the same executable. The handle
// is never dereferenced by this crate.
unsafe impl Send for GraphExec {}
// SAFETY: as for Send above; a shared reference only reads the handle, which
// `launch` passes to the driver.
unsafe impl Sync for GraphExec {}

impl std::fmt::Debug for GraphExec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("GraphExec")
            .field(&(self.raw as usize))
            .finish()
    }
}

impl GraphExec {
    /// Instantiates the captured graph `graph` (a `CUgraph`, the runtime's
    /// `cudaGraph_t`, as `torch.cuda.CUDAGraph.raw_cuda_graph` returns it)
    /// on `device`. The executable does not depend on the graph afterwards,
    /// which its owner may destroy.
    ///
    /// # Safety
    /// `graph` must be a live captured graph of `device`'s context, which
    /// no other thread modifies during the call.
    ///
    /// # Errors
    ///
    /// Fails when the driver cannot instantiate the graph.
    pub unsafe fn instantiate(device: DeviceId, graph: usize) -> Result<Self> {
        let context = PrimaryContext::retain(device)?;
        let mut raw = std::ptr::null_mut();
        in_context(context.raw(), |driver| {
            driver.check(
                // SAFETY: the graph is live by the caller's contract; the
                // driver writes one executable handle to the local `raw`.
                unsafe { (driver.graph_instantiate)(&mut raw, graph as Handle, 0) },
                "cuGraphInstantiateWithFlags",
            )
        })?;
        Ok(Self { context, raw })
    }

    /// The `CUgraphExec` handle.
    pub fn raw(&self) -> usize {
        self.raw as usize
    }

    /// Submits one execution of the graph to `stream` and returns without
    /// waiting for it.
    ///
    /// The launch is ordered after the work already queued on `stream` and
    /// after every earlier launch of this executable. The graph may have
    /// been captured on another stream of the same device.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        let driver = driver()?;
        // The launch resolves the executable in the stream's context; the
        // calling thread may have none.
        let _context = stream.enter()?;
        // SAFETY: the executable is live while this value is, and the stream
        // is a live handle.
        driver.check(
            unsafe { (driver.graph_launch)(self.raw, stream.handle()) },
            "cuGraphLaunch",
        )
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        let (context, raw) = (self.context.clone(), self.raw as usize);
        // The owner drained every stream the executable was launched on.
        reclaim::defer(move || {
            in_context(context.raw(), |driver| {
                // SAFETY: the executable is owned and destroyed once.
                driver.check(
                    unsafe { (driver.graph_exec_destroy)(raw as Handle) },
                    "cuGraphExecDestroy",
                )
            })
        });
    }
}
