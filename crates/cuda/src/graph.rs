//! Launch of CUDA graph executables instantiated outside this crate.

use crate::driver::{Handle, driver};
use crate::{Result, Stream};

/// A borrowed CUDA graph executable (`CUgraphExec`, the driver's name for
/// the runtime's `cudaGraphExec_t`).
///
/// The executable is instantiated and destroyed by its owner, for example a
/// `torch.cuda.CUDAGraph`, which also owns every allocation the graph's
/// nodes address. This handle only launches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphExec {
    raw: Handle,
}

// SAFETY: the driver accepts `cuGraphLaunch` from any host thread and orders
// each launch behind the earlier launches of the same executable. The handle
// is never dereferenced by this crate.
unsafe impl Send for GraphExec {}
// SAFETY: as for Send above; a shared reference only reads the handle, which
// `launch` passes to the driver.
unsafe impl Sync for GraphExec {}

impl GraphExec {
    /// Borrows an instantiated executable from its integer handle, as
    /// `torch.cuda.CUDAGraph.raw_cuda_graph_exec` returns it.
    ///
    /// # Safety
    /// `raw` must be a live graph executable. Its owner must keep the
    /// executable, and the memory its nodes read and write, alive until the
    /// work of every launch through the returned value and its copies has
    /// completed.
    pub unsafe fn borrowed(raw: usize) -> Self {
        Self { raw: raw as Handle }
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
    /// been captured on another stream of the same device, including a
    /// stream outside `stream`'s SM partition.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        let driver = driver()?;
        // The launch resolves the executable in the stream's context; the
        // calling thread may have none.
        let _context = stream.enter()?;
        // SAFETY: the executable is live by `borrowed`'s contract and the
        // stream is a live handle.
        driver.check(
            unsafe { (driver.graph_launch)(self.raw, stream.handle()) },
            "cuGraphLaunch",
        )
    }
}
