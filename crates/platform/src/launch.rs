//! Loaded kernels, their launch configuration, and per-function attributes.
//!
//! A kernel program's module carries the launch configuration its metadata
//! declares (Triton's `num_warps`, `shared`, `num_ctas`, `launch_pdl` and
//! `launch_cooperative_grid`); the operator computes only the grid. This
//! module turns that configuration into one `cuLaunchKernelEx` call, so a
//! three-dimensional grid, dynamic shared memory, a thread-block cluster,
//! programmatic dependent launch and cooperative launch all work, and sets a
//! function's attributes with `cuFuncSetAttribute` before its first launch
//! (a dynamic shared-memory size beyond 48 KiB needs
//! [`FunctionAttribute::MaxDynamicSharedMemory`]).

use std::ffi::c_void;

use crate::driver::{
    self, CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, CU_LAUNCH_ATTRIBUTE_COOPERATIVE,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, ClusterDim, Handle, LaunchAttribute,
    LaunchAttributeValue,
};
use crate::{Device, Error, Result, Stream};

/// A kernel of a loaded [`crate::Library`], resolved in its device's primary
/// context.
///
/// A `CUfunction` is valid only while the library it belongs to is loaded, so
/// an owner keeps the library beside every function it launches, as the
/// design's `Kernel::Cubin` holds both. The handle is the same in every
/// thread; only the context it was resolved in matters.
#[derive(Clone)]
pub struct Function {
    raw: Handle,
    device: Device,
}

// SAFETY: a function handle is a driver object valid in every thread; the
// context it carries is retained and thread-safe.
unsafe impl Send for Function {}
// SAFETY: as for Send above; a shared reference only passes the immutable
// handle to driver calls.
unsafe impl Sync for Function {}

impl Function {
    pub(crate) fn new(raw: Handle, device: Device) -> Self {
        Self { raw, device }
    }

    pub(crate) fn raw(&self) -> Handle {
        self.raw
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// Launches the kernel on `stream` with `config` and `parameters`, which
    /// points at each kernel parameter's value in declaration order, and
    /// returns without waiting for the kernel. A launch on a capturing stream
    /// becomes a kernel node of the captured graph, carrying the same launch
    /// attributes.
    ///
    /// # Safety
    ///
    /// `parameters` matches the kernel's parameter list, and the storage the
    /// kernel addresses outlives every execution of the launch, including
    /// every replay of a graph that captured it.
    pub fn launch_ex(
        &self,
        stream: &Stream,
        config: &LaunchConfig,
        parameters: &mut [*mut c_void],
    ) -> Result<()> {
        config.check()?;

        let driver = driver::driver()?;
        let _context = stream.enter()?;
        let mut attributes = [empty_attribute(); 3];
        let mut count = 0;
        if let Some(cluster) = config.cluster {
            attributes[count] = LaunchAttribute {
                id: CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
                pad: [0; 4],
                value: LaunchAttributeValue {
                    cluster: ClusterDim {
                        x: cluster.dims[0],
                        y: cluster.dims[1],
                        z: cluster.dims[2],
                    },
                },
            };
            count += 1;
            // A cluster's scheduling preference only applies with a cluster
            // dimension, so it is set beside one.
            if cluster.scheduling != ClusterScheduling::Default {
                attributes[count] = LaunchAttribute {
                    id: CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
                    pad: [0; 4],
                    value: LaunchAttributeValue {
                        cluster_scheduling: cluster.scheduling as i32,
                    },
                };
                count += 1;
            }
        }
        if config.pdl {
            attributes[count] = LaunchAttribute {
                id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                pad: [0; 4],
                value: LaunchAttributeValue {
                    programmatic_stream_serialization: 1,
                },
            };
            count += 1;
        }
        if config.cooperative {
            attributes[count] = LaunchAttribute {
                id: CU_LAUNCH_ATTRIBUTE_COOPERATIVE,
                pad: [0; 4],
                value: LaunchAttributeValue { cooperative: 1 },
            };
            count += 1;
        }

        let config = driver::LaunchConfig {
            grid_dim: config.grid,
            block_dim: config.block,
            shared_mem_bytes: config.shared_memory,
            stream: stream.handle(),
            attrs: attributes.as_mut_ptr(),
            num_attrs: count as u32,
        };
        driver.check(
            // SAFETY: forwarded from the caller; the attribute array and the
            // parameter array outlive the call, and the stream is live.
            unsafe {
                (driver.launch_kernel_ex)(
                    &config,
                    self.raw,
                    parameters.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernelEx",
        )
    }

    /// Sets one of the function's attributes, such as the largest dynamic
    /// shared-memory size it may be launched with.
    pub fn set_attribute(&self, attribute: FunctionAttribute, value: i32) -> Result<()> {
        let driver = driver::driver()?;
        let _context = self.device.enter()?;
        driver.check(
            // SAFETY: the function is live while its library is; the value is
            // passed by value.
            unsafe { (driver.func_set_attribute)(self.raw, attribute as i32, value) },
            "cuFuncSetAttribute",
        )
    }
}

/// One launch's configuration: the grid and block dimensions, the dynamic
/// shared-memory size, and what the module's metadata declares beyond them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Blocks per grid dimension.
    pub grid: [u32; 3],
    /// Threads per block dimension.
    pub block: [u32; 3],
    /// Dynamic shared memory per block, in bytes.
    pub shared_memory: u32,
    /// The thread-block cluster the blocks are grouped into, if any.
    pub cluster: Option<Cluster>,
    /// Programmatic dependent launch: the kernel resolves its stream
    /// dependency itself (`cudaGridDependencySynchronize`), so the driver may
    /// start it before the previous kernel finishes.
    pub pdl: bool,
    /// Cooperative launch: every block of the grid is resident together.
    pub cooperative: bool,
}

impl LaunchConfig {
    /// A configuration of `grid` blocks of `block` threads with no dynamic
    /// shared memory and no launch attributes.
    pub fn new(grid: [u32; 3], block: [u32; 3]) -> Self {
        Self {
            grid,
            block,
            ..Self::default()
        }
    }

    /// Rejects a configuration the driver would only refuse at launch, so the
    /// failure names the operator's launch instead of a driver status.
    fn check(&self) -> Result<()> {
        if self.grid.contains(&0) || self.block.contains(&0) {
            return Err(Error::invalid(format!(
                "a launch needs a non-zero grid and block, got grid {:?} and block {:?}",
                self.grid, self.block
            )));
        }
        if let Some(cluster) = self.cluster
            && cluster.dims.contains(&0)
        {
            return Err(Error::invalid(format!(
                "a cluster needs non-zero dimensions, got {:?}",
                cluster.dims
            )));
        }
        Ok(())
    }
}

/// The thread-block cluster a launch groups its blocks into: the cluster's
/// dimensions in blocks, and the order the device schedules clusters in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cluster {
    pub dims: [u32; 3],
    pub scheduling: ClusterScheduling,
}

impl Cluster {
    pub fn new(dims: [u32; 3]) -> Self {
        Self {
            dims,
            scheduling: ClusterScheduling::Default,
        }
    }
}

/// `CUclusterSchedulingPolicy`: how the device places clusters on the
/// multiprocessors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClusterScheduling {
    /// `CU_CLUSTER_SCHEDULING_POLICY_DEFAULT`: the driver's choice.
    #[default]
    Default,
    /// `CU_CLUSTER_SCHEDULING_POLICY_SPREAD`: spread clusters over the GPCs.
    Spread,
    /// `CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING`: pack clusters to fill
    /// multiprocessors.
    LoadBalancing,
}

/// A function attribute `cuFuncSetAttribute` sets, with the numeric values of
/// `CUfunction_attribute`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum FunctionAttribute {
    /// The largest dynamic shared-memory size, in bytes, the function may be
    /// launched with.
    MaxDynamicSharedMemory = 8,
    /// Requires every launch of the function to be a cluster launch.
    ClusterSizeMustBeSet = 10,
    /// The cluster width the function requires, in blocks.
    RequiredClusterWidth = 11,
    /// The cluster height the function requires, in blocks.
    RequiredClusterHeight = 12,
    /// The cluster depth the function requires, in blocks.
    RequiredClusterDepth = 13,
    /// Allows clusters larger than eight blocks where the architecture
    /// permits.
    NonPortableClusterSizeAllowed = 14,
}

/// An attribute slot the launch may leave unused.
fn empty_attribute() -> LaunchAttribute {
    LaunchAttribute {
        id: 0,
        pad: [0; 4],
        value: LaunchAttributeValue { pad: [0; 64] },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_launch_rejects_empty_grid_block_and_cluster_dimensions() {
        // A zero dimension is the caller's mistake, named before the driver
        // sees the launch.
        for config in [
            LaunchConfig {
                grid: [0, 1, 1],
                ..LaunchConfig::new([1, 1, 1], [1, 1, 1])
            },
            LaunchConfig {
                block: [1, 1, 0],
                ..LaunchConfig::new([1, 1, 1], [1, 1, 1])
            },
            LaunchConfig {
                cluster: Some(Cluster::new([1, 0, 1])),
                ..LaunchConfig::new([1, 1, 1], [1, 1, 1])
            },
        ] {
            assert!(matches!(config.check(), Err(Error::InvalidArgument(_))));
        }

        let well_formed = LaunchConfig {
            cluster: Some(Cluster::new([2, 1, 1])),
            pdl: true,
            cooperative: false,
            ..LaunchConfig::new([2, 4, 1], [128, 1, 1])
        };
        assert!(well_formed.check().is_ok());
    }
}
