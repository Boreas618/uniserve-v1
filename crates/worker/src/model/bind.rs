//! Component bindings: prepare, allocate workspaces in Rust, bind.
//!
//! A binding is one component call set's root module prepared for one
//! stream and one admitted size. Binding is three steps, so every workspace
//! is a Rust allocation with a fixed address:
//!
//! 1. `component.prepare` walks the module and declares the workspaces its
//!    operators need, without allocating. Shared transient work areas carry
//!    a role (`matmul`, `moe`, `attention`, `vsa.0`, `vsa.1`); the other
//!    requirements are persistent per binding.
//! 2. [`bind`] allocates through the stream's [`Workspaces`]: each shared
//!    field one backing per stream and device, sized to the largest binding
//!    prepared on that stream (they run serially there), and one allocation
//!    per persistent requirement.
//! 3. `component.bind` receives the allocations as DLPack tensors once and
//!    creates operators over them.
//!
//! Every binding prepared for a stream must be prepared before any of them
//! is bound, so the shared backings never grow after a graph captured them.

use std::collections::{BTreeMap, BTreeSet};

use tvm_ffi::Any;

use crate::config::Device;
use crate::ffi::{DType, Fields, Function, Functions, Handle, Record};
use crate::parallel::host::HostGroup;
use crate::storage::memory::StorageView;
use crate::storage::{WorkspaceRequest, Workspaces};
use crate::{Error, Result};

use super::interface::Interface;

/// Operator backends selected for a binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Backends {
    pub attention: String,
    pub vsa: String,
    pub matmul: String,
    pub moe: String,
}

impl Default for Backends {
    fn default() -> Self {
        Self {
            attention: "auto".into(),
            vsa: "auto".into(),
            matmul: "auto".into(),
            moe: "auto".into(),
        }
    }
}

/// The admitted text workload a binding is prepared for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextSize {
    pub tokens: usize,
    pub rows: usize,
}

/// What to bind.
pub struct BindingRequest {
    pub component: String,
    /// The binding's root module path.
    pub path: String,
    pub device: Device,
    /// The CUDA stream handle every call of the binding enqueues on; zero
    /// for a host binding.
    pub stream: usize,
    pub size: Option<TextSize>,
    pub backends: Backends,
    /// Derive missing host sequence lengths from device metadata.
    pub derive_host_lengths: bool,
    /// Capability-owned values (expert exchange, weight prefetch, KV cache
    /// views) passed through unchanged.
    pub experts: Option<Handle>,
    pub weights: Option<Handle>,
    pub cache: Option<Handle>,
}

/// One declared workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub name: String,
    /// Shared transient role, or `None` for a persistent workspace.
    pub role: Option<String>,
    pub dtype: DType,
    pub shape: Vec<i64>,
    /// Host (pinned on CUDA ranks) rather than device memory.
    pub host: bool,
    /// CUDA ordinal, `-1` for the host.
    pub device: i32,
}

impl Requirement {
    fn from_record(record: &Record) -> Result<Self> {
        let dtype = record.string("dtype")?;
        Ok(Self {
            name: record.string("name")?,
            role: record.optional_string("role")?,
            dtype: DType::from_name(&dtype).ok_or_else(|| {
                Error::Interface(format!(
                    "component.prepare declares the unknown dtype {dtype:?}"
                ))
            })?,
            shape: record.shape("shape")?,
            host: record.flag("host")?,
            device: i32::try_from(record.get::<i64>("device")?)
                .map_err(|_| Error::Interface("component.prepare device is invalid".into()))?,
        })
    }
}

/// A declared binding awaiting its workspaces.
pub struct Prepared {
    pub request: BindingRequest,
    handle: Handle,
    pub requirements: Vec<Requirement>,
}

/// A bound component call set.
pub struct ComponentBinding {
    pub component: String,
    pub path: String,
    pub device: Device,
    pub stream: usize,
    pub size: Option<TextSize>,
    /// The prepared Python object; capability runtimes create their step
    /// functions over it (`<area>.bind(prepared, ...)`).
    prepared: Handle,
    kernels: Function,
    close: Function,
    /// Workspaces retained until every graph and call that addresses them
    /// has retired; Python views also retain them.
    workspaces: Vec<StorageView>,
    closed: bool,
}

impl ComponentBinding {
    pub fn prepared(&self) -> &Handle {
        &self.prepared
    }

    /// Selected kernels by module path, after numerical warmup.
    pub fn kernels(&self) -> Result<Vec<Record>> {
        let value: tvm_ffi::Array<Any> = self.kernels.call_as(&[])?;
        value
            .iter()
            .enumerate()
            .map(|(index, record)| {
                Record::new(format!("{}.kernels[{index}]", self.component), record)
            })
            .collect()
    }

    /// Release operators. An aborted close drops Python references without
    /// waiting for device work, and keeps the workspaces until exit.
    #[allow(clippy::mem_forget)]
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
        if aborted {
            // Device work may still address the workspaces; keep them.
            std::mem::forget(std::mem::take(&mut self.workspaces));
            return Ok(());
        }
        let closed = self.close.call_values(&[Any::from(false)]).map(drop);
        self.workspaces.clear();
        closed
    }
}

/// Declare the workspaces of one binding.
pub fn prepare(
    interface: &Interface,
    loaded: &Handle,
    request: BindingRequest,
) -> Result<Prepared> {
    let device = match request.device {
        Device::Cpu => -1,
        Device::Cuda(index) => i64::from(index),
    };
    let size = request.size.map(|size| {
        Fields::new()
            .with("tokens", size.tokens as i64)
            .with("rows", size.rows as i64)
            .map()
    });
    let fields = Fields::new()
        .text("component", &request.component)
        .text("path", &request.path)
        .with("device", device)
        .with("stream", request.stream as i64)
        .optional("size", size)
        .text("attention", &request.backends.attention)
        .text("vsa", &request.backends.vsa)
        .text("matmul", &request.backends.matmul)
        .text("moe", &request.backends.moe)
        .with("derive_host_lengths", request.derive_host_lengths)
        .optional("experts", request.experts.as_ref().map(Handle::to_any))
        .optional("weights", request.weights.as_ref().map(Handle::to_any))
        .optional("cache", request.cache.as_ref().map(Handle::to_any))
        .into_any();
    let result: tvm_ffi::Array<Any> = interface
        .prepare
        .call_as(&[loaded.view(), (&fields).into()])?;
    if result.len() != 2 {
        return Err(Error::Interface(
            "component.prepare returns [prepared, requirements]".into(),
        ));
    }
    let handle = Handle::new(
        result
            .get(0)
            .map_err(|error| Error::numerical("component.prepare", &error))?,
    )?;
    let requirements = Record::new(
        "component.prepare",
        result
            .get(1)
            .map_err(|error| Error::numerical("component.prepare", &error))?,
    )?
    .records("workspaces")?
    .iter()
    .map(Requirement::from_record)
    .collect::<Result<Vec<_>>>()?;
    Ok(Prepared {
        request,
        handle,
        requirements,
    })
}

/// The workspaces of each stream and device bindings run on.
#[derive(Default)]
pub struct StreamWorkspaces {
    streams: BTreeMap<(usize, i32), Workspaces>,
}

impl StreamWorkspaces {
    fn requests(binding: &Prepared, device: i32) -> Vec<WorkspaceRequest> {
        binding
            .requirements
            .iter()
            .filter(|requirement| requirement.device == device)
            .map(|requirement| WorkspaceRequest {
                name: requirement.name.clone(),
                dtype: requirement.dtype,
                shape: requirement.shape.clone(),
                host: requirement.host,
                // A shared field is named `{role}/{field}`: every binding on
                // the stream borrows one backing per field of a role.
                role: requirement.role.as_ref().map(|_| requirement.name.clone()),
            })
            .collect()
    }

    fn devices(binding: &Prepared) -> BTreeSet<i32> {
        binding
            .requirements
            .iter()
            .map(|requirement| requirement.device)
            .collect()
    }

    fn workspaces(&mut self, stream: usize, device: i32) -> &mut Workspaces {
        self.streams
            .entry((stream, device))
            .or_insert_with(|| Workspaces::new((device >= 0).then_some(device)))
    }

    /// Refuse new workspace storage: captured graphs address the current
    /// backings.
    pub fn seal(&mut self) {
        self.streams.values_mut().for_each(Workspaces::seal);
    }
}

/// The worker's ranks and the mesh groups bindings communicate over.
pub struct PeerGroup<'a> {
    /// Host collectives over the worker's ranks; member `i` is rank `i`.
    pub group: &'a HostGroup,
    /// Each component's multi-member mesh groups on this rank.
    pub collectives: BTreeMap<String, Vec<crate::parallel::CollectiveGroup>>,
}

impl PeerGroup<'_> {
    /// Bind `binding`'s collectives to NCCL communicators on its stream
    /// (`owner`), collectively over each group's members, and route the
    /// binding's numerical collectives to them (`distributed.bind_collectives`).
    /// `None` when the binding's component communicates over no group, and
    /// for a host binding: it has no stream, and its collectives run on the
    /// component's host process groups.
    pub fn bind_collectives(
        &self,
        interface: &Interface,
        binding: &ComponentBinding,
        owner: Option<&crate::stream::CUDAStream>,
    ) -> Result<Option<crate::parallel::StreamCollectives>> {
        let Some(groups) = self
            .collectives
            .get(&binding.component)
            .filter(|groups| !groups.is_empty())
        else {
            return Ok(None);
        };
        if binding.device == Device::Cpu {
            return Ok(None);
        }
        let owner = owner
            .ok_or_else(|| Error::Unsupported("stream collectives require a CUDA stream".into()))?;
        let collectives = crate::parallel::StreamCollectives::bind(owner, self.group, groups)?;
        interface
            .function("distributed.bind_collectives")?
            .call_values(&[binding.prepared().to_any(), collectives.tables()])?;
        Ok(Some(collectives))
    }
}

/// Allocate every prepared binding's workspaces and bind them, in order.
///
/// All bindings are declared to their streams' workspaces before any
/// allocation, so a shared field's backing holds the largest of them.
pub fn bind(
    interface: &Interface,
    workspaces: &mut StreamWorkspaces,
    prepared: Vec<Prepared>,
) -> Result<Vec<ComponentBinding>> {
    let _range = crate::profiling::range(c"uniserve.worker.bind", None);
    for binding in &prepared {
        for device in StreamWorkspaces::devices(binding) {
            let requests = StreamWorkspaces::requests(binding, device);
            workspaces
                .workspaces(binding.request.stream, device)
                .declare(&requests)?;
        }
    }

    let mut bound = Vec::with_capacity(prepared.len());
    for binding in prepared {
        let mut views = Vec::new();
        let mut tensors = Fields::new();
        for device in StreamWorkspaces::devices(&binding) {
            let requests = StreamWorkspaces::requests(&binding, device);
            for (name, view) in workspaces
                .workspaces(binding.request.stream, device)
                .bind(&requests)?
            {
                tensors = tensors.with(&name, view.to_tensor()?);
                views.push(view);
            }
        }

        let resources = Fields::new().with("workspaces", tensors.map()).into_any();
        let functions = Functions::new(
            "component.bind",
            interface
                .bind
                .call(&[binding.handle.view(), (&resources).into()])?,
        )?;
        let request = binding.request;
        bound.push(ComponentBinding {
            component: request.component,
            path: request.path,
            device: request.device,
            stream: request.stream,
            size: request.size,
            prepared: binding.handle,
            kernels: functions.get("kernels")?,
            close: functions.get("close")?,
            workspaces: views,
            closed: false,
        });
    }
    Ok(bound)
}
