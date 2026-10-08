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
//! 2. [`bind`] allocates: one backing per stream, role, device and schema,
//!    sized to the maximum over every binding prepared on that stream (they
//!    run serially there), and one allocation per persistent requirement.
//! 3. `component.bind` receives the allocations as DLPack tensors once and
//!    creates operators over them.
//!
//! Every binding prepared for a stream must be prepared before any of them
//! is bound, so the shared backings never grow after a graph captured them.

use std::collections::BTreeMap;
use std::sync::Arc;

use tvm_ffi::{Any, Tensor};

use crate::config::Device;
use crate::ffi::{Backing, DType, Fields, Function, Functions, Handle, Record};
use crate::storage::memory::{Memory, StorageView};
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

    fn bytes(&self) -> Result<usize> {
        let elements = self.shape.iter().try_fold(1usize, |total, &extent| {
            usize::try_from(extent)
                .ok()
                .and_then(|extent| total.checked_mul(extent))
        });
        elements
            .and_then(|elements| elements.checked_mul(self.dtype.bytes()))
            .ok_or_else(|| {
                Error::Interface(format!("workspace {} has an invalid shape", self.name))
            })
    }

    /// Requirements sharing a backing must describe the same tensor.
    fn schema(&self) -> (String, &'static str, bool, usize) {
        (
            self.name.clone(),
            self.dtype.name(),
            self.host,
            self.shape.len(),
        )
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
    workspaces: Vec<Arc<Memory>>,
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

    /// Bytes of the Rust allocations backing this binding.
    pub fn workspace_bytes(&self) -> usize {
        self.workspaces.iter().map(|memory| memory.len()).sum()
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
    );
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

/// A shared backing's key: stream, role and device, then the schema of its
/// fields.
type SharedKey = (usize, String, i32, Vec<(String, &'static str, bool, usize)>);

/// The largest shape of each field over every binding sharing a backing.
fn envelopes(prepared: &[Prepared]) -> Result<BTreeMap<SharedKey, BTreeMap<String, Vec<i64>>>> {
    let mut shared: BTreeMap<SharedKey, BTreeMap<String, Vec<i64>>> = BTreeMap::new();
    for binding in prepared {
        let mut roles: BTreeMap<(String, i32), Vec<&Requirement>> = BTreeMap::new();
        for requirement in &binding.requirements {
            if let Some(role) = &requirement.role {
                roles
                    .entry((role.clone(), requirement.device))
                    .or_default()
                    .push(requirement);
            }
        }
        for ((role, device), fields) in roles {
            let schema = fields.iter().map(|field| field.schema()).collect();
            let envelope = shared
                .entry((binding.request.stream, role, device, schema))
                .or_default();
            for field in fields {
                let bound = envelope
                    .entry(field.name.clone())
                    .or_insert_with(|| vec![0; field.shape.len()]);
                for (bound, &extent) in bound.iter_mut().zip(&field.shape) {
                    *bound = (*bound).max(extent);
                }
            }
        }
    }
    Ok(shared)
}

fn allocate(requirement: &Requirement, shape: &[i64]) -> Result<(Arc<Memory>, Tensor)> {
    let sized = Requirement {
        shape: shape.to_vec(),
        ..requirement.clone()
    };
    let bytes = sized.bytes()?.max(1);
    let memory = match (requirement.device, requirement.host) {
        (-1, _) => Memory::host(bytes),
        (device, true) => Memory::pinned(device, bytes)?,
        (device, false) => Memory::device(device, bytes)?,
    };
    let backing: Arc<dyn Backing> = Arc::<Memory>::clone(&memory);
    let tensor = StorageView::contiguous(backing, 0, requirement.dtype, shape)?.to_tensor()?;
    Ok((memory, tensor))
}

/// Allocate every prepared binding's workspaces and bind them, in order.
///
/// Fails without binding the remaining requests when one fails; bindings
/// already returned by earlier calls are unaffected.
pub fn bind(interface: &Interface, prepared: Vec<Prepared>) -> Result<Vec<ComponentBinding>> {
    let _range = crate::profiling::range(c"uniserve.worker.bind", None);
    let envelopes = envelopes(&prepared)?;
    let mut backings: BTreeMap<(SharedKey, String), (Arc<Memory>, Tensor)> = BTreeMap::new();
    let mut bound = Vec::with_capacity(prepared.len());
    for binding in prepared {
        let mut workspaces = Vec::new();
        let mut tensors = Fields::new();
        let mut roles: BTreeMap<(String, i32), Vec<&Requirement>> = BTreeMap::new();
        for requirement in &binding.requirements {
            match &requirement.role {
                Some(role) => roles
                    .entry((role.clone(), requirement.device))
                    .or_default()
                    .push(requirement),
                None => {
                    let (memory, tensor) = allocate(requirement, &requirement.shape)?;
                    workspaces.push(memory);
                    tensors = tensors.with(&requirement.name, tensor);
                }
            }
        }
        for ((role, device), fields) in roles {
            let schema: Vec<_> = fields.iter().map(|field| field.schema()).collect();
            let key: SharedKey = (binding.request.stream, role, device, schema);
            for field in fields {
                let entry = (key.clone(), field.name.clone());
                if !backings.contains_key(&entry) {
                    let shape = envelopes
                        .get(&key)
                        .and_then(|envelope| envelope.get(&field.name))
                        .ok_or_else(|| {
                            Error::Invariant("shared workspace has no envelope".into())
                        })?;
                    let allocation = allocate(field, shape)?;
                    backings.insert(entry.clone(), allocation);
                }
                let (memory, tensor) = &backings[&entry];
                workspaces.push(Arc::clone(memory));
                tensors = tensors.with(&field.name, tensor.clone());
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
            workspaces,
            closed: false,
        });
    }
    Ok(bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirement(name: &str, role: Option<&str>, shape: &[i64]) -> Requirement {
        Requirement {
            name: name.into(),
            role: role.map(str::to_owned),
            dtype: DType::U8,
            shape: shape.to_vec(),
            host: false,
            device: 0,
        }
    }

    fn prepared(stream: usize, requirements: Vec<Requirement>) -> Prepared {
        Prepared {
            request: BindingRequest {
                component: "model".into(),
                path: String::new(),
                device: Device::Cuda(0),
                stream,
                size: None,
                backends: Backends::default(),
                derive_host_lengths: true,
                experts: None,
                weights: None,
                cache: None,
            },
            handle: Handle::new(Any::default()),
            requirements,
        }
    }

    #[test]
    fn shared_workspaces_take_the_largest_size_on_one_stream() {
        let bindings = vec![
            prepared(
                7,
                vec![
                    requirement("matmul/workspace", Some("matmul"), &[1024]),
                    requirement("0000/exchange", None, &[64]),
                ],
            ),
            prepared(
                7,
                vec![requirement("matmul/workspace", Some("matmul"), &[4096])],
            ),
            // Another stream runs concurrently and needs its own backing.
            prepared(
                9,
                vec![requirement("matmul/workspace", Some("matmul"), &[512])],
            ),
        ];
        let envelopes = envelopes(&bindings).unwrap();
        assert_eq!(envelopes.len(), 2);
        let sizes: Vec<_> = envelopes
            .iter()
            .map(|((stream, role, _, _), fields)| {
                (*stream, role.clone(), fields["matmul/workspace"].clone())
            })
            .collect();
        assert_eq!(
            sizes,
            vec![
                (7, "matmul".to_owned(), vec![4096]),
                (9, "matmul".to_owned(), vec![512])
            ]
        );
    }
}
