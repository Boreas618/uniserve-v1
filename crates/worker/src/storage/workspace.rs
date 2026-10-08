//! Bind-time workspaces requested by numerical bindings.
//!
//! A binding's preparation declares the buffers its operators need instead
//! of allocating them. Persistent requests get one allocation each. Shared
//! requests name a role (`attention`, `matmul`, ...): every binding on one
//! stream that declares a role and element type borrows the same backing,
//! sized to the largest request, because those bindings run serialized on
//! that stream. Backings are fixed before the first graph capture, so
//! captured graphs see stable addresses; growth after sealing is refused.

use std::collections::HashMap;
use std::sync::Arc;

use super::memory::{Memory, StorageView};
use crate::ffi::{Backing, DType};
use crate::{Error, Result};

/// One requested buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceRequest {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<i64>,
    /// Page-locked host memory instead of device memory.
    pub host: bool,
    /// The shared role, or `None` for a persistent buffer.
    pub role: Option<String>,
}

impl WorkspaceRequest {
    fn bytes(&self) -> Result<usize> {
        self.shape
            .iter()
            .try_fold(self.dtype.bytes(), |bytes, &extent| {
                usize::try_from(extent)
                    .ok()
                    .and_then(|extent| bytes.checked_mul(extent))
            })
            .ok_or_else(|| Error::Invalid(format!("workspace {} has an invalid shape", self.name)))
    }
}

type SharedKey = (String, bool);

/// The workspaces of one stream's bindings.
pub struct Workspaces {
    device: Option<i32>,
    /// Largest declared bytes per shared role and memory kind.
    declared: HashMap<SharedKey, usize>,
    shared: HashMap<SharedKey, Arc<Memory>>,
    /// Allocation bytes so far, for capacity accounting.
    bytes: u64,
    sealed: bool,
}

impl Workspaces {
    pub fn new(device: Option<i32>) -> Self {
        Self {
            device,
            declared: HashMap::new(),
            shared: HashMap::new(),
            bytes: 0,
            sealed: false,
        }
    }

    /// Record shared requests before any allocation, so each role's backing
    /// is sized to the largest binding that uses it.
    pub fn declare(&mut self, requests: &[WorkspaceRequest]) -> Result<()> {
        for request in requests {
            if let Some(role) = &request.role {
                let bytes = request.bytes()?;
                let declared = self
                    .declared
                    .entry((role.clone(), request.host))
                    .or_default();
                *declared = (*declared).max(bytes);
            }
        }
        Ok(())
    }

    fn allocate(&self, bytes: usize, host: bool) -> Result<Arc<Memory>> {
        match (self.device, host) {
            (Some(device), false) => Memory::device(device, bytes.max(1)),
            (Some(device), true) => Memory::pinned(device, bytes.max(1)),
            (None, _) => Ok(Memory::host(bytes.max(1))),
        }
    }

    /// Allocate (or borrow, for shared roles) every requested buffer.
    /// Fails after [`seal`](Self::seal) when a request needs new memory.
    pub fn bind(&mut self, requests: &[WorkspaceRequest]) -> Result<Vec<(String, StorageView)>> {
        self.declare(requests)?;
        requests
            .iter()
            .map(|request| {
                let bytes = request.bytes()?;
                let memory = match &request.role {
                    Some(role) => {
                        let key = (role.clone(), request.host);
                        match self.shared.get(&key) {
                            Some(memory) if memory.len() >= bytes => Arc::clone(memory),
                            _ => {
                                self.require_open(&request.name)?;
                                let size = self.declared.get(&key).copied().unwrap_or(bytes);
                                let memory = self.allocate(size.max(bytes), request.host)?;
                                self.bytes += memory.len() as u64;
                                self.shared.insert(key, Arc::clone(&memory));
                                memory
                            }
                        }
                    }
                    None => {
                        self.require_open(&request.name)?;
                        let memory = self.allocate(bytes, request.host)?;
                        self.bytes += memory.len() as u64;
                        memory
                    }
                };
                let view = StorageView::contiguous(
                    memory as Arc<dyn Backing>,
                    0,
                    request.dtype,
                    &request.shape,
                )?;
                Ok((request.name.clone(), view))
            })
            .collect()
    }

    fn require_open(&self, name: &str) -> Result<()> {
        if self.sealed {
            return Err(Error::Invalid(format!(
                "workspace {name} needs new storage after graph capture sealed it"
            )));
        }
        Ok(())
    }

    /// Refuse new allocations: captured graphs address the current backings.
    pub fn seal(&mut self) {
        self.sealed = true;
    }

    /// Bytes allocated so far.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(name: &str, elements: i64, role: Option<&str>) -> WorkspaceRequest {
        WorkspaceRequest {
            name: name.into(),
            dtype: DType::F32,
            shape: vec![elements],
            host: false,
            role: role.map(str::to_owned),
        }
    }

    #[test]
    fn shared_roles_take_the_largest_declaration_and_sealing_stops_growth() -> Result<()> {
        let mut workspaces = Workspaces::new(None);
        workspaces.declare(&[request("big.scratch", 64, Some("attention"))])?;
        let small = workspaces.bind(&[
            request("small.scratch", 8, Some("attention")),
            request("small.table", 4, None),
        ])?;
        let big = workspaces.bind(&[request("big.scratch", 64, Some("attention"))])?;
        assert_eq!(small[0].1.address(), big[0].1.address());
        assert_eq!(workspaces.bytes(), 64 * 4 + 4 * 4);

        workspaces.seal();
        assert!(workspaces.bind(&[request("late", 4, None)]).is_err());
        assert!(
            workspaces
                .bind(&[request("other", 4, Some("matmul"))])
                .is_err()
        );
        assert!(
            workspaces
                .bind(&[request("reuse", 16, Some("attention"))])
                .is_ok()
        );
        Ok(())
    }
}
