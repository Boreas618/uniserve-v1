//! The rank's storage owners and their part in request lifecycles.
//!
//! Startup allocates them on the capacity plan: the device event pool, the
//! product store (relay lanes and the buffer pool), the product transports
//! and, for a text model, the KV cache (unit planes, block tables and import
//! workspaces). Capability runtimes borrow them to execute calls; lifecycle
//! commands release their request storage through [`RequestStorage`].

use std::collections::HashSet;
use std::sync::Arc;

use uniserve_core::CallId;
use uniserve_cuda::Stream;
use uniserve_ipc::{BufferId, RequestKey, Wake};

use crate::Result;
use crate::executor::lifecycle::RequestStorage;
use crate::storage::{KVCache, TensorStore};
use crate::stream::EventPool;
use crate::transfer::{Notify, Transports};

/// Deliver transfer-capacity notifications.
pub(super) fn notify(callbacks: Vec<Notify>) {
    for callback in callbacks {
        callback();
    }
}

/// The storage owners of a rank.
pub struct RankStorage {
    pub events: Arc<EventPool>,
    pub tensors: TensorStore,
    pub transports: Transports,
    pub kv: Option<KVCache>,
    /// Stream that orders slot releases after the batches reading them: the
    /// stream serving token calls. A borrowed view of a stream the worker's
    /// lanes own; the lanes close after this owner.
    release_stream: Option<Stream>,
}

impl RankStorage {
    pub fn new(
        events: Arc<EventPool>,
        tensors: TensorStore,
        transports: Transports,
        kv: Option<KVCache>,
        release_stream: Option<Stream>,
    ) -> Self {
        Self {
            events,
            tensors,
            transports,
            kv,
            release_stream,
        }
    }

    /// Release the products (and KV exports) of finished calls.
    pub fn release_calls(&mut self, calls: &[(RequestKey, CallId)]) -> Result<()> {
        if calls.is_empty() {
            return Ok(());
        }
        self.tensors.release_calls(calls);
        if let Some(kv) = &mut self.kv {
            let released = kv.manager().release_calls(calls);
            kv.manager().release_exports(&released);
        }
        Ok(())
    }
}

impl RequestStorage for RankStorage {
    fn admit_slots(&mut self, _slots: &[usize]) -> Result<()> {
        // Block tables are installed per batch; a new slot has no state.
        Ok(())
    }

    fn release_buffers(&mut self, buffers: &HashSet<BufferId>) -> Result<()> {
        self.tensors.release_buffers(buffers)?;
        if let Some(kv) = &mut self.kv {
            kv.release_imports(buffers)?;
            kv.manager()
                .release_exports(&buffers.iter().copied().collect::<Vec<_>>());
        }
        Ok(())
    }

    fn release_requests(
        &mut self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<()> {
        self.tensors.release_requests(requests, retained)?;
        if let Some(kv) = &mut self.kv {
            kv.cancel_request_imports(requests, retained)?;
            let exports: Vec<BufferId> = kv
                .manager()
                .exported_buffers()
                .into_iter()
                .filter(|buffer| requests.contains(&buffer.owner) && !retained.contains(buffer))
                .collect();
            kv.manager().release_exports(&exports);
        }
        Ok(())
    }

    fn retirement_ready(
        &mut self,
        buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        Ok(self.tensors.retirement_ready(buffers, requests, retained)?
            && self
                .kv
                .as_mut()
                .is_none_or(|kv| kv.manager().retirement_ready(buffers, requests, retained)))
    }

    fn retire_slots(&mut self, slots: &[usize]) -> Result<()> {
        if let Some(kv) = &mut self.kv {
            let slots: Vec<u32> = slots.iter().map(|&slot| slot as u32).collect();
            kv.release_slots(&slots, self.release_stream.as_ref())?;
        }
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        self.events.reap()?;
        self.transports.reap()?;
        if let Some(kv) = &self.kv {
            kv.reap_imports()?;
        }
        Ok(())
    }

    fn awaiting_acknowledgment(&self) -> bool {
        self.transports.awaiting_acknowledgment()
    }

    fn set_wake(&self, wake: Option<Wake>) {
        self.events.set_wake(wake.clone());
        self.transports.set_wake(wake.clone());
        if let Some(kv) = &self.kv {
            kv.set_wake(wake);
        }
    }

    #[allow(clippy::mem_forget)]
    fn close(&mut self, aborted: bool) -> Result<()> {
        if aborted {
            // Device work may still address the pool and the staging.
            std::mem::forget(self.kv.take());
            return Ok(());
        }
        let mut result = Ok(());
        if let Some(kv) = &mut self.kv {
            result = result.and(kv.close());
        }
        result = result.and(self.transports.close());
        self.tensors.close();
        result.and(self.events.close())
    }
}
