//! Request slots, call dependencies, and accepted execution progress.

pub(crate) mod decode_state;
pub(crate) mod diffusion;
pub(crate) mod video;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use indexmap::IndexMap;
use uniserve_core::CallId;
use uniserve_ipc::{CallStatus, CanvasStep, NewRequest, RequestKey};

use crate::{Error, Result};

/// Accepted coordinates of a state-consuming call.
///
/// Cache lengths and logical positions count tokens. `flow_step` counts solver
/// steps; `rng_counter` counts draws in the request's sampling stream.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RequestProgress {
    pub logical_position: u64,
    pub rng_counter: u64,
    pub flow_step: u64,
    pub cache_visible_len: u64,
    pub cache_computed_len: u64,
    pub prompt_logits_ready: bool,
}

#[derive(Debug, Default)]
struct RequestState {
    progress: RequestProgress,
    // Last submitted canvas step, independent of accepted device results.
    canvas: Option<CanvasStep>,
    accepted_call: CallId,
    state_call: CallId,
    // Insertion order is submission order, which need not equal call-id order.
    pending: IndexMap<CallId, bool>,
    // Latent encodings recorded so far: the index of the next one's
    // posterior noise draw.
    posterior_draws: u64,
    closed: bool,
    retired: bool,
}

/// One admitted epoch of a request slot.
///
/// Only its pool changes execution state; a batch that binds its calls to
/// the request reads its progress. No foreign callback runs while the state
/// is locked.
#[derive(Debug)]
pub(crate) struct Request {
    admission: NewRequest,
    state: Mutex<RequestState>,
}

impl Request {
    fn new(admission: NewRequest) -> Self {
        let prefix = admission
            .ar
            .as_ref()
            .map_or(0, |ar| u64::from(ar.initial_position));

        Self {
            admission,
            state: Mutex::new(RequestState {
                progress: RequestProgress {
                    logical_position: prefix,
                    cache_visible_len: prefix,
                    cache_computed_len: prefix,
                    ..RequestProgress::default()
                },
                ..RequestState::default()
            }),
        }
    }

    pub(crate) fn admission(&self) -> &NewRequest {
        &self.admission
    }

    pub(crate) fn key(&self) -> RequestKey {
        self.admission.request_key
    }

    pub(crate) fn slot(&self) -> usize {
        self.admission.request_pool_idx as usize
    }

    pub(crate) fn progress(&self) -> Result<RequestProgress> {
        Ok(self.state()?.progress)
    }

    /// The posterior noise draw index of the request's next latent
    /// encoding: its latent encodings recorded before it. A request's
    /// encodings are recorded in their call order, so the index does not
    /// depend on batching.
    pub(crate) fn posterior_draw(&self) -> Result<u64> {
        let mut state = self.state()?;
        let index = state.posterior_draws;
        state.posterior_draws += 1;
        Ok(index)
    }

    pub(crate) fn retired(&self) -> Result<bool> {
        Ok(self.state()?.retired)
    }

    fn state(&self) -> Result<MutexGuard<'_, RequestState>> {
        self.state
            .lock()
            .map_err(|_| Error::State("request state lock is poisoned"))
    }
}

/// Own scheduler-assigned slots and advance their admitted request epochs.
pub(crate) struct RequestPool {
    // Slot zero belongs to inactive graph rows.
    slots: Vec<Option<u64>>,
    requests: HashMap<u64, Arc<Request>>,
    closed: bool,
}

impl RequestPool {
    pub(crate) fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Invalid(
                "request-pool capacity must be positive".into(),
            ));
        }

        Ok(Self {
            slots: vec![None; capacity + 1],
            requests: HashMap::new(),
            closed: false,
        })
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.len() - 1
    }

    pub(crate) fn get(&self, request_id: u64) -> Result<&Arc<Request>> {
        self.peek(request_id)
            .ok_or_else(|| Error::Invalid(format!("unknown request {request_id}")))
    }

    pub(crate) fn peek(&self, request_id: u64) -> Option<&Arc<Request>> {
        self.requests.get(&request_id)
    }

    pub(crate) fn has_open_requests(&self) -> Result<bool> {
        for request in self.requests.values() {
            let state = request.state()?;
            if !state.closed && !state.retired {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Install an admission at its assigned slot; identical retries retain
    /// progress. A different epoch may replace only a retired occupant.
    pub(crate) fn start(&mut self, admission: NewRequest) -> Result<Option<usize>> {
        let key = admission.request_key;
        let slot = self.validate_slot(admission.request_pool_idx as usize)?;

        if let Some(resident) = self.peek(key.request_id.0) {
            if resident.key() == key {
                if resident.admission() != &admission {
                    return Err(Error::Invalid(
                        "request admission conflicts with resident state".into(),
                    ));
                }
                return Ok(None);
            }
            if !resident.retired()? {
                return Err(Error::Invalid(
                    "request admission conflicts with resident state".into(),
                ));
            }
        }
        if let Some(id) = self.slots[slot]
            && !self.get(id)?.retired()?
        {
            return Err(Error::Invalid(format!(
                "request-pool index {slot} is occupied"
            )));
        }

        // Check both occupants before eviction: a failed admission must leave
        // the previous request available to its remaining result observers.
        self.remove(key.request_id.0);
        if let Some(id) = self.slots[slot] {
            self.remove(id);
        }
        self.requests
            .insert(key.request_id.0, Arc::new(Request::new(admission)));
        self.slots[slot] = Some(key.request_id.0);
        Ok(Some(slot))
    }

    /// Bind a batch's calls to their live request epochs.
    ///
    /// A batch names each request once (`Batch::validate`, where the engine
    /// builds it) and live requests hold distinct slots (`Self::start`), so
    /// only live state is checked: an unknown or stale request, a closed
    /// one, and a call already executing.
    pub(crate) fn bind_calls(&self, calls: &[(RequestKey, CallId)]) -> Result<Vec<Arc<Request>>> {
        if self.closed {
            return Err(Error::State("request pool is closed"));
        }
        let mut requests = Vec::with_capacity(calls.len());
        for &(key, id) in calls {
            let request = self.get(key.request_id.0)?;
            if request.key() != key {
                return Err(Error::Invalid(format!(
                    "call {id:?} has a stale request key"
                )));
            }
            let state = request.state()?;
            if state.closed {
                return Err(Error::Invalid(format!(
                    "call {id:?} targets a closed request"
                )));
            }
            if state.pending.contains_key(&id) {
                return Err(Error::Invalid("request call is already executing".into()));
            }
            requests.push(Arc::clone(request));
        }
        Ok(requests)
    }

    /// Check the complete submission before reserving any pending call.
    /// Each tuple names the request, call, and whether it advances request state.
    pub(crate) fn validate_pending(&self, calls: &[(RequestKey, CallId, bool)]) -> Result<()> {
        let mut seen = HashSet::new();
        for &(key, id, _) in calls {
            let request = self.get(key.request_id.0)?;
            if request.key() != key {
                return Err(Error::State("request commit lost its admitted slot"));
            }
            if request.state()?.pending.contains_key(&id) || !seen.insert((key, id)) {
                return Err(Error::State("request commit repeats an executing call"));
            }
        }
        Ok(())
    }

    /// Reserve all calls together in their submission order.
    pub(crate) fn add_pending(&mut self, calls: &[(RequestKey, CallId, bool)]) -> Result<()> {
        self.validate_pending(calls)?;
        for &(key, id, advances) in calls {
            self.get(key.request_id.0)?
                .state()?
                .pending
                .insert(id, advances);
        }
        Ok(())
    }

    /// Follow the last submitted state call, or the accepted admission root.
    /// Independent video encoders and transfers have no state predecessor.
    pub(crate) fn predecessor(&self, key: RequestKey, advances: bool) -> Result<Option<CallId>> {
        let Some(request) = self.peek(key.request_id.0) else {
            return Ok(None);
        };
        if request.key() != key || (request.admission.diffusion.is_some() && !advances) {
            return Ok(None);
        }

        let state = request.state()?;
        Ok(Some(
            state
                .pending
                .iter()
                .rev()
                .find_map(|(&id, &advances)| advances.then_some(id))
                .unwrap_or(state.state_call),
        ))
    }

    /// Submit the next canvas step of a retained request epoch.
    ///
    /// Queued steps can precede device completion, so these coordinates are
    /// distinct from accepted progress. A new epoch starts without a canvas;
    /// its device banks are initialized by step zero before their first read.
    pub(crate) fn advance_canvas(&self, key: RequestKey, step: CanvasStep) -> Result<()> {
        let request = self.get(key.request_id.0)?;
        let mut state = request.state()?;
        if request.key() != key || state.retired {
            return Err(Error::Invalid(
                "canvas step requires the retained request epoch".into(),
            ));
        }

        let follows = if step.step == 0 {
            state
                .canvas
                .is_none_or(|last| last.block.checked_add(1) == Some(step.block))
        } else {
            state.canvas
                == Some(CanvasStep {
                    block: step.block,
                    step: step.step - 1,
                })
        };
        if !follows {
            return Err(Error::Invalid(format!(
                "canvas step {step:?} does not continue {:?}",
                state.canvas
            )));
        }

        state.canvas = Some(step);
        Ok(())
    }

    /// Retire one call's pending state. Stale epochs and duplicate results
    /// cannot affect a reused slot; late results cannot regress progress.
    pub(crate) fn apply_result(
        &mut self,
        key: RequestKey,
        id: CallId,
        status: CallStatus,
        progress: Option<RequestProgress>,
    ) -> Result<()> {
        let Some(request) = self
            .peek(key.request_id.0)
            .filter(|request| request.key() == key)
        else {
            return Ok(());
        };
        let mut state = request.state()?;
        let Some(advances) = state.pending.shift_remove(&id) else {
            return Ok(());
        };

        if let Some(progress) = progress.filter(|_| id > state.accepted_call) {
            state.progress = progress;
            state.accepted_call = id;
        }
        if status == CallStatus::Ok && advances && id > state.state_call {
            state.state_call = id;
        }
        if status == CallStatus::Error {
            state.closed = true;
        }
        Ok(())
    }

    pub(crate) fn cancel_calls(&mut self, calls: &[(RequestKey, CallId)]) -> Result<()> {
        for &(key, id) in calls {
            if let Some(request) = self
                .peek(key.request_id.0)
                .filter(|request| request.key() == key)
            {
                let mut state = request.state()?;
                state.pending.shift_remove(&id);
                state.closed = true;
            }
        }
        Ok(())
    }

    pub(crate) fn finish(&mut self, key: RequestKey) -> Result<()> {
        if let Some(request) = self
            .peek(key.request_id.0)
            .filter(|request| request.key() == key)
        {
            request.state()?.closed = true;
        }
        Ok(())
    }

    pub(crate) fn retirement_ready(&self, key: RequestKey) -> Result<bool> {
        match self
            .peek(key.request_id.0)
            .filter(|request| request.key() == key)
        {
            Some(request) => Ok(request.state()?.pending.is_empty()),
            None => Ok(true),
        }
    }

    /// Mark a closed epoch reusable after its physical readers have drained.
    /// The execution owner must finish device and host writes before calling.
    pub(crate) fn retire(&mut self, request_id: u64) -> Result<()> {
        let mut state = self.get(request_id)?.state()?;
        if !state.closed || !state.pending.is_empty() {
            return Err(Error::State(
                "request retirement requires closed, completed execution",
            ));
        }
        state.retired = true;
        Ok(())
    }

    /// Remove the pool's reference after the execution owner releases a request.
    pub(crate) fn remove(&mut self, request_id: u64) {
        if let Some(request) = self.requests.remove(&request_id) {
            self.slots[request.slot()] = None;
        }
    }

    /// Release request references after the execution owner has drained work.
    pub(crate) fn close(&mut self) {
        self.closed = true;
        self.requests.clear();
        self.slots.fill(None);
    }

    fn validate_slot(&self, slot: usize) -> Result<usize> {
        if self.closed {
            return Err(Error::State("request pool is closed"));
        }
        if slot == 0 || slot > self.capacity() {
            return Err(Error::Invalid(format!(
                "request-pool index {slot} exceeds capacity"
            )));
        }
        Ok(slot)
    }
}

#[cfg(test)]
mod tests {
    use uniserve_core::{DiffusionSamplingParams, RequestId, SamplingParams, VideoTask};
    use uniserve_ipc::{ArRequestParams, VideoAdmission};

    use super::*;

    fn admission(id: u64, epoch: u64, slot: u32) -> NewRequest {
        NewRequest {
            request_key: RequestKey::new(1, RequestId(id), epoch),
            request_pool_idx: slot,
            ar: Some(ArRequestParams {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                finish_token_ids: Vec::new(),
                initial_position: 5,
                canvas: None,
            }),
            image: None,
            diffusion: None,
            video: None,
            prompt_token_ids: Vec::new(),
            input_images: 0,
        }
    }

    #[test]
    fn predecessors_follow_submission_order_and_skip_independent_work() -> Result<()> {
        for order in [[1, 2], [2, 1]] {
            let mut pool = RequestPool::new(1)?;
            let request = NewRequest {
                ar: None,
                diffusion: Some(DiffusionSamplingParams {
                    num_frames: 1,
                    video_units: 1,
                    seed: 0,
                    width: 16,
                    height: 16,
                }),
                video: Some(VideoAdmission {
                    task: VideoTask::T2va,
                    text_tags: vec![1],
                    conditions: Vec::new(),
                }),
                prompt_token_ids: vec![1],
                ..admission(7, 1, 1)
            };
            let key = request.request_key;
            pool.start(request)?;
            assert_eq!(pool.predecessor(key, true)?, Some(CallId::new(0, 0)));

            let [first, second] = order.map(|batch| CallId::new(batch, 0));
            pool.add_pending(&[(key, first, true), (key, second, true)])?;
            assert_eq!(pool.predecessor(key, true)?, Some(second));
            assert_eq!(pool.predecessor(key, false)?, None);

            pool.apply_result(key, second, CallStatus::Predicated, None)?;
            assert_eq!(pool.predecessor(key, true)?, Some(first));
            pool.apply_result(key, first, CallStatus::Ok, None)?;
            assert_eq!(pool.predecessor(key, true)?, Some(first));
        }
        Ok(())
    }

    #[test]
    fn failed_admission_preserves_both_resident_slots() -> Result<()> {
        let mut pool = RequestPool::new(2)?;
        let first = admission(7, 1, 1);
        let second = admission(8, 1, 2);
        pool.start(first.clone())?;
        pool.finish(first.request_key)?;
        pool.retire(7)?;
        pool.start(second.clone())?;

        // Replacing the first epoch would be legal, but its requested
        // destination still belongs to the second request. Neither
        // admission may be evicted.
        assert!(pool.start(admission(7, 2, 2)).is_err());
        assert_eq!(pool.get(7)?.admission(), &first);
        assert_eq!(pool.get(8)?.admission(), &second);
        assert_eq!(pool.start(first)?, None);
        assert_eq!(pool.start(second)?, None);
        Ok(())
    }

    #[test]
    fn a_reused_slot_starts_its_epoch_at_the_admitted_position() -> Result<()> {
        let mut pool = RequestPool::new(1)?;
        let first = admission(7, 1, 1);
        let key = first.request_key;
        pool.start(first)?;
        let call = CallId::new(1, 0);
        pool.add_pending(&[(key, call, true)])?;
        pool.apply_result(
            key,
            call,
            CallStatus::Ok,
            Some(RequestProgress {
                logical_position: 9,
                rng_counter: 4,
                cache_computed_len: 9,
                cache_visible_len: 9,
                prompt_logits_ready: true,
                ..RequestProgress::default()
            }),
        )?;
        assert_eq!(pool.get(7)?.progress()?.logical_position, 9);
        pool.finish(key)?;
        pool.retire(7)?;
        pool.start(admission(7, 2, 1))?;

        let current = pool.get(7)?;
        assert_eq!(current.progress()?.logical_position, 5);
        assert!(!current.retired()?);
        Ok(())
    }

    #[test]
    fn canvas_steps_follow_submission_and_restart_on_slot_reuse() -> Result<()> {
        let mut pool = RequestPool::new(1)?;
        let first = admission(7, 1, 1);
        let key = first.request_key;
        pool.start(first)?;
        assert!(
            pool.advance_canvas(key, CanvasStep { block: 0, step: 1 })
                .is_err()
        );
        pool.advance_canvas(key, CanvasStep { block: 0, step: 0 })?;
        assert!(
            pool.advance_canvas(key, CanvasStep { block: 0, step: 0 })
                .is_err()
        );
        pool.advance_canvas(key, CanvasStep { block: 0, step: 1 })?;
        pool.advance_canvas(key, CanvasStep { block: 1, step: 0 })?;

        pool.finish(key)?;
        pool.retire(7)?;
        let second = admission(7, 2, 1);
        let next = second.request_key;
        pool.start(second)?;
        assert!(
            pool.advance_canvas(key, CanvasStep { block: 1, step: 1 })
                .is_err()
        );
        assert!(
            pool.advance_canvas(next, CanvasStep { block: 0, step: 1 })
                .is_err()
        );
        pool.advance_canvas(next, CanvasStep { block: 0, step: 0 })?;
        Ok(())
    }
}
