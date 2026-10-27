//! Paged diffusion trajectories and the physical lifetime of their two banks.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use ipc::{BufferId, Call, CallKind, LatentParams, MediaCall, RequestKey};

use crate::transfer::{ReadRef, Retirement};
use crate::{Error, Result, lock};

#[derive(Default)]
struct LatentSlot {
    pages: Vec<usize>,
    bank: u8,
    step_index: i64,
    generation: i64,
    units: usize,
    height: i64,
    width: i64,
}

/// The version of a slot's trajectory a call names, and the size of its
/// latent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TrajectoryVersion {
    /// The solver steps the latent's trajectory has completed, which is
    /// also the index of its next step.
    pub step_index: i64,
    /// The generation every commit of the trajectory advances.
    pub generation: i64,
    /// Latent units of the trajectory.
    pub units: i64,
    /// Latent raster height.
    pub height: i64,
    /// Latent raster width.
    pub width: i64,
}

/// A batch's prepared trajectory commit or release. The executor validates
/// all updates before making any result visible, then applies the same values.
#[derive(Clone)]
pub(crate) struct LatentUpdate {
    pub request_pool_idx: i64,
    pub params: LatentParams,
    pub expected_generation: i64,
    pub expected_step_index: i64,
    pub generation: i64,
    pub step_index: i64,
    pub release: bool,
    /// Whether an initialization commits a stored trajectory at
    /// `step_index` rather than noise at step zero.
    pub restored: bool,
}

impl LatentUpdate {
    /// Derive a completed numerical call's trajectory change from its plan
    /// and the slot's committed `step_index`, the pool's ledger
    /// ([`LatentPool::step_index`]).
    ///
    /// Latent preparation opens a trajectory at step zero, a denoising call
    /// advances the committed one by one step, and image decoding releases
    /// it as committed. Standalone media keeps one slot-local trajectory,
    /// versioned by step; transferred image trajectories use the call's
    /// explicit generations.
    pub(crate) fn for_call(
        slot: usize,
        call: &Call,
        params: &LatentParams,
        step_index: i64,
    ) -> Result<Self> {
        let (expected_generation, expected_step_index, generation, step_index, release) =
            match call.kind {
                CallKind::Media(MediaCall::LatentPreparation) => (
                    0,
                    0,
                    call.latent_output
                        .as_ref()
                        .map_or(1, |output| i64::from(output.generation)),
                    0,
                    false,
                ),
                CallKind::Media(MediaCall::Denoising) => (
                    call.latent_input
                        .as_ref()
                        .map_or(step_index + 1, |input| i64::from(input.generation)),
                    step_index,
                    call.latent_output
                        .as_ref()
                        .map_or(step_index + 2, |output| i64::from(output.generation)),
                    step_index + 1,
                    false,
                ),
                CallKind::Media(MediaCall::ImageDecoding) => {
                    let input = call.latent_input.as_ref().ok_or_else(|| {
                        Error::Invalid("image decoding has no latent trajectory".into())
                    })?;
                    (0, 0, i64::from(input.generation), step_index, true)
                }
                _ => {
                    return Err(Error::Invalid(
                        "call does not update a latent trajectory".into(),
                    ));
                }
            };

        Ok(Self {
            request_pool_idx: slot as i64,
            params: params.clone(),
            expected_generation,
            expected_step_index,
            generation,
            step_index,
            release,
            restored: false,
        })
    }

    /// The initialization `initial` of a slot-local trajectory, committing
    /// the stored trajectory a resumed request's latent preparation copied
    /// in at `step` instead of noise at step zero. A slot-local trajectory's
    /// generation is one past its step, as its denoising calls advance it.
    pub(crate) fn restored(initial: Self, step: i64) -> Result<Self> {
        if initial.expected_generation != 0 || initial.release || step < 1 {
            return Err(Error::Invalid(
                "a trajectory is restored by its initialization at a positive step".into(),
            ));
        }
        Ok(Self {
            generation: step + 1,
            step_index: step,
            restored: true,
            ..initial
        })
    }
}

struct ImportState {
    transfers: Vec<Arc<ReadRef>>,
    adopted: bool,
    released: bool,
}

/// Destination pages retained until adoption or abandonment and physical
/// completion of the reads writing them. Imported values occupy bank zero.
pub(crate) struct LatentImport {
    pub buffer: BufferId,
    pub request_pool_idx: usize,
    pub pages: Vec<usize>,
    pub units: usize,
    state: Mutex<ImportState>,
}

impl LatentImport {
    pub(crate) fn adopted(&self) -> bool {
        self.state().adopted
    }

    pub(crate) fn released(&self) -> bool {
        self.state().released
    }

    fn state(&self) -> MutexGuard<'_, ImportState> {
        lock(&self.state)
    }
}

struct ExportState {
    retirements: Vec<Arc<Retirement>>,
    released: bool,
}

/// An immutable page-bank version retained by transport readers, including
/// versions exported before the first trajectory commit.
pub(crate) struct LatentExport {
    pub buffer: BufferId,
    pub request_pool_idx: usize,
    pub bank: u8,
    pub pages: Vec<usize>,
    state: Mutex<ExportState>,
}

impl LatentExport {
    pub(crate) fn retirements(&self) -> Vec<Arc<Retirement>> {
        self.state().retirements.clone()
    }

    fn state(&self) -> MutexGuard<'_, ExportState> {
        lock(&self.state)
    }

    /// Whether transport readers may still read this version: it is not
    /// released, or a reader has not completed successfully (a failed or
    /// cancelled read establishes nothing about the device).
    fn retained(&self) -> bool {
        let state = self.state();
        !state.released
            || state
                .retirements
                .iter()
                .any(|completion| !completion.succeeded())
    }
}

/// Own scheduler-assigned pages, committed trajectories and transfer access.
/// [`LatentStorage`](super::LatentStorage) holds the pages' memory.
pub(crate) struct LatentPool {
    request_pool_size: usize,
    num_pages: usize,
    page_units: usize,
    slots: Vec<LatentSlot>,
    owners: Vec<usize>,
    imports: HashMap<usize, Arc<LatentImport>>,
    exports: HashMap<BufferId, Arc<LatentExport>>,
    retiring: HashSet<usize>,
}

impl LatentPool {
    pub(crate) fn new(
        request_pool_size: usize,
        num_pages: usize,
        page_units: usize,
    ) -> Result<Self> {
        if request_pool_size == 0 || num_pages < 2 || page_units == 0 {
            return Err(Error::Invalid(
                "latent-pool shape must contain slots, pages, and elements".into(),
            ));
        }

        Ok(Self {
            request_pool_size,
            num_pages,
            page_units,
            slots: (0..=request_pool_size)
                .map(|_| LatentSlot::default())
                .collect(),
            owners: vec![0; num_pages],
            imports: HashMap::new(),
            exports: HashMap::new(),
            retiring: HashSet::new(),
        })
    }

    pub(crate) fn slot(&self, slot: i64) -> Result<usize> {
        if slot < 1 || slot as usize > self.request_pool_size {
            return Err(Error::Invalid(
                "latent request slot is outside physical capacity".into(),
            ));
        }
        Ok(slot as usize)
    }

    pub(crate) fn page_units(&self) -> usize {
        self.page_units
    }

    /// The step index of the slot's committed trajectory: the solver steps
    /// it has completed, which is also the index of the next one. The pool
    /// is the rank's ledger of steps: a batch's commit advances it and an
    /// adopted import sets it, so a call reads the step it advances here; an
    /// empty slot is at step zero.
    pub(crate) fn step_index(&self, slot: usize) -> i64 {
        self.slots[slot].step_index
    }

    pub(crate) fn capacity_units(&self) -> usize {
        (self.num_pages - 1) * self.page_units
    }

    pub(crate) fn validate_pages(&self, pages: &[i64], units: i64) -> Result<Vec<usize>> {
        if units < 1
            || pages.len() != (units as usize).div_ceil(self.page_units)
            || pages
                .iter()
                .any(|&page| page < 1 || page as usize >= self.num_pages)
            || pages.iter().collect::<HashSet<_>>().len() != pages.len()
        {
            return Err(Error::Invalid(
                "latent page table is outside physical pool bounds".into(),
            ));
        }
        Ok(pages.iter().map(|&page| page as usize).collect())
    }

    pub(crate) fn exports(&self) -> impl Iterator<Item = &Arc<LatentExport>> {
        self.exports.values()
    }

    /// Check the trajectory addressed by a call and return its committed bank.
    pub(crate) fn current_bank(
        &self,
        slot: usize,
        trajectory: &TrajectoryVersion,
        pages: &[usize],
    ) -> Result<u8> {
        let current = &self.slots[slot];
        if current.step_index != trajectory.step_index
            || current.generation != trajectory.generation
            || current.units as i64 != trajectory.units
            || current.height != trajectory.height
            || current.width != trajectory.width
        {
            return Err(Error::Invalid(
                "latent allocation does not name the committed trajectory".into(),
            ));
        }
        if current.pages != pages {
            return Err(Error::Invalid(
                "latent page table does not match its committed trajectory".into(),
            ));
        }

        self.require_owners(pages, slot, None)?;
        Ok(current.bank)
    }

    pub(crate) fn next_bank(&self, slot: usize) -> u8 {
        1 - self.slots[slot].bank
    }

    /// The bank holding the slot's committed step.
    pub(crate) fn committed_bank(&self, slot: usize) -> u8 {
        self.slots[slot].bank
    }

    /// Prepare an export before the backend constructs its numerical views.
    /// Register it before allowing another owner operation on this pool.
    pub(crate) fn prepare_export(
        &self,
        buffer: BufferId,
        slot: usize,
        bank: u8,
        pages: Vec<usize>,
    ) -> Result<LatentExport> {
        if self.exports.contains_key(&buffer) {
            return Err(Error::Invalid(
                "latent export generation is already registered".into(),
            ));
        }

        Ok(LatentExport {
            buffer,
            request_pool_idx: slot,
            bank,
            pages,
            state: Mutex::new(ExportState {
                retirements: Vec::new(),
                released: false,
            }),
        })
    }

    /// Install the prepared object after numerical view creation succeeds.
    pub(crate) fn register_export(&mut self, source: Arc<LatentExport>) {
        self.exports.insert(source.buffer, source);
    }

    pub(crate) fn retain_export(
        &self,
        source: &LatentExport,
        retirement: Arc<Retirement>,
    ) -> Result<()> {
        let mut state = source.state();
        if state.released
            || self
                .exports
                .get(&source.buffer)
                .is_none_or(|current| !std::ptr::eq(&**current, source))
        {
            return Err(Error::Invalid(
                "latent export reservation is no longer active".into(),
            ));
        }
        state.retirements.push(retirement);
        Ok(())
    }

    pub(crate) fn release_exports(&self, buffers: &[BufferId]) {
        for buffer in buffers {
            if let Some(source) = self.exports.get(buffer) {
                source.state().released = true;
            }
        }
    }

    /// Release `request`'s published versions that a denoising step's
    /// write into `slot`'s next bank overwrites, and return their buffers,
    /// whose transport locations the caller revokes.
    ///
    /// The next bank holds the version two steps behind the committed one:
    /// the trajectory has moved past it, so its publication ends when the
    /// trajectory writes over it, as it does when it is freed. Readers that
    /// claimed it before keep the bank until they finished
    /// ([`Self::write_dependencies`]); a later read finds it revoked. A
    /// version of another request, or one not older than the committed
    /// version, is left to its Free.
    pub(crate) fn supersede(
        &self,
        slot: usize,
        request: RequestKey,
        pages: &[usize],
    ) -> Vec<BufferId> {
        let bank = self.next_bank(slot);
        let committed = self.slots[slot].generation;
        self.exports
            .values()
            .filter(|source| {
                source.request_pool_idx == slot
                    && source.bank == bank
                    && source.buffer.owner == request
                    && i64::from(source.buffer.generation) < committed
                    && source.pages.iter().any(|page| pages.contains(page))
            })
            .filter_map(|source| {
                let mut state = source.state();
                (!std::mem::replace(&mut state.released, true)).then_some(source.buffer)
            })
            .collect()
    }

    pub(crate) fn write_dependencies(&self, slot: usize, pages: &[usize]) -> Vec<Arc<Retirement>> {
        let bank = self.next_bank(slot);
        self.exports
            .values()
            .filter(|source| {
                source.bank == bank && source.pages.iter().any(|page| pages.contains(page))
            })
            .flat_map(|source| source.retirements())
            .collect()
    }

    /// Prepare bank-zero ownership before any numerical write. The owner may
    /// build views and initialize padding, then installs the returned object
    /// without changing pool assignments between these operations.
    pub(crate) fn prepare_import(
        &self,
        buffer: BufferId,
        slot: usize,
        pages: Vec<usize>,
        units: usize,
    ) -> Result<LatentImport> {
        self.require_empty(slot)?;
        self.require_owners(&pages, 0, None)?;

        Ok(LatentImport {
            buffer,
            request_pool_idx: slot,
            pages,
            units,
            state: Mutex::new(ImportState {
                transfers: Vec::new(),
                adopted: false,
                released: false,
            }),
        })
    }

    /// Install the prepared import after destination views are constructed.
    pub(crate) fn register_import(&mut self, write: Arc<LatentImport>) {
        let slot = write.request_pool_idx;
        for &page in &write.pages {
            self.owners[page] = slot;
        }
        self.slots[slot].pages.clone_from(&write.pages);
        self.imports.insert(slot, write);
    }

    pub(crate) fn retain_transfer(&self, write: &LatentImport, transfer: ReadRef) -> Result<()> {
        self.require_import(write)?;
        let mut state = write.state();
        if state.adopted {
            return Err(Error::Invalid(
                "resident latent import cannot accept another read".into(),
            ));
        }
        state.transfers.push(Arc::new(transfer));
        Ok(())
    }

    pub(crate) fn validate_adoption(
        &self,
        write: &LatentImport,
        generation: i64,
        step_index: i64,
        height: i64,
        width: i64,
    ) -> Result<()> {
        self.require_import(write)?;
        let state = write.state();
        if state.adopted || state.transfers.is_empty() {
            return Err(Error::Invalid("latent import cannot be adopted".into()));
        }
        self.validate_metadata(&TrajectoryVersion {
            step_index,
            generation,
            units: write.units as i64,
            height,
            width,
        })?;
        if generation != i64::from(write.buffer.generation) {
            return Err(Error::Invalid(
                "latent import generation disagrees with its product".into(),
            ));
        }
        Ok(())
    }

    /// Apply a validated adoption after every transfer result has ordered its
    /// producer fence on the consuming stream. Readiness alone is insufficient.
    pub(crate) fn adopt_import(
        &mut self,
        write: &LatentImport,
        generation: i64,
        step_index: i64,
        height: i64,
        width: i64,
    ) {
        let current = &mut self.slots[write.request_pool_idx];
        current.bank = 0;
        current.step_index = step_index;
        current.generation = generation;
        current.units = write.units;
        current.height = height;
        current.width = width;
        write.state().adopted = true;
    }

    /// Revoke the import and return transfers to cancel outside native locks.
    pub(crate) fn abandon_import(&self, write: &LatentImport) -> Result<Vec<Arc<ReadRef>>> {
        if write.released() {
            return Ok(Vec::new());
        }
        self.require_import(write)?;
        let mut state = write.state();
        if state.adopted {
            return Err(Error::Invalid(
                "resident latent import cannot be abandoned".into(),
            ));
        }

        state.released = true;
        Ok(state.transfers.clone())
    }

    pub(crate) fn cancel_imports(&self, requests: &HashSet<RequestKey>) -> Vec<Arc<ReadRef>> {
        let mut transfers = Vec::new();
        for write in self.imports.values() {
            if !requests.contains(&write.buffer.owner) {
                continue;
            }
            let mut state = write.state();
            if !state.adopted && !state.released {
                state.released = true;
                transfers.extend(state.transfers.iter().cloned());
            }
        }
        transfers
    }

    /// Validate all changes without partially committing a rejected batch.
    pub(crate) fn validate_updates(&self, updates: &[LatentUpdate]) -> Result<()> {
        let mut slots = HashSet::new();
        let mut claimed = HashSet::new();
        for update in updates {
            let params = &update.params;
            let slot = self.slot(update.request_pool_idx)?;
            if !slots.insert(slot) {
                return Err(Error::Invalid(
                    "latent commit repeats a request slot".into(),
                ));
            }
            let units = i64::from(params.latent_units);
            let height = i64::from(params.height);
            let width = i64::from(params.width);
            let pages = self.validate_pages(
                &params
                    .page_table
                    .iter()
                    .map(|&page| i64::from(page))
                    .collect::<Vec<_>>(),
                units,
            )?;

            let trajectory = |step_index, generation| TrajectoryVersion {
                step_index,
                generation,
                units,
                height,
                width,
            };
            if update.release {
                self.current_bank(
                    slot,
                    &trajectory(update.step_index, update.generation),
                    &pages,
                )?;
                continue;
            }

            self.validate_metadata(&trajectory(update.step_index, update.generation))?;
            if update.expected_generation == 0 {
                // An initialization publishes step zero, or the stored step
                // of the trajectory a resumed request restored.
                if update.expected_step_index != 0 || (update.step_index != 0 && !update.restored) {
                    return Err(Error::Invalid(
                        "latent initialization must publish step zero".into(),
                    ));
                }
                self.require_empty(slot)?;
                // A provisional export from this slot retains the bank, but
                // does not conflict with its own initial trajectory commit.
                self.require_owners(&pages, 0, Some(slot))?;
            } else {
                self.current_bank(
                    slot,
                    &trajectory(update.expected_step_index, update.expected_generation),
                    &pages,
                )?;
                if update.step_index <= update.expected_step_index {
                    return Err(Error::Invalid(
                        "latent successor does not advance its step".into(),
                    ));
                }
            }

            if update.generation <= update.expected_generation {
                return Err(Error::Invalid(
                    "latent commit does not advance its generation".into(),
                ));
            }
            for page in pages {
                if !claimed.insert(page) {
                    return Err(Error::Invalid(
                        "latent commits overlap physical pages".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Apply the previously validated updates. The executor must not mutate
    /// assignments or updates between validation and application. Returned
    /// transfers belong to released slots and must be cancelled by their owner.
    pub(crate) fn apply_updates(&mut self, updates: &[LatentUpdate]) -> Vec<Arc<ReadRef>> {
        let mut transfers = Vec::new();
        for update in updates {
            let params = &update.params;
            let slot = update.request_pool_idx as usize;
            if update.release {
                transfers.extend(self.clear_slot(slot));
                continue;
            }

            let current = &mut self.slots[slot];
            if update.expected_generation == 0 {
                current.pages = params
                    .page_table
                    .iter()
                    .map(|&page| page as usize)
                    .collect();
                for &page in &current.pages {
                    self.owners[page] = slot;
                }
            }
            current.bank = 1 - current.bank;
            current.generation = update.generation;
            current.step_index = update.step_index;
            current.units = params.latent_units as usize;
            current.height = i64::from(params.height);
            current.width = i64::from(params.width);
        }
        transfers
    }

    pub(crate) fn release_slots(&mut self, slots: &[i64]) -> Result<Vec<Arc<ReadRef>>> {
        let slots = slots
            .iter()
            .map(|&slot| self.slot(slot))
            .collect::<Result<Vec<_>>>()?;
        if slots.iter().copied().collect::<HashSet<_>>().len() != slots.len() {
            return Err(Error::Invalid(
                "latent release repeats a request slot".into(),
            ));
        }
        Ok(slots
            .into_iter()
            .flat_map(|slot| self.clear_slot(slot))
            .collect())
    }

    pub(crate) fn retirement_ready(&self, requests: &HashSet<RequestKey>) -> bool {
        !self
            .imports
            .values()
            .any(|write| requests.contains(&write.buffer.owner))
            && !self
                .exports
                .values()
                .any(|source| requests.contains(&source.buffer.owner))
    }

    fn require_empty(&self, slot: usize) -> Result<()> {
        let current = &self.slots[slot];
        if self.imports.contains_key(&slot)
            || current.generation != 0
            || current.units != 0
            || self.retiring.contains(&slot)
        {
            return Err(Error::Invalid(
                "request slot already owns a committed trajectory".into(),
            ));
        }
        Ok(())
    }

    fn require_owners(
        &self,
        pages: &[usize],
        owner: usize,
        export_slot: Option<usize>,
    ) -> Result<()> {
        if owner == 0
            && self.exports.values().any(|source| {
                Some(source.request_pool_idx) != export_slot
                    && source.pages.iter().any(|page| pages.contains(page))
            })
        {
            return Err(Error::Invalid(
                "latent pages are owned by a published version".into(),
            ));
        }
        if pages.iter().any(|&page| self.owners[page] != owner) {
            return Err(Error::Invalid(
                "latent page table is not owned by its request slot".into(),
            ));
        }
        Ok(())
    }

    fn validate_metadata(&self, trajectory: &TrajectoryVersion) -> Result<()> {
        let TrajectoryVersion {
            step_index,
            generation,
            units,
            height,
            width,
        } = *trajectory;
        if generation < 1 || step_index < 0 || units < 1 || height < 1 || width < 1 {
            return Err(Error::Invalid(
                "latent trajectory metadata is invalid".into(),
            ));
        }
        if units as usize > self.capacity_units() {
            return Err(Error::Invalid(
                "latent trajectory exceeds physical capacity".into(),
            ));
        }
        Ok(())
    }

    fn require_import(&self, write: &LatentImport) -> Result<()> {
        if write.released()
            || self
                .imports
                .get(&write.request_pool_idx)
                .is_none_or(|current| !std::ptr::eq(&**current, write))
        {
            return Err(Error::Invalid(
                "latent import reservation is no longer writable".into(),
            ));
        }
        Ok(())
    }

    fn clear_slot(&mut self, slot: usize) -> Vec<Arc<ReadRef>> {
        self.retiring.insert(slot);
        if let Some(write) = self.imports.get(&slot) {
            let mut state = write.state();
            state.released = true;
            if !state.adopted {
                return state.transfers.clone();
            }
        }
        Vec::new()
    }

    pub(crate) fn require_initial(&self, slot: usize, pages: &[usize]) -> Result<()> {
        self.require_empty(slot)?;
        self.require_owners(pages, 0, None)?;
        self.require_writable(1, pages)
    }

    /// Fail when a version that transport readers still retain covers
    /// `pages` of `bank`. A released version whose readers all completed
    /// retains nothing, whether or not the pool has reaped it yet: its
    /// readers' completions are applied by the owning thread in their own
    /// order, and a write that waited for them may be recorded before the
    /// next reap.
    pub(crate) fn require_writable(&self, bank: u8, pages: &[usize]) -> Result<()> {
        if self.exports.values().any(|source| {
            source.bank == bank
                && source.pages.iter().any(|page| pages.contains(page))
                && source.retained()
        }) {
            return Err(Error::Invalid(
                "latent bank is retained by a published version".into(),
            ));
        }
        Ok(())
    }

    /// Failed or cancelled physical completions retain their pages. Logical
    /// cancellation revokes visibility; it never establishes physical reuse.
    pub(crate) fn reap(&mut self) {
        self.exports.retain(|_, source| source.retained());
        self.imports.retain(|&slot, write| {
            let state = write.state();
            if (!state.adopted && !state.released)
                || state.transfers.iter().any(|transfer| !transfer.succeeded())
            {
                return true;
            }
            if state.released {
                self.retiring.insert(slot);
            }
            false
        });

        self.retiring.retain(|&slot| {
            if self.imports.contains_key(&slot)
                || self
                    .exports
                    .values()
                    .any(|source| source.request_pool_idx == slot)
            {
                return true;
            }

            let current = &mut self.slots[slot];
            for &page in &current.pages {
                self.owners[page] = 0;
            }
            *current = LatentSlot::default();
            false
        });
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use ipc::{BufferId, LatentParams, RequestKey};
    use ipc::{CallId, RequestId};

    use super::*;
    use crate::EventPool;
    use crate::transfer::{Read, ReadRef, Retirement};

    /// The first commit of a seven-unit and of a four-unit trajectory.
    const SEVEN_UNITS: TrajectoryVersion = TrajectoryVersion {
        step_index: 0,
        generation: 1,
        units: 7,
        height: 16,
        width: 112,
    };
    const FOUR_UNITS: TrajectoryVersion = TrajectoryVersion {
        step_index: 0,
        generation: 1,
        units: 4,
        height: 16,
        width: 64,
    };

    fn buffer(slot: u64) -> BufferId {
        BufferId {
            owner: RequestKey::new(1, RequestId(slot), 1),
            producer_call_id: CallId::new(1, 0),
            output_index: 0,
            generation: 1,
        }
    }

    /// A transport read of an import, whose physical retirement the test
    /// completes.
    fn read() -> ReadRef {
        ReadRef::new(&Read::new(EventPool::new(), None))
    }

    fn initial(slot: i64, pages: &[u32], units: u32) -> LatentUpdate {
        LatentUpdate {
            request_pool_idx: slot,
            params: LatentParams {
                request_key: buffer(slot as u64).owner,
                call_id: CallId::new(1, 0),
                page_table: pages.to_vec(),
                latent_units: units,
                height: 16,
                width: units * 16,
            },
            expected_generation: 0,
            expected_step_index: 0,
            generation: 1,
            step_index: 0,
            release: false,
            restored: false,
        }
    }

    #[test]
    fn pages_keep_their_order_and_rejected_commits_leave_slots_available() -> Result<()> {
        let mut pool = LatentPool::new(2, 5, 4)?;
        let first = pool.validate_pages(&[3, 1], 7)?;
        let second = pool.validate_pages(&[4, 2], 7)?;
        assert_eq!(first, [3, 1]);
        assert_eq!(second, [4, 2]);
        // Too few pages for the units, the padding page and repeated pages
        // are refused.
        assert!(pool.validate_pages(&[1], 7).is_err());
        assert!(pool.validate_pages(&[0, 1], 7).is_err());
        assert!(pool.validate_pages(&[2, 2], 7).is_err());

        let rejected = [initial(1, &[3, 1], 7), initial(2, &[1], 4)];
        assert!(pool.validate_updates(&rejected).is_err());
        pool.require_initial(1, &first)?;
        pool.require_initial(2, &second)?;

        let updates = [initial(1, &[3, 1], 7), initial(2, &[4, 2], 7)];
        pool.validate_updates(&updates)?;
        pool.apply_updates(&updates);
        assert_eq!(pool.current_bank(1, &SEVEN_UNITS, &[3, 1])?, 1);
        assert_eq!(pool.current_bank(2, &SEVEN_UNITS, &[4, 2])?, 1);
        assert!(pool.current_bank(1, &SEVEN_UNITS, &[1, 3]).is_err());
        Ok(())
    }

    #[test]
    fn an_exported_bank_waits_for_all_readers_across_trajectory_commits() -> Result<()> {
        let mut pool = LatentPool::new(2, 3, 4)?;
        pool.require_initial(1, &[1])?;
        let source = Arc::new(pool.prepare_export(buffer(1), 1, 1, vec![1])?);
        pool.register_export(Arc::clone(&source));
        let readers = [
            Arc::new(Retirement::default()),
            Arc::new(Retirement::default()),
        ];
        for reader in &readers {
            pool.retain_export(&source, Arc::clone(reader))?;
        }
        assert!(pool.require_initial(2, &[1]).is_err());

        let mut update = initial(1, &[1], 4);
        pool.validate_updates(std::slice::from_ref(&update))?;
        pool.apply_updates(std::slice::from_ref(&update));
        let bank = pool.current_bank(1, &FOUR_UNITS, &[1])?;
        pool.require_writable(1 - bank, &[1])?;
        update.expected_generation = 1;
        update.generation = 2;
        update.step_index = 1;
        pool.validate_updates(std::slice::from_ref(&update))?;
        pool.apply_updates(&[update]);

        pool.release_exports(&[buffer(1)]);
        readers[0].complete(Ok(()))?;
        pool.reap();
        assert!(pool.require_writable(pool.next_bank(1), &[1]).is_err());
        assert!(
            pool.write_dependencies(1, &[1])
                .iter()
                .any(|signal| !signal.done())
        );
        readers[1].complete(Ok(()))?;
        pool.reap();
        pool.require_writable(pool.next_bank(1), &[1])?;
        Ok(())
    }

    #[test]
    fn a_denoising_write_supersedes_the_version_two_steps_back() -> Result<()> {
        let mut pool = LatentPool::new(2, 3, 4)?;
        let request = buffer(1).owner;
        pool.require_initial(1, &[1])?;
        let mut update = initial(1, &[1], 4);
        pool.validate_updates(std::slice::from_ref(&update))?;
        pool.apply_updates(std::slice::from_ref(&update));

        // The prepared version is published, and one reader has claimed it.
        let prepared = pool.current_bank(1, &FOUR_UNITS, &[1])?;
        let source = Arc::new(pool.prepare_export(buffer(1), 1, prepared, vec![1])?);
        pool.register_export(Arc::clone(&source));
        let reader = Arc::new(Retirement::default());
        pool.retain_export(&source, Arc::clone(&reader))?;

        // The first step writes the other bank: nothing is superseded.
        assert!(pool.supersede(1, request, &[1]).is_empty());
        update.expected_generation = 1;
        update.generation = 2;
        update.step_index = 1;
        pool.validate_updates(std::slice::from_ref(&update))?;
        pool.apply_updates(&[update]);

        // The second step writes over the prepared version, which ends its
        // publication once, without a Free. The write still waits for the
        // reader that claimed it.
        assert_eq!(pool.next_bank(1), prepared);
        assert_eq!(pool.supersede(1, request, &[1]), [buffer(1)]);
        assert!(pool.supersede(1, request, &[1]).is_empty());
        assert!(
            pool.write_dependencies(1, &[1])
                .iter()
                .any(|signal| !signal.done())
        );
        assert!(pool.require_writable(prepared, &[1]).is_err());
        reader.complete(Ok(()))?;
        pool.reap();
        pool.require_writable(prepared, &[1])
    }

    #[test]
    fn abandoned_and_adopted_imports_keep_pages_until_physical_completion() -> Result<()> {
        let mut pool = LatentPool::new(2, 3, 4)?;
        let write = Arc::new(pool.prepare_import(buffer(1), 1, vec![1], 4)?);
        pool.register_import(Arc::clone(&write));
        let transfer = read();
        let retirement = transfer.read().retirement();
        pool.retain_transfer(&write, transfer)?;
        pool.abandon_import(&write)?;
        pool.reap();
        assert!(write.released());
        assert!(pool.prepare_import(buffer(2), 2, vec![1], 4).is_err());

        retirement.complete(Ok(()))?;
        pool.reap();
        let replacement = Arc::new(pool.prepare_import(buffer(2), 2, vec![1], 4)?);
        pool.register_import(Arc::clone(&replacement));
        assert!(pool.retain_transfer(&write, read()).is_err());

        let transfer = read();
        let retirement = transfer.read().retirement();
        pool.retain_transfer(&replacement, transfer)?;
        pool.validate_adoption(&replacement, 1, 0, 16, 64)?;
        // The consumer orders the producer fences before exposing the bank;
        // the copy may still be executing when its stream-ordered consumers
        // are admitted.
        pool.adopt_import(&replacement, 1, 0, 16, 64);
        assert!(replacement.adopted());
        assert_eq!(pool.current_bank(2, &FOUR_UNITS, &[1])?, 0);
        pool.release_slots(&[2])?;
        pool.reap();
        assert!(pool.require_initial(1, &[1]).is_err());

        retirement.complete(Ok(()))?;
        pool.reap();
        pool.require_initial(1, &[1])
    }

    #[test]
    fn failed_export_retirement_retains_only_its_own_pages() -> Result<()> {
        let mut pool = LatentPool::new(2, 3, 4)?;
        let source = Arc::new(pool.prepare_export(buffer(1), 1, 1, vec![1])?);
        pool.register_export(Arc::clone(&source));
        let retirement = Arc::new(Retirement::default());
        pool.retain_export(&source, Arc::clone(&retirement))?;
        retirement.complete(Err(Error::transport("reader completion unknown")))?;
        pool.release_exports(&[buffer(1)]);
        pool.release_slots(&[1])?;
        pool.reap();

        assert!(pool.require_initial(2, &[1]).is_err());
        assert!(!pool.retirement_ready(&HashSet::from([buffer(1).owner])));
        pool.require_initial(2, &[2])?;
        let updates = [initial(2, &[2], 4)];
        pool.validate_updates(&updates)?;
        pool.apply_updates(&updates);
        assert_eq!(pool.current_bank(2, &FOUR_UNITS, &[2])?, 1);
        assert!(pool.retirement_ready(&HashSet::from([buffer(2).owner])));
        Ok(())
    }
}
