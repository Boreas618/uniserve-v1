//! The trajectory ledger of a denoising lane: which bank of the latent
//! pool holds each request's committed step.
//!
//! A lane that denoises latent trajectories (a video or an image model's
//! denoiser) keeps every request's samples in two banks of pages of a
//! latent pool the worker owns: a step reads the bank holding the
//! committed step and writes the other, and its commit, applied to the
//! ledger at the batch's commit point, makes the written bank the committed
//! one. The executor reads and writes the banks itself; the worker decides
//! which pages and banks a call addresses ([`LatentLedger::bind`]), checks
//! every commit before the commit point and applies it after.
//!
//! The pages of a trajectory follow its model's layout ([`Layout`]): a
//! video request slot owns a fixed run of pages after the padding page 0,
//! and its raster is the canvas its request was admitted with; an image
//! trajectory occupies the pages the scheduler assigned its call
//! (`Batch::latent_params`), at its request's admitted image size, and its
//! versions are published to and imported from other ranks
//! ([`crate::executor::transfer::latents`]).
//!
//! Batches of lanes on different streams may address the pool (an image
//! model's denoiser and its decoder): each waits for the previous user on
//! another stream ([`LatentLedger::enter`]) and records its end
//! ([`LatentLedger::leave`]).

use std::collections::{HashMap, HashSet};

use ipc::{BufferId, Call, CallKind, ImageParams, LatentParams, MediaCall, RequestKey};

use crate::Error;
use crate::executor::batch::StreamOrder;
use crate::memory::TrajectoryVersion;
use crate::memory::allocation::StorageView;
use crate::stream::Issue;
use crate::{LatentPool, LatentStorage, LatentUpdate, Result};

/// How a ledger's trajectories occupy its pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// Every request slot owns `pages` consecutive pages after the padding
    /// page (a video model).
    Slots { pages: u64 },
    /// Each call names the pages the scheduler assigned it (an image model).
    Scheduled,
}

/// The trajectory ledger of a denoising rank: the latent pool whose two
/// banks of pages hold every request's samples, which bank holds a
/// request's committed step, and the storage of the banks.
pub(crate) struct LatentLedger {
    pool: LatentPool,
    storage: LatentStorage,
    /// Elements (video) or latent rows (image) of one page.
    page_units: u64,
    layout: Layout,
    /// Each admitted video request's canvas `(width, height)`, by slot,
    /// which its trajectory's raster is.
    canvases: HashMap<usize, (u32, u32)>,
    /// Orders batches addressing the banks across the lanes' streams.
    order: StreamOrder,
}

/// What [`LatentLedger::bind`] binds of one call.
pub(crate) struct Bound {
    pub(crate) banks: Banks,
    pub(crate) update: LatentUpdate,
    /// The trajectory's pages and size.
    pub(crate) params: LatentParams,
}

impl Bound {
    /// The binding as the executor reads it (`calls[i].latent` of the
    /// step): the banks, the step, the trajectory's pages in logical order,
    /// its latent units and its size.
    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "read": self.banks.read,
            "write": self.banks.write,
            "step": self.banks.step,
            "pages": self.params.page_table,
            "units": self.params.latent_units,
            "height": self.params.height,
            "width": self.params.width,
        })
    }
}

impl LatentLedger {
    /// The ledger of a video model: every request slot owns `pages`
    /// consecutive pages of `page_units` elements.
    pub(crate) fn new(
        pool: LatentPool,
        storage: LatentStorage,
        page_units: u64,
        pages: u64,
    ) -> Self {
        Self {
            pool,
            storage,
            page_units,
            layout: Layout::Slots { pages },
            canvases: HashMap::new(),
            order: StreamOrder::default(),
        }
    }

    /// The ledger of an image model, whose trajectories occupy the pages
    /// the scheduler assigns each call.
    pub(crate) fn scheduled(pool: LatentPool, storage: LatentStorage) -> Self {
        let page_units = pool.page_units() as u64;
        Self {
            pool,
            storage,
            page_units,
            layout: Layout::Scheduled,
            canvases: HashMap::new(),
            order: StreamOrder::default(),
        }
    }

    /// Note the canvas `(width, height)` of the video request admitted to
    /// `slot`.
    pub(crate) fn admit(&mut self, slot: usize, canvas: (u32, u32)) {
        if let Layout::Slots { .. } = self.layout {
            self.canvases.insert(slot, canvas);
        }
    }

    /// The pool and the storage of its banks, for trajectory transfers.
    pub(crate) fn parts(&mut self) -> (&mut LatentPool, &LatentStorage) {
        (&mut self.pool, &self.storage)
    }

    /// The pool.
    pub(crate) fn pool(&self) -> &LatentPool {
        &self.pool
    }

    /// Begin a batch's use of the banks, behind the batch that used them
    /// last on another stream.
    pub(crate) fn enter(&self, issue: impl Issue) -> Result<()> {
        self.order.wait(issue)
    }

    /// End a batch's use of the banks after what `issue` issued for it.
    pub(crate) fn leave(&mut self, issue: impl Issue) -> Result<()> {
        self.order.record(issue)
    }

    /// The banks a video trajectory call of `kind` reads and writes `slot`'s
    /// trajectory in, the step it evaluates, and its commit: a latent
    /// preparation initializes the trajectory, a denoising step advances it
    /// by one.
    pub(crate) fn bind(
        &self,
        slot: usize,
        call: &Call,
        kind: MediaCall,
    ) -> Result<(Banks, LatentUpdate)> {
        let Layout::Slots { pages } = self.layout else {
            return Err(Error::Invariant(
                "an image trajectory binds its scheduled pages".into(),
            ));
        };
        let canvas = *self
            .canvases
            .get(&slot)
            .ok_or_else(|| Error::Invalid("a denoising request has no canvas".into()))?;
        let params = self.slot_params(slot, call, canvas, pages);
        let bound = match kind {
            MediaCall::LatentPreparation => self.initial(slot, call, params)?,
            MediaCall::Denoising => {
                let step = self.pool.step_index(slot);
                self.step(slot, call, params, step + 1)?
            }
            other => {
                return Err(Error::Invalid(format!(
                    "{} writes no latent trajectory",
                    other.as_str()
                )));
            }
        };
        Ok((bound.banks, bound.update))
    }

    /// The banks and pages an image model's trajectory call reads and
    /// writes, the step it evaluates and its commit. `params` are the
    /// scheduler's pages of the call and `image` its request's admitted
    /// image.
    ///
    /// A latent preparation opens the trajectory in bank one at step zero;
    /// a denoising call advances the committed version by its next step of
    /// the admitted schedule; an image decoding reads the completed
    /// trajectory and releases it.
    pub(crate) fn bind_image(
        &self,
        slot: usize,
        call: &Call,
        params: &LatentParams,
        image: &ImageParams,
    ) -> Result<Bound> {
        if self.layout != Layout::Scheduled {
            return Err(Error::Invariant(
                "a video trajectory occupies its slot's pages".into(),
            ));
        }
        if (params.height, params.width) != (image.height, image.width) {
            return Err(Error::Invalid(
                "latent allocation disagrees with the admitted image".into(),
            ));
        }
        let steps = i64::from(image.steps);
        match call.kind {
            CallKind::Media(MediaCall::LatentPreparation) => {
                if call
                    .latent_output
                    .as_ref()
                    .is_none_or(|output| output.generation == 0)
                {
                    return Err(Error::Invalid(
                        "latent preparation requires a latent output".into(),
                    ));
                }
                self.initial(slot, call, params.clone())
            }
            CallKind::Media(MediaCall::Denoising) => {
                let generations = (
                    call.latent_input.as_ref().map(|input| input.generation),
                    call.latent_output.as_ref().map(|output| output.generation),
                );
                match generations {
                    (Some(input), Some(output)) if input > 0 && output > 0 && input != output => {}
                    _ => {
                        return Err(Error::Invalid(
                            "denoising requires distinct latent input and output generations"
                                .into(),
                        ));
                    }
                }
                // Each denoising call advances the slot's committed
                // trajectory by the next step of its admitted schedule.
                if self.pool.step_index(slot) >= steps {
                    return Err(Error::Invalid(
                        "denoising continues past the admitted schedule".into(),
                    ));
                }
                let generation = call
                    .latent_input
                    .as_ref()
                    .map_or(0, |input| i64::from(input.generation));
                self.step(slot, call, params.clone(), generation)
            }
            CallKind::Media(MediaCall::ImageDecoding) => {
                let input = call.latent_input.as_ref().ok_or_else(|| {
                    Error::Invalid("image decoding has no latent trajectory".into())
                })?;
                // The committed step is the end of the admitted schedule
                // once denoising completed.
                let step = self.pool.step_index(slot);
                if step != steps {
                    return Err(Error::Invalid(
                        "image decoding requires a completed trajectory".into(),
                    ));
                }
                let pages = self.pages(params)?;
                let version = TrajectoryVersion {
                    step_index: step,
                    generation: i64::from(input.generation),
                    units: i64::from(params.latent_units),
                    height: i64::from(params.height),
                    width: i64::from(params.width),
                };
                let read = self.pool.current_bank(slot, &version, &pages)?;
                Ok(Bound {
                    banks: Banks {
                        read,
                        write: read,
                        step,
                    },
                    update: LatentUpdate::for_call(slot, call, params, step)?,
                    params: params.clone(),
                })
            }
            other => Err(Error::Invalid(format!(
                "{} reads no latent trajectory",
                other.as_str()
            ))),
        }
    }

    /// The two banks, `[pages, page_units, width]` each, which the executor
    /// reads trajectories from and writes them to.
    pub(crate) fn banks(&self) -> Result<[StorageView; 2]> {
        Ok([self.storage.bank(0)?, self.storage.bank(1)?])
    }

    /// The pages, latent units and raster of a video `slot`'s trajectory:
    /// every slot owns the same run of consecutive pages after the padding
    /// page, so the rank names them itself; the raster is the request's
    /// admitted canvas.
    fn slot_params(
        &self,
        slot: usize,
        call: &Call,
        canvas: (u32, u32),
        pages: u64,
    ) -> LatentParams {
        let first = 1 + (slot as u64 - 1) * pages;
        LatentParams {
            request_key: call.request_key,
            call_id: call.call_id,
            page_table: (first..first + pages).map(|page| page as u32).collect(),
            latent_units: (pages * self.page_units) as u32,
            width: canvas.0,
            height: canvas.1,
        }
    }

    fn pages(&self, params: &LatentParams) -> Result<Vec<usize>> {
        let pages: Vec<i64> = params
            .page_table
            .iter()
            .map(|&page| i64::from(page))
            .collect();
        self.pool
            .validate_pages(&pages, i64::from(params.latent_units))
    }

    /// The bank a latent preparation initializes `slot`'s trajectory in, and
    /// its commit.
    fn initial(&self, slot: usize, call: &Call, params: LatentParams) -> Result<Bound> {
        let pages = self.pages(&params)?;
        self.pool.require_initial(slot, &pages)?;
        let update = LatentUpdate::for_call(slot, call, &params, 0)?;
        Ok(Bound {
            banks: Banks {
                read: 1,
                write: 1,
                step: 0,
            },
            update,
            params,
        })
    }

    /// The banks a denoising call reads and writes `slot`'s trajectory in,
    /// the step it evaluates, and its commit: it advances the step the pool
    /// committed by one. `generation` is the committed version's.
    fn step(
        &self,
        slot: usize,
        call: &Call,
        params: LatentParams,
        generation: i64,
    ) -> Result<Bound> {
        let pages = self.pages(&params)?;
        let step = self.pool.step_index(slot);
        let trajectory = TrajectoryVersion {
            step_index: step,
            generation,
            units: i64::from(params.latent_units),
            height: i64::from(params.height),
            width: i64::from(params.width),
        };
        let read = self.pool.current_bank(slot, &trajectory, &pages)?;
        let write = 1 - read;
        self.pool.require_writable(write, &pages)?;
        let update = LatentUpdate::for_call(slot, call, &params, step)?;
        Ok(Bound {
            banks: Banks { read, write, step },
            update,
            params,
        })
    }

    /// Validate trajectory commits before any becomes visible.
    pub(crate) fn validate(&self, updates: &[LatentUpdate]) -> Result<()> {
        self.pool.validate_updates(updates)
    }

    /// Make validated commits visible. A released trajectory's pending
    /// imports end: their reads are cancelled and their pages retire with
    /// them.
    pub(crate) fn apply(&mut self, updates: &[LatentUpdate]) -> Result<()> {
        let mut result = Ok(());
        for transfer in self.pool.apply_updates(updates) {
            result = result.and(transfer.read().cancel());
        }
        result
    }

    /// Release retired requests' pages and forget their canvases.
    pub(crate) fn release_slots(&mut self, slots: &[usize]) -> Result<()> {
        for slot in slots {
            self.canvases.remove(slot);
        }
        let slots: Vec<i64> = slots.iter().map(|&slot| slot as i64).collect();
        let mut result = Ok(());
        for transfer in self.pool.release_slots(&slots)? {
            result = result.and(transfer.read().cancel());
        }
        self.pool.reap();
        result
    }

    /// Forget the published versions of released buffers.
    pub(crate) fn release_buffers(&self, buffers: &HashSet<BufferId>) {
        self.pool
            .release_exports(&buffers.iter().copied().collect::<Vec<_>>());
    }

    /// Retire finished requests' published versions, but `retained` ones,
    /// and cancel their pending imports.
    pub(crate) fn release_requests(
        &self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<()> {
        let exports: Vec<BufferId> = self
            .pool
            .exports()
            .map(|source| source.buffer)
            .filter(|buffer| requests.contains(&buffer.owner) && !retained.contains(buffer))
            .collect();
        self.pool.release_exports(&exports);
        let mut result = Ok(());
        for transfer in self.pool.cancel_imports(requests) {
            result = result.and(transfer.read().cancel());
        }
        result
    }

    /// Whether `requests` hold no access to the pool any more.
    pub(crate) fn retirement_ready(&self, requests: &HashSet<RequestKey>) -> bool {
        self.pool.retirement_ready(requests)
    }

    /// Reclaim pages whose accesses retired.
    pub(crate) fn reap(&mut self) {
        self.pool.reap();
    }
}

/// The banks a trajectory call reads and writes, and the step it evaluates.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Banks {
    pub(crate) read: u8,
    pub(crate) write: u8,
    pub(crate) step: i64,
}

#[cfg(test)]
mod tests {
    use ipc::{CallId, CallKind, RequestId, TensorRef};

    use super::*;
    use crate::ffi::DType;
    use crate::memory::{LatentPlan, LocalMemory};

    /// The ledger of a rank whose two request slots own two pages of four
    /// units each.
    fn ledger() -> Result<LatentLedger> {
        let plan = LatentPlan {
            request_pool_size: 2,
            num_pages: 5,
            page_units: 4,
            latent_width: 1,
            dtype: DType::F32,
        };
        let storage = LatentStorage::allocate(&plan, None, None, "latent_banks", &mut LocalMemory)?;
        Ok(LatentLedger::new(LatentPool::new(2, 5, 4)?, storage, 4, 2))
    }

    fn call(slot: u64, kind: MediaCall) -> Call {
        Call::new(
            ipc::RequestKey::new(1, RequestId(slot), 1),
            CallId::new(1, 0),
            CallKind::Media(kind),
        )
    }

    /// A latent preparation opens a trajectory in bank one at step zero;
    /// each denoising step reads the bank holding the committed step and
    /// writes the other, and a released slot starts a new trajectory once
    /// its next request was admitted.
    #[test]
    fn a_trajectory_alternates_banks_from_its_committed_step() -> Result<()> {
        let mut ledger = ledger()?;
        let canvas = (1344, 768);
        let preparation = call(2, MediaCall::LatentPreparation);

        // A slot whose request was not admitted has no raster to denoise.
        assert!(
            ledger
                .bind(2, &preparation, MediaCall::LatentPreparation)
                .is_err()
        );
        ledger.admit(2, canvas);

        let (banks, update) = ledger.bind(2, &preparation, MediaCall::LatentPreparation)?;
        assert_eq!((banks.read, banks.write, banks.step), (1, 1, 0));
        // Slot two owns the two pages after slot one's and the padding page.
        assert_eq!(update.params.page_table, [3, 4]);
        assert_eq!(update.params.latent_units, 8);
        assert_eq!((update.params.width, update.params.height), canvas);
        ledger.validate(std::slice::from_ref(&update))?;
        ledger.apply(&[update])?;

        let denoising = call(2, MediaCall::Denoising);
        for (step, read) in [(0, 1), (1, 0), (2, 1)] {
            let (banks, update) = ledger.bind(2, &denoising, MediaCall::Denoising)?;
            assert_eq!(
                (banks.read, banks.write, banks.step),
                (read, 1 - read, step)
            );
            ledger.validate(std::slice::from_ref(&update))?;
            ledger.apply(&[update])?;
        }

        // A trajectory in use cannot be opened again until its slot retires,
        // and a retired slot forgets its request's canvas.
        assert!(
            ledger
                .bind(2, &preparation, MediaCall::LatentPreparation)
                .is_err()
        );
        ledger.release_slots(&[2])?;
        assert!(
            ledger
                .bind(2, &preparation, MediaCall::LatentPreparation)
                .is_err()
        );
        ledger.admit(2, canvas);
        let (banks, _) = ledger.bind(2, &preparation, MediaCall::LatentPreparation)?;
        assert_eq!((banks.read, banks.write, banks.step), (1, 1, 0));
        Ok(())
    }

    /// The ledger of an image rank: two slots, five usable pages of two
    /// latent rows of three values.
    fn image_ledger() -> Result<LatentLedger> {
        let plan = LatentPlan {
            request_pool_size: 2,
            num_pages: 6,
            page_units: 2,
            latent_width: 3,
            dtype: DType::F32,
        };
        let storage = LatentStorage::allocate(&plan, None, None, "latent_banks", &mut LocalMemory)?;
        Ok(LatentLedger::scheduled(LatentPool::new(2, 6, 2)?, storage))
    }

    fn latent(owner: RequestKey, generation: u32) -> TensorRef {
        TensorRef {
            request_key: owner,
            producer_call_id: CallId::new(1, 0),
            output_index: 0,
            generation,
            dtype: ipc::DType::F32,
            shape_bound: ipc::ShapeBound { dims: Vec::new() },
        }
    }

    fn image_call(kind: MediaCall, input: Option<u32>, output: Option<u32>) -> Call {
        let mut call = call(1, kind);
        let owner = call.request_key;
        call.latent_input = input.map(|generation| latent(owner, generation));
        call.latent_output = output.map(|generation| latent(owner, generation));
        call
    }

    fn image(steps: u16) -> ImageParams {
        ImageParams {
            steps,
            cfg_text_scale: 1.0,
            cfg_img_scale: 1.0,
            cfg_renorm_type: ipc::CfgRenorm::None,
            cfg_renorm_min: 0.0,
            cfg_interval: (0.0, 1.0),
            timestep_shift: 0.0,
            height: 16,
            width: 16,
            seed: None,
        }
    }

    /// An image trajectory lives in the pages its scheduler parameters
    /// name: its preparation opens it, each denoising step of the admitted
    /// schedule advances it, and its decoding reads the completed version
    /// and releases it; a size other than the admitted image's, a step past
    /// the schedule and a decode before its end are refused.
    #[test]
    fn an_image_trajectory_follows_its_scheduled_pages() -> Result<()> {
        let mut ledger = image_ledger()?;
        let admitted = image(2);
        let preparation = image_call(MediaCall::LatentPreparation, None, Some(1));
        let params = LatentParams {
            request_key: preparation.request_key,
            call_id: preparation.call_id,
            page_table: vec![4, 2],
            latent_units: 3,
            height: 16,
            width: 16,
        };
        let mut other = params.clone();
        other.width = 32;
        assert!(
            ledger
                .bind_image(1, &preparation, &other, &admitted)
                .is_err()
        );

        let bound = ledger.bind_image(1, &preparation, &params, &admitted)?;
        assert_eq!(
            (bound.banks.read, bound.banks.write, bound.banks.step),
            (1, 1, 0)
        );
        assert_eq!(bound.to_json()["pages"], serde_json::json!([4, 2]));
        ledger.validate(std::slice::from_ref(&bound.update))?;
        ledger.apply(&[bound.update])?;

        // A decode before the schedule's end reads no completed version.
        let decoding = image_call(MediaCall::ImageDecoding, Some(3), None);
        assert!(ledger.bind_image(1, &decoding, &params, &admitted).is_err());

        for (step, (input, output), read) in [(0, (1, 2), 1), (1, (2, 3), 0)] {
            let denoising = image_call(MediaCall::Denoising, Some(input), Some(output));
            let bound = ledger.bind_image(1, &denoising, &params, &admitted)?;
            assert_eq!(
                (bound.banks.read, bound.banks.write, bound.banks.step),
                (read, 1 - read, step)
            );
            ledger.validate(std::slice::from_ref(&bound.update))?;
            ledger.apply(&[bound.update])?;
        }
        let past = image_call(MediaCall::Denoising, Some(3), Some(4));
        assert!(ledger.bind_image(1, &past, &params, &admitted).is_err());

        let bound = ledger.bind_image(1, &decoding, &params, &admitted)?;
        assert_eq!((bound.banks.read, bound.banks.step), (1, 2));
        ledger.validate(std::slice::from_ref(&bound.update))?;
        ledger.apply(&[bound.update])?;
        // The released slot opens a new trajectory once the pool reaped it.
        ledger.reap();
        assert_eq!(ledger.pool().step_index(1), 0);
        let bound = ledger.bind_image(1, &preparation, &params, &admitted)?;
        assert_eq!(bound.banks.step, 0);
        Ok(())
    }
}
