//! Denoising an image batch: the active trajectories of its denoising calls,
//! their initial latents and guidance plans, and the solver step each call
//! covers, which evaluates their rows grouped by image size and integrates
//! their predictions, each denoising resumed between the jobs it records
//! ([`DenoiseResume`], [`PlanResume`]).

use std::collections::HashMap;
use std::sync::Arc;

use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::ImageParams;
use uniserve_ipc::{ForwardStats, LatentParams, RequestKey};

use super::{DenoiseRow, IntegrateRow};
use crate::cuda_graph::GraphMode;
use crate::executor::image::ImageParts;
use crate::executor::image::{ImageBatch, Planned, Progress, Resume};
use crate::executor::transfer::latents;
use crate::lane::{Completion, Job};
use crate::parallel::ExpertUnion;
use crate::request::diffusion::{BranchSet, BranchSource, Prefix};
use crate::request::diffusion::{Coordinates, ForwardSlot, ImageTrajectory};
use crate::runner::causal_lm::{CausalLMRunners, Step};
use crate::runner::denoiser::image_model::ImageDenoiserLayout;
use crate::storage::LatentUpdate;
use crate::storage::Scratch;
use crate::storage::TrajectoryVersion;
use crate::stream::Issue;
use crate::{Error, Result};

/// The denoiser serving a batch, with its device and stream.
pub(crate) fn image_placement(
    image: &ImageParts<'_>,
) -> Result<(Option<DeviceId>, Option<Stream>, usize)> {
    let runner = image.denoiser().ok_or_else(|| {
        Error::Unsupported("image trajectories are served by an image denoiser".into())
    })?;
    let stream = runner.stream().cloned();
    Ok((runner.device().cuda(), stream, 0))
}

/// The denoising rows of one image size: the size, its rows and, per
/// stepping image, its position among the batch's trajectories and its
/// first row.
type SizeGroup = ((u32, u32), Vec<DenoiseRow>, Vec<(usize, usize)>);

/// The stepping trajectories of one solver step: each one's position among
/// the batch's trajectories, its step, its guidance branches and its
/// timestep.
type Stepping = Vec<(usize, u32, BranchSet, f32)>;

/// The guidance prefixes a step misses: the position of the trajectory, the
/// slot the prefix is written to and its tokens.
type Fills = Vec<(usize, u32, Arc<[u32]>)>;

/// What a denoising keeps between the jobs it records.
pub(crate) struct Denoising {
    /// Each active call's trajectory and admitted image, with its placed
    /// samples in `scratch`.
    trajectories: Vec<(Trajectory, ImageParams)>,
    scratch: Vec<Scratch>,
    /// The forward rows of each call, by call index.
    descriptors: HashMap<usize, Vec<ForwardSlot>>,
    layout: ImageDenoiserLayout,
    component: String,
    stats: ForwardStats,
}

/// A denoising between two of its jobs: the job on the lane ends with the
/// text runner's evaluation of the guidance prefixes its step misses.
pub(crate) struct DenoiseResume {
    state: Denoising,
    stepping: Stepping,
    fills: Fills,
    /// The evaluation, completed from the job's completion.
    step: Step,
}

/// A denoising between two of its jobs: the job on the lane ends with the
/// plan calls of the trajectories the denoising opens (one this rank did
/// not prepare, or one resumed at another size). Their plans say which
/// branches each step evaluates, so the steps are recorded once they are
/// known.
///
/// A rank opens a trajectory when it prepares its latent, so a denoising
/// reaches this boundary for a trajectory whose committed version was
/// transferred from the rank that prepared it: its head placed the
/// preparation and the denoising on different workers. The engine's
/// scheduler keeps a request's image calls on the replica that holds its
/// state, so its own plans open every trajectory at its preparation; the
/// engine's multiprocess tests place the two on different workers.
pub(crate) struct PlanResume {
    /// Each active call's trajectory, admitted image and conditioning
    /// coordinates.
    trajectories: Vec<(Trajectory, ImageParams, Coordinates)>,
    /// The forward rows of each call, by call index.
    descriptors: HashMap<usize, Vec<ForwardSlot>>,
    /// The trajectories to open, in the order their plan calls were
    /// recorded.
    plans: Vec<Planned>,
}

/// One active trajectory call of a batch.
struct Trajectory {
    index: usize,
    slot: usize,
    params: LatentParams,
    /// Steps in the admitted schedule.
    steps: u32,
    /// The slot's committed step index, from the latent pool's ledger: the
    /// step a denoising call advances.
    step_index: u32,
}

impl ImageBatch<'_> {
    /// The latent allocation of call `index`.
    fn latent_params(&self, index: usize) -> Result<LatentParams> {
        let call = &self.plan.calls[index];
        self.plan
            .latent_params
            .iter()
            .find(|params| params.request_key == call.request_key && params.call_id == call.call_id)
            .cloned()
            .ok_or_else(|| Error::Invalid("trajectory call has no latent allocation".into()))
    }

    /// Validate call `index`'s allocation against its admitted image and
    /// return the admission.
    fn trajectory(&self, index: usize) -> Result<(Trajectory, uniserve_ipc::ImageParams)> {
        let call = &self.plan.calls[index];
        let request = self.requests.get(call.request_key.request_id.0)?;
        let image =
            request.admission().image.clone().ok_or_else(|| {
                Error::Invalid("image trajectory requires image parameters".into())
            })?;
        let params = self.latent_params(index)?;
        let layout = *self
            .parts
            .denoiser()
            .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?
            .layout()?;
        if (params.height, params.width) != (image.height, image.width)
            || params.latent_units != layout.latent_rows(params.height, params.width)?
        {
            return Err(Error::Invalid(
                "latent allocation disagrees with the admitted image".into(),
            ));
        }
        let slot = self.batch.calls[index].slot;
        let step_index = u32::try_from(self.parts.latents()?.pool.step_index(slot))
            .map_err(|_| Error::Invariant("a committed latent step exceeds u32".into()))?;
        Ok((
            Trajectory {
                index,
                slot,
                params,
                steps: u32::from(image.steps),
                step_index,
            },
            image,
        ))
    }

    /// The request's own conditioning prefix cache: the export it published and its
    /// visible extent.
    fn conditioning(&mut self, index: usize) -> Result<Coordinates> {
        let call = &self.plan.calls[index];
        let buffer = call.cache_input.ok_or_else(|| {
            Error::Invalid("image trajectory requires its conditioning prefix cache".into())
        })?;
        let slot = self.batch.calls[index].slot as u32;
        let visible = self.batch.calls[index].output.progress.cache_visible_len;
        let prefix_cache = self.storage.prefix_cache.as_mut().ok_or_else(|| {
            Error::Unsupported("image trajectories require a prefix cache".into())
        })?;
        let coordinates = prefix_cache.tables().coordinates(slot, visible)?;
        prefix_cache.validate_conditioning(call.request_key, buffer, slot, visible)?;
        Ok(coordinates.into())
    }

    /// Export the version call `index` just wrote into `bank` (at
    /// `step_index`) to other ranks and workers, when the rank publishes there
    /// and reports the component's results. Readers on this rank use the
    /// committed trajectory directly.
    fn publish_latent(
        &mut self,
        index: usize,
        bank: u8,
        step_index: u32,
        params: &LatentParams,
        stream: impl Issue,
    ) -> Result<()> {
        let call = &self.plan.calls[index];
        let Some(product) = &call.latent_output else {
            return Ok(());
        };
        if !self.batch.reports || !self.storage.transports.exports_remotely() {
            return Ok(());
        }
        let (pool, storage) = self.parts.latent_pool()?;
        let export = latents::issue_export(
            pool,
            storage,
            &self.storage.transports,
            call,
            product,
            self.batch.calls[index].slot,
            params,
            bank,
            step_index,
            stream,
        )?;
        // The batch's commit locates it once this work was performed.
        self.latent_exports.push(export.buffer());
        self.exports.push((index, export));
        Ok(())
    }

    /// Open the batch's use of the latent pool, behind its previous user
    /// in what `stream` issues, and place its trajectories.
    fn place(&mut self, trajectories: &[Trajectory], stream: impl Issue) -> Result<Vec<Scratch>> {
        let latents = self.parts.latents()?;
        latents.enter(stream)?;
        self.latents_entered = true;
        let params: Vec<&LatentParams> = trajectories.iter().map(|value| &value.params).collect();
        latents.place(&params)
    }

    /// Record into `job` the calls a trajectory of request `key` opens
    /// with: its guidance plan, and the prefix of each source one of its
    /// guidance branches may condition on. The conditioning prefix is the
    /// request's own context and the negative prefix the admitted negative
    /// tokens as they are ([`BranchSource::admitted`]); the model frames and
    /// tokenizes the others.
    /// [`Self::planned_trajectory`] opens the trajectory from the job's
    /// completion.
    fn plan_trajectory(
        &self,
        slot: usize,
        key: RequestKey,
        size: (u32, u32),
        image: &ImageParams,
        job: &Job,
    ) -> Result<Planned> {
        let denoiser = self
            .parts
            .denoiser()
            .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?;
        denoiser.plan(image, job)?;

        let negative = self
            .requests
            .get(key.request_id.0)?
            .admission()
            .generation
            .as_ref()
            .map(|generation| generation.negative_token_ids.clone())
            .unwrap_or_default();
        let mut prefixes: Vec<(BranchSource, Option<Prefix>)> = Vec::new();
        for source in [
            BranchSource::Conditioning,
            BranchSource::NegativeOrStart,
            denoiser.layout()?.image_unconditional,
        ] {
            if prefixes.iter().any(|(known, _)| *known == source) {
                continue;
            }
            let given = source.admitted(&negative);
            if given.is_none() {
                self.parts.policy.ask_prefix(source, job);
            }
            prefixes.push((source, given));
        }
        Ok(Planned {
            slot,
            key,
            size,
            params: Arc::new(image.clone()),
            prefixes,
        })
    }

    /// The trajectory `planned` opens and its slot, from the `completion`
    /// of the job that made its calls.
    pub(crate) fn planned_trajectory(
        &self,
        planned: Planned,
        completion: &mut Completion,
    ) -> Result<(usize, ImageTrajectory)> {
        let Planned {
            slot,
            key,
            size,
            params,
            prefixes,
        } = planned;
        let denoiser = self
            .parts
            .denoiser()
            .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?;
        let plan = denoiser.planned(params, completion.values()?)?;
        let prefixes = prefixes
            .into_iter()
            .map(|(source, given)| {
                let prefix = match given {
                    Some(prefix) => prefix,
                    None => self.parts.policy.prefix(&completion.returned()?)?,
                };
                Ok((source, prefix))
            })
            .collect::<Result<_>>()?;
        Ok((slot, ImageTrajectory::new(key, size, plan, prefixes)))
    }

    /// Draw each trajectory's seeded initial latent and stage it as its
    /// first committed version.
    pub(crate) fn prepare_latents(&mut self) -> Result<ForwardStats> {
        // The forward records into the batch's job, taken for its length so
        // that the batch's owners stay borrowable beside it.
        let job = std::mem::take(&mut self.job);
        let stats = self.record_latents(&job);
        self.job = job;
        stats
    }

    fn record_latents(&mut self, job: &Job) -> Result<ForwardStats> {
        let stream = self.batch.stream.clone();
        let issue = job.on(stream.as_ref());
        let mut trajectories = Vec::new();
        for index in self.active() {
            let call = &self.plan.calls[index];
            // The pool refuses a slot whose trajectory is still open
            // (`require_initial`, and the commit's `validate_updates`).
            let (trajectory, image) = self.trajectory(index)?;
            // The image's noise comes from the admitted image seed and the
            // index the engine assigned the image it initializes.
            let image_index = call
                .image_index
                .ok_or_else(|| Error::Invalid("latent preparation names no image index".into()))?;
            if call
                .latent_output
                .as_ref()
                .is_none_or(|output| output.generation == 0)
            {
                return Err(Error::Invalid(
                    "latent preparation requires a latent output".into(),
                ));
            }
            self.conditioning(index)?;
            trajectories.push((trajectory, image, image_index));
        }
        if trajectories.is_empty() {
            return Ok(ForwardStats::default());
        }
        let scratch = {
            let values: Vec<Trajectory> = trajectories
                .iter()
                .map(|(value, ..)| Trajectory {
                    index: value.index,
                    slot: value.slot,
                    params: value.params.clone(),
                    steps: value.steps,
                    step_index: value.step_index,
                })
                .collect();
            self.place(&values, issue)?
        };

        for ((trajectory, image, image_index), scratch) in trajectories.iter().zip(&scratch) {
            let call = &self.plan.calls[trajectory.index];
            let denoiser = self.parts.denoiser().ok_or_else(|| {
                Error::Unsupported("image trajectories require a denoiser".into())
            })?;
            let planned = self.plan_trajectory(
                trajectory.slot,
                call.request_key,
                (trajectory.params.height, trajectory.params.width),
                image,
                job,
            )?;
            let latents = self.parts.latents()?;
            let page_units = latents.pool.page_units();
            latents
                .pool
                .require_initial(trajectory.slot, &scratch.pages)?;
            denoiser.initial_latent(
                scratch.sample,
                scratch.units,
                scratch.padded(page_units),
                trajectory.params.height,
                trajectory.params.width,
                image.seed.unwrap_or(0),
                *image_index,
                job,
            )?;
            latents.scatter(1, scratch, issue)?;
            self.publish_latent(trajectory.index, 1, 0, &trajectory.params, issue)?;
            // The trajectory opens with the batch's commit, once its
            // calls were made.
            self.plans.push(planned);
            self.updates.push((
                trajectory.index,
                LatentUpdate::for_call(trajectory.slot, call, &trajectory.params, 0)?,
            ));
        }
        Ok(ForwardStats::default())
    }

    /// Install the block tables of every slot the active calls' forward
    /// rows address, reset their new units and retain the extents the rows
    /// read and write until the batch's device work retires. Rows on slots
    /// other than a call's own hold its guidance prefixes.
    fn reserve_rows(
        &mut self,
        active: &[usize],
        stream: impl Issue,
    ) -> Result<HashMap<usize, Vec<ForwardSlot>>> {
        let plan = self.plan;
        let forward = &plan.forward;
        let mut descriptors: HashMap<usize, Vec<ForwardSlot>> = HashMap::new();
        for (row, &call) in forward.call_indices.iter().enumerate() {
            let index = call as usize;
            if index >= plan.calls.len() {
                return Err(Error::Invalid("forward row names no call".into()));
            }
            if !active.contains(&index) {
                continue;
            }
            descriptors.entry(index).or_default().push(ForwardSlot {
                slot: forward.request_pool_indices[row],
                seq_len: forward.seq_lens[row],
                query_len: forward.query_lens[row],
            });
        }
        let slots: std::collections::HashSet<u32> = descriptors
            .values()
            .flatten()
            .map(|row| row.slot)
            .chain(
                active
                    .iter()
                    .map(|&index| self.batch.calls[index].slot as u32),
            )
            .collect();
        let units = self.parts.denoisers.units.as_ref();
        let prefix_cache = self.storage.prefix_cache.as_mut().ok_or_else(|| {
            Error::Unsupported("image trajectories require a prefix cache".into())
        })?;
        let assignments: Vec<_> = plan
            .block_tables
            .iter()
            .filter(|table| slots.contains(&table.request_pool_idx))
            .cloned()
            .collect();
        prefix_cache.install(&assignments, stream)?;
        let new_units: Vec<u32> = plan
            .new_cache_units
            .iter()
            .filter(|allocation| slots.contains(&allocation.request_pool_idx))
            .flat_map(|allocation| allocation.unit_ids.iter().map(|unit| unit.0))
            .collect();
        if !new_units.is_empty() {
            let column = units.ok_or_else(|| {
                Error::Unsupported("image trajectories reset cache units on their stream".into())
            })?;
            prefix_cache.recycle(&new_units, column, stream)?;
        }

        for (row, &call) in forward.call_indices.iter().enumerate() {
            let index = call as usize;
            if !active.contains(&index) {
                continue;
            }
            let key = plan.calls[index].request_key;
            let slot = forward.request_pool_indices[row];
            let own = self.batch.calls[index].slot as u32;
            if slot != own {
                let prefix = forward.seq_lens[row] - forward.query_lens[row];
                if prefix > prefix_cache.tables().allocated_length(slot) {
                    return Err(Error::Invalid(
                        "forward row exceeds alternative-prefix capacity".into(),
                    ));
                }
                prefix_cache.retain_prefix(key, slot);
            }
            // Readers retain their prefix, writers also their query tokens.
            let length = u64::from(if forward.write_cache[row] {
                forward.seq_lens[row]
            } else {
                forward.seq_lens[row] - forward.query_lens[row]
            });
            prefix_cache.retain(key, slot, length, &self.batch.retirement)?;
        }
        Ok(descriptors)
    }

    /// Advance every active trajectory by its one solver step. Missing
    /// guidance prefixes are evaluated by `text` first, as expert steps on a
    /// rank with `experts`.
    pub(crate) fn denoise(
        &mut self,
        text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        // The forward records into the batch's job, taken for its length so
        // that the batch's owners stay borrowable beside it.
        let job = std::mem::take(&mut self.job);
        let progress = self.begin_denoise(&job, text, experts);
        self.job = job;
        progress
    }

    /// Continue the denoising from the `completion` of the job that ended
    /// with an evaluation of guidance prefixes.
    pub(crate) fn resume_denoise(
        &mut self,
        resume: DenoiseResume,
        completion: &mut Completion,
        text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        let job = std::mem::take(&mut self.job);
        let DenoiseResume {
            state,
            stepping,
            fills,
            step,
        } = resume;
        let progress = self.denoise_steps(
            &job,
            state,
            Some((stepping, fills, step, completion)),
            text,
            experts,
        );
        self.job = job;
        progress
    }

    /// Continue the denoising from the `completion` of the job that ended
    /// with the plan calls of the trajectories it opens.
    pub(crate) fn resume_planned(
        &mut self,
        resume: PlanResume,
        completion: &mut Completion,
        text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        let PlanResume {
            trajectories,
            descriptors,
            plans,
        } = resume;
        for planned in plans {
            let (slot, trajectory) = self.planned_trajectory(planned, completion)?;
            self.parts.trajectories.open(slot, trajectory);
        }

        let job = std::mem::take(&mut self.job);
        let progress = self.gather_denoise(&job, trajectories, descriptors, text, experts);
        self.job = job;
        progress
    }

    /// Record what precedes the solver step: the wait on the text runners'
    /// order, the reservation of the rows' slots, the resumption of each
    /// trajectory and the gather of its committed version. The denoising is
    /// one job unless it opens a trajectory, whose plan call then ends the
    /// job, or the step misses guidance prefixes: the job then ends with the
    /// text runner's evaluation of them, which is completed before the step
    /// is recorded.
    fn begin_denoise(
        &mut self,
        job: &Job,
        text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        let plan = self.plan;
        let active = self.active();
        if active.is_empty() {
            return Ok(Progress::done(ForwardStats::default()));
        }
        let stream = self.batch.stream.clone();
        if let Some(text) = text.as_deref() {
            // Conditioning K/V and earlier prefixes were written by text
            // batches, possibly on another stream.
            text.order().wait(job.on(stream.as_ref()))?;
        }
        let descriptors = self.reserve_rows(&active, job.on(stream.as_ref()))?;

        // Validate every call and find the trajectories to open.
        let mut trajectories = Vec::with_capacity(active.len());
        let mut plans = Vec::new();
        for &index in &active {
            let call = &plan.calls[index];
            let (trajectory, image) = self.trajectory(index)?;
            let params = &trajectory.params;
            let generations = (
                call.latent_input.as_ref().map(|input| input.generation),
                call.latent_output.as_ref().map(|output| output.generation),
            );
            match generations {
                (Some(input), Some(output)) if input > 0 && output > 0 && input != output => {}
                _ => {
                    return Err(Error::Invalid(
                        "denoising requires distinct latent input and output generations".into(),
                    ));
                }
            }
            // Each denoising call advances the slot's committed trajectory
            // by the next step of its admitted schedule.
            if trajectory.step_index >= trajectory.steps {
                return Err(Error::Invalid(
                    "denoising continues past the admitted schedule".into(),
                ));
            }
            let conditioning = self.conditioning(index)?;
            let size = (params.height, params.width);
            let key = call.request_key;
            let reopen = self
                .parts
                .trajectories
                .get(trajectory.slot, key)
                .is_none_or(|open| open.size != size);
            if reopen {
                plans.push(self.plan_trajectory(trajectory.slot, key, size, &image, job)?);
            }
            trajectories.push((trajectory, image, conditioning));
        }

        // The steps of an opened trajectory are recorded from its plan: the
        // job ends with the plan calls.
        if !plans.is_empty() {
            return Ok(Progress::Boundary(Resume::Plan(Box::new(PlanResume {
                trajectories,
                descriptors,
                plans,
            }))));
        }
        self.gather_denoise(job, trajectories, descriptors, text, experts)
    }

    /// Record the resumption of each trajectory, whose guidance plan is
    /// open, the gather of its committed version and the solver step.
    fn gather_denoise(
        &mut self,
        job: &Job,
        trajectories: Vec<(Trajectory, ImageParams, Coordinates)>,
        descriptors: HashMap<usize, Vec<ForwardSlot>>,
        text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        let plan = self.plan;
        let stream = self.batch.stream.clone();
        let trajectories: Vec<(Trajectory, ImageParams)> = trajectories
            .into_iter()
            .map(|(trajectory, image, conditioning)| {
                let key = plan.calls[trajectory.index].request_key;
                if let Some(open) = self.parts.trajectories.get(trajectory.slot, key) {
                    open.begin_call(conditioning);
                }
                (trajectory, image)
            })
            .collect();
        let scratch = {
            let values: Vec<Trajectory> = trajectories
                .iter()
                .map(|(value, _)| Trajectory {
                    index: value.index,
                    slot: value.slot,
                    params: value.params.clone(),
                    steps: value.steps,
                    step_index: value.step_index,
                })
                .collect();
            self.place(&values, job.on(stream.as_ref()))?
        };

        // Gather each committed version into the step workspace.
        {
            let latents = self.parts.latents()?;
            for ((trajectory, _), scratch) in trajectories.iter().zip(&scratch) {
                let call = &plan.calls[trajectory.index];
                let params = &trajectory.params;
                let generation = call
                    .latent_input
                    .as_ref()
                    .map_or(0, |input| i64::from(input.generation));
                let committed = TrajectoryVersion {
                    step_index: i64::from(trajectory.step_index),
                    generation,
                    units: i64::from(params.latent_units),
                    height: i64::from(params.height),
                    width: i64::from(params.width),
                };
                let bank =
                    latents
                        .pool
                        .current_bank(trajectory.slot, &committed, &scratch.pages)?;
                latents.gather(bank, scratch, job.on(stream.as_ref()))?;
            }
        }

        let layout = *self
            .parts
            .denoiser()
            .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?
            .layout()?;
        let component = trajectories
            .first()
            .map(|(trajectory, _)| plan.calls[trajectory.index].component.clone())
            .unwrap_or_default();
        self.denoise_steps(
            job,
            Denoising {
                trajectories,
                scratch,
                descriptors,
                layout,
                component,
                stats: ForwardStats::default(),
            },
            None,
            text,
            experts,
        )
    }

    /// Record the solver step of every trajectory, then the release of
    /// finished trajectories' guidance prefixes and the record on the text
    /// runners' order. `resumed` is the step whose guidance prefixes the
    /// last job evaluated, with that job's completion.
    fn denoise_steps(
        &mut self,
        job: &Job,
        state: Denoising,
        resumed: Option<(Stepping, Fills, Step, &mut Completion)>,
        mut text: Option<&mut CausalLMRunners>,
        experts: Option<&Arc<ExpertUnion>>,
    ) -> Result<Progress> {
        let plan = self.plan;
        let stream = self.batch.stream.clone();
        let Denoising {
            trajectories,
            scratch,
            descriptors,
            layout,
            component,
            mut stats,
        } = state;
        let stepping = if let Some((stepping, fills, step, completion)) = resumed {
            // The last job evaluated this step's guidance prefixes.
            let text = text.as_deref_mut().ok_or_else(|| {
                Error::Unsupported("guidance prefixes require the text image".into())
            })?;
            text.complete_prefill(step, completion)?;
            text.order().wait(job.on(stream.as_ref()))?;
            for (position, slot, tokens) in &fills {
                let (trajectory, _) = &trajectories[*position];
                let key = plan.calls[trajectory.index].request_key;
                if let Some(open) = self.parts.trajectories.get(trajectory.slot, key) {
                    open.prefilled(*slot, tokens.len() as u64);
                }
            }
            stepping
        } else {
            // Timesteps and guidance branches of each trajectory's step.
            let mut stepping: Stepping = Vec::new();
            let mut fills: Fills = Vec::new();
            for (position, (trajectory, _)) in trajectories.iter().enumerate() {
                let step = trajectory.step_index;
                let key = plan.calls[trajectory.index].request_key;
                let call_rows = descriptors
                    .get(&trajectory.index)
                    .map(Vec::as_slice)
                    .unwrap_or_default();
                let prefix_cache = self.storage.prefix_cache.as_ref().ok_or_else(|| {
                    Error::Unsupported("image trajectories require a prefix cache".into())
                })?;
                let tables = prefix_cache.tables();
                let open = self
                    .parts
                    .trajectories
                    .get(trajectory.slot, key)
                    .ok_or_else(|| Error::Invariant("an image trajectory closed".into()))?;
                let (set, time) = open.plan.step(step as usize)?;
                let filled = open.resolve_branches(
                    set,
                    layout.image_unconditional,
                    call_rows,
                    |slot, visible| tables.coordinates(slot, visible).map(Coordinates::from),
                )?;
                fills.extend(
                    filled
                        .into_iter()
                        .map(|fill| (position, fill.slot, fill.tokens)),
                );
                self.parts.latents()?.fill_timestep(
                    trajectory.slot as u32,
                    time,
                    job.on(stream.as_ref()),
                )?;
                stepping.push((position, step, set, time));
            }

            // Missing guidance prefixes are written before the step reads
            // them.
            if !fills.is_empty() {
                let text = text.as_deref_mut().ok_or_else(|| {
                    Error::Unsupported("guidance prefixes require the text image".into())
                })?;
                let prefix_cache = self.storage.prefix_cache.as_mut().ok_or_else(|| {
                    Error::Unsupported("image trajectories require a prefix cache".into())
                })?;
                let prefixes: Vec<(u32, &[u32])> = fills
                    .iter()
                    .map(|(_, slot, tokens)| (*slot, &tokens[..]))
                    .collect();
                // The text runner's evaluation of the prefixes is recorded
                // into this forward's job, behind the block table installs
                // it reads; the job ends with it, because the evaluation
                // is completed from the job that performed it, and the
                // step follows in the next job.
                let step = text.prefill_cache(experts, &component, &prefixes, prefix_cache, job)?;
                return Ok(Progress::Boundary(Resume::Denoise(Box::new(
                    DenoiseResume {
                        state: Denoising {
                            trajectories,
                            scratch,
                            descriptors,
                            layout,
                            component,
                            stats,
                        },
                        stepping,
                        fills,
                        step,
                    },
                ))));
            }
            stepping
        };
        self.denoise_step(job, &trajectories, &scratch, &stepping, &mut stats)?;

        // A finished trajectory's guidance prefixes retire with it.
        for (trajectory, _) in &trajectories {
            if trajectory.step_index + 1 < trajectory.steps {
                continue;
            }
            let own = trajectory.slot as u32;
            let mut slots: Vec<u32> = descriptors
                .get(&trajectory.index)
                .map(|rows| {
                    rows.iter()
                        .map(|row| row.slot)
                        .filter(|&slot| slot != own)
                        .collect()
                })
                .unwrap_or_default();
            slots.sort_unstable();
            slots.dedup();
            if slots.is_empty() {
                continue;
            }
            let key = plan.calls[trajectory.index].request_key;
            let prefix_cache = self.storage.prefix_cache.as_mut().ok_or_else(|| {
                Error::Unsupported("image trajectories require a prefix cache".into())
            })?;
            prefix_cache.release_prefixes(key, &slots, job.on(stream.as_ref()))?;
        }
        if let Some(text) = text {
            text.order_mut().record(job.on(stream.as_ref()))?;
        }
        Ok(Progress::done(stats))
    }

    /// Record one solver step of the stepping trajectories: the rows of
    /// their guidance branches, grouped by image size, each group's
    /// prediction and integration, and the scatter of each successor into
    /// the other bank with its export.
    fn denoise_step(
        &mut self,
        job: &Job,
        trajectories: &[(Trajectory, ImageParams)],
        scratch: &[Scratch],
        stepping: &Stepping,
        stats: &mut ForwardStats,
    ) -> Result<()> {
        let plan = self.plan;
        let stream = self.batch.stream.clone();
        // One denoising row per branch of every stepping image; images
        // of one size share a numerical call.
        let mut groups: Vec<SizeGroup> = Vec::new();
        for &(position, _, set, time) in stepping {
            let (trajectory, _) = &trajectories[position];
            let key = plan.calls[trajectory.index].request_key;
            let logical = self.batch.calls[trajectory.index]
                .output
                .progress
                .logical_position;
            let open = self
                .parts
                .trajectories
                .get(trajectory.slot, key)
                .ok_or_else(|| Error::Invariant("an image trajectory closed".into()))?;
            let rows = open.branch_rows(set, logical)?;
            let size = (trajectory.params.height, trajectory.params.width);
            let group = match groups.iter().position(|(value, ..)| *value == size) {
                Some(group) => group,
                None => {
                    groups.push((size, Vec::new(), Vec::new()));
                    groups.len() - 1
                }
            };
            let (_, values, owners) = &mut groups[group];
            owners.push((position, values.len()));
            for (_, coordinates, temporal) in rows {
                values.push(DenoiseRow {
                    sample: scratch[position].sample,
                    units: scratch[position].units,
                    height: size.0,
                    width: size.1,
                    slot: coordinates.slot,
                    prefix: u32::try_from(coordinates.visible).map_err(|_| {
                        Error::Invalid("guidance prefix exceeds the token range".into())
                    })?,
                    temporal,
                    timestep: time,
                });
            }
        }
        for (_, rows, owners) in &groups {
            let prefix_cache = self.storage.prefix_cache.as_mut().ok_or_else(|| {
                Error::Unsupported("image trajectories require a prefix cache".into())
            })?;
            let denoiser = self.parts.denoiser_mut().ok_or_else(|| {
                Error::Unsupported("image trajectories require a denoiser".into())
            })?;
            let predicted = denoiser.predict(rows, prefix_cache, None, job)?;
            let mode = predicted.mode.as_str();
            *stats.mode_counts.entry(mode.into()).or_default() += 1;
            if predicted.mode == GraphMode::Replay {
                stats.cuda_graph_replays += 1;
            }
            let mut updates = Vec::with_capacity(owners.len());
            for &(position, first) in owners {
                let (trajectory, _) = &trajectories[position];
                let &(_, step, set, _) = stepping
                    .iter()
                    .find(|(value, ..)| *value == position)
                    .ok_or_else(|| Error::Invariant("a stepping image lost its step".into()))?;
                let key = plan.calls[trajectory.index].request_key;
                let open = self
                    .parts
                    .trajectories
                    .get(trajectory.slot, key)
                    .ok_or_else(|| Error::Invariant("an image trajectory closed".into()))?;
                updates.push(IntegrateRow {
                    sample: scratch[position].sample,
                    units: scratch[position].units,
                    slot: trajectory.slot as u32,
                    step,
                    prediction: predicted.offsets[first],
                    branches: set,
                    params: Arc::clone(&open.plan.params),
                });
            }
            self.parts
                .denoiser()
                .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?
                .integrate(&updates, job)?;
        }

        // Every trajectory publishes its successor into the other bank.
        for ((trajectory, _), scratch) in trajectories.iter().zip(scratch) {
            let latents = self.parts.latents()?;
            let bank = latents.pool.next_bank(trajectory.slot);
            latents.pool.require_writable(bank, &scratch.pages)?;
            latents.scatter(bank, scratch, job.on(stream.as_ref()))?;
            let params = &trajectory.params;
            self.publish_latent(
                trajectory.index,
                bank,
                trajectory.step_index + 1,
                params,
                job.on(stream.as_ref()),
            )?;
            let call = &plan.calls[trajectory.index];
            self.updates.push((
                trajectory.index,
                LatentUpdate::for_call(
                    trajectory.slot,
                    call,
                    &trajectory.params,
                    i64::from(trajectory.step_index),
                )?,
            ));
        }
        Ok(())
    }
}
