//! Image latent preparation and KV-conditioned denoising batches
//! (`LatentPreparation`, `Denoising` of `ImageDenoiser` models).
//!
//! A trajectory starts with latent preparation: its guidance plan opens,
//! its seeded initial latent is drawn into the step workspace and scattered
//! into bank 1 of its pages. Each denoising call then advances it by an
//! interval of solver steps: the committed bank is gathered into the step
//! workspace; every step fills the slot's timestep, resolves the guidance
//! branches it evaluates (prefilling missing guidance prefixes through the
//! text runtime), predicts every branch of every image (one numerical call
//! per image size) and integrates each image's predictions; the interval's
//! result is scattered into the other bank. Commits become visible when the
//! batch commits ([`super::batch::ImageBatch::commit`]).

use std::collections::HashMap;

use uniserve_cuda::Stream;
use uniserve_ipc::{DrawLayout, ForwardStats, LatentParams};

use super::ImageRuntime;
use super::batch::ImageBatch;
use super::latents::Scratch;
use crate::executor::forward::experts::ExpertRuntime;
use crate::executor::forward::token::TextRuntime;
use crate::executor::forward::transfer::latents;
use crate::request::diffusion::{Coordinates, ForwardSlot, ImageTrajectory};
use crate::runner::diffusion::image::{DenoiseRow, IntegrateRow};
use crate::runner::graphs::GraphMode;
use crate::storage::LatentUpdate;
use crate::{Error, Result};

/// The denoiser serving a batch, with its device and stream.
pub(super) fn placement(runtime: &ImageRuntime) -> Result<(Option<i32>, Option<Stream>, usize)> {
    let runner = runtime.diffusion().ok_or_else(|| {
        Error::Unsupported("image trajectories are served by an image denoiser".into())
    })?;
    let stream = runner.stream().cloned();
    Ok((runner.device().cuda_index(), stream, 0))
}

/// The denoising rows of one image size: the size, its rows and, per
/// stepping image, its position among the batch's trajectories and its
/// first row.
type SizeGroup = ((u32, u32), Vec<DenoiseRow>, Vec<(usize, usize)>);

/// One active trajectory call of a batch.
struct Trajectory {
    index: usize,
    slot: usize,
    params: LatentParams,
    steps: u32,
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
    fn trajectory(&self, index: usize) -> Result<(Trajectory, uniserve_core::ImageParams)> {
        let call = &self.plan.calls[index];
        let request = self.requests.get(call.request_key.request_id.0)?;
        let image =
            request.admission().image.clone().ok_or_else(|| {
                Error::Invalid("image trajectory requires image parameters".into())
            })?;
        let params = self.latent_params(index)?;
        let layout = *self
            .runtime
            .diffusion()
            .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?
            .layout();
        if (params.height, params.width) != (image.height, image.width)
            || params.latent_units != layout.latent_rows(params.height, params.width)?
        {
            return Err(Error::Invalid(
                "latent allocation disagrees with the admitted image".into(),
            ));
        }
        Ok((
            Trajectory {
                index,
                slot: self.calls[index].slot,
                params,
                steps: u32::from(image.steps),
            },
            image,
        ))
    }

    /// The request's own conditioning KV: the export it published and its
    /// visible extent.
    fn conditioning(&mut self, index: usize) -> Result<Coordinates> {
        let call = &self.plan.calls[index];
        let buffer = call.kv_input.ok_or_else(|| {
            Error::Invalid("image trajectory requires its conditioning KV".into())
        })?;
        let slot = self.calls[index].slot as u32;
        let visible = self.calls[index].output.progress.kv_visible_len;
        let kv =
            self.storage.kv.as_mut().ok_or_else(|| {
                Error::Unsupported("image trajectories require a KV cache".into())
            })?;
        let (tables, manager) = kv.split();
        let coordinates = tables.coordinates(slot, visible)?;
        manager.validate_conditioning(call.request_key, buffer, slot, visible, tables)?;
        Ok(coordinates.into())
    }

    /// Export the version call `index` just wrote into `bank` (at solver
    /// `step`) to other ranks and workers, when the rank publishes there
    /// and reports the component's results. Readers on this rank use the
    /// committed trajectory directly.
    fn publish_latent(
        &mut self,
        index: usize,
        bank: u8,
        step: u32,
        params: &LatentParams,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let call = &self.plan.calls[index];
        let Some(product) = &call.latent_output else {
            return Ok(());
        };
        if !self.reports || !self.storage.transports.exports_remotely() {
            return Ok(());
        }
        let (pool, storage) = self.runtime.latent_pool()?;
        let buffer = latents::export(
            pool,
            storage,
            &self.storage.transports,
            call,
            product,
            self.calls[index].slot,
            params,
            bank,
            step,
            stream,
            &mut self.publications,
            &mut self.calls[index].output,
        )?;
        self.latent_exports.push(buffer);
        Ok(())
    }

    /// Open the batch's use of the latent pool and place its trajectories.
    fn place(&mut self, trajectories: &[Trajectory]) -> Result<Vec<Scratch>> {
        let latents = self.runtime.latents()?;
        latents.enter(self.stream.as_ref())?;
        self.latents_entered = true;
        let params: Vec<&LatentParams> = trajectories.iter().map(|value| &value.params).collect();
        latents.place(&params)
    }

    /// Draw each trajectory's seeded initial latent and stage it as its
    /// first committed version.
    pub(super) fn prepare_latents(&mut self) -> Result<ForwardStats> {
        let mut trajectories = Vec::new();
        for index in self.active() {
            let call = &self.plan.calls[index];
            let (trajectory, image) = self.trajectory(index)?;
            if trajectory.params.start_step != 0 || trajectory.params.step_count != 0 {
                return Err(Error::Invalid(
                    "latent preparation assigns no denoising steps".into(),
                ));
            }
            if self.calls[index].output.progress.flow_step != 0 {
                return Err(Error::Invalid(
                    "latent preparation requires a fresh trajectory".into(),
                ));
            }
            let rng = call
                .rng
                .ok_or_else(|| Error::Invalid("latent preparation requires an rng".into()))?;
            if rng.draw_layout != DrawLayout::FlowNoise
                || rng.seed != image.seed.unwrap_or(0)
                || rng.semantic_index_base == 0
            {
                return Err(Error::Invalid(
                    "latent preparation rng disagrees with the admitted image".into(),
                ));
            }
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
            trajectories.push((trajectory, image, rng));
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
                })
                .collect();
            self.place(&values)?
        };

        let stream = self.stream.clone();
        for ((trajectory, image, rng), scratch) in trajectories.iter().zip(&scratch) {
            let call = &self.plan.calls[trajectory.index];
            let diffusion = self.runtime.diffusion().ok_or_else(|| {
                Error::Unsupported("image trajectories require a denoiser".into())
            })?;
            let plan = diffusion.plan(image)?;
            let latents = self.runtime.latents()?;
            let page_units = latents.pool.page_units();
            latents
                .pool
                .require_initial(trajectory.slot, &scratch.pages)?;
            let semantic = u32::try_from(rng.semantic_index_base)
                .map_err(|_| Error::Invalid("image index exceeds its range".into()))?;
            diffusion.initial_latent(
                scratch.sample,
                scratch.units,
                scratch.padded(page_units),
                trajectory.params.height,
                trajectory.params.width,
                rng.seed,
                semantic,
            )?;
            latents.scatter(1, scratch, stream.as_ref())?;
            self.publish_latent(trajectory.index, 1, 0, &trajectory.params, stream.as_ref())?;
            self.runtime.trajectories.open(
                trajectory.slot,
                ImageTrajectory::new(
                    call.request_key,
                    (trajectory.params.height, trajectory.params.width),
                    plan,
                ),
            );
            self.updates.push((
                trajectory.index,
                LatentUpdate::for_call(trajectory.slot, call, &trajectory.params)?,
            ));
        }
        Ok(ForwardStats::default())
    }

    /// Install the block tables of every slot the active calls' forward
    /// rows address, reset their new units and retain the extents the rows
    /// read and write until the batch's device work retires. Rows on slots
    /// other than a call's own hold its guidance prefixes.
    fn reserve_rows(&mut self, active: &[usize]) -> Result<HashMap<usize, Vec<ForwardSlot>>> {
        let plan = self.plan;
        let forward = &plan.forward;
        let stream = self.stream.clone();
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
            .chain(active.iter().map(|&index| self.calls[index].slot as u32))
            .collect();
        let units = self.runtime.units.as_ref();
        let kv =
            self.storage.kv.as_mut().ok_or_else(|| {
                Error::Unsupported("image trajectories require a KV cache".into())
            })?;
        let assignments: Vec<_> = plan
            .block_tables
            .iter()
            .filter(|table| slots.contains(&table.request_pool_idx))
            .cloned()
            .collect();
        kv.install(&assignments, stream.as_ref())?;
        let new_units: Vec<u32> = plan
            .new_cache_units
            .iter()
            .filter(|allocation| slots.contains(&allocation.request_pool_idx))
            .flat_map(|allocation| allocation.unit_ids.iter().map(|unit| unit.0))
            .collect();
        if !new_units.is_empty() {
            let column = units.ok_or_else(|| {
                Error::Unsupported("image trajectories reset KV units on their stream".into())
            })?;
            kv.recycle(&new_units, column, stream.as_ref())?;
        }

        for (row, &call) in forward.call_indices.iter().enumerate() {
            let index = call as usize;
            if !active.contains(&index) {
                continue;
            }
            let key = plan.calls[index].request_key;
            let slot = forward.request_pool_indices[row];
            let own = self.calls[index].slot as u32;
            let tables = kv.tables_mut();
            if slot != own {
                if forward.seq_lens[row] - forward.query_lens[row] > tables.allocated_length(slot) {
                    return Err(Error::Invalid(
                        "forward row exceeds alternative-prefix capacity".into(),
                    ));
                }
                tables.retain_prefix(key, slot);
            }
            // Readers retain their prefix, writers also their query tokens.
            let length = u64::from(if forward.write_kv[row] {
                forward.seq_lens[row]
            } else {
                forward.seq_lens[row] - forward.query_lens[row]
            });
            let mut spans = Vec::new();
            for group in 0..tables.groups().len() as u32 {
                let table = tables.table(slot, group)?;
                let start = u64::from(table.start_page) * u64::from(table.shape.page_tokens);
                if length > start {
                    spans.extend(table.spans(start, length - start)?);
                }
            }
            kv.manager()
                .retain_execution(key, &spans, std::sync::Arc::clone(&self.retirement));
        }
        Ok(descriptors)
    }

    /// Advance every active trajectory by its assigned interval. Missing
    /// guidance prefixes are evaluated by `text` first, as expert steps on a
    /// rank with `experts`.
    pub(super) fn denoise(
        &mut self,
        mut text: Option<&mut TextRuntime>,
        experts: Option<&ExpertRuntime>,
    ) -> Result<ForwardStats> {
        let plan = self.plan;
        let active = self.active();
        if active.is_empty() {
            return Ok(ForwardStats::default());
        }
        let stream = self.stream.clone();
        if let Some(text) = text.as_deref() {
            // Conditioning K/V and earlier prefixes were written by text
            // batches, possibly on another stream.
            text.order().wait(stream.as_ref())?;
        }
        let descriptors = self.reserve_rows(&active)?;

        // Validate every call and resume its trajectory.
        let mut trajectories = Vec::with_capacity(active.len());
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
            if call.rng.is_some() {
                return Err(Error::Invalid("denoising draws no random values".into()));
            }
            if params.step_count == 0
                || params.start_step + params.step_count > trajectory.steps
                || params.step_count > call.bounds.max_tokens
            {
                return Err(Error::Invalid(
                    "denoising interval lies outside the admitted schedule".into(),
                ));
            }
            let conditioning = self.conditioning(index)?;
            let size = (params.height, params.width);
            let key = call.request_key;
            let reopen = self
                .runtime
                .trajectories
                .get(trajectory.slot, key)
                .is_none_or(|open| open.size != size);
            if reopen {
                let diffusion = self.runtime.diffusion().ok_or_else(|| {
                    Error::Unsupported("image trajectories require a denoiser".into())
                })?;
                let guidance = diffusion.plan(&image)?;
                self.runtime
                    .trajectories
                    .open(trajectory.slot, ImageTrajectory::new(key, size, guidance));
            }
            if let Some(open) = self.runtime.trajectories.get(trajectory.slot, key) {
                open.begin_call(conditioning);
            }
            trajectories.push((trajectory, image));
        }
        let scratch = {
            let values: Vec<Trajectory> = trajectories
                .iter()
                .map(|(value, _)| Trajectory {
                    index: value.index,
                    slot: value.slot,
                    params: value.params.clone(),
                    steps: value.steps,
                })
                .collect();
            self.place(&values)?
        };

        // Gather each committed version into the step workspace.
        {
            let latents = self.runtime.latents()?;
            for ((trajectory, _), scratch) in trajectories.iter().zip(&scratch) {
                let call = &plan.calls[trajectory.index];
                let params = &trajectory.params;
                let generation = call
                    .latent_input
                    .as_ref()
                    .map_or(0, |input| i64::from(input.generation));
                let bank = latents.pool.current_bank(
                    trajectory.slot,
                    i64::from(params.start_step),
                    generation,
                    i64::from(params.latent_units),
                    i64::from(params.height),
                    i64::from(params.width),
                    &scratch.pages,
                )?;
                latents.gather(bank, scratch, stream.as_ref())?;
            }
        }

        let layout = *self
            .runtime
            .diffusion()
            .ok_or_else(|| Error::Unsupported("image trajectories require a denoiser".into()))?
            .layout();
        let component = plan.calls[active[0]].component.clone();
        let mut stats = ForwardStats::default();
        let intervals = trajectories
            .iter()
            .map(|(trajectory, _)| trajectory.params.step_count)
            .max()
            .unwrap_or(0);
        for offset in 0..intervals {
            // Timesteps and guidance branches of this step.
            let mut stepping = Vec::new();
            let mut fills: Vec<(usize, u32, std::sync::Arc<[u32]>)> = Vec::new();
            for (position, ((trajectory, image), _)) in
                trajectories.iter().zip(&scratch).enumerate()
            {
                let params = &trajectory.params;
                if offset >= params.step_count {
                    continue;
                }
                let step = params.start_step + offset;
                let key = plan.calls[trajectory.index].request_key;
                let call_rows = descriptors
                    .get(&trajectory.index)
                    .map(Vec::as_slice)
                    .unwrap_or_default();
                let policy = self.runtime.policy.clone();
                let kv = self.storage.kv.as_ref().ok_or_else(|| {
                    Error::Unsupported("image trajectories require a KV cache".into())
                })?;
                let tables = kv.tables();
                let negative = self
                    .requests
                    .get(key.request_id.0)?
                    .admission()
                    .ar
                    .as_ref()
                    .map(|ar| ar.negative_token_ids.clone())
                    .unwrap_or_default();
                let open = self
                    .runtime
                    .trajectories
                    .get(trajectory.slot, key)
                    .ok_or_else(|| Error::Invariant("an image trajectory closed".into()))?;
                let (set, time) = open.plan.step(step as usize)?;
                let prompt = image.image_prompts.first().cloned().unwrap_or_default();
                let filled = open.resolve_branches(
                    set,
                    layout.image_unconditional,
                    call_rows,
                    |source| {
                        policy.guidance_prefix(source, &prompt, &image.negative_prompt, &negative)
                    },
                    |slot, visible| tables.coordinates(slot, visible).map(Coordinates::from),
                )?;
                fills.extend(
                    filled
                        .into_iter()
                        .map(|fill| (position, fill.slot, fill.tokens)),
                );
                self.runtime.latents()?.fill_timestep(
                    trajectory.slot as u32,
                    time,
                    stream.as_ref(),
                )?;
                stepping.push((position, step, set, time));
            }

            // Missing guidance prefixes are written before the step reads
            // them.
            if !fills.is_empty() {
                let text = text.as_deref_mut().ok_or_else(|| {
                    Error::Unsupported("guidance prefixes require the text runtime".into())
                })?;
                let kv = self.storage.kv.as_mut().ok_or_else(|| {
                    Error::Unsupported("image trajectories require a KV cache".into())
                })?;
                let (tables, manager) = kv.split();
                let prefixes: Vec<(u32, &[u32])> = fills
                    .iter()
                    .map(|(_, slot, tokens)| (*slot, &tokens[..]))
                    .collect();
                text.prefill_cache(experts, &component, &prefixes, tables, manager)?;
                text.order().wait(stream.as_ref())?;
                for (position, slot, tokens) in &fills {
                    let (trajectory, _) = &trajectories[*position];
                    let key = plan.calls[trajectory.index].request_key;
                    if let Some(open) = self.runtime.trajectories.get(trajectory.slot, key) {
                        open.prefilled(*slot, tokens.len() as u64);
                    }
                }
            }

            // One denoising row per branch of every stepping image; images
            // of one size share a numerical call.
            let mut groups: Vec<SizeGroup> = Vec::new();
            for &(position, _, set, time) in &stepping {
                let (trajectory, _) = &trajectories[position];
                let key = plan.calls[trajectory.index].request_key;
                let logical = self.calls[trajectory.index]
                    .output
                    .progress
                    .logical_position;
                let open = self
                    .runtime
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
                let kv = self.storage.kv.as_ref().ok_or_else(|| {
                    Error::Unsupported("image trajectories require a KV cache".into())
                })?;
                let diffusion = self.runtime.diffusion_mut().ok_or_else(|| {
                    Error::Unsupported("image trajectories require a denoiser".into())
                })?;
                let predicted = diffusion.predict(rows, kv.tables(), |_, _, _| Ok(()))?;
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
                        .runtime
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
                        plan: open.plan.handle.clone(),
                    });
                }
                self.runtime
                    .diffusion()
                    .ok_or_else(|| {
                        Error::Unsupported("image trajectories require a denoiser".into())
                    })?
                    .integrate(&updates)?;
            }

            // Intervals ending at this step publish their successor into
            // the other bank.
            for ((trajectory, _), scratch) in trajectories.iter().zip(&scratch) {
                if offset + 1 != trajectory.params.step_count {
                    continue;
                }
                let latents = self.runtime.latents()?;
                let bank = latents.pool.next_bank(trajectory.slot);
                latents.pool.require_writable(bank, &scratch.pages)?;
                latents.scatter(bank, scratch, stream.as_ref())?;
                let params = &trajectory.params;
                self.publish_latent(
                    trajectory.index,
                    bank,
                    params.start_step + params.step_count,
                    params,
                    stream.as_ref(),
                )?;
                let call = &plan.calls[trajectory.index];
                self.updates.push((
                    trajectory.index,
                    LatentUpdate::for_call(trajectory.slot, call, &trajectory.params)?,
                ));
            }
        }

        // A finished trajectory's guidance prefixes retire with it.
        for (trajectory, _) in &trajectories {
            let params = &trajectory.params;
            if params.start_step + params.step_count < trajectory.steps {
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
            let kv = self.storage.kv.as_mut().ok_or_else(|| {
                Error::Unsupported("image trajectories require a KV cache".into())
            })?;
            kv.release_prefixes(key, &slots, stream.as_ref())?;
        }
        if let Some(text) = text {
            text.order_mut().record(stream.as_ref())?;
        }
        Ok(stats)
    }
}
