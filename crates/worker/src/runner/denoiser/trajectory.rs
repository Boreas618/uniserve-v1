//! A video request's trajectory in a media batch: its initialization
//! ([`prepare`]: prepared sources, refined prompt and encoded conditions) and
//! its solver steps ([`denoise`]), the final one publishing the rank's rows of
//! each modality's latents.

use std::sync::Arc;

use uniserve_core::VideoCondition;
use uniserve_ipc::Batch as BatchPlan;

use crate::executor::products::{OutputShape, Placement, bind_outputs};
use crate::executor::video::MediaBatch;
use crate::executor::video::{Parts, product_name, stream_of};
use crate::lane::Job;
use crate::storage::memory::StorageView;
use crate::{Error, Result};

/// Initialize a request's trajectory: its prepared sources, its refined
/// prompt where this rank retains conditioning
/// ([`DenoiserRunner::conditions`]) and its encoded conditions.
pub(crate) fn prepare(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    let (id, seed, conditioned, size) = {
        let request = parts.video.request(slot, call.request_key)?;
        let id = request
            .layout
            .ok_or_else(|| Error::Invariant("a denoising request has no layout".into()))?;
        let size = request
            .admission
            .as_ref()
            .map(|admission| admission.size.clone())
            .ok_or_else(|| Error::Invariant("a denoising request has no size".into()))?;
        (id, request.seed, !request.conditions.is_empty(), size)
    };
    if call.inputs.is_empty() || (call.inputs.len() > 1) != conditioned {
        return Err(Error::Invalid(
            "video preparation reads its conditioning and its conditions".into(),
        ));
    }
    let refines = parts
        .denoisers
        .video
        .as_ref()
        .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?
        .runner
        .conditions();
    let store = parts.owners().store;
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    // A pipeline stage without the conditioner leaves the prompt features
    // unread: its forward receives the refined prompt among the preceding
    // stage's hidden rows.
    let features = refines
        .then(|| batch.read(store, &call.inputs[0], call.call_id, issue))
        .transpose()?;
    let conditions: Vec<Arc<crate::ProductRead>> = call.inputs[1..]
        .iter()
        .map(|input| batch.read(store, input, call.call_id, issue))
        .collect::<Result<_>>()?;
    let denoiser = parts
        .denoisers
        .video
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
    // The request's noise and the host sources its slot is initialized from.
    denoiser
        .runner
        .prepare_latents(id, slot, &size, seed, job)?;
    let update = denoiser
        .runner
        .initialize(id, slot, call, size.canvas, job)?;
    if let Some(features) = features {
        let value = features.value();
        let count = u64::from(size.tokens);
        if value.shape().first().copied() != Some(count as i64) {
            return Err(Error::Invalid(
                "prompt features disagree with the request's prompt".into(),
            ));
        }
        let mode = denoiser.runner.condition(id, slot, value, count, job)?;
        batch.record(mode);
    }
    if conditioned {
        let request = parts.video.request(slot, call.request_key)?;
        let latents = condition_latents(&request.conditions, &conditions)?;
        let denoiser = parts
            .denoisers
            .video
            .as_ref()
            .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
        denoiser
            .runner
            .encode_conditions(id, slot, &size, &latents, job)?;
    }
    batch.batch.calls[index].forward.latent = Some(update);
    Ok(())
}

/// Rows of one condition read: `(read, first row, rows)`.
type RowSpan = (usize, u64, u64);

/// Locate every condition latent in latent preparation's condition reads.
///
/// `conditions` are each condition's `(visual rows, audio rows)` in request
/// order. The reads are the visual rounds' rows in round order (`rounds`
/// holds each round's row count), then, when any condition carries audio,
/// one read of every track's rows (`tracks`). The visual rounds hold the
/// conditions' visual rows back to back, and a round ends wherever its
/// units do, so a condition's rows may continue into the next round.
///
/// Returns the latents in request order, a condition's visual rows before
/// its audio rows, each as the spans that hold its rows in row order.
fn condition_spans(
    conditions: &[(u64, u64)],
    rounds: &[u64],
    tracks: Option<u64>,
) -> Result<Vec<Vec<RowSpan>>> {
    let mismatch =
        || Error::Invalid("condition latents disagree with the request's planned rows".into());
    let visual: u64 = conditions.iter().map(|rows| rows.0).sum();
    let audio: u64 = conditions.iter().map(|rows| rows.1).sum();
    if rounds.iter().sum::<u64>() != visual || tracks.unwrap_or(0) != audio {
        return Err(mismatch());
    }
    // The audio read follows the visual rounds.
    let track_read = rounds.len();
    let mut result = Vec::new();
    let (mut round, mut row, mut audio_row) = (0usize, 0u64, 0u64);
    for &(visual_rows, audio_rows) in conditions {
        if visual_rows > 0 {
            let mut spans = Vec::new();
            let mut needed = visual_rows;
            while needed > 0 {
                let available = *rounds.get(round).ok_or_else(mismatch)? - row;
                let taken = available.min(needed);
                if taken > 0 {
                    spans.push((round, row, taken));
                }
                needed -= taken;
                row += taken;
                if taken == available {
                    round += 1;
                    row = 0;
                }
            }
            result.push(spans);
        }
        if audio_rows > 0 {
            result.push(vec![(track_read, audio_row, audio_rows)]);
            audio_row += audio_rows;
        }
    }
    Ok(result)
}

/// The condition latents of latent preparation's condition `reads`, each a
/// list of row tensors that together hold its rows (see
/// [`condition_spans`]).
fn condition_latents(
    conditions: &[VideoCondition],
    reads: &[Arc<crate::ProductRead>],
) -> Result<Vec<Vec<StorageView>>> {
    let audio = conditions.iter().any(|condition| condition.audio_rows > 0);
    let (rounds, tracks) = match (audio, reads.split_last()) {
        (true, Some((tracks, rounds))) => (rounds, Some(tracks)),
        (true, None) => {
            return Err(Error::Invalid(
                "condition latents lack their audio rows".into(),
            ));
        }
        (false, _) => (reads, None),
    };
    let rows = |read: &Arc<crate::ProductRead>| {
        read.value().shape().first().copied().unwrap_or(0).max(0) as u64
    };
    let planned: Vec<(u64, u64)> = conditions
        .iter()
        .map(|condition| (condition.video_rows(), u64::from(condition.audio_rows)))
        .collect();
    let round_rows: Vec<u64> = rounds.iter().map(rows).collect();
    condition_spans(&planned, &round_rows, tracks.map(rows))?
        .into_iter()
        .map(|spans| {
            spans
                .into_iter()
                .map(|(read, first, count)| {
                    reads[read].value().narrow(0, first as i64, count as i64)
                })
                .collect()
        })
        .collect()
}

/// Advance a request's trajectory by one step; the final step publishes the
/// rank's rows of each modality's latents.
pub(crate) fn denoise(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    let (id, canvas) = {
        let request = parts.video.request(slot, call.request_key)?;
        let id = request
            .layout
            .ok_or_else(|| Error::Invariant("a denoising request has no layout".into()))?;
        (id, request.canvas)
    };
    let num_steps = parts.video.model.facts.num_steps;
    let names: Vec<String> = call
        .outputs
        .iter()
        .map(|output| product_name(parts.info, call, output).map(str::to_owned))
        .collect::<Result<_>>()?;
    let denoiser = parts
        .denoisers
        .video
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no video denoiser".into()))?;
    // Each call advances the slot's committed trajectory, the latent pool's
    // ledger, by the next step of the checkpoint's fixed schedule.
    let step = denoiser.runner.committed_step(slot)?;
    if step >= i64::from(num_steps) {
        return Err(Error::Invalid(
            "denoising continues past the fixed schedule".into(),
        ));
    }
    // Only the call that completes denoising declares products: the final
    // samples its step publishes.
    if !call.outputs.is_empty() && step + 1 != i64::from(num_steps) {
        return Err(Error::Invalid(
            "final latent products require completed denoising".into(),
        ));
    }
    let (update, mode) = denoiser.runner.denoise(id, slot, call, canvas, job)?;
    batch.record(mode);
    batch.batch.calls[index].forward.latent = Some(update);
    if call.outputs.is_empty() {
        return Ok(());
    }
    let samples = denoiser.runner.final_samples(id)?;
    let rows = denoiser.runner.sample_rows(id)?;
    let modalities = parts.video.model.facts.modalities.clone();
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    for (output, name) in call.outputs.iter().zip(&names) {
        let modality = modalities
            .iter()
            .position(|modality| format!("{modality}_latents") == *name)
            .ok_or_else(|| Error::Invalid(format!("the denoiser publishes no {name}")))?;
        let (Some(sample), sample_rows) = (&samples[modality], rows[modality]) else {
            // This rank holds none of the modality's rows.
            continue;
        };
        let shape = vec![sample_rows.rows, sample_rows.width];
        let region = Some(vec![
            sample_rows.local.0..sample_rows.local.1,
            0..sample_rows.width,
        ]);
        let bound = bind_outputs(
            owners.store,
            &[OutputShape {
                product: output.clone(),
                shape,
                region,
            }],
            batch.target(),
            placement,
        )?;
        let write = bound
            .into_iter()
            .next()
            .ok_or_else(|| Error::Invariant("a latent product was not bound".into()))?;
        batch.batch.writes.push(write.clone());
        let issue = job.on(Some(&stream));
        owners.store.write(&write, sample, issue, None)?;
        batch.batch.calls[index]
            .forward
            .export(owners, call, output, &write, false, issue)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_latents_follow_their_rows_across_encoding_rounds() {
        // A keyframe of 4 rows, a reference video of 10 rows with 3 audio
        // rows and a voice of 2 audio rows; the encoder published the
        // visual rows in rounds of 6, 5 and 3 rows.
        let conditions = [(4, 0), (10, 3), (0, 2)];
        let latents = condition_spans(&conditions, &[6, 5, 3], Some(5)).unwrap();
        assert_eq!(
            latents,
            vec![
                vec![(0, 0, 4)],
                // The video continues through the second round into the third.
                vec![(0, 4, 2), (1, 0, 5), (2, 0, 3)],
                // The audio read follows the three visual rounds.
                vec![(3, 0, 3)],
                vec![(3, 3, 2)],
            ]
        );

        // Reads that hold other rows than the request planned are refused.
        assert!(condition_spans(&conditions, &[6, 5], Some(5)).is_err());
        assert!(condition_spans(&conditions, &[6, 5, 3], Some(4)).is_err());
        assert!(condition_spans(&conditions, &[6, 5, 3], None).is_err());
        // Without audio, every read is a visual round.
        assert_eq!(
            condition_spans(&[(4, 0)], &[4], None).unwrap(),
            vec![vec![(0, 0, 4)]]
        );
    }
}
