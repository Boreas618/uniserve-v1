//! Joining rank reports: the calls a set of agreeing rank results completes are
//! taken from the per-rank buffers, each report is validated and ordered to
//! match the submitted calls, and the participants' records are merged into the
//! report of the calls' output owner.

use std::collections::{BTreeSet, VecDeque};

use anyhow::Context;
use uniserve_ipc::Batch;

use super::{CallIdentity, call_identity};
use crate::executor::{CallResult, WorkerResult};

/// Consume selected calls while retaining the rank results that hold the
/// rest. A rank result's aggregate statistics are emitted once, when its final
/// call is consumed; they are never divided or copied.
///
/// An empty `identities` visits every result of the batch without taking a
/// call. `try_join` sets `retain_forward_stats` only when taking from the
/// output owner. Fails when a selected call is missing or a drained result
/// keeps a product, and, with `retain_forward_stats`, when the selection
/// drains more than one result carrying forward statistics.
pub(super) fn take_rank_calls(
    buffer: &mut VecDeque<WorkerResult>,
    batch: &Batch,
    identities: &[CallIdentity],
    retain_forward_stats: bool,
) -> anyhow::Result<WorkerResult> {
    let mut output = WorkerResult {
        batch_id: batch.batch_id,
        done: false,
        results: Vec::with_capacity(identities.len()),
        products: Vec::new(),
        worker_exec_us: None,
        forward_stats: None,
    };
    for report in buffer
        .iter_mut()
        .filter(|report| report.batch_id == batch.batch_id)
    {
        let selected = report.results.iter().any(|completion| {
            identities.contains(&call_identity(
                completion.output.request_key,
                completion.output.call_id,
            ))
        });
        if !selected && !identities.is_empty() {
            continue;
        }
        output
            .results
            .extend(report.results.extract_if(.., |completion| {
                identities.contains(&call_identity(
                    completion.output.request_key,
                    completion.output.call_id,
                ))
            }));
        output
            .products
            .extend(report.products.extract_if(.., |product| {
                identities.contains(&call_identity(
                    product.product.request_key,
                    product.product.producer_call_id,
                ))
            }));
        if report.results.is_empty() {
            anyhow::ensure!(
                report.products.is_empty(),
                "rank result retained an unowned product"
            );
            if retain_forward_stats && report.forward_stats.is_some() {
                anyhow::ensure!(
                    output.forward_stats.is_none(),
                    "canonical call selection spans statistics fragments"
                );
                output.forward_stats = report.forward_stats.take();
            }
        }
    }
    // A drained result stays buffered while it carries `worker_exec_us`, which
    // `try_join` sums per rank once the batch is done; `finish_batch` then
    // removes it.
    buffer.retain(|report| {
        report.batch_id != batch.batch_id
            || !report.results.is_empty()
            || report.worker_exec_us.is_some()
    });
    anyhow::ensure!(
        output.results.len() == identities.len(),
        "rank result omitted selected calls"
    );
    Ok(output)
}

/// Returns the call identifiers carried by a rank report.
pub(super) fn report_call_ids(report: &WorkerResult) -> Vec<CallIdentity> {
    let mut ids = report
        .results
        .iter()
        .map(|output| call_identity(output.output.request_key, output.output.call_id))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

/// Join cooperative completion reports into the designated output owner's report.
/// Every participant agrees on semantic completion; only the output owner
/// publishes host products.
pub(super) fn merge_rank_report(
    batch: &Batch,
    canonical_report: &mut WorkerResult,
    participant_report: &WorkerResult,
    rank: usize,
) -> anyhow::Result<()> {
    let batch_id = batch.batch_id;
    if participant_report.batch_id != batch_id {
        anyhow::bail!(
            "rank {rank} completion report batch ID mismatch while joining pending batch {batch_id}: got {}",
            participant_report.batch_id
        );
    }
    if participant_report.results.len() != canonical_report.results.len() {
        anyhow::bail!(
            "rank {rank} report for batch {batch_id} has {} completions, expected {}",
            participant_report.results.len(),
            canonical_report.results.len()
        );
    }
    for (completion_index, (canonical, actual)) in canonical_report
        .results
        .iter_mut()
        .zip(&participant_report.results)
        .enumerate()
    {
        anyhow::ensure!(
            batch
                .calls
                .iter()
                .any(|call| call.request_key == canonical.output.request_key
                    && call.call_id == canonical.output.call_id),
            "rank join received an unplanned call for batch {batch_id}"
        );
        if let Err(error) = merge_completion_record(canonical, actual) {
            anyhow::bail!(
                "rank {rank} report for batch {batch_id} completion {completion_index} differs from the output owner: {error:#}"
            );
        }
    }
    for product in &participant_report.products {
        anyhow::ensure!(
            batch
                .calls
                .iter()
                .flat_map(|call| call.tensor_outputs())
                .any(|output| output == &product.product),
            "rank {rank} published locations for an undeclared product"
        );
        if let Some(existing) = canonical_report
            .products
            .iter_mut()
            .find(|existing| existing.product == product.product)
        {
            existing.merge_locations(product)?;
        } else {
            canonical_report.products.push(product.clone());
        }
    }
    Ok(())
}

/// Validates a rank report and orders completions to match the submitted call sequence.
pub(super) fn validate_and_order_rank_report(
    batch: &Batch,
    report: &mut WorkerResult,
    rank: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        report.batch_id == batch.batch_id,
        "rank {rank} returned pending batch {} for pending batch {}",
        report.batch_id,
        batch.batch_id
    );
    let report_count = report.results.len();
    let returned = report_call_ids(report).into_iter().collect::<BTreeSet<_>>();
    let mut ordered = Vec::with_capacity(report_count);
    for planned in &batch.calls {
        let identity = call_identity(planned.request_key, planned.call_id);
        if !returned.contains(&identity) {
            continue;
        }
        let index = report
            .results
            .iter()
            .position(|completion| {
                completion.output.request_key == planned.request_key
                    && completion.output.call_id == planned.call_id
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "rank {rank} omitted a call for pending batch {} collective {}",
                    batch.batch_id,
                    batch.collective_seq
                )
            })?;
        let completion = report.results.swap_remove(index);
        anyhow::ensure!(
            completion.output.status != uniserve_ipc::CallStatus::Predicated
                || report.products.iter().all(|export| {
                    export.product.request_key != completion.output.request_key
                        || export.product.producer_call_id != completion.output.call_id
                }),
            "predicated call published a tensor"
        );
        if planned.code == uniserve_ipc::CallKind::Transfer(uniserve_ipc::TransferMode::CacheExport)
            && completion.output.status == uniserve_ipc::CallStatus::Ok
        {
            let export = completion
                .output
                .cache_output
                .as_ref()
                .context("successful cache export has no transfer descriptor")?;
            anyhow::ensure!(
                Some(export.source) == planned.cache_output,
                "cache export differs from its declared output"
            );
            let bytes = export.tensors().try_fold(0_u64, |sum, tensor| {
                Ok::<_, uniserve_ipc::ValidationError>(sum.saturating_add(tensor.validate()?))
            })?;
            anyhow::ensure!(
                bytes <= planned.bounds.max_transfer_bytes,
                "cache export exceeds its transfer-byte bound"
            );
        }
        ordered.push(completion);
    }
    anyhow::ensure!(
        ordered.len() == report_count,
        "rank {rank} returned an unplanned call for pending batch {}",
        batch.batch_id
    );
    report.results = ordered;
    Ok(())
}

/// Merge one participant into the output owner's result. Accepted progress and
/// allocation generations agree; cache exports contribute immutable rank locations.
fn merge_completion_record(
    canonical: &mut CallResult,
    rank_completion: &CallResult,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        canonical.output.request_key == rank_completion.output.request_key
            && canonical.output.call_id == rank_completion.output.call_id,
        "completion identity diverged"
    );
    anyhow::ensure!(
        canonical.output.committed_tokens.as_slice()
            == rank_completion.output.committed_tokens.as_slice()
            && canonical.output.status == rank_completion.output.status
            && canonical.output.position == rank_completion.output.position
            && canonical.output.cache_visible_len == rank_completion.output.cache_visible_len
            && canonical.output.cache_computed_len == rank_completion.output.cache_computed_len
            && canonical.output.num_completed_steps == rank_completion.output.num_completed_steps
            && canonical.output.finish_flags == rank_completion.output.finish_flags,
        "completion result fields diverged"
    );
    anyhow::ensure!(
        rank_completion.media.as_ref().is_ok_and(Option::is_none)
            && rank_completion.output.sampled_logprob.is_none()
            && rank_completion.output.top_logprobs.is_empty()
            && rank_completion.output.prompt_logprobs.is_empty()
            && rank_completion.output.candidate_logprobs.is_empty(),
        "non-designated rank returned public output"
    );
    anyhow::ensure!(
        canonical.output.product_generations == rank_completion.output.product_generations,
        "designated-rank product generations diverged"
    );
    match (
        &mut canonical.output.cache_output,
        &rank_completion.output.cache_output,
    ) {
        (Some(stored), Some(export)) => stored.merge_locations(export)?,
        (None, None) => {}
        _ => anyhow::bail!("rank cache export presence diverged"),
    }
    Ok(())
}
