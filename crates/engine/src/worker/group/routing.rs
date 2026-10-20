//! Where a group's calls run: the refusal of placements that put host-shared
//! storage off the head's host, the ranks that read each media product
//! (`media_consumer_slots`), the projection of a batch onto the calls each
//! rank's components execute (`rank_projection`), and the rank a call's
//! results join on (`output_owner`).

use std::collections::{BTreeMap, BTreeSet, HashSet};

use anyhow::Context;
use uniserve_ipc::Batch;

use super::WorkerGroup;
use crate::executor::RankBuffer;
use crate::worker::IncarnationRegistry;

/// Refuses a placement whose muxer or media reader is not on the head's host.
///
/// Both exchange POSIX shared-storage objects with the head by name, and such
/// a name resolves in one host's namespace: the head opens the artifact a
/// muxer publishes, and the media reader opens the condition media the head
/// published. Either placed elsewhere would fail on the first request, so the
/// placement is refused at startup, naming the component and the host it was
/// placed on.
pub(crate) fn refuse_shared_storage_off_head(
    head: &str,
    ranks: &[crate::WorkerRank],
    components: &BTreeMap<String, crate::executor::ComponentConfig>,
    media_components: &BTreeMap<uniserve_ipc::MediaCall, String>,
) -> anyhow::Result<()> {
    use uniserve_ipc::MediaCall;

    for (call, work, object) in [
        (
            MediaCall::Muxing,
            "muxing",
            "the artifact it publishes is a shared-storage object",
        ),
        (
            MediaCall::MediaReading,
            "media reading",
            "the condition media it reads are shared-storage objects",
        ),
    ] {
        let Some(component) = media_components.get(&call) else {
            continue;
        };
        let placement = components.get(component).with_context(|| {
            format!("component {component} serves {work} but the placement binds no such component")
        })?;
        for &rank in &placement.ranks {
            let host = ranks
                .get(rank)
                .map(|placed| placed.host.as_str())
                .with_context(|| {
                    format!("component {component} names rank {rank} outside the placement")
                })?;
            anyhow::ensure!(
                host == head,
                "component {component} serves {work} on rank {rank} on host {host}, but the head \
                 is on host {head}, and {object} named in one host's namespace"
            );
        }
    }
    Ok(())
}

/// Returns the rank a call's results join on: the first of its component's
/// ranks, in configured order, among the `members` the call is planned on.
///
/// That is the component's first rank, which publishes its host products,
/// unless the call's units were dealt to other ranks. `None` when the call
/// is planned on none of the component's ranks.
pub(super) fn output_owner(component_ranks: &[usize], members: &[usize]) -> Option<usize> {
    component_ranks
        .iter()
        .copied()
        .find(|rank| members.contains(rank))
}

/// Acknowledgment slots of the ranks reading a media product produced on
/// `rank` of `worker`, where `members` are the ranks executing the call.
///
/// The consuming component may belong to another worker, whose ranks all
/// read the product; on the producing worker the ranks producing their own
/// copy read that copy instead of another rank's. `routed` names the worker
/// the request was routed to for a consuming component; only that replica
/// reads the product. A component absent from `routed` is read by whichever
/// replica serves it when its call is placed, so every worker holding it is
/// named.
///
/// Naming a replica the request was not routed to is not only wasted: a
/// producer publishes over every mechanism that reaches a named reader, so a
/// replica on another host would make each product cross hosts unread.
#[expect(
    clippy::too_many_arguments,
    reason = "the consuming worker's routing, components, peers and transfer placement"
)]
pub(crate) fn media_consumer_slots(
    consuming: &[uniserve_ipc::MediaCall],
    routing: &BTreeMap<uniserve_ipc::MediaCall, String>,
    routed: &BTreeMap<String, crate::WorkerGroupId>,
    worker: &str,
    components: &BTreeMap<String, crate::ComponentConfig>,
    peers: &BTreeMap<String, BTreeMap<String, crate::ComponentConfig>>,
    transfer: &crate::executor::TransferConfig,
    members: &[usize],
    rank: usize,
    producer: Option<&crate::ComponentConfig>,
) -> Vec<u32> {
    let mut slots = BTreeSet::new();
    for consumer in consuming {
        let Some(name) = routing.get(consumer) else {
            continue;
        };
        let mut owners = peers
            .iter()
            .filter_map(|(id, entries)| entries.get(name).map(|config| (id.as_str(), config)))
            .collect::<Vec<_>>();
        if !owners.iter().any(|(id, _)| *id == worker)
            && let Some(config) = components.get(name)
        {
            owners.push((worker, config));
        }
        if let Some(reader) = routed.get(name) {
            owners.retain(|(id, _)| *id == reader.0);
        }
        for (owner, component) in owners {
            // A round deals its media units to the producing ranks in
            // `members` order and to a distributed consumer's ranks in
            // configured order, `units_per_rank` each, so a distributed
            // consumer reads only the positions this producer rank wrote.
            let dealt = producer
                .filter(|producer| {
                    producer.distribution.is_some() && component.distribution.is_some()
                })
                .and_then(|producer| {
                    members
                        .iter()
                        .position(|&member| member == rank)
                        .map(|index| (producer, index))
                });
            let readers: Vec<usize> = match dealt {
                Some((producer, index)) => {
                    let per_producer = producer.units_per_rank.max(1);
                    let per_reader = component.units_per_rank.max(1);
                    (index * per_producer..(index + 1) * per_producer)
                        .filter_map(|position| component.ranks.get(position / per_reader).copied())
                        .collect()
                }
                None => component.ranks.clone(),
            };
            for reader in readers {
                if owner != worker || (reader != rank && !members.contains(&reader)) {
                    slots.insert(transfer.acknowledgment_slot(owner, reader as u32));
                }
            }
        }
    }
    slots.into_iter().collect()
}

impl WorkerGroup {
    /// Acknowledgment slots of the ranks that read the products a call
    /// produces on `rank`, where `members` are the ranks executing the call.
    ///
    /// A media call's readers are the ranks of the components serving the
    /// calls that consume it, less the ranks that produce their own copy: a
    /// consumer reads the copy it holds before any other. For a video
    /// request's call, `routed` names each component its request's graph
    /// reads the products with, and the worker the request was routed to for
    /// it. An image generation call's readers follow from its kind
    /// (`generation_consuming_calls`). Other work is read by the destinations
    /// of the rank's transfer edges.
    pub(super) fn consumer_slots(
        &self,
        call: &uniserve_ipc::Call,
        routed: Option<&BTreeMap<String, crate::WorkerGroupId>>,
        members: &[usize],
        rank: usize,
    ) -> Vec<u32> {
        let transfer = &self.process_args.transfer;
        let worker = &self.process_args.worker_id;
        let consuming: Vec<uniserve_ipc::MediaCall> = match (routed, call.kind) {
            (Some(readers), _) => self
                .media_routing
                .iter()
                .filter(|(_, component)| readers.contains_key(*component))
                .map(|(media_call, _)| *media_call)
                .collect(),
            (None, uniserve_ipc::CallKind::Media(media_call)) => {
                match crate::scheduler::generation_consuming_calls(media_call) {
                    Some(consuming) => consuming.to_vec(),
                    None => return transfer.product_consumers(worker, rank as u32),
                }
            }
            _ => return transfer.product_consumers(worker, rank as u32),
        };
        media_consumer_slots(
            &consuming,
            &self.media_routing,
            routed.unwrap_or(&BTreeMap::new()),
            worker,
            &self.process_args.components,
            &self.process_args.peers,
            transfer,
            members,
            rank,
            self.process_args.components.get(&call.component),
        )
    }

    /// The placement this group was launched with: its ranks' hosts and
    /// devices, and its components.
    pub fn placement(
        &self,
    ) -> (
        &[crate::WorkerRank],
        &BTreeMap<String, crate::ComponentConfig>,
    ) {
        (&self.process_args.ranks, &self.process_args.components)
    }

    /// Ranks in the group, as launched.
    pub(crate) fn world_size(&self) -> usize {
        self.process_args.ranks.len()
    }

    /// Whether the group placed rank `rank` on a CUDA device.
    pub(crate) fn rank_on_cuda(&self, rank: usize) -> bool {
        self.process_args
            .ranks
            .get(rank)
            .is_some_and(|placed| placed.device.starts_with("cuda"))
    }

    /// Batches the group admits at once: the queue depth it launched its
    /// ranks with, at least one.
    pub(crate) fn queue_depth(&self) -> usize {
        self.depth
    }

    /// Joins this group's ranks to the deployment's registry of live rank
    /// incarnations, which every group of the deployment then shares.
    pub(crate) fn share_incarnations(&mut self, incarnations: IncarnationRegistry) {
        incarnations.assign(
            &crate::WorkerGroupId(self.process_args.worker_id.clone()),
            self.workers.iter().map(|worker| &worker.info().endpoint),
        );
        self.incarnations = incarnations;
    }

    /// States the deployment-wide media routing once every worker reported.
    pub fn set_media_routing(&mut self, routing: BTreeMap<uniserve_ipc::MediaCall, String>) {
        self.media_routing = routing;
    }
}

/// Restrict a physical invocation to each component's actual members. Request and
/// storage commands retain group-wide visibility, including on otherwise idle
/// ranks; they do not create synthetic call completions.
///
/// `placed` holds the persistent spans of each rank's own buffer pool; a
/// rank's projection keeps those of the buffers its calls read and write.
pub(super) fn rank_projection(
    batch: &Batch,
    components: &BTreeMap<String, crate::ComponentConfig>,
    rank_count: usize,
    placed: &[RankBuffer],
    consumer_slots: impl Fn(&uniserve_ipc::Call, &[usize], usize) -> Vec<u32>,
) -> anyhow::Result<Vec<(usize, Batch)>> {
    anyhow::ensure!(
        batch.buffer_allocations.is_empty(),
        "a group batch carries no buffer spans; each rank's spans arrive beside it"
    );
    let members = batch
        .calls
        .iter()
        .map(|call| {
            let entry = components
                .get(&call.component)
                .with_context(|| format!("unknown component {}", call.component))?;
            // A distributed component deals a call's units `units_per_rank`
            // at a time to the ranks the call names, or to its ranks in
            // configured order when the call names none, so only the ranks
            // the call's `max_units` reach take part.
            // `WorkerGroupConfig::validate_members`, which `spawn_workers` calls,
            // rejects a zero `units_per_rank`.
            if entry.distribution.is_none() {
                return Ok(entry.ranks.clone());
            }
            let range = batch
                .decode_ranges
                .iter()
                .find(|range| {
                    range.request_key == call.request_key && range.call_id == call.call_id
                })
                .context("temporally distributed component requires a decode range")?;
            let holders = (range.max_units as usize).div_ceil(entry.units_per_rank);
            if range.ranks.is_empty() {
                return Ok(entry.ranks[..holders.min(entry.ranks.len())].to_vec());
            }
            let named = range
                .ranks
                .iter()
                .map(|rank| *rank as usize)
                .collect::<Vec<_>>();
            anyhow::ensure!(
                named.len() == holders && named.iter().all(|rank| entry.ranks.contains(rank)),
                "a call's ranks hold its units within component {}",
                call.component
            );
            Ok(named)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut batches = Vec::new();
    for rank in 0..rank_count {
        let indices = members
            .iter()
            .enumerate()
            .filter_map(|(index, members)| members.contains(&rank).then_some(index))
            .collect::<Vec<_>>();
        // A rank also takes part to copy its share of the batch's pages.
        let copies_pages = batch
            .page_writes
            .iter()
            .any(|copy| copy.rank as usize == rank);
        if indices.is_empty() && batch.commands.is_empty() && !copies_pages {
            continue;
        }
        // A rank receives the batch's projection onto the calls it owns,
        // under the same identity: commands travel to every participating rank.
        // Each call states which ranks read its products, which only the head
        // can derive from the placement.
        let mut projection = batch.clone();
        projection.calls = indices
            .iter()
            .map(|index| {
                let mut call = batch.calls[*index].clone();
                call.consumer_slots = consumer_slots(&call, &members[*index], rank);
                call
            })
            .collect();
        // Forward rows are narrowed to the selected calls, and the batch's
        // side tables to what those calls use: their request pool slots,
        // identities, inputs, cache sources, and buffers.
        projection.forward = batch.forward.select(&indices);
        // A cache transfer reads or writes its request's pages without a
        // forward row; a batch carries one call kind, so every page table
        // of a cache transfer batch belongs to its calls.
        let transfers_cache = projection.calls.iter().any(|call| {
            matches!(
                call.kind,
                uniserve_ipc::CallKind::Transfer(
                    uniserve_ipc::TransferMode::CacheExport
                        | uniserve_ipc::TransferMode::CacheInstall
                )
            )
        });
        let slots = projection
            .forward
            .request_pool_indices
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        projection
            .block_tables
            .retain(|table| transfers_cache || slots.contains(&table.request_pool_idx));
        projection
            .new_cache_units
            .retain(|units| transfers_cache || slots.contains(&units.request_pool_idx));
        // Each rank copies its own share of a page through its own pool.
        projection
            .page_reads
            .retain(|copy| copy.rank as usize == rank);
        projection
            .page_writes
            .retain(|copy| copy.rank as usize == rank);
        let identities = projection
            .calls
            .iter()
            .map(|call| (call.request_key, call.call_id))
            .collect::<HashSet<_>>();
        projection
            .latent_params
            .retain(|params| identities.contains(&(params.request_key, params.call_id)));
        projection
            .decode_ranges
            .retain(|range| identities.contains(&(range.request_key, range.call_id)));
        let inputs = projection
            .calls
            .iter()
            .flat_map(|call| call.tensor_inputs().chain(call.predicate.as_ref()))
            .collect::<HashSet<_>>();
        projection
            .input_products
            .retain(|payload| inputs.contains(&payload.product));
        projection.cache_inputs.retain(|export| {
            projection
                .calls
                .iter()
                .any(|call| call.cache_input == Some(export.source))
        });
        let buffers = inputs
            .iter()
            .copied()
            .chain(
                projection
                    .calls
                    .iter()
                    .flat_map(|call| call.tensor_outputs()),
            )
            .map(|product| product.buffer_id())
            .collect::<HashSet<_>>();
        projection.buffer_allocations = placed
            .iter()
            .filter(|span| span.rank as usize == rank && buffers.contains(&span.allocation.buffer))
            .map(|span| span.allocation)
            .collect();
        if projection.calls.is_empty()
            && projection.commands.is_empty()
            && projection.page_writes.is_empty()
        {
            continue;
        }
        projection.validate()?;
        batches.push((rank, projection));
    }
    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::{output_owner, refuse_shared_storage_off_head};
    use crate::WorkerRank;
    use crate::executor::ComponentConfig;
    use std::collections::BTreeMap;
    use uniserve_ipc::MediaCall;

    #[test]
    fn a_call_joins_on_the_first_component_rank_it_is_planned_on() {
        // The component deals its units to rank 4 first.
        let ranks = [4, 5, 6, 7, 0, 1, 2, 3];

        assert_eq!(output_owner(&ranks, &[0, 1, 2, 3, 4, 5, 6, 7]), Some(4));
        assert_eq!(output_owner(&ranks, &[5, 6]), Some(5));
        // A round dealt to the ranks the component lists last has no member
        // on its first rank.
        assert_eq!(output_owner(&ranks, &[0, 1, 2, 3]), Some(0));
        assert_eq!(output_owner(&ranks, &[]), None);
    }

    fn placement(hosts: &[&str]) -> Vec<WorkerRank> {
        hosts
            .iter()
            .enumerate()
            .map(|(index, host)| WorkerRank {
                host: (*host).to_owned(),
                device: format!("cuda:{index}"),
            })
            .collect()
    }

    fn components(muxer_ranks: Vec<usize>) -> BTreeMap<String, ComponentConfig> {
        BTreeMap::from([(
            "muxer".to_owned(),
            ComponentConfig::parallel(muxer_ranks, Default::default()),
        )])
    }

    fn muxing(component: &str) -> BTreeMap<MediaCall, String> {
        BTreeMap::from([(MediaCall::Muxing, component.to_owned())])
    }

    #[test]
    fn a_muxer_on_the_head_host_is_admitted() {
        let ranks = placement(&["rank-0", "rank-0", "rank-1", "rank-1"]);
        refuse_shared_storage_off_head("rank-0", &ranks, &components(vec![0]), &muxing("muxer"))
            .expect("a muxer on the head's host serves the artifact the head opens");
        // A deployment without a muxer publishes no artifact and is unaffected.
        refuse_shared_storage_off_head("rank-0", &ranks, &components(vec![3]), &BTreeMap::new())
            .expect("a placement without muxing has no artifact to place");
    }

    #[test]
    fn a_muxer_off_the_head_host_is_refused_by_name() {
        let ranks = placement(&["rank-0", "rank-0", "rank-1", "rank-1"]);
        let error = refuse_shared_storage_off_head(
            "rank-0",
            &ranks,
            &components(vec![3]),
            &muxing("muxer"),
        )
        .expect_err("a muxer on another host cannot publish an artifact the head opens");
        let message = format!("{error:#}");
        assert!(
            message.contains("muxer"),
            "the refusal names the component: {message}"
        );
        assert!(
            message.contains("rank-1"),
            "the refusal names the muxer's host: {message}"
        );
        assert!(
            message.contains("rank-0"),
            "the refusal names the head's host: {message}"
        );

        let error = refuse_shared_storage_off_head(
            "rank-0",
            &ranks,
            &components(vec![0]),
            &muxing("output"),
        )
        .expect_err("a muxing component the placement does not bind is refused");
        assert!(format!("{error:#}").contains("output"));
    }

    /// The media reader opens the condition media the head published by name,
    /// so it is refused off the head's host as the muxer is.
    #[test]
    fn a_media_reader_off_the_head_host_is_refused_by_name() {
        let ranks = placement(&["rank-0", "rank-0", "rank-1", "rank-1"]);
        let reading = |ranks: Vec<usize>| {
            (
                BTreeMap::from([(
                    "media_reader".to_owned(),
                    ComponentConfig::parallel(ranks, Default::default()),
                )]),
                BTreeMap::from([(MediaCall::MediaReading, "media_reader".to_owned())]),
            )
        };

        let (components, calls) = reading(vec![1]);
        refuse_shared_storage_off_head("rank-0", &ranks, &components, &calls)
            .expect("a media reader on the head's host reads the published media");

        let (components, calls) = reading(vec![2]);
        let error = refuse_shared_storage_off_head("rank-0", &ranks, &components, &calls)
            .expect_err("a media reader on another host cannot open the published media");
        let message = format!("{error:#}");
        assert!(
            message.contains("media_reader") && message.contains("rank-1"),
            "the refusal names the component and its host: {message}"
        );
    }
}
