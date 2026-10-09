//! Step agreement for distributed routed experts.
//!
//! Every rank of an expert union executes the same sequence of expert layer
//! calls: a rank with tokens runs its forward, every other rank joins each
//! layer with no tokens of its own. Before each step the ranks agree over the
//! union's [`HostGroup`] on which source group runs and on the step's
//! per-rank token capacity ([`ExpertParallel::agree`]); [`ExpertExchange`]
//! makes that selection from the gathered records without communication.
//!
//! During startup warmup the source ranks announce each eager expert step
//! instead ([`ExpertParallel::warmup`]): dedicated expert ranks run no
//! forward of their own and execute the announced capacities until the
//! sources announce zero.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde::Serialize;
use tvm_ffi::{Any, AnyView};

use super::host::HostGroup;
use crate::ffi::{Fields, Record};
use crate::profiling::range;
use crate::{Error, Result};

/// Bytes of one agreement record: `(tokens, leaving, kind)` as little-endian
/// `i64` words.
const RECORD_BYTES: usize = 24;

/// How the ranks of an expert union move tokens to their experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertTransport {
    /// FlashInfer's NVLink all-to-all around each rank's grouped experts.
    AllToAll,
    /// DeepEP elastic dispatch and combine.
    DeepEp,
    /// The fused MegaMoE kernel; split over DeepGEMM when experts run on
    /// dedicated ranks.
    MegaMoe,
}

impl ExpertTransport {
    /// Parse a launch exchange name. Weight prefetch (`dwdp`) moves no
    /// tokens and has no transport.
    pub fn parse(name: &str) -> Result<Option<Self>> {
        match name {
            "alltoall" => Ok(Some(Self::AllToAll)),
            "deepep" => Ok(Some(Self::DeepEp)),
            "megamoe" => Ok(Some(Self::MegaMoe)),
            "dwdp" => Ok(None),
            other => Err(Error::Invalid(format!("unknown expert exchange {other:?}"))),
        }
    }

    /// The transport name the numerical library constructs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AllToAll => "alltoall",
            Self::DeepEp => "deepep",
            Self::MegaMoe => "megamoe",
        }
    }

    /// The fused transport retains one symmetric layout of the maximum
    /// capacity; the others exchange at power-of-two capacities.
    const fn fused(self) -> bool {
        matches!(self, Self::MegaMoe)
    }
}

/// Coordinate a serialized sequence of expert calls over one rank group.
///
/// Sources executing one forward together must all have the same call kind
/// ready. Other sources and expert-only ranks participate without local rows.
/// The caller supplies gathered host records; selection performs no
/// communication.
pub struct ExpertExchange {
    rank: usize,
    ranks: usize,
    sources: Vec<Vec<usize>>,
    max_tokens: usize,
    capacities: Vec<usize>,
    kind: i64,
    active: bool,
    released: bool,
}

impl ExpertExchange {
    /// Memberships use global ranks. `rank` is the local index in `ranks`.
    /// Fused transports retain maximum-sized symmetric storage; other
    /// transports use power-of-two capacities plus the configured maximum.
    pub fn new(
        ranks: &[usize],
        rank: usize,
        memberships: &[Vec<usize>],
        attention_ranks: usize,
        max_tokens: usize,
        fused: bool,
    ) -> Result<Self> {
        if ranks.len() < 2 || rank >= ranks.len() || max_tokens == 0 {
            return Err(Error::Invalid(
                "an expert exchange spans two or more ranks and one token".into(),
            ));
        }

        let source_count = if attention_ranks == 0 {
            ranks.len()
        } else {
            attention_ranks
        };
        if source_count > ranks.len() || memberships.len() != ranks.len() {
            return Err(Error::Invalid(
                "expert source membership is incomplete".into(),
            ));
        }

        let mut sources = Vec::new();
        for (index, members) in memberships.iter().enumerate() {
            let peers = members
                .iter()
                .map(|member| ranks.iter().position(|rank| rank == member))
                .collect::<Option<Vec<_>>>()
                .filter(|peers| {
                    if peers.is_empty() {
                        return index >= source_count;
                    }

                    peers.contains(&index)
                        && peers.iter().copied().collect::<HashSet<_>>().len() == peers.len()
                        && peers
                            .iter()
                            .all(|&peer| peer < source_count && memberships[peer] == *members)
                })
                .ok_or_else(|| {
                    Error::Invalid(
                        "expert sources must declare disjoint, consistent groups of ranks that execute each forward together".into(),
                    )
                })?;

            if peers.first() == Some(&index) {
                sources.push(peers);
            }
        }

        let capacities = if fused {
            vec![max_tokens]
        } else {
            (0..usize::BITS - (max_tokens - 1).leading_zeros())
                .map(|power| 1 << power)
                .chain([max_tokens])
                .collect()
        };

        Ok(Self {
            rank,
            ranks: ranks.len(),
            sources,
            max_tokens,
            capacities,
            kind: -1,
            active: false,
            released: false,
        })
    }

    /// Select from completed host records `(tokens, leaving, kind)`.
    ///
    /// The accessor borrows the fixed receive buffer without copying it into
    /// a second record array. It must return stable values throughout this
    /// call. Ready kinds rotate in increasing cyclic order.
    pub fn select(&mut self, record: impl Fn(usize) -> (usize, bool, i64)) -> Result<usize> {
        self.active = false;
        self.released = (0..self.ranks).all(|rank| {
            let (tokens, leaving, _) = record(rank);
            tokens == 0 && leaving
        });

        let mut selected: Option<(i64, usize, bool)> = None;
        for members in &self.sources {
            let (_, _, kind) = record(members[0]);
            let mut most = 0;
            if !members.iter().all(|&rank| {
                let (tokens, _, tag) = record(rank);
                most = most.max(tokens);
                tokens != 0 && tag == kind
            }) {
                continue;
            }

            let active = members.contains(&self.rank);
            match &mut selected {
                Some((tag, tokens, local)) if *tag == kind => {
                    *tokens = (*tokens).max(most);
                    *local |= active;
                }
                Some((tag, _, _)) if (kind <= self.kind, kind) >= (*tag <= self.kind, *tag) => {}
                _ => selected = Some((kind, most, active)),
            }
        }

        let Some((kind, most, active)) = selected else {
            return Ok(0);
        };
        self.kind = kind;
        self.active = active;

        self.capacities
            .iter()
            .copied()
            .find(|&capacity| capacity >= most)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "expert step of {most} tokens exceeds the configured {} tokens per rank",
                    self.max_tokens
                ))
            })
    }

    /// Step capacities, ascending.
    pub fn capacities(&self) -> &[usize] {
        &self.capacities
    }

    /// The smallest capacity holding `tokens`.
    pub fn capacity_for(&self, tokens: usize) -> Result<usize> {
        self.capacities
            .iter()
            .copied()
            .find(|&capacity| capacity >= tokens.max(1))
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "{tokens} tokens exceed the expert exchange's {} tokens per rank",
                    self.max_tokens
                ))
            })
    }

    pub fn kind(&self) -> i64 {
        self.kind
    }

    pub fn active(&self) -> bool {
        self.active
    }

    pub fn released(&self) -> bool {
        self.released
    }
}

/// The outcome of one agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Agreement {
    /// Per-rank token capacity of the selected step; zero when no source
    /// group is ready.
    pub capacity: usize,
    /// This rank's pending forward was selected. Otherwise it joins the
    /// step with no tokens and keeps its forward for a later agreement.
    pub active: bool,
    /// Every rank is leaving without tokens: the union may shut down.
    pub released: bool,
}

/// One expert layer as `experts.layers` describes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExpertLayer {
    pub num_experts: usize,
    pub top_k: usize,
    pub hidden_size: usize,
}

impl ExpertLayer {
    pub fn from_record(record: &Record) -> Result<Self> {
        Ok(Self {
            num_experts: record.count("num_experts")?,
            top_k: record.count("top_k")?,
            hidden_size: record.count("hidden_size")?,
        })
    }
}

/// What every rank of a union must agree on before warmup enters
/// collectives: step capacities, microbatches and the expert layer
/// sequence on every rank, and the source ranks' warmup and capture plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExpertPlan {
    pub capacities: Vec<usize>,
    pub microbatches: usize,
    pub layers: Vec<ExpertLayer>,
    /// The source's graph and runner plan; `None` on expert-only ranks.
    pub calls: Option<serde_json::Value>,
}

/// The host side of an expert union: its host group and step selection.
pub struct ExpertParallel {
    host: Arc<Mutex<HostGroup>>,
    control: Mutex<Control>,
    attention_ranks: usize,
    transport: ExpertTransport,
    max_tokens: usize,
    capacities: Vec<usize>,
}

/// Step selection and the fixed agreement buffers, used under one lock.
struct Control {
    exchange: ExpertExchange,
    records: Vec<u8>,
}

impl ExpertParallel {
    /// Form the union's step protocol. Collective over the union.
    ///
    /// `sources` lists the union members (global ranks) that execute each
    /// of this rank's forwards together; it is empty on an expert-only
    /// rank. With dedicated expert ranks, the first member's `max_tokens`
    /// bounds every rank, so expert ranks size their exchanges for the
    /// source's capacity, graph padding included.
    pub fn configure(
        host: Arc<Mutex<HostGroup>>,
        attention_ranks: usize,
        sources: &[usize],
        max_tokens: usize,
        transport: ExpertTransport,
    ) -> Result<Self> {
        let (exchange, max_tokens) = {
            let group = lock(&host);
            if attention_ranks >= group.size() {
                return Err(Error::Invalid(
                    "attention ranks must leave at least one expert rank".into(),
                ));
            }
            let membership: Vec<u8> = sources
                .iter()
                .flat_map(|&rank| (rank as u64).to_le_bytes())
                .collect();
            let memberships = group
                .all_gather_bytes(&membership)?
                .iter()
                .map(|bytes| {
                    if !bytes.len().is_multiple_of(8) {
                        return Err(Error::Transport(
                            "expert source membership record is malformed".into(),
                        ));
                    }
                    Ok(bytes
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|word| u64::from_le_bytes(*word) as usize)
                        .collect())
                })
                .collect::<Result<Vec<Vec<usize>>>>()?;
            let max_tokens = if attention_ranks == 0 {
                max_tokens
            } else {
                let value = (group.rank() == 0).then(|| (max_tokens as u64).to_le_bytes());
                let received = group.broadcast_bytes(value.as_ref().map(|value| &value[..]), 0)?;
                let word: [u8; 8] = received.as_slice().try_into().map_err(|_| {
                    Error::Transport("expert capacity broadcast is malformed".into())
                })?;
                u64::from_le_bytes(word) as usize
            };
            let ranks: Vec<usize> = group.ranks().to_vec();
            let exchange = ExpertExchange::new(
                &ranks,
                group.rank(),
                &memberships,
                attention_ranks,
                max_tokens,
                transport.fused(),
            )?;
            (exchange, max_tokens)
        };
        let size = lock(&host).size();
        Ok(Self {
            host,
            capacities: exchange.capacities().to_vec(),
            control: Mutex::new(Control {
                exchange,
                records: vec![0; RECORD_BYTES * size],
            }),
            attention_ranks,
            transport,
            max_tokens,
        })
    }

    pub fn transport(&self) -> ExpertTransport {
        self.transport
    }

    /// Leading source-only members; zero for colocated experts.
    pub fn attention_ranks(&self) -> usize {
        self.attention_ranks
    }

    /// Tokens each rank may send per step.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Step capacities, ascending.
    pub fn capacities(&self) -> &[usize] {
        &self.capacities
    }

    /// The smallest step capacity holding `tokens`.
    pub fn capacity_for(&self, tokens: usize) -> Result<usize> {
        self.control().exchange.capacity_for(tokens)
    }

    /// The union's host group, shared with the transports' bootstrap.
    pub fn host(&self) -> &Arc<Mutex<HostGroup>> {
        &self.host
    }

    /// This rank's member index and the union size.
    pub fn member(&self) -> (usize, usize) {
        let group = lock(&self.host);
        (group.rank(), group.size())
    }

    fn control(&self) -> MutexGuard<'_, Control> {
        self.control.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Agree with the union on the next step.
    ///
    /// Every rank contributes the tokens its local forward sends (its live
    /// rows or its graph's padded extent), or zero when it only joins, and
    /// the call `kind` it has ready. Only source groups whose members all
    /// have one kind ready may send tokens; ready kinds take turns in cyclic
    /// order. A rank shutting down agrees with `leaving` until
    /// [`Agreement::released`]. One host round trip; no device work.
    pub fn agree(&self, tokens: usize, kind: i64, leaving: bool) -> Result<Agreement> {
        let _range = range(c"uniserve.expert.agree", None);
        let mut local = [0_u8; RECORD_BYTES];
        local[..8].copy_from_slice(&(tokens as i64).to_le_bytes());
        local[8..16].copy_from_slice(&i64::from(leaving).to_le_bytes());
        local[16..].copy_from_slice(&kind.to_le_bytes());

        let mut control = self.control();
        let Control { exchange, records } = &mut *control;
        lock(&self.host).all_gather_fixed(&local, records)?;
        let word = |rank: usize, field: usize| {
            let start = rank * RECORD_BYTES + field * 8;
            let mut value = [0_u8; 8];
            value.copy_from_slice(&records[start..start + 8]);
            i64::from_le_bytes(value)
        };
        let capacity = exchange.select(|rank| {
            (
                word(rank, 0).max(0) as usize,
                word(rank, 1) != 0,
                word(rank, 2),
            )
        })?;
        Ok(Agreement {
            capacity,
            active: capacity != 0 && exchange.active(),
            released: exchange.released(),
        })
    }

    /// Announce one eager warmup step of `capacity` tokens, or receive the
    /// sources' announcement on other ranks; zero ends warmup.
    ///
    /// With dedicated expert ranks the first source broadcasts and every
    /// rank returns its value, so expert ranks execute the same layer
    /// sequence. Colocated ranks run identical warmup plans and need no
    /// announcement: the value is returned unchanged.
    pub fn warmup(&self, capacity: usize) -> Result<usize> {
        if self.attention_ranks == 0 {
            return Ok(capacity);
        }
        let group = lock(&self.host);
        let value = (group.rank() == 0).then(|| (capacity as u64).to_le_bytes());
        let received = group.broadcast_bytes(value.as_ref().map(|value| &value[..]), 0)?;
        let word: [u8; 8] = received
            .as_slice()
            .try_into()
            .map_err(|_| Error::Transport("expert warmup announcement is malformed".into()))?;
        Ok(u64::from_le_bytes(word) as usize)
    }

    /// Wait until every rank of the union reaches this point.
    pub fn barrier(&self) -> Result<()> {
        lock(&self.host).barrier()
    }

    /// Compare every rank's plan before warmup enters collectives. Every
    /// rank must declare the same capacities, microbatches and expert layer
    /// sequence; source ranks also the same warmup and capture plan.
    pub fn compare_plans(&self, plan: &ExpertPlan) -> Result<()> {
        let encoded = serde_json::to_vec(plan)
            .map_err(|error| Error::Invariant(format!("expert plan encoding: {error}")))?;
        let plans = lock(&self.host).all_gather_bytes(&encoded)?;
        let decoded = plans
            .iter()
            .map(|bytes| {
                serde_json::from_slice::<serde_json::Value>(bytes)
                    .map_err(|_| Error::Transport("an expert plan record is malformed".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let common = |plan: &serde_json::Value| {
            (
                plan.get("capacities").cloned(),
                plan.get("microbatches").cloned(),
                plan.get("layers").cloned(),
            )
        };
        let sources = if self.attention_ranks == 0 {
            decoded.len()
        } else {
            self.attention_ranks
        };
        let consistent = decoded
            .iter()
            .all(|plan| common(plan) == common(&decoded[0]))
            && decoded[..sources].iter().all(|plan| *plan == decoded[0]);
        if !consistent {
            return Err(Error::Invalid(format!(
                "expert-parallel ranks have different plans: {}",
                serde_json::Value::Array(decoded)
            )));
        }
        Ok(())
    }
}

fn lock(host: &Mutex<HostGroup>) -> MutexGuard<'_, HostGroup> {
    host.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Host collectives over `host` as TVM-FFI functions, for numerical
/// transports that bootstrap their own device communicators (FlashInfer's
/// communication backend, DeepEP and NVSHMEM unique ids, barriers).
///
/// The map holds `rank` and `size` and the functions `all_gather(bytes) ->
/// [bytes]`, `broadcast(bytes or None, root) -> bytes`, `barrier()` and
/// `split(color, key) -> map`. Callers run them in the same order on every
/// member and never concurrently with the expert step agreement, which
/// shares the connections. TVM-FFI releases the GIL while they block.
pub fn host_functions(host: &Arc<Mutex<HostGroup>>) -> Any {
    let (rank, size) = {
        let group = lock(host);
        (group.rank(), group.size())
    };
    type HostCall = dyn Fn(&HostGroup, &[AnyView<'_>]) -> Result<Any>;
    let function = |call: Box<HostCall>| {
        let host = Arc::clone(host);
        tvm_ffi::Function::from_packed(move |args: &[AnyView<'_>]| {
            call(&lock(&host), args).map_err(|error| ffi_error(&error))
        })
    };
    let all_gather = function(Box::new(|group, args| {
        let [value] = args else {
            return Err(Error::Invalid("all_gather takes one value".into()));
        };
        let value = tvm_ffi::Bytes::try_from(*value)
            .map_err(|error| Error::numerical("host.all_gather", &error))?;
        let values = group.all_gather_bytes(value.as_slice())?;
        Ok(Any::from(crate::ffi::list(
            values
                .into_iter()
                .map(|value| Any::from(tvm_ffi::Bytes::from(value))),
        )))
    }));
    let broadcast = function(Box::new(|group, args| {
        let [value, root] = args else {
            return Err(Error::Invalid("broadcast takes a value and a root".into()));
        };
        let root = i64::try_from(*root)
            .ok()
            .and_then(|root| usize::try_from(root).ok())
            .ok_or_else(|| Error::Invalid("broadcast root is not a member index".into()))?;
        let value = if crate::ffi::is_none(&Any::from(*value)) {
            None
        } else {
            Some(
                tvm_ffi::Bytes::try_from(*value)
                    .map_err(|error| Error::numerical("host.broadcast", &error))?,
            )
        };
        let received = group.broadcast_bytes(value.as_ref().map(|value| value.as_slice()), root)?;
        Ok(Any::from(tvm_ffi::Bytes::from(received)))
    }));
    let barrier = function(Box::new(|group, _| {
        group.barrier()?;
        Ok(Any::default())
    }));
    let split = function(Box::new(|group, args| {
        let [color, key] = args else {
            return Err(Error::Invalid("split takes a color and a key".into()));
        };
        let integer = |value: &AnyView<'_>| {
            i64::try_from(*value).map_err(|error| Error::numerical("host.split", &error))
        };
        let subgroup = group.split(integer(color)?, integer(key)?)?;
        Ok(host_functions(&Arc::new(Mutex::new(subgroup))))
    }));
    Fields::new()
        .with("rank", rank as i64)
        .with("size", size as i64)
        .with("all_gather", all_gather)
        .with("broadcast", broadcast)
        .with("barrier", barrier)
        .with("split", split)
        .into_any()
}

/// A worker error raised back into the calling numerical code.
fn ffi_error(error: &Error) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), "")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn exchange(rank: usize, fused: bool) -> ExpertExchange {
        ExpertExchange::new(
            &[8, 12, 20, 40, 41],
            rank,
            &[vec![8, 20], vec![12], vec![8, 20], vec![], vec![]],
            3,
            192,
            fused,
        )
        .unwrap()
    }

    #[test]
    fn source_groups_wait_together_and_ready_kinds_take_turns() {
        for rank in 0..5 {
            let mut exchange = exchange(rank, false);
            let mut records = [
                (129, false, 2),
                (7, false, 0),
                (0, false, 2),
                (0, false, 0),
                (0, false, 0),
            ];
            assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 8);
            assert_eq!(exchange.kind(), 0);
            assert_eq!(exchange.active(), rank == 1);

            records[2].0 = 64;
            assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 192);
            assert_eq!(exchange.kind(), 2);
            assert_eq!(exchange.active(), rank == 0 || rank == 2);
            assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 8);

            records[2].2 = 1;
            assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 8);
            records[2].2 = 2;
            records[0].0 = 193;
            assert!(exchange.select(|rank| records[rank]).is_err());
        }
    }

    #[test]
    fn release_waits_for_every_rank_and_no_remaining_tokens() {
        let mut exchange = exchange(1, true);
        let mut records = [(0, true, 0); 5];
        records[4].1 = false;
        assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 0);
        assert!(!exchange.released());

        records[4].1 = true;
        records[1].0 = 1;
        assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 192);
        assert!(!exchange.released());

        records[1].0 = 0;
        assert_eq!(exchange.select(|rank| records[rank]).unwrap(), 0);
        assert!(exchange.released());
    }

    #[test]
    fn capacities_are_powers_of_two_below_the_maximum() {
        let exchange = exchange(0, false);
        assert_eq!(exchange.capacities(), [1, 2, 4, 8, 16, 32, 64, 128, 192]);
        assert_eq!(exchange.capacity_for(0).unwrap(), 1);
        assert_eq!(exchange.capacity_for(65).unwrap(), 128);
        assert_eq!(exchange.capacity_for(192).unwrap(), 192);
        assert!(exchange.capacity_for(193).is_err());
        assert_eq!(super::tests::exchange(0, true).capacities(), [192]);
    }

    /// Run `body` on every member of a loopback union of `size` members.
    fn union<T: Send>(size: usize, body: impl Fn(usize, HostGroup) -> T + Sync) -> Vec<T> {
        use super::super::host::Rendezvous;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let mut listener = Some(listener);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..size)
                .map(|rank| {
                    let rendezvous = match listener.take().filter(|_| rank == 0) {
                        Some(listener) => Rendezvous::serve(address.clone(), listener),
                        None => Rendezvous::connect(address.clone()),
                    };
                    let body = &body;
                    scope.spawn(move || {
                        let group = HostGroup::connect(
                            rendezvous,
                            rank,
                            size,
                            std::time::Duration::from_secs(30),
                        )
                        .unwrap();
                        body(rank, group)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        })
    }

    #[test]
    fn colocated_ranks_agree_on_the_selected_step_and_release_together() {
        let results = union(3, |rank, group| {
            let parallel = ExpertParallel::configure(
                Arc::new(Mutex::new(group)),
                0,
                &[rank],
                64,
                ExpertTransport::AllToAll,
            )
            .unwrap();
            // Rank 0 sends 5 tokens of kind 0, rank 2 9 tokens of kind 1;
            // kinds take turns, the idle rank joins both.
            let tokens = [5, 0, 9][rank];
            let kind = [0, 0, 1][rank];
            let first = parallel.agree(tokens, kind, false).unwrap();
            let second = parallel.agree(tokens, kind, false).unwrap();
            let leaving = parallel.agree(0, 0, true).unwrap();
            assert_eq!(parallel.warmup(16).unwrap(), 16);
            (first, second, leaving)
        });
        for (rank, (first, second, leaving)) in results.into_iter().enumerate() {
            assert_eq!((first.capacity, first.active), (8, rank == 0));
            assert_eq!((second.capacity, second.active), (16, rank == 2));
            assert_eq!(leaving.capacity, 0);
            assert!(leaving.released);
        }
    }

    #[test]
    fn dedicated_expert_ranks_borrow_the_source_bound_and_follow_announcements() {
        let results = union(3, |rank, group| {
            // Two source ranks, one expert rank whose own bound is ignored.
            let sources: Vec<usize> = if rank < 2 { vec![rank] } else { Vec::new() };
            let parallel = ExpertParallel::configure(
                Arc::new(Mutex::new(group)),
                2,
                &sources,
                [48, 48, 1][rank],
                ExpertTransport::DeepEp,
            )
            .unwrap();
            let announced = parallel.warmup(if rank == 0 { 32 } else { 0 }).unwrap();
            let plan = ExpertPlan {
                capacities: parallel.capacities().to_vec(),
                microbatches: 1,
                layers: vec![ExpertLayer {
                    num_experts: 8,
                    top_k: 2,
                    hidden_size: 16,
                }],
                calls: (rank < 2).then(|| serde_json::json!({"decode": [1, 2]})),
            };
            parallel.compare_plans(&plan).unwrap();
            (parallel.max_tokens(), announced)
        });
        assert!(results.iter().all(|&result| result == (48, 32)));
    }

    #[test]
    fn different_source_plans_are_refused_on_every_rank() {
        let results = union(2, |rank, group| {
            let parallel = ExpertParallel::configure(
                Arc::new(Mutex::new(group)),
                0,
                &[rank],
                8,
                ExpertTransport::AllToAll,
            )
            .unwrap();
            parallel
                .compare_plans(&ExpertPlan {
                    capacities: parallel.capacities().to_vec(),
                    microbatches: 1,
                    layers: Vec::new(),
                    calls: Some(serde_json::json!({"decode": [rank + 1]})),
                })
                .is_err()
        });
        assert_eq!(results, vec![true, true]);
    }

    #[test]
    fn overlapping_source_groups_are_rejected() {
        assert!(
            ExpertExchange::new(
                &[0, 1, 2],
                0,
                &[vec![0, 2], vec![1, 2], vec![0, 2]],
                0,
                32,
                false,
            )
            .is_err()
        );
    }
}
