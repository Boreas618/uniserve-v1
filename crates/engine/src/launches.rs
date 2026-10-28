//! The launches a world's ranks start with, computed without starting any
//! process.
//!
//! An engine starts its world's ranks from the launch arguments of their
//! worker groups (`SchedulerHandle::start_all`): each rank's process gets
//! its executable, its environment and, once it registered, its launch
//! descriptor. A daemon that plans a world before starting it reads the same
//! launches from the world's spec here ([`rank_launches`]): which model each
//! rank loads, on which device, and in which parallel position, exactly as
//! the start states them. Only what a started launch has and a planned one
//! cannot (the claim at the Storage Server, the rendezvous sockets, the
//! engine's address) is absent.

use std::collections::HashMap;

use proto::request::wire::EngineSpec;

use crate::core::{EngineConfig, SchedulerHandle};

/// The ranks that load their weights together: their union's when the
/// world's replicas share their experts, else their worker group's.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Agreement {
    /// The expert-parallel union of the deployment.
    ExpertUnion,
    /// The worker group of this identity.
    Group(String),
}

/// One rank's launch, as the world's start states it.
#[derive(Debug, Clone)]
pub struct RankLaunch {
    /// The request scheduler that owns this rank; dedicated experts have none.
    pub scheduler: Option<usize>,
    /// The rank's host, by host identity.
    pub host: String,
    /// The rank's worker group and its rank there.
    pub group: String,
    pub rank: u32,
    /// The ranks the rank loads its weights with, and its index among them
    /// (its union rank, or its rank in its group).
    pub agreement: Agreement,
    pub agreement_index: u32,
    /// The executable the rank's process runs.
    pub executable: String,
    /// The variables its process is started with, over the daemon's.
    pub environment: HashMap<String, String>,
    /// Its launch descriptor; its Storage Server claim and rendezvous are
    /// unset.
    pub descriptor: ipc::LaunchDescriptor,
}

/// Every rank's launch of the world `spec` states, in the order of its
/// groups' arguments and their ranks.
///
/// # Errors
///
/// Fails as the world's start would before starting any process: on a
/// spec that does not resolve, groups that do not form its schedulers, or
/// a rank whose descriptor cannot be built.
pub fn rank_launches(spec: &EngineSpec) -> anyhow::Result<Vec<RankLaunch>> {
    let config = EngineConfig::from_spec(spec)?;
    let groups = SchedulerHandle::group_arguments(&config)?;
    let arguments = groups.arguments;
    let mut launches = Vec::new();
    for (index, args) in arguments.iter().enumerate() {
        let executable = args
            .executable
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("the worker executable's path is not valid text"))?
            .to_owned();
        for rank in 0..args.world_size() {
            // A planned rank holds no claim: it retains nothing.
            let storage = ipc::StorageLaunch {
                socket: String::new(),
                deployment: args.deployment.clone(),
                claim: [0; 16],
                host_kv_cache: args.host_kv_cache,
                tcp: None,
                peers: Vec::new(),
                world: args.world,
                standby: false,
                standby_memory: args.standby_memory,
            };
            let descriptor = args.launch_descriptor(rank, storage, None, None, None)?;
            let (agreement, agreement_index) = match &descriptor.expert_parallel {
                Some(union) => (Agreement::ExpertUnion, union.rank),
                None => (Agreement::Group(args.worker_id.clone()), rank),
            };
            launches.push(RankLaunch {
                scheduler: args
                    .role
                    .serves_requests()
                    .then_some(index / groups.per_core),
                host: args.ranks[rank as usize].host.clone(),
                group: args.worker_id.clone(),
                rank,
                agreement,
                agreement_index,
                executable: executable.clone(),
                environment: args.rank_environment(rank)?,
                descriptor,
            });
        }
    }
    Ok(launches)
}

/// The launches of the world `spec` states whose groups hold device
/// storage: every rank of a group with a rank on a CUDA device, in
/// [`rank_launches`]'s order. A group of CPU ranks alone (media codecs,
/// muxers and readers) holds no weights or cache on a GPU, so it has nothing
/// to plan or assign. A group is planned whole, since its ranks register
/// with their engine by their rank in the group.
///
/// # Errors
///
/// Fails as [`rank_launches`] does.
pub fn device_launches(spec: &EngineSpec) -> anyhow::Result<Vec<RankLaunch>> {
    let launches = rank_launches(spec)?;
    let groups: std::collections::BTreeSet<String> = launches
        .iter()
        .filter(|launch| launch.descriptor.device.starts_with("cuda:"))
        .map(|launch| launch.group.clone())
        .collect();
    Ok(launches
        .into_iter()
        .filter(|launch| groups.contains(&launch.group))
        .collect())
}

/// Describes target weights and measures source links on the actual node of
/// each rank [`device_launches`] names, answering in that order. Planning
/// uses the same process ownership and registration as serving; closing this
/// operation's process connection releases all planning ranks.
pub fn plan_ranks(
    spec: &EngineSpec,
    process_address: &str,
    source_worlds: &[u64],
) -> anyhow::Result<Vec<storage::manifest::weights::PlannedRank>> {
    use std::collections::BTreeMap;
    use std::time::{Duration, Instant};

    use crate::worker::{claims::Claims, processes, registry};
    use anyhow::Context;

    let config = EngineConfig::from_spec(spec)?;
    let arguments = SchedulerHandle::group_arguments(&config)?.arguments;
    let launches = device_launches(spec)?;
    let processes = processes::ProcessClient::connect(process_address, &spec.deployment)?;
    let claims = Claims::claim(&arguments, &processes)?;
    let registry =
        registry::WorkerRegistry::bind(launches.iter().any(|rank| rank.host != spec.host))?;
    let reachable = processes::lock(&processes)?.address(&spec.host)?;
    let mut generations = BTreeMap::new();
    for launch in &launches {
        let generation = match generations.entry(launch.group.clone()) {
            std::collections::btree_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                *entry.insert(registry::lock(&registry)?.begin_launch(&launch.group))
            }
        };
        let address = registry::lock(&registry)?.address(if launch.host == spec.host {
            None
        } else {
            reachable
        })?;
        processes::lock(&processes)?.spawn(proto::process::Spawn {
            host: launch.host.clone(),
            group: launch.group.clone(),
            rank: launch.rank,
            generation,
            executable: launch.executable.clone(),
            arguments: vec![
                "worker".into(),
                "--plan".into(),
                "--engine".into(),
                address,
                "--worker-id".into(),
                launch.group.clone(),
                "--rank".into(),
                launch.rank.to_string(),
                "--generation".into(),
                generation.to_string(),
            ],
            environment: launch.environment.clone(),
            ..Default::default()
        })?;
    }
    let mut channels = BTreeMap::new();
    for (group, generation) in &generations {
        let ranks: Vec<_> = launches
            .iter()
            .filter(|launch| &launch.group == group)
            .collect();
        let registered = registry::collect(&registry, group, ranks.len(), || {
            let mut processes = processes::lock(&processes)?;
            for rank in &ranks {
                if let Some(status) = processes.exited(group, *generation, rank.rank) {
                    anyhow::bail!("planning rank {} of {group} exited: {status}", rank.rank);
                }
            }
            Ok(())
        })?;
        for (rank, mut channel) in registered.into_iter().enumerate() {
            let launch = ranks
                .iter()
                .find(|launch| launch.rank == rank as u32)
                .context("registered planning rank has no launch")?;
            let mut descriptor = launch.descriptor.clone();
            descriptor.storage = claims.launch(&launch.host)?;
            channel.send(&descriptor)?;
            channel.send(&source_worlds)?;
            channels.insert((group.clone(), rank as u32), channel);
        }
    }
    let deadline = Instant::now() + Duration::from_secs(900);
    launches
        .iter()
        .map(|launch| {
            let channel = channels
                .get_mut(&(launch.group.clone(), launch.rank))
                .context("planning rank did not register")?;
            loop {
                if let Some(result) =
                    channel.try_recv::<Result<storage::manifest::weights::PlannedRank, String>>()?
                {
                    return result.map_err(anyhow::Error::msg);
                }
                let remaining = deadline.saturating_duration_since(Instant::now());
                anyhow::ensure!(!remaining.is_zero(), "rank planning exceeded 900 seconds");
                channel.wait(remaining)?;
            }
        })
        .collect()
}
