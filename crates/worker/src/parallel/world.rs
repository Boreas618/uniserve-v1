//! The rank's process world: host collectives, the torch process world and
//! the component meshes numerical code communicates over.
//!
//! Startup order (identical on every rank):
//!
//! 1. Host groups: the worker group's [`HostGroup`] over its rendezvous, or,
//!    in an expert-parallel union, the union's group and the worker's
//!    subgroup of it.
//! 2. The torch world (`distributed.initialize`): the store's first rank
//!    binds a listener beside the rendezvous and publishes its address over
//!    the host group.
//! 3. The component group plan ([`GroupPlan`]): every torch group in plan
//!    order (`distributed.create_groups`), then one `DeviceMesh` per member
//!    binding (`distributed.mesh`).
//!
//! At bind, the stream groups follow ([`World::bind_stream_groups`]): each
//! stream a component's collectives run on gets torch groups of its own.

use std::collections::BTreeMap;
use std::os::fd::IntoRawFd;
use std::time::Duration;

use tvm_ffi::Any;

use crate::config::placement::Declarations;
use crate::config::{RendezvousConfig, WorkerLaunch};
use crate::ffi::{Handle, Interpreter, integer_list, list, strings};
use crate::{Error, Result};

use super::host::{HostGroup, Rendezvous, STARTUP_TIMEOUT, bind_beside};
use super::topology::{ComponentGroups, GroupEntry, GroupPlan, MeshBinding, PlanInputs};

/// One component's bound meshes on this rank.
pub(crate) struct ComponentMeshes {
    pub plan: ComponentGroups,
    /// The numerical `DeviceMesh` handle, for a member.
    pub mesh: Option<Handle>,
    /// The temporal unit ring's `DeviceMesh` handle, for a member.
    pub units: Option<Handle>,
}

/// One torch group a component's numerical collectives run on from one of
/// its stream copies ([`World::bind_stream_groups`]).
pub(crate) struct CollectiveGroup {
    pub component: String,
    /// The stream copy (expert microbatch) whose collectives use it.
    pub copy: usize,
    /// The mesh communicators (`tp`, `pp.tp`) whose fiber it spans.
    pub names: Vec<String>,
    /// The torch process group.
    pub handle: Handle,
}

/// The rank's process world.
pub(crate) struct World {
    /// Host collectives over this worker's ranks.
    pub host: HostGroup,
    /// Host collectives over the expert-parallel union.
    pub union: Option<HostGroup>,
    /// The torch world (`distributed.initialize`), absent for a one-rank
    /// worker outside a union, which needs no process group.
    torch: Handle,
    pub components: Vec<ComponentMeshes>,
    destroy: Option<crate::ffi::Function>,
}

impl World {
    /// Form the host groups and the torch world.
    pub(crate) fn initialize(launch: &WorkerLaunch, timeout: Duration) -> Result<Self> {
        let execution = &launch.execution;
        let (host, union, store) = match &launch.expert_parallel {
            Some(union) => {
                let union_rendezvous = union.rendezvous.as_ref().ok_or_else(|| {
                    Error::Invalid("an expert-parallel union requires its store address".into())
                })?;
                let group = HostGroup::connect(
                    rendezvous(union_rendezvous)?,
                    union.rank,
                    union.size,
                    timeout,
                )?;
                let start = union.rank - execution.rank;
                let members: Vec<usize> = (start..start + execution.world_size).collect();
                let host = group.subgroup(&members)?;
                let store = store(&group, union_rendezvous)?;
                (host, Some(group), Some(store))
            }
            None => match &launch.rendezvous {
                Some(rendezvous_config) if execution.world_size > 1 => {
                    let group = HostGroup::connect(
                        rendezvous(rendezvous_config)?,
                        execution.rank,
                        execution.world_size,
                        timeout,
                    )?;
                    let store = store(&group, rendezvous_config)?;
                    (group, None, Some(store))
                }
                _ if execution.world_size > 1 => {
                    return Err(Error::Invalid(
                        "a multi-rank worker requires a rendezvous address".into(),
                    ));
                }
                _ => (HostGroup::single(execution.rank), None, None),
            },
        };

        // The store of the torch world this rank joins (the union's when
        // there is one) and this rank's place in the union; `None` crosses
        // for what a one-rank worker or a worker outside a union lacks.
        let text = |value: &str| Any::from(tvm_ffi::String::from(value));
        let optional = |value: Option<i64>| value.map(Any::from).unwrap_or_default();
        let member = launch.expert_parallel.as_ref();
        let torch = Handle::new(
            Interpreter::import("uniserve.distributed.world.initialize")?.call_values(&[
                Any::from(execution.rank as i64),
                Any::from(launch.local_rank as i64),
                Any::from(execution.world_size as i64),
                text(&execution.device.to_string()),
                launch
                    .distributed_backend
                    .as_deref()
                    .map(text)
                    .unwrap_or_default(),
                store
                    .as_ref()
                    .map(|store| text(&store.host))
                    .unwrap_or_default(),
                optional(store.as_ref().map(|store| store.port)),
                optional(store.as_ref().and_then(|store| store.listen_fd)),
                optional(member.map(|union| union.rank as i64)),
                optional(member.map(|union| union.size as i64)),
            ])?,
        )?;
        Ok(Self {
            host,
            union,
            torch,
            components: Vec::new(),
            destroy: Some(Interpreter::import("uniserve.distributed.world.destroy")?),
        })
    }

    /// The world a rank plans its partition in, before the world it belongs
    /// to starts (`distributed.world.planning`): its communicators and
    /// component meshes ([`World::bind_components`]) name the ranks of the
    /// worker and of its expert union, as [`World::initialize`]'s do, but
    /// no host or torch group is joined or created, so the rank needs no
    /// peer and any collective raises.
    pub(crate) fn planning(launch: &WorkerLaunch) -> Result<Self> {
        let execution = &launch.execution;
        let member = launch.expert_parallel.as_ref();
        let optional = |value: Option<i64>| value.map(Any::from).unwrap_or_default();
        let torch = Handle::new(
            Interpreter::import("uniserve.distributed.world.planning")?.call_values(&[
                Any::from(execution.rank as i64),
                Any::from(launch.local_rank as i64),
                Any::from(execution.world_size as i64),
                Any::from(tvm_ffi::String::from(execution.device.to_string().as_str())),
                optional(member.map(|union| union.rank as i64)),
                optional(member.map(|union| union.size as i64)),
            ])?,
        )?;
        Ok(Self {
            host: HostGroup::single(execution.rank),
            union: None,
            torch,
            components: Vec::new(),
            destroy: None,
        })
    }

    pub(crate) fn torch(&self) -> &Handle {
        &self.torch
    }

    /// Create every component's groups and meshes on the plan.
    ///
    /// A rank that holds no component fails only after every group exists,
    /// so its peers are never left waiting in a creation collective.
    pub(crate) fn bind_components(
        &mut self,
        launch: &WorkerLaunch,
        declarations: &Declarations,
    ) -> Result<()> {
        let selections = declarations.selections();
        let physical = physical_ranks(launch);
        let (plan, owns) = GroupPlan::components(&PlanInputs {
            rank: launch.execution.rank,
            physical: &physical,
            local_sync: self.union.is_some(),
            placements: &launch.components,
            selections: Some(&selections),
        })?;
        let groups = self.create_groups(&plan.entries)?;

        for component in plan.components {
            let mesh = component
                .mesh
                .as_ref()
                .map(|binding| self.mesh(binding, &groups))
                .transpose()?;
            let units = component
                .units
                .as_ref()
                .map(|binding| self.mesh(binding, &groups))
                .transpose()?;
            self.components.push(ComponentMeshes {
                plan: component,
                mesh,
                units,
            });
        }
        if !owns {
            return Err(Error::Unsupported(
                "rank has no configured component".into(),
            ));
        }
        Ok(())
    }

    /// Create the stream groups of every component that runs collectives on
    /// `copies` streams of its own (its token computation, each expert
    /// microbatch of it): one torch group per copy and multi-member fiber of
    /// the component's mesh ([`GroupPlan::streams`]). Collective over the
    /// process world in plan order, at the same point of every rank's startup;
    /// returns the groups this rank belongs to, in plan order.
    pub(crate) fn bind_stream_groups(
        &self,
        launch: &WorkerLaunch,
        declarations: &Declarations,
        copies: &BTreeMap<String, usize>,
    ) -> Result<Vec<CollectiveGroup>> {
        let selections = declarations.selections();
        let physical = physical_ranks(launch);
        let (entries, streams) = GroupPlan::streams(
            &PlanInputs {
                rank: launch.execution.rank,
                physical: &physical,
                local_sync: self.union.is_some(),
                placements: &launch.components,
                selections: Some(&selections),
            },
            copies,
        )?;
        let groups = self.create_groups(&entries)?;
        Ok(streams
            .into_iter()
            .map(|stream| CollectiveGroup {
                component: stream.component,
                copy: stream.copy,
                names: stream.names,
                handle: groups[stream.entry].clone(),
            })
            .collect())
    }

    /// Create the torch groups of `entries` in order
    /// (`distributed.create_groups`): one value per entry, the group of a
    /// kept entry.
    fn create_groups(&self, entries: &[GroupEntry]) -> Result<Vec<Handle>> {
        // Each planned group's members cross back to back, with their
        // counts.
        let groups: tvm_ffi::Array<Any> =
            Interpreter::import("uniserve.distributed.world.create_groups")?
                .call_values(&[
                    self.torch.to_any(),
                    Any::from(integer_list(
                        entries.iter().map(|entry| entry.members.len() as i64),
                    )),
                    Any::from(integer_list(
                        entries
                            .iter()
                            .flat_map(|entry| &entry.members)
                            .map(|&rank| rank as i64),
                    )),
                    Any::from(list(
                        entries.iter().map(|entry| Any::from(entry.local_sync)),
                    )),
                    Any::from(list(entries.iter().map(|entry| Any::from(entry.keep)))),
                ])?
                .try_into()
                .map_err(|error| {
                    Error::numerical("uniserve.distributed.world.create_groups", &error)
                })?;
        if groups.len() != entries.len() {
            return Err(Error::Interface(
                "create_groups returns one value per plan entry".into(),
            ));
        }
        groups.iter().map(Handle::new).collect()
    }

    /// The meshes of a component this rank is a member of.
    pub(crate) fn meshes(&self, component: &str) -> (Option<Handle>, Option<Handle>) {
        self.components
            .iter()
            .find(|meshes| meshes.plan.name == component)
            .map_or((None, None), |meshes| {
                (meshes.mesh.clone(), meshes.units.clone())
            })
    }

    fn mesh(&self, binding: &MeshBinding, groups: &[Handle]) -> Result<Handle> {
        // Each communicator's members cross back to back, with their
        // counts; a communicator of one member has no created group.
        let communicators = &binding.groups;
        Handle::new(
            Interpreter::import("uniserve.distributed.world.mesh")?.call_values(&[
                self.torch.to_any(),
                Any::from(integer_list(
                    binding.mesh.ranks.iter().map(|&rank| rank as i64),
                )),
                Any::from(integer_list(
                    binding.mesh.shape.iter().map(|&size| size as i64),
                )),
                Any::from(strings(&binding.mesh.axes)),
                Any::from(binding.rank as i64),
                Any::from(strings(communicators.iter().map(|group| group.name()))),
                Any::from(integer_list(
                    communicators.iter().map(|group| group.members.len() as i64),
                )),
                Any::from(integer_list(
                    communicators
                        .iter()
                        .flat_map(|group| &group.members)
                        .map(|&rank| rank as i64),
                )),
                Any::from(list(communicators.iter().map(|group| {
                    group
                        .entry
                        .map(|entry| groups[entry].to_any())
                        .unwrap_or_default()
                }))),
            ])?,
        )
    }

    /// Destroy the torch groups after every communication user retired.
    /// An aborted close keeps them until the process exits: a failed peer
    /// may never join the teardown.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let Some(destroy) = self.destroy.take() else {
            return Ok(());
        };
        self.components.clear();
        if aborted {
            crate::leak(std::mem::replace(&mut self.torch, Handle::none()));
            return Ok(());
        }
        destroy.invoke(&[self.torch.view()])
    }
}

impl Drop for World {
    fn drop(&mut self) {
        // A world dropped without a close (a failed startup) is abandoned:
        // its torch groups stay referenced until the process exits.
        let _ = self.close(true);
    }
}

/// The torch world rank of each of this worker's ranks: the union's ranks
/// of the worker in an expert union, its own ranks otherwise.
fn physical_ranks(launch: &WorkerLaunch) -> Vec<usize> {
    let execution = &launch.execution;
    match &launch.expert_parallel {
        Some(union) => {
            let start = union.rank - execution.rank;
            (start..start + execution.world_size).collect()
        }
        None => (0..execution.world_size).collect(),
    }
}

fn rendezvous(config: &RendezvousConfig) -> Result<Rendezvous> {
    match config.listen_fd {
        // SAFETY: the daemon passed this descriptor for this process
        // alone; the launch configuration hands it out once.
        Some(fd) => unsafe { Rendezvous::inherited(config.address.clone(), fd) },
        None => Ok(Rendezvous::connect(config.address.clone())),
    }
}

/// The address of a torch world's rendezvous store.
struct Store {
    host: String,
    port: i64,
    /// The listening descriptor on the rank that serves the store.
    listen_fd: Option<i64>,
}

/// The torch store: its first rank binds a listener beside the rendezvous
/// and publishes the address; every member learns it over `group`.
fn store(group: &HostGroup, rendezvous: &RendezvousConfig) -> Result<Store> {
    let (listener, address) = if group.rank() == 0 {
        let (listener, address) = bind_beside(&rendezvous.address)?;
        (Some(listener), Some(address))
    } else {
        (None, None)
    };
    let address = group.broadcast_bytes(address.as_deref().map(str::as_bytes), 0)?;
    let address = String::from_utf8(address)
        .map_err(|_| Error::transport("the store address is not text"))?;
    let (host, port) = address
        .rsplit_once(':')
        .ok_or_else(|| Error::transport("the store address must be host:port"))?;
    let port: i64 = port
        .parse()
        .map_err(|_| Error::transport("the store port is invalid"))?;
    // The store takes ownership of the listening descriptor.
    let listen_fd = listener.map(|listener| i64::from(listener.into_raw_fd()));
    Ok(Store {
        host: host.to_owned(),
        port,
        listen_fd,
    })
}

/// How long startup collectives wait for every member.
pub(crate) const fn startup_timeout() -> Duration {
    STARTUP_TIMEOUT
}
