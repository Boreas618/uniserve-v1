//! The process-group creation plan for numerical collectives.
//!
//! Model code communicates through torch process groups bound to
//! `DeviceMesh` communicators. Creating a torch group is a collective over
//! the whole process world: every rank must create every group, members or
//! not, in the same order, or the ranks deadlock. Rust fixes that order here
//! and the numerical library creates the groups on the plan
//! (`distributed.create_groups`, then `distributed.mesh` per bound mesh).
//!
//! The order (identical on every rank):
//!
//! 1. Components in alphabetical order.
//! 2. Within a component, the unit ring of a temporally distributed
//!    component first (every rank binds it), then the numerical mesh (every
//!    rank binds it for an undistributed component; only members bind their
//!    one-rank local mesh for a distributed one).
//! 3. Within a mesh, axis selections in lexicographic order of their axis
//!    name lists, and each selection's fibers in topology order.
//! 4. One group per distinct member set within one mesh binding; two
//!    bindings of the same membership are distinct ordering domains.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use uniserve_core::ComponentConfig;

use crate::{Error, Result};

/// Host components own no numerical method and no process group.
pub const HOST_COMPONENTS: [&str; 3] = ["media_reader", "video_codec", "muxer"];

pub fn is_host_component(name: &str) -> bool {
    HOST_COMPONENTS.contains(&name)
}

/// An ordered mathematical topology: `ranks` laid out row-major over
/// `shape`, with one name per axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mesh {
    /// Logical (process-world) ranks in topology order.
    pub ranks: Vec<usize>,
    pub shape: Vec<usize>,
    pub axes: Vec<String>,
}

impl Mesh {
    pub fn new(ranks: Vec<usize>, dimensions: &[(&str, usize)]) -> Result<Self> {
        let shape: Vec<usize> = dimensions.iter().map(|(_, size)| *size).collect();
        if shape.iter().product::<usize>() != ranks.len() || shape.contains(&0) {
            return Err(Error::Invalid(
                "mesh shape product must equal its membership".into(),
            ));
        }
        Ok(Self {
            ranks,
            shape,
            axes: dimensions
                .iter()
                .map(|(axis, _)| (*axis).to_owned())
                .collect(),
        })
    }

    /// A member's coordinates in the declared rank ordering.
    fn coordinate(&self, index: usize) -> Vec<usize> {
        let mut index = index;
        let mut coordinate = vec![0; self.shape.len()];
        for (axis, &size) in self.shape.iter().enumerate().rev() {
            coordinate[axis] = index % size;
            index /= size;
        }
        coordinate
    }

    /// Fibers of `selection` in topology order, each ordered by the selected
    /// axes' coordinates in selection order. This is `DeviceMesh.members`.
    pub fn members(&self, selection: &[String]) -> Result<Vec<Vec<usize>>> {
        let varying = selection
            .iter()
            .map(|axis| {
                self.axes
                    .iter()
                    .position(|name| name == axis)
                    .ok_or_else(|| Error::Invalid(format!("mesh has no axis {axis:?}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let fixed: Vec<usize> = (0..self.axes.len())
            .filter(|axis| !varying.contains(axis))
            .collect();

        // Fibers keep first-appearance order of their fixed coordinates; each
        // member keeps its varying coordinates for ordering.
        type Fiber = (Vec<usize>, Vec<(Vec<usize>, usize)>);
        let mut fibers: Vec<Fiber> = Vec::new();
        for (index, &rank) in self.ranks.iter().enumerate() {
            let coordinate = self.coordinate(index);
            let key: Vec<usize> = fixed.iter().map(|&axis| coordinate[axis]).collect();
            let position: Vec<usize> = varying.iter().map(|&axis| coordinate[axis]).collect();
            match fibers.iter_mut().find(|(fiber, _)| *fiber == key) {
                Some((_, members)) => members.push((position, rank)),
                None => fibers.push((key, vec![(position, rank)])),
            }
        }
        Ok(fibers
            .into_iter()
            .map(|(_, mut members)| {
                members.sort();
                members.into_iter().map(|(_, rank)| rank).collect()
            })
            .collect())
    }
}

/// One torch group every rank creates, in plan order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupEntry {
    /// Physical (torch world) ranks, sorted.
    pub members: Vec<usize>,
    /// Create with local synchronization: only members synchronize, which
    /// an expert union needs so a replica's groups never wait for others.
    pub local_sync: bool,
    /// Whether this rank is a member and keeps the group.
    pub keep: bool,
}

/// A communicator of one bound mesh that this rank belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshGroup {
    /// Selected axes; empty for the rank-local selection.
    pub axes: Vec<String>,
    /// Physical members in logical (fiber) order.
    pub members: Vec<usize>,
    /// Index into [`GroupPlan::entries`], `None` for a one-rank fiber.
    pub entry: Option<usize>,
}

impl MeshGroup {
    /// The communicator name: axes joined by `.`, or `local`.
    pub fn name(&self) -> String {
        if self.axes.is_empty() {
            "local".to_owned()
        } else {
            self.axes.join(".")
        }
    }
}

/// A mesh this rank binds, with the communicators it belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshBinding {
    pub mesh: Mesh,
    /// This rank's world rank (it may be outside `mesh.ranks`).
    pub rank: usize,
    /// Communicators of fibers containing this rank, in creation order.
    pub groups: Vec<MeshGroup>,
}

/// One component's process groups on this rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentGroups {
    pub name: String,
    /// Whether this rank runs the component.
    pub member: bool,
    /// The temporal unit ring, for a member of a distributed component.
    pub units: Option<MeshBinding>,
    /// The numerical mesh, for a member.
    pub mesh: Option<MeshBinding>,
}

/// The rank's complete group-creation plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupPlan {
    pub entries: Vec<GroupEntry>,
    pub components: Vec<ComponentGroups>,
}

/// Inputs that fix the plan; identical on every rank except `rank`.
pub struct PlanInputs<'a> {
    /// This rank's world rank.
    pub rank: usize,
    /// Physical rank of each world rank (the torch ranks of this instance).
    pub physical: &'a [usize],
    /// Synchronize group creation locally (expert union present).
    pub local_sync: bool,
    pub placements: &'a BTreeMap<String, ComponentConfig>,
    /// Axis selections each component's calls communicate over, from the
    /// model description; `None` selects every single axis.
    pub selections: Option<&'a BTreeMap<String, BTreeSet<Vec<String>>>>,
}

impl GroupPlan {
    /// Plan every component's groups.
    ///
    /// Fails with "rank has no configured component" only after the complete
    /// plan exists, so the caller can still create every group and release
    /// its peers before refusing.
    pub fn components(inputs: &PlanInputs<'_>) -> Result<(Self, bool)> {
        let mut plan = Self {
            entries: Vec::new(),
            components: Vec::new(),
        };
        let owns = inputs
            .placements
            .values()
            .any(|placement| placement.ranks.contains(&inputs.rank));

        for (name, placement) in inputs.placements {
            let member = placement.ranks.contains(&inputs.rank);
            let temporal = placement.distribution.is_some();
            let mut units = None;
            if temporal && !is_host_component(name) {
                // Every process creates the overlap ring, including
                // non-members; members then use a rank-local numerical mesh.
                let ring = Mesh::new(placement.ranks.clone(), &[("units", placement.ranks.len())])?;
                let selections = BTreeSet::from([vec!["units".to_owned()]]);
                let binding = plan.bind(inputs, ring, &selections)?;
                if member {
                    units = Some(binding);
                }
            }

            let mut mesh = None;
            if !temporal || member {
                let ranks = if temporal {
                    vec![inputs.rank]
                } else {
                    placement.ranks.clone()
                };
                let dimensions = placement.parallel_config.dimensions();
                let topology = Mesh::new(ranks, &dimensions)?;
                let selections = match inputs.selections {
                    None => dimensions
                        .iter()
                        .map(|(axis, _)| vec![(*axis).to_owned()])
                        .collect(),
                    Some(selections) => selections.get(name).cloned().unwrap_or_default(),
                };
                let binding = plan.bind(inputs, topology, &selections)?;
                if member {
                    mesh = Some(binding);
                }
            }

            plan.components.push(ComponentGroups {
                name: name.clone(),
                member,
                units,
                mesh,
            });
        }
        Ok((plan, owns))
    }

    /// Plan this rank's binding of `mesh`: the communicators of the
    /// selected fibers that contain the rank.
    fn bind(
        &mut self,
        inputs: &PlanInputs<'_>,
        mesh: Mesh,
        selections: &BTreeSet<Vec<String>>,
    ) -> Result<MeshBinding> {
        if mesh.ranks.iter().any(|&rank| rank >= inputs.physical.len()) {
            return Err(Error::Invalid(
                "mesh ranks disagree with the process world".into(),
            ));
        }
        let mut created = HashMap::<Vec<usize>, usize>::new();
        let mut groups = Vec::new();
        for selection in selections {
            for fiber in mesh.members(selection)? {
                let mut ordered = fiber.clone();
                ordered.sort_unstable();
                let mapped: Vec<usize> = fiber.iter().map(|&rank| inputs.physical[rank]).collect();
                if fiber.len() > 1 && !created.contains_key(&ordered) {
                    let mut members = mapped.clone();
                    members.sort_unstable();
                    created.insert(ordered.clone(), self.entries.len());
                    self.entries.push(GroupEntry {
                        members,
                        local_sync: inputs.local_sync,
                        keep: fiber.contains(&inputs.rank),
                    });
                }
                if fiber.contains(&inputs.rank) {
                    groups.push(MeshGroup {
                        axes: selection.clone(),
                        members: mapped,
                        entry: created.get(&ordered).copied(),
                    });
                }
            }
        }
        Ok(MeshBinding {
            mesh,
            rank: inputs.rank,
            groups,
        })
    }
}

#[cfg(test)]
mod tests {
    use uniserve_core::{ComponentDistribution, ParallelConfig, SequenceParallel};

    use super::*;

    fn axes(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    #[test]
    fn fibers_follow_topology_order() {
        // pp=2 x tp=2 over ranks 0..4: tp fibers are rows, pp fibers columns.
        let mesh = Mesh::new(vec![0, 1, 2, 3], &[("pp", 2), ("tp", 2)]).unwrap();
        assert_eq!(
            mesh.members(&axes(&["tp"])).unwrap(),
            vec![vec![0, 1], vec![2, 3]]
        );
        assert_eq!(
            mesh.members(&axes(&["pp"])).unwrap(),
            vec![vec![0, 2], vec![1, 3]]
        );
        // Selection order orders members within a fiber.
        assert_eq!(
            mesh.members(&axes(&["tp", "pp"])).unwrap(),
            vec![vec![0, 2, 1, 3]]
        );
        assert_eq!(
            mesh.members(&[]).unwrap(),
            vec![vec![0], vec![1], vec![2], vec![3]]
        );
    }

    #[test]
    fn every_rank_plans_the_same_groups_in_component_order() {
        let tensor = ComponentConfig::parallel(
            vec![0, 1],
            ParallelConfig {
                tensor_parallel_size: 2,
                ..ParallelConfig::default()
            },
        );
        let decoder = ComponentConfig {
            ranks: vec![2, 3],
            parallel_config: ParallelConfig::default(),
            distribution: Some(ComponentDistribution::TemporalUnits),
            units_per_rank: 1,
        };
        let sequence = ComponentConfig::parallel(
            vec![0, 1, 2, 3],
            ParallelConfig {
                tensor_parallel_size: 2,
                sequence_parallel: SequenceParallel::Ulysses { ulysses_degree: 2 },
                ..ParallelConfig::default()
            },
        );
        let placements = BTreeMap::from([
            ("text".to_owned(), tensor),
            ("decoder".to_owned(), decoder),
            ("denoiser".to_owned(), sequence),
        ]);
        let selections = BTreeMap::from([
            ("text".to_owned(), BTreeSet::from([axes(&["tp"])])),
            ("decoder".to_owned(), BTreeSet::new()),
            (
                "denoiser".to_owned(),
                BTreeSet::from([axes(&["ulysses"]), axes(&["tp"]), axes(&["tp", "ulysses"])]),
            ),
        ]);
        let physical = [0, 1, 2, 3];
        let plans: Vec<_> = (0..4)
            .map(|rank| {
                GroupPlan::components(&PlanInputs {
                    rank,
                    physical: &physical,
                    local_sync: false,
                    placements: &placements,
                    selections: Some(&selections),
                })
                .unwrap()
            })
            .collect();

        // Every rank creates the same groups in the same order.
        let order = |(plan, _): &(GroupPlan, bool)| {
            plan.entries
                .iter()
                .map(|entry| entry.members.clone())
                .collect::<Vec<_>>()
        };
        let expected = vec![
            // decoder: the unit ring (alphabetically first component).
            vec![2, 3],
            // denoiser over dims [pp, cp, tp, ulysses] = [1, 1, 2, 2]:
            // ["tp"], then ["tp", "ulysses"], then ["ulysses"].
            vec![0, 2],
            vec![1, 3],
            vec![0, 1, 2, 3],
            vec![0, 1],
            vec![2, 3],
            // text: one tp group.
            vec![0, 1],
        ];
        for plan in &plans {
            assert_eq!(order(plan), expected);
            assert!(plan.1);
        }

        // Rank 2 keeps only its own fibers and binds the decoder ring.
        let (plan, _) = &plans[2];
        let keep: Vec<_> = plan.entries.iter().map(|entry| entry.keep).collect();
        assert_eq!(keep, vec![true, true, false, true, false, true, false]);
        let decoder = &plan.components[0];
        assert_eq!(decoder.name, "decoder");
        assert_eq!(decoder.units.as_ref().unwrap().groups[0].entry, Some(0));
        assert_eq!(decoder.mesh.as_ref().unwrap().mesh.ranks, vec![2]);
        let text = &plan.components[2];
        assert!(!text.member && text.mesh.is_none());
    }

    #[test]
    fn an_unplaced_rank_still_plans_every_group() {
        let placements = BTreeMap::from([(
            "text".to_owned(),
            ComponentConfig::parallel(
                vec![0, 1],
                ParallelConfig {
                    tensor_parallel_size: 2,
                    ..ParallelConfig::default()
                },
            ),
        )]);
        let (plan, owns) = GroupPlan::components(&PlanInputs {
            rank: 2,
            physical: &[0, 1, 2],
            local_sync: true,
            placements: &placements,
            selections: None,
        })
        .unwrap();
        assert!(!owns);
        assert_eq!(plan.entries.len(), 1);
        assert!(plan.entries[0].local_sync && !plan.entries[0].keep);
    }
}
