//! One captured graph: its executable, the template it came from, the nodes each operator launch
//! added, and the kernels it references.
//!
//! `Captured` is the design's struct (specs/worker.md 4.5) and it exists for two reasons beyond
//! replay:
//!
//! - **Rebind** needs the node of each operator launch. `nodes` records, per launch, the graph
//!   nodes a capture added: `(operator name, descriptor key, nodes)`, in launch order. A program
//!   whose launch structure is unchanged is rebound by setting the new launch's parameters on those
//!   nodes; one whose structure changed is recaptured instead (4.5's table).
//! - **Purge** needs the graph to hold its kernels. `kernels` holds an `Arc<Kernel>` for every
//!   kernel the graph references, so the last reference to a replaced generation's kernel drops
//!   only when the graph that calls it is rebound or destroyed. CUDA does not define what happens
//!   when a library is unloaded while a graph still calls its kernel, so the order matters
//!   (4.5, 7.3's purge step).
//!
//! The executable is instantiated from the template, and the template stays alive with it because
//! `cuGraphExecKernelNodeSetParams` addresses a node of the graph the executable was instantiated
//! from.

use std::sync::Arc;

use ops::Kernel;
use platform::{Graph, GraphExec, GraphNode, OwnedKernelNodeParams, Stream};

use crate::{Error, Result};

/// `(operator name, descriptor key, the graph nodes this operator launch added)`.
///
/// The vector is in the order the model launched the operators during capture, so a rebind walk
/// that launches the same operators in the same order can pair the two launch by launch.
pub type Nodes = Vec<(&'static str, String, Vec<GraphNode>)>;

/// One captured graph executable and what it was captured from.
pub struct Captured {
    /// The executable the stream launches.
    exec: GraphExec,
    /// The graph the executable was instantiated from; its nodes are the handles a rebind sets.
    template: Graph,
    /// The nodes of each operator launch, in capture order.
    nodes: Nodes,
    /// Every kernel the graph calls, so it stays loaded while the graph can be replayed.
    kernels: Vec<Arc<Kernel>>,
    /// How many kernel nodes the graph has: what one replay launches (7.3's counters).
    kernel_nodes: u64,
}

impl Captured {
    /// Instantiates `template` and records what a capture of it produced.
    ///
    /// `nodes` is the capture's per-launch node table and `kernels` every kernel the forward
    /// launched; both are the caller's record of the capture it just made.
    pub fn new(template: Graph, nodes: Nodes, kernels: Vec<Arc<Kernel>>) -> Result<Self> {
        let exec = template.instantiate()?;
        Self::with_executable(template, exec, nodes, kernels)
    }

    /// The same, from an executable the caller instantiated itself.
    pub fn with_executable(
        template: Graph,
        exec: GraphExec,
        nodes: Nodes,
        kernels: Vec<Arc<Kernel>>,
    ) -> Result<Self> {
        if nodes.is_empty() && !kernels.is_empty() {
            return Err(Error::Invariant(
                "a captured graph with kernels has no node table".into(),
            ));
        }
        let mut kernel_nodes = 0u64;
        for (_, _, launch) in &nodes {
            for node in launch {
                if node.kind()? == platform::GraphNodeKind::Kernel {
                    kernel_nodes += 1;
                }
            }
        }
        Ok(Self {
            exec,
            template,
            nodes,
            kernels,
            kernel_nodes,
        })
    }

    /// Replays the graph on `stream`, after the work already queued on it.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        Ok(self.exec.launch(stream)?)
    }

    /// The executable, for a caller that needs its raw handle in a report.
    pub fn exec(&self) -> &GraphExec {
        &self.exec
    }

    /// The template graph the executable was instantiated from.
    pub fn template(&self) -> &Graph {
        &self.template
    }

    /// The per-launch node table.
    pub fn nodes(&self) -> &Nodes {
        &self.nodes
    }

    /// Every kernel the graph calls.
    pub fn kernels(&self) -> &[Arc<Kernel>] {
        &self.kernels
    }

    /// How many kernel nodes the graph has: the launches one replay issues.
    pub fn kernel_nodes(&self) -> u64 {
        self.kernel_nodes
    }

    /// Whether the graph launches `operator` at all.
    pub fn uses(&self, operator: &'static str) -> bool {
        self.nodes.iter().any(|(name, _, _)| *name == operator)
    }

    /// The descriptor keys under which the graph launches `operator`, in capture order.
    pub fn descriptors(&self, operator: &'static str) -> Vec<&str> {
        self.nodes
            .iter()
            .filter(|(name, _, _)| *name == operator)
            .map(|(_, key, _)| key.as_str())
            .collect()
    }

    /// Rewrites the nodes of one operator launch with new parameters read from a precopy walk.
    ///
    /// `launch` is the ordinal of this operator's launch within the capture (0 for its first
    /// launch), and `params` holds one owned parameter set per node that launch added, in the
    /// template's node order. The number of parameter sets must equal the number of nodes the
    /// capture recorded for that launch; a difference means the new program's launch structure
    /// changed, which is a recapture, not a rebind (4.5), and is reported as an error here.
    ///
    /// The call affects only future launches of the executable: a replay already enqueued runs
    /// with the parameters it was captured with.
    ///
    /// # Preconditions
    ///
    /// Called once per `(bucket, generation)` from that generation's precopy walk; a bucket
    /// rebound for generation `N` is not rebound again without a fresh walk, because the walk's
    /// parameters are what the model computed for those launch arguments.
    pub fn rebind_launch(
        &self,
        operator: &'static str,
        launch: usize,
        params: &[OwnedKernelNodeParams],
    ) -> Result<usize> {
        let (_, key, nodes) = self
            .nodes
            .iter()
            .filter(|(name, _, _)| *name == operator)
            .nth(launch)
            .ok_or_else(|| {
                Error::Invariant(format!(
                    "the captured graph has no launch {launch} of operator {operator}"
                ))
            })?;
        if nodes.len() != params.len() {
            return Err(Error::Invariant(format!(
                "operator {operator} ({key}) added {} nodes when captured and {} in the walk; \
                 the program's launch structure changed and the bucket must be recaptured",
                nodes.len(),
                params.len()
            )));
        }
        for (node, owned) in nodes.iter().zip(params) {
            self.exec.set_kernel_node_params_owned(node.clone(), owned)?;
        }
        Ok(nodes.len())
    }

    /// Whether `nodes` is the same launch structure as this graph's: the same launches, in the
    /// same order, each adding the same number of graph nodes.
    ///
    /// This is 4.5's rebind condition. A launch that added a different number of nodes - a
    /// split-K reduction kernel beside the main one, a native library that launched more than one
    /// kernel inside - is a structure change, and its bucket is recaptured instead of rebound.
    /// Descriptor keys are not compared: a new program under a new key of the same operator is
    /// exactly the rebind case, and the parameters are what the walk recomputes.
    pub fn same_structure(&self, nodes: &Nodes) -> bool {
        self.nodes.len() == nodes.len()
            && self
                .nodes
                .iter()
                .zip(nodes)
                .all(|((operator, _, mine), (walked, _, theirs))| {
                    operator == walked && mine.len() == theirs.len()
                })
    }

    /// How many launches of `operator` the capture recorded.
    pub fn launches(&self, operator: &'static str) -> usize {
        self.nodes
            .iter()
            .filter(|(name, _, _)| *name == operator)
            .count()
    }

    /// The kernels of the graph that belong to `generation`, by (operator, descriptor key).
    ///
    /// A purge report names what a graph still holds of an old generation; the ownership itself is
    /// the `Arc`s in `kernels`.
    pub fn kernel_names(&self) -> Vec<(&'static str, &str)> {
        self.nodes
            .iter()
            .map(|(operator, key, _)| (*operator, key.as_str()))
            .collect()
    }
}

impl std::fmt::Debug for Captured {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Captured")
            .field("launches", &self.nodes.len())
            .field("kernels", &self.kernels.len())
            .finish()
    }
}
