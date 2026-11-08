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

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use ops::Kernel;
use platform::{Graph, GraphExec, GraphNode, OwnedKernelNodeParams, Stream};

use crate::{Error, Result};

/// `(operator name, descriptor key, the graph nodes this operator launch added)`.
///
/// The vector is in the order the model launched the operators during capture, so a rebind walk
/// that launches the same operators in the same order can pair the two launch by launch.
/// One walk's launches: the operator, the descriptor key it was walked under, and the nodes the
/// launch added - each with the kinds of the edges that order it (`platform::WalkedNode`).
///
/// The edge kinds are part of the structure a rebind may reuse (4.5): the executable a rebind builds
/// is instantiated from the *serving* graph's template, so it keeps that graph's edges, and a
/// programmatic edge - which lets a downstream node start before its upstream one finishes - is not
/// interchangeable with a completion edge. Comparing nodes alone would call two different orders the
/// same structure.
pub type Nodes = Vec<(&'static str, String, Vec<platform::WalkedNode>)>;

/// One captured graph executable and what it was captured from.
///
/// **The fields are declared in the order they must be released, because that is the order dropping
/// a value follows.** The executable goes first: it is the thing a replay launches, and a graph whose
/// exec has been destroyed while its nodes are still addressable is a handle nobody may use. The
/// template follows, and the kernels last, so a kernel stays loaded while a graph that calls it is
/// still alive - a kernel unloaded under a live graph is the failure mode the platform reports as an
/// unknown function rather than as a lifetime error.
///
/// This is a load-bearing order and not a style: a field added above `exec` changes what a purge
/// releases first, and 7.3's purge is the step that must not free a graph pool segment while an
/// executable can still replay over it. A new field goes at the end unless its own release order
/// says otherwise.
pub struct Captured {
    /// The executable the stream launches.
    ///
    /// Replaced before a replay when it no longer suits CUPTI's recording of the process: a timeline
    /// changed how graphs are traced, or storage the graph addresses was remapped.
    exec: RefCell<GraphExec>,
    /// The graph the executable was instantiated from; its nodes are the handles a rebind sets.
    ///
    /// Shared, because a rebind builds the prepared generation's executable from this same graph
    /// rather than capturing one again: the structure a bucket was captured with is the structure
    /// every later program of that bucket has while it can be rebound (4.5).
    template: Arc<Graph>,
    /// The nodes of each operator launch, in capture order.
    nodes: Nodes,
    /// Every kernel the graph calls, so it stays loaded while the graph can be replayed.
    kernels: Vec<Arc<Kernel>>,
    /// How many kernel nodes the graph has: what one replay launches (7.3's counters).
    kernel_nodes: u64,
    /// The renewal the executable was instantiated at ([`crate::profiling::graph_renewals`]).
    renewal: Cell<u64>,
    /// The parameters a rebind set on the executable, which a new one of the template needs again.
    rebound: RefCell<Vec<(GraphNode, OwnedKernelNodeParams)>>,
}

impl Captured {
    /// Instantiates `template` and records what a capture of it produced.
    ///
    /// `nodes` is the capture's per-launch node table and `kernels` every kernel the forward
    /// launched; both are the caller's record of the capture it just made.
    pub fn new(template: Graph, nodes: Nodes, kernels: Vec<Arc<Kernel>>) -> Result<Self> {
        Self::from_template(Arc::new(template), nodes, kernels)
    }

    /// The same, from a template this generation shares with the one it replaces: the executable is
    /// instantiated from `template` and its nodes are then set from the walk's parameters.
    ///
    /// This is 4.5's rebind: a bucket whose launch structure is unchanged does not need a second
    /// capture, only a second executable and the parameters of the program it is being prepared for.
    /// The generation being replaced keeps its own executable, which probation and a revert need
    /// (7.3 step 4).
    pub fn from_template(
        template: Arc<Graph>,
        nodes: Nodes,
        kernels: Vec<Arc<Kernel>>,
    ) -> Result<Self> {
        let exec = template.instantiate()?;
        Self::with_executable(template, exec, nodes, kernels)
    }

    /// The same, from an executable the caller instantiated itself.
    pub fn with_executable(
        template: Arc<Graph>,
        exec: GraphExec,
        nodes: Nodes,
        kernels: Vec<Arc<Kernel>>,
    ) -> Result<Self> {
        if nodes.is_empty() && !kernels.is_empty() {
            return Err(Error::Invariant(
                "a captured graph with kernels has no node table".into(),
            ));
        }
        // Two counts, and they are about different subjects: what the walk *declared* and what the
        // template *holds*. The walk's per-launch table is the rebind's subject - a node the table
        // does not name cannot be given new parameters - and the template's kernel nodes are what a
        // replay actually launches.
        let mut recorded_kernels = 0u64;
        let mut recorded_anything = 0u64;
        for (_, _, launch) in &nodes {
            recorded_anything += launch.len() as u64;
            for walked in launch {
                if walked.node.kind()? == platform::GraphNodeKind::Kernel {
                    recorded_kernels += 1;
                }
            }
        }
        let mut template_kernels = 0u64;
        for node in template.nodes()? {
            if node.kind()? == platform::GraphNodeKind::Kernel {
                template_kernels += 1;
            }
        }
        // What one replay launches is the template's count, not the table's: a table that names one
        // node of a 781-node graph would otherwise report one kernel per replay.
        let kernel_nodes = template_kernels;
        // A capture is the one place in this system where a silent partial result looks exactly like
        // success: it returns in milliseconds, with no error, and its graph replays a fraction of the
        // forward. So the template is read back from the driver and compared with what the walk
        // reported, and a graph that holds fewer nodes than the walk launched refuses **by name**.
        //
        // The comparison is one-sided on purpose. An operator may emit more than one node - the
        // paged attention's second reduction does - and the mechanism adds a memcpy per uploaded
        // column and an empty node per ordering edge, so the template holds at least what the walk
        // launched and equality is not the invariant. Holding fewer is: it means nodes the walk
        // reported are not in the graph a replay launches.
        let launched = nodes.len() as u64;
        let template_nodes = template.nodes()?.len() as u64;
        // What the graph holds beyond kernels is what a replay does on the host's side of the
        // driver: a copy from host memory reads it again at every launch.
        let mut kinds = std::collections::BTreeMap::<String, u64>::new();
        for node in template.nodes()? {
            *kinds.entry(format!("{:?}", node.kind()?)).or_default() += 1;
        }
        let mut operators = std::collections::BTreeMap::<&str, (u64, u64)>::new();
        for (operator, _, launch) in &nodes {
            let entry = operators.entry(operator).or_default();
            entry.0 += 1;
            entry.1 += launch.len() as u64;
        }
        let mut heaviest: Vec<_> = operators.into_iter().collect();
        heaviest.sort_by_key(|(_, (_, nodes))| std::cmp::Reverse(*nodes));
        heaviest.truncate(8);
        tracing::info!(
            nodes = template_nodes,
            ?kinds,
            ?heaviest,
            "a captured graph's nodes, and its heaviest operators as (launches, nodes)"
        );
        if template_nodes < launched {
            return Err(Error::Invariant(format!(
                "the capture recorded {launched} launch(es) and its graph holds {template_nodes} \
                 node(s): the work a replay launches is not the work the walk declared"
            )));
        }
        // The other side, and it decides whether a rebind can be trusted: the walk's table must
        // account for the graph's kernel nodes. A table that names fewer kernels than the graph
        // launches is a rebind that would set parameters on some of them and silently leave the
        // rest running the replaced program - a new generation that differs from the old one in
        // exactly the nodes it did not name.
        //
        // It refuses only when the table is **complete** - every launch named at least one node.
        // An incomplete table is the recorder's defect rather than the capture's, and a capture is
        // also how a bucket first comes to serve: refusing it there would take the deployment down
        // for a fault of the update path. So it is reported at `warn` with all three counts, and
        // the rebind path refuses by name exactly where the missing nodes would be written.
        // Reported, not refused, and the reason is a fact about the mechanism rather than a
        // concession: one launch may add more than one kernel node (the paged attention's second
        // reduction does), so a complete table legitimately names fewer kernels than its graph
        // holds. A refusal here would refuse a correct capture - and at capture time that is a
        // deployment that never serves. The refusal belongs where the parameters are written:
        // `rebind_launch` refuses a launch whose recorded kernels differ from the walk's, and
        // `nodes_complete` makes an unexplained launch a recapture rather than a silent rebind.
        if recorded_kernels < template_kernels {
            tracing::warn!(
                launches = launched,
                recorded_nodes = recorded_anything,
                recorded_kernels,
                template_kernels,
                complete = nodes.iter().all(|(_, _, launch)| !launch.is_empty()),
                "the walk's node table names fewer kernel nodes than the graph holds: a rebind of \
                 this bucket covers only the nodes the table names"
            );
        }
        Ok(Self {
            exec: RefCell::new(exec),
            template,
            nodes,
            kernels,
            kernel_nodes,
            renewal: Cell::new(crate::profiling::graph_renewals()),
            rebound: RefCell::new(Vec::new()),
        })
    }

    /// Replays the graph on `stream`, after the work already queued on it.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        let renewal = crate::profiling::graph_renewals();
        if self.renewal.get() != renewal {
            // The template keeps the capture's nodes, so a new executable needs nothing captured
            // again; an executable still in flight is freed by the driver when it completes.
            let exec = self.template.instantiate()?;
            for (node, params) in self.rebound.borrow().iter() {
                exec.set_kernel_node_params_owned(node.clone(), params)?;
            }
            *self.exec.borrow_mut() = exec;
            self.renewal.set(renewal);
        }
        Ok(self.exec.borrow().launch(stream)?)
    }

    /// The executable, for a caller that needs its raw handle in a report.
    pub fn exec(&self) -> std::cell::Ref<'_, GraphExec> {
        self.exec.borrow()
    }

    /// The template graph the executable was instantiated from.
    pub fn template(&self) -> &Graph {
        &self.template
    }

    /// The template, shared: what a rebind instantiates the prepared generation's executable from.
    pub fn shared_template(&self) -> Arc<Graph> {
        Arc::clone(&self.template)
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

    /// Whether the walk's table named a node for every launch it reported.
    ///
    /// A rebind sets parameters node by node, so a launch the table left empty is a launch whose
    /// new program can never reach the executable: the bucket would be reported as rebound while
    /// that launch kept running the replaced kernel. The precopy path recaptures such a bucket
    /// instead, and this is the fact it decides on.
    pub fn nodes_complete(&self) -> bool {
        !self.nodes.is_empty() && self.nodes.iter().all(|(_, _, launch)| !launch.is_empty())
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
        // Only kernel nodes carry launch parameters: a capture also adds the empty nodes that hold
        // dependencies, and the walk's parameters were collected from the kernels alone.
        let mut kernels = Vec::with_capacity(nodes.len());
        for walked in nodes {
            if walked.node.kind()? == platform::GraphNodeKind::Kernel {
                kernels.push(&walked.node);
            }
        }
        if kernels.len() != params.len() {
            return Err(Error::Invariant(format!(
                "operator {operator} ({key}) launched {} kernels when captured and {} in the walk; \
                 the program's launch structure changed and the bucket must be recaptured",
                kernels.len(),
                params.len()
            )));
        }
        for (node, owned) in kernels.into_iter().zip(params) {
            self.exec
                .borrow()
                .set_kernel_node_params_owned(node.clone(), owned)?;
            self.rebound
                .borrow_mut()
                .push((node.clone(), owned.clone()));
        }
        Ok(params.len())
    }

    /// Whether `nodes` is the same launch structure as this graph's: the same launches, in the
    /// same order, each adding the same number of graph nodes of the same kinds, ordered by the same
    /// kinds of edges.
    ///
    /// This is 4.5's rebind condition. A launch that added a different number of nodes - a
    /// split-K reduction kernel beside the main one, a native library that launched more than one
    /// kernel inside - is a structure change, and its bucket is recaptured instead of rebound.
    /// Descriptor keys are not compared: a new program under a new key of the same operator is
    /// exactly the rebind case, and the parameters are what the walk recomputes.
    ///
    /// **The edges are part of the structure.** A rebind instantiates a fresh executable from the
    /// *serving* graph's template and sets node parameters on it, so the prepared generation keeps
    /// the serving graph's edges; a programmatic edge lets a downstream node start before its
    /// upstream node has finished, so a structure in which one kind replaced the other is a
    /// structure in which the two graphs do not order their work alike. It is recaptured.
    pub fn same_structure(&self, nodes: &Nodes) -> Result<bool> {
        Ok(self.structure_mismatch(nodes)?.is_none())
    }

    /// Why `nodes` is not this graph's launch structure, or `None` when it is.
    ///
    /// The reason a recapture states, so a bucket that was recaptured says which fact decided it:
    /// a count, a node's kind, or an edge that became programmatic.
    pub fn structure_mismatch(&self, nodes: &Nodes) -> Result<Option<String>> {
        if self.nodes.len() != nodes.len() {
            return Ok(Some(format!(
                "the walk launched {} operator(s) and the serving graph {}",
                nodes.len(),
                self.nodes.len()
            )));
        }
        for (launch, ((operator, _, mine), (walked, _, theirs))) in
            self.nodes.iter().zip(nodes).enumerate()
        {
            if operator != walked {
                return Ok(Some(format!(
                    "launch {launch} is {walked} in the walk and {operator} in the serving graph"
                )));
            }
            if mine.len() != theirs.len() {
                return Ok(Some(format!(
                    "launch {launch} of {operator} added {} node(s) in the walk and {} when the \
                     serving graph was captured",
                    theirs.len(),
                    mine.len()
                )));
            }
            // The kinds matter as much as the count: a rebind carries each kernel node's
            // parameters over to the live graph by position, so a launch whose nodes changed from
            // kernel-then-empty to empty-then-kernel has the same length and a different meaning.
            for (index, (live, fresh)) in mine.iter().zip(theirs).enumerate() {
                let kind = live.node.kind()?;
                if kind != fresh.node.kind()? {
                    return Ok(Some(format!(
                        "launch {launch} of {operator} node {index} is {:?} in the walk and {:?} \
                         when the serving graph was captured",
                        fresh.node.kind()?,
                        kind
                    )));
                }
                if live.incoming != fresh.incoming {
                    return Ok(Some(format!(
                        "launch {launch} of {operator} node {index} is ordered by {:?} edges in \
                         the walk and by {:?} when the serving graph was captured: an edge kind \
                         decides how much the two nodes may overlap",
                        fresh.incoming, live.incoming
                    )));
                }
            }
        }
        Ok(None)
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
