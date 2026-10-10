//! CUDA graph policy shared by every numerical runner.
//!
//! Rust decides which shapes are captured, in which order, into which
//! allocator pool, and which captured graph a call replays. Python owns the
//! capture mechanics: a binding's `capture(...)` records one bound step with
//! `torch.cuda.CUDAGraph` in the pool Rust passes and returns the graph as
//! an opaque handle, its `replay(graph, ...)` replays it, and its
//! `release(graph)` drops it (`uniserve_models.interface.graphs`).
//!
//! The pieces are independent of any runner:
//!
//! - [`bucket`]: the smallest configured size that holds a live count;
//! - [`capture_order`]: shapes largest footprint first, so later captures
//!   reuse the pool blocks and workspaces of earlier ones;
//! - [`reserve_pools`]: one allocator pool per device for an owner, or the
//!   pools of an owner it shares a stream with, registered with
//!   [`GraphStorage`] for budget accounting;
//! - [`GraphTable`]: the replay table of one runner, keyed by the runner's
//!   own graph key and an optional expert capacity.
//!
//! A table is filled at startup and sealed with the rank's graph storage;
//! afterwards a configured shape without a resident graph is an error, never
//! a silent eager fallback, because every rank of a group must make the
//! same collective calls in the same order.

use std::hash::Hash;
use std::sync::Arc;

use indexmap::IndexMap;
use tvm_ffi::Any;

use super::{GraphPool, GraphStorage, graph_storage_budget_bytes};
use crate::ffi::{Function, Handle};
use crate::{Error, Result};

/// Routed-expert token capacity a graph was captured for; `None` for a
/// runner without disaggregated experts.
pub type Capacity = Option<usize>;

/// How one numerical invocation ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphMode {
    Eager,
    Capture,
    Replay,
}

impl GraphMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Capture => "graph_capture",
            Self::Replay => "graph_replay",
        }
    }
}

/// The smallest configured size holding `count`, or `None` when every size
/// is smaller. `sizes` need not be sorted.
pub fn bucket(sizes: &[usize], count: usize) -> Option<usize> {
    sizes.iter().copied().filter(|&size| size >= count).min()
}

/// Order shapes for capture: largest `footprint` first, ties in their given
/// order, duplicates removed. Pool blocks freed by a large capture are
/// reused by the smaller ones that follow.
pub fn capture_order<T, F>(shapes: impl IntoIterator<Item = T>, footprint: F) -> Vec<T>
where
    T: PartialEq,
    F: Fn(&T) -> (usize, usize),
{
    let mut ordered: Vec<T> = Vec::new();
    for shape in shapes {
        if !ordered.contains(&shape) {
            ordered.push(shape);
        }
    }
    // A stable sort keeps the configured order among equal footprints.
    ordered.sort_by_key(|shape| std::cmp::Reverse(footprint(shape)));
    ordered
}

/// Reserve the allocator pools of graph owner `key` on `devices`.
///
/// With `share`, the owner borrows that registered owner's pools (owners on
/// one stream replay serially, so they can share intermediates); devices it
/// has no pool on get a new one. `create` is the interface's `graphs.pool`,
/// returning `[pool, id_a, id_b, total_bytes]`. The owner is registered only
/// after every pool exists, so a failure leaves the storage unchanged.
pub fn reserve_pools(
    storage: &mut GraphStorage<String, Handle>,
    create: &Function,
    key: usize,
    owner: String,
    devices: &[i32],
    share: Option<usize>,
) -> Result<Vec<Arc<GraphPool<Handle>>>> {
    let shared = storage.prepare(key, share)?;
    let mut pools = Vec::new();
    let mut defaults = Vec::new();
    for &device in devices {
        if pools
            .iter()
            .any(|pool: &Arc<GraphPool<Handle>>| pool.device == device)
        {
            continue;
        }
        if let Some(pool) = shared.iter().find(|pool| pool.device == device) {
            pools.push(Arc::clone(pool));
            continue;
        }
        let value: tvm_ffi::Array<Any> =
            create.call_as(&[(&Any::from(i64::from(device))).into()])?;
        let item = |index: usize| {
            value
                .get(index)
                .map_err(|error| Error::numerical(create.name(), &error))
        };
        if value.len() != 4 {
            return Err(Error::Interface(format!(
                "{} returns [pool, id_a, id_b, total_bytes]",
                create.name()
            )));
        }
        let number = |index: usize| -> Result<i64> {
            i64::try_from(item(index)?).map_err(|error| Error::numerical(create.name(), &error))
        };
        let id = (number(1)? as u64, number(2)? as u64);
        defaults.push((device, graph_storage_budget_bytes(number(3)?)));
        pools.push(Arc::new(GraphPool {
            device,
            id,
            value: Handle::new(item(0)?)?,
        }));
    }
    storage.commit(key, owner, pools.clone(), defaults);
    Ok(pools)
}

/// What a runner does with a graph key.
pub enum Lookup<'a> {
    /// Replay this resident graph.
    Replay(&'a Handle),
    /// Startup has not captured the key yet: capture it now.
    Capture,
}

/// Captures, replays and eager invocations of one table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphCounts {
    pub captures: u64,
    pub replays: u64,
    pub eager: u64,
}

/// The captured graphs of one runner.
///
/// `K` is the runner's graph key: every input property its captured step
/// depends on (shape bucket, numerical variant, output selection). Each key
/// holds one graph per expert capacity. Graphs are released through the
/// binding's `release` in reverse capture order.
pub struct GraphTable<K> {
    /// Names the runner in errors.
    owner: String,
    graphs: IndexMap<(K, Capacity), Handle>,
    release: Function,
    sealed: bool,
    counts: GraphCounts,
}

impl<K: Clone + Eq + Hash + std::fmt::Debug> GraphTable<K> {
    /// An empty table whose graphs `release` drops.
    pub fn new(owner: impl Into<String>, release: Function) -> Self {
        Self {
            owner: owner.into(),
            graphs: IndexMap::new(),
            release,
            sealed: false,
            counts: GraphCounts::default(),
        }
    }

    /// Resolve a configured graph shape. After [`Self::seal`] a missing graph
    /// is an error: startup captured every shape serving can select.
    pub fn lookup(&self, key: &K, capacity: Capacity) -> Result<Lookup<'_>> {
        match self.graphs.get(&(key.clone(), capacity)) {
            Some(graph) => Ok(Lookup::Replay(graph)),
            None if self.sealed => Err(Error::Invalid(format!(
                "{}: configured graph bucket {key:?} (capacity {capacity:?}) is not resident",
                self.owner
            ))),
            None => Ok(Lookup::Capture),
        }
    }

    /// Record a captured graph. Refused after seal and for a key captured
    /// before; the refused handle is released.
    pub fn insert(&mut self, key: K, capacity: Capacity, graph: Handle) -> Result<()> {
        let refusal = if self.sealed {
            Some("graphs are captured only before the runner is sealed".to_owned())
        } else if self.graphs.contains_key(&(key.clone(), capacity)) {
            Some(format!("graph bucket {key:?} was already captured"))
        } else {
            None
        };
        if let Some(reason) = refusal {
            self.release.invoke(&[graph.view()])?;
            return Err(Error::Invalid(format!("{}: {reason}", self.owner)));
        }
        self.graphs.insert((key, capacity), graph);
        Ok(())
    }

    pub fn contains(&self, key: &K, capacity: Capacity) -> bool {
        self.graphs.contains_key(&(key.clone(), capacity))
    }

    /// Keys in capture order.
    pub fn keys(&self) -> impl Iterator<Item = &(K, Capacity)> {
        self.graphs.keys()
    }

    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Refuse further captures; serving may only replay.
    pub fn seal(&mut self) {
        self.sealed = true;
    }

    pub fn sealed(&self) -> bool {
        self.sealed
    }

    /// Count one invocation by how it ran.
    pub fn record(&mut self, mode: GraphMode) {
        match mode {
            GraphMode::Eager => self.counts.eager += 1,
            GraphMode::Capture => self.counts.captures += 1,
            GraphMode::Replay => self.counts.replays += 1,
        }
    }

    pub fn counts(&self) -> GraphCounts {
        self.counts
    }

    /// Release every graph, newest first, attempting all before returning
    /// the first failure. The caller has drained the streams that replay
    /// them. An aborted close keeps the Python graphs (and the inputs and
    /// pools they address) referenced until the process exits.
    #[allow(clippy::mem_forget)]
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        let graphs = std::mem::take(&mut self.graphs);
        self.sealed = true;
        if aborted {
            std::mem::forget(graphs);
            return Ok(());
        }
        let mut failure = None;
        for (_, graph) in graphs.into_iter().rev() {
            if let Err(error) = self.release.invoke(&[graph.view()]) {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use tvm_ffi::AnyView;

    use super::*;

    /// A release function recording the graph values it drops.
    fn recorder() -> (Function, Arc<Mutex<Vec<i64>>>) {
        let released = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&released);
        let function = tvm_ffi::Function::from_packed(move |args: &[AnyView]| {
            let value = i64::try_from(args[0])?;
            sink.lock().unwrap().push(value);
            Ok(Any::from(()))
        });
        (Function::new("text.release", function), released)
    }

    fn graph(value: i64) -> Handle {
        Handle::new(Any::from(value)).unwrap()
    }

    #[test]
    fn sealed_tables_replay_captured_keys_and_refuse_the_rest() -> Result<()> {
        let (release, released) = recorder();
        let mut table = GraphTable::new("text", release);
        assert!(matches!(table.lookup(&8, None)?, Lookup::Capture));
        table.insert(8, None, graph(1))?;
        table.insert(8, Some(64), graph(2))?;
        assert!(matches!(table.lookup(&8, None)?, Lookup::Replay(_)));

        // A second capture of one key is refused and its graph released.
        assert!(table.insert(8, None, graph(3)).is_err());
        assert_eq!(*released.lock().unwrap(), vec![3]);

        table.seal();
        assert!(matches!(table.lookup(&8, Some(64))?, Lookup::Replay(_)));
        assert!(table.lookup(&4, None).is_err());
        assert!(table.insert(4, None, graph(4)).is_err());
        assert_eq!(*released.lock().unwrap(), vec![3, 4]);
        Ok(())
    }

    #[test]
    fn close_releases_graphs_newest_first_unless_aborted() -> Result<()> {
        let (release, released) = recorder();
        let mut table = GraphTable::new("text", release);
        for (key, value) in [(16, 1), (8, 2), (4, 3)] {
            table.insert(key, None, graph(value))?;
        }
        table.close(false)?;
        assert_eq!(*released.lock().unwrap(), vec![3, 2, 1]);
        assert!(table.is_empty());

        let (release, released) = recorder();
        let mut aborted = GraphTable::new("text", release);
        aborted.insert(1, None, graph(9))?;
        aborted.close(true)?;
        assert!(released.lock().unwrap().is_empty());
        Ok(())
    }

    #[test]
    fn buckets_and_capture_order_follow_configured_sizes() {
        assert_eq!(bucket(&[8, 1, 32, 4], 3), Some(4));
        assert_eq!(bucket(&[8, 1, 32, 4], 32), Some(32));
        assert_eq!(bucket(&[8, 1, 32, 4], 33), None);

        // (rows, tokens): largest product first, then tokens; duplicates
        // removed; equal footprints keep their configured order.
        let order = capture_order([(2, 16), (4, 64), (8, 8), (2, 16), (4, 16)], |&(r, t)| {
            (r * t, t)
        });
        assert_eq!(order, vec![(4, 64), (4, 16), (8, 8), (2, 16)]);
    }
}
