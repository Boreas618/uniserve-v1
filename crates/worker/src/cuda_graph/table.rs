//! The captured graphs of one runner.

use std::hash::Hash;

use indexmap::IndexMap;

use super::{CUDAGraph, GraphMode};
use crate::ffi::Function;
use crate::{Error, Result};

/// Routed-expert token capacity a graph was captured for; `None` for a
/// runner without disaggregated experts.
pub type Capacity = Option<usize>;

/// What a runner does with a graph key.
pub enum Lookup<'a> {
    /// Run this resident graph.
    Replay(&'a CUDAGraph),
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
    graphs: IndexMap<(K, Capacity), CUDAGraph>,
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
    pub fn insert(&mut self, key: K, capacity: Capacity, graph: CUDAGraph) -> Result<()> {
        let refusal = if self.sealed {
            Some("graphs are captured only before the runner is sealed".to_owned())
        } else if self.graphs.contains_key(&(key.clone(), capacity)) {
            Some(format!("graph bucket {key:?} was already captured"))
        } else {
            None
        };
        if let Some(reason) = refusal {
            self.release.invoke(&[graph.handle().view()])?;
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
            if let Err(error) = self.release.invoke(&[graph.handle().view()]) {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tvm_ffi::{Any, AnyView};

    use super::*;
    use crate::ffi::{integer_list, list};

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

    /// A one-segment capture result whose Python graph is `value`.
    fn graph(value: i64) -> CUDAGraph {
        CUDAGraph::captured(Any::from(list([
            Any::from(value),
            Any::from(integer_list([0x1000 + value])),
        ])))
        .unwrap()
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
}
