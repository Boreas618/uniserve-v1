//! The captured graphs of one runner.

use std::hash::Hash;

use indexmap::{Equivalent, IndexMap};

use super::CUDAGraph;
use crate::ffi::Function;
use crate::{Error, Result};

/// Routed-expert token capacity a graph was captured for; `None` for a
/// runner without disaggregated experts.
pub(crate) type Capacity = Option<usize>;

/// A graph's key and capacity, borrowed: a lookup that clones no key. It
/// hashes as the owned `(key, capacity)` pair does, since a tuple and a
/// derived struct hash their fields in order.
#[derive(Hash)]
struct Slot<'a, K>(&'a K, Capacity);

impl<K: Eq> Equivalent<(K, Capacity)> for Slot<'_, K> {
    fn equivalent(&self, (key, capacity): &(K, Capacity)) -> bool {
        self.0 == key && self.1 == *capacity
    }
}

/// What a runner does with a graph key.
pub(crate) enum Lookup<'a> {
    /// Run this resident graph.
    Replay(&'a CUDAGraph),
    /// Startup has not captured the key yet: capture it now.
    Capture,
}

/// The captured graphs of one runner.
///
/// `K` is the runner's graph key: every input property its captured step
/// depends on (shape bucket, numerical variant, output selection). Each key
/// holds one graph per expert capacity. Graphs are released through the
/// binding's `release` in reverse capture order.
pub(crate) struct GraphTable<K> {
    /// Names the runner in errors.
    owner: String,
    graphs: IndexMap<(K, Capacity), CUDAGraph>,
    release: Function,
    sealed: bool,
}

impl<K: Eq + Hash + std::fmt::Debug> GraphTable<K> {
    /// An empty table whose graphs `release` drops.
    pub(crate) fn new(owner: impl Into<String>, release: Function) -> Self {
        Self {
            owner: owner.into(),
            graphs: IndexMap::new(),
            release,
            sealed: false,
        }
    }

    /// Resolve a configured graph shape. After [`Self::seal`] a missing graph
    /// is an error: startup captured every shape serving can select.
    pub(crate) fn lookup(&self, key: &K, capacity: Capacity) -> Result<Lookup<'_>> {
        match self.graphs.get(&Slot(key, capacity)) {
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
    pub(crate) fn insert(&mut self, key: K, capacity: Capacity, graph: CUDAGraph) -> Result<()> {
        let refusal = if self.sealed {
            Some("graphs are captured only before the runner is sealed".to_owned())
        } else if self.contains(&key, capacity) {
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

    pub(crate) fn contains(&self, key: &K, capacity: Capacity) -> bool {
        self.graphs.contains_key(&Slot(key, capacity))
    }

    /// Keys in capture order.
    pub(crate) fn keys(&self) -> impl Iterator<Item = &(K, Capacity)> {
        self.graphs.keys()
    }

    pub(crate) fn len(&self) -> usize {
        self.graphs.len()
    }

    /// Refuse further captures; serving may only replay.
    pub(crate) fn seal(&mut self) {
        self.sealed = true;
    }

    /// Release every graph, newest first, attempting all before returning
    /// the first failure. The caller has drained the streams that replay
    /// them. An aborted close keeps the Python graphs (and the inputs and
    /// pools they address) referenced until the process exits.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let graphs = std::mem::take(&mut self.graphs);
        self.sealed = true;
        if aborted {
            crate::leak(graphs);
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
        (
            Function::new("uniserve.cuda_graph.release", function),
            released,
        )
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

        let (release, released) = recorder();
        let mut aborted = GraphTable::new("text", release);
        aborted.insert(1, None, graph(9))?;
        aborted.close(true)?;
        assert!(released.lock().unwrap().is_empty());
        Ok(())
    }
}
