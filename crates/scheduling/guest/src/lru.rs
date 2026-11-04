//! Least-recently-used order over cache objects, maintained from the
//! mechanism's cache events.
//!
//! Design §4.4: an ordered structure such as this one is the policy's own,
//! built from `Cached`, `Hit` and `Freed`, so that a shortage takes its
//! victims in order without evaluating every cached object. The mechanism
//! records the state change; the policy keeps the order.
//!
//! One `Lru` holds one kind of object. Device KV units, host KV pages and
//! encoder-cache entries are three separate orders, because a shortage of one
//! storage may not be relieved with another.
//!
//! `init` builds the order from a snapshot: the mechanism states each object's
//! last use, so a policy put in mid-flight knows the order that already exists.

use std::collections::HashMap;
use std::hash::Hash;

/// The least-recently-used order of objects a policy may be asked to give up.
///
/// The most recently used object is at the tail and the least recently used at
/// the head; [`Lru::least`] is the first victim of a shortage and
/// [`Lru::pop_least`] takes it.
#[derive(Clone, Debug)]
pub struct Lru<I: Copy + Eq + Hash> {
    /// The links of every object in the order.
    nodes: HashMap<I, Node<I>>,
    /// The least recently used object.
    head: Option<I>,
    /// The most recently used object.
    tail: Option<I>,
}

#[derive(Clone, Copy, Debug)]
struct Node<I> {
    /// The object used just before this one.
    before: Option<I>,
    /// The object used just after this one.
    after: Option<I>,
}

impl<I: Copy + Eq + Hash> Default for Lru<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Copy + Eq + Hash> Lru<I> {
    /// An empty order.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            head: None,
            tail: None,
        }
    }

    /// An order from the least recently used object to the most recently
    /// used, as a snapshot of the mechanism states it.
    pub fn from_least_recent(ids: impl IntoIterator<Item = I>) -> Self {
        let mut lru = Self::new();
        for id in ids {
            lru.touch(id);
        }
        lru
    }

    /// Records an object as the most recently used, whether it is new to the
    /// order or already in it: the `Cached` and `Hit` case.
    pub fn touch(&mut self, object: I) {
        if self.nodes.contains_key(&object) {
            self.unlink(object);
        }
        let before = self.tail;
        self.nodes.insert(
            object,
            Node {
                before,
                after: None,
            },
        );
        if let Some(before) = before
            && let Some(node) = self.nodes.get_mut(&before)
        {
            node.after = Some(object);
        } else {
            self.head = Some(object);
        }
        self.tail = Some(object);
    }

    /// Takes an object out of the order, returning whether it was in it: the
    /// `Freed` case, and the object a shortage took.
    pub fn remove(&mut self, object: I) -> bool {
        if !self.nodes.contains_key(&object) {
            return false;
        }
        self.unlink(object);
        self.nodes.remove(&object);
        true
    }

    /// The object a shortage takes first: the least recently used.
    pub fn least(&self) -> Option<I> {
        self.head
    }

    /// Takes the least recently used object out of the order.
    pub fn pop_least(&mut self) -> Option<I> {
        let object = self.head?;
        self.remove(object);
        Some(object)
    }

    /// The objects from the least to the most recently used.
    pub fn iter(&self) -> impl Iterator<Item = I> + '_ {
        let mut next = self.head;
        std::iter::from_fn(move || {
            let object = next?;
            next = self.nodes.get(&object).and_then(|node| node.after);
            Some(object)
        })
    }

    /// How many objects are in the order.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether no object is in the order.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Splices one object out of the links, leaving it in `nodes`.
    fn unlink(&mut self, object: I) {
        let Some(node) = self.nodes.get(&object).copied() else {
            return;
        };
        match node.before {
            Some(before) => {
                if let Some(before) = self.nodes.get_mut(&before) {
                    before.after = node.after;
                }
            }
            None => self.head = node.after,
        }
        match node.after {
            Some(after) => {
                if let Some(after) = self.nodes.get_mut(&after) {
                    after.before = node.before;
                }
            }
            None => self.tail = node.before,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order<I: Copy + Eq + Hash>(lru: &Lru<I>) -> Vec<I> {
        lru.iter().collect()
    }

    #[test]
    fn the_latest_touch_is_the_tail() {
        let mut lru = Lru::<u32>::new();
        lru.touch(1);
        lru.touch(2);
        lru.touch(3);
        assert_eq!(order(&lru), vec![1, 2, 3]);
        assert_eq!(lru.least(), Some(1));
    }

    #[test]
    fn touching_a_live_object_moves_it_to_the_tail() {
        let mut lru = Lru::<u32>::new();
        for object in [1, 2, 3] {
            lru.touch(object);
        }
        lru.touch(1);
        assert_eq!(order(&lru), vec![2, 3, 1]);
        lru.touch(3);
        assert_eq!(order(&lru), vec![2, 1, 3]);
    }

    #[test]
    fn removing_an_object_keeps_the_rest_ordered() {
        let mut lru = Lru::<u32>::new();
        for object in [1, 2, 3, 4] {
            lru.touch(object);
        }
        assert!(lru.remove(2));
        assert_eq!(order(&lru), vec![1, 3, 4]);
        lru.touch(1);
        assert_eq!(order(&lru), vec![3, 4, 1]);
        assert!(lru.remove(4));
        assert_eq!(order(&lru), vec![3, 1]);
        assert!(!lru.remove(4));
    }

    #[test]
    fn popping_takes_the_least_recently_used_first() {
        let mut lru = Lru::<u32>::new();
        for object in [7, 8, 9] {
            lru.touch(object);
        }
        assert_eq!(lru.pop_least(), Some(7));
        assert_eq!(lru.pop_least(), Some(8));
        assert_eq!(lru.pop_least(), Some(9));
        assert_eq!(lru.pop_least(), None);
        assert!(lru.is_empty());
    }

    #[test]
    fn a_snapshot_order_is_kept_as_stated() {
        let lru = Lru::from_least_recent([5, 6, 7]);
        assert_eq!(order(&lru), vec![5, 6, 7]);
        assert_eq!(lru.least(), Some(5));
    }
}
