//! Least-recently-used order over cache objects, maintained from the
//! mechanism's cache events.
//!
//! Design §4.4: an ordered structure such as this one is the policy's own,
//! built from `Cached`, `Hit`, `Released`, `Held` and `Freed`, so that a
//! shortage takes its victims in order without evaluating every cached object.
//! The mechanism records the state change; the policy keeps the order.
//!
//! One `Lru` holds one kind of object. Device KV units, host KV pages and
//! encoder-cache entries are separate orders, because a shortage of one storage
//! may not be relieved with another. Each object carries the state the policy
//! keeps of it: a page its view, an encoder entry nothing.
//!
//! The order is a doubly linked list over the slots of a vector, and an ordered
//! index finds an object's slot. An event costs one index lookup and a few slot
//! writes whatever the size of the cache, and walking the order reads the slots
//! alone: a step's events and a shortage's walk cost what they touch, not a
//! lookup per link.
//!
//! `init` builds the order from a snapshot: the mechanism states each object's
//! last use, so a policy put in mid-flight knows the order that already exists.

use std::collections::BTreeMap;

/// The least-recently-used order of objects a policy may be asked to give up,
/// with the state `V` the policy keeps of each.
///
/// The most recently used object is at the tail and the least recently used at
/// the head; [`Lru::least`] is the first victim of a shortage and
/// [`Lru::pop_least`] takes it.
#[derive(Clone, Debug)]
pub(crate) struct Lru<I: Copy + Ord, V = ()> {
    /// The slot of every object in the order.
    index: BTreeMap<I, usize>,
    /// The objects in no particular order, each linked to its neighbours in the
    /// order by slot. A removal moves the last slot into the freed one, so the
    /// vector holds exactly the objects in the order.
    slots: Vec<Slot<I, V>>,
    /// The slot of the least recently used object.
    head: Option<usize>,
    /// The slot of the most recently used object.
    tail: Option<usize>,
}

#[derive(Clone, Debug)]
struct Slot<I, V> {
    object: I,
    value: V,
    /// The slot of the object used just before this one.
    before: Option<usize>,
    /// The slot of the object used just after this one.
    after: Option<usize>,
}

impl<I: Copy + Ord, V> Default for Lru<I, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Copy + Ord, V> Lru<I, V> {
    /// An empty order.
    pub(crate) fn new() -> Self {
        Self {
            index: BTreeMap::new(),
            slots: Vec::new(),
            head: None,
            tail: None,
        }
    }

    /// An order from the least recently used object to the most recently
    /// used, as a snapshot of the mechanism states it.
    pub(crate) fn from_least_recent(objects: impl IntoIterator<Item = (I, V)>) -> Self {
        let mut lru = Self::new();
        for (object, value) in objects {
            lru.insert(object, value);
        }
        lru
    }

    /// Records an object as the most recently used with `value`, whether it is
    /// new to the order or already in it: the `Cached`, `Released` and `Held`
    /// case, which states the object anew.
    pub(crate) fn insert(&mut self, object: I, value: V) {
        if let Some(slot) = self.touch(object) {
            *slot = value;
            return;
        }

        let at = self.slots.len();
        self.slots.push(Slot {
            object,
            value,
            before: self.tail,
            after: None,
        });
        self.index.insert(object, at);
        self.attach_last(at);
    }

    /// Records an object already in the order as the most recently used and
    /// returns its state, for the caller to update: the `Hit` case. `None`, and
    /// no change, for an object not in the order.
    pub(crate) fn touch(&mut self, object: I) -> Option<&mut V> {
        let at = *self.index.get(&object)?;
        if self.tail != Some(at) {
            self.detach(at);
            self.slots[at].before = self.tail;
            self.slots[at].after = None;
            self.attach_last(at);
        }
        Some(&mut self.slots[at].value)
    }

    /// The state kept of an object, if it is in the order.
    pub(crate) fn get(&self, object: I) -> Option<&V> {
        let at = *self.index.get(&object)?;
        Some(&self.slots[at].value)
    }

    /// Takes an object out of the order and returns its state: the `Freed`
    /// case, and the object a shortage took. `None` for an object not in it.
    pub(crate) fn remove(&mut self, object: I) -> Option<V> {
        let at = self.index.remove(&object)?;
        self.detach(at);

        // The last slot moves into the freed one: its neighbours and the
        // index follow it.
        let removed = self.slots.swap_remove(at);
        if at < self.slots.len() {
            let Slot {
                object: moved,
                before,
                after,
                ..
            } = self.slots[at];
            self.index.insert(moved, at);
            match before {
                Some(before) => self.slots[before].after = Some(at),
                None => self.head = Some(at),
            }
            match after {
                Some(after) => self.slots[after].before = Some(at),
                None => self.tail = Some(at),
            }
        }
        Some(removed.value)
    }

    /// The object a shortage takes first: the least recently used.
    pub(crate) fn least(&self) -> Option<I> {
        self.head.map(|at| self.slots[at].object)
    }

    /// Takes the least recently used object out of the order.
    pub(crate) fn pop_least(&mut self) -> Option<I> {
        let object = self.least()?;
        self.remove(object);
        Some(object)
    }

    /// The objects with their state, from the least to the most recently used.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (I, &V)> + '_ {
        let mut next = self.head;
        std::iter::from_fn(move || {
            let slot = &self.slots[next?];
            next = slot.after;
            Some((slot.object, &slot.value))
        })
    }

    /// Links the slot `at`, whose `before` names the current tail, as the most
    /// recently used.
    fn attach_last(&mut self, at: usize) {
        match self.tail {
            Some(tail) => self.slots[tail].after = Some(at),
            None => self.head = Some(at),
        }
        self.tail = Some(at);
    }

    /// Splices the slot `at` out of the links, leaving it in `slots` and the
    /// index.
    fn detach(&mut self, at: usize) {
        let (before, after) = (self.slots[at].before, self.slots[at].after);
        match before {
            Some(before) => self.slots[before].after = after,
            None => self.head = after,
        }
        match after {
            Some(after) => self.slots[after].before = before,
            None => self.tail = before,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order<V>(lru: &Lru<u32, V>) -> Vec<u32> {
        lru.iter().map(|(object, _)| object).collect()
    }

    fn of(objects: &[u32]) -> Lru<u32> {
        Lru::from_least_recent(objects.iter().map(|object| (*object, ())))
    }

    #[test]
    fn the_latest_insert_is_the_tail() {
        let lru = of(&[1, 2, 3]);
        assert_eq!(order(&lru), vec![1, 2, 3]);
        assert_eq!(lru.least(), Some(1));
        assert_eq!(order(&lru).len(), 3);
    }

    #[test]
    fn touching_a_live_object_moves_it_to_the_tail() {
        let mut lru = of(&[1, 2, 3]);
        assert!(lru.touch(1).is_some());
        assert_eq!(order(&lru), vec![2, 3, 1]);
        assert!(lru.touch(3).is_some());
        assert_eq!(order(&lru), vec![2, 1, 3]);
        assert!(lru.touch(3).is_some());
        assert_eq!(order(&lru), vec![2, 1, 3]);
    }

    #[test]
    fn touching_an_object_not_in_the_order_changes_nothing() {
        let mut lru = of(&[1, 2]);
        assert!(lru.touch(9).is_none());
        assert_eq!(order(&lru), vec![1, 2]);
    }

    #[test]
    fn inserting_a_live_object_restates_it_as_the_most_recent() {
        let mut lru = Lru::from_least_recent([(1, 'a'), (2, 'b'), (3, 'c')]);
        lru.insert(1, 'z');
        assert_eq!(order(&lru), vec![2, 3, 1]);
        assert_eq!(lru.get(1), Some(&'z'));
        assert_eq!(order(&lru).len(), 3);

        *lru.touch(2).unwrap() = 'y';
        assert_eq!(order(&lru), vec![3, 1, 2]);
        assert_eq!(lru.get(2), Some(&'y'));
    }

    #[test]
    fn removing_an_object_keeps_the_rest_ordered() {
        let mut lru = Lru::from_least_recent([(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]);
        assert_eq!(lru.remove(2), Some('b'));
        assert_eq!(order(&lru), vec![1, 3, 4]);
        lru.touch(1);
        assert_eq!(order(&lru), vec![3, 4, 1]);
        assert_eq!(lru.remove(4), Some('d'));
        assert_eq!(order(&lru), vec![3, 1]);
        assert_eq!(lru.remove(4), None);

        // The objects that remain keep their state wherever they are stored.
        assert_eq!(lru.get(3), Some(&'c'));
        assert_eq!(lru.get(1), Some(&'a'));
        lru.insert(5, 'e');
        assert_eq!(order(&lru), vec![3, 1, 5]);
        assert_eq!(lru.remove(3), Some('c'));
        assert_eq!(order(&lru), vec![1, 5]);
        assert_eq!(lru.get(5), Some(&'e'));
    }

    #[test]
    fn popping_takes_the_least_recently_used_first() {
        let mut lru = of(&[7, 8, 9]);
        assert_eq!(lru.pop_least(), Some(7));
        assert_eq!(lru.pop_least(), Some(8));
        assert_eq!(lru.pop_least(), Some(9));
        assert_eq!(lru.pop_least(), None);
        assert!(order(&lru).is_empty());
    }

    #[test]
    fn a_snapshot_order_is_kept_as_stated() {
        let lru = of(&[5, 6, 7]);
        assert_eq!(order(&lru), vec![5, 6, 7]);
        assert_eq!(lru.least(), Some(5));
    }
}
