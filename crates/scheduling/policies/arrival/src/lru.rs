//! Least-recently-used orders over cache objects, maintained from the
//! mechanism's cache events.
//!
//! Design §4.4: an ordered structure such as this one is the policy's own,
//! built from `Cached`, `Hit`, `Released`, `Held` and `Freed`, so that a
//! shortage takes its victims in order without evaluating every cached object.
//! The mechanism records the state change; the policy keeps the order.
//!
//! One `Lru` holds one kind of object, in one or more orders: each object is in
//! exactly one of them, and an object moves to another when the state the
//! policy keeps of it changes which one it belongs to. Device KV units, host KV
//! pages and encoder-cache entries are separate orders, because a shortage of
//! one storage may not be relieved with another (`crate::pages`). Each object
//! carries the state the policy keeps of it: a page its view and whether it
//! waits to be offered to the host, an encoder entry nothing.
//!
//! The orders are doubly linked lists over one array of slots, and one ordered
//! index finds an object's slot. An event costs one index lookup and a few slot
//! writes whatever the size of the cache, moving an object to another order
//! included, and walking an order reads its own slots alone: a step's events
//! and a shortage's walk cost what they touch, not a lookup per link. The array
//! grows a chunk of slots at a time and never moves the slots it holds
//! ([`Slots`]), so the event that adds an object costs the same at every size.
//!
//! `init` builds the orders from a snapshot: the mechanism states each object's
//! last use, so a policy put in mid-flight knows the order that already exists.

use std::collections::BTreeMap;
use std::ops::{Index, IndexMut};

/// Least-recently-used orders of objects a policy may be asked to give up,
/// with the state `V` the policy keeps of each.
///
/// The orders are numbered from zero. In each, the most recently used object is
/// at the tail and the least recently used at the head; [`Lru::iter`] reads an
/// order from its head and [`Lru::pop_least`] takes it.
#[derive(Clone, Debug)]
pub(crate) struct Lru<I: Copy + Ord, V = ()> {
    /// The slot of every object, whichever order holds it.
    index: BTreeMap<I, usize>,
    /// The objects in no particular order, each linked to its neighbours in its
    /// order by slot. A removal moves the last slot into the freed one, so the
    /// array holds exactly the objects of the orders.
    slots: Slots<Slot<I, V>>,
    /// The least and the most recently used slot of each order.
    ends: Vec<Ends>,
}

#[derive(Clone, Debug)]
struct Slot<I, V> {
    object: I,
    value: V,
    /// The order that holds the object.
    order: usize,
    /// The slot of the object of the same order used just before this one.
    before: Option<usize>,
    /// The slot of the object of the same order used just after this one.
    after: Option<usize>,
}

/// The ends of one order: `None` both for an empty order.
#[derive(Clone, Copy, Debug, Default)]
struct Ends {
    /// The slot of the least recently used object.
    head: Option<usize>,
    /// The slot of the most recently used object.
    tail: Option<usize>,
}

impl<I: Copy + Ord, V> Lru<I, V> {
    /// `orders` empty orders.
    pub(crate) fn new(orders: usize) -> Self {
        Self {
            index: BTreeMap::new(),
            slots: Slots::default(),
            ends: vec![Ends::default(); orders],
        }
    }

    /// Records an object as the most recently used of `order` with `value`,
    /// whether it is new or already in an order: the `Cached` case, which
    /// states the object anew. Returns the state now kept.
    ///
    /// # Panics
    ///
    /// When `order` is not one of the orders.
    pub(crate) fn insert(&mut self, object: I, value: V, order: usize) -> &mut V {
        let at = match self.index.get(&object).copied() {
            Some(at) => {
                self.detach(at);
                self.slots[at].value = value;
                at
            }
            None => {
                let at = self.slots.len();
                self.slots.push(Slot {
                    object,
                    value,
                    order,
                    before: None,
                    after: None,
                });
                self.index.insert(object, at);
                at
            }
        };
        self.attach_last(at, order);
        &mut self.slots[at].value
    }

    /// Records an object already in an order as used just now: `change`
    /// updates its state and names the order that state belongs to, and the
    /// object becomes that order's most recently used. Returns the state, for
    /// the caller to update further: the `Hit`, `Released` and `Held` case.
    /// `None`, and no change, for an object in no order.
    ///
    /// # Panics
    ///
    /// When `change` names an order that is not one of the orders.
    pub(crate) fn touch(
        &mut self,
        object: I,
        change: impl FnOnce(&mut V) -> usize,
    ) -> Option<&mut V> {
        let at = *self.index.get(&object)?;
        let order = change(&mut self.slots[at].value);
        self.detach(at);
        self.attach_last(at, order);
        Some(&mut self.slots[at].value)
    }

    /// The state kept of an object, if it is in an order.
    pub(crate) fn get(&self, object: I) -> Option<&V> {
        let at = *self.index.get(&object)?;
        Some(&self.slots[at].value)
    }

    /// The state kept of an object, for the caller to update, if it is in an
    /// order; its place stays. The caller changes nothing that decides which
    /// order holds it.
    pub(crate) fn get_mut(&mut self, object: I) -> Option<&mut V> {
        let at = *self.index.get(&object)?;
        Some(&mut self.slots[at].value)
    }

    /// Takes an object out of its order and returns its state: the `Freed`
    /// case, and the object a shortage took. `None` for an object in no order.
    pub(crate) fn remove(&mut self, object: I) -> Option<V> {
        let at = self.index.remove(&object)?;
        self.detach(at);

        // The last slot moves into the freed one: its neighbours, the ends of
        // its order and the index follow it.
        let removed = self.slots.swap_remove(at)?;
        if at < self.slots.len() {
            let Slot {
                object: moved,
                order,
                before,
                after,
                ..
            } = self.slots[at];
            self.index.insert(moved, at);
            match before {
                Some(before) => self.slots[before].after = Some(at),
                None => self.ends[order].head = Some(at),
            }
            match after {
                Some(after) => self.slots[after].before = Some(at),
                None => self.ends[order].tail = Some(at),
            }
        }
        Some(removed.value)
    }

    /// Takes the least recently used object of `order` out of it.
    pub(crate) fn pop_least(&mut self, order: usize) -> Option<I> {
        let object = self.slots[self.ends[order].head?].object;
        self.remove(object);
        Some(object)
    }

    /// The objects of `order` with their state, from the least to the most
    /// recently used.
    pub(crate) fn iter(&self, order: usize) -> impl Iterator<Item = (I, &V)> + '_ {
        let mut next = self.ends[order].head;
        std::iter::from_fn(move || {
            let slot = &self.slots[next?];
            next = slot.after;
            Some((slot.object, &slot.value))
        })
    }

    /// Links the slot `at`, which no order links, as the most recently used of
    /// `order`.
    fn attach_last(&mut self, at: usize, order: usize) {
        let tail = self.ends[order].tail;
        let slot = &mut self.slots[at];
        slot.order = order;
        slot.before = tail;
        slot.after = None;
        match tail {
            Some(tail) => self.slots[tail].after = Some(at),
            None => self.ends[order].head = Some(at),
        }
        self.ends[order].tail = Some(at);
    }

    /// Splices the slot `at` out of its order's links, leaving it in `slots`
    /// and the index.
    fn detach(&mut self, at: usize) {
        let Slot {
            order,
            before,
            after,
            ..
        } = self.slots[at];
        match before {
            Some(before) => self.slots[before].after = after,
            None => self.ends[order].head = after,
        }
        match after {
            Some(after) => self.slots[after].before = before,
            None => self.ends[order].tail = before,
        }
    }
}

/// The slots per chunk of [`Slots`].
const CHUNK: usize = 4096;

/// An array of slots stored in chunks of [`CHUNK`]: every chunk but the last is
/// full, and the last holds at least one slot.
///
/// A vector that doubles when it is full copies every slot it holds in the one
/// call that crosses its capacity, a cost in proportion to the cache within one
/// call's fuel (about 48 fuel a page of a policy's page record). A new chunk
/// moves no slot, and the vector of chunks it grows holds one handle a chunk.
#[derive(Clone, Debug)]
struct Slots<T> {
    chunks: Vec<Vec<T>>,
    len: usize,
}

impl<T> Default for Slots<T> {
    fn default() -> Self {
        Self {
            chunks: Vec::new(),
            len: 0,
        }
    }
}

impl<T> Slots<T> {
    /// The slots held.
    fn len(&self) -> usize {
        self.len
    }

    /// Appends `slot` at index `len`.
    fn push(&mut self, slot: T) {
        match self.chunks.last_mut() {
            Some(chunk) if chunk.len() < CHUNK => chunk.push(slot),
            _ => {
                let mut chunk = Vec::with_capacity(CHUNK);
                chunk.push(slot);
                self.chunks.push(chunk);
            }
        }
        self.len += 1;
    }

    /// Takes the slot at `at` out, moving the last slot into its place; `None`,
    /// and no change, for an index past the slots.
    fn swap_remove(&mut self, at: usize) -> Option<T> {
        if at >= self.len {
            return None;
        }
        let chunk = self.chunks.last_mut()?;
        let last = chunk.pop()?;
        if chunk.is_empty() {
            self.chunks.pop();
        }
        self.len -= 1;
        if at == self.len {
            return Some(last);
        }
        Some(std::mem::replace(&mut self[at], last))
    }
}

impl<T> Index<usize> for Slots<T> {
    type Output = T;

    fn index(&self, at: usize) -> &T {
        &self.chunks[at / CHUNK][at % CHUNK]
    }
}

impl<T> IndexMut<usize> for Slots<T> {
    fn index_mut(&mut self, at: usize) -> &mut T {
        &mut self.chunks[at / CHUNK][at % CHUNK]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order<V>(lru: &Lru<u32, V>, order: usize) -> Vec<u32> {
        lru.iter(order).map(|(object, _)| object).collect()
    }

    fn of(objects: &[u32]) -> Lru<u32> {
        let mut lru = Lru::new(1);
        for object in objects {
            lru.insert(*object, (), 0);
        }
        lru
    }

    fn lettered(objects: &[(u32, char)]) -> Lru<u32, char> {
        let mut lru = Lru::new(1);
        for (object, letter) in objects {
            lru.insert(*object, *letter, 0);
        }
        lru
    }

    #[test]
    fn the_latest_insert_is_the_tail() {
        let lru = of(&[1, 2, 3]);
        assert_eq!(order(&lru, 0), vec![1, 2, 3]);
    }

    #[test]
    fn inserting_a_live_object_restates_it_as_the_most_recent() {
        let mut lru = lettered(&[(1, 'a'), (2, 'b'), (3, 'c')]);
        lru.insert(1, 'z', 0);
        assert_eq!(order(&lru, 0), vec![2, 3, 1]);
        assert_eq!(lru.get(1), Some(&'z'));

        // The state `insert` returns is the one kept.
        *lru.insert(2, 'b', 0) = 'y';
        assert_eq!(order(&lru, 0), vec![3, 1, 2]);
        assert_eq!(lru.get(2), Some(&'y'));
    }

    #[test]
    fn touching_an_object_moves_it_to_the_tail_of_the_order_its_state_names() {
        let mut lru = Lru::new(2);
        for (object, letter) in [(1, 'a'), (2, 'b'), (3, 'c')] {
            lru.insert(object, letter, 0);
        }
        assert_eq!(lru.touch(1, |_| 0), Some(&mut 'a'));
        assert_eq!(order(&lru, 0), vec![2, 3, 1]);

        // A change of state that names the other order moves the object there.
        let moved = lru.touch(2, |letter| {
            *letter = 'x';
            1
        });
        assert_eq!(moved, Some(&mut 'x'));
        assert_eq!((order(&lru, 0), order(&lru, 1)), (vec![3, 1], vec![2]));
        lru.touch(3, |_| 1);
        assert_eq!((order(&lru, 0), order(&lru, 1)), (vec![1], vec![2, 3]));

        assert_eq!(lru.touch(9, |_| 0), None);
        assert_eq!((order(&lru, 0), order(&lru, 1)), (vec![1], vec![2, 3]));
    }

    #[test]
    fn removing_an_object_keeps_the_rest_ordered() {
        let mut lru = lettered(&[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]);
        assert_eq!(lru.remove(2), Some('b'));
        assert_eq!(order(&lru, 0), vec![1, 3, 4]);
        lru.insert(1, 'a', 0);
        assert_eq!(order(&lru, 0), vec![3, 4, 1]);
        assert_eq!(lru.remove(4), Some('d'));
        assert_eq!(order(&lru, 0), vec![3, 1]);
        assert_eq!(lru.remove(4), None);

        // The objects that remain keep their state wherever they are stored.
        assert_eq!(lru.get(3), Some(&'c'));
        assert_eq!(lru.get(1), Some(&'a'));
        lru.insert(5, 'e', 0);
        assert_eq!(order(&lru, 0), vec![3, 1, 5]);
        assert_eq!(lru.remove(3), Some('c'));
        assert_eq!(order(&lru, 0), vec![1, 5]);
        assert_eq!(lru.get(5), Some(&'e'));
    }

    /// A removal moves the last slot into the freed one; when that slot's object
    /// is of another order, that order's links and ends follow it.
    #[test]
    fn removing_an_object_keeps_the_other_orders_ordered() {
        let mut lru = Lru::new(2);
        for (object, order) in [(1, 0), (2, 1), (3, 0), (4, 1)] {
            lru.insert(object, object, order);
        }
        assert_eq!(lru.remove(1), Some(1));
        assert_eq!((order(&lru, 0), order(&lru, 1)), (vec![3], vec![2, 4]));
        assert_eq!(lru.remove(2), Some(2));
        assert_eq!((order(&lru, 0), order(&lru, 1)), (vec![3], vec![4]));
        lru.insert(5, 5, 1);
        assert_eq!(order(&lru, 1), vec![4, 5]);
        assert_eq!(lru.pop_least(1), Some(4));
        assert_eq!((order(&lru, 0), order(&lru, 1)), (vec![3], vec![5]));
        assert_eq!((lru.get(3), lru.get(5)), (Some(&3), Some(&5)));
    }

    /// The slots span several chunks: removals that move a slot across a chunk
    /// boundary, and that empty the last chunk, keep every order and state.
    #[test]
    fn orders_spanning_chunks_stay_ordered_through_removals() {
        let count = 2 * CHUNK as u32 + 3;
        let mut lru = Lru::new(2);
        for object in 0..count {
            lru.insert(object, object, (object % 2) as usize);
        }
        // The first objects leave, each replaced by the last slot of the array.
        for object in 0..CHUNK as u32 + 5 {
            assert_eq!(lru.remove(object), Some(object));
        }
        let left = |parity: u32| -> Vec<u32> {
            (CHUNK as u32 + 5..count)
                .filter(|object| object % 2 == parity)
                .collect()
        };
        assert_eq!((order(&lru, 0), order(&lru, 1)), (left(0), left(1)));
        assert!((CHUNK as u32 + 5..count).all(|object| lru.get(object) == Some(&object)));
    }

    #[test]
    fn popping_takes_the_least_recently_used_first() {
        let mut lru = of(&[7, 8, 9]);
        assert_eq!(lru.pop_least(0), Some(7));
        assert_eq!(lru.pop_least(0), Some(8));
        assert_eq!(lru.pop_least(0), Some(9));
        assert_eq!(lru.pop_least(0), None);
        assert!(order(&lru, 0).is_empty());
    }
}
