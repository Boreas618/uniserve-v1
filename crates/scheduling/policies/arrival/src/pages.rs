//! The policy's record of the cached pages, in one order per storage a shortage takes from.
//!
//! A device shortage may take only a device page no running request continues from, and a host
//! shortage only a page on the host alone (design §4.4). Each kind of victim has its own
//! least-recently-used order, so a shortage reads the first `need` pages of its order and nothing
//! else; the device pages running requests hold have an order of their own, which no shortage
//! reads. Within each order the pages keep the order the whole cache's last uses give them: every
//! event that moves a page to another order uses it now, so it joins its new order as the most
//! recently used.

use scheduling::{PageId, PageView};

use crate::lru::Lru;

/// The order of the device pages no running request continues from: what a device shortage
/// takes, and what a step offloads.
const FREE: usize = 0;
/// The order of the device pages a running request continues from, which no shortage takes.
const HELD: usize = 1;
/// The order of the pages on the host alone: what a host shortage takes.
const HOST: usize = 2;

/// The order a page belongs to by its view.
fn order(view: &PageView) -> usize {
    if view.on_host {
        HOST
    } else if view.evictable {
        FREE
    } else {
        HELD
    }
}

/// What the policy keeps of a cached page.
#[derive(Clone, Debug)]
pub(crate) struct Page {
    /// The page as the mechanism last stated it: whether it sits in host memory and whether no
    /// running request continues from it.
    pub(crate) view: PageView,
    /// Whether the page waits in the policy's offers to be offered to the host.
    pub(crate) queued: bool,
}

impl Page {
    /// A page as the mechanism states it, waiting for no offer.
    pub(crate) fn stated(view: &PageView) -> Self {
        Self {
            view: view.clone(),
            queued: false,
        }
    }

    /// Whether the page is one a step offloads: a device page no running request continues
    /// from.
    pub(crate) fn offloadable(&self) -> bool {
        order(&self.view) == FREE
    }
}

/// The cached pages, by the storage a shortage may take each from.
#[derive(Clone, Debug)]
pub(crate) struct Pages {
    orders: Lru<PageId, Page>,
}

impl Default for Pages {
    fn default() -> Self {
        Self {
            orders: Lru::new(3),
        }
    }
}

impl Pages {
    /// The pages a snapshot states, from the least recently used to the most recently used.
    pub(crate) fn from_least_recent<'a>(views: impl IntoIterator<Item = &'a PageView>) -> Self {
        let mut pages = Self::default();
        for view in views {
            pages.state(Page::stated(view));
        }
        pages
    }

    /// What the policy keeps of `page`, if the cache holds it.
    pub(crate) fn get(&self, page: PageId) -> Option<&Page> {
        self.orders.get(page)
    }

    /// What the policy keeps of `page`, for the caller to update without moving it: the caller
    /// changes nothing of its view.
    pub(crate) fn get_mut(&mut self, page: PageId) -> Option<&mut Page> {
        self.orders.get_mut(page)
    }

    /// Records `page` as the mechanism states it now (`Cached`), as the most recently used of
    /// the order its view names.
    pub(crate) fn state(&mut self, page: Page) {
        let order = order(&page.view);
        self.orders.insert(page.view.id, page, order);
    }

    /// Records `page` as used just now, with its view changed by `change` (a hit, a release, a
    /// hold): it is the most recently used of the order its view now names. `None`, and no
    /// change, for a page the cache does not hold.
    pub(crate) fn used(
        &mut self,
        page: PageId,
        change: impl FnOnce(&mut PageView),
    ) -> Option<&mut Page> {
        self.orders.touch(page, |kept| {
            change(&mut kept.view);
            order(&kept.view)
        })
    }

    /// Takes `page` out of the record (`Freed`).
    pub(crate) fn remove(&mut self, page: PageId) {
        self.orders.remove(page);
    }

    /// The `need` least recently used device pages no running request continues from, or as
    /// many as there are.
    pub(crate) fn free_victims(&self, need: usize) -> impl Iterator<Item = PageId> + '_ {
        self.orders.iter(FREE).take(need).map(|(page, _)| page)
    }

    /// The `need` least recently used pages on the host alone, or as many as there are.
    pub(crate) fn host_victims(&self, need: usize) -> impl Iterator<Item = PageId> + '_ {
        self.orders.iter(HOST).take(need).map(|(page, _)| page)
    }
}
