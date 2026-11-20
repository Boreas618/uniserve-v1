//! The versions of an interface a peer speaks, and the one two peers agree on at their hello.
//!
//! Every interface between the processes of a deployment is versioned: the engine interface
//! (`proto::ENGINE_INTERFACE`), the Storage Server's protocol (`storage::protocol::VERSIONS`),
//! the buffer pool's runtime interface (`storage::runtime::VERSIONS`) and the program
//! interfaces (`program::versions`). A build speaks a window of each, [`Versions`]: every version
//! from its oldest to its newest. At an interface's hello each peer states its window, and the
//! two agree on the newest version both speak ([`Versions::agree`]); a peer whose window shares
//! no version with the other's is refused with a refusal that names both windows.
//!
//! What a window holds is one schema: the messages of every version in it are read by one set of
//! types, because a version changes its interface's messages only by additions (a field, a
//! variant, a value), as Protobuf's rules for compatible changes state them. A sender uses what a
//! version added only on a connection whose agreed version has it, and a receiver of an older
//! agreed version never sees it. A change that is not an addition keeps the previous version's
//! handling beside its own, selected by the connection's agreed version, until the window's
//! oldest version passes it: the change that raises an interface's oldest version removes every
//! handling of the versions below it.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A window of an interface's versions: every version from [`Versions::oldest`] to
/// [`Versions::newest`], both included, of which a peer speaks each.
///
/// Versions start at 1, so 0 is never a version a window holds: a hello's field that states no
/// newest version is 0, and [`Versions::stated`] reads it as a window of the oldest alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Versions {
    oldest: u32,
    newest: u32,
}

impl Versions {
    /// The window from `oldest` to `newest`.
    ///
    /// # Panics
    ///
    /// When `oldest` is 0 or above `newest`; in a constant, the build refuses it.
    pub const fn new(oldest: u32, newest: u32) -> Self {
        assert!(oldest > 0, "versions start at 1");
        assert!(
            oldest <= newest,
            "a window's oldest version is not above its newest"
        );
        Self { oldest, newest }
    }

    /// The window of the one version `version`.
    ///
    /// # Panics
    ///
    /// When `version` is 0.
    pub const fn one(version: u32) -> Self {
        Self::new(version, version)
    }

    /// The window a hello states as its `oldest` and `newest` versions, where a `newest` of 0
    /// (a field the peer did not set) is the oldest alone; `None` for a window no peer speaks: an
    /// oldest of 0, or a newest below the oldest.
    pub fn stated(oldest: u32, newest: u32) -> Option<Self> {
        let newest = if newest == 0 { oldest } else { newest };
        (oldest > 0 && oldest <= newest).then_some(Self { oldest, newest })
    }

    /// The oldest version of the window.
    pub const fn oldest(self) -> u32 {
        self.oldest
    }

    /// The newest version of the window: the one a build's own changes are written in.
    pub const fn newest(self) -> u32 {
        self.newest
    }

    /// Whether the window holds `version`.
    pub const fn speaks(self, version: u32) -> bool {
        self.oldest <= version && version <= self.newest
    }

    /// The version two peers of the windows `self` and `other` agree on: the newest both speak,
    /// or `None` when the windows share no version.
    pub fn agree(self, other: Self) -> Option<u32> {
        let newest = self.newest.min(other.newest);
        (newest >= self.oldest.max(other.oldest)).then_some(newest)
    }
}

/// `14` for a window of one version, `14 to 15` for a wider one.
impl fmt::Display for Versions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.oldest == self.newest {
            write!(formatter, "{}", self.oldest)
        } else {
            write!(formatter, "{} to {}", self.oldest, self.newest)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Versions;

    /// Two peers agree on the newest version both speak, whichever side holds the wider window;
    /// windows that share no version agree on none.
    #[test]
    fn peers_agree_on_the_newest_version_both_speak() {
        let current = Versions::new(14, 15);

        assert_eq!(current.agree(Versions::new(14, 15)), Some(15));
        assert_eq!(current.agree(Versions::new(15, 16)), Some(15));
        assert_eq!(Versions::new(15, 16).agree(current), Some(15));
        assert_eq!(current.agree(Versions::one(14)), Some(14));
        assert_eq!(current.agree(Versions::new(10, 20)), Some(15));
        assert_eq!(current.agree(Versions::new(16, 17)), None);
        assert_eq!(current.agree(Versions::one(13)), None);
    }

    /// A hello that states no newest version states a window of its oldest alone, the window of
    /// a peer that speaks one version; a window no peer speaks is none.
    #[test]
    fn a_hello_without_a_newest_version_states_its_oldest_alone() {
        assert_eq!(Versions::stated(14, 0), Some(Versions::one(14)));
        assert_eq!(Versions::stated(14, 16), Some(Versions::new(14, 16)));
        assert_eq!(Versions::stated(0, 0), None);
        assert_eq!(Versions::stated(0, 3), None);
        assert_eq!(Versions::stated(5, 4), None);
    }

    #[test]
    fn a_window_names_its_versions() {
        assert_eq!(Versions::one(14).to_string(), "14");
        assert_eq!(Versions::new(14, 15).to_string(), "14 to 15");
        assert!(Versions::new(14, 15).speaks(14));
        assert!(!Versions::new(14, 15).speaks(16));
    }
}
