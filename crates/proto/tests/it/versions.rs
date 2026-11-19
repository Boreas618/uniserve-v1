//! The engine interface's hello: a daemon serves a peer in the newest version both speak, and
//! refuses one whose window shares none with a refusal that names both windows.

use proto::{process, request::wire};
use service::Versions;

/// A daemon serves a peer in the newest version both windows hold, whichever is wider; a hello
/// that states no newest version offers its oldest alone.
#[test]
fn a_daemon_agrees_on_the_newest_version_both_speak() {
    let daemon = Versions::new(14, 15);

    assert_eq!(proto::agree(14, 15, daemon), Ok(15));
    assert_eq!(proto::agree(15, 16, daemon), Ok(15));
    assert_eq!(proto::agree(13, 14, daemon), Ok(14));
    assert_eq!(proto::agree(14, 0, daemon), Ok(14));
}

/// A peer outside the daemon's window is refused, on either link, with the refusal that names
/// the peer's window and the daemon's; a hello that states no version is refused alike.
#[test]
fn a_peer_outside_the_window_is_refused_with_both_windows() {
    let daemon = Versions::new(14, 15);

    let disagreement = proto::agree(16, 17, daemon).unwrap_err();
    let refusal = process::Refusal::of_versions(disagreement);
    assert_eq!(
        refusal.reason,
        Some(process::refusal::Reason::Version(process::VersionRefusal {
            offered: 16,
            spoken: 15,
            offered_newest: 17,
            spoken_oldest: 14,
        }))
    );
    assert_eq!(
        refusal.to_string(),
        "the peer speaks versions 16 to 17 of the engine interface and the daemon 14 to 15, \
         which share none"
    );

    let refusal = wire::Refusal::of_versions(proto::agree(13, 0, daemon).unwrap_err());
    assert_eq!(
        refusal.reason,
        Some(wire::refusal::Reason::Version(wire::VersionRefusal {
            offered: 13,
            spoken: 15,
            offered_newest: 0,
            spoken_oldest: 14,
        }))
    );
    assert_eq!(
        refusal.to_string(),
        "the engine speaks versions 13 of the engine interface and the daemon 14 to 15, which \
         share none"
    );

    assert!(proto::agree(0, 0, daemon).is_err());
}

/// A peer reads the version its welcome states, which its offer must hold; a welcome that
/// states none is a daemon of one version, which accepted the offer's oldest.
#[test]
fn a_peer_speaks_the_version_its_welcome_states() {
    let offered = Versions::new(14, 15);

    assert_eq!(proto::agreed(offered, 15), Ok(15));
    assert_eq!(proto::agreed(offered, 0), Ok(14));
    assert!(proto::agreed(offered, 16).is_err());
}

/// The refusal of a daemon of one version reads in a peer as that one version, so a peer of a
/// window and a peer of one version read each other's refusals.
#[test]
fn a_refusal_of_one_version_names_it_alone() {
    let refusal = process::Refusal {
        reason: Some(process::refusal::Reason::Version(process::VersionRefusal {
            offered: 15,
            spoken: 14,
            offered_newest: 0,
            spoken_oldest: 0,
        })),
    };
    assert_eq!(
        refusal.to_string(),
        "the peer speaks versions 15 of the engine interface and the daemon 14, which share none"
    );
}

/// An engine that agreed on a version offers it alone at every later hello, so a daemon of a
/// wider window serves its later links in that version too.
#[test]
fn a_process_offers_its_agreed_version_alone() {
    let agreed = proto::Agreed(14);
    let successor = Versions::new(14, 15);

    assert_eq!(agreed.offer(), Versions::one(14));
    assert_eq!(
        proto::agree(agreed.offer().oldest(), agreed.offer().newest(), successor),
        Ok(14)
    );
}
