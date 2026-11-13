//! What orders host work behind a device's streams: the events storage owners wait on before
//! they release storage that device work addressed ([`events::EventPool`]).

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "no storage owner defers a release behind the event pool"
    )
)]
pub(crate) mod events;
