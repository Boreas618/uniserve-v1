//! What a program crate starts from: `use scheduling::prelude::*;`.
//!
//! The [`Program`] trait and its [`export!`] macro, the inputs and outputs of the four programs,
//! durations and instants, serde's derives (a program crate depends on `serde` itself, which
//! the derives expand to) and the `log` crate's macros, whose records reach the host's `events`
//! table.

pub use log::{debug, error, info, trace, warn};
pub use serde::{Deserialize, Serialize};

pub use crate::control::{
    Config, Events, Kind, Record, Records, Report, Series, State, Stats, Window,
};
pub use crate::export;
pub use crate::kv::{
    Directory, Found, Handoff, Holding, Key, Kv, Location, Move, Node, Path, Tier, Tiering,
    Transport,
};
pub use crate::program::Program;
pub use crate::routing::{Hit, Replica, Request, Role, Route, Routing, Usage};
pub use crate::time::{Duration, Instant, hours, millis, mins, secs};
