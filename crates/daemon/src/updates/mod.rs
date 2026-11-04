//! The deployment's updates: an epoch switch and the trial that decides it
//! (`specs/worker.md` §7.4, §7.3).
//!
//! An update that changes what builds a main process re-builds the epoch
//! ([`crate::epoch::layer`] returns [`crate::epoch::Layer::Epoch`]): each
//! node's main process of the target executable starts as a parked standby
//! (precopy), the serving epoch is drained and the target serves
//! (switchover), the trial decides, and a commit or a revert follows.
//! Participants are node main processes, not an engine and its workers, and
//! the wire between them and the node's daemon is `proto::process`
//! ([`crate::proto::process`]): commands down, reports up, the roster of
//! children behind them.
//!
//! [`switch`] holds the switch itself: the phases, what ends each one, the
//! pause it measures, the trial's verdict and the commit, revert and purge
//! it commands.

pub(crate) mod change;
pub(crate) mod switch;
