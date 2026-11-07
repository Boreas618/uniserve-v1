//! `uniserve engine`: a node's main process for one epoch.
//!
//! The daemon starts one main process per node and epoch and hands it one launch file
//! (`--launch <path>`): the epoch's build config, the model, the request address, the Storage
//! Server namespacing, the Program Server's socket, the devices, the rank and the durations
//! (`specs/executor.md` §2.1). The process itself - its control thread, its schedulers, one executor
//! per device and its blocking pool - lives in [`crate::engine_main`].
//!
//! The command ends with its daemon: the lifeline the supervising daemon gave this process ends
//! when the daemon, and every daemon a hot restart handed its processes to, ended.

pub(crate) use crate::engine_main::run;
