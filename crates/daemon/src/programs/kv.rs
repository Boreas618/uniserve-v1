//! The deployment's KV program as the leader hands it to every node's Storage Server.
//!
//! The KV program runs on the Storage Servers, not in the daemon (`storage_server::programs`):
//! the leader fixes its module with the rest of the package's programs and has each node's daemon
//! set it on that node's server ([`crate::engine::control::EngineControl::program`]), with the
//! tiering hook on the leader's node alone, and reads back what each server recorded of it
//! ([`crate::engine::control::EngineControl::program_records`]).

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;

/// The package's KV program: what every node's Storage Server runs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KvSetup {
    /// Its name in the package (`[programs.<name>]`), which its saved state is filed under.
    pub(crate) name: String,
    /// The wasm32 module.
    pub(crate) module: Arc<[u8]>,
    /// The package's `args` table for it.
    pub(crate) args: toml::Table,
    /// How often its periodic work runs: the tiering hook on the leader's node, and the save of
    /// its state on every node.
    pub(crate) period: Duration,
}

impl KvSetup {
    /// The program as the process interface carries it to the node of `host`, which steps the
    /// tiering hook when `tiering`.
    ///
    /// # Errors
    ///
    /// Fails when the `args` table has no JSON form (a datetime is carried as its text) or the
    /// period does not fit in milliseconds.
    pub(crate) fn wire(&self, tiering: bool) -> anyhow::Result<proto::process::KvProgram> {
        let args = serde_json::to_string(&self.args).context("writing the KV program's args")?;
        let period_ms = u64::try_from(self.period.as_millis())
            .context("the KV program's period is too long")?;
        Ok(proto::process::KvProgram {
            name: self.name.clone(),
            module: self.module.to_vec(),
            args,
            period_ms,
            tiering,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The program crosses to a node with its module, its args as the JSON of the table, its
    /// period in milliseconds and the node's part in the tiering hook.
    #[test]
    fn a_kv_program_crosses_to_a_node_whole() {
        let setup = KvSetup {
            name: "kv".to_owned(),
            module: Arc::from(&b"\0asm\x01\0\0\0"[..]),
            args: toml::from_str("watermark = 0.5\ntiers = [\"host\", \"store\"]").unwrap(),
            period: Duration::from_secs(10),
        };
        let wire = setup.wire(true).unwrap();
        assert_eq!(wire.name, "kv");
        assert_eq!(wire.module, b"\0asm\x01\0\0\0");
        assert_eq!(wire.period_ms, 10_000);
        assert!(wire.tiering);
        let args: serde_json::Value = serde_json::from_str(&wire.args).unwrap();
        assert_eq!(
            args,
            serde_json::json!({ "watermark": 0.5, "tiers": ["host", "store"] })
        );
        assert!(!setup.wire(false).unwrap().tiering);
    }
}
