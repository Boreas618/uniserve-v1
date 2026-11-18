//! The etcd the sidecar registers its workers in and reads the planner's decisions from, through
//! etcd v3's JSON gateway: the same keys, values, leases and transactions Dynamo's own gRPC
//! client writes (`lib/runtime/src/transports/etcd.rs`), each a POST of the request message in
//! JSON with keys and values in base64 and 64-bit integers as decimal strings.

use anyhow::{Context as _, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Value, json};

/// A client of one etcd endpoint's JSON gateway (`http://host:2379`).
#[derive(Clone)]
pub(super) struct Etcd {
    base: String,
    http: reqwest::Client,
}

fn encode(bytes: &[u8]) -> String {
    STANDARD.encode(bytes)
}

fn decode(text: &str) -> Result<Vec<u8>> {
    STANDARD
        .decode(text)
        .with_context(|| format!("`{text}` is not base64"))
}

/// The key after every key that starts with `prefix`: a range's end.
fn range_end(prefix: &str) -> Vec<u8> {
    let mut end = prefix.as_bytes().to_vec();
    while let Some(last) = end.pop() {
        if last < 0xff {
            end.push(last + 1);
            return end;
        }
    }
    vec![0]
}

/// A 64-bit integer the gateway writes as a decimal string, or as a number.
fn integer(value: &Value) -> Option<i64> {
    match value {
        Value::String(text) => text.parse().ok(),
        Value::Number(number) => number.as_i64(),
        _ => None,
    }
}

impl Etcd {
    pub(super) fn new(base: &str) -> Self {
        Self {
            base: base.trim_end_matches('/').to_owned(),
            http: reqwest::Client::new(),
        }
    }

    /// Posts `body` to `path` and reads the answer's first JSON message.
    async fn call(&self, path: &str, body: Value) -> Result<Value> {
        let response = self
            .http
            .post(format!("{}{path}", self.base))
            .json(&body)
            .send()
            .await
            .with_context(|| format!("posting {path} to etcd at {}", self.base))?;
        let status = response.status();
        // A stream's answer (a keep-alive's) is one message per line; the first is the answer.
        let mut response = response;
        let mut text = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            text.extend_from_slice(&chunk);
            if text.contains(&b'\n') {
                break;
            }
        }
        let line = text.split(|&byte| byte == b'\n').next().unwrap_or_default();
        let value: Value = serde_json::from_slice(line)
            .with_context(|| format!("etcd answered {path} with no JSON ({status})"))?;
        if !status.is_success() || value.get("error").is_some() {
            bail!("etcd refused {path} ({status}): {value}");
        }
        Ok(value.get("result").cloned().unwrap_or(value))
    }

    /// A new lease of `ttl` seconds: its id.
    pub(super) async fn grant(&self, ttl: u64) -> Result<i64> {
        let answer = self
            .call("/v3/lease/grant", json!({"TTL": ttl.to_string()}))
            .await?;
        answer
            .get("ID")
            .and_then(integer)
            .context("etcd granted a lease of no id")
    }

    /// Renews lease `lease`: the seconds it lives on, 0 once it expired.
    pub(super) async fn keep_alive(&self, lease: i64) -> Result<i64> {
        let answer = self
            .call("/v3/lease/keepalive", json!({"ID": lease.to_string()}))
            .await?;
        Ok(answer.get("TTL").and_then(integer).unwrap_or(0))
    }

    /// Revokes lease `lease`, deleting every key attached to it.
    pub(super) async fn revoke(&self, lease: i64) -> Result<()> {
        self.call("/v3/lease/revoke", json!({"ID": lease.to_string()}))
            .await
            .map(drop)
    }

    /// Creates `key` with `value` under `lease` unless it exists (Dynamo's create: a transaction
    /// that puts only at version 0): whether it created it.
    pub(super) async fn create(&self, key: &str, value: &Value, lease: i64) -> Result<bool> {
        let key = encode(key.as_bytes());
        let answer = self
            .call(
                "/v3/kv/txn",
                json!({
                    "compare": [{"key": key, "target": "VERSION", "result": "EQUAL", "version": "0"}],
                    "success": [{"request_put": {
                        "key": key,
                        "value": encode(&serde_json::to_vec(value)?),
                        "lease": lease.to_string(),
                    }}],
                    "failure": [{"request_range": {"key": key}}],
                }),
            )
            .await?;
        Ok(answer
            .get("succeeded")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    /// Puts `value` at `key`, under `lease` when one is given.
    pub(super) async fn put(&self, key: &str, value: &[u8], lease: Option<i64>) -> Result<()> {
        let mut body = json!({"key": encode(key.as_bytes()), "value": encode(value)});
        if let Some(lease) = lease {
            body["lease"] = json!(lease.to_string());
        }
        self.call("/v3/kv/put", body).await.map(drop)
    }

    /// Deletes `key`.
    pub(super) async fn delete(&self, key: &str) -> Result<()> {
        self.call("/v3/kv/deleterange", json!({"key": encode(key.as_bytes())}))
            .await
            .map(drop)
    }

    /// Every key that starts with `prefix`, with its value.
    pub(super) async fn range(&self, prefix: &str) -> Result<Vec<(String, Vec<u8>)>> {
        let answer = self
            .call(
                "/v3/kv/range",
                json!({"key": encode(prefix.as_bytes()), "range_end": encode(&range_end(prefix))}),
            )
            .await?;
        let Some(kvs) = answer.get("kvs").and_then(Value::as_array) else {
            return Ok(Vec::new());
        };
        kvs.iter()
            .map(|kv| {
                let key = kv.get("key").and_then(Value::as_str).unwrap_or_default();
                let value = kv.get("value").and_then(Value::as_str).unwrap_or_default();
                let key = String::from_utf8(decode(key)?).context("an etcd key is not UTF-8")?;
                Ok((key, decode(value)?))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::range_end;

    #[test]
    fn a_prefix_range_ends_after_every_key_it_begins() {
        assert_eq!(range_end("v1/dynamo/planner/"), b"v1/dynamo/planner0");
        assert_eq!(range_end("a\u{7f}"), b"a\x80");
    }
}
