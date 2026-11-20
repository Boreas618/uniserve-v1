//! The configuration sections as records that cross builds: an engine's launch (JSON) and a hot
//! restart's handoff (MessagePack with named fields), which a build of another release writes.

use proto::config::KvCacheConfig;

/// A KV cache section of an earlier release states the retired `boundary_interval`: this build
/// reads it in both encodings as the same section without it, and writes it in neither, so a
/// build of that release reads what this one writes.
#[test]
fn a_retired_option_is_read_and_never_written() {
    let earlier = serde_json::json!({ "host_ratio": 4.0, "boundary_interval": 64 });
    let expected = KvCacheConfig {
        host_ratio: 4.0,
        ..KvCacheConfig::default()
    };

    let launch: KvCacheConfig =
        serde_json::from_value(earlier.clone()).expect("the launch's section reads");
    assert_eq!(launch, expected);
    let handoff: KvCacheConfig =
        service::msgpack::decode(&service::msgpack::encode(&earlier).unwrap())
            .expect("the handoff's section reads");
    assert_eq!(handoff, expected);

    let launch = serde_json::to_value(&expected).unwrap();
    assert!(launch.get("boundary_interval").is_none(), "{launch}");
    let handoff: serde_json::Value =
        service::msgpack::decode(&service::msgpack::encode(&expected).unwrap()).unwrap();
    assert!(handoff.get("boundary_interval").is_none(), "{handoff}");
}
