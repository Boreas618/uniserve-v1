//! The sidecar against fake Dynamo peers that speak Dynamo's protocols: an etcd that serves
//! etcd v3's JSON gateway, a frontend that discovers the workers there and sends token requests
//! over the TCP request plane to their endpoints, receiving the responses on a response server
//! of its own, a planner that writes its decisions through the virtual connector's keys, and a
//! router that subscribes to the workers' KV event channels. The deployment is a node serving a
//! real leader over engines whose schedulers run as they serve, above fake executors.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::routing::post;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::CancellationToken;

use crate::api::openai::video_jobs::VideoJobs;
use crate::http::Ready;
use crate::http::node::{Brought, First, Start, run};
use crate::policy::Package;
use crate::policy::revision::{KEPT, Revisions};

/// How long a test waits for what the sidecar does.
const WAIT: Duration = Duration::from_secs(30);

/// The fake etcd's state: keys with their values and leases, and the leases.
#[derive(Default)]
struct Store {
    keys: BTreeMap<Vec<u8>, (Vec<u8>, Option<i64>)>,
    leases: BTreeMap<i64, i64>,
    next: i64,
}

type Shared = Arc<Mutex<Store>>;

fn bytes(value: &Value) -> Vec<u8> {
    STANDARD.decode(value.as_str().unwrap_or_default()).unwrap()
}

fn number(value: &Value) -> i64 {
    match value {
        Value::String(text) => text.parse().unwrap(),
        other => other.as_i64().unwrap(),
    }
}

/// An etcd v3 JSON gateway of the calls the sidecar makes, on a port of its own.
async fn fake_etcd() -> (String, Shared) {
    let store = Shared::default();
    let router = axum::Router::new()
        .route(
            "/v3/lease/grant",
            post(
                |State(store): State<Shared>, Json(body): Json<Value>| async move {
                    let mut store = store.lock().unwrap();
                    store.next += 1;
                    let id = 0x694d_0000 + store.next;
                    let ttl = number(&body["TTL"]);
                    store.leases.insert(id, ttl);
                    Json(json!({"ID": id.to_string(), "TTL": ttl.to_string()}))
                },
            ),
        )
        .route(
            "/v3/lease/keepalive",
            post(
                |State(store): State<Shared>, Json(body): Json<Value>| async move {
                    let id = number(&body["ID"]);
                    let ttl = store.lock().unwrap().leases.get(&id).copied().unwrap_or(0);
                    format!(
                        "{}\n",
                        json!({"result": {"ID": id.to_string(), "TTL": ttl.to_string()}})
                    )
                },
            ),
        )
        .route(
            "/v3/lease/revoke",
            post(
                |State(store): State<Shared>, Json(body): Json<Value>| async move {
                    let id = number(&body["ID"]);
                    let mut store = store.lock().unwrap();
                    store.leases.remove(&id);
                    store.keys.retain(|_, (_, lease)| *lease != Some(id));
                    Json(json!({}))
                },
            ),
        )
        .route(
            "/v3/kv/txn",
            post(
                |State(store): State<Shared>, Json(body): Json<Value>| async move {
                    // The sidecar's one transaction: put when the key's version is 0.
                    let put = &body["success"][0]["request_put"];
                    let key = bytes(&put["key"]);
                    let mut store = store.lock().unwrap();
                    let succeeded = !store.keys.contains_key(&key);
                    if succeeded {
                        let lease = put.get("lease").map(number);
                        store.keys.insert(key, (bytes(&put["value"]), lease));
                    }
                    Json(json!({"succeeded": succeeded}))
                },
            ),
        )
        .route(
            "/v3/kv/put",
            post(
                |State(store): State<Shared>, Json(body): Json<Value>| async move {
                    let lease = body.get("lease").map(number);
                    store
                        .lock()
                        .unwrap()
                        .keys
                        .insert(bytes(&body["key"]), (bytes(&body["value"]), lease));
                    Json(json!({}))
                },
            ),
        )
        .route(
            "/v3/kv/deleterange",
            post(
                |State(store): State<Shared>, Json(body): Json<Value>| async move {
                    store.lock().unwrap().keys.remove(&bytes(&body["key"]));
                    Json(json!({}))
                },
            ),
        )
        .route(
            "/v3/kv/range",
            post(
                |State(store): State<Shared>, Json(body): Json<Value>| async move {
                    let (from, to) = (bytes(&body["key"]), bytes(&body["range_end"]));
                    let store = store.lock().unwrap();
                    let kvs: Vec<Value> = store
                    .keys
                    .range(from..to)
                    .map(|(key, (value, _))| {
                        json!({"key": STANDARD.encode(key), "value": STANDARD.encode(value)})
                    })
                    .collect();
                    Json(json!({"kvs": kvs, "count": kvs.len().to_string()}))
                },
            ),
        )
        .with_state(Arc::clone(&store));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await });
    (format!("http://{address}"), store)
}

/// The values of the keys that start with `prefix`, as JSON, by key.
fn values(store: &Shared, prefix: &str) -> BTreeMap<String, Value> {
    store
        .lock()
        .unwrap()
        .keys
        .iter()
        .filter(|(key, _)| key.starts_with(prefix.as_bytes()))
        .map(|(key, (value, _))| {
            let value = serde_json::from_slice(value)
                .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(value).into_owned()));
            (String::from_utf8_lossy(key).into_owned(), value)
        })
        .collect()
}

/// Waits until `check` holds of the fake etcd's keys.
async fn until(store: &Shared, what: &str, check: impl Fn(&Shared) -> bool) {
    let deadline = tokio::time::Instant::now() + WAIT;
    while !check(store) {
        assert!(tokio::time::Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The instances registered under `component`.
fn instances(store: &Shared, component: &str) -> Vec<Value> {
    values(store, &format!("v1/instances/dynamo/{component}/generate/"))
        .into_values()
        .collect()
}

/// Reads one two-part message: its header and its body.
async fn read_message(stream: &mut TcpStream) -> (Vec<u8>, Vec<u8>) {
    let mut lengths = [0u8; 24];
    stream.read_exact(&mut lengths).await.unwrap();
    let header = u64::from_be_bytes(lengths[..8].try_into().unwrap()) as usize;
    let body = u64::from_be_bytes(lengths[8..16].try_into().unwrap()) as usize;
    let mut bytes = vec![0u8; header + body];
    stream.read_exact(&mut bytes).await.unwrap();
    let body = bytes.split_off(header);
    (bytes, body)
}

fn two_part(header: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_be_bytes());
    out.extend_from_slice(&(body.len() as u64).to_be_bytes());
    out.extend_from_slice(&0u64.to_be_bytes());
    out.extend_from_slice(header);
    out.extend_from_slice(body);
    out
}

/// A frontend's request to the worker of `instance`: the request frame of `request` to the
/// endpoint the instance states, acknowledged, and the response stream it opens back to the
/// frontend's response server, read to its sentinel. Returns each output in order.
async fn frontend_request(instance: &Value, request: Value) -> Vec<Value> {
    let transport = instance["transport"]["tcp"].as_str().unwrap();
    let (address, path) = transport.split_once('/').unwrap();
    let home = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let context = uuid::Uuid::new_v4().to_string();
    let subject = uuid::Uuid::new_v4().to_string();
    let info = json!({
        "address": home.local_addr().unwrap().to_string(),
        "subject": subject,
        "context": context,
        "stream_type": "response",
    });
    let control = json!({
        "id": context,
        "request_type": "single_in",
        "response_type": "many_out",
        "payload_codec": "json",
        "connection_info": {"transport": "tcp_server", "info": info.to_string()},
    });
    let payload = two_part(
        &serde_json::to_vec(&control).unwrap(),
        &serde_json::to_vec(&request).unwrap(),
    );
    let headers =
        serde_json::to_vec(&json!({"x-endpoint-path": path, "request-id": context})).unwrap();
    let mut frame = Vec::new();
    frame.extend_from_slice(&(path.len() as u16).to_be_bytes());
    frame.extend_from_slice(path.as_bytes());
    frame.extend_from_slice(&(headers.len() as u16).to_be_bytes());
    frame.extend_from_slice(&headers);
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);

    let mut worker = TcpStream::connect(address).await.unwrap();
    worker.write_all(&frame).await.unwrap();
    let mut ack = [0u8; 4];
    worker.read_exact(&mut ack).await.unwrap();
    assert_eq!(u32::from_be_bytes(ack), 0, "the worker accepts the request");

    let (mut stream, _) = tokio::time::timeout(WAIT, home.accept())
        .await
        .unwrap()
        .unwrap();
    let (handshake, _) = read_message(&mut stream).await;
    assert_eq!(
        serde_json::from_slice::<Value>(&handshake).unwrap(),
        json!({"subject": subject, "stream_type": "response"})
    );
    let (prologue, _) = read_message(&mut stream).await;
    assert_eq!(
        serde_json::from_slice::<Value>(&prologue).unwrap()["error"],
        Value::Null
    );
    let mut outputs = Vec::new();
    loop {
        let (header, body) = tokio::time::timeout(WAIT, read_message(&mut stream))
            .await
            .unwrap();
        if !header.is_empty() {
            assert_eq!(
                serde_json::from_slice::<Value>(&header).unwrap(),
                json!("sentinel")
            );
            return outputs;
        }
        let data: Value = serde_json::from_slice(&body).unwrap();
        if data["complete_final"] == json!(true) {
            continue;
        }
        assert!(data["data"].get("error").is_none(), "{data}");
        outputs.push(data["data"]["data"].clone());
    }
}

/// The tokens of a response's outputs, and its last output.
fn tokens(outputs: &[Value]) -> (Vec<u64>, Value) {
    let tokens = outputs
        .iter()
        .flat_map(|output| output["token_ids"].as_array().cloned().unwrap_or_default())
        .map(|token| token.as_u64().unwrap())
        .collect();
    (tokens, outputs.last().cloned().unwrap_or_default())
}

/// A node serving a leader over `replicas` test engines of the roles `roles` states, with
/// `dynamo` as its Dynamo sidecar's settings and `events` as its KV events' endpoint: its
/// revisions, its stop signal and the directory of its state.
async fn node(
    replicas: usize,
    roles: &[(u32, scheduling::routing::Role)],
    dynamo: crate::config::DynamoConfig,
    events: Option<String>,
) -> (Arc<Revisions>, CancellationToken, tempfile::TempDir) {
    let clients = (0..replicas)
        .map(|ordinal| {
            let mut launch = testkit::args(crate::test_support::SERVED_MODEL);
            launch.replica = u32::try_from(ordinal).unwrap();
            launch.engine_config.kv_events.endpoint = events.clone();
            Arc::new(crate::test_support::fake_client(
                launch,
                testkit::Fixture::new(crate::test_support::letter_fake),
            ))
        })
        .collect();
    let leader = Arc::new(crate::test_support::leader_over(
        crate::test_support::test_model(crate::test_support::CHAT),
        crate::test_support::configured_tokenizer(),
        crate::test_support::SERVED_MODEL.to_owned(),
        clients,
    ));
    let mut config = service::lock(&leader.replicas()[0].config).clone();
    config.dynamo_config = dynamo;
    config.kv_events_config.endpoint = events;
    for (number, role) in roles {
        config.replicas.insert(
            number.to_string(),
            crate::config::ReplicaConfig { role: *role },
        );
    }
    for replica in leader.replicas() {
        *service::lock(&replica.config) = config.clone();
    }
    let package = Package::of(config.clone(), BTreeMap::new()).unwrap();

    let state = tempfile::tempdir().unwrap();
    let revisions = Arc::new(Revisions::open(state.path().join("sim")).unwrap());
    let changes = proto::reload::changes(None, &package.config);
    let first = revisions.begin(&package, "start", None, &changes).unwrap();
    revisions.converged(first, 0.0, KEPT).unwrap();

    let runtime = service::RuntimeDirectory::within(state.path()).unwrap();
    let instance = service::Instance::create(&runtime, "daemon").unwrap();
    let listener = crate::http::listener::Listener::bind(&crate::config::AdminConfig::default())
        .await
        .unwrap();
    let admin = crate::http::admin::Listener::bind(instance.path()).unwrap();
    let shutdown = CancellationToken::new();
    let jobs = Arc::new(VideoJobs::new(
        config.video_job_config.clone(),
        shutdown.child_token(),
    ));
    let recorder = leader.recorder().clone();
    let brought = Brought {
        config: config.clone(),
        stated: package.stated.clone(),
        leader,
        jobs,
        adopted: Vec::new(),
        daemon: None,
        token: shutdown.clone(),
        then: None,
    };
    let start = Start {
        families: processing::Registry::new([]).unwrap(),
        options: config,
        deployment: "sim".to_owned(),
        revisions: Arc::clone(&revisions),
        recorder,
        ready: Ready::new(None, instance.path()),
        instance,
        shutdown: shutdown.clone(),
    };
    tokio::spawn(run(
        start,
        listener,
        admin,
        First::Serving(Box::new(brought)),
    ));
    (revisions, shutdown, state)
}

fn settings(etcd: &str) -> crate::config::DynamoConfig {
    crate::config::DynamoConfig {
        etcd: Some(etcd.to_owned()),
        host: Some("127.0.0.1".to_owned()),
        period: Duration::from_millis(50),
        ..crate::config::DynamoConfig::default()
    }
}

/// A port no socket of the host listens on now.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A replica is an aggregated worker: its instance, card and event channels are in discovery
/// under its lease, a frontend's token request to it streams the tokens it generates and ends
/// with Dynamo's finish reason and usage, a router subscribed to its KV event channel reads the
/// pages its scheduler committed as Dynamo's KV events, and its keys leave discovery when the
/// deployment comes down.
#[tokio::test(flavor = "multi_thread")]
async fn a_replica_is_a_worker_a_frontend_sends_token_requests_to() {
    let (etcd, store) = fake_etcd().await;
    let events = format!("tcp://127.0.0.1:{}", free_port());
    let (_revisions, shutdown, _state) = node(1, &[], settings(&etcd), Some(events)).await;
    until(&store, "the replica registers as a worker", |store| {
        instances(store, "backend").len() == 1
    })
    .await;

    // Its instance: the request plane's address and its endpoint's path, JSON requests.
    let instance = instances(&store, "backend").remove(0);
    let id = instance["instance_id"].as_u64().unwrap();
    assert_eq!(instance["type"], "Endpoint");
    assert_eq!(instance["namespace"], "dynamo");
    assert_eq!(instance["request_plane_codec"], "json");
    let transport = instance["transport"]["tcp"].as_str().unwrap();
    assert!(
        transport.starts_with("127.0.0.1:")
            && transport.ends_with(&format!("/{id:x}/dynamo/backend/generate")),
        "{transport}"
    );
    // Its card: a token model of the served name and its page size, an aggregated worker.
    let cards = values(&store, &format!("v1/mdc/dynamo/backend/generate/{id:x}"));
    let card = &cards.values().next().unwrap()["card_json"];
    assert_eq!(card["display_name"], crate::test_support::SERVED_MODEL);
    assert_eq!(card["model_input"], "Tokens");
    assert_eq!(card["worker_type"], "aggregated");
    assert_eq!(
        card["kv_cache_block_size"],
        proto::config::DEFAULT_BLOCK_SIZE
    );
    // Its event channels: its KV events and its load.
    let channels = values(
        &store,
        "v1/event_channels/namespace/dynamo/component/backend/",
    );
    let topics: Vec<&str> = channels
        .values()
        .map(|channel| channel["topic"].as_str().unwrap())
        .collect();
    assert!(
        topics.contains(&"kv-events") && topics.contains(&"kv_metrics"),
        "{topics:?}"
    );
    let sources = values(
        &store,
        "v1/event_sources/namespace/dynamo/component/backend/",
    );
    let source = sources.values().next().unwrap();
    assert_eq!(
        source["metadata"]["worker"],
        json!({"worker_id": id, "dp_rank": 0})
    );

    // A router subscribes to the KV events before the request.
    let kv = channels
        .values()
        .find(|channel| channel["topic"] == "kv-events")
        .unwrap();
    use zeromq::Socket as _;
    let mut router = zeromq::SubSocket::new();
    {
        router
            .connect(kv["transport"]["config"]["endpoint"].as_str().unwrap())
            .await
            .unwrap();
        router.subscribe("kv-events").await.unwrap();
    }
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Two full pages of prompt, and three tokens to generate.
    let block = proto::config::DEFAULT_BLOCK_SIZE;
    let prompt: Vec<u32> = (0..2 * block).map(|token| 100 + token).collect();
    let outputs = frontend_request(
        &instance,
        json!({
            "model": crate::test_support::SERVED_MODEL,
            "token_ids": prompt,
            "stop_conditions": {"max_tokens": 3},
            "sampling_options": {"temperature": 0.0},
            "eos_token_ids": [],
        }),
    )
    .await;
    let (generated, last) = tokens(&outputs);
    assert_eq!(generated, [u64::from(crate::test_support::LETTER); 3]);
    assert_eq!(last["finish_reason"], "length");
    assert_eq!(
        last["completion_usage"],
        json!({"prompt_tokens": 2 * block, "completion_tokens": 3, "total_tokens": 2 * block + 3})
    );

    // The router reads the stored pages: four frames, the envelope after its frame header,
    // Dynamo's KV events of the worker with the hash of each page's tokens.
    let stored = tokio::time::timeout(WAIT, async {
        use zeromq::SocketRecv as _;
        loop {
            let frames = router.recv().await.unwrap().into_vec();
            assert_eq!(frames.len(), 4);
            assert_eq!(&frames[0][..], b"kv-events");
            let frame = &frames[3];
            assert_eq!(frame[0], 1);
            let length = u32::from_be_bytes(frame[1..5].try_into().unwrap()) as usize;
            #[derive(serde::Deserialize)]
            struct Envelope {
                topic: String,
                #[serde(with = "serde_bytes")]
                payload: Vec<u8>,
            }
            let envelope: Envelope = rmp_serde::from_slice(&frame[5..5 + length]).unwrap();
            assert_eq!(envelope.topic, "kv-events");
            let events: Vec<Value> = rmp_serde::from_slice(&envelope.payload).unwrap();
            if let Some(stored) = events
                .iter()
                .find(|event| event["event"]["data"].get("stored").is_some())
            {
                break stored.clone();
            }
        }
    })
    .await
    .expect("the router reads the stored pages");
    assert_eq!(stored["worker_id"], id);
    assert_eq!(stored["storage_tier"], "device");
    let blocks = stored["event"]["data"]["stored"]["blocks"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        blocks[0]["tokens_hash"],
        super::wire::tokens_hash(&prompt[..block as usize])
    );

    // The deployment comes down: its workers leave discovery.
    shutdown.cancel();
    until(&store, "the workers leave discovery", |store| {
        values(store, "v1/").is_empty()
    })
    .await;
}

/// The planner's decision of one prefill and one decode worker, which it states through its
/// virtual connector, becomes the roles of the deployment's two mixed replicas: the next
/// revision, written by `dynamo`, states replica 0 prefill and replica 1 decode. The decision is
/// acknowledged only once that change converged, which the test's engines, started without a
/// launch of their own, do not let a change do.
#[tokio::test(flavor = "multi_thread")]
async fn the_planners_decision_is_written_as_the_replicas_roles() {
    use scheduling::routing::Role;

    let (etcd, store) = fake_etcd().await;
    let (revisions, shutdown, _state) = node(2, &[], settings(&etcd), None).await;
    until(&store, "both replicas register", |store| {
        instances(store, "backend").len() == 2
    })
    .await;

    for (key, value) in [
        ("num_prefill_workers", "1"),
        ("num_decode_workers", "1"),
        ("decision_id", "0"),
    ] {
        store.lock().unwrap().keys.insert(
            format!("v1/dynamo/planner/{key}").into_bytes(),
            (value.as_bytes().to_vec(), None),
        );
    }
    let deadline = tokio::time::Instant::now() + WAIT;
    let written = loop {
        if let Ok(record) = revisions.record(2)
            && record.result != crate::policy::revision::Outcome::Converging
        {
            break record;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the decision writes a revision"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(written.writer, "dynamo");
    let roles: HashMap<String, Role> = revisions
        .load(2)
        .unwrap()
        .package
        .config
        .replicas
        .iter()
        .map(|(id, replica)| (id.clone(), replica.role))
        .collect();
    assert_eq!(
        roles,
        HashMap::from([
            ("0".to_owned(), Role::Prefill),
            ("1".to_owned(), Role::Decode)
        ])
    );
    // The change did not converge here: the planner is not told it did.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(values(&store, "v1/dynamo/planner/scaled_decision_id").is_empty());

    shutdown.cancel();
}

/// Replicas of the prefill and the decode role are a prefill worker, whose card states its
/// bootstrap, and a decode worker. A frontend's disaggregated request, its prefill half to the
/// prefill worker and its decode half to the decode worker in one bootstrap room, streams its
/// tokens on the decode half, and the prefill half answers with the room and ends.
#[tokio::test(flavor = "multi_thread")]
async fn a_frontend_pairs_a_prefill_and_a_decode_replica_in_one_room() {
    use scheduling::routing::Role;

    let (etcd, store) = fake_etcd().await;
    let roles = [(0, Role::Prefill), (1, Role::Decode)];
    let (_revisions, shutdown, _state) = node(2, &roles, settings(&etcd), None).await;
    until(&store, "the workers register under their roles", |store| {
        instances(store, "prefill").len() == 1 && instances(store, "backend").len() == 1
    })
    .await;
    let prefill = instances(&store, "prefill").remove(0);
    let decode = instances(&store, "backend").remove(0);
    let prefill_id = prefill["instance_id"].as_u64().unwrap();
    let card = values(
        &store,
        &format!("v1/mdc/dynamo/prefill/generate/{prefill_id:x}"),
    )
    .into_values()
    .next()
    .unwrap();
    assert_eq!(card["card_json"]["worker_type"], "prefill");
    assert_eq!(card["card_json"]["model_type"], "Prefill");
    assert_eq!(card["card_json"]["needs"], json!([["decode"]]));
    let bootstrap = card["card_json"]["runtime_config"]["disaggregated_endpoint"].clone();
    assert_eq!(bootstrap["bootstrap_host"], "127.0.0.1");
    let decode_id = decode["instance_id"].as_u64().unwrap();
    let decode_card = values(
        &store,
        &format!("v1/mdc/dynamo/backend/generate/{decode_id:x}"),
    )
    .into_values()
    .next()
    .unwrap();
    assert_eq!(decode_card["card_json"]["worker_type"], "decode");
    assert_eq!(decode_card["card_json"]["needs"], json!([["prefill"]]));

    // The frontend's disaggregated request: both halves in one room, sent together.
    let room = 4_611_686_018_427_387_904u64;
    let info = json!({
        "bootstrap_host": bootstrap["bootstrap_host"],
        "bootstrap_port": bootstrap["bootstrap_port"],
        "bootstrap_room": room,
    });
    let half = |max_tokens: u32| {
        json!({
            "token_ids": [5, 6, 7],
            "stop_conditions": {"max_tokens": max_tokens},
            "sampling_options": {"temperature": 0.0},
            "routing": {"prefill_worker_id": prefill_id, "prefill_dp_rank": 0},
            "bootstrap_info": info,
        })
    };
    let (prefilled, decoded) = tokio::join!(
        frontend_request(&prefill, half(1)),
        frontend_request(&decode, half(4)),
    );
    assert_eq!(
        prefilled[0],
        json!({"token_ids": [], "disaggregated_params": info})
    );
    assert_eq!(tokens(&prefilled).1["finish_reason"], "length");
    let (generated, last) = tokens(&decoded);
    assert_eq!(generated, [u64::from(crate::test_support::LETTER); 4]);
    assert_eq!(last["finish_reason"], "length");

    shutdown.cancel();
}
