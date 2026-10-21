//! Launch descriptor resolution.

use cuda::DeviceId;
use ipc::{CallKind, ForwardMode};
use serde_json::{Value, json};
use worker::config::{Device, WorkerLaunch, resolve};

/// Neutral values for every descriptor field, as the engine emits them for
/// a single-rank CPU worker.
fn defaults() -> Value {
    json!({
        "acknowledgment_slot": 0,
        "host_slots": [0],
        "products_cross_hosts": false,
        "queue_depth": 2,
        "watchdog_timeout_ms": 300000,
        "model": "model",
        "storage": {
            "socket": "/run/uniserve/storage.sock",
            "deployment": "model",
            "claim": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "page_tier": false,
            "standby": false,
        },
        "host": "localhost",
        "device": "cpu",
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "components": {"model": {"ranks": [0]}},
        "deployment_components": ["model"],
        "transfer_backends": "local",
        "export_backends": "local",
        "rendezvous_address": null,
        "rendezvous_listen_fd": null,
        "expert_parallel": null,
        "role": "model",
        "expert_microbatches": 1,
        "distributed_backend": null,
        "mesh": null,
        "lane": [],
        "no_model": false,
        "load_format": "auto",
        "download_dir": null,
        "load_threads": null,
        "model_dtype": "bfloat16",
        "quantization_config": {},
        "prefix_cache_dtype": null,
        "mem_fraction_static": 0.70,
        "max_total_tokens": null,
        "attention_backend": "auto",
        "block_size": 64,
        "max_batch_calls": 256,
        "max_batch_tokens": 8192,
        "max_request_pool_size": 129,
        "max_model_len": 8192,
        "max_video_seconds": 15.0,
        "max_condition_rows": 0,
        "encoder_cache_entries": 256,
        "ffmpeg": "ffmpeg",
        "min_video_seconds": null,
        "graph_policy": "off",
        "decode_graph_batch_sizes": null,
        "prefill_cuda_graph": false,
        "prefill_outputs": true,
        "prefill_graph_token_sizes": null,
        "flow_cuda_graph": true,
        "flow_graph_batch_sizes": null,
        "flow_graph_shapes": null,
        "video_text_capacities": null,
        "video_resolutions": [],
        "video_aspect_ratios": [],
        "canvas_sampling": null,
        "flashinfer_workspace_size": 536870912,
        "flashinfer_use_tensor_core": null,
        "flashinfer_decode_backend": "fa2",
        "flashinfer_prefill_backend": "auto",
        "flashinfer_decode_split_tile_size": null,
        "flashinfer_prefill_split_tile_size": null,
        "flashinfer_disable_split_kv": false
    })
}

/// Resolve a descriptor with `overrides` applied over the defaults.
fn resolved(overrides: Value) -> Result<WorkerLaunch, String> {
    let mut descriptor = defaults();
    let fields = descriptor.as_object_mut().unwrap();
    for (key, value) in overrides.as_object().unwrap() {
        fields.insert(key.clone(), value.clone());
    }
    resolve_text(&descriptor.to_string())
}

/// Resolve the descriptor `text` states, as the rank does with the one the
/// head sends it.
fn resolve_text(text: &str) -> Result<WorkerLaunch, String> {
    let descriptor = serde_json::from_str(text).map_err(|error| error.to_string())?;
    resolve(descriptor, "worker", 1).map_err(|error| error.to_string())
}

fn two_rank_model() -> Value {
    json!({"model": {"ranks": [0, 1], "parallel_config": {"tensor_parallel_size": 2}}})
}

#[test]
fn the_tower_placement_resolves_the_generation_device() {
    let launch = resolved(json!({"device": "cuda:0", "mesh": "tower=gen:cuda:1"})).unwrap();
    assert_eq!(launch.execution.device, Device::Cuda(DeviceId(0)));
    assert_eq!(
        launch.execution.generation_device,
        Some(Device::Cuda(DeviceId(1)))
    );
    assert_eq!(launch.device(), Device::Cuda(DeviceId(0)));

    // The text tower runs on the rank's device, which the declaration does
    // not restate, and the generation tower on another one.
    for (mesh, message) in [
        ("tower=text:cuda:0;gen:cuda:1", "tower=gen:<device>"),
        ("tower=gen:cuda", "other than the rank's"),
        ("tower=gen:bogus", "worker_config.mesh tower gen"),
    ] {
        let error = launch_error(json!({"device": "cuda:0", "mesh": mesh}));
        assert!(error.contains(message), "{mesh}: {error}");
    }
}

#[test]
fn a_malformed_device_names_its_option() {
    for device in ["gpu0", "cuda:x"] {
        let error = launch_error(json!({"device": device}));
        assert!(error.contains("rank device"), "{error}");
    }
}

#[test]
fn engine_capacities_and_precision_reach_the_worker() {
    let launch = resolved(json!({
        "max_batch_calls": 128,
        "max_batch_tokens": 16384,
        "max_request_pool_size": 257,
        "max_model_len": 32768,
        "max_total_tokens": 65536,
        "mem_fraction_static": 0.5,
        "prefix_cache_dtype": "float16",
        "quantization_config": {"mode": "performance", "components": {"text_encoder": "bf16"}},
    }))
    .unwrap();
    assert_eq!(launch.execution.max_batch_calls, 128);
    assert_eq!(launch.execution.max_batch_tokens, 16384);
    assert_eq!(launch.execution.max_request_pool_size, 257);
    assert_eq!(launch.execution.max_model_len, 32768);
    assert_eq!(launch.execution.max_total_tokens, Some(65536));
    assert_eq!(launch.execution.mem_fraction_static, 0.5);
    assert_eq!(
        launch.execution.prefix_cache_dtype.as_deref(),
        Some("float16")
    );
    let quantization: Value =
        serde_json::from_str(&launch.model.as_ref().unwrap().quantization).unwrap();
    assert_eq!(
        quantization,
        json!({"mode": "performance", "components": {"text_encoder": "bf16"}})
    );
}

fn launch_error(overrides: Value) -> String {
    resolved(overrides).unwrap_err()
}

#[test]
fn execution_lanes_bind_disjoint_calls() {
    let launch = resolved(json!({"lane": [
        {"lane_id": "decode", "sm_budget": 64, "call_kinds": ["decode", "verify", "token_denoising"],
         "max_batch_calls": null, "max_batch_tokens": null, "max_inflight": null},
        {"lane_id": "compute", "sm_budget": 88, "call_kinds": ["prefill", "denoising"],
         "max_batch_calls": null, "max_batch_tokens": null, "max_inflight": null},
    ]}))
    .unwrap();
    let lanes = &launch.execution.lanes;
    assert_eq!(lanes[0].lane_id, "decode");
    assert_eq!(lanes[1].sm_budget, 88);
    assert_eq!(
        lanes[0].call_kinds,
        vec![
            CallKind::Forward(ForwardMode::Decode),
            CallKind::Forward(ForwardMode::Verify),
            CallKind::Forward(ForwardMode::TokenDenoising),
        ]
    );

    // A misspelled optional limit fails at parsing instead of leaving the
    // lane without that limit.
    for field in ["max_batch_token", "max_inflght"] {
        let mut lane = json!({"lane_id": "decode", "sm_budget": 64, "call_kinds": ["decode"]});
        lane[field] = json!(2);
        assert!(
            launch_error(json!({ "lane": [lane] })).contains("unknown field"),
            "{field}"
        );
    }
    let duplicate = json!({"lane": [
        {"lane_id": "a", "sm_budget": 64, "call_kinds": ["decode"],
         "max_batch_calls": null, "max_batch_tokens": null, "max_inflight": null},
        {"lane_id": "b", "sm_budget": 64, "call_kinds": ["decode"],
         "max_batch_calls": null, "max_batch_tokens": null, "max_inflight": null},
    ]});
    assert!(launch_error(duplicate).contains("one execution lane"));
}

/// A stated capture catalog or capacity list replaces the rank's own, and an
/// unstated one keeps it. The launching side validates the lists it states.
#[test]
fn stated_catalogs_replace_the_rank_defaults() {
    let launch = resolved(json!({
        "video_text_capacities": [1024, 4096, 8192],
        "min_video_seconds": 4.0,
        "flow_graph_shapes": [[1024, 512], [512, 1024]],
    }))
    .unwrap();
    assert_eq!(
        launch.execution.video_text_capacities,
        vec![1024, 4096, 8192]
    );
    assert_eq!(launch.execution.min_video_seconds, Some(4.0));
    assert_eq!(
        launch.execution.flow_graph_shapes,
        vec![(1024, 512), (512, 1024)]
    );

    // Without settings the worker derives its video capacities itself and
    // captures its own image shapes.
    let unset = resolved(json!({})).unwrap();
    assert!(unset.execution.video_text_capacities.is_empty());
    assert_eq!(unset.execution.min_video_seconds, None);
    assert!(!unset.execution.flow_graph_shapes.is_empty());
}

#[test]
fn video_canvas_names_keep_the_order_the_deployment_names() {
    let launch = resolved(json!({
        "video_resolutions": ["480p", "768p"],
        "video_aspect_ratios": ["9:16", "16:9"],
    }))
    .unwrap();
    assert_eq!(launch.execution.video_resolutions, ["480p", "768p"]);
    assert_eq!(launch.execution.video_aspect_ratios, ["9:16", "16:9"]);

    // Without names the worker prepares the canvases its denoiser offers.
    let execution = resolved(json!({})).unwrap().execution;
    assert!(execution.video_resolutions.is_empty() && execution.video_aspect_ratios.is_empty());
}

#[test]
fn attention_options_reach_the_backend() {
    let launch = resolved(
        json!({"flashinfer_workspace_size": 64 << 20, "flashinfer_use_tensor_core": true}),
    )
    .unwrap();
    assert_eq!(launch.attention.workspace_size, 64 << 20);
    assert_eq!(launch.attention.use_tensor_core, Some(true));
    assert!(launch_error(json!({"flashinfer_use_tensor_core": "maybe"})).contains("bool"));
}

/// The group's first rank serves the rendezvous on the socket it inherits.
#[test]
fn the_first_rank_inherits_its_rendezvous_socket() {
    let launch = resolved(json!({
        "rank": 0, "world_size": 2, "components": two_rank_model(),
        "rendezvous_address": "127.0.0.1:29500", "rendezvous_listen_fd": 7,
    }))
    .unwrap();
    let rendezvous = launch.rendezvous.unwrap();
    assert_eq!(
        (rendezvous.address.as_str(), rendezvous.listen_fd),
        ("127.0.0.1:29500", Some(7))
    );
}

#[test]
fn an_expert_parallel_replica_names_its_union_and_store() {
    let launch = resolved(json!({"expert_parallel": {
        "rank": 0, "size": 4, "attention_ranks": 0, "address": "10.0.0.2:29600",
        "listen_fd": 9, "exchange": "megamoe",
    }}))
    .unwrap();
    let union = launch.expert_parallel.unwrap();
    assert_eq!((union.rank, union.size), (0, 4));
    assert_eq!(union.rendezvous.address, "10.0.0.2:29600");
    assert_eq!(union.rendezvous.listen_fd, Some(9));
    assert_eq!(launch.execution.expert_exchange, "megamoe");
}

#[test]
fn disaggregated_ranks_keep_their_local_rank_and_role() {
    for (role, rank, world_size, union_rank) in [("model", 1, 2, 3), ("experts", 1, 4, 5)] {
        let components = if role == "experts" {
            json!({})
        } else {
            json!({"model": {"ranks": (0..world_size).collect::<Vec<_>>(),
                "parallel_config": {"tensor_parallel_size": world_size}}})
        };
        let launch = resolved(json!({
            "role": role, "expert_microbatches": 2, "rank": rank, "local_rank": rank,
            "world_size": world_size, "components": components,
            "expert_parallel": {"rank": union_rank, "size": 8, "attention_ranks": 4,
                "address": "10.0.0.2:29600", "listen_fd": null, "exchange": "deepep"},
        }))
        .unwrap();
        assert_eq!(
            (launch.execution.rank, launch.execution.world_size),
            (rank, world_size)
        );
        let union = launch.expert_parallel.as_ref().unwrap();
        assert_eq!((union.rank, union.attention_ranks), (union_rank, 4));
        assert_eq!(launch.execution.expert_microbatches, 2);
        assert_eq!(launch.components.is_empty(), role == "experts");
    }
}

#[test]
fn no_model_alone_launches_the_stub_model() {
    let launch = resolved(json!({"model": "", "no_model": true})).unwrap();
    assert!(launch.stub && launch.model.is_none());
    assert!(launch_error(json!({"model": " "})).contains("--model is required"));
}

#[test]
fn the_encoder_cache_holds_the_launched_entry_count() {
    let launch = resolved(json!({"encoder_cache_entries": 1024})).unwrap();
    assert_eq!(launch.execution.encoder_cache_entries, 1024);

    // The stub model takes its size from the descriptor like any other.
    let stub = resolved(json!({
        "model": "",
        "no_model": true,
        "encoder_cache_entries": 32
    }))
    .unwrap();
    assert_eq!(stub.execution.encoder_cache_entries, 32);

    assert!(launch_error(json!({"encoder_cache_entries": 0})).contains("must be positive"));
}

#[test]
fn channel_and_transfer_values_are_checked() {
    assert!(launch_error(json!({"transfer_backends": "local,local"})).contains("unique"));
    assert!(launch_error(json!({"export_backends": "posix_shm"})).contains("bound transports"));
    assert!(
        launch_error(json!({"transfer_backends": "local,cuda_vmm", "export_backends": "local"}))
            .contains("CUDA VMM")
    );
}
