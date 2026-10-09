//! Launch descriptor resolution, through the rank's command line.

#![recursion_limit = "256"]
#![allow(clippy::unwrap_used)]

use std::path::Path;

use serde_json::{Value, json};
use uniserve_ipc::{CallKind, ForwardMode, MediaCall};
use uniserve_worker::config::{Device, WorkerLaunch};
use uniserve_worker::launch::configure;

/// Neutral values for every descriptor field, as the engine emits them for
/// a single-rank CPU worker.
fn defaults() -> Value {
    json!({
        "worker_id": "worker",
        "registration_address": "127.0.0.1:0",
        "channel_transport": "iceoryx2",
        "acknowledgment_slot": 0,
        "host_slots": [0],
        "products_cross_hosts": false,
        "queue_depth": 2,
        "ipc_payload_cap": 65536,
        "model": "model",
        "device": "cpu",
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "components": {"model": {"ranks": [0]}},
        "deployment_components": ["model"],
        "supported_calls": null,
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
        "allow_stub": false,
        "load_format": "auto",
        "download_dir": null,
        "load_threads": null,
        "checksum_manifest": null,
        "model_dtype": "bfloat16",
        "quantization_config": {},
        "kv_cache_dtype": null,
        "kv_memory_fraction": 0.70,
        "kv_token_capacity": null,
        "attention_backend": "auto",
        "block_size": 64,
        "max_batch_calls": 256,
        "max_batch_tokens": 8192,
        "max_request_pool_size": 129,
        "max_model_len": 8192,
        "max_video_seconds": 15.0,
        "max_condition_rows": 0,
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
        "video_canvases": [],
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

fn resolve_text(text: &str) -> Result<WorkerLaunch, String> {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("launch.json");
    std::fs::write(&path, text).unwrap();
    configure(&["--launch-descriptor", path_text(&path)]).map_err(|error| error.to_string())
}

fn path_text(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn two_rank_model() -> Value {
    json!({"model": {"ranks": [0, 1], "parallel_config": {"tensor_parallel_size": 2}}})
}

#[test]
fn the_tower_placement_resolves_the_generation_device() {
    for mesh in ["tower=gen:cuda:1", "tower=text:cuda;gen:cuda:1"] {
        let launch = resolved(json!({"device": "cuda:0", "mesh": mesh})).unwrap();
        assert_eq!(launch.execution.device, "cuda:0");
        assert_eq!(
            launch.execution.generation_device.as_deref(),
            Some("cuda:1")
        );
        assert_eq!(launch.device(), Device::Cuda(0));
    }
    for mesh in ["tower=text:cuda:2;gen:cuda:1", "tower=gen:cuda"] {
        assert!(
            resolved(json!({"device": "cuda:0", "mesh": mesh})).is_err(),
            "{mesh}"
        );
    }
}

#[test]
fn a_malformed_device_names_its_option() {
    for (device, mesh, option) in [
        ("gpu0", Value::Null, "--device"),
        ("cuda:x", Value::Null, "--device"),
        (
            "cuda:0",
            json!("tower=text:cpu:x;gen:cuda:1"),
            "--mesh tower text",
        ),
        ("cuda:0", json!("tower=gen:bogus"), "--mesh tower gen"),
    ] {
        let error = resolved(json!({"device": device, "mesh": mesh})).unwrap_err();
        assert!(error.contains(option), "{error}");
    }
}

#[test]
fn the_storage_fraction_lies_in_the_unit_interval() {
    assert_eq!(
        resolved(json!({"kv_memory_fraction": 1.0}))
            .unwrap()
            .execution
            .kv_storage_fraction,
        1.0
    );
    for fraction in [0.0, -0.5, 1.5] {
        assert!(resolved(json!({"kv_memory_fraction": fraction})).is_err());
    }
}

#[test]
fn engine_capacities_and_precision_reach_the_worker() {
    let launch = resolved(json!({
        "max_batch_calls": 128,
        "max_batch_tokens": 16384,
        "max_request_pool_size": 257,
        "quantization_config": {"mode": "performance", "components": {"text_encoder": "bf16"}},
    }))
    .unwrap();
    assert_eq!(launch.execution.max_batch_calls, 128);
    assert_eq!(launch.execution.max_batch_tokens, 16384);
    assert_eq!(launch.execution.max_request_pool_size, 257);
    let quantization: Value =
        serde_json::from_str(&launch.model.as_ref().unwrap().quantization).unwrap();
    assert_eq!(
        quantization,
        json!({"mode": "performance", "components": {"text_encoder": "bf16"}})
    );
}

#[test]
fn a_kv_dtype_selector_overrides_the_launch_dtype() {
    let launch = resolved(json!({
        "kv_cache_dtype": "bfloat16",
        "quantization_config": {"kv_cache_dtype": "float16"},
    }))
    .unwrap();
    assert_eq!(launch.execution.kv_cache_dtype.as_deref(), Some("float16"));
    assert!(
        launch_error(json!({"quantization_config": {"kv_cache_dtype": 7}})).contains("KV dtype")
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

    // A lane partitions compute only: it has no pool capacity of its own.
    assert!(
        launch_error(json!({"lane": [{"lane_id": "decode", "sm_budget": 64,
            "call_kinds": ["decode"], "kv_capacity_tokens": 4096}]}))
        .contains("unknown field")
    );
    let duplicate = json!({"lane": [
        {"lane_id": "a", "sm_budget": 64, "call_kinds": ["decode"],
         "max_batch_calls": null, "max_batch_tokens": null, "max_inflight": null},
        {"lane_id": "b", "sm_budget": 64, "call_kinds": ["decode"],
         "max_batch_calls": null, "max_batch_tokens": null, "max_inflight": null},
    ]});
    assert!(launch_error(duplicate).contains("one execution lane"));
}

#[test]
fn capability_selectors_choose_concrete_call_kinds() {
    let selected = |selector: &str| {
        resolved(json!({"supported_calls": selector}))
            .unwrap()
            .supported_calls
            .into_iter()
            .collect::<Vec<_>>()
    };
    assert_eq!(
        selected("ar_decode"),
        vec![CallKind::Forward(ForwardMode::Decode)]
    );
    assert_eq!(
        selected("diffusion_finalize"),
        vec![
            CallKind::Media(MediaCall::ImageDecoding),
            CallKind::Media(MediaCall::Muxing)
        ]
    );
    assert_eq!(
        selected("media_append"),
        vec![
            CallKind::Media(MediaCall::VideoEncoding),
            CallKind::Media(MediaCall::AudioEncoding)
        ]
    );
    // Without a selector the worker serves every call kind it implements.
    let every = resolved(json!({})).unwrap().supported_calls;
    assert_eq!(every.len(), CallKind::ALL.len());
    assert!(launch_error(json!({"supported_calls": "ar_decode,ar_decode"})).contains("duplicate"));
    assert!(launch_error(json!({"supported_calls": "everything"})).contains("unknown call"));
}

#[test]
fn bucket_lists_are_positive_and_strictly_increasing() {
    let launch =
        resolved(json!({"video_text_capacities": "1024,4096,8192", "min_video_seconds": 4.0}))
            .unwrap();
    assert_eq!(
        launch.execution.video_text_capacities,
        vec![1024, 4096, 8192]
    );
    assert_eq!(launch.execution.min_video_seconds, Some(4.0));

    // Without settings the worker derives its video capacities itself.
    let unset = resolved(json!({})).unwrap();
    assert!(unset.execution.video_text_capacities.is_empty());
    assert_eq!(unset.execution.min_video_seconds, None);

    for capacities in ["0,1024", "4096,1024", "1024,1024"] {
        assert!(resolved(json!({"video_text_capacities": capacities})).is_err());
    }
    assert_eq!(
        resolved(json!({"flow_graph_shapes": "1024x512, 512X1024"}))
            .unwrap()
            .execution
            .flow_graph_shapes,
        vec![(1024, 512), (512, 1024)]
    );
    assert!(resolved(json!({"flow_graph_shapes": "1024x512,1024x512"})).is_err());
}

#[test]
fn video_canvases_keep_the_order_the_deployment_names() {
    let canvas = |width, height| uniserve_core::Canvas { width, height };
    let launch = resolved(json!({"video_canvases": [
        {"width": 768, "height": 1344},
        {"width": 1344, "height": 768},
    ]}))
    .unwrap();
    assert_eq!(
        launch.execution.video_canvases,
        vec![canvas(768, 1344), canvas(1344, 768)]
    );

    // Without a selection the worker prepares the canvases its denoiser
    // offers.
    assert!(
        resolved(json!({}))
            .unwrap()
            .execution
            .video_canvases
            .is_empty()
    );

    for canvases in [
        json!([{"width": 1344, "height": 768}, {"width": 1344, "height": 768}]),
        json!([{"width": 0, "height": 768}]),
    ] {
        assert!(resolved(json!({"video_canvases": canvases})).is_err());
    }
}

#[test]
fn attention_options_and_cuda_representations_are_checked() {
    let launch = resolved(
        json!({"flashinfer_workspace_size": 64 << 20, "flashinfer_use_tensor_core": "True"}),
    )
    .unwrap();
    assert_eq!(launch.attention.workspace_size, 64 << 20);
    assert_eq!(launch.attention.use_tensor_core, Some(true));
    assert!(
        launch_error(json!({"flashinfer_workspace_size": 0})).contains("FlashInfer workspace size")
    );
    assert!(resolved(json!({"flashinfer_workspace_size": -5})).is_err());

    for (overrides, option) in [
        (json!({"model_dtype": "float32"}), "--dtype float32"),
        (
            json!({"kv_cache_dtype": "float8_e4m3fn"}),
            "--kv-cache-dtype float8_e4m3fn",
        ),
        (
            json!({"quantization_config": {"kv_cache_dtype": "float8_e4m3fn"}}),
            "--quantization-config kv_cache_dtype float8_e4m3fn",
        ),
    ] {
        let mut cuda = overrides.clone();
        cuda.as_object_mut()
            .unwrap()
            .insert("device".into(), json!("cuda:0"));
        assert!(
            launch_error(cuda).contains(&format!("{option} has no native CUDA attention kernel"))
        );
        // A CPU rank keeps the portable representations.
        let requested = overrides
            .get("model_dtype")
            .and_then(Value::as_str)
            .unwrap_or("bfloat16")
            .to_owned();
        let cpu = resolved(overrides).unwrap();
        assert_eq!(cpu.execution.device, "cpu");
        assert_eq!(cpu.execution.model_dtype, requested);
    }
}

#[test]
fn the_rendezvous_socket_belongs_to_exactly_the_first_rank() {
    for (rank, address, fd) in [
        (0, json!("127.0.0.1:29500"), Value::Null),
        (1, json!("127.0.0.1:29500"), json!(7)),
        (0, Value::Null, json!(7)),
    ] {
        let error = launch_error(json!({
            "rank": rank, "local_rank": rank, "world_size": 2, "components": two_rank_model(),
            "rendezvous_address": address, "rendezvous_listen_fd": fd,
        }));
        assert!(error.contains("rendezvous socket"), "{error}");
    }
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

    for (union, message) in [
        (
            json!({"rank": 1, "size": 4, "address": "h:1", "listen_fd": 9, "exchange": "alltoall"}),
            "rank 0",
        ),
        (
            json!({"rank": 0, "size": 4, "address": "h:1", "listen_fd": null, "exchange": "alltoall"}),
            "rank 0",
        ),
        (
            json!({"rank": 0, "size": 1, "address": "h:1", "listen_fd": 9, "exchange": "alltoall"}),
            "two",
        ),
        (
            json!({"rank": 4, "size": 4, "address": "h:1", "listen_fd": null, "exchange": "alltoall"}),
            "rank",
        ),
        (
            json!({"rank": 0, "size": 4, "address": "h:1", "listen_fd": 9, "exchange": "ring"}),
            "exchange",
        ),
    ] {
        let mut union = union;
        union
            .as_object_mut()
            .unwrap()
            .insert("attention_ranks".into(), json!(0));
        let error = launch_error(json!({"expert_parallel": union}));
        assert!(error.contains(message), "{error}");
    }
    let error = launch_error(json!({
        "world_size": 2, "components": two_rank_model(),
        "rendezvous_address": "127.0.0.1:29500", "rendezvous_listen_fd": 7,
        "expert_parallel": {"rank": 0, "size": 2, "attention_ranks": 0, "address": "h:1",
            "listen_fd": 9, "exchange": "alltoall"},
    }));
    assert!(error.contains("only rank"), "{error}");
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
        assert_eq!(launch.supported_calls.is_empty(), role == "experts");
    }
    for count in [json!(0), json!(5), json!(true), json!(2)] {
        assert!(resolved(json!({"expert_microbatches": count})).is_err());
    }
    for (role, union_rank, attention, exchange) in [
        ("model", 3, 4, "deepep"),
        ("model", 4, 4, "deepep"),
        ("experts", 2, 4, "deepep"),
        ("experts", 4, 4, "alltoall"),
        ("experts", 4, 8, "deepep"),
    ] {
        let components = if role == "experts" {
            json!({})
        } else {
            two_rank_model()
        };
        assert!(
            resolved(json!({
                "role": role, "rank": 0, "world_size": 2, "components": components,
                "expert_parallel": {"rank": union_rank, "size": 8, "attention_ranks": attention,
                    "address": "10.0.0.2:29600", "listen_fd": null, "exchange": exchange},
            }))
            .is_err(),
            "{role} {union_rank} {attention} {exchange}"
        );
    }
}

#[test]
fn the_stub_model_must_be_acknowledged() {
    let launch = resolved(json!({"model": "", "no_model": true, "allow_stub": true})).unwrap();
    assert!(launch.stub && launch.model.is_none());
    assert!(launch_error(json!({"model": "", "no_model": true})).contains("--allow-stub"));
    assert!(launch_error(json!({"model": " "})).contains("--model is required"));
}

#[test]
fn the_descriptor_carries_exactly_the_launch_fields() {
    let mut extra = defaults();
    extra
        .as_object_mut()
        .unwrap()
        .insert("tensor_parallel".into(), json!(2));
    assert!(
        resolve_text(&extra.to_string())
            .unwrap_err()
            .contains("unknown field")
    );

    let mut missing = defaults();
    missing.as_object_mut().unwrap().remove("queue_depth");
    assert!(
        resolve_text(&missing.to_string())
            .unwrap_err()
            .contains("queue_depth")
    );

    assert!(launch_error(json!({"queue_depth": 0})).contains("--queue-depth must be positive"));
    assert!(launch_error(json!({"transfer_backends": "local,local"})).contains("unique"));
    assert!(launch_error(json!({"export_backends": "shm"})).contains("bound transports"));
    assert!(
        launch_error(json!({"transfer_backends": "local,cuda_vmm", "export_backends": "local"}))
            .contains("CUDA VMM")
    );
}

#[test]
fn argument_errors_are_usage_errors() {
    assert!(configure(&["--rank", "0"]).is_err());
    assert!(
        configure(&["--launch-descriptor", "/nonexistent/launch.json"])
            .unwrap_err()
            .to_string()
            .contains("cannot read launch descriptor")
    );
}
