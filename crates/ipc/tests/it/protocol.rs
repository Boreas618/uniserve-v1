//! Protocol round trips and validation behavior for every message family.
//!
//! Tests carry messages through a [`Frame`] as a channel does, and drive the
//! message validators.

use std::io::Write as _;
use std::net::TcpListener;
use std::time::{Duration, Instant};

use ipc::*;
use ipc::{CacheGroup, CacheGroupKind, RequestId};
use serde::Serialize;

/// Returns `message` as a receiver decodes it from the frame a sender made.
fn round_trip<M: Serialize + serde::de::DeserializeOwned>(message: &M) -> M {
    Frame::encode(message).unwrap().decode().unwrap()
}

fn request_key() -> RequestKey {
    RequestKey::new(4, RequestId(7), 2)
}

fn output_product(call: CallId) -> TensorRef {
    TensorRef {
        request_key: request_key(),
        producer_call_id: call,
        output_index: 0,
        generation: 3,
        dtype: DType::I64,
        shape_bound: ShapeBound::default(),
    }
}

fn ar_decode_call() -> Call {
    let kind = CallKind::Forward(ForwardMode::Decode);
    Call {
        consumer_slots: Vec::new(),
        token_input: None,
        coordinates: CallCoordinates {
            logical_position: 7,
            cache_visible_len: 5,
            cache_computed_len: 6,
        },

        token_output: Some(output_product(CallId::new(11, 0))),
        vision_inputs: Vec::new(),
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        input_image: None,
        draft: Default::default(),
        cache_input: None,
        cache_output: None,
        input_token_ids: Vec::new(),
        canvas: None,
        sampling_state: None,
        request_key: request_key(),
        call_id: CallId::new(11, 0),
        component: "model".into(),
        kind,
        bounds: Bounds {
            max_tokens: 1,
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        image_index: (kind == CallKind::Media(MediaCall::LatentPreparation)).then_some(4),
        latent_restore: None,
        host_output: None,
        latent_draw: (kind == CallKind::Media(MediaCall::LatentEncoding)).then_some(2),
        carried_image: None,
    }
}

fn call_for(kind: CallKind, call_id: CallId) -> Call {
    let denoises = kind == CallKind::Forward(ForwardMode::TokenDenoising);
    Call {
        consumer_slots: Vec::new(),
        coordinates: CallCoordinates::default(),
        token_input: None,

        token_output: None,
        vision_inputs: Vec::new(),
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,

        input_image: None,
        draft: Default::default(),
        cache_input: (kind == CallKind::Transfer(TransferMode::CacheInstall)).then_some(BufferId {
            owner: request_key(),
            producer_call_id: CallId::new(1, 0),
            output_index: 0,
            generation: 1,
        }),
        cache_output: matches!(
            kind,
            CallKind::Transfer(TransferMode::CacheExport | TransferMode::CacheInstall)
        )
        .then_some(BufferId {
            owner: request_key(),
            producer_call_id: call_id,
            output_index: 0,
            generation: 2,
        }),
        // A token-denoising call steps its request's resident canvas.
        input_token_ids: Vec::new(),
        canvas: denoises.then_some(CanvasStep::default()),
        sampling_state: None,
        request_key: request_key(),
        call_id,
        component: "model".into(),
        kind,
        bounds: Bounds {
            max_tokens: if denoises { 4 } else { 0 },
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        image_index: None,
        latent_restore: None,
        host_output: None,
        latent_draw: None,
        carried_image: None,
    }
}

fn media_sampling() -> DiffusionSamplingParams {
    DiffusionSamplingParams {
        num_frames: 22,
        video_units: 3,
        seed: 17,
        width: 1024,
        height: 768,
        condition_rows: 0,
    }
}

/// A latent preparation names the image whose noise it draws only when it
/// initializes an image trajectory's latent product; a video denoiser's
/// latent preparation, whose rank holds the slot's pages, names none.
#[test]
fn only_an_image_latent_preparation_names_its_image_index() {
    let call_id = CallId::new(12, 0);
    let mut video = call_for(CallKind::Media(MediaCall::LatentPreparation), call_id);
    video
        .validate()
        .expect("a video latent preparation names no image");
    video.image_index = Some(1);
    assert!(
        video.validate().is_err(),
        "a video preparation draws no image"
    );

    let mut image = call_for(CallKind::Media(MediaCall::LatentPreparation), call_id);
    image.latent_output = Some(output_product(call_id));
    assert!(
        image.validate().is_err(),
        "an image preparation names its image"
    );
    image.image_index = Some(0);
    assert!(image.validate().is_err(), "image indices count from one");
    image.image_index = Some(1);
    image
        .validate()
        .expect("an image latent preparation with its image index");
}

/// A `[Static(2), Device { max: 4 }]` bound requires rank two, an exact first
/// extent, and a second extent within the maximum. Rebinding the same tensor to
/// flat capacity accepts any rank whose element count fits.
#[test]
fn tensor_export_preserves_static_axes_within_dynamic_capacity() {
    let mut reference = output_product(CallId::new(11, 0));
    reference.dtype = DType::F32;
    reference.shape_bound.dims = vec![DimBound::Static(2), DimBound::Device { max: 4 }];
    for (shape, valid) in [
        (vec![2, 3], true),
        (vec![2, 4], true),
        (vec![1, 4], false),
        (vec![2, 5], false),
        (vec![8], false),
    ] {
        let export = TensorExport {
            product: reference.clone(),
            value: TransferHandle::DeviceProduct {
                height: 0,
                width: 0,
                value_range: String::new(),
                tensor: TensorTransfer {
                    shape: shape.clone(),
                    locations: vec![Locator {
                        source: MainProcessInfo::default().endpoint,
                        transport: TransferTransport::PosixShm {
                            endpoint: "tensor-publisher".into(),
                            name: "bounded-tensor".into(),
                        },
                        nbytes: shape.iter().product::<u64>() * 4,
                        dtype: "float32".into(),
                        offset: vec![0; shape.len()],
                        shape,
                        device: "cpu".into(),
                    }],
                },
            },
        };
        assert_eq!(export.validate().is_ok(), valid);
        let mut flat = export;
        flat.product.shape_bound.dims = vec![DimBound::Device { max: 8 }];
        assert_eq!(
            flat.validate().is_ok(),
            flat.value.tensors()[0].shape.iter().product::<u64>() <= 8
        );
    }
}

/// Shards of a 4x4 tensor: the left half plus the top-right quarter leave the
/// bottom-right quarter uncovered until it is added. A full replica then
/// substitutes for the left-half shard, and removing that replica uncovers
/// the left half again.
#[test]
fn tensor_coverage_preserves_replicas_and_detects_missing_regions() {
    let shard = |offset: Vec<u64>, shape: Vec<u64>| Locator {
        source: MainProcessInfo::default().endpoint,
        transport: TransferTransport::PosixShm {
            endpoint: "tensor-publisher".into(),
            name: "tensor-shard".into(),
        },
        nbytes: shape.iter().product::<u64>() * 4,
        dtype: "float32".into(),
        device: "cpu".into(),
        offset,
        shape,
    };
    let mut tensor = TensorTransfer {
        shape: vec![4, 4],
        locations: vec![shard(vec![0, 0], vec![4, 2]), shard(vec![0, 2], vec![2, 2])],
    };
    assert!(!tensor.has_complete_coverage());
    tensor.locations.push(shard(vec![2, 2], vec![2, 2]));
    assert!(tensor.has_complete_coverage());
    tensor.locations.push(shard(vec![0, 0], vec![4, 4]));
    tensor.locations.remove(0);
    assert!(tensor.has_complete_coverage());
    tensor.locations.pop();
    assert!(!tensor.has_complete_coverage());
}

/// A video admission without its conditioning is refused.
#[test]
fn a_media_admission_without_conditioning_is_refused() {
    let admission = NewRequest::new_media(
        request_key(),
        1,
        vec![5, 6, 7],
        media_sampling(),
        VideoAdmission {
            conditioning: Vec::new(),
        },
    );
    assert!(admission.is_err());
}

#[test]
fn validation_rejects_an_output_owned_by_another_call() {
    let mut call = ar_decode_call();
    call.token_output.as_mut().unwrap().producer_call_id = CallId::new(999, 0);
    assert!(call.validate().is_err());
}

#[test]
fn validation_allows_shared_encoder_features_and_rejects_foreign_request_state() {
    let foreign_key = RequestKey::new(4, RequestId(8), 2);
    let mut feature = output_product(CallId::new(3, 0));
    feature.request_key = foreign_key;
    let mut call = ar_decode_call();
    call.vision_inputs = vec![VisionInput { offset: 0, feature }];
    call.validate().unwrap();

    // Only encoder features may belong to another request; the same tensor
    // as a generic input is rejected.
    let shared = call.vision_inputs.pop().unwrap().feature;
    call.inputs.push(shared);
    assert!(call.validate().is_err());
}

#[test]
fn product_validation_enforces_generation_and_shape_bounds() {
    let mut product = output_product(CallId::new(11, 0));
    product.generation = 0;
    assert!(product.validate().is_err());
    product.generation = 1;
    product.shape_bound.dims = vec![DimBound::Device { max: 2 }, DimBound::Device { max: 4 }];
    assert!(product.validate().is_err());
}

#[test]
fn main_process_info_round_trips() {
    let info = MainProcessInfo {
        prefix_cache: Some(PrefixCacheInfo {
            groups: vec![CacheGroup {
                layer_ids: (11..20).collect(),
                ..MainProcessInfo::default().prefix_cache.unwrap().groups[0].clone()
            }],
            ..MainProcessInfo::default().prefix_cache.unwrap()
        }),
        components: vec![ComponentInfo {
            name: "denoiser".into(),
            outputs: vec![
                OutputInfo {
                    name: "conditioning".into(),
                    dtype: DType::BF16,
                    shape_bound: ShapeBound {
                        dims: vec![
                            DimBound::Static(1),
                            DimBound::Device { max: 16384 },
                            DimBound::Static(2560),
                        ],
                    },
                    raster_axes: None,
                },
                // A product laid out at the request's raster names it.
                OutputInfo {
                    name: "video_units".into(),
                    dtype: DType::U8,
                    shape_bound: ShapeBound {
                        dims: vec![
                            DimBound::Device { max: 8 },
                            DimBound::Static(22),
                            DimBound::Static(1344),
                            DimBound::Static(1536),
                            DimBound::Static(3),
                        ],
                    },
                    raster_axes: Some(RasterAxes {
                        height: 2,
                        width: 3,
                    }),
                },
            ],
        }],
        ..Default::default()
    };
    let response = MainProcessResponse::info(info.clone());
    let decoded = round_trip(&response);
    let MainProcessResponse::Info { info: decoded, .. } = decoded else {
        panic!("decoded response must preserve its info variant");
    };
    assert_eq!(decoded, info);
}

/// A component's distribution is omitted from its encoding when absent, so
/// both forms must decode, each with its own units per rank.
#[test]
fn a_component_round_trips_with_and_without_a_distribution() {
    use proto::config::{BlockConfig, BlockDistribution};

    let grouped = BlockConfig::parallel(vec![0]);
    let distributed = BlockConfig {
        distribution: Some(BlockDistribution::TemporalUnits),
        units_per_rank: 3,
        ..grouped.clone()
    };
    for config in [grouped, distributed] {
        assert_eq!(round_trip(&config), config);
    }
}

/// A component's layout reaches its ranks in the launch descriptor with
/// every sequence-parallel strategy, a Ulysses group's head projection
/// included.
#[test]
fn a_component_round_trips_with_every_sequence_parallel_strategy() {
    use proto::config::{BlockConfig, SequenceParallel};

    let strategies = [
        SequenceParallel::Local,
        SequenceParallel::Ulysses {
            ulysses_degree: 2,
            projection: Default::default(),
            projection_rows: None,
        },
        SequenceParallel::Allgather {
            allgather_degree: 2,
        },
        SequenceParallel::Hybrid {
            ulysses_degree: 2,
            allgather_degree: 2,
        },
    ];
    for sequence_parallel in strategies {
        let shape = BlockConfig {
            sequence_parallel,
            ..BlockConfig::parallel(Vec::new())
        };
        let count = shape.world_size().unwrap();
        let config = BlockConfig {
            ranks: (0..count).rev().collect(),
            ..shape
        };
        assert_eq!(round_trip(&config), config);
    }
}

#[test]
fn cache_free_main_process_info_round_trips() {
    let info = MainProcessInfo {
        media_components: video_components(),
        supported_calls: video_components()
            .into_keys()
            .map(CallKind::Media)
            .collect(),
        prefix_cache: None,
        latent_page_units: 64,
        latent_pages: 3,
        ..MainProcessInfo::default()
    };
    let response = MainProcessResponse::info(info.clone());
    let decoded = round_trip(&response);
    let MainProcessResponse::Info { info: decoded, .. } = decoded else {
        panic!("decoded response must preserve its info variant");
    };
    assert_eq!(decoded, info);
}

#[test]
fn main_process_info_rejects_duplicate_set_members() {
    let info = MainProcessInfo {
        supported_calls: vec![
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::Decode),
            CallKind::Forward(ForwardMode::Prefill),
        ],
        ..Default::default()
    };
    assert!(info.validate().is_err());
}

/// A cache pool needs an allocatable unit; every block size must be a power of
/// two dividing the largest one; a layer belongs to one group; heads stay
/// within their logical bounds.
#[test]
fn main_process_info_rejects_inconsistent_cache_groups() {
    let valid = full_caps();
    assert!(valid.clone().validate().is_ok());
    let cases: [fn(&mut PrefixCacheInfo); 5] = [
        |cache| cache.num_units = 1,
        |cache| cache.groups[1].block_size = 48,
        |cache| cache.groups[1].layer_ids.push(0),
        |cache| cache.groups[0].layer_ids.clear(),
        |cache| cache.groups[0].kv_head_offset = 1,
    ];
    for invalidate in cases {
        let mut info = valid.clone();
        invalidate(info.prefix_cache.as_mut().unwrap());
        assert!(info.validate().is_err());
    }
}

/// Stages combine into the narrowest pool with the union of their layers,
/// and must agree on every group's page shape.
#[test]
fn cache_stages_merge_into_the_narrowest_pool() {
    let mut first = full_caps().prefix_cache.unwrap();
    first.identities = vec!["layer-0:function-1".into(), "layer-1:function-1".into()];
    let mut second = first.clone();
    second.num_units = 1024;
    second.groups[0].layer_ids = vec![28, 29];
    second.identities[0] = "layer-28:function-1|layer-29:function-1".into();
    let merged = first.merge(&second).unwrap();
    assert_eq!(merged.num_units, 1024);
    assert_eq!(merged.groups[0].layer_ids.len(), 16);
    assert_eq!(merged.groups[1].layer_ids, first.groups[1].layer_ids);
    assert_eq!(merged.seeds(), second.merge(&first).unwrap().seeds());
    assert_ne!(merged.seeds()[0], first.seeds()[0]);
    assert_eq!(merged.seeds()[1], first.seeds()[1]);

    second.groups[1].units_per_page = 3;
    assert!(first.merge(&second).is_err());
}

fn key_for_request(request: u64) -> RequestKey {
    RequestKey::new(4, RequestId(request), 2)
}

/// A decode batch of two calls, as the scheduler states one: its columns, its shape and its
/// generation.
fn token_batch() -> execution::Batch {
    use execution::{BatchColumns, BatchRow, BatchRows, RowKind, Shape, Variant};
    let calls = [(1, 0..1), (2, 1..2)].map(|(request, rows)| BatchRow {
        request,
        kind: RowKind::Decode,
        cache_group: 0,
        rows,
    });
    execution::Batch {
        id: execution::BatchId(7),
        phase: PhaseId(1),
        calls: vec![execution::CallId(0), execution::CallId(1)],
        shape: Shape::decode(2, Variant::DecodeSampled),
        columns: BatchColumns {
            rows: BatchRows {
                calls: calls.to_vec(),
                positions: vec![11, 4],
                slot_mapping: vec![43, -1],
                seq_lens: vec![12, 5],
                query_offsets: vec![0, 1, 2],
                logits_rows: vec![0, 1],
            },
            input_ids: vec![17, 23],
            slots: vec![1, 2],
            parameters: vec![0.5; 16],
            draws: vec![philox::DrawAddress::default(); 2],
            commit: vec![0; 16],
            page_size: 16,
            max_kv_len: 4096,
        },
        generation: 3,
        cache_units: Vec::new(),
        block_tables: Vec::new(),
        page_loads: Vec::new(),
        page_offloads: Vec::new(),
        latent_restores: Vec::new(),
        host_outputs: Vec::new(),
        media: None,
    }
}

/// The output of [`token_batch`]: one sampled token for each call.
fn batch_output() -> execution::BatchOutput {
    let state = execution::RowState {
        step_index: 12,
        cache_computed_len: 12,
        cache_visible_len: 12,
    };
    execution::BatchOutput {
        calls: vec![
            execution::CallOutput::completed(execution::CallId(0), vec![99], state),
            execution::CallOutput {
                token_logprobs: vec![-0.25],
                ..execution::CallOutput::completed(execution::CallId(1), vec![5], state)
            },
        ],
        timing: TimingCounters {
            queued_us: 3,
            device_us: 40,
            copy_us: 0,
            host_us: 2,
        },
        forward_stats: None,
    }
}

/// One fixture per `RequestKind`, plus call-id coverage on the submit frame.
fn request_fixtures() -> Vec<MainProcessRequest> {
    let mut submit = MainProcessRequest::submit(token_batch());
    submit.set_message_id(Some(91));
    vec![
        MainProcessRequest::info(),
        submit,
        MainProcessRequest::close(),
    ]
}

fn response_fixtures() -> Vec<MainProcessResponse> {
    vec![
        MainProcessResponse::Info {
            message_id: None,
            info: MainProcessInfo::default(),
        },
        MainProcessResponse::Info {
            message_id: Some(17),
            info: full_caps(),
        },
        MainProcessResponse::Result {
            message_id: Some(17),
            result: batch_output(),
        },
        MainProcessResponse::Ok {
            message_id: Some(17),
        },
        MainProcessResponse::Error {
            message_id: Some(17),
            error: MainProcessResponseError {
                message: "device fault on decode".into(),
                code: Some(ErrorCode::ComputeError),
                fatal: false,
                phase: Some("execute".into()),
                route: Some("und.decode".into()),
                calls: vec![
                    ErrorCallIdentity {
                        request_key: key_for_request(100),
                        call_id: CallId::new(11, 0),
                    },
                    ErrorCallIdentity {
                        request_key: key_for_request(101),
                        call_id: CallId::new(12, 0),
                    },
                ],
            },
        },
        MainProcessResponse::Stalled {
            stall: Stall {
                batch_id: Some(12),
                route: Some("decode".into()),
                method: Some("uniserve.runner.forward".into()),
                calls: vec![ErrorCallIdentity {
                    request_key: key_for_request(101),
                    call_id: CallId::new(12, 0),
                }],
                running_ms: 150_000,
            },
        },
    ]
}

fn full_caps() -> MainProcessInfo {
    MainProcessInfo {
        encoder_cache_entries: 64,
        encoder_entry_bytes: 128 << 20,
        media_components: video_components(),
        supported_calls: CallKind::ALL.to_vec(),
        // A full-attention group with 64-token pages of one unit and a
        // sliding-window group with 32-token pages of two units share one
        // pool.
        prefix_cache: Some(PrefixCacheInfo {
            groups: vec![
                CacheGroup {
                    layer_ids: (0..14).collect(),
                    ..MainProcessInfo::default().prefix_cache.unwrap().groups[0].clone()
                },
                CacheGroup {
                    kind: CacheGroupKind::SlidingWindow { window: 4096 },
                    block_size: 32,
                    units_per_page: 2,
                    layer_ids: (14..28).collect(),
                    ..MainProcessInfo::default().prefix_cache.unwrap().groups[0].clone()
                },
            ],
            ..MainProcessInfo::default().prefix_cache.unwrap()
        }),
        max_batch_calls: 64,
        max_batch_tokens: 4096,
        request_slots: 96,
        max_unresolved_calls: 3,
        latent_page_units: 64,
        latent_pages: 17,
        ..MainProcessInfo::default()
    }
}

fn video_components() -> std::collections::BTreeMap<MediaCall, String> {
    [
        (MediaCall::TextEncoding, "text_encoder"),
        (MediaCall::LatentPreparation, "denoiser"),
        (MediaCall::Denoising, "denoiser"),
        (MediaCall::VideoDecoding, "video_decoder"),
        (MediaCall::AudioDecoding, "audio_decoder"),
        (MediaCall::VideoEncoding, "output"),
        (MediaCall::AudioEncoding, "output"),
        (MediaCall::Muxing, "output"),
    ]
    .into_iter()
    .map(|(call, component)| (call, component.to_owned()))
    .collect()
}

#[test]
fn every_request_kind_round_trips_through_ipc() {
    let fixtures = request_fixtures();
    for kind in [RequestKind::Info, RequestKind::Submit, RequestKind::Close] {
        assert!(
            fixtures.iter().any(|request| request.kind() == kind),
            "no fixture covers request kind {kind:?}"
        );
    }
    for request in fixtures {
        assert_eq!(
            round_trip(&request),
            request,
            "round trip for {:?}",
            request.kind()
        );
    }
}

#[test]
fn every_response_kind_round_trips_through_ipc() {
    let fixtures = response_fixtures();
    for kind in [
        ResponseKind::Info,
        ResponseKind::Result,
        ResponseKind::Ok,
        ResponseKind::Error,
        ResponseKind::Stalled,
    ] {
        assert!(
            fixtures.iter().any(|response| response.kind() == kind),
            "no fixture covers response kind {kind:?}"
        );
    }
    for response in fixtures {
        assert_eq!(
            round_trip(&response),
            response,
            "round trip for {:?}",
            response.kind()
        );
    }
}

#[test]
fn a_frame_states_the_length_of_the_message_it_carries() {
    let frame = Frame::encode(&MainProcessRequest::info()).unwrap();
    let (field, message) = frame.as_bytes().split_at(Frame::LENGTH_BYTES);

    assert_eq!(Frame::message_len(field.try_into().unwrap()), message.len());
}

#[test]
fn a_frame_decodes_only_as_the_message_it_carries() {
    let request = Frame::encode(&MainProcessRequest::info()).unwrap();
    assert!(matches!(
        request.decode::<MainProcessResponse>(),
        Err(IpcError::Encoding(_))
    ));

    // A message cut short, as a frame of another build may be, is refused:
    // the frame's length field counts the shortened message, so a channel
    // receives it whole and its decoding fails.
    let mut cut = Frame::encode(&MainProcessResponse::ok())
        .unwrap()
        .into_bytes();
    cut.pop();
    let length = u32::try_from(cut.len() - Frame::LENGTH_BYTES).unwrap();
    cut[..Frame::LENGTH_BYTES].copy_from_slice(&length.to_le_bytes());

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut channel = Channel::connect(&listener.local_addr().unwrap().to_string()).unwrap();
    let (mut peer, _) = listener.accept().unwrap();
    peer.write_all(&cut).unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    let received = loop {
        match channel.try_recv::<MainProcessResponse>() {
            Ok(None) => {
                assert!(Instant::now() < deadline, "the cut frame did not arrive");
                channel.wait(Duration::from_millis(50)).unwrap();
            }
            received => break received,
        }
    };
    assert!(
        matches!(received, Err(IpcError::Encoding(_))),
        "{received:?}"
    );
}

/// A decode range that names the ranks holding its units names each once,
/// in unit order, and crosses the channel as stated.
#[test]
fn a_decode_range_names_each_holding_rank_once() {
    let range = DecodeRange {
        request_key: request_key(),
        call_id: CallId::new(11, 0),
        cursor: 4,
        max_units: 2,
        ranks: vec![5, 1],
    };
    assert!(range.validate().is_ok());
    assert_eq!(round_trip(&range), range);
    let repeated = DecodeRange {
        ranks: vec![5, 5],
        ..range
    };
    assert!(repeated.validate().is_err());
}

/// The message a refused batch or output states, so a test names the rule that refused it.
fn refusal(result: ValidationResult<()>) -> String {
    result.expect_err("the record is refused").to_string()
}

/// A block table of slot `slot` in group 0 over `units`.
fn table(slot: u32, units: &[u32]) -> BlockTable {
    BlockTable {
        request_pool_idx: slot,
        group_id: 0,
        start_page: 0,
        unit_ids: units.iter().copied().map(UnitId).collect(),
        allocated_tokens: 16 * units.len() as u32,
    }
}

/// The units newly acquired for slot `slot` in group 0.
fn allocation(slot: u32, units: &[u32]) -> CacheUnitAllocation {
    CacheUnitAllocation {
        request_pool_idx: slot,
        group_id: 0,
        unit_ids: units.iter().copied().map(UnitId).collect(),
        copies: Vec::new(),
    }
}

/// A host page copy of group 0 over `units`.
fn page_copy(units: &[u32]) -> PageCopy {
    PageCopy {
        rank: 0,
        pool: 1,
        group_id: 0,
        slot: 3,
        units: units.iter().copied().map(UnitId).collect(),
    }
}

#[test]
fn a_scheduled_token_batch_and_its_output_are_valid() {
    let batch = execution::Batch {
        block_tables: vec![table(1, &[5, 6]), table(2, &[7])],
        cache_units: vec![allocation(1, &[6])],
        page_loads: vec![page_copy(&[6])],
        page_offloads: vec![page_copy(&[5])],
        ..token_batch()
    };
    batch.validate().expect("the batch is valid");
    batch_output()
        .validate(&batch)
        .expect("its output is valid");
}

#[test]
fn a_batch_names_each_call_once_with_one_call_row_each() {
    let mut repeated = token_batch();
    repeated.calls[1] = repeated.calls[0];
    assert!(refusal(repeated.validate()).contains("repeats a call"));

    let mut missing = token_batch();
    missing.calls.pop();
    assert!(refusal(missing.validate()).contains("number of call rows"));
}

#[test]
fn a_batch_carries_one_call_per_request() {
    let mut batch = token_batch();
    batch.columns.rows.calls[1].request = batch.columns.rows.calls[0].request;
    assert!(refusal(batch.validate()).contains("two calls for one request"));
}

#[test]
fn a_token_batch_columns_fit_its_shape() {
    let mut batch = token_batch();
    batch.columns.input_ids.pop();
    assert!(refusal(batch.validate()).contains("input_ids"));
}

#[test]
fn a_batch_states_each_block_table_once() {
    let batch = execution::Batch {
        block_tables: vec![table(1, &[5]), table(1, &[6])],
        ..token_batch()
    };
    assert!(refusal(batch.validate()).contains("repeats a block table"));
}

#[test]
fn a_cache_unit_allocation_lies_inside_its_block_table() {
    let unmatched = execution::Batch {
        block_tables: vec![table(1, &[5])],
        cache_units: vec![allocation(2, &[5])],
        ..token_batch()
    };
    assert!(refusal(unmatched.validate()).contains("no matching block table"));

    let outside = execution::Batch {
        block_tables: vec![table(1, &[5])],
        cache_units: vec![allocation(1, &[9])],
        ..token_batch()
    };
    assert!(refusal(outside.validate()).contains("outside its block table"));

    let repeated = execution::Batch {
        block_tables: vec![table(1, &[5, 6])],
        cache_units: vec![allocation(1, &[5]), allocation(1, &[6])],
        ..token_batch()
    };
    assert!(refusal(repeated.validate()).contains("repeats a cache-unit allocation"));
}

#[test]
fn a_page_read_fills_only_new_units_of_the_batch() {
    let batch = execution::Batch {
        block_tables: vec![table(1, &[5, 6])],
        cache_units: vec![allocation(1, &[6])],
        page_loads: vec![page_copy(&[5])],
        ..token_batch()
    };
    assert!(refusal(batch.validate()).contains("not new units"));

    let empty = execution::Batch {
        page_offloads: vec![page_copy(&[])],
        ..token_batch()
    };
    assert!(refusal(empty.validate()).contains("page copy is empty"));
}

#[test]
fn a_batch_names_one_host_encoder_output_per_request() {
    let output = HostOutput {
        pool: 1,
        import: true,
        row_bytes: 64,
        rows: [0, 4],
        slots: vec![Some(2)],
    };
    let batch = execution::Batch {
        host_outputs: vec![
            (key_for_request(1), output.clone()),
            (key_for_request(1), output),
        ],
        ..token_batch()
    };
    assert!(refusal(batch.validate()).contains("two host encoder outputs"));
}

#[test]
fn a_completion_agrees_with_its_status() {
    let state = execution::RowState {
        step_index: 3,
        cache_computed_len: 3,
        cache_visible_len: 3,
    };
    let completed = || execution::CallOutput::completed(execution::CallId(0), vec![1], state);

    let longer = execution::CallOutput {
        cache_visible_len: 4,
        ..completed()
    };
    assert!(refusal(longer.validate()).contains("visible cache length"));

    let uncoded = execution::CallOutput {
        status: CallStatus::Error,
        tokens: Vec::new(),
        ..completed()
    };
    assert!(refusal(uncoded.validate()).contains("must carry an error code"));

    let coded = execution::CallOutput {
        error_code: Some(ErrorCode::ComputeError),
        ..completed()
    };
    assert!(refusal(coded.validate()).contains("must not carry an error code"));

    let explained = execution::CallOutput {
        status: CallStatus::Error,
        error_code: Some(ErrorCode::ComputeError),
        error_message: Some("the prompt is too long".into()),
        tokens: Vec::new(),
        ..completed()
    };
    assert!(refusal(explained.clone().validate()).contains("only an input-error completion"));
    execution::CallOutput {
        error_code: Some(ErrorCode::InputError),
        ..explained
    }
    .validate()
    .expect("an input error states its reason");

    let predicated = execution::CallOutput {
        status: CallStatus::Predicated,
        ..completed()
    };
    assert!(refusal(predicated.validate()).contains("predicated completion"));

    let misaligned = execution::CallOutput {
        token_logprobs: vec![-0.5, -0.25],
        ..completed()
    };
    assert!(refusal(misaligned.validate()).contains("do not align"));

    let empty_media = execution::CallOutput {
        media_output: Some(MediaOutput {
            handle: ArtifactHandle::PosixShm {
                name: "uniserve-artifact".into(),
            },
            bytes: 0,
        }),
        ..completed()
    };
    assert!(refusal(empty_media.validate()).contains("byte length must be positive"));
}

#[test]
fn a_batch_output_answers_each_call_of_its_batch_once() {
    let batch = token_batch();

    let mut foreign = batch_output();
    foreign.calls[1].call = execution::CallId(9);
    assert!(refusal(foreign.validate(&batch)).contains("did not carry"));

    let mut repeated = batch_output();
    repeated.calls[1].call = repeated.calls[0].call;
    assert!(refusal(repeated.validate(&batch)).contains("repeats a call"));
}
