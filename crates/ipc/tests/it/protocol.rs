//! Protocol round trips and validation behavior for every message family.
//!
//! Tests carry messages through a [`Frame`] as a channel does, and drive the
//! message validators. The `batch_with_calls` fixture derives the planner
//! columns `Batch::validate` expects, so a test can mutate a single field to
//! exercise one rule.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::net::TcpListener;
use std::time::{Duration, Instant};

use ipc::*;
use ipc::{
    CacheGroup, CacheGroupKind, CfgRenorm, ImageParams, RequestId, SamplingParams, TokenLogprob,
    UnitId,
};
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
        readout: None,
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

/// Two readout slots over a four-token canvas: slot 0 reads two candidates at
/// canvas token 0, slot 1 one candidate at canvas token 2.
fn readout() -> Readout {
    Readout {
        slot_tokens: vec![0, 2],
        candidate_offsets: vec![0, 2, 3],
        candidate_ids: vec![31, 32, 33],
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
        // A token-denoising call carries its canvas and the slots it reads.
        input_token_ids: if denoises {
            vec![4, 9, 4, 1]
        } else {
            Vec::new()
        },
        readout: denoises.then(readout),
        canvas: None,
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

fn completion_record() -> CallOutput {
    CallOutput {
        latent_layouts: Vec::new(),
        token_logprobs: vec![vec![
            TokenLogprob {
                token_id: 271,
                logprob: -0.25,
                rank: 1,
            },
            TokenLogprob {
                token_id: 42,
                logprob: f32::NEG_INFINITY,
                rank: 1000,
            },
        ]],
        prompt_logprobs: vec![vec![TokenLogprob {
            token_id: 7,
            logprob: -0.5,
            rank: 2,
        }]],
        request_key: request_key(),
        call_id: CallId::new(11, 0),

        status: CallStatus::Ok,

        error_code: None,
        error_message: None,
        timing_counters: TimingCounters::default(),
        cache_visible_len: 5,
        cache_computed_len: 5,
        step_index: 0,

        committed_tokens: vec![271],
        candidate_logprobs: Vec::new(),
        finish_flags: FinishFlags::default(),
        media_output: None,
        cache_output: None,
        draft_token_ids: Vec::new(),
    }
}

/// An autoregressive admission of a request with three input images.
fn admission() -> NewRequest {
    NewRequest::new(
        request_key(),
        u32::try_from(request_key().request_id.0).unwrap(),
        Some(GenerationParams {
            sampling: SamplingParams::default(),
            negative_token_ids: Vec::new(),
            finish_token_ids: vec![2, 7],
            initial_position: 0,
            generated_token_ids: Vec::new(),
            adapter: 0,
        }),
        None,
        3,
    )
    .unwrap()
}

fn media_sampling() -> DiffusionSamplingParams {
    DiffusionSamplingParams {
        num_frames: 22,
        video_units: 3,
        seed: 17,
        width: 1024,
        height: 768,
    }
}

/// A video admission whose conditioning the model family encoded; the
/// protocol carries it opaque.
fn media_admission(prompt_token_ids: Vec<u32>) -> NewRequest {
    NewRequest::new_media(
        request_key(),
        u32::try_from(request_key().request_id.0).unwrap(),
        prompt_token_ids,
        media_sampling(),
        VideoAdmission {
            conditioning: br#"{"task":"t2va"}"#.to_vec(),
        },
    )
    .unwrap()
}

/// Encodes `batch` as a submit request, decodes it, and returns the decoded batch.
fn execute_round_trip(batch: Batch) -> Batch {
    let request = WorkerRequest::submit(batch);
    let decoded = round_trip(&request);
    decoded.batch().unwrap().clone()
}

#[test]
fn forward_columns_preserve_cfg_rows_and_reject_misalignment() {
    let call = call_for(CallKind::Media(MediaCall::Denoising), CallId::new(11, 0));
    let mut run = batch_with_calls(3, vec![], vec![call]);
    // One alternative-prefix write followed by two denoising rows. Prefix
    // plus query lengths form total attention lengths even for read-only rows.
    run.forward = ForwardBatch {
        call_indices: vec![0, 0, 0],
        request_pool_indices: vec![2, 1, 2],
        seq_lens: vec![2, 17, 6],
        query_lens: vec![2, 4, 4],
        write_cache: vec![true, false, false],
    };
    assert_eq!(execute_round_trip(run.clone()), run);

    // Columns of different lengths, a sequence shorter than its query, and a
    // row naming a call outside the batch are each rejected.
    let mut missing_length = run.clone();
    missing_length.forward.seq_lens.pop();
    assert!(missing_length.validate().is_err());

    let mut short_sequence = run.clone();
    short_sequence.forward.seq_lens[2] = 3;
    assert!(short_sequence.validate().is_err());

    run.forward.call_indices[2] = 1;
    assert!(run.validate().is_err());
}

#[test]
fn call_coordinates_survive_physical_dispatch_and_reject_collisions() {
    let batch_id = u64::MAX - 7;
    let first = call_for(
        CallKind::Media(MediaCall::TextEncoding),
        CallId::new(batch_id, u32::MAX - 1),
    );
    let mut second = call_for(
        CallKind::Media(MediaCall::TextEncoding),
        CallId::new(batch_id, u32::MAX),
    );
    second.request_key = key_for_request(101);
    let run = batch_with_calls(3, vec![], vec![first.clone(), second]);
    assert_eq!(execute_round_trip(run.clone()), run);

    // A rank's projection of a batch can hold sparse logical indices. The
    // projection neither renumbers them nor narrows their unsigned range.
    let projection = batch_with_calls(4, vec![], vec![first.clone()]);
    assert_eq!(
        execute_round_trip(projection).calls[0].call_id,
        first.call_id
    );

    let mut collision = run;
    collision.calls[1].call_id = first.call_id;
    assert!(collision.validate().is_err());
}

/// Builds a batch around `calls` with the planner columns `Batch::validate`
/// expects for them: a buffer allocation for every declared buffer output, a
/// block table, cache-page allocation, and forward row for every call with cache
/// page capacity, latent params for every call that addresses a trajectory,
/// and a decode range for every video or audio decode or encode call.
fn batch_with_calls(batch_id: u64, admissions: Vec<NewRequest>, calls: Vec<Call>) -> Batch {
    // A call id names its batch, so the first call's batch id overrides the
    // `batch_id` argument.
    let batch_id = calls.first().map_or(batch_id, |call| call.call_id.batch_id);
    let mut run = Batch::new(batch_id, admissions, calls);
    run.collective_seq = batch_id.max(1);
    let mut next_buffer_offset = 0_u64;
    for (call_index, call) in run.calls.iter().enumerate() {
        for output in call.buffer_outputs() {
            next_buffer_offset = next_buffer_offset.div_ceil(256) * 256;
            let bytes = output.max_bytes();
            run.buffer_allocations.push(BufferAllocation {
                buffer: output.buffer_id(),
                offset: next_buffer_offset,
                bytes,
            });
            next_buffer_offset += bytes;
        }
        // Each forward call holds one fresh cache unit.
        let capacity_units = u32::from(matches!(call.kind, CallKind::Forward(_)));
        if capacity_units > 0 {
            // The request id doubles as the request pool slot, which forward
            // rows, block tables, and unit allocations require to be positive.
            let request_pool_idx = u32::try_from(call.request_key.request_id.0).unwrap();
            let unit_ids = (1..=capacity_units).map(UnitId).collect::<Vec<_>>();
            run.block_tables.push(BlockTable {
                request_pool_idx,
                group_id: 0,
                start_page: 0,
                unit_ids: unit_ids.clone(),
                allocated_tokens: call.bounds.max_tokens.max(1),
            });
            run.new_cache_units.push(CacheUnitAllocation {
                request_pool_idx,
                group_id: 0,
                unit_ids,
            });
            run.forward.push(
                call_index as u32,
                request_pool_idx,
                call.bounds.max_tokens.max(1),
                call.bounds.max_tokens.max(1),
                true,
            );
        }
        if call.addresses_latent_pages() {
            run.latent_params.push(LatentParams {
                request_key: call.request_key,
                call_id: call.call_id,
                page_table: vec![u32::try_from(call_index + 1).unwrap()],
                latent_units: 1,
                height: 1,
                width: 1,
            });
        }
        if matches!(
            call.kind,
            CallKind::Media(
                MediaCall::VideoDecoding
                    | MediaCall::VideoEncoding
                    | MediaCall::AudioDecoding
                    | MediaCall::AudioEncoding
            )
        ) {
            run.decode_ranges.push(DecodeRange {
                request_key: call.request_key,
                call_id: call.call_id,
                cursor: 0,
                max_units: 1,
                ranks: Vec::new(),
            });
        }
    }
    run
}

/// Builds a `BatchOutput`; the first completion's batch id, when present,
/// overrides `batch_id`.
fn batch_output(
    batch_id: u64,
    completions: Vec<CallOutput>,
    products: Vec<TensorExport>,
    worker_exec_us: Option<u64>,
    forward_stats: Option<ForwardStats>,
) -> BatchOutput {
    BatchOutput {
        batch_id: completions
            .first()
            .map_or(batch_id, |record| record.call_id.batch_id),
        completions,
        products,

        worker_exec_us,
        forward_stats,
    }
}

#[test]
fn every_call_kind_round_trips_through_ipc() {
    let variants = CallKind::ALL;
    for (index, work) in variants.into_iter().enumerate() {
        let call = call_for(work, CallId::new(100 + index as u64, 0));
        let batch = execute_round_trip(batch_with_calls(1, Vec::new(), vec![call.clone()]));
        assert_eq!(batch.calls().next().unwrap(), &call);
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

/// A token-denoising readout reaches the rank with its canvas and slots, and
/// its completion returns the candidates' log-probabilities in candidate
/// order.
#[test]
fn a_readout_and_its_candidate_logprobs_round_trip() {
    let call_id = CallId::new(5, 0);
    let call = call_for(CallKind::Forward(ForwardMode::TokenDenoising), call_id);
    let batch = execute_round_trip(batch_with_calls(5, Vec::new(), vec![call.clone()]));
    assert_eq!(batch.calls().next().unwrap(), &call);

    let mut record = completion_record();
    record.call_id = call_id;
    record.token_logprobs.clear();
    record.prompt_logprobs.clear();
    record.committed_tokens.clear();
    record.candidate_logprobs = vec![-0.25, -3.5, -17.75];
    let report = batch_output(5, vec![record], Vec::new(), None, None);
    let decoded = round_trip(&WorkerResponse::result(report.clone()));
    assert_eq!(decoded.output().unwrap(), &report);
}

/// A readout belongs to a token-denoising call and must address the canvas
/// that call carries, with every slot reading at least one candidate.
#[test]
fn a_readout_must_address_the_canvas_of_a_token_denoising_call() {
    let call_id = CallId::new(5, 0);
    let valid = call_for(CallKind::Forward(ForwardMode::TokenDenoising), call_id);
    assert!(valid.validate().is_ok());

    let mut unread = valid.clone();
    unread.readout = None;
    let mut misplaced = call_for(CallKind::Forward(ForwardMode::Prefill), call_id);
    misplaced.readout = Some(readout());
    misplaced.input_token_ids = vec![1, 2, 3, 4];
    misplaced.bounds.max_tokens = 4;
    let mut outside = valid.clone();
    outside.readout.as_mut().unwrap().slot_tokens[1] = 4;
    let mut empty_slot = valid.clone();
    empty_slot.readout.as_mut().unwrap().candidate_offsets = vec![0, 3, 3];
    let mut short_offsets = valid.clone();
    short_offsets.readout.as_mut().unwrap().candidate_offsets = vec![0, 2];
    let mut unordered = valid.clone();
    unordered.readout.as_mut().unwrap().slot_tokens.reverse();
    let mut sampling = valid.clone();
    sampling.token_output = Some(output_product(call_id));
    for call in [
        unread,
        misplaced,
        outside,
        empty_slot,
        short_offsets,
        unordered,
        sampling,
    ] {
        assert!(call.validate().is_err(), "{:?}", call.readout);
    }
}

/// A canvas step of a block-diffusion request whose admission names its
/// canvas sampling.
fn canvas_step_batch() -> Batch {
    let call_id = CallId::new(6, 0);
    let mut call = call_for(CallKind::Forward(ForwardMode::TokenDenoising), call_id);
    call.readout = None;
    call.input_token_ids.clear();
    call.bounds.max_tokens = 256;
    call.canvas = Some(CanvasStep { block: 2, step: 5 });
    let admission = NewRequest::new(
        request_key(),
        u32::try_from(request_key().request_id.0).unwrap(),
        Some(GenerationParams {
            sampling: SamplingParams {
                seed: Some(11),
                ..SamplingParams::default()
            },
            negative_token_ids: Vec::new(),
            finish_token_ids: vec![1, 50, 106],
            initial_position: 0,
            generated_token_ids: Vec::new(),
            adapter: 0,
        }),
        None,
        0,
    )
    .unwrap();
    batch_with_calls(6, vec![admission], vec![call])
}

/// A canvas step reaches the rank with its block and step and its request's
/// admitted sampling, and its completion returns the stopped block's tokens.
#[test]
fn a_canvas_step_its_sampling_and_its_block_round_trip() {
    let batch = canvas_step_batch();
    assert_eq!(execute_round_trip(batch.clone()), batch);

    let mut record = completion_record();
    record.call_id = CallId::new(6, 0);
    record.token_logprobs.clear();
    record.prompt_logprobs.clear();
    record.committed_tokens = (0..256).collect();
    let report = batch_output(6, vec![record], Vec::new(), None, None);
    let decoded = round_trip(&WorkerResponse::result(report.clone()));
    assert_eq!(decoded.output().unwrap(), &report);
}

/// A canvas step belongs to a token-denoising call, which carries either a
/// readout or a canvas step, and denoises its resident canvas without input
/// tokens.
#[test]
fn a_canvas_step_is_a_token_denoising_call_without_input_tokens() {
    let valid = canvas_step_batch().calls().next().unwrap().clone();
    assert!(valid.validate().is_ok());

    let mut misplaced = call_for(CallKind::Forward(ForwardMode::Prefill), valid.call_id);
    misplaced.canvas = valid.canvas;
    let mut both = valid.clone();
    both.readout = Some(readout());
    both.input_token_ids = vec![4, 9, 4, 1];
    let mut with_tokens = valid.clone();
    with_tokens.input_token_ids = vec![1; 256];
    let mut empty = valid.clone();
    empty.bounds.max_tokens = 0;
    for call in [misplaced, both, with_tokens, empty] {
        assert!(call.validate().is_err(), "{:?}", call.canvas);
    }
}

/// Only a successful completion reports candidate log-probabilities; that
/// it answers a token-denoising call is the engine's check, which holds the
/// call.
#[test]
fn candidate_logprobs_belong_to_a_successful_completion() {
    let mut record = completion_record();
    record.candidate_logprobs = vec![-1.0];
    assert!(record.validate().is_ok());
    record.status = CallStatus::Error;
    record.error_code = Some(ErrorCode::ComputeError);
    assert!(record.validate().is_err());
}

#[test]
fn every_call_kind_round_trips_in_its_own_batch() {
    let batches = comprehensive_batches();
    assert_eq!(batches.len(), CallKind::ALL.len());
    for batch in batches {
        assert_eq!(execute_round_trip(batch.clone()), batch);
    }
}

#[test]
fn a_batch_carries_one_call_kind_through_one_entry() {
    let decode = call_for(CallKind::Forward(ForwardMode::Decode), CallId::new(1, 0));
    let prefill = call_for(CallKind::Forward(ForwardMode::Prefill), CallId::new(1, 1));
    let mixed = batch_with_calls(1, Vec::new(), vec![decode.clone(), prefill]);
    let error = mixed.validate().unwrap_err().to_string();
    assert!(error.contains("mixes call kinds or components"), "{error}");

    let mut other_entry = call_for(CallKind::Forward(ForwardMode::Decode), CallId::new(1, 1));
    other_entry.component = "encoder".into();
    let split = batch_with_calls(1, Vec::new(), vec![decode, other_entry]);
    let error = split.validate().unwrap_err().to_string();
    assert!(error.contains("mixes call kinds or components"), "{error}");
}

#[test]
fn latent_parameters_round_trip_with_request_or_paged_storage() {
    let mut call = call_for(CallKind::Media(MediaCall::Denoising), CallId::new(101, 0));
    call.latent_output = Some(output_product(call.call_id));
    let mut run = batch_with_calls(1, Vec::new(), vec![call]);
    assert_eq!(execute_round_trip(run.clone()), run);

    // Paged storage names pages and a positive unit count; request-owned
    // storage has neither. A mixture of the two is rejected.
    run.latent_params[0].page_table.clear();
    run.latent_params[0].latent_units = 0;
    assert_eq!(execute_round_trip(run.clone()), run);
    run.latent_params[0].latent_units = 1;
    assert!(run.validate().is_err());
    run.latent_params[0].latent_units = 0;
    run.latent_params[0].page_table.push(1);
    assert!(run.validate().is_err());
}

#[test]
fn media_tracks_preserve_independent_ranges_and_tensor_dependencies() {
    let mut latent = output_product(CallId::new(55, 0));
    latent.dtype = DType::F32;
    latent.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(8), DimBound::Static(32)],
    };
    for (index, (entry, stage)) in [
        ("video_decoder", MediaCall::VideoDecoding),
        ("audio_decoder", MediaCall::AudioDecoding),
    ]
    .iter()
    .enumerate()
    {
        let call = Call {
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
            cache_input: None,
            cache_output: None,
            input_token_ids: Vec::new(),
            readout: None,
            canvas: None,
            sampling_state: None,
            request_key: request_key(),
            call_id: CallId::new(56 + index as u64, 0),
            component: (*entry).into(),
            kind: CallKind::Media(*stage),
            bounds: Bounds::default(),
            inputs: vec![latent.clone()],
            outputs: Vec::new(),
            predicate: None,
            image_index: None,
            latent_restore: None,
            host_output: None,
            latent_draw: None,
            carried_image: None,
        };
        // A video decode range may start past the first unit, while an audio
        // decode range must start at it. Neither call may omit its range.
        let mut run = batch_with_calls(index as u64 + 1, Vec::new(), vec![call]);
        if index == 0 {
            run.decode_ranges[0].cursor = 3;
            run.decode_ranges[0].max_units = 2;
        }
        assert_eq!(execute_round_trip(run.clone()), run);
        if index == 1 {
            run.decode_ranges[0].cursor = 1;
            assert!(run.validate().is_err());
        }
        run.decode_ranges.clear();
        assert!(run.validate().is_err());
    }
}

#[test]
fn block_table_allocation_round_trips_with_the_call() {
    let base = ar_decode_call();
    let batch = execute_round_trip(batch_with_calls(9, Vec::new(), vec![base.clone()]));
    assert_eq!(batch.calls().next().unwrap(), &base);
    assert_eq!(batch.block_tables[0].unit_ids, vec![UnitId(1)]);
}

/// A sliding-window group's table carries only its in-window pages: its
/// start page and page-major units survive the round trip, and its fresh
/// units must lie inside the table and never name the unit-zero sentinel.
#[test]
fn windowed_block_table_round_trips_its_start_page_and_units() {
    let mut run = batch_with_calls(9, Vec::new(), vec![ar_decode_call()]);
    let slot = run.block_tables[0].request_pool_idx;
    run.block_tables.push(BlockTable {
        request_pool_idx: slot,
        group_id: 1,
        start_page: 3,
        unit_ids: vec![UnitId(7), UnitId(8), UnitId(9), UnitId(10)],
        allocated_tokens: 5 * 16,
    });
    run.new_cache_units.push(CacheUnitAllocation {
        request_pool_idx: slot,
        group_id: 1,
        unit_ids: vec![UnitId(9), UnitId(10)],
    });
    assert_eq!(execute_round_trip(run.clone()), run);

    let mut outside = run.clone();
    outside.new_cache_units[1].unit_ids = vec![UnitId(11)];
    assert!(outside.validate().is_err());
    // A state group's page without a checkpoint names the padding sentinel
    // for its units; a real unit appears once.
    let mut sentinel = run.clone();
    sentinel.block_tables[1].unit_ids[0] = UnitId(0);
    sentinel.block_tables[1].unit_ids[1] = UnitId(0);
    assert!(sentinel.validate().is_ok());
    let mut repeated = run;
    repeated.block_tables[1].unit_ids[0] = UnitId(8);
    assert!(repeated.validate().is_err());
}

#[test]
fn export_round_trips_its_registered_view_and_endpoint() {
    let mut product = output_product(CallId::new(11, 0));
    product.dtype = DType::F32;
    product.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(4)],
    };
    for transport in [
        TransferTransport::CudaVmm {
            endpoint: "uniserve-cuda-physical-incarnation".into(),
            export_id: "0123456789abcdef0123456789abcdef".into(),
            storage_size_bytes: 4096,
            storage_offsets_bytes: vec![128, 64],
            span_lengths: vec![2],
            span_counts: vec![2],
            tensor_stride: vec![2],
            ready_event_handle: vec![9; 64],
            allocation_handle: vec![7; 64],
            acknowledgment_offset: -1,
            resident: true,
        },
        TransferTransport::PosixShm {
            endpoint: "uniserve-shm-physical-incarnation".into(),
            name: "uniserve-export".into(),
        },
    ] {
        let mut report = batch_output(
            5,
            vec![completion_record()],
            vec![TensorExport {
                product: product.clone(),
                value: TransferHandle::DeviceProduct {
                    height: 0,
                    width: 0,
                    value_range: String::new(),
                    tensor: TensorTransfer {
                        shape: vec![4],
                        locations: vec![Locator {
                            source: WorkerInfo::default().endpoint,
                            transport,
                            nbytes: 16,
                            dtype: "float32".into(),
                            shape: vec![4],
                            offset: vec![0],
                            device: "cuda:1".into(),
                        }],
                    },
                },
            }],
            None,
            None,
        );

        // A replica differs only in its source incarnation. A report for another
        // generation names a different identity: merging it fails and leaves
        // the export unchanged.
        let original = &mut report.products[0];
        let mut replica = original.clone();
        if let TransferHandle::DeviceProduct { tensor, .. } = &mut replica.value {
            tensor.locations[0].source.incarnation = "second-rank".into();
        }
        let mut stale = replica.clone();
        stale.product.generation += 1;
        let before = original.clone();
        assert!(original.merge_locations(&stale).is_err());
        assert_eq!(*original, before);
        original.merge_locations(&replica).unwrap();
        let decoded = round_trip(&WorkerResponse::result(report.clone()));
        assert_eq!(decoded.output().unwrap(), &report);

        // Other product dtypes round-trip with locator byte sizes that follow
        // their element widths.
        for (dtype, storage_dtype, nbytes) in [(DType::I64, "int64", 32), (DType::I16, "int16", 8)]
        {
            let mut typed_report = report.clone();
            typed_report.products[0].product.dtype = dtype;
            let TransferHandle::DeviceProduct { tensor, .. } = &mut typed_report.products[0].value
            else {
                unreachable!()
            };
            for location in &mut tensor.locations {
                location.dtype = storage_dtype.into();
                location.nbytes = nbytes;
            }
            let decoded = round_trip(&WorkerResponse::result(typed_report.clone()));
            assert_eq!(decoded.output().unwrap(), &typed_report);
        }
    }
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
                        source: WorkerInfo::default().endpoint,
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

#[test]
fn unchanged_cache_export_round_trips_without_physical_tensors() {
    let source = BufferId {
        owner: request_key(),
        producer_call_id: CallId::new(11, 0),
        output_index: 0,
        generation: 3,
    };
    let mut report = batch_output(
        5,
        vec![CallOutput {
            cache_output: Some(CacheTransfer {
                groups: Vec::new(),
                source,
                destination: "decoder".into(),
                base: Some(source),
                base_extent: 16,
                exported_extent: 16,
                compute_dtype: "bfloat16".into(),
            }),
            ..completion_record()
        }],
        Vec::new(),
        None,
        None,
    );
    let decoded = round_trip(&WorkerResponse::result(report.clone()));
    assert_eq!(decoded.output().unwrap(), &report);

    // A grown extent must carry the tensors of its suffix.
    let CacheTransfer {
        exported_extent, ..
    } = report.completions[0].cache_output.as_mut().unwrap();
    *exported_extent = 17;
    assert!(report.validate().is_err());
}

/// Shards of a 4x4 tensor: the left half plus the top-right quarter leave the
/// bottom-right quarter uncovered until it is added. A full replica then
/// substitutes for the left-half shard, and removing that replica uncovers
/// the left half again.
#[test]
fn tensor_coverage_preserves_replicas_and_detects_missing_regions() {
    let shard = |offset: Vec<u64>, shape: Vec<u64>| Locator {
        source: WorkerInfo::default().endpoint,
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

#[test]
fn raw_cache_export_round_trips_page_representation_and_exact_request() {
    let source = BufferId {
        owner: request_key(),
        producer_call_id: CallId::new(11, 0),
        output_index: 0,
        generation: 3,
    };
    let tensor = |name: &str, dtype: &str, shape: Vec<u64>, itemsize: u64| TensorTransfer {
        shape: shape.clone(),
        locations: vec![Locator {
            source: WorkerInfo::default().endpoint,
            transport: TransferTransport::PosixShm {
                endpoint: "uniserve-cache-publisher".into(),
                name: name.into(),
            },
            nbytes: shape.iter().product::<u64>() * itemsize,
            dtype: dtype.into(),
            offset: vec![0; shape.len()],
            shape,
            device: "cpu".into(),
        }],
    };
    // Five tokens past a base of three, with four-token pages, touch two
    // source pages of the full-attention group. Each tensor is one rank's
    // shard: three of six kv heads starting at head three, and for FP8 one of
    // two scale head groups. A sliding-window group with a two-token history
    // carries only tokens [6, 8) of its own layers.
    for (dtype, itemsize) in [("bfloat16", 2), ("float8_e4m3fn", 1)] {
        let mut tensors = vec![
            tensor("keys", dtype, vec![5, 2, 3, 4], itemsize),
            tensor("values", dtype, vec![5, 2, 3, 4], itemsize),
        ];
        for field in &mut tensors {
            field.shape[2] = 6;
            field.locations[0].offset[2] = 3;
        }
        let mut window = vec![
            tensor("window-keys", dtype, vec![2, 1, 2, 8], itemsize),
            tensor("window-values", dtype, vec![2, 1, 2, 8], itemsize),
        ];
        if dtype == "float8_e4m3fn" {
            let mut scales = tensor("scales", "float32", vec![2, 2, 2, 1], 4);
            scales.shape[3] = 2;
            scales.locations[0].offset[3] = 1;
            tensors.push(scales);
            // Tokens [6, 8) lie in one eight-token source page.
            window.push(tensor("window-scales", "float32", vec![1, 2, 1, 1], 4));
        }
        let report = batch_output(
            5,
            vec![CallOutput {
                cache_output: Some(CacheTransfer {
                    groups: vec![
                        CacheGroupTransfer {
                            start: 3,
                            block_size: 4,
                            tensors,
                        },
                        CacheGroupTransfer {
                            start: 6,
                            block_size: 8,
                            tensors: window,
                        },
                    ],
                    source,
                    destination: "decoder".into(),
                    base: Some(source),
                    base_extent: 3,
                    exported_extent: 8,
                    compute_dtype: "bfloat16".into(),
                }),
                ..completion_record()
            }],
            Vec::new(),
            None,
            None,
        );
        let decoded = round_trip(&WorkerResponse::result(report.clone()));
        assert_eq!(decoded.output().unwrap(), &report);

        // Zero generations, a zero block size, a group starting before the
        // base, an unchanged extent with groups, and malformed or unexpected
        // scales are each rejected on encode.
        for invalid_field in ["source", "base", "block_size", "start", "extent", "scales"] {
            let mut invalid = report.clone();
            let CacheTransfer {
                source,
                base,
                base_extent,
                exported_extent,
                groups,
                ..
            } = invalid.completions[0].cache_output.as_mut().unwrap();
            match invalid_field {
                "source" => source.generation = 0,
                "base" => base.as_mut().unwrap().generation = 0,
                "block_size" => groups[0].block_size = 0,
                "start" => groups[1].start = 2,
                "extent" => *base_extent = *exported_extent,
                "scales" if dtype == "float8_e4m3fn" => {
                    groups[0].tensors[2] = tensor("scales", "float32", vec![1, 2, 2], 4);
                }
                "scales" => groups[0]
                    .tensors
                    .push(tensor("scales", "float32", vec![2, 2, 2], 4)),
                _ => unreachable!(),
            }
            assert!(invalid.validate().is_err());
        }
    }
}

#[test]
fn error_completion_round_trips_with_its_error_code() {
    let mut record = completion_record();
    record.status = CallStatus::Error;
    record.error_code = Some(ErrorCode::ComputeError);
    let report = batch_output(6, vec![record.clone()], Vec::new(), None, None);
    let decoded = round_trip(&WorkerResponse::result(report));
    assert_eq!(
        decoded.output().unwrap().completions().next().unwrap(),
        &record
    );
}

/// A refusal carries its reason across the channel, and only an input-error
/// completion carries one.
#[test]
fn a_refused_completion_round_trips_with_its_reason() {
    let mut record = completion_record();
    record.status = CallStatus::Error;
    record.error_code = Some(ErrorCode::InputError);
    record.error_message = Some("the conditions take 7808 denoiser rows".to_owned());
    let report = batch_output(6, vec![record.clone()], Vec::new(), None, None);
    let decoded = round_trip(&WorkerResponse::result(report));
    assert_eq!(
        decoded.output().unwrap().completions().next().unwrap(),
        &record
    );

    let mut silent = record.clone();
    silent.error_message = None;
    assert!(silent.validate().is_ok());
    let mut explained = record;
    explained.error_code = Some(ErrorCode::ComputeError);
    assert!(explained.validate().is_err());
}

#[test]
fn request_retirement_requires_unique_buffers_from_its_request() {
    let product = tensor_for(
        request_key(),
        CallId::new(7, 0),
        0,
        DType::F16,
        ShapeBound {
            dims: vec![DimBound::Static(2), DimBound::Device { max: 16 }],
        },
    );

    // Both a repeated buffer and a buffer owned by another request are
    // rejected.
    for retained_buffers in [
        vec![product.buffer_id(), product.buffer_id()],
        vec![BufferId {
            owner: key_for_request(900),
            ..product.buffer_id()
        }],
    ] {
        let command = BatchCommand::Finish {
            request_key: request_key(),
            retained_buffers,
        };
        assert!(
            batch_with_calls(3, vec![admission()], vec![ar_decode_call()])
                .with_commands(vec![command])
                .validate()
                .is_err()
        );
    }
}

#[test]
fn every_batch_command_variant_round_trips_through_ipc() {
    let finish = BatchCommand::Finish {
        request_key: request_key(),
        retained_buffers: vec![
            tensor_for(
                request_key(),
                CallId::new(7, 0),
                0,
                DType::F16,
                ShapeBound {
                    dims: vec![DimBound::Static(2), DimBound::Device { max: 16 }],
                },
            )
            .buffer_id(),
        ],
    };
    let free = BatchCommand::Free {
        buffer: tensor_for(
            request_key(),
            CallId::new(8, 0),
            0,
            DType::F16,
            ShapeBound {
                dims: vec![DimBound::Static(2), DimBound::Device { max: 16 }],
            },
        )
        .buffer_id(),
    };
    let batch = batch_with_calls(3, vec![admission()], vec![ar_decode_call()])
        .with_commands(vec![finish.clone(), free.clone()]);
    let decoded = execute_round_trip(batch);
    assert_eq!(
        decoded.commands,
        vec![
            BatchCommand::Start {
                request: Box::new(admission())
            },
            finish,
            free
        ]
    );
}

#[test]
fn new_request_round_trips() {
    let batch = execute_round_trip(batch_with_calls(
        4,
        vec![admission()],
        vec![ar_decode_call()],
    ));
    assert_eq!(batch.admissions().next(), Some(&admission()));
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
fn maximum_media_prompt_round_trips() {
    let prompt_token_ids = (0..16_384_u32).map(|index| 100_000 + index).collect();
    let admission = media_admission(prompt_token_ids);
    let request = WorkerRequest::submit(batch_with_calls(
        5,
        vec![admission.clone()],
        vec![call_for(
            CallKind::Media(MediaCall::LatentPreparation),
            CallId::new(12, 0),
        )],
    ));

    let decoded = round_trip(&request);
    assert_eq!(
        decoded.batch().unwrap().admissions().next(),
        Some(&admission)
    );
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
fn batch_inputs_require_a_consumer_for_their_payload_kind() {
    let mut feature = output_product(CallId::new(3, 0));
    feature.dtype = DType::F32;
    feature.shape_bound.dims = vec![DimBound::Static(2)];
    let mut call = ar_decode_call();
    call.vision_inputs = vec![VisionInput {
        offset: 0,
        feature: feature.clone(),
    }];
    let mut batch = batch_with_calls(11, Vec::new(), vec![call]);
    batch.input_products.push(TensorExport {
        product: feature,
        value: TransferHandle::Encoder {
            height: 1,
            width: 1,
            payload_kind: FeatureKind::Vision,
            tensor: TensorTransfer {
                shape: vec![2],
                locations: vec![Locator {
                    source: WorkerInfo::default().endpoint,
                    transport: TransferTransport::Local {
                        endpoint: "encoder".into(),
                        key: 1,
                    },
                    nbytes: 8,
                    dtype: "float32".into(),
                    shape: vec![2],
                    offset: vec![0],
                    device: "cpu".into(),
                }],
            },
        },
    });
    batch.validate().unwrap();

    // The same numerical shape cannot turn a vision input into a latent input.
    let TransferHandle::Encoder { payload_kind, .. } = &mut batch.input_products[0].value else {
        unreachable!()
    };
    *payload_kind = FeatureKind::Latent;
    assert!(batch.validate().is_err());

    let feature = batch.calls[0].vision_inputs.pop().unwrap().feature;
    batch.calls[0].latent_feature_input = Some(feature);
    batch.validate().unwrap();
}

#[test]
fn a_group_batch_is_unplaced_and_each_rank_projection_is_placed() {
    let mut call = call_for(CallKind::Transfer(TransferMode::Tensor), CallId::new(1, 0));
    let input = output_product(CallId::new(9, 0));
    let output = output_product(CallId::new(1, 0));
    call.bounds.max_transfer_bytes = input.max_bytes();
    call.inputs = vec![input];
    call.outputs = vec![output];

    // A rank's batch names the span of every persistent output in that
    // rank's own pool.
    let placed = batch_with_calls(1, Vec::new(), vec![call]);
    assert_eq!(placed.buffer_allocations.len(), 1);
    placed.validate().unwrap();

    // A group's batch is validated before any rank's spans are chosen.
    let mut unplaced = placed.clone();
    unplaced.buffer_allocations.clear();
    assert!(unplaced.validate().is_err());
    unplaced.validate_unplaced().unwrap();
    assert!(placed.validate_unplaced().is_err());
}

#[test]
fn batch_rejects_two_calls_for_one_request() {
    let batch = Batch {
        batch_id: 1,
        collective_seq: 1,
        calls: vec![ar_decode_call(), ar_decode_call()],
        block_tables: Vec::new(),
        new_cache_units: Vec::new(),
        forward: ForwardBatch::default(),
        latent_params: Vec::new(),
        decode_ranges: Vec::new(),
        buffer_allocations: Vec::new(),
        commands: Vec::new(),
        input_products: Vec::new(),
        cache_inputs: Vec::new(),
        page_loads: Vec::new(),
        page_offloads: Vec::new(),
        latent_offloads: Vec::new(),
    };
    assert!(batch.validate().is_err());
}

#[test]
fn a_batch_may_carry_page_writes_alone() {
    let mut batch = Batch {
        batch_id: 1,
        collective_seq: 1,
        calls: Vec::new(),
        block_tables: Vec::new(),
        new_cache_units: Vec::new(),
        forward: ForwardBatch::default(),
        latent_params: Vec::new(),
        decode_ranges: Vec::new(),
        buffer_allocations: Vec::new(),
        commands: Vec::new(),
        input_products: Vec::new(),
        cache_inputs: Vec::new(),
        page_loads: Vec::new(),
        page_offloads: Vec::new(),
        latent_offloads: Vec::new(),
    };
    assert!(batch.validate().is_err());
    batch.page_offloads.push(PageCopy {
        rank: 0,
        pool: 1,
        group_id: 0,
        slot: 3,
        units: vec![UnitId(5)],
    });
    batch.validate().unwrap();
}

#[test]
fn token_inputs_preserve_order_and_enforce_capacity() {
    for tokens in [vec![], vec![42], vec![1, 2, 3, 4, 5], vec![u32::MAX, 0, 7]] {
        let mut call = ar_decode_call();
        call.bounds.max_tokens = 5;
        call.input_token_ids = tokens.clone();
        let decoded = execute_round_trip(batch_with_calls(1, Vec::new(), vec![call]));
        assert_eq!(decoded.calls[0].input_token_ids, tokens);
    }
    let mut call = ar_decode_call();
    call.bounds.max_tokens = 1;
    call.input_token_ids = vec![7, 8];
    assert!(call.validate().is_err());
}

#[test]
fn call_sampling_state_preserves_whitelist_presence() {
    for allowed_token_ids in [None, Some(Vec::new()), Some(vec![2, 7])] {
        let mut call = ar_decode_call();
        call.sampling_state = Some(SamplingState {
            allowed_token_ids,
            suppressed_token_ids: vec![3, 9],
            finish_token_ids: vec![5, 11],
            transition_token_ids: vec![13, 29],
            force_finish: true,
        });
        let request = WorkerRequest::submit(batch_with_calls(1, Vec::new(), vec![call]));
        assert_eq!(round_trip(&request), request);
    }
}

#[test]
fn image_encoder_input_round_trips_and_rejects_an_incompatible_call() {
    let mut call = ar_decode_call();
    call.kind = CallKind::Media(MediaCall::VisionEncoding);
    call.input_image = Some("iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAGUlEQVR4nGN0SGhgIAUwkaR6VMOohiGlAQCjvQFA6eri4wAAAABJRU5ErkJggg==".into());
    call.token_output = None;
    let batch = batch_with_calls(1, vec![admission()], vec![call.clone()]);
    let decoded = execute_round_trip(batch);
    assert_eq!(
        decoded.calls().next().unwrap().input_image,
        call.input_image
    );

    for (kind, image) in [
        (
            CallKind::Forward(ForwardMode::Decode),
            call.input_image.clone(),
        ),
        (
            CallKind::Media(MediaCall::VisionEncoding),
            Some(String::new().into()),
        ),
    ] {
        let mut invalid = call.clone();
        invalid.kind = kind;
        invalid.input_image = image;
        let batch = batch_with_calls(1, vec![admission()], vec![invalid]);
        assert!(batch.validate().is_err());
    }
}

#[test]
fn batch_rejects_a_conflicting_command_identity() {
    let finish = |retained_buffers| BatchCommand::Finish {
        request_key: request_key(),
        retained_buffers,
    };
    let buffer = tensor_for(
        request_key(),
        CallId::new(7, 0),
        0,
        DType::F16,
        ShapeBound {
            dims: vec![DimBound::Static(2), DimBound::Device { max: 16 }],
        },
    )
    .buffer_id();
    let batch = batch_with_calls(1, vec![admission()], vec![ar_decode_call()])
        .with_commands(vec![finish(vec![]), finish(vec![buffer])]);
    assert!(batch.validate().is_err());
}

#[test]
fn batch_allows_a_duplicate_identical_command() {
    let finish = BatchCommand::Finish {
        request_key: request_key(),
        retained_buffers: vec![],
    };
    let batch = batch_with_calls(1, vec![admission()], vec![ar_decode_call()])
        .with_commands(vec![finish.clone(), finish]);
    assert!(batch.validate().is_ok());
}

#[test]
fn worker_info_round_trips() {
    let info = WorkerInfo {
        prefix_cache: Some(PrefixCacheInfo {
            groups: vec![CacheGroup {
                layer_ids: (11..20).collect(),
                ..WorkerInfo::default().prefix_cache.unwrap().groups[0].clone()
            }],
            ..WorkerInfo::default().prefix_cache.unwrap()
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
    let response = WorkerResponse::info(info.clone());
    let decoded = round_trip(&response);
    let WorkerResponse::Info { info: decoded, .. } = decoded else {
        panic!("decoded response must preserve its info variant");
    };
    assert_eq!(decoded, info);
}

/// A component's distribution is omitted from its encoding when absent, so
/// both forms must decode, each with its own units per rank.
#[test]
fn a_component_round_trips_with_and_without_a_distribution() {
    use ipc::{ComponentConfig, ComponentDistribution, ParallelConfig};

    let grouped = ComponentConfig::parallel(vec![0], ParallelConfig::default());
    let distributed = ComponentConfig {
        distribution: Some(ComponentDistribution::TemporalUnits),
        units_per_rank: 3,
        ..grouped.clone()
    };
    for config in [grouped, distributed] {
        assert_eq!(round_trip(&config), config);
    }
}

/// A component's placement reaches its ranks in the launch descriptor with
/// every sequence-parallel strategy, a Ulysses group's head projection
/// included.
#[test]
fn a_component_round_trips_with_every_sequence_parallel_strategy() {
    use ipc::{ComponentConfig, ParallelConfig, SequenceParallel};

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
        let parallel = ParallelConfig {
            sequence_parallel,
            ..Default::default()
        };
        let count = parallel.world_size().unwrap();
        let config = ComponentConfig::parallel((0..count).rev().collect(), parallel);
        assert_eq!(round_trip(&config), config);
    }
}

#[test]
fn cache_free_worker_info_round_trips() {
    let info = WorkerInfo {
        media_components: video_components(),
        supported_calls: video_components()
            .into_keys()
            .map(CallKind::Media)
            .collect(),
        prefix_cache: None,
        latent_page_units: 64,
        latent_pages: 3,
        ..WorkerInfo::default()
    };
    let response = WorkerResponse::info(info.clone());
    let decoded = round_trip(&response);
    let WorkerResponse::Info { info: decoded, .. } = decoded else {
        panic!("decoded response must preserve its info variant");
    };
    assert_eq!(decoded, info);
}

#[test]
fn worker_info_rejects_duplicate_set_members() {
    let info = WorkerInfo {
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
fn worker_info_rejects_inconsistent_cache_groups() {
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
    let first = full_caps().prefix_cache.unwrap();
    let mut second = first.clone();
    second.num_units = 1024;
    second.groups[0].layer_ids = vec![28, 29];
    let merged = first.merge(&second).unwrap();
    assert_eq!(merged.num_units, 1024);
    assert_eq!(merged.groups[0].layer_ids.len(), 16);
    assert_eq!(merged.groups[1].layer_ids, first.groups[1].layer_ids);

    second.groups[1].units_per_page = 3;
    assert!(first.merge(&second).is_err());
}

fn key_for_request(request: u64) -> RequestKey {
    RequestKey::new(4, RequestId(request), 2)
}

fn tensor_for(
    key: RequestKey,
    call: CallId,
    output_index: u16,
    dtype: DType,
    shape_bound: ShapeBound,
) -> TensorRef {
    TensorRef {
        request_key: key,
        producer_call_id: call,
        output_index,
        generation: 3 + u32::from(output_index),
        dtype,
        shape_bound,
    }
}

fn full_sampling() -> SamplingParams {
    SamplingParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        ignore_eos: true,
        seed: Some(0xdead_beef),
        min_p: 0.05,
        repetition_penalty: 1.1,
        frequency_penalty: 0.25,
        presence_penalty: -0.5,
        logit_bias: vec![(5, 1.5), (11, -2.0)],
        min_tokens: 3,
        n_logprobs: 4,
        return_logprobs: true,
        return_prompt_logprobs: true,
        n_prompt_logprobs: 2,
        bad_words_ids: vec![vec![1, 2, 3], vec![42]],
        allowed_token_ids: Some(vec![10, 11, 12]),
        typical_p: 0.8,
        forced_token_ids: vec![13, 14],
    }
}

fn full_image() -> ImageParams {
    ImageParams {
        steps: 20,
        cfg_text_scale: 5.0,
        cfg_img_scale: 1.5,
        cfg_renorm_type: CfgRenorm::Global,
        cfg_renorm_min: 0.1,
        cfg_interval: (0.2, 0.9),
        timestep_shift: 3.0,
        height: 1024,
        width: 768,
        seed: Some(77),
    }
}

/// One batch per closed `CallKind` variant, each carrying that variant's
/// single call on its own request key. A batch is one numerical call on one
/// component, so the variants cannot share one. The first batch also admits
/// requests 100 and 110, and the last retires request 201.
fn comprehensive_batches() -> Vec<Batch> {
    let ar_params = NewRequest::new(
        key_for_request(100),
        100,
        Some(GenerationParams {
            sampling: full_sampling(),
            negative_token_ids: vec![100, 101],
            finish_token_ids: vec![2, 7],
            initial_position: 128,
            generated_token_ids: Vec::new(),
            adapter: 0,
        }),
        None,
        0,
    )
    .unwrap();
    let image_params =
        NewRequest::new(key_for_request(110), 110, None, Some(full_image()), 2).unwrap();
    let variants = CallKind::ALL;
    let mut batches = Vec::new();
    for (index, kind) in variants.into_iter().enumerate() {
        let mut calls = Vec::new();
        let key = key_for_request(100 + index as u64);
        let call_id = CallId::new(42 + index as u64, 0);
        calls.push(Call {
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
            cache_input: (kind == CallKind::Transfer(TransferMode::CacheInstall)).then_some(
                BufferId {
                    owner: key,
                    producer_call_id: CallId::new(2, 0),
                    output_index: 0,
                    generation: 1,
                },
            ),
            cache_output: matches!(
                kind,
                CallKind::Transfer(TransferMode::CacheExport | TransferMode::CacheInstall)
            )
            .then_some(BufferId {
                owner: key,
                producer_call_id: call_id,
                output_index: 1,
                generation: 2,
            }),
            input_token_ids: if kind == CallKind::Forward(ForwardMode::TokenDenoising) {
                vec![4, 9, 4, 1]
            } else {
                Vec::new()
            },
            readout: (kind == CallKind::Forward(ForwardMode::TokenDenoising)).then(readout),
            canvas: None,
            sampling_state: None,
            request_key: key,
            call_id,
            component: "model".into(),
            kind,
            bounds: Bounds {
                max_tokens: 7 + index as u32,
                max_completion_bytes: 4096,
                max_transfer_bytes: 1 << 16,
                ..Bounds::default()
            },
            inputs: vec![tensor_for(
                key,
                CallId::new(2, 0),
                0,
                DType::F16,
                ShapeBound {
                    dims: vec![DimBound::Static(2), DimBound::Device { max: 16 }],
                },
            )],
            outputs: Vec::new(),
            predicate: Some(tensor_for(
                key,
                CallId::new(3, 0),
                0,
                DType::U8,
                ShapeBound::default(),
            )),
            image_index: (kind == CallKind::Media(MediaCall::LatentPreparation))
                .then_some(4 + index as u32),
            latent_restore: None,
            host_output: None,
            latent_draw: None,
            carried_image: None,
        });
        // The admissions ride on the first batch of the pass, as a scheduler
        // places them ahead of the calls that use them.
        let admissions = if index == 0 {
            calls[0].input_token_ids = vec![7, 8, 9, 10];
            vec![ar_params.clone(), image_params.clone()]
        } else {
            Vec::new()
        };
        batches.push(batch_with_calls(42 + index as u64, admissions, calls));
    }
    // The pass's retirement rides on its last batch.
    let last = batches.pop().expect("one batch per call kind");
    batches.push(last.with_commands(vec![BatchCommand::Finish {
        request_key: key_for_request(201),
        retained_buffers: Vec::new(),
    }]));
    batches
}

/// One fixture per `RequestKind`, plus call-id coverage on the submit frame.
fn request_fixtures() -> Vec<WorkerRequest> {
    let mut submit = WorkerRequest::submit(
        comprehensive_batches()
            .into_iter()
            .next()
            .expect("one batch per call kind"),
    );
    submit.set_message_id(Some(91));
    vec![WorkerRequest::info(), submit, WorkerRequest::close()]
}

fn full_caps() -> WorkerInfo {
    WorkerInfo {
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
                    ..WorkerInfo::default().prefix_cache.unwrap().groups[0].clone()
                },
                CacheGroup {
                    kind: CacheGroupKind::SlidingWindow { window: 4096 },
                    block_size: 32,
                    units_per_page: 2,
                    layer_ids: (14..28).collect(),
                    ..WorkerInfo::default().prefix_cache.unwrap().groups[0].clone()
                },
            ],
            ..WorkerInfo::default().prefix_cache.unwrap()
        }),
        max_batch_calls: 64,
        max_batch_tokens: 4096,
        request_slots: 96,
        max_unresolved_calls: 3,
        latent_page_units: 64,
        latent_pages: 17,
        ..WorkerInfo::default()
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

fn full_forward_stats() -> ForwardStats {
    let map = |prefix: &str, base: u64| {
        BTreeMap::from([
            (format!("{prefix}.a"), base),
            (format!("{prefix}.b"), base + 1),
        ])
    };
    ForwardStats {
        mode_counts: map("mode_counts", 1),
        mode_tokens: map("mode_tokens", 3),
        mode_us: map("mode_us", 5),
        component_us: map("component_us", 7),
        stream_us: map("stream_us", 9),
        blocking_calls: map("blocking_calls", 40),
        fence_us: [(
            "decode/step".to_owned(),
            ipc::FenceHistogram {
                buckets: vec![1; ipc::FENCE_BUCKETS_US.len() + 1],
                sum_us: 41,
            },
        )]
        .into(),
        fence_records_lost: 42,
        profile: map("profile", 43),
        attention_launches: 11,
        attention_us: 12,
        attention_backend_counts: map("backend", 13),
        cuda_graph_captures: 15,
        cuda_graph_replays: 16,
        cuda_graph_misses: 17,
        cuda_graph_fallbacks: 18,
        cuda_graph_unpadded_tokens: 19,
        cuda_graph_padded_tokens: 20,
        cuda_graph_runtime_mode_counts: map("runtime_mode", 21),
        text_decode_token_relay_hits: 23,
        text_decode_token_relay_misses: 24,
        text_decode_position_relay_hits: 25,
        text_decode_position_relay_misses: 26,
        flashinfer_decode_plan_calls: 27,
        flashinfer_decode_plan_reuses: 28,
        flashinfer_decode_plan_rows: 29,
        flashinfer_decode_plan_indices: 30,
        flashinfer_decode_graph_plan_calls: 31,
        flashinfer_decode_graph_plan_reuses: 32,
        spec_verify_rows: 33,
        spec_verify_draft_tokens: 34,
        spec_verify_accepted_tokens: 35,
        spec_verify_rejected_tokens: 36,
        spec_verify_committed_tokens: 37,
        spec_verify_path_counts: map("spec_path", 38),
    }
}

fn full_run_result() -> BatchOutput {
    let mut ok_record = CallOutput {
        latent_layouts: Vec::new(),
        token_logprobs: Vec::new(),
        prompt_logprobs: Vec::new(),
        request_key: key_for_request(100),
        call_id: CallId::new(11, 0),

        status: CallStatus::Ok,

        error_code: None,
        error_message: None,
        timing_counters: TimingCounters {
            queued_us: 41,
            device_us: 42,
            copy_us: 43,
            host_us: 44,
        },
        cache_visible_len: 6,
        step_index: 7,
        cache_computed_len: 8,

        committed_tokens: vec![271, 272],
        candidate_logprobs: Vec::new(),
        finish_flags: FinishFlags {
            eos: true,
            length: false,
            stop: false,
        },
        media_output: None,
        cache_output: None,
        draft_token_ids: Vec::new(),
    };
    let mut predicated_record = ok_record.clone();
    predicated_record.request_key = key_for_request(101);
    predicated_record.call_id = CallId::new(11, 1);
    predicated_record.status = CallStatus::Predicated;
    predicated_record.committed_tokens.clear();
    predicated_record.finish_flags = FinishFlags::default();
    let mut error_record = ok_record.clone();
    error_record.request_key = key_for_request(102);
    error_record.call_id = CallId::new(11, 2);
    error_record.status = CallStatus::Error;
    error_record.error_code = Some(ErrorCode::ResourceError);
    error_record.finish_flags = FinishFlags {
        eos: false,
        length: false,
        stop: true,
    };
    ok_record.media_output = Some(MediaOutput {
        handle: ArtifactHandle::PosixShm {
            name: "rendered-image".into(),
        },
        bytes: 256,
    });
    let products = Vec::new();
    batch_output(
        5,
        vec![ok_record, predicated_record, error_record],
        products,
        Some(1234),
        Some(full_forward_stats()),
    )
}

/// One fixture per `ResponseKind`, plus a second info frame that
/// exercises non-default worker information.
fn response_fixtures() -> Vec<WorkerResponse> {
    vec![
        WorkerResponse::Info {
            message_id: None,
            info: WorkerInfo::default(),
        },
        WorkerResponse::Info {
            message_id: Some(17),
            info: full_caps(),
        },
        WorkerResponse::Result {
            message_id: Some(17),
            result: full_run_result(),
        },
        WorkerResponse::Ok {
            message_id: Some(17),
        },
        WorkerResponse::Error {
            message_id: Some(17),
            error: WorkerResponseError {
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
        WorkerResponse::Stalled {
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
    let frame = Frame::encode(&WorkerRequest::info()).unwrap();
    let (field, message) = frame.as_bytes().split_at(Frame::LENGTH_BYTES);

    assert_eq!(Frame::message_len(field.try_into().unwrap()), message.len());
}

#[test]
fn a_frame_decodes_only_as_the_message_it_carries() {
    let request = Frame::encode(&WorkerRequest::info()).unwrap();
    assert!(matches!(
        request.decode::<WorkerResponse>(),
        Err(IpcError::Encoding(_))
    ));

    // A message cut short, as a frame of another build may be, is refused:
    // the frame's length field counts the shortened message, so a channel
    // receives it whole and its decoding fails.
    let mut cut = Frame::encode(&WorkerResponse::ok()).unwrap().into_bytes();
    cut.pop();
    let length = u32::try_from(cut.len() - Frame::LENGTH_BYTES).unwrap();
    cut[..Frame::LENGTH_BYTES].copy_from_slice(&length.to_le_bytes());

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut channel = Channel::connect(&listener.local_addr().unwrap().to_string()).unwrap();
    let (mut peer, _) = listener.accept().unwrap();
    peer.write_all(&cut).unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    let received = loop {
        match channel.try_recv::<WorkerResponse>() {
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
