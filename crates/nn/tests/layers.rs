//! What a model crate can rely on when it builds layers: the geometry each layer derives, and the
//! arguments it refuses.
//!
//! A layer validates its weights once, at construction, and reports a wiring error as an error
//! rather than at the first launch: a model crate that loads a checkpoint with a shape the layer
//! cannot compute must fail at load, with the parameter named. The weights here are views of a
//! region that is never read, because construction only inspects extents and strides.

use nn::{
    Activation, AdaptivePool, Arena, CausalPad, Conv3d, ConvGeometry, DType, DevicePtr, Epilogue,
    Forward, GateBias, GatedActivation, GroupNorm, Halves, KvGroup, KvPlanes, KvView, LayerNorm,
    Linear, Modulation, Residual, ResidualStream, RmsNorm, Rotary, Rotate, Rotation, Rounding,
    Scale, Shape, Snake, TensorView, VsaTiles, Weight,
};
use ops::gemm::WeightOrder;

/// A weight of `dims` elements, row-major, over a region that is never read.
fn weight(dims: &[u64]) -> Weight {
    let shape = Shape::new(dims);
    let elements = shape.numel();
    // Every element is four bytes, so a BF16 view of the same extents also fits.
    let backing = nn::Backing::new(DevicePtr::new(0x1_0000), elements * 4);
    Weight {
        backing,
        view: TensorView::new(DevicePtr::new(0x1_0000), DType::F32, shape),
    }
}

#[test]
fn a_projection_derives_its_contraction_and_channels_from_the_storage_order() {
    let nxk = Linear::new(weight(&[8, 64]), None, WeightOrder::NxK).unwrap();
    assert_eq!((nxk.k(), nxk.n()), (64, 8));

    let kxn = Linear::new(weight(&[64, 8]), None, WeightOrder::KxN).unwrap();
    assert_eq!((kxn.k(), kxn.n()), (64, 8));

    assert!(Linear::new(weight(&[64]), None, WeightOrder::KxN).is_err());
    assert!(
        Linear::new(weight(&[64, 8]), Some(weight(&[9])), WeightOrder::KxN).is_err(),
        "a bias of the wrong width is a loading error"
    );
    assert!(Linear::new(weight(&[64, 8]), Some(weight(&[8])), WeightOrder::KxN).is_ok());
}

/// A residual's bias is the update's, added under a channel scale of its width; a row scale is a
/// table with one row per conditioning parameter.
#[test]
fn a_residual_refuses_parameters_its_scale_does_not_read() {
    let channel = |width| Scale::Channel(weight(&[width]));
    assert!(Residual::new(channel(16), Some(weight(&[8]))).is_err());
    assert!(Residual::new(channel(16), Some(weight(&[16]))).is_ok());
    assert!(Residual::new(Scale::Channel(weight(&[4, 16])), None).is_err());
    assert!(Residual::new(Scale::None, Some(weight(&[16]))).is_err());

    assert!(Residual::new(Scale::Row(weight(&[4, 32])), None).is_ok());
    assert!(Residual::new(Scale::Row(weight(&[4, 32])), Some(weight(&[32]))).is_err());
    assert!(Residual::new(Scale::Row(weight(&[32])), None).is_err());
    assert!(Residual::new(Scale::Row(weight(&[4, 0])), None).is_err());
}

/// Each scale writes one operand in place: the unscaled add either input, a channel scale the
/// residual, a row scale the update. Another output, or row indices the scale does not read, is
/// refused before anything launches.
#[test]
fn a_residual_writes_only_the_operand_its_scale_updates() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let rows = |columns| {
        arena
            .tensor(DType::BF16, Shape::new(&[3, columns]))
            .unwrap()
    };
    let (residual, update, other) = (rows(16), rows(16), rows(16));
    let indices = arena.tensor(DType::I32, Shape::new(&[3])).unwrap();

    let plain = Residual::default();
    assert!(plain.forward(&f, residual, update, None, residual).is_ok());
    assert!(plain.forward(&f, residual, update, None, update).is_ok());
    assert!(plain.forward(&f, residual, update, None, other).is_err());
    assert!(
        plain
            .forward(&f, residual, update, Some(indices), residual)
            .is_err()
    );

    let gains = arena.tensor(DType::BF16, Shape::new(&[16])).unwrap();
    let channel = Residual::new(
        Scale::Channel(Weight {
            backing: nn::Backing::new(gains.ptr, 32),
            view: gains,
        }),
        None,
    )
    .unwrap();
    assert!(
        channel
            .forward(&f, residual, update, None, residual)
            .is_ok()
    );
    assert!(channel.forward(&f, residual, update, None, update).is_err());

    let table = arena.tensor(DType::BF16, Shape::new(&[4, 16])).unwrap();
    let gated = Residual::new(
        Scale::Row(Weight {
            backing: nn::Backing::new(table.ptr, 128),
            view: table,
        }),
        None,
    )
    .unwrap();
    assert!(
        gated
            .forward(&f, residual, update, Some(indices), update)
            .is_ok()
    );
    assert!(
        gated
            .forward(&f, residual, update, Some(indices), residual)
            .is_err()
    );
    assert!(gated.forward(&f, residual, update, None, update).is_err());
}

/// A normalization's epilogue parameters are the normalization's width: a scaled residual's scale
/// and bias are vectors of it, a modulation's tables are `[parameters, width]`.
#[test]
fn a_normalization_refuses_epilogue_parameters_of_another_width() {
    let rows = TensorView::new(DevicePtr::new(0x1_0000), DType::I32, Shape::new(&[3]));
    let scaled = |scale: &[u64], bias: Option<&[u64]>| Epilogue::ScaledResidual {
        scale: weight(scale),
        bias: bias.map(weight),
    };
    assert!(RmsNorm::new(weight(&[16, 8]), 1e-6, Epilogue::None).is_err());
    assert!(RmsNorm::new(weight(&[16]), 1e-6, scaled(&[8], None)).is_err());
    assert!(RmsNorm::new(weight(&[16]), 1e-6, scaled(&[16], Some(&[8]))).is_err());
    assert!(RmsNorm::new(weight(&[16]), 1e-6, scaled(&[16], Some(&[16]))).is_ok());

    let modulated = |shift: &[u64], gate: Option<&[u64]>| {
        Epilogue::Modulated(Modulation {
            shift: weight(shift),
            scale: weight(&[4, 64]),
            gate: gate.map(weight),
            rows,
            retained: None,
        })
    };
    assert!(RmsNorm::new(weight(&[64]), 1e-6, modulated(&[4, 64], None)).is_ok());
    assert!(RmsNorm::new(weight(&[64]), 1e-6, modulated(&[4, 64], Some(&[4, 64]))).is_ok());
    assert!(
        RmsNorm::new(weight(&[64]), 1e-6, modulated(&[64], None)).is_err(),
        "a shift table is [parameters, width], not one row"
    );
    assert!(
        RmsNorm::new(weight(&[64]), 1e-6, modulated(&[4, 32], None)).is_err(),
        "a table of another width is not this layer's"
    );

    let (gain, bias) = (weight(&[16]), Some(weight(&[16])));
    assert!(LayerNorm::new(gain, bias, 1e-6, scaled(&[16], None)).is_ok());
    assert!(
        LayerNorm::new(gain, None, 1e-6, scaled(&[16], None)).is_err(),
        "a scaled residual layer normalization adds its bias"
    );
    assert!(LayerNorm::new(gain, bias, 1e-6, modulated(&[4, 16], None)).is_err());
}

/// The residual stream is a call operand: a plain normalization adds into it when one is passed, a
/// scaled residual requires one updated in place, and a modulation takes one exactly when it gates
/// an update, writing the sum over that update.
#[test]
fn a_normalization_reads_the_residual_stream_its_epilogue_folds() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let view = |dtype, dims: &[u64]| arena.tensor(dtype, Shape::new(dims)).unwrap();
    let held = |view: TensorView| Weight {
        backing: nn::Backing::new(view.ptr, view.numel() * 2),
        view,
    };
    let (x, out, stream, other) = (
        view(DType::BF16, &[3, 16]),
        view(DType::BF16, &[3, 16]),
        view(DType::BF16, &[3, 16]),
        view(DType::BF16, &[3, 16]),
    );
    let gain = held(view(DType::BF16, &[16]));

    let plain = RmsNorm::new(gain, 1e-6, Epilogue::None).unwrap();
    assert!(plain.forward(&f, x, None, out).is_ok());
    assert!(
        plain
            .forward(&f, x, Some(ResidualStream::in_place(stream)), out)
            .is_ok()
    );
    let moved = ResidualStream {
        input: other,
        output: stream,
    };
    assert!(plain.forward(&f, x, Some(moved), out).is_ok());

    let scale = held(view(DType::BF16, &[16]));
    let epilogue = Epilogue::ScaledResidual { scale, bias: None };
    let scaled = RmsNorm::new(gain, 1e-6, epilogue).unwrap();
    assert!(
        scaled
            .forward(&f, x, Some(ResidualStream::in_place(stream)), out)
            .is_ok()
    );
    assert!(scaled.forward(&f, x, None, out).is_err());
    assert!(scaled.forward(&f, x, Some(moved), out).is_err());

    let table = held(view(DType::BF16, &[4, 16]));
    let rows = view(DType::I32, &[3]);
    let modulation = |gate| Modulation {
        shift: table,
        scale: table,
        gate,
        rows,
        retained: None,
    };
    let ungated = RmsNorm::new(gain, 1e-6, Epilogue::Modulated(modulation(None))).unwrap();
    assert!(ungated.forward(&f, x, None, out).is_ok());
    assert!(
        ungated
            .forward(&f, x, Some(ResidualStream::in_place(x)), out)
            .is_err()
    );
    let gated = RmsNorm::new(gain, 1e-6, Epilogue::Modulated(modulation(Some(table)))).unwrap();
    let into_update = ResidualStream {
        input: stream,
        output: x,
    };
    assert!(gated.forward(&f, x, Some(into_update), out).is_ok());
    assert!(gated.forward(&f, x, None, out).is_err());
    assert!(
        gated
            .forward(&f, x, Some(ResidualStream::in_place(stream)), out)
            .is_err()
    );
}

#[test]
fn a_gated_product_reads_halves_and_biases_of_its_output_width() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let view = |dims: &[u64]| arena.tensor(DType::BF16, Shape::new(dims)).unwrap();
    let (gate, value, out) = (view(&[3, 64]), view(&[3, 64]), view(&[3, 64]));
    let bias = |width| {
        let view = view(&[width]);
        Weight {
            backing: nn::Backing::new(view.ptr, width * 2),
            view,
        }
    };
    let halves = Halves::Split { gate, value };
    let swiglu = |value: Option<Weight>| {
        GatedActivation::new(
            Activation::Silu,
            Rounding::ValueProduct,
            GateBias::Split { value, gate: None },
        )
        .unwrap()
    };
    assert!(swiglu(None).forward(&f, halves, out).is_ok());
    assert!(swiglu(Some(bias(64))).forward(&f, halves, out).is_ok());
    assert!(swiglu(Some(bias(32))).forward(&f, halves, out).is_err());
    assert!(
        GatedActivation::new(
            Activation::Silu,
            Rounding::ValueProduct,
            GateBias::Split {
                value: None,
                gate: Some(weight(&[64, 2]))
            },
        )
        .is_err()
    );

    // A packed projection is twice the output's width, and its bias is the packed one.
    let packed = view(&[3, 128]);
    let fused = GatedActivation::new(Activation::Silu, Rounding::Fused, GateBias::None).unwrap();
    assert!(fused.forward(&f, Halves::ValueFirst(packed), out).is_ok());
    assert!(
        fused
            .forward(&f, Halves::ValueFirst(view(&[3, 96])), out)
            .is_err()
    );
    assert!(
        swiglu(Some(bias(64)))
            .forward(&f, Halves::ValueFirst(packed), out)
            .is_err()
    );
}

#[test]
fn a_rotary_needs_a_pair_per_head() {
    let frequencies = weight(&[32]);
    assert!(Rotary::new(frequencies, 64, 8, 8, 1e-6, Rotation::Half, 0.125).is_ok());
    assert!(
        Rotary::new(frequencies, 63, 8, 8, 1e-6, Rotation::Half, 0.125).is_err(),
        "an odd head dimension has no rotation pair"
    );
}

#[test]
fn a_group_normalization_states_its_groups_and_its_affine_pair() {
    let pad = CausalPad {
        top: 1,
        left: 1,
        front: 2,
        reflect: false,
    };
    assert!(
        GroupNorm::new(32, 1e-6, Some(weight(&[64])), None, pad).is_err(),
        "a weight without a bias is not an affine pair"
    );
    assert!(GroupNorm::new(32, 1e-6, Some(weight(&[64])), Some(weight(&[64])), pad).is_ok());
    assert!(GroupNorm::new(32, 1e-6, None, None, pad).is_ok());
    assert!(GroupNorm::new(0, 1e-6, None, None, pad).is_err());
}

/// The frame a normalization writes holds the source's channels and is at least its extent on
/// every padded axis.
#[test]
fn a_group_normalization_writes_a_frame_no_smaller_than_its_source() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let view = |dims: &[u64]| arena.tensor(DType::F32, Shape::new(dims)).unwrap();
    let pad = CausalPad {
        top: 1,
        left: 1,
        front: 2,
        reflect: true,
    };
    let norm = GroupNorm::new(4, 1e-6, None, None, pad).unwrap();
    let values = view(&[1, 8, 3, 6, 6]);
    let (mean, rstd) = (view(&[3, 4]), view(&[3, 4]));
    assert!(
        norm.forward(&f, values, mean, rstd, view(&[1, 8, 5, 8, 8]))
            .is_ok()
    );
    assert!(
        norm.forward(&f, values, mean, rstd, view(&[1, 4, 5, 8, 8]))
            .is_err()
    );
    assert!(
        norm.forward(&f, values, mean, rstd, view(&[1, 8, 5, 4, 8]))
            .is_err()
    );
    assert!(
        norm.forward(&f, values, mean, rstd, view(&[8, 5, 8, 8]))
            .is_err()
    );
}

#[test]
fn the_tile_selection_needs_a_whole_tile_and_a_pair_of_rotated_features() {
    // The programmes handle BLOCK_ROWS rows at a time, so a tile must be a whole number of them.
    assert!(VsaTiles::new(64, 8, 128, 1e-6).is_ok());
    assert!(VsaTiles::new(40, 8, 128, 1e-6).is_err());
    assert!(VsaTiles::new(64, 0, 128, 1e-6).is_err());

    let layer = VsaTiles::new(64, 8, 128, 1e-6).unwrap();
    assert_eq!(layer.tile_rows(), 64);
    assert_eq!(layer.heads(), 8);
    assert_eq!(layer.width(), 128);
    assert!(
        VsaTiles::new(64, 8, 128, 1e-6)
            .unwrap()
            .with_rotation(128)
            .is_ok()
    );
    assert!(
        VsaTiles::new(64, 8, 128, 1e-6)
            .unwrap()
            .with_rotation(127)
            .is_err(),
        "an odd rotation has no pair"
    );
    assert!(
        VsaTiles::new(64, 8, 128, 1e-6)
            .unwrap()
            .with_rotation(256)
            .is_err(),
        "the rotation cannot be wider than the features"
    );
}

/// The page scales belong to the layer, not the group: two layers of one FP8 group keep their own
/// encodings, and the cache view hands each layer's out with it.
#[test]
fn a_group_carries_each_layers_own_page_scales() {
    let pages = Shape::new(&[16, 1, 1, 1]);
    let planes_shape = Shape::new(&[16, 64, 8, 128]);
    let fp8 = |address: u64| TensorView::new(DevicePtr::new(address), DType::F8E4M3, planes_shape);
    let flags = |address: u64| TensorView::new(DevicePtr::new(address), DType::I32, pages);
    let scale = |address: u64| TensorView::new(DevicePtr::new(address), DType::F32, pages);
    let planes = |base: u64| KvPlanes {
        key: fp8(base),
        value: fp8(base + 0x100),
        key_initialized: flags(base + 0x200),
        value_initialized: flags(base + 0x300),
        key_scale: Some(scale(base + 0x400)),
        value_scale: Some(scale(base + 0x500)),
    };
    let layers = [planes(0x1_0000), planes(0x2_0000)];
    let table = TensorView::new(DevicePtr::new(0x3_0000), DType::I32, Shape::new(&[4, 8]));
    let groups = [KvGroup {
        block_tables: table,
        page_size: 64,
        dtype: DType::F8E4M3,
        layers: &layers,
    }];
    let view = KvView { groups: &groups };

    let first = view.planes(0, 0).unwrap();
    let second = view.planes(0, 1).unwrap();
    assert_eq!(first.key_scale.unwrap().ptr, DevicePtr::new(0x1_0400));
    assert_eq!(second.key_scale.unwrap().ptr, DevicePtr::new(0x2_0400));
    assert_eq!(first.value_scale.unwrap().ptr, DevicePtr::new(0x1_0500));
    assert_eq!(second.value_scale.unwrap().ptr, DevicePtr::new(0x2_0500));
}

/// The audio encoder's activation and head pool are built from the checkpoint's channel counts, so
/// a mismatched checkpoint fails where it is loaded rather than at the first launch.
#[test]
fn the_audio_layers_hold_their_channels() {
    assert!(Snake::new(0).is_err());
    assert!(Snake::new(8).is_ok());

    let pool = AdaptivePool::new(8, 256, 32).unwrap();
    assert_eq!(pool.output(), 32);
    // A pool of nothing, or into nothing, has no geometry to compile.
    assert!(AdaptivePool::new(0, 256, 32).is_err());
    assert!(AdaptivePool::new(8, 0, 32).is_err());
    assert!(AdaptivePool::new(8, 256, 0).is_err());
    // Wider than the descriptor's fields hold.
    assert!(AdaptivePool::new(u64::from(u32::MAX) + 1, 256, 32).is_err());
}

/// The vision tower's rotation is built from its head geometry alone: the tables are the caller's
/// and the whole head turns, so a narrower cut or an odd head has no program.
#[test]
fn a_vision_rotation_holds_a_head_and_its_heads() {
    let rotate = Rotate::new(64, 12, Rotation::Half).unwrap();
    assert_eq!((rotate.head_dim(), rotate.heads()), (64, 12));
    assert!(Rotate::new(63, 12, Rotation::Half).is_err());
    assert!(Rotate::new(0, 12, Rotation::Half).is_err());
    assert!(Rotate::new(64, 0, Rotation::Half).is_err());
}

/// A layer's planes are the ones its own index names, and the refusal for an index the cache has not
/// got names the count.
///
/// This is the seam the paged attention reads through: `Attention::forward` writes
/// `cache.planes(group, layer)` and `Attention::attend` reads the same. A reader that took the
/// group's first layer instead would answer another layer's keys and values — the defect that made
/// every layer above the first attend over layer 0's cache — and no shape or dtype check can see it,
/// because both answers are planes of the same geometry. A cache of one layer cannot tell the two
/// apart, which is why this test builds two.
#[test]
fn a_layer_reads_the_planes_its_own_index_names() {
    let plane = |marker: u64| {
        let page = |offset: u64| {
            TensorView::new(
                DevicePtr::new(marker + offset),
                DType::BF16,
                Shape::new(&[2, 4, 2, 8]),
            )
        };
        let flag = |offset: u64| {
            TensorView::new(
                DevicePtr::new(marker + offset),
                DType::I32,
                Shape::new(&[2, 1, 1, 1]),
            )
        };
        KvPlanes {
            key: page(0x0),
            value: page(0x100),
            key_initialized: flag(0x200),
            value_initialized: flag(0x300),
            key_scale: None,
            value_scale: None,
        }
    };
    let first = plane(0x10_0000);
    let second = plane(0x20_0000);
    let layers = [first, second];
    let table = TensorView::new(DevicePtr::new(0x30_0000), DType::I32, Shape::new(&[1, 4]));
    let group = KvGroup {
        block_tables: table,
        page_size: 4,
        dtype: DType::BF16,
        layers: &layers,
    };
    let groups = [group];
    let cache = KvView { groups: &groups };

    // Every field of the second layer's planes is the second layer's, not the first's: a reader that
    // took one field from the wrong plane would read another layer's keys, values or flags.
    let read = cache.planes(0, 1).expect("the second layer");
    assert_eq!(read.key.ptr.address(), second.key.ptr.address());
    assert_eq!(read.value.ptr.address(), second.value.ptr.address());
    assert_eq!(
        read.key_initialized.ptr.address(),
        second.key_initialized.ptr.address()
    );
    assert_eq!(
        read.value_initialized.ptr.address(),
        second.value_initialized.ptr.address()
    );
    assert_ne!(read.key.ptr.address(), first.key.ptr.address());
    assert_eq!(
        cache
            .planes(0, 0)
            .expect("the first layer")
            .key
            .ptr
            .address(),
        first.key.ptr.address()
    );

    // A layer the group has not got, and a group the cache has not got, are refusals that name what
    // the cache holds rather than addressing whatever follows the last plane.
    let past = cache
        .planes(0, 2)
        .err()
        .expect("a third layer in a two-layer group");
    assert!(format!("{past}").contains("holds 2"), "{past}");
    let group = cache
        .planes(1, 0)
        .err()
        .expect("a second group in a one-group cache");
    assert!(format!("{group}").contains("the cache has 1"), "{group}");
}

/// The programs a recording forward reported, as `(operator, key)` pairs.
struct Launches(Vec<(String, String)>);

impl ops::capture::Sink for Launches {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator.to_owned(), key.to_owned()));
    }
}

/// The iteration order is a call's, not a module's: a convolution built channels-last launches the
/// same program as one built channels-first over the same geometry, reading the order from its
/// call. An order in the key would compile the same convolution twice.
#[test]
fn a_channels_last_convolution_launches_the_channels_first_program() {
    let spatial = ConvGeometry::new(3, [2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1).unwrap();
    let spatial_weight = weight(&[8, 4, 2, 2, 2]);
    let input = Shape::new(&[1, 4, 3, 3, 3]);
    let output = Shape::new(&[1, 8, 2, 2, 2]);
    // `NDHWC` storage viewed as the logical `[n, c, d, h, w]` shape: channel fastest.
    let ndhwc = |shape: Shape| {
        let [c, d, h, w] = [1, 2, 3, 4].map(|axis| shape.dim(axis));
        ops::Strides::new(&[c * d * h * w, 1, h * w * c, w * c, c])
    };

    let launched = |channels_last: bool| {
        let arena = Arena::counting();
        let f = Forward::recording(&arena);
        let view = |shape: Shape| {
            let reserved = arena.tensor(DType::F32, shape).unwrap();
            if channels_last {
                TensorView::with_strides(reserved.ptr, reserved.dtype, shape, ndhwc(shape))
            } else {
                reserved
            }
        };
        let (x, out) = (view(input), view(output));
        let conv = Conv3d::new(spatial_weight, None, spatial, false, channels_last).unwrap();
        let mut launches = Launches(Vec::new());
        ops::capture::set_sink(Some(&mut launches));
        let walked = conv.forward(&f, x, out);
        ops::capture::set_sink(None);
        walked.unwrap();
        launches.0
    };

    let channels_first = launched(false);
    assert_eq!(channels_first.len(), 1, "{channels_first:?}");
    assert_eq!(channels_first[0].0, "conv3d");
    assert_eq!(launched(true), channels_first, "the order keys no module");
}
