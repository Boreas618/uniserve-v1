//! What a model crate can rely on when it builds layers: the geometry each layer derives, and the
//! arguments it refuses.
//!
//! A layer validates its weights once, at construction, and reports a wiring error as an error
//! rather than at the first launch: a model crate that loads a checkpoint with a shape the layer
//! cannot compute must fail at load, with the parameter named. The weights here are views of a
//! region that is never read, because construction only inspects extents and strides.

use nn::{
    AdaptivePool, DType, DevicePtr, FrameNormPad, GatedResidual, KvGroup, KvPlanes, KvView, Linear,
    ModulatedRmsNorm, Patch, PatchMode, Rotary, Rotate, Rotation, ScaledResidual,
    ScaledResidualRmsNorm, Shape, Snake, SwiGlu, TensorView, VsaTiles, Weight, WeightedRmsNorm,
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

#[test]
fn the_residual_layers_refuse_parameters_of_different_channel_counts() {
    assert!(WeightedRmsNorm::new(weight(&[16, 8]), 1e-6).is_err());
    assert!(WeightedRmsNorm::new(weight(&[8]), 1e-6).is_ok());

    assert!(
        ScaledResidualRmsNorm::new(weight(&[8]), None, weight(&[16]), 1e-6).is_err(),
        "the residual's scale and the normalization's weight are the same width"
    );
    assert!(
        ScaledResidualRmsNorm::new(weight(&[16]), Some(weight(&[8])), weight(&[16]), 1e-6).is_err(),
        "the update's bias is the residual's width"
    );
    assert!(
        ScaledResidualRmsNorm::new(weight(&[16]), Some(weight(&[16])), weight(&[16]), 1e-6).is_ok()
    );

    assert!(ScaledResidual::new(weight(&[16]), Some(weight(&[8]))).is_err());
    assert!(ScaledResidual::new(weight(&[16]), Some(weight(&[16]))).is_ok());
}

#[test]
fn a_modulation_needs_two_dimensional_parameter_tables_of_its_width() {
    let learned = weight(&[64]);
    let table = weight(&[4, 64]);
    let modulation = ModulatedRmsNorm::new(learned, table, weight(&[4, 64]), None, 1e-6).unwrap();
    assert_eq!(modulation.width(), 64);
    assert!(!modulation.has_update());

    let gated = ModulatedRmsNorm::new(
        weight(&[64]),
        weight(&[4, 64]),
        weight(&[4, 64]),
        Some(weight(&[4, 64])),
        1e-6,
    )
    .unwrap();
    assert!(gated.has_update());

    assert!(
        ModulatedRmsNorm::new(weight(&[64]), weight(&[64]), weight(&[4, 64]), None, 1e-6).is_err(),
        "a shift table is [parameters, width], not one row"
    );
    assert!(
        ModulatedRmsNorm::new(
            weight(&[64]),
            weight(&[4, 32]),
            weight(&[4, 64]),
            None,
            1e-6
        )
        .is_err(),
        "a table of another width is not this layer's"
    );
}

#[test]
fn the_fused_activation_needs_a_width_and_biases_that_match_it() {
    assert!(SwiGlu::new(0).is_err());
    let layer = SwiGlu::new(64).unwrap();
    assert_eq!(layer.width(), 64);

    assert!(layer.with_biases(Some(weight(&[64])), None).is_ok());
    assert!(
        SwiGlu::new(64)
            .unwrap()
            .with_biases(Some(weight(&[32])), None)
            .is_err()
    );
    assert!(
        SwiGlu::new(64)
            .unwrap()
            .with_biases(None, Some(weight(&[64, 2])))
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
fn the_frame_normalization_states_its_layout_and_its_affine_pair() {
    assert!(
        FrameNormPad::new(32, Some(weight(&[64])), None, false).is_err(),
        "a weight without a bias is not an affine pair"
    );
    assert!(FrameNormPad::new(32, Some(weight(&[64])), Some(weight(&[64])), false).is_ok());
    assert!(FrameNormPad::new(32, None, None, true).is_ok());
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

/// The gate table of a diffusion block is one row per conditioning parameter, and a table of no
/// rows or a vector is a loading error rather than a launch-time surprise.
#[test]
fn a_gated_residual_holds_the_gate_table_it_gates_with() {
    assert!(GatedResidual::new(weight(&[4, 32])).is_ok());
    let gate = GatedResidual::new(weight(&[4, 32])).unwrap();
    assert_eq!(gate.width(), 32);
    assert!(GatedResidual::new(weight(&[32])).is_err());
    assert!(GatedResidual::new(weight(&[4, 0])).is_err());
}

/// The patch mapping is built from a layout's latent geometry: channels and the patch grid are
/// fixed by the layout, frames are the call's launch value, and an odd extent has no whole patches.
#[test]
fn a_patch_mapping_holds_a_layouts_latent_geometry() {
    let patch = Patch::new(24, 32, 32).unwrap();
    assert_eq!(patch.channels(), 24);
    assert_eq!(PatchMode::Patchify.to_string(), "patchify");
    assert_eq!(PatchMode::Unpatchify.to_string(), "unpatchify");
    assert!(Patch::new(0, 32, 32).is_err());
    assert!(Patch::new(24, 0, 32).is_err());
    assert!(
        Patch::new(24, 33, 32).is_err(),
        "an odd height has no whole patch"
    );
    assert!(Patch::new(24, 32, 31).is_err());
    assert!(Patch::new(u64::from(u32::MAX) + 1, 32, 32).is_err());
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
