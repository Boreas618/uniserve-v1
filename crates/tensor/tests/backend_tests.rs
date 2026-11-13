//! The device-backend interface as a backend outside the crate sees it: how tensors come to be
//! on a backend device, which operands may meet, and how a backend reaches the storage under a
//! tensor.

#[macro_use]
mod common;

use common::{host_device, HostStorage};
use uniserve_tensor::{
    CpuStorage, DType, Device, DeviceLocation, IndexOp, Result, Storage, Tensor,
};

#[test]
fn location_names_the_backend_kind_and_ordinal() {
    assert_eq!(Device::Cpu.location(), DeviceLocation::Cpu);
    assert_eq!(
        host_device(3).location(),
        DeviceLocation::Backend {
            kind: "host",
            ordinal: 3
        }
    );
}

#[test]
fn same_device_is_decided_by_the_backend() {
    // Two values of the host device with one ordinal are the same device to their backend.
    assert!(host_device(0).same_device(&host_device(0)));
    assert!(!host_device(0).same_device(&host_device(1)));

    assert!(Device::Cpu.same_device(&Device::Cpu));
    assert!(!Device::Cpu.same_device(&host_device(0)));
    assert!(!host_device(0).same_device(&Device::Cpu));
}

#[test]
fn to_device_copies_between_the_cpu_and_a_backend() -> Result<()> {
    let backend = host_device(0);

    // A strided view: its layout is kept by both copies.
    let cpu = Tensor::arange(0f32, 12., &Device::Cpu)?
        .reshape((3, 4))?
        .t()?
        .narrow(0, 1, 2)?;
    let expected = [[1f32, 5., 9.], [2., 6., 10.]];
    assert_eq!(cpu.to_vec2::<f32>()?, expected);

    let on_backend = cpu.to_device(&backend)?;
    assert!(on_backend.device().same_device(&backend));
    assert!(matches!(
        &*on_backend.storage_and_layout().0,
        Storage::Backend(_)
    ));
    assert_eq!(on_backend.to_vec2::<f32>()?, expected);

    let on_cpu = on_backend.to_device(&Device::Cpu)?;
    assert!(on_cpu.device().is_cpu());
    assert!(matches!(&*on_cpu.storage_and_layout().0, Storage::Cpu(_)));
    assert_eq!(on_cpu.to_vec2::<f32>()?, expected);
    Ok(())
}

#[test]
fn to_device_on_the_same_device_shares_the_storage() -> Result<()> {
    for device in [Device::Cpu, host_device(0)] {
        // A device value of its own: the backend decides that it is the same device.
        let same = match &device {
            Device::Cpu => Device::Cpu,
            Device::Backend(_) => host_device(0),
        };

        let tensor = Tensor::zeros(3, DType::F32, &device)?;
        let moved = tensor.to_device(&same)?;

        // A write through one tensor is read through the other.
        moved.one_set()?;
        assert_eq!(tensor.to_vec1::<f32>()?, [1., 1., 1.]);
    }
    Ok(())
}

#[test]
fn to_device_between_different_backend_devices_is_an_error() -> Result<()> {
    let tensor = Tensor::zeros(3, DType::F32, &host_device(0))?;
    let err = tensor.to_device(&host_device(1)).unwrap_err();
    assert!(
        err.to_string().starts_with("cannot move a tensor from"),
        "unexpected error: {err}"
    );
    Ok(())
}

/// Operations whose operands are on two devices that may not meet fail with a device mismatch.
fn assert_device_mismatch(lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    let ids = Tensor::new(&[0u32, 1], rhs.device())?;
    let cond = Tensor::ones(lhs.shape(), DType::U8, rhs.device())?;

    let results = [
        ("add", lhs.add(rhs)),
        ("broadcast_mul", lhs.broadcast_mul(rhs)),
        ("maximum", lhs.maximum(rhs)),
        ("eq", lhs.eq(rhs)),
        ("matmul", lhs.matmul(rhs)),
        ("where_cond", cond.where_cond(lhs, lhs)),
        ("index_select", lhs.index_select(&ids, 0)),
        ("cat", Tensor::cat(&[lhs, rhs], 0)),
        ("slice_scatter", lhs.slice_scatter(rhs, 0, 0)),
        (
            "conv1d",
            lhs.unsqueeze(0)?.conv1d(&rhs.unsqueeze(0)?, 0, 1, 1, 1),
        ),
    ];

    for (name, result) in results {
        let err = match result {
            Ok(_) => panic!("{name} accepted operands on different devices"),
            Err(err) => err,
        };
        assert!(
            err.to_string().starts_with("device mismatch in"),
            "unexpected error for {name}: {err}"
        );
    }
    Ok(())
}

#[test]
fn operands_on_different_devices_are_an_error() -> Result<()> {
    let data = &[[1f32, 2.], [3., 4.]];
    let cpu = Tensor::new(data, &Device::Cpu)?;
    let backend0 = Tensor::new(data, &host_device(0))?;
    let backend1 = Tensor::new(data, &host_device(1))?;

    assert_device_mismatch(&cpu, &backend0)?;
    assert_device_mismatch(&backend0, &cpu)?;
    assert_device_mismatch(&backend0, &backend1)?;
    Ok(())
}

#[test]
fn from_storage_builds_a_tensor_over_a_backend_storage() -> Result<()> {
    let Device::Backend(backend) = host_device(0) else {
        unreachable!("the host device is a backend device")
    };

    let storage = backend.storage_from_cpu_storage(&CpuStorage::F32(vec![1., 2., 3., 4.]))?;
    let tensor = Tensor::from_storage(Storage::Backend(storage), (2, 2));

    assert!(tensor.device().same_device(&host_device(0)));
    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.to_vec2::<f32>()?, [[1., 2.], [3., 4.]]);
    assert_eq!(tensor.t()?.sum(0)?.to_vec1::<f32>()?, [3., 7.]);
    Ok(())
}

#[test]
fn from_storage_builds_a_tensor_over_a_cpu_storage() -> Result<()> {
    let storage = Storage::Cpu(CpuStorage::U32(vec![1, 2, 3]));
    let tensor = Tensor::from_storage(storage, 3);

    assert!(tensor.device().is_cpu());
    assert_eq!(tensor.to_vec1::<u32>()?, [1, 2, 3]);
    Ok(())
}

#[test]
fn storage_and_layout_reach_the_backend_storage() -> Result<()> {
    let tensor = Tensor::arange(0f32, 6., &host_device(0))?.reshape((2, 3))?;
    let row = tensor.i(1)?;

    {
        let (storage, layout) = row.storage_and_layout();
        let Storage::Backend(storage) = &*storage else {
            panic!("expected a backend storage")
        };

        // The view shares the storage of the whole tensor and starts at its second row.
        let CpuStorage::F32(data) = HostStorage::from_dyn(storage.as_ref())?.cpu() else {
            panic!("expected F32 elements")
        };
        assert_eq!(data, &[0., 1., 2., 3., 4., 5.]);
        assert_eq!(layout.start_offset(), 3);
        assert_eq!(layout.dims(), [3]);
    }

    {
        let (mut storage, layout) = row.storage_mut_and_layout();
        let Storage::Backend(storage) = &mut *storage else {
            panic!("expected a backend storage")
        };

        let CpuStorage::F32(data) = HostStorage::from_dyn_mut(storage.as_mut())?.cpu_mut() else {
            panic!("expected F32 elements")
        };
        data[layout.start_offset()] = 30.;
    }

    // Every tensor over the storage reads what was written through the guard.
    assert_eq!(tensor.to_vec2::<f32>()?, [[0., 1., 2.], [30., 4., 5.]]);
    Ok(())
}

#[test]
fn display_names_the_backend_device() -> Result<()> {
    let tensor = Tensor::new(&[1u32, 2], &host_device(2))?;
    assert_eq!(format!("{tensor}"), "[1, 2]\nTensor[[2], u32, host:2]");
    Ok(())
}

fn powf(device: &Device) -> Result<()> {
    let tensor = Tensor::new(&[1f32, 2., 3.], device)?;
    assert_eq!(tensor.powf(2.)?.to_vec1::<f32>()?, [1., 4., 9.]);
    Ok(())
}

fn copy_has_its_own_storage(device: &Device) -> Result<()> {
    let tensor = Tensor::new(&[[1f32, 2.], [3., 4.]], device)?.t()?;
    let copy = tensor.copy()?;
    assert!(copy.device().same_device(device));
    assert_eq!(copy.to_vec2::<f32>()?, [[1., 3.], [2., 4.]]);

    copy.zero_set()?;
    assert_eq!(copy.to_vec2::<f32>()?, [[0., 0.], [0., 0.]]);
    assert_eq!(tensor.to_vec2::<f32>()?, [[1., 3.], [2., 4.]]);
    Ok(())
}

fn upsample_nearest1d(device: &Device) -> Result<()> {
    let tensor = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.]]], device)?;
    let upsampled = tensor.upsample_nearest1d(6)?;
    assert_eq!(
        upsampled.to_vec3::<f32>()?,
        [[[1., 1., 2., 2., 3., 3.], [4., 4., 5., 5., 6., 6.]]]
    );
    Ok(())
}

/// Index tensors may be `U8`, `U32` or `I64` for every index operation.
fn index_dtypes(device: &Device) -> Result<()> {
    let tensor = Tensor::new(&[[0f32, 1., 2.], [3., 4., 5.]], device)?;
    let ones = tensor.ones_like()?;

    for dtype in [DType::U8, DType::U32, DType::I64] {
        let ids = Tensor::new(&[2u32, 0], device)?.to_dtype(dtype)?;
        assert_eq!(
            tensor.index_select(&ids, 1)?.to_vec2::<f32>()?,
            [[2., 0.], [5., 3.]],
            "index_select with {dtype:?} indices"
        );

        let ids = Tensor::new(&[[2u32, 0, 1], [1, 1, 0]], device)?.to_dtype(dtype)?;
        assert_eq!(
            tensor.gather(&ids, 1)?.to_vec2::<f32>()?,
            [[2., 0., 1.], [4., 4., 3.]],
            "gather with {dtype:?} indices"
        );
        assert_eq!(
            ones.scatter(&ids, &tensor, 1)?.to_vec2::<f32>()?,
            [[1., 2., 0.], [5., 4., 1.]],
            "scatter with {dtype:?} indices"
        );
        assert_eq!(
            ones.scatter_add(&ids, &tensor, 1)?.to_vec2::<f32>()?,
            [[2., 3., 1.], [6., 8., 1.]],
            "scatter_add with {dtype:?} indices"
        );

        let ids = Tensor::new(&[1u32, 1, 0], device)?.to_dtype(dtype)?;
        assert_eq!(
            ones.index_add(&ids, &tensor, 1)?.to_vec2::<f32>()?,
            [[3., 2., 1.], [6., 8., 1.]],
            "index_add with {dtype:?} indices"
        );
    }
    Ok(())
}

test_device!(powf);
test_device!(copy_has_its_own_storage);
test_device!(upsample_nearest1d);
test_device!(index_dtypes);
