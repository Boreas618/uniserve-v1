#![cfg(test)]

#[macro_use]
mod common;

use tensor::{CpuStorage, DType, Device, Result, Storage, Tensor};

// A packed dtype has no host element type: its tensor keeps the bytes it was built from.
fn packed_dtype_keeps_its_bytes(device: &Device) -> Result<()> {
    let bytes = [0x12u8, 0x34, 0xab];
    let t = Tensor::from_raw_buffer(&bytes, DType::F4, &[6], device)?;
    assert!(t.device().same_device(device));
    assert_eq!(t.dtype(), DType::F4);
    assert_eq!(t.dims(), [6]);

    let on_cpu = t.to_device(&Device::Cpu)?;
    let (storage, _) = on_cpu.storage_and_layout();
    let Storage::Cpu(CpuStorage::F4(data)) = &*storage else {
        panic!("expected a packed F4 cpu storage")
    };
    assert_eq!(data, &bytes);
    Ok(())
}

test_device!(packed_dtype_keeps_its_bytes);
