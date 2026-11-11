#[macro_use]
mod common;

use uniserve_tensor::{CpuStorage, DType, Device, Result, Storage, Tensor};

struct TmpFile(std::path::PathBuf);

impl TmpFile {
    fn create(base: &str) -> TmpFile {
        let filename = std::env::temp_dir().join(format!(
            "uniserve-tensor-{}-{}-{:?}",
            base,
            std::process::id(),
            std::thread::current().id(),
        ));
        TmpFile(filename)
    }
}

impl std::convert::AsRef<std::path::Path> for TmpFile {
    fn as_ref(&self) -> &std::path::Path {
        self.0.as_path()
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap()
    }
}

fn safetensors(device: &Device) -> Result<()> {
    use uniserve_tensor::safetensors::Load;

    let tmp_file = TmpFile::create("st");
    let t = Tensor::arange(0f32, 24f32, &Device::Cpu)?;
    t.save_safetensors("t", &tmp_file)?;
    let t = t.to_device(device)?;

    // Load from file.
    let st = uniserve_tensor::safetensors::load(&tmp_file, device)?;
    let t2 = st.get("t").unwrap();
    assert!(t2.device().same_device(device));
    let diff = (&t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0f32);

    // Load from bytes.
    let bytes = std::fs::read(tmp_file)?;
    let st = uniserve_tensor::safetensors::SliceSafetensors::new(&bytes)?;
    let t2 = st.get("t").unwrap().load(device)?;
    assert!(t2.device().same_device(device));
    let diff = (&t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0f32);
    Ok(())
}

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

test_device!(safetensors);
test_device!(packed_dtype_keeps_its_bytes);
