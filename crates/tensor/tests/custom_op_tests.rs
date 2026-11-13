#[macro_use]
mod common;

use common::HostStorage;
use uniserve_tensor::backend::BackendStorage;
use uniserve_tensor::cpu_backend;
use uniserve_tensor::test_utils::to_vec1_round;
use uniserve_tensor::{
    AccessPattern, CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Device, InplaceOp1,
    InplaceOp2, InplaceOp3, InplaceOpN, Layout, Result, Shape, Src, Tensor,
};

fn fwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        v
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        (v.exp() - T::one()) * alpha
    }
}

struct Elu {
    alpha: f64,
}

impl CustomOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let storage = uniserve_tensor::map_dtype!(
            "elu",
            s,
            |s| cpu_backend::unary_map(s, l, |v| fwd(v, self.alpha)),
            (F8E4M3, BF16, F16, F32, F64)
        );
        Ok((storage, l.shape().clone()))
    }

    fn backend_fwd(
        &self,
        s: &dyn BackendStorage,
        l: &Layout,
    ) -> Result<(Box<dyn BackendStorage>, Shape)> {
        let s = HostStorage::from_dyn(s)?;
        let (storage, shape) = CustomOp1::cpu_fwd(self, s.cpu(), l)?;
        Ok((s.on_same_device(storage), shape))
    }
}

fn custom_op1(device: &Device) -> Result<()> {
    let t = Tensor::arange(0u32, 12u32, device)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    let elu_t = t.apply_op1(&Elu { alpha: 1. })?;
    assert!(elu_t.device().same_device(device));
    assert_eq!(
        to_vec1_round(&elu_t, 4)?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}

/// The F32 elements a contiguous layout views in a storage.
fn f32_elements<'a>(storage: &'a CpuStorage, layout: &Layout) -> Result<&'a [f32]> {
    let CpuStorage::F32(data) = storage else {
        uniserve_tensor::bail!("expected F32, got {:?}", storage.dtype())
    };
    match layout.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => uniserve_tensor::bail!("expected a contiguous layout"),
    }
}

/// `x * y`, element-wise, as a binary op; `x * y + z` as a ternary op.
struct MulAdd;

impl CustomOp2 for MulAdd {
    fn name(&self) -> &'static str {
        "mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, y) = (f32_elements(s1, l1)?, f32_elements(s2, l2)?);
        let product = x.iter().zip(y).map(|(x, y)| x * y).collect();
        Ok((CpuStorage::F32(product), l1.shape().clone()))
    }

    fn backend_fwd(
        &self,
        s1: &dyn BackendStorage,
        l1: &Layout,
        s2: &dyn BackendStorage,
        l2: &Layout,
    ) -> Result<(Box<dyn BackendStorage>, Shape)> {
        let (s1, s2) = (HostStorage::from_dyn(s1)?, HostStorage::from_dyn(s2)?);
        let (storage, shape) = CustomOp2::cpu_fwd(self, s1.cpu(), l1, s2.cpu(), l2)?;
        Ok((s1.on_same_device(storage), shape))
    }
}

impl CustomOp3 for MulAdd {
    fn name(&self) -> &'static str {
        "mul-add"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (x, y, z) = (
            f32_elements(s1, l1)?,
            f32_elements(s2, l2)?,
            f32_elements(s3, l3)?,
        );
        let sum = x.iter().zip(y).zip(z).map(|((x, y), z)| x * y + z);
        Ok((CpuStorage::F32(sum.collect()), l1.shape().clone()))
    }

    fn backend_fwd(
        &self,
        s1: &dyn BackendStorage,
        l1: &Layout,
        s2: &dyn BackendStorage,
        l2: &Layout,
        s3: &dyn BackendStorage,
        l3: &Layout,
    ) -> Result<(Box<dyn BackendStorage>, Shape)> {
        let (s1, s2, s3) = (
            HostStorage::from_dyn(s1)?,
            HostStorage::from_dyn(s2)?,
            HostStorage::from_dyn(s3)?,
        );
        let (storage, shape) = CustomOp3::cpu_fwd(self, s1.cpu(), l1, s2.cpu(), l2, s3.cpu(), l3)?;
        Ok((s1.on_same_device(storage), shape))
    }
}

fn custom_op2_and_op3(device: &Device) -> Result<()> {
    let x = Tensor::new(&[1f32, 2., 3.], device)?;
    let y = Tensor::new(&[4f32, 5., 6.], device)?;
    let z = Tensor::new(&[0.5f32, 0.25, 0.125], device)?;

    let product = x.apply_op2(&y, &MulAdd)?;
    assert!(product.device().same_device(device));
    assert_eq!(product.to_vec1::<f32>()?, [4., 10., 18.]);

    let sum = x.apply_op3(&y, &z, &MulAdd)?;
    assert!(sum.device().same_device(device));
    assert_eq!(sum.to_vec1::<f32>()?, [4.5, 10.25, 18.125]);
    Ok(())
}

/// An op with a CPU forward only.
struct Identity;

impl CustomOp1 for Identity {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        Ok((s.clone(), l.shape().clone()))
    }
}

#[test]
fn custom_op_without_backend_forward_is_an_error_on_a_backend_device() -> Result<()> {
    let cpu = Tensor::new(&[1f32, 2.], &Device::Cpu)?;
    assert_eq!(cpu.apply_op1(&Identity)?.to_vec1::<f32>()?, [1., 2.]);

    let backend = cpu.to_device(&common::host_device(0))?;
    let err = backend.apply_op1(&Identity).unwrap_err();
    assert!(
        err.to_string()
            .starts_with("no backend implementation for identity"),
        "unexpected error: {err}"
    );
    Ok(())
}

fn elu_inplace(storage: &mut CpuStorage, alpha: f64) -> Result<()> {
    match storage {
        CpuStorage::F8E4M3(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
        CpuStorage::BF16(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
        CpuStorage::F16(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
        CpuStorage::F32(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
        CpuStorage::F64(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
        _ => uniserve_tensor::bail!("unsupported dtype for inplace elu"),
    }
    Ok(())
}

impl InplaceOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        elu_inplace(s, self.alpha)
    }

    fn backend_fwd(&self, s: &mut dyn BackendStorage, _l: &Layout) -> Result<()> {
        elu_inplace(HostStorage::from_dyn_mut(s)?.cpu_mut(), self.alpha)
    }
}

fn inplace_op1(device: &Device) -> Result<()> {
    let t = Tensor::arange(0u32, 12u32, device)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    t.inplace_op1(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round(&t, 4)?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}

struct InplaceAdd;

fn add_inplace(s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<()> {
    let elems = l1.shape().elem_count();
    let start1 = l1.start_offset();
    let start2 = l2.start_offset();
    let (CpuStorage::F32(d1), CpuStorage::F32(d2)) = (s1, s2) else {
        uniserve_tensor::bail!("unsupported dtype for inplace-add")
    };
    let rhs: Vec<f32> = d2[start2..start2 + elems].to_vec();
    for (v, r) in d1[start1..start1 + elems].iter_mut().zip(rhs) {
        *v += r;
    }
    Ok(())
}

impl InplaceOp2 for InplaceAdd {
    fn name(&self) -> &'static str {
        "inplace-add"
    }

    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<()> {
        add_inplace(s1, l1, s2, l2)
    }

    fn backend_fwd(
        &self,
        s1: &mut dyn BackendStorage,
        l1: &Layout,
        s2: &dyn BackendStorage,
        l2: &Layout,
    ) -> Result<()> {
        let s1 = HostStorage::from_dyn_mut(s1)?.cpu_mut();
        add_inplace(s1, l1, HostStorage::from_dyn(s2)?.cpu(), l2)
    }
}

/// `x += y + z`.
impl InplaceOp3 for InplaceAdd {
    fn name(&self) -> &'static str {
        "inplace-add3"
    }

    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<()> {
        add_inplace(s1, l1, s2, l2)?;
        add_inplace(s1, l1, s3, l3)
    }

    fn backend_fwd(
        &self,
        s1: &mut dyn BackendStorage,
        l1: &Layout,
        s2: &dyn BackendStorage,
        l2: &Layout,
        s3: &dyn BackendStorage,
        l3: &Layout,
    ) -> Result<()> {
        let s1 = HostStorage::from_dyn_mut(s1)?.cpu_mut();
        add_inplace(s1, l1, HostStorage::from_dyn(s2)?.cpu(), l2)?;
        add_inplace(s1, l1, HostStorage::from_dyn(s3)?.cpu(), l3)
    }
}

fn inplace_op2_and_op3(device: &Device) -> Result<()> {
    let a = Tensor::from_vec(vec![1f32, 2.], 2, device)?;
    let b = Tensor::from_vec(vec![10f32, 20.], 2, device)?;
    let c = Tensor::from_vec(vec![100f32, 200.], 2, device)?;

    a.inplace_op2(&b, &InplaceAdd)?;
    assert_eq!(a.to_vec1::<f32>()?, [11., 22.]);

    a.inplace_op3(&b, &c, &InplaceAdd)?;
    assert_eq!(a.to_vec1::<f32>()?, [121., 242.]);
    Ok(())
}

fn inplace_op_shared_storage_errors_instead_of_deadlocking(device: &Device) -> Result<()> {
    let t = Tensor::from_vec(vec![1f32, 2., 3., 4.], 4, device)?;
    let dst = t.narrow(0, 0, 2)?;
    let src = t.narrow(0, 2, 2)?;
    let res = dst.inplace_op2(&src, &InplaceAdd);
    assert!(res.is_err(), "shared-storage inplace_op2 must error");
    let res = dst.inplace_op3(&src, &src, &InplaceAdd);
    let _ = res.expect_err("shared-storage inplace_op3 must error");
    assert_eq!(t.to_vec1::<f32>()?, [1., 2., 3., 4.]);
    Ok(())
}

/// `x += y` that reads element `i` of `y` only to write element `i` of `x`, so `y` may be `x`
/// itself.
struct ElementwiseAdd;

/// `x += y` where `y` views elements of the storage of `x`.
fn add_inplace_aliased(storage: &mut CpuStorage, dst_l: &Layout, src_l: &Layout) -> Result<()> {
    let CpuStorage::F32(data) = storage else {
        uniserve_tensor::bail!("unsupported dtype for elementwise-add")
    };
    let elems = dst_l.shape().elem_count();
    let src: Vec<f32> = data[src_l.start_offset()..][..elems].to_vec();
    for (v, r) in data[dst_l.start_offset()..][..elems].iter_mut().zip(src) {
        *v += r;
    }
    Ok(())
}

impl InplaceOpN<1> for ElementwiseAdd {
    fn name(&self) -> &'static str {
        "elementwise-add"
    }

    fn src_access_pattern(&self) -> Option<AccessPattern> {
        Some(AccessPattern::Elementwise)
    }

    fn cpu_fwd(
        &self,
        dst: &mut CpuStorage,
        dst_l: &Layout,
        srcs: [(&CpuStorage, &Layout); 1],
    ) -> Result<()> {
        let [(src, src_l)] = srcs;
        add_inplace(dst, dst_l, src, src_l)
    }

    fn cpu_fwd_aliased(
        &self,
        dst: &mut CpuStorage,
        dst_l: &Layout,
        srcs: [(Src<'_, CpuStorage>, &Layout); 1],
    ) -> Result<()> {
        match srcs {
            [(Src::Distinct(src), src_l)] => add_inplace(dst, dst_l, src, src_l),
            [(Src::Aliased(_), src_l)] => add_inplace_aliased(dst, dst_l, src_l),
        }
    }

    fn backend_fwd(
        &self,
        dst: &mut dyn BackendStorage,
        dst_l: &Layout,
        srcs: [(&dyn BackendStorage, &Layout); 1],
    ) -> Result<()> {
        let [(src, src_l)] = srcs;
        let dst = HostStorage::from_dyn_mut(dst)?.cpu_mut();
        add_inplace(dst, dst_l, HostStorage::from_dyn(src)?.cpu(), src_l)
    }

    fn backend_fwd_aliased(
        &self,
        dst: &mut dyn BackendStorage,
        dst_l: &Layout,
        srcs: [(Src<'_, dyn BackendStorage>, &Layout); 1],
    ) -> Result<()> {
        let dst = HostStorage::from_dyn_mut(dst)?.cpu_mut();
        match srcs {
            [(Src::Distinct(src), src_l)] => {
                add_inplace(dst, dst_l, HostStorage::from_dyn(src)?.cpu(), src_l)
            }
            [(Src::Aliased(_), src_l)] => add_inplace_aliased(dst, dst_l, src_l),
        }
    }
}

fn inplace_op_reads_an_aliased_source_elementwise(device: &Device) -> Result<()> {
    let t = Tensor::from_vec(vec![1f32, 2., 3., 4.], 4, device)?;

    // The source is the destination itself.
    t.inplace_op2(&t, &ElementwiseAdd)?;
    assert_eq!(t.to_vec1::<f32>()?, [2., 4., 6., 8.]);

    // A source in another storage takes the forward without aliasing.
    let other = Tensor::from_vec(vec![1f32, 1., 1., 1.], 4, device)?;
    t.inplace_op2(&other, &ElementwiseAdd)?;
    assert_eq!(t.to_vec1::<f32>()?, [3., 5., 7., 9.]);

    // The source is another part of the destination's storage.
    let dst = t.narrow(0, 0, 2)?;
    let src = t.narrow(0, 2, 2)?;
    dst.inplace_op2(&src, &ElementwiseAdd)?;
    assert_eq!(t.to_vec1::<f32>()?, [10., 14., 7., 9.]);

    // Overlapping views that are not the same elements stay an error.
    let dst = t.narrow(0, 0, 3)?;
    let src = t.narrow(0, 1, 3)?;
    assert!(dst.inplace_op2(&src, &ElementwiseAdd).is_err());
    assert_eq!(t.to_vec1::<f32>()?, [10., 14., 7., 9.]);
    Ok(())
}

#[test]
fn custom_op_operands_on_different_devices_are_an_error() -> Result<()> {
    let cpu = Tensor::new(&[1f32, 2.], &Device::Cpu)?;
    let backend = cpu.to_device(&common::host_device(0))?;

    let err = cpu.apply_op2(&backend, &MulAdd).unwrap_err();
    assert!(
        err.to_string().starts_with("device mismatch in mul"),
        "unexpected error: {err}"
    );

    let err = backend.inplace_op2(&cpu, &InplaceAdd).unwrap_err();
    assert!(
        err.to_string()
            .starts_with("device mismatch in inplace-add"),
        "unexpected error: {err}"
    );
    Ok(())
}

test_device!(custom_op1);
test_device!(custom_op2_and_op3);
test_device!(inplace_op1);
test_device!(inplace_op2_and_op3);
test_device!(inplace_op_shared_storage_errors_instead_of_deadlocking);
test_device!(inplace_op_reads_an_aliased_source_elementwise);
