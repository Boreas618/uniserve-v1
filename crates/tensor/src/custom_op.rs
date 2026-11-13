use crate::backend::BackendStorage;
use crate::tensor::from_storage;
use crate::{bail, CpuStorage, Layout, LayoutRelation, Result, Shape, Storage, Tensor};

/// Unary ops that can be defined in user-land.
///
/// An op computes on CPU storage in [`CustomOp1::cpu_fwd`] and on the storage of a device
/// backend in [`CustomOp1::backend_fwd`]. The input storage can use arbitrary strides and offsets,
/// so each forward accesses it through the associated layout, and returns its output storage
/// together with the output shape, which views it contiguously.
pub trait CustomOp1 {
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on the cpu device.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a backend device. The op downcasts the storage to the
    /// backend's type through [`BackendStorage::as_any`].
    fn backend_fwd(
        &self,
        storage: &dyn BackendStorage,
        layout: &Layout,
    ) -> Result<(Box<dyn BackendStorage>, Shape)> {
        let _ = (storage, layout);
        bail!("no backend implementation for {}", self.name())
    }
}

/// Binary ops that can be defined in user-land, see [`CustomOp1`].
pub trait CustomOp2 {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on the cpu device.
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a backend device.
    fn backend_fwd(
        &self,
        s1: &dyn BackendStorage,
        l1: &Layout,
        s2: &dyn BackendStorage,
        l2: &Layout,
    ) -> Result<(Box<dyn BackendStorage>, Shape)> {
        let _ = (s1, l1, s2, l2);
        bail!("no backend implementation for {}", self.name())
    }
}

/// Ternary ops that can be defined in user-land, see [`CustomOp1`].
pub trait CustomOp3 {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on the cpu device.
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a backend device.
    fn backend_fwd(
        &self,
        s1: &dyn BackendStorage,
        l1: &Layout,
        s2: &dyn BackendStorage,
        l2: &Layout,
        s3: &dyn BackendStorage,
        l3: &Layout,
    ) -> Result<(Box<dyn BackendStorage>, Shape)> {
        let _ = (s1, l1, s2, l2, s3, l3);
        bail!("no backend implementation for {}", self.name())
    }
}

impl Tensor {
    /// Applies a unary custom op.
    pub fn apply_op1(&self, c: &dyn CustomOp1) -> Result<Self> {
        let (storage, shape) = self.storage().apply_op1(self.layout(), c)?;
        Ok(from_storage(storage, shape))
    }

    /// Applies a binary custom op.
    pub fn apply_op2(&self, rhs: &Self, c: &dyn CustomOp2) -> Result<Self> {
        let (storage, shape) =
            self.storage()
                .apply_op2(self.layout(), &rhs.storage(), rhs.layout(), c)?;
        Ok(from_storage(storage, shape))
    }

    /// Applies a ternary custom op.
    pub fn apply_op3(&self, t2: &Self, t3: &Self, c: &dyn CustomOp3) -> Result<Self> {
        let (storage, shape) = self.storage().apply_op3(
            self.layout(),
            &t2.storage(),
            t2.layout(),
            &t3.storage(),
            t3.layout(),
            c,
        )?;
        Ok(from_storage(storage, shape))
    }
}

// In place ops.

/// Unary ops that can be defined in user-land and that write their result into their operand.
pub trait InplaceOp1 {
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on the cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> Result<()>;

    /// The forward pass, as run on a backend device. Note that the storage can use arbitrary
    /// strides, offsets etc so the associated layout should be used to access it.
    fn backend_fwd(&self, storage: &mut dyn BackendStorage, layout: &Layout) -> Result<()> {
        let _ = (storage, layout);
        bail!("no backend implementation for {}", self.name())
    }
}

/// In-place ops that can be defined in user-land: `N` sources are read and the destination is
/// written.
pub trait InplaceOpN<const N: usize> {
    fn name(&self) -> &'static str;

    /// Defines the source access pattern of this in-place op.
    /// Defaults to `None`, which rejects all in-place source aliasing.
    fn src_access_pattern(&self) -> Option<AccessPattern> {
        None
    }

    fn cpu_fwd(
        &self,
        dst: &mut CpuStorage,
        dst_l: &Layout,
        srcs: [(&CpuStorage, &Layout); N],
    ) -> Result<()> {
        let _ = (dst, dst_l, srcs);
        bail!("no cpu implementation for {}", self.name())
    }

    fn cpu_fwd_aliased(
        &self,
        dst: &mut CpuStorage,
        dst_l: &Layout,
        srcs: [(Src<'_, CpuStorage>, &Layout); N],
    ) -> Result<()> {
        let _ = (dst, dst_l, srcs);
        bail!("no aliased cpu implementation for {}", self.name())
    }

    fn backend_fwd(
        &self,
        dst: &mut dyn BackendStorage,
        dst_l: &Layout,
        srcs: [(&dyn BackendStorage, &Layout); N],
    ) -> Result<()> {
        let _ = (dst, dst_l, srcs);
        bail!("no backend implementation for {}", self.name())
    }

    fn backend_fwd_aliased(
        &self,
        dst: &mut dyn BackendStorage,
        dst_l: &Layout,
        srcs: [(Src<'_, dyn BackendStorage>, &Layout); N],
    ) -> Result<()> {
        let _ = (dst, dst_l, srcs);
        bail!("no aliased backend implementation for {}", self.name())
    }
}

#[derive(Debug)]
pub enum Src<'a, S: ?Sized> {
    Distinct(&'a S),
    Aliased(LayoutRelation),
}

impl<S: ?Sized> Copy for Src<'_, S> {}

impl<S: ?Sized> Clone for Src<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}

// If all sources are distinct we return the entire array of tensors without the `Src` wrapper.
pub(crate) fn all_distinct<'a, B: ?Sized, const N: usize>(
    srcs: &[(Src<'a, B>, &'a Layout); N],
) -> Option<[(&'a B, &'a Layout); N]> {
    srcs.iter()
        .all(|(s, _)| matches!(s, Src::Distinct(_)))
        .then(|| {
            std::array::from_fn(|i| match srcs[i] {
                (Src::Distinct(s), l) => (s, l),
                _ => unreachable!("checked immediately above"),
            })
        })
}

/// Indicates which indices a kernel reads, relative to the destination indices it writes.
///
/// When used with [`crate::LayoutRelation`] we can describe safe access patterns.
/// For example `Elementwise` is safe to use with both `LayoutRelation::Identical` and `LayoutRelation::Disjoint`,
/// while `Arbitrary` is only guaranteed to be safe with `LayoutRelation::Disjoint`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum AccessPattern {
    /// Reads only source index `i` when writing destination index `i`.
    Elementwise,
    /// Reads arbitrary source indices.
    Arbitrary,
}

impl AccessPattern {
    fn supports(self, rel: LayoutRelation) -> bool {
        matches!(
            (self, rel),
            (
                AccessPattern::Elementwise,
                LayoutRelation::Identical | LayoutRelation::Disjoint
            ) | (AccessPattern::Arbitrary, LayoutRelation::Disjoint)
        )
    }
}

impl<C: InplaceOp1> InplaceOpN<0> for C {
    fn name(&self) -> &'static str {
        InplaceOp1::name(self)
    }

    fn cpu_fwd(
        &self,
        dst: &mut CpuStorage,
        dl: &Layout,
        _: [(&CpuStorage, &Layout); 0],
    ) -> Result<()> {
        InplaceOp1::cpu_fwd(self, dst, dl)
    }

    fn cpu_fwd_aliased(
        &self,
        dst: &mut CpuStorage,
        dl: &Layout,
        _: [(Src<'_, CpuStorage>, &Layout); 0],
    ) -> Result<()> {
        InplaceOp1::cpu_fwd(self, dst, dl)
    }

    fn backend_fwd(
        &self,
        dst: &mut dyn BackendStorage,
        dl: &Layout,
        _: [(&dyn BackendStorage, &Layout); 0],
    ) -> Result<()> {
        InplaceOp1::backend_fwd(self, dst, dl)
    }

    fn backend_fwd_aliased(
        &self,
        dst: &mut dyn BackendStorage,
        dl: &Layout,
        _: [(Src<'_, dyn BackendStorage>, &Layout); 0],
    ) -> Result<()> {
        InplaceOp1::backend_fwd(self, dst, dl)
    }
}

impl<C: InplaceOp2> InplaceOpN<1> for C {
    fn name(&self) -> &'static str {
        InplaceOp2::name(self)
    }

    fn cpu_fwd(
        &self,
        dst: &mut CpuStorage,
        dl: &Layout,
        srcs: [(&CpuStorage, &Layout); 1],
    ) -> Result<()> {
        let [(s, sl)] = srcs;
        InplaceOp2::cpu_fwd(self, dst, dl, s, sl)
    }

    fn cpu_fwd_aliased(
        &self,
        dst: &mut CpuStorage,
        dl: &Layout,
        srcs: [(Src<'_, CpuStorage>, &Layout); 1],
    ) -> Result<()> {
        match srcs {
            [(Src::Distinct(s), sl)] => InplaceOp2::cpu_fwd(self, dst, dl, s, sl),
            _ => bail!("{}: an aliased input requires InplaceOpN", self.name()),
        }
    }

    fn backend_fwd(
        &self,
        dst: &mut dyn BackendStorage,
        dl: &Layout,
        srcs: [(&dyn BackendStorage, &Layout); 1],
    ) -> Result<()> {
        let [(s, sl)] = srcs;
        InplaceOp2::backend_fwd(self, dst, dl, s, sl)
    }

    fn backend_fwd_aliased(
        &self,
        dst: &mut dyn BackendStorage,
        dl: &Layout,
        srcs: [(Src<'_, dyn BackendStorage>, &Layout); 1],
    ) -> Result<()> {
        match srcs {
            [(Src::Distinct(s), sl)] => InplaceOp2::backend_fwd(self, dst, dl, s, sl),
            _ => bail!("{}: an aliased input requires InplaceOpN", self.name()),
        }
    }
}

/// Binary ops that can be defined in user-land and that write their result into their first
/// operand.
pub trait InplaceOp2 {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on the cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(&self, s1: &mut CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout)
        -> Result<()>;

    /// The forward pass, as run on a backend device. Note that the storage can use arbitrary
    /// strides, offsets etc so the associated layout should be used to access it.
    fn backend_fwd(
        &self,
        s1: &mut dyn BackendStorage,
        l1: &Layout,
        s2: &dyn BackendStorage,
        l2: &Layout,
    ) -> Result<()> {
        let _ = (s1, l1, s2, l2);
        bail!("no backend implementation for {}", self.name())
    }
}

impl<C: InplaceOp3> InplaceOpN<2> for C {
    fn name(&self) -> &'static str {
        InplaceOp3::name(self)
    }

    fn cpu_fwd(
        &self,
        dst: &mut CpuStorage,
        dl: &Layout,
        srcs: [(&CpuStorage, &Layout); 2],
    ) -> Result<()> {
        let [(s1, l1), (s2, l2)] = srcs;
        InplaceOp3::cpu_fwd(self, dst, dl, s1, l1, s2, l2)
    }

    fn cpu_fwd_aliased(
        &self,
        dst: &mut CpuStorage,
        dl: &Layout,
        srcs: [(Src<'_, CpuStorage>, &Layout); 2],
    ) -> Result<()> {
        match srcs {
            [(Src::Distinct(s1), l1), (Src::Distinct(s2), l2)] => {
                InplaceOp3::cpu_fwd(self, dst, dl, s1, l1, s2, l2)
            }
            _ => bail!("{}: an aliased input requires InplaceOpN", self.name()),
        }
    }

    fn backend_fwd(
        &self,
        dst: &mut dyn BackendStorage,
        dl: &Layout,
        srcs: [(&dyn BackendStorage, &Layout); 2],
    ) -> Result<()> {
        let [(s1, l1), (s2, l2)] = srcs;
        InplaceOp3::backend_fwd(self, dst, dl, s1, l1, s2, l2)
    }

    fn backend_fwd_aliased(
        &self,
        dst: &mut dyn BackendStorage,
        dl: &Layout,
        srcs: [(Src<'_, dyn BackendStorage>, &Layout); 2],
    ) -> Result<()> {
        match srcs {
            [(Src::Distinct(s1), l1), (Src::Distinct(s2), l2)] => {
                InplaceOp3::backend_fwd(self, dst, dl, s1, l1, s2, l2)
            }
            _ => bail!("{}: an aliased input requires InplaceOpN", self.name()),
        }
    }
}

/// Ternary ops that can be defined in user-land and that write their result into their first
/// operand.
pub trait InplaceOp3 {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on the cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<()>;

    /// The forward pass, as run on a backend device. Note that the storage can use arbitrary
    /// strides, offsets etc so the associated layout should be used to access it.
    fn backend_fwd(
        &self,
        s1: &mut dyn BackendStorage,
        l1: &Layout,
        s2: &dyn BackendStorage,
        l2: &Layout,
        s3: &dyn BackendStorage,
        l3: &Layout,
    ) -> Result<()> {
        let _ = (s1, l1, s2, l2, s3, l3);
        bail!("no backend implementation for {}", self.name())
    }
}

impl Tensor {
    /// Applies a custom op in-place for the `self` tensor.
    ///
    /// Tensors sharing underlying storage with `self` are classified and passed as [`Src::Aliased`].
    /// Separate tensors are locked and passed as [`Src::Distinct`].
    fn inplace_op<const N: usize, C: InplaceOpN<N>>(&self, srcs: [&Self; N], c: &C) -> Result<()> {
        let name = c.name();

        // Ensure writes cannot collide with themselves
        if self.layout().has_internal_overlap() {
            bail!("{name}: dst has repeated elements (zero-stride). Can not write in-place")
        }

        // Classify srcs wrt dst
        let access = c.src_access_pattern();
        let mut rels: [Option<LayoutRelation>; N] = [None; N];
        for i in 0..N {
            if !self.same_storage(srcs[i]) {
                continue;
            }
            let rel = Layout::relation(self.layout(), srcs[i].layout());
            match access {
                Some(a) if a.supports(rel) => rels[i] = Some(rel),
                Some(a) => bail!(
                    "src {i} shares storage with dst ({rel:?}), which is not supported for the access pattern of `{name}` ({a:?})."
                ),
                None => bail!(
                    "src {i} shares storage with dst, and `{name}` does not support aliased operands."
                ),
            }
        }

        // Acquire locks in order sorted by `Tensor::storage_key`.
        // Avoids deadlock from cycle of waiting locks.
        let dst_key = self.storage_key();

        let mut order: [usize; N] = std::array::from_fn(|i| i);
        order.sort_unstable_by_key(|&i| srcs[i].storage_key());

        let mut guards: [Option<_>; N] = std::array::from_fn(|_| None);
        let mut dst: Option<_> = None;

        for &i in order.iter() {
            if rels[i].is_some() {
                continue; // Aliased. Read through `dst`
            }
            let key = srcs[i].storage_key();
            if key > dst_key && dst.is_none() {
                dst = Some(self.storage_mut());
            }
            // Two sources sharing the same allocation, but distinct from dst.
            // `read_recursive` allows for shared read access
            guards[i] = Some(srcs[i].storage());
        }
        // If `dst` is not yet set we acquire it from `self` now.
        let mut dst = match dst {
            Some(g) => g,
            None => self.storage_mut(),
        };

        let operands: [(Src<'_, Storage>, &Layout); N] = std::array::from_fn(|i| {
            let s = match (&guards[i], rels[i]) {
                (Some(g), None) => Src::Distinct(&**g),
                (None, Some(rel)) => Src::Aliased(rel),
                _ => unreachable!(
                    "Source is either distinct or aliased. Other match patterns should be impossible"
                ),
            };
            (s, srcs[i].layout())
        });

        dst.inplace_op(self.layout(), operands, c)
    }

    /// Applies a unary custom op in place.
    pub fn inplace_op1<C: InplaceOp1>(&self, c: &C) -> Result<()> {
        self.inplace_op([], c)
    }

    /// Applies a binary custom op in place (for the first tensor).
    pub fn inplace_op2<C: InplaceOpN<1>>(&self, rhs: &Self, c: &C) -> Result<()> {
        self.inplace_op([rhs], c)
    }

    /// Applies a ternary custom op in place (for the first tensor).
    pub fn inplace_op3<C: InplaceOpN<2>>(&self, t2: &Self, t3: &Self, c: &C) -> Result<()> {
        self.inplace_op([t2, t3], c)
    }
}
