//! What a storage's bytes are: the fields it holds, the part of a logical
//! tensor each field is, how the field's elements lie in the bytes, and
//! the chunks the storage's storage is allocated in.
//!
//! A worker's weights are storages, each a range of device storage holding
//! one or more fields (a merged projection's branches, the small tensors
//! packed together). A field is a local tensor of one logical tensor:
//! the model library declares which indices of the logical tensor it holds
//! ([`Logical`]) and the field's own shape and strides say where each local
//! element lies ([`Encoding::Affine`]); a field whose bytes no strides
//! describe (swizzled scales, permuted rows) names its encoding, which the
//! library converts to and from an affine canonical form
//! ([`Encoding::Encoded`]).
//!
//! # Chunks
//!
//! A storage's storage is allocated in chunks, each a CUDA VMM allocation of
//! its own mapped back to back with the next, so that another world can
//! map any chunk it needs on its GPU without the rest. A chunk can be
//! mapped into another world's storage only if it starts and ends at the
//! same byte layout there, on allocation-granularity boundaries.
//! [`StorageLayout::chunks`] cuts a storage where another parallel
//! configuration would cut the logical tensor: at its fields' bounds, at
//! the bounds of the segments a field holds, and wherever a degree in use
//! divides a dimension's partition units, each along a field's outermost
//! dimension, where a cut keeps the parts contiguous. Every cut rounds down
//! and up to the granularity, so the chunks between two cuts are those any
//! world holding the same region at the same phase also has. A chunk's key
//! ([`StorageLayout::chunk_key`]) names the logical regions its bytes hold
//! and where they lie in it, so equal keys hold equal bytes whatever world
//! wrote them.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use super::region::{Span, volume};
use super::{Error, Result};

/// Logical indices `span` of one dimension, held from local position
/// `local` on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Segment {
    pub span: Span,
    pub local: u64,
}

impl Segment {
    /// The local positions the segment covers.
    pub fn locals(&self) -> Range<u64> {
        self.local..self.local + self.span.count
    }
}

/// A local tensor's part of a logical tensor, as the model library declares
/// it (`uniserve.distributed.logical.Logical`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Logical {
    /// The logical tensor's shape.
    pub shape: Vec<u64>,
    /// What each dimension indexes.
    pub dims: Vec<String>,
    /// How many consecutive indices of each dimension every partition keeps
    /// together; zero for a dimension no partition divides.
    pub units: Vec<u64>,
    /// Per dimension, the logical indices held and their local positions;
    /// the local tensor holds their Cartesian product. Local positions no
    /// segment covers are zero padding.
    pub segments: Vec<Vec<Segment>>,
}

/// How a field's elements lie in its bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Encoding {
    /// Local element `i` lies at byte `offset + sum(i[d] * strides[d]) *
    /// element`.
    Affine,
    /// The field's bytes are a function of its whole local tensor under the
    /// encoding `name`, and the field's `logical` part names the part of
    /// the tensor (`group`) it encodes; the model library converts the
    /// group's fields to and from its canonical encoding, in which this
    /// field is the affine `canonical`. A field without a canonical form
    /// moves only as the bytes of a source that holds the same.
    Encoded {
        name: String,
        group: String,
        canonical: Option<Canonical>,
    },
}

/// A field's affine form in its group's canonical encoding: where it lies
/// in the staging bytes the canonical group occupies.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Canonical {
    pub offset: u64,
    pub shape: Vec<u64>,
    pub strides: Vec<u64>,
    pub logical: Logical,
}

/// One local tensor of a storage.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Field {
    /// The logical tensor: `<tensor id>/<field>`, as the retained manifest
    /// names a parameter's or buffer's field.
    pub tensor: String,
    /// The checkpoint source the tensor is loaded from; empty for a tensor
    /// no source loads.
    pub source: String,
    /// DLPack element type name.
    pub dtype: String,
    /// Bytes per element.
    pub element: u64,
    /// Byte offset of local element zero in the storage.
    pub offset: u64,
    /// The local tensor's shape.
    pub shape: Vec<u64>,
    /// The local tensor's strides, in elements.
    pub strides: Vec<u64>,
    pub logical: Logical,
    pub encoding: Encoding,
}

/// What one part of a byte window holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    /// The logical indices `logical` of field `field`, whose first element
    /// lies at local position `local` (one per dimension).
    Data {
        field: usize,
        logical: Vec<Span>,
        local: Vec<u64>,
    },
    /// Local positions `local..local + counts` of field `field` that hold
    /// padding.
    Padding {
        field: usize,
        local: Vec<u64>,
        counts: Vec<u64>,
    },
    /// Bytes `bytes` (relative to the field's offset) of an encoded field,
    /// whose meaning is the whole field's.
    Encoded { field: usize, bytes: Range<u64> },
}

impl Field {
    /// The local tensor's element count.
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    /// The bytes of the field in its storage.
    pub fn bytes(&self) -> Range<u64> {
        let extent = if self.numel() == 0 {
            0
        } else {
            self.shape
                .iter()
                .zip(&self.strides)
                .map(|(extent, stride)| (extent - 1) * stride)
                .sum::<u64>()
                + 1
        };
        self.offset..self.offset + extent * self.element
    }

    /// The byte offset in the storage of local position `local`.
    pub fn address(&self, local: &[u64]) -> u64 {
        self.offset
            + local
                .iter()
                .zip(&self.strides)
                .map(|(index, stride)| index * stride)
                .sum::<u64>()
                * self.element
    }

    /// The byte stride of each dimension.
    pub fn byte_strides(&self) -> Vec<u64> {
        self.strides
            .iter()
            .map(|stride| stride * self.element)
            .collect()
    }

    fn validate(&self) -> Result<()> {
        if self.strides.len() != self.shape.len() || self.element == 0 {
            return Err(Error::Invalid(format!(
                "field {} declares its shape and strides inconsistently",
                self.tensor
            )));
        }
        match &self.encoding {
            // An affine field holds its logical part at its own positions.
            Encoding::Affine => validate_part(&self.tensor, &self.shape, &self.logical),
            // An encoded field's part is of its group's tensor; its
            // canonical form holds it affinely.
            Encoding::Encoded { canonical, .. } => match canonical {
                Some(canonical) => {
                    if canonical.strides.len() != canonical.shape.len() {
                        return Err(Error::Invalid(format!(
                            "the canonical form of field {} declares its shape and strides \
                             inconsistently",
                            self.tensor
                        )));
                    }
                    validate_part(&self.tensor, &canonical.shape, &canonical.logical)
                }
                None => Ok(()),
            },
        }
    }

    /// The local positions of one dimension, split at the segments' bounds:
    /// each part with the segment holding it, `None` for padding.
    fn local_parts(&self, dim: usize, locals: Range<u64>) -> Vec<(Range<u64>, Option<Segment>)> {
        local_parts(&self.logical, dim, locals)
    }
}

/// Checks that a local tensor of `shape` holds the logical part `logical`
/// within its extents and the logical tensor's.
fn validate_part(tensor: &str, shape: &[u64], logical: &Logical) -> Result<()> {
    let rank = shape.len();
    if logical.shape.len() != rank || logical.units.len() != rank || logical.segments.len() != rank
    {
        return Err(Error::Invalid(format!(
            "field {tensor} declares its shape and logical part inconsistently"
        )));
    }
    for (dim, segments) in logical.segments.iter().enumerate() {
        for segment in segments {
            let fits_local = segment.local + segment.span.count <= shape[dim];
            let fits_logical = segment
                .span
                .last()
                .is_none_or(|last| last < logical.shape[dim]);
            if !fits_local || !fits_logical || segment.span.step == 0 {
                return Err(Error::Invalid(format!(
                    "field {tensor} holds a segment outside its shapes along dimension {dim}"
                )));
            }
        }
    }
    Ok(())
}

/// The local positions `locals` of dimension `dim` of a local tensor
/// holding `logical`, split at the segments' bounds: each part with the
/// segment holding it, `None` for padding.
fn local_parts(
    logical: &Logical,
    dim: usize,
    locals: Range<u64>,
) -> Vec<(Range<u64>, Option<Segment>)> {
    let mut parts = Vec::new();
    let mut position = locals.start;
    let mut segments: Vec<&Segment> = logical.segments[dim]
        .iter()
        .filter(|segment| {
            segment.locals().start < locals.end && segment.locals().end > locals.start
        })
        .collect();
    segments.sort_by_key(|segment| segment.local);
    for segment in segments {
        let start = segment.local.max(locals.start);
        let end = segment.locals().end.min(locals.end);
        if position < start {
            parts.push((position..start, None));
        }
        parts.push((start..end, Some(*segment)));
        position = end;
    }
    if position < locals.end {
        parts.push((position..locals.end, None));
    }
    parts
}

impl Field {
    /// What the local box `local..local + counts` holds: logical boxes and
    /// padding.
    fn pieces_of_box(&self, field: usize, local: &[u64], counts: &[u64], into: &mut Vec<Piece>) {
        let per_dim: Vec<Vec<(Range<u64>, Option<Segment>)>> = (0..local.len())
            .map(|dim| self.local_parts(dim, local[dim]..local[dim] + counts[dim]))
            .collect();
        let mut choice = vec![0_usize; per_dim.len()];
        loop {
            let parts: Vec<&(Range<u64>, Option<Segment>)> = choice
                .iter()
                .enumerate()
                .map(|(dim, index)| &per_dim[dim][*index])
                .collect();
            let start: Vec<u64> = parts.iter().map(|(range, _)| range.start).collect();
            // Every dimension's part held by a segment: logical data.
            let held: Option<Vec<Span>> = parts
                .iter()
                .map(|(range, segment)| {
                    segment.map(|segment| {
                        segment
                            .span
                            .narrow(range.start - segment.local, range.end - range.start)
                    })
                })
                .collect();
            if let Some(logical) = held {
                into.push(Piece::Data {
                    field,
                    logical,
                    local: start,
                });
            } else {
                into.push(Piece::Padding {
                    field,
                    local: start,
                    counts: parts
                        .iter()
                        .map(|(range, _)| range.end - range.start)
                        .collect(),
                });
            }
            // The next combination, the last dimension fastest.
            let mut dim = per_dim.len();
            loop {
                if dim == 0 {
                    return;
                }
                dim -= 1;
                choice[dim] += 1;
                if choice[dim] < per_dim[dim].len() {
                    break;
                }
                choice[dim] = 0;
            }
        }
    }

    /// What the field's bytes `window` (in the storage) hold, in byte
    /// order.
    pub fn pieces(&self, field: usize, window: &Range<u64>, into: &mut Vec<Piece>) {
        let bytes = self.bytes();
        let start = window.start.max(bytes.start);
        let end = window.end.min(bytes.end);
        if start >= end {
            return;
        }
        let order = match (&self.encoding, self.nested_order()) {
            (Encoding::Affine, Some(order)) => order,
            _ => {
                into.push(Piece::Encoded {
                    field,
                    bytes: start - self.offset..end - self.offset,
                });
                return;
            }
        };
        let mut boxes = Vec::new();
        let mut local = vec![0; self.shape.len()];
        self.boxes_in(
            &order,
            0,
            self.offset,
            &(start..end),
            &mut local,
            &mut boxes,
        );
        for (local, counts) in boxes {
            self.pieces_of_box(field, &local, &counts, into);
        }
    }

    /// The dimensions of more than one index in order of decreasing stride,
    /// when each one's indices lie wholly apart from the next (its stride
    /// covers the span of the dimensions inside it): the local tensor is a
    /// row-major tensor whose rows may leave gaps, as a column view of a
    /// wider matrix does. `None` when the strides interleave.
    fn nested_order(&self) -> Option<Vec<usize>> {
        let mut order: Vec<usize> = (0..self.shape.len())
            .filter(|&dim| self.shape[dim] > 1)
            .collect();
        order.sort_by_key(|&dim| (std::cmp::Reverse(self.strides[dim]), dim));
        let mut span = 1;
        for &dim in order.iter().rev() {
            if self.strides[dim] < span {
                return None;
            }
            span += self.strides[dim] * (self.shape[dim] - 1);
        }
        Some(order)
    }

    /// The local boxes of the elements of dimensions `order[depth..]`, the
    /// outer ones fixed at `local` and starting at byte `base`, whose bytes
    /// lie in `window`: whole runs of an outer dimension where they lie
    /// inside it, its indices one by one where the window cuts them.
    fn boxes_in(
        &self,
        order: &[usize],
        depth: usize,
        base: u64,
        window: &Range<u64>,
        local: &mut Vec<u64>,
        boxes: &mut Vec<(Vec<u64>, Vec<u64>)>,
    ) {
        let inner = &order[depth..];
        let element = self.element;
        // The bytes one index of the dimensions in `dims` spans.
        let span = |dims: &[usize]| {
            (dims
                .iter()
                .map(|&dim| self.strides[dim] * (self.shape[dim] - 1))
                .sum::<u64>()
                + 1)
                * element
        };
        let counts_from = |dims: &[usize]| {
            let mut counts = vec![1; self.shape.len()];
            for &dim in dims {
                counts[dim] = self.shape[dim];
            }
            counts
        };
        // Dense dimensions from here on: their elements are a contiguous
        // run, cut by the window into row-major boxes.
        let dense = inner.iter().rev().try_fold(1_u64, |expected, &dim| {
            (self.strides[dim] == expected).then(|| expected * self.shape[dim])
        });
        if let Some(numel) = dense {
            let first = window.start.saturating_sub(base).div_ceil(element);
            let last = (window.end.saturating_sub(base) / element).min(numel);
            let shape: Vec<u64> = inner.iter().map(|&dim| self.shape[dim]).collect();
            for (start, counts) in linear_boxes(&shape, first, last) {
                let mut at = local.clone();
                let mut extents = vec![1; self.shape.len()];
                for (position, &dim) in inner.iter().enumerate() {
                    at[dim] = start[position];
                    extents[dim] = counts[position];
                }
                boxes.push((at, extents));
            }
            return;
        }
        let dim = inner[0];
        let stride = self.strides[dim] * element;
        let row = span(&inner[1..]);
        // Indices whose bytes reach into the window.
        let first = (window.start.saturating_sub(base) + stride).saturating_sub(row) / stride;
        let first = if base + first * stride + row <= window.start {
            first + 1
        } else {
            first
        };
        let last = if window.end <= base {
            0
        } else {
            ((window.end - 1 - base) / stride + 1).min(self.shape[dim])
        };
        let mut index = first;
        while index < last {
            let row_start = base + index * stride;
            let whole = row_start >= window.start && row_start + row <= window.end;
            if whole {
                // A run of indices wholly inside the window is one box.
                let mut stop = index + 1;
                while stop < last && base + stop * stride + row <= window.end {
                    stop += 1;
                }
                let mut at = local.clone();
                at[dim] = index;
                let mut counts = counts_from(&inner[1..]);
                counts[dim] = stop - index;
                boxes.push((at, counts));
                index = stop;
            } else {
                local[dim] = index;
                self.boxes_in(order, depth + 1, row_start, window, local, boxes);
                local[dim] = 0;
                index += 1;
            }
        }
    }

    /// The local positions of the outermost dimension a cut may fall
    /// between: the field's bounds, its segments' bounds, and where a
    /// degree of `degrees` divides the dimension's partition units.
    fn cuts(&self, degrees: &[u64]) -> Vec<u64> {
        let Some(order) = self.nested_order() else {
            return Vec::new();
        };
        if !matches!(self.encoding, Encoding::Affine) {
            return Vec::new();
        }
        let Some(&outer) = order.iter().find(|&&dim| self.shape[dim] > 1) else {
            return Vec::new();
        };
        let extent = self.logical.shape[outer];
        let unit = self.logical.units[outer];
        let mut cuts = Vec::new();
        for segment in &self.logical.segments[outer] {
            cuts.push(segment.local);
            cuts.push(segment.locals().end);
            if unit == 0 || !extent.is_multiple_of(unit) {
                continue;
            }
            let units = extent / unit;
            for &degree in degrees {
                if degree <= 1 || !units.is_multiple_of(degree) {
                    continue;
                }
                let part = extent / degree;
                for boundary in (1..degree).map(|index| index * part) {
                    if let Some(position) = segment.span.position(boundary) {
                        cuts.push(segment.local + position);
                    }
                }
            }
        }
        let stride = self.strides[outer] * self.element;
        cuts.into_iter()
            .map(|local| self.offset + local * stride)
            .collect()
    }
}

/// The boxes of a row-major tensor of `shape` whose elements, in row-major
/// order, are `first..last`: each a start and the extents of every
/// dimension, at most two per dimension.
pub fn linear_boxes(shape: &[u64], first: u64, last: u64) -> Vec<(Vec<u64>, Vec<u64>)> {
    let mut boxes = Vec::new();
    let mut prefix = Vec::new();
    split_linear(shape, first, last, &mut prefix, &mut boxes);
    boxes
}

fn split_linear(
    shape: &[u64],
    first: u64,
    last: u64,
    prefix: &mut Vec<u64>,
    boxes: &mut Vec<(Vec<u64>, Vec<u64>)>,
) {
    if first >= last {
        return;
    }
    let Some((&outer, inner_shape)) = shape.split_first() else {
        // A zero-dimensional tensor's one element.
        boxes.push((prefix.clone(), Vec::new()));
        return;
    };
    let inner: u64 = inner_shape.iter().product();
    if inner == 0 || outer == 0 {
        return;
    }
    let (mut row, offset) = (first / inner, first % inner);
    let (end_row, end_offset) = (last / inner, last % inner);
    let full = |row: u64, count: u64, boxes: &mut Vec<(Vec<u64>, Vec<u64>)>, prefix: &[u64]| {
        let mut start = prefix.to_vec();
        start.push(row);
        start.extend(std::iter::repeat_n(0, inner_shape.len()));
        let mut counts = vec![1; prefix.len()];
        counts.push(count);
        counts.extend_from_slice(inner_shape);
        boxes.push((start, counts));
    };
    if row == end_row {
        prefix.push(row);
        split_linear(inner_shape, offset, end_offset, prefix, boxes);
        prefix.pop();
        return;
    }
    if offset > 0 {
        prefix.push(row);
        split_linear(inner_shape, offset, inner, prefix, boxes);
        prefix.pop();
        row += 1;
    }
    if end_row > row {
        full(row, end_row - row, boxes, prefix);
    }
    if end_offset > 0 {
        prefix.push(end_row);
        split_linear(inner_shape, 0, end_offset, prefix, boxes);
        prefix.pop();
    }
}

/// One storage: a range of device storage and the fields it holds.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StorageLayout {
    /// The bytes the fields occupy; the storage's storage rounds them up to
    /// the allocation granularity.
    pub bytes: u64,
    pub fields: Vec<Field>,
}

impl StorageLayout {
    /// Checks that every field fits the storage and its declarations.
    pub fn validate(&self) -> Result<()> {
        for field in &self.fields {
            field.validate()?;
            if field.bytes().end > self.bytes {
                return Err(Error::Invalid(format!(
                    "field {} ends past its storage's {} bytes",
                    field.tensor, self.bytes
                )));
            }
        }
        Ok(())
    }

    /// The storage's storage: its bytes rounded up to `granularity`.
    pub fn padded(&self, granularity: u64) -> u64 {
        self.bytes.div_ceil(granularity).max(1) * granularity
    }

    /// The chunks the storage's storage is allocated in, in address order,
    /// for the parallel `degrees` in use (module documentation).
    pub fn chunks(&self, granularity: u64, degrees: &[u64]) -> Vec<Range<u64>> {
        let padded = self.padded(granularity);
        let mut bounds = vec![0, padded];
        for field in &self.fields {
            let bytes = field.bytes();
            for cut in [bytes.start, bytes.end]
                .into_iter()
                .chain(field.cuts(degrees))
            {
                bounds.push(cut / granularity * granularity);
                bounds.push(cut.div_ceil(granularity) * granularity);
            }
        }
        bounds.retain(|bound| *bound <= padded);
        bounds.sort_unstable();
        bounds.dedup();
        bounds.windows(2).map(|pair| pair[0]..pair[1]).collect()
    }

    /// What the bytes `window` hold, field by field in storage order.
    pub fn pieces(&self, window: &Range<u64>) -> Vec<Piece> {
        let mut order: Vec<usize> = (0..self.fields.len()).collect();
        order.sort_by_key(|&index| (self.fields[index].offset, index));
        let mut pieces = Vec::new();
        for index in order {
            self.fields[index].pieces(index, window, &mut pieces);
        }
        pieces
    }

    /// The key of the chunk `window`: the digest of what each of its pieces
    /// holds and where in the chunk it lies. `fingerprint` gives the
    /// digest of what determines a checkpoint source's bytes (`""` for the
    /// tensors no source loads).
    pub fn chunk_key(
        &self,
        window: &Range<u64>,
        fingerprint: &dyn Fn(&str) -> Option<[u8; 32]>,
    ) -> Result<crate::Key> {
        let mut digest = crate::key::Digest::new(b"uniserve-weight-chunk");
        digest.number(window.end - window.start);
        for piece in self.pieces(window) {
            let index = match &piece {
                Piece::Data { field, .. }
                | Piece::Padding { field, .. }
                | Piece::Encoded { field, .. } => *field,
            };
            let field = &self.fields[index];
            let source = fingerprint(&field.source).ok_or_else(|| {
                Error::Invalid(format!(
                    "field {} is loaded from checkpoint source {:?}, which has no fingerprint",
                    field.tensor, field.source
                ))
            })?;
            digest.text(&field.tensor);
            digest.bytes(&source);
            digest.text(&field.dtype);
            digest.number(field.element);
            digest.numbers(&field.logical.shape);
            match &piece {
                Piece::Data { logical, local, .. } => {
                    digest.text("data");
                    for span in logical {
                        digest.numbers(&[span.start, span.count, span.step]);
                    }
                    digest.number(field.address(local) - window.start);
                    digest.numbers(&field.byte_strides());
                }
                Piece::Padding { local, counts, .. } => {
                    digest.text("padding");
                    digest.numbers(counts);
                    digest.number(field.address(local) - window.start);
                    digest.numbers(&field.byte_strides());
                }
                Piece::Encoded { bytes, .. } => {
                    // The bytes are the encoding's function of the whole
                    // local tensor: name everything that determines it.
                    digest.text("encoded");
                    if let Encoding::Encoded { name, .. } = &field.encoding {
                        digest.text(name);
                    }
                    digest.numbers(&field.shape);
                    digest.numbers(&field.strides);
                    for segments in &field.logical.segments {
                        for segment in segments {
                            digest.numbers(&[
                                segment.span.start,
                                segment.span.count,
                                segment.span.step,
                                segment.local,
                            ]);
                        }
                        digest.text("/");
                    }
                    digest.numbers(&[bytes.start, bytes.end]);
                    digest.number(field.offset + bytes.start - window.start);
                }
            }
        }
        Ok(crate::Key(digest.finish()))
    }
}

/// The number of elements of the logical box `spans`.
pub fn elements(spans: &[Span]) -> u64 {
    volume(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    /// A row-major bfloat16 field of `rows` x `columns` holding logical
    /// rows `first..first + rows` of a `[total, columns]` tensor whose
    /// partitions keep `unit` rows together.
    fn rows_field(
        tensor: &str,
        offset: u64,
        total: u64,
        first: u64,
        rows: u64,
        columns: u64,
        unit: u64,
    ) -> Field {
        Field {
            tensor: tensor.into(),
            source: "model".into(),
            dtype: "bfloat16".into(),
            element: 2,
            offset,
            shape: vec![rows, columns],
            strides: vec![columns, 1],
            logical: Logical {
                shape: vec![total, columns],
                dims: vec!["output".into(), "input".into()],
                units: vec![unit, 0],
                segments: vec![
                    vec![Segment {
                        span: Span::range(first, rows),
                        local: 0,
                    }],
                    vec![Segment {
                        span: Span::range(0, columns),
                        local: 0,
                    }],
                ],
            },
            encoding: Encoding::Affine,
        }
    }

    /// The query, key and value rows a tensor-parallel rank `rank` of
    /// `degree` holds of a projection of 32 query and 4 KV heads of 128
    /// channels over 2048 inputs, back to back as loading coalesces them.
    fn qkv(degree: u64, rank: u64) -> StorageLayout {
        let (q, kv) = (4096 / degree, 512 / degree);
        let fields = vec![
            rows_field("q", 0, 4096, rank * q, q, 2048, 128),
            rows_field("k", q * 4096, 512, rank * kv, kv, 2048, 128),
            rows_field("v", (q + kv) * 4096, 512, rank * kv, kv, 2048, 128),
        ];
        StorageLayout {
            bytes: (q + 2 * kv) * 4096,
            fields,
        }
    }

    fn keys(storage: &StorageLayout, degrees: &[u64]) -> Vec<(Range<u64>, crate::Key)> {
        storage
            .chunks(2 * MIB, degrees)
            .into_iter()
            .map(|chunk| {
                let key = storage.chunk_key(&chunk, &|_| Some([7; 32])).unwrap();
                (chunk, key)
            })
            .collect()
    }

    #[test]
    fn query_groups_of_every_tensor_parallel_rank_share_chunks() {
        let degrees = [1, 2, 4];
        let whole = keys(&qkv(1, 0), &degrees);
        // One chunk per KV-head group of queries, then the keys and the
        // values: the groups of 4 MiB lie on granularity bounds.
        let bounds: Vec<Range<u64>> = whole.iter().map(|(chunk, _)| chunk.clone()).collect();
        assert_eq!(
            bounds,
            vec![
                0..4 * MIB,
                4 * MIB..8 * MIB,
                8 * MIB..12 * MIB,
                12 * MIB..16 * MIB,
                16 * MIB..18 * MIB,
                18 * MIB..20 * MIB,
            ]
        );
        for degree in [2, 4] {
            for rank in 0..degree {
                let part = keys(&qkv(degree, rank), &degrees);
                let groups = 4 / degree;
                for group in 0..groups {
                    // The rank's group lies at its own offset in the whole
                    // projection.
                    let global = (rank * groups + group) as usize;
                    assert_eq!(part[group as usize].1, whole[global].1);
                }
                // Its keys and values are a fraction of a granule: they
                // share no chunk with the whole projection's.
                for (_, key) in &part[groups as usize..] {
                    assert!(whole.iter().all(|(_, other)| other != key));
                }
            }
        }
    }

    #[test]
    fn a_vocabulary_shard_shares_the_aligned_rows_of_the_whole() {
        // 151936 rows of 4 KiB: a quarter ends mid-granule, and the chunks
        // up to the granule before it are the whole table's.
        let degrees = [1, 2, 4];
        let whole = StorageLayout {
            bytes: 151_936 * 4096,
            fields: vec![rows_field("embedding", 0, 151_936, 0, 151_936, 2048, 1)],
        };
        let quarter = StorageLayout {
            bytes: 37_984 * 4096,
            fields: vec![rows_field("embedding", 0, 151_936, 0, 37_984, 2048, 1)],
        };
        let whole = keys(&whole, &degrees);
        let quarter = keys(&quarter, &degrees);
        assert_eq!(quarter[0].0, 0..37_888 * 4096);
        assert_eq!(quarter[0].1, whole[0].1);
        // The tail holds rows the whole table holds with other rows.
        assert!(whole.iter().all(|(_, key)| *key != quarter[1].1));
    }

    #[test]
    fn windows_decompose_into_their_logical_boxes() {
        // Experts [4, 6, 8] whose second dimension is two sub-blocks of
        // three rows, of which the field holds rows 1..3 and 4..6.
        let field = Field {
            tensor: "up_gate".into(),
            source: String::new(),
            dtype: "float32".into(),
            element: 4,
            offset: 64,
            shape: vec![4, 4, 8],
            strides: vec![32, 8, 1],
            logical: Logical {
                shape: vec![4, 6, 8],
                dims: vec!["expert".into(), "output".into(), "input".into()],
                units: vec![1, 1, 0],
                segments: vec![
                    vec![Segment {
                        span: Span::range(0, 4),
                        local: 0,
                    }],
                    vec![
                        Segment {
                            span: Span::range(1, 2),
                            local: 0,
                        },
                        Segment {
                            span: Span::range(4, 2),
                            local: 2,
                        },
                    ],
                    vec![Segment {
                        span: Span::range(0, 8),
                        local: 0,
                    }],
                ],
            },
            encoding: Encoding::Affine,
        };
        // A window from mid-row of expert 0 to mid-row of expert 2.
        let window = 64 + (3 * 8 + 4) * 4..64 + (2 * 32 + 9) * 4;
        let mut pieces = Vec::new();
        field.pieces(0, &window, &mut pieces);
        let mut covered = Vec::new();
        for piece in &pieces {
            let Piece::Data { logical, local, .. } = piece else {
                panic!("an affine field without padding holds data only");
            };
            for e in 0..logical[0].count {
                for r in 0..logical[1].count {
                    for c in 0..logical[2].count {
                        let address = field.address(&[local[0] + e, local[1] + r, local[2] + c]);
                        covered.push((
                            address,
                            [
                                logical[0].index(e),
                                logical[1].index(r),
                                logical[2].index(c),
                            ],
                        ));
                    }
                }
            }
        }
        covered.sort_unstable();
        let addresses: Vec<u64> = covered.iter().map(|(address, _)| *address).collect();
        let expected: Vec<u64> = window.clone().step_by(4).collect();
        assert_eq!(addresses, expected);
        // Local row 2 of each expert is logical row 4 (the second block).
        let (_, first_of_block) = covered
            .iter()
            .find(|(address, _)| *address == field.address(&[1, 2, 0]))
            .unwrap();
        assert_eq!(*first_of_block, [1, 4, 0]);
    }
}
