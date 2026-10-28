//! The computation of a [`Manifest`] (module documentation of
//! [`super`]).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;

use super::layout::{Encoding, Field, Logical, Piece, Segment};
use super::region::{Span, atoms, volume};
use super::{
    Conversion, Copy, Endpoint, Error, Fill, Gpu, LinkLoad, Manifest, Mapping, Missing, Request,
    Result, Side, Space, Staging, uuid,
};
use crate::Key;
use crate::protocol::Memory;

/// The most tied GPU assignments whose transfers are planned to find the
/// one whose busiest link finishes first: every assignment of up to five
/// ranks to five GPUs.
const TIED_ASSIGNMENTS: usize = 120;

/// The most target ranks whose GPU assignments are enumerated; larger
/// worlds take the assignment of most local bytes the Hungarian method
/// finds.
const ENUMERATED_RANKS: usize = 8;

/// Where an affine view of a field lies on its rank: the field itself in
/// its storage, or an encoded field's canonical form in a staging space.
#[derive(Clone, Copy)]
struct View<'a> {
    space: Space,
    offset: u64,
    strides: &'a [u64],
    element: u64,
    logical: &'a Logical,
}

impl View<'_> {
    fn address(&self, local: &[u64]) -> u64 {
        self.offset
            + local
                .iter()
                .zip(self.strides)
                .map(|(index, stride)| index * stride)
                .sum::<u64>()
                * self.element
    }

    /// The segment of dimension `dim` holding all of `span`.
    fn segment(&self, dim: usize, span: &Span) -> Option<&Segment> {
        self.logical.segments[dim]
            .iter()
            .find(|segment| span.within(&segment.span))
    }

    /// The local position of `span`'s first index along `dim`, and the
    /// local step between its indices; `None` when no one segment holds it.
    fn locate(&self, dim: usize, span: &Span) -> Option<(u64, u64)> {
        let segment = self.segment(dim, span)?;
        let position = segment.span.position(span.start)?;
        Some((segment.local + position, span.step / segment.span.step))
    }
}

/// A field one side's rank holds, by its storage and index.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct FieldRef {
    rank: u32,
    storage: u32,
    field: u32,
}

/// A source chunk: its rank, storage and index.
type SourceChunk = (u32, u32, u32);

/// The source fields that hold a block, each read in a form.
type Candidates = Vec<(FieldRef, Form)>;

/// How a field is read or written: as stored, or in canonical form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Form {
    Stored,
    Canonical,
}

/// A block of a target field's region and the sources that hold it.
struct Cell {
    target: FieldRef,
    form: Form,
    what: Block,
    bytes: u64,
    /// The sources holding the block, as fields read in a form.
    candidates: Candidates,
    /// The chosen source.
    chosen: Option<(FieldRef, Form)>,
}

enum Block {
    /// The logical box `logical` of the field's tensor.
    Region(Vec<Span>),
    /// The bytes (relative to the field's offset) of an encoded field,
    /// copied as they are from a source of the same encoding and region.
    Bytes(Range<u64>),
}

struct Context<'r> {
    request: &'r Request,
    /// Source fields by logical tensor.
    sources: HashMap<&'r str, Vec<FieldRef>>,
    /// Source chunks by GPU and key.
    chunks: HashMap<(Gpu, Key), SourceChunk>,
    /// Each measured link's bandwidth in bytes per second.
    bandwidth: HashMap<(Gpu, Gpu), f64>,
}

impl<'r> Context<'r> {
    fn new(request: &'r Request) -> Result<Self> {
        let mut sources: HashMap<&str, Vec<FieldRef>> = HashMap::new();
        let mut chunks = HashMap::new();
        for (rank, source) in request.source.ranks.iter().enumerate() {
            let gpu = source
                .gpu
                .ok_or_else(|| Error::Invalid(format!("source rank {rank} names no GPU")))?;
            for (storage, held) in source.storages.iter().enumerate() {
                held.layout.validate()?;
                for (field, value) in held.layout.fields.iter().enumerate() {
                    sources.entry(&value.tensor).or_default().push(FieldRef {
                        rank: rank as u32,
                        storage: storage as u32,
                        field: field as u32,
                    });
                }
                for (chunk, value) in held.chunks.iter().enumerate() {
                    chunks.entry((gpu, value.key)).or_insert((
                        rank as u32,
                        storage as u32,
                        chunk as u32,
                    ));
                }
            }
        }
        for target in &request.target.ranks {
            for held in &target.storages {
                held.layout.validate()?;
            }
        }
        if request.gpus.len() < request.target.ranks.len() {
            return Err(Error::Invalid(format!(
                "{} target ranks cannot take {} GPUs",
                request.target.ranks.len(),
                request.gpus.len()
            )));
        }
        // Links between GPUs, as the Transfer Server measured them; a copy
        // within one GPU takes the link from it to itself. A GPU of another
        // host reaches this host's over the NVLink fabric (a link between
        // the two GPUs) or by READ (a link from that host's GPU): the
        // Transfer Server copies over the fabric whenever it can, and a
        // pair measured both ways takes the faster.
        let mut bandwidth: HashMap<(Gpu, Gpu), f64> = HashMap::new();
        for measured in &request.links {
            let pair = match (&measured.link.source, &measured.link.destination) {
                (
                    Memory::Device { gpu: source }
                    | Memory::Remote {
                        gpu: Some(source), ..
                    },
                    Memory::Device { gpu: destination },
                ) => (*source, *destination),
                _ => continue,
            };
            let entry = bandwidth.entry(pair).or_default();
            *entry = entry.max(measured.bytes_per_second as f64);
        }
        Ok(Self {
            request,
            sources,
            chunks,
            bandwidth,
        })
    }

    fn field(&self, side: Side, at: FieldRef) -> &'r Field {
        let world = match side {
            Side::Source => &self.request.source,
            Side::Target => &self.request.target,
        };
        &world.ranks[at.rank as usize].storages[at.storage as usize]
            .layout
            .fields[at.field as usize]
    }

    /// The GPU of source rank `rank`, which `Context::new` checked it
    /// names.
    fn source_gpu(&self, rank: u32) -> Gpu {
        self.request.source.ranks[rank as usize]
            .gpu
            .unwrap_or_default()
    }

    /// The bytes of target rank `rank`'s regions that the source ranks on
    /// GPU `gpu` hold.
    fn local_bytes(&self, rank: usize, gpu: Gpu) -> u64 {
        let mut bytes = 0;
        for held in &self.request.target.ranks[rank].storages {
            for field in &held.layout.fields {
                let Some(sources) = self.sources.get(field.tensor.as_str()) else {
                    continue;
                };
                for source in sources {
                    if self.source_gpu(source.rank) != gpu {
                        continue;
                    }
                    let other = self.field(Side::Source, *source);
                    bytes += shared_elements(&field.logical, &other.logical) * field.element;
                }
            }
        }
        bytes
    }
}

/// The number of logical elements both parts hold.
fn shared_elements(one: &Logical, other: &Logical) -> u64 {
    if one.shape != other.shape {
        return 0;
    }
    one.segments
        .iter()
        .zip(&other.segments)
        .map(|(mine, theirs)| {
            mine.iter()
                .flat_map(|a| theirs.iter().map(move |b| (a, b)))
                .filter_map(|(a, b)| a.span.intersect(&b.span))
                .map(|span| span.count)
                .sum::<u64>()
        })
        .product()
}

/// The GPU assignments (target rank to index of a GPU) that maximize the
/// bytes each rank finds on its GPU, at most [`TIED_ASSIGNMENTS`] of them.
fn assignments(local: &[Vec<u64>], gpus: usize) -> Vec<Vec<usize>> {
    let ranks = local.len();
    if ranks > ENUMERATED_RANKS {
        return vec![hungarian(local, gpus)];
    }
    let mut best = (0_u64, Vec::new());
    let mut chosen = Vec::with_capacity(ranks);
    let mut used = vec![false; gpus];
    enumerate(local, &mut chosen, &mut used, 0, &mut best);
    best.1
}

fn enumerate(
    local: &[Vec<u64>],
    chosen: &mut Vec<usize>,
    used: &mut [bool],
    bytes: u64,
    best: &mut (u64, Vec<Vec<usize>>),
) {
    let rank = chosen.len();
    if rank == local.len() {
        if bytes > best.0 || best.1.is_empty() {
            *best = (bytes, vec![chosen.clone()]);
        } else if bytes == best.0 && best.1.len() < TIED_ASSIGNMENTS {
            best.1.push(chosen.clone());
        }
        return;
    }
    for gpu in 0..used.len() {
        if used[gpu] {
            continue;
        }
        used[gpu] = true;
        chosen.push(gpu);
        enumerate(local, chosen, used, bytes + local[rank][gpu], best);
        chosen.pop();
        used[gpu] = false;
    }
}

/// The assignment of ranks to distinct GPUs of the most local bytes, by
/// the Hungarian method on the negated bytes.
fn hungarian(local: &[Vec<u64>], gpus: usize) -> Vec<usize> {
    let ranks = local.len();
    let cost = |rank: usize, gpu: usize| -> i128 { -i128::from(local[rank][gpu]) };
    // Potentials and matching with 1-based sentinels (rows: ranks, columns:
    // GPUs).
    let (mut u, mut v) = (vec![0_i128; ranks + 1], vec![0_i128; gpus + 1]);
    let mut matched = vec![0_usize; gpus + 1];
    let mut way = vec![0_usize; gpus + 1];
    for rank in 1..=ranks {
        matched[0] = rank;
        let mut column = 0;
        let mut least = vec![i128::MAX; gpus + 1];
        let mut visited = vec![false; gpus + 1];
        loop {
            visited[column] = true;
            let row = matched[column];
            let mut delta = i128::MAX;
            let mut next = 0;
            for gpu in 1..=gpus {
                if visited[gpu] {
                    continue;
                }
                let reduced = cost(row - 1, gpu - 1) - u[row] - v[gpu];
                if reduced < least[gpu] {
                    least[gpu] = reduced;
                    way[gpu] = column;
                }
                if least[gpu] < delta {
                    delta = least[gpu];
                    next = gpu;
                }
            }
            for gpu in 0..=gpus {
                if visited[gpu] {
                    u[matched[gpu]] += delta;
                    v[gpu] -= delta;
                } else {
                    least[gpu] -= delta;
                }
            }
            column = next;
            if matched[column] == 0 {
                break;
            }
        }
        loop {
            let previous = way[column];
            matched[column] = matched[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }
    let mut assignment = vec![0; ranks];
    for gpu in 1..=gpus {
        if matched[gpu] > 0 {
            assignment[matched[gpu] - 1] = gpu - 1;
        }
    }
    assignment
}

pub(super) fn plan(request: &Request) -> Result<Manifest> {
    let context = Context::new(request)?;
    let ranks = request.target.ranks.len();
    // Target ranks already running on their GPUs keep them; otherwise the
    // manifest assigns every rank a GPU.
    let fixed: Option<Vec<usize>> = request
        .target
        .ranks
        .iter()
        .map(|rank| {
            rank.gpu
                .and_then(|gpu| request.gpus.iter().position(|candidate| *candidate == gpu))
        })
        .collect();
    let candidates = match fixed {
        Some(assignment) => vec![assignment],
        None if request.target.ranks.iter().any(|rank| rank.gpu.is_some()) => {
            return Err(Error::Invalid(
                "target ranks name GPUs for some ranks only, or GPUs outside the request's".into(),
            ));
        }
        None => {
            let local: Vec<Vec<u64>> = (0..ranks)
                .map(|rank| {
                    request
                        .gpus
                        .iter()
                        .map(|gpu| context.local_bytes(rank, *gpu))
                        .collect()
                })
                .collect();
            assignments(&local, request.gpus.len())
        }
    };
    let mut best: Option<(f64, Manifest)> = None;
    for assignment in candidates {
        let gpus: Vec<Gpu> = assignment.iter().map(|&gpu| request.gpus[gpu]).collect();
        let manifest = Draft::new(&context, gpus).build()?;
        let busiest = manifest
            .links
            .iter()
            .map(|link| link.seconds)
            .fold(0.0, f64::max);
        if best.as_ref().is_none_or(|(seconds, _)| busiest < *seconds) {
            best = Some((busiest, manifest));
        }
    }
    let (_, mut manifest) = best.ok_or_else(|| Error::Invalid("no GPU assignment".into()))?;
    manifest.request = request.digest()?;
    manifest.digest = manifest.compute_digest()?;
    Ok(manifest)
}

/// One GPU assignment's manifest under construction.
struct Draft<'c, 'r> {
    context: &'c Context<'r>,
    gpus: Vec<Gpu>,
    mappings: Vec<Mapping>,
    stagings: Vec<Staging>,
    staging_index: HashMap<(Side, u32, String), u32>,
    encodes: BTreeMap<u32, Conversion>,
    decodes: BTreeMap<u32, Conversion>,
    cells: Vec<Cell>,
    fills: Vec<Fill>,
    missing: Vec<Missing>,
}

impl<'c, 'r> Draft<'c, 'r> {
    fn new(context: &'c Context<'r>, gpus: Vec<Gpu>) -> Self {
        Self {
            context,
            gpus,
            mappings: Vec::new(),
            stagings: Vec::new(),
            staging_index: HashMap::new(),
            encodes: BTreeMap::new(),
            decodes: BTreeMap::new(),
            cells: Vec::new(),
            fills: Vec::new(),
            missing: Vec::new(),
        }
    }

    fn build(mut self) -> Result<Manifest> {
        for rank in 0..self.context.request.target.ranks.len() {
            self.rank(rank as u32)?;
        }
        let links = self.choose_sources()?;
        let copies = self.copies()?;
        let stagings = std::mem::take(&mut self.stagings);
        Ok(Manifest {
            request: [0; 32],
            gpus: self.gpus,
            mappings: self.mappings,
            stagings,
            encodes: self.encodes.into_values().collect(),
            copies,
            decodes: self.decodes.into_values().collect(),
            fills: merge_fills(self.fills),
            missing: self.missing,
            links,
            digest: [0; 32],
        })
    }

    /// Map, copy or fill everything target rank `rank` holds.
    fn rank(&mut self, rank: u32) -> Result<()> {
        let context = self.context;
        let target = &context.request.target.ranks[rank as usize];
        let gpu = self.gpus[rank as usize];
        // Chunks a source on the same GPU holds under the same key.
        let mut mapped: Vec<Vec<Option<SourceChunk>>> = target
            .storages
            .iter()
            .map(|storage| {
                storage
                    .chunks
                    .iter()
                    .map(|chunk| context.chunks.get(&(gpu, chunk.key)).copied())
                    .collect()
            })
            .collect();
        // An encoded field is written whole by its group's conversion, so
        // every chunk touching a group that is not mapped whole is written.
        loop {
            let written = self.written_groups(rank, &mapped);
            let mut changed = false;
            for (storage, held) in target.storages.iter().enumerate() {
                for (chunk, value) in held.chunks.iter().enumerate() {
                    if mapped[storage][chunk].is_none() {
                        continue;
                    }
                    let window = value.offset..value.offset + value.length;
                    let touches = held.layout.fields.iter().any(|field| {
                        let bytes = field.bytes();
                        bytes.start < window.end
                            && bytes.end > window.start
                            && matches!(&field.encoding, Encoding::Encoded { group, .. }
                                if written.contains(group.as_str()))
                    });
                    if touches {
                        mapped[storage][chunk] = None;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let mut encoded_needs = BTreeSet::new();
        for (storage, held) in target.storages.iter().enumerate() {
            for (chunk, value) in held.chunks.iter().enumerate() {
                if let Some((source_rank, source_storage, source_chunk)) = mapped[storage][chunk] {
                    self.mappings.push(Mapping {
                        rank,
                        storage: storage as u32,
                        chunk: chunk as u32,
                        key: value.key,
                        source_rank,
                        source_storage,
                        source_chunk,
                    });
                    continue;
                }
                let window = value.offset..value.offset + value.length;
                for piece in held.layout.pieces(&window) {
                    match piece {
                        Piece::Data {
                            field,
                            logical,
                            local: _,
                        } => {
                            let at = FieldRef {
                                rank,
                                storage: storage as u32,
                                field: field as u32,
                            };
                            self.need(at, Form::Stored, logical);
                        }
                        Piece::Padding {
                            field,
                            local,
                            counts,
                        } => {
                            let field = &held.layout.fields[field];
                            self.fill(rank, storage as u32, field, &local, &counts);
                        }
                        Piece::Encoded { field, .. } => {
                            encoded_needs.insert((storage as u32, field as u32));
                        }
                    }
                }
            }
        }
        // Encoded fields are written by group.
        let mut groups: BTreeMap<&'r str, Vec<FieldRef>> = BTreeMap::new();
        for (storage, field) in encoded_needs {
            let at = FieldRef {
                rank,
                storage,
                field,
            };
            match &context.field(Side::Target, at).encoding {
                Encoding::Encoded { group, .. } => {
                    groups.entry(group.as_str()).or_default().push(at);
                }
                // An affine field whose strides interleave has no byte
                // window of its own: it is needed whole.
                Encoding::Affine => {
                    let field = context.field(Side::Target, at);
                    for logical in product_spans(&field.logical) {
                        self.need(at, Form::Stored, logical);
                    }
                }
            }
        }
        for (group, needed) in groups {
            self.encoded_group(rank, group, &needed)?;
        }
        Ok(())
    }

    /// The encoded groups of target rank `rank` some chunk of which is not
    /// mapped.
    fn written_groups(&self, rank: u32, mapped: &[Vec<Option<SourceChunk>>]) -> BTreeSet<&'r str> {
        let target = &self.context.request.target.ranks[rank as usize];
        let mut written = BTreeSet::new();
        for (storage, held) in target.storages.iter().enumerate() {
            for (chunk, value) in held.chunks.iter().enumerate() {
                if mapped[storage][chunk].is_some() {
                    continue;
                }
                let window = value.offset..value.offset + value.length;
                for field in &held.layout.fields {
                    let bytes = field.bytes();
                    if let Encoding::Encoded { group, .. } = &field.encoding
                        && bytes.start < window.end
                        && bytes.end > window.start
                    {
                        written.insert(group.as_str());
                    }
                }
            }
        }
        written
    }

    /// The staging space of `side`'s rank `rank` for `group`, created with
    /// its conversion on first use.
    fn staging(&mut self, side: Side, rank: u32, group: &str) -> Result<u32> {
        if let Some(&index) = self.staging_index.get(&(side, rank, group.to_owned())) {
            return Ok(index);
        }
        let world = match side {
            Side::Source => &self.context.request.source,
            Side::Target => &self.context.request.target,
        };
        let mut bytes = 0;
        let mut fields = Vec::new();
        let mut encoding = None;
        for (storage, held) in world.ranks[rank as usize].storages.iter().enumerate() {
            for field in &held.layout.fields {
                if let Encoding::Encoded {
                    name,
                    group: named,
                    canonical,
                } = &field.encoding
                    && named == group
                {
                    let canonical = canonical.as_ref().ok_or_else(|| {
                        Error::Invalid(format!(
                            "field {} of group {group} has no canonical form",
                            field.tensor
                        ))
                    })?;
                    let extent = canonical
                        .shape
                        .iter()
                        .zip(&canonical.strides)
                        .map(|(extent, stride)| extent.saturating_sub(1) * stride)
                        .sum::<u64>()
                        + 1;
                    bytes = bytes.max(canonical.offset + extent * field.element);
                    fields.push((storage as u32, field.tensor.clone()));
                    encoding = Some(name.clone());
                }
            }
        }
        let encoding = encoding.ok_or_else(|| {
            Error::Invalid(format!("rank {rank} holds no field of group {group}"))
        })?;
        let index = self.stagings.len() as u32;
        self.stagings.push(Staging {
            side,
            rank,
            group: group.to_owned(),
            bytes,
        });
        self.staging_index
            .insert((side, rank, group.to_owned()), index);
        let conversion = Conversion {
            staging: index,
            encoding,
            fields,
        };
        match side {
            Side::Source => self.encodes.insert(index, conversion),
            Side::Target => self.decodes.insert(index, conversion),
        };
        Ok(index)
    }

    /// The affine view of `at` on `side` in `form`.
    fn view(&mut self, side: Side, at: FieldRef, form: Form) -> Result<View<'r>> {
        let field = self.context.field(side, at);
        match (form, &field.encoding) {
            (Form::Stored, _) => Ok(View {
                space: Space::Storage {
                    storage: at.storage,
                },
                offset: field.offset,
                strides: &field.strides,
                element: field.element,
                logical: &field.logical,
            }),
            (
                Form::Canonical,
                Encoding::Encoded {
                    group,
                    canonical: Some(canonical),
                    ..
                },
            ) => {
                let staging = self.staging(side, at.rank, group)?;
                Ok(View {
                    space: Space::Staging { staging },
                    offset: canonical.offset,
                    strides: &canonical.strides,
                    element: field.element,
                    logical: &canonical.logical,
                })
            }
            (Form::Canonical, _) => Err(Error::Invalid(format!(
                "field {} has no canonical form",
                field.tensor
            ))),
        }
    }

    /// Record that target field `at` (read in `form`) needs the logical box
    /// `logical`, split into blocks each inside or outside every source.
    fn need(&mut self, at: FieldRef, form: Form, logical: Vec<Span>) {
        let context = self.context;
        let field = context.field(Side::Target, at);
        // Sources of the same tensor, element type and logical shape, each
        // read as stored when affine and in canonical form otherwise.
        let sources: Vec<(FieldRef, Form, &Logical)> = context
            .sources
            .get(field.tensor.as_str())
            .into_iter()
            .flatten()
            .filter_map(|&source| {
                let other = context.field(Side::Source, source);
                if other.dtype != field.dtype || other.logical.shape != field.logical.shape {
                    return None;
                }
                // An encoded source without a canonical form gives its
                // bytes whole only (`encoded_group`).
                match &other.encoding {
                    Encoding::Affine => Some((source, Form::Stored, &other.logical)),
                    Encoding::Encoded { canonical, .. } => canonical
                        .as_ref()
                        .map(|canonical| (source, Form::Canonical, &canonical.logical)),
                }
            })
            .collect();
        // Each dimension's atoms against every source segment.
        let per_dim: Vec<Vec<Span>> = logical
            .iter()
            .enumerate()
            .map(|(dim, span)| {
                let spans: Vec<Span> = sources
                    .iter()
                    .flat_map(|(_, _, held)| held.segments[dim].iter().map(|segment| segment.span))
                    .collect();
                atoms(span, &spans)
            })
            .collect();
        let mut choice = vec![0_usize; per_dim.len()];
        if per_dim.iter().any(Vec::is_empty) {
            return;
        }
        loop {
            let cell: Vec<Span> = choice
                .iter()
                .enumerate()
                .map(|(dim, index)| per_dim[dim][*index])
                .collect();
            let candidates: Candidates = sources
                .iter()
                .filter(|(_, _, held)| {
                    cell.iter().enumerate().all(|(dim, span)| {
                        held.segments[dim]
                            .iter()
                            .any(|segment| span.within(&segment.span))
                    })
                })
                .map(|(source, form, _)| (*source, *form))
                .collect();
            if candidates.is_empty() {
                self.missing.push(Missing {
                    rank: at.rank,
                    storage: at.storage,
                    tensor: field.tensor.clone(),
                    logical: cell,
                });
            } else {
                self.cells.push(Cell {
                    target: at,
                    form,
                    bytes: volume(&cell) * field.element,
                    what: Block::Region(cell),
                    candidates,
                    chosen: None,
                });
            }
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

    /// The sources holding exactly what encoded target field `at` holds,
    /// in the same encoding: their bytes are its bytes.
    fn same_sources(&self, at: FieldRef) -> Candidates {
        let context = self.context;
        let field = context.field(Side::Target, at);
        context
            .sources
            .get(field.tensor.as_str())
            .into_iter()
            .flatten()
            .filter(|source| {
                let other = context.field(Side::Source, **source);
                other.encoding == field.encoding
                    && other.dtype == field.dtype
                    && other.shape == field.shape
                    && other.strides == field.strides
                    && other.logical == field.logical
            })
            .map(|source| (*source, Form::Stored))
            .collect()
    }

    /// Record that the encoded group `group` of target rank `rank` is
    /// written, `needed` being its fields that lie in written chunks: each
    /// of them copied as it is when a source holds each in the same
    /// encoding, the whole group converted from its canonical form
    /// otherwise, as the conversion writes all of it.
    fn encoded_group(&mut self, rank: u32, group: &str, needed: &[FieldRef]) -> Result<()> {
        let context = self.context;
        let same: Option<Vec<(FieldRef, Candidates)>> = needed
            .iter()
            .map(|&at| {
                let same = self.same_sources(at);
                (!same.is_empty()).then_some((at, same))
            })
            .collect();
        if let Some(same) = same {
            for (at, candidates) in same {
                let bytes = context.field(Side::Target, at).bytes();
                self.cells.push(Cell {
                    target: at,
                    form: Form::Stored,
                    bytes: bytes.end - bytes.start,
                    what: Block::Bytes(0..bytes.end - bytes.start),
                    candidates,
                    chosen: None,
                });
            }
            return Ok(());
        }
        let target = &context.request.target.ranks[rank as usize];
        let convertible = target.storages.iter().all(|held| {
            held.layout
                .fields
                .iter()
                .all(|field| match &field.encoding {
                    Encoding::Encoded {
                        group: named,
                        canonical,
                        ..
                    } if named == group => canonical.is_some(),
                    _ => true,
                })
        });
        if !convertible {
            // Without a canonical form the fields no source holds as they
            // are stored are loaded.
            for &at in needed {
                let field = context.field(Side::Target, at);
                let candidates = self.same_sources(at);
                if candidates.is_empty() {
                    for logical in product_spans(&field.logical) {
                        self.missing.push(Missing {
                            rank,
                            storage: at.storage,
                            tensor: field.tensor.clone(),
                            logical,
                        });
                    }
                    continue;
                }
                let bytes = field.bytes();
                self.cells.push(Cell {
                    target: at,
                    form: Form::Stored,
                    bytes: bytes.end - bytes.start,
                    what: Block::Bytes(0..bytes.end - bytes.start),
                    candidates,
                    chosen: None,
                });
            }
            return Ok(());
        }
        // The group converts from canonical once every field of it is
        // there: need each field's canonical region.
        let staging = self.staging(Side::Target, rank, group)?;
        let Some(conversion) = self.decodes.get(&staging) else {
            return Err(Error::Invalid(format!("group {group} has no conversion")));
        };
        let fields = conversion.fields.clone();
        for (storage, tensor) in fields {
            let held = &target.storages[storage as usize].layout;
            let Some((index, member)) = held
                .fields
                .iter()
                .enumerate()
                .find(|(_, member)| member.tensor == tensor)
            else {
                continue;
            };
            let Encoding::Encoded {
                canonical: Some(canonical),
                ..
            } = &member.encoding
            else {
                continue;
            };
            let member_at = FieldRef {
                rank,
                storage,
                field: index as u32,
            };
            let boxes: Vec<Vec<Span>> = product_spans(&canonical.logical);
            for logical in boxes {
                self.need(member_at, Form::Canonical, logical);
            }
        }
        Ok(())
    }

    /// Zero the padding at local `local..local + counts` of `field`.
    fn fill(&mut self, rank: u32, storage: u32, field: &Field, local: &[u64], counts: &[u64]) {
        let strides = field.byte_strides();
        for (offset, length, rows, pitch) in
            runs_one_side(field.address(local), counts, &strides, field.element)
        {
            self.fills.push(Fill {
                rank,
                storage,
                offset,
                length,
                rows,
                pitch,
            });
        }
    }

    /// Choose every block's source, largest first, minimizing the latest
    /// link completion; returns the links' loads.
    fn choose_sources(&mut self) -> Result<Vec<LinkLoad>> {
        let context = self.context;
        let mut order: Vec<usize> = (0..self.cells.len()).collect();
        order.sort_by_key(|&index| std::cmp::Reverse(self.cells[index].bytes));
        let mut loads: BTreeMap<(Gpu, Gpu), u64> = BTreeMap::new();
        let mut busiest = 0.0_f64;
        for index in order {
            let cell = &self.cells[index];
            let target_gpu = self.gpus[cell.target.rank as usize];
            let mut best: Option<((f64, f64), (FieldRef, Form))> = None;
            for &(source, form) in &cell.candidates {
                let source_gpu = context.source_gpu(source.rank);
                let bandwidth = *context
                    .bandwidth
                    .get(&(source_gpu, target_gpu))
                    .filter(|bandwidth| **bandwidth > 0.0)
                    .ok_or_else(|| {
                        Error::Invalid(format!(
                            "no bandwidth is measured from GPU {} to GPU {}",
                            uuid(&source_gpu),
                            uuid(&target_gpu)
                        ))
                    })?;
                let load = loads.get(&(source_gpu, target_gpu)).copied().unwrap_or(0);
                let finish = (load + cell.bytes) as f64 / bandwidth;
                let score = (busiest.max(finish), finish);
                if best.as_ref().is_none_or(|(current, _)| score < *current) {
                    best = Some((score, (source, form)));
                }
            }
            let Some(((latest, _), chosen)) = best else {
                continue;
            };
            busiest = latest;
            let source_gpu = context.source_gpu(chosen.0.rank);
            *loads.entry((source_gpu, target_gpu)).or_default() += cell.bytes;
            self.cells[index].chosen = Some(chosen);
        }
        loads
            .into_iter()
            .map(|((source, target), bytes)| {
                let bandwidth = context
                    .bandwidth
                    .get(&(source, target))
                    .copied()
                    .unwrap_or(f64::INFINITY);
                Ok(LinkLoad {
                    seconds: bytes as f64 / bandwidth,
                    source,
                    target,
                    bytes,
                })
            })
            .collect()
    }

    /// The byte copies of every block from its chosen source.
    fn copies(&mut self) -> Result<Vec<Copy>> {
        let cells = std::mem::take(&mut self.cells);
        let mut copies = Vec::new();
        for cell in &cells {
            let Some((source, source_form)) = cell.chosen else {
                continue;
            };
            match &cell.what {
                Block::Bytes(range) => {
                    let target = self.context.field(Side::Target, cell.target);
                    let from = self.context.field(Side::Source, source);
                    copies.push(Copy {
                        source: Endpoint {
                            rank: source.rank,
                            space: Space::Storage {
                                storage: source.storage,
                            },
                            offset: from.offset + range.start,
                        },
                        target: Endpoint {
                            rank: cell.target.rank,
                            space: Space::Storage {
                                storage: cell.target.storage,
                            },
                            offset: target.offset + range.start,
                        },
                        length: range.end - range.start,
                        rows: 1,
                        source_pitch: 0,
                        target_pitch: 0,
                    });
                }
                Block::Region(logical) => {
                    let into = self.view(Side::Target, cell.target, cell.form)?;
                    let from = self.view(Side::Source, source, source_form)?;
                    region_copies(
                        logical,
                        (source.rank, &from),
                        (cell.target.rank, &into),
                        &mut copies,
                    )?;
                }
            }
        }
        Ok(merge_copies(copies))
    }
}

/// Every box of a part: the product of its segments.
fn product_spans(logical: &Logical) -> Vec<Vec<Span>> {
    let mut boxes = vec![Vec::new()];
    for segments in &logical.segments {
        boxes = boxes
            .into_iter()
            .flat_map(|prefix| {
                segments.iter().map(move |segment| {
                    let mut next = prefix.clone();
                    next.push(segment.span);
                    next
                })
            })
            .collect();
    }
    boxes
}

/// The copies that move the logical box `logical` from the source view to
/// the target view.
fn region_copies(
    logical: &[Span],
    (source_rank, from): (u32, &View<'_>),
    (target_rank, into): (u32, &View<'_>),
    copies: &mut Vec<Copy>,
) -> Result<()> {
    let mut dims = Vec::with_capacity(logical.len());
    let mut source_local = Vec::with_capacity(logical.len());
    let mut target_local = Vec::with_capacity(logical.len());
    for (dim, span) in logical.iter().enumerate() {
        let missing = || Error::Invalid(format!("a block is not held along dimension {dim}"));
        let (source_start, source_step) = from.locate(dim, span).ok_or_else(missing)?;
        let (target_start, target_step) = into.locate(dim, span).ok_or_else(missing)?;
        source_local.push(source_start);
        target_local.push(target_start);
        dims.push((
            span.count,
            source_step * from.strides[dim] * from.element,
            target_step * into.strides[dim] * into.element,
        ));
    }
    if from.element != into.element {
        return Err(Error::Invalid(
            "a block's source and target elements differ in size".into(),
        ));
    }
    let source_base = from.address(&source_local);
    let target_base = into.address(&target_local);
    for (offsets, length, rows, (source_pitch, target_pitch)) in
        runs_two_sides(source_base, target_base, &dims, from.element)
    {
        copies.push(Copy {
            source: Endpoint {
                rank: source_rank,
                space: from.space,
                offset: offsets.0,
            },
            target: Endpoint {
                rank: target_rank,
                space: into.space,
                offset: offsets.1,
            },
            length,
            rows,
            source_pitch,
            target_pitch,
        });
    }
    Ok(())
}

type Run = ((u64, u64), u64, u64, (u64, u64));

/// The strided runs covering a box whose dimension `d` has `dims[d] =
/// (count, source byte stride, target byte stride)`: the innermost
/// dimensions contiguous on both sides form a run, the next dimension its
/// rows, and the rest are enumerated.
fn runs_two_sides(source: u64, target: u64, dims: &[(u64, u64, u64)], element: u64) -> Vec<Run> {
    let mut ordered: Vec<(u64, u64, u64)> = dims
        .iter()
        .copied()
        .filter(|(count, _, _)| *count > 1)
        .collect();
    if dims.iter().any(|(count, _, _)| *count == 0) {
        return Vec::new();
    }
    ordered.sort_by_key(|&(_, source_stride, target_stride)| {
        (
            std::cmp::Reverse(target_stride),
            std::cmp::Reverse(source_stride),
        )
    });
    let mut length = element;
    while let Some(&(count, source_stride, target_stride)) = ordered.last() {
        if source_stride == length && target_stride == length {
            length *= count;
            ordered.pop();
        } else {
            break;
        }
    }
    let (rows, source_pitch, target_pitch) = ordered.pop().unwrap_or((1, 0, 0));
    let mut runs = Vec::new();
    let mut index = vec![0_u64; ordered.len()];
    loop {
        let (mut s, mut t) = (source, target);
        for (position, (_, source_stride, target_stride)) in ordered.iter().enumerate() {
            s += index[position] * source_stride;
            t += index[position] * target_stride;
        }
        runs.push(((s, t), length, rows, (source_pitch, target_pitch)));
        let mut dim = ordered.len();
        loop {
            if dim == 0 {
                return runs;
            }
            dim -= 1;
            index[dim] += 1;
            if index[dim] < ordered[dim].0 {
                break;
            }
            index[dim] = 0;
        }
    }
}

/// The strided runs covering a box of one side, as [`runs_two_sides`].
fn runs_one_side(
    base: u64,
    counts: &[u64],
    strides: &[u64],
    element: u64,
) -> Vec<(u64, u64, u64, u64)> {
    let dims: Vec<(u64, u64, u64)> = counts
        .iter()
        .zip(strides)
        .map(|(count, stride)| (*count, *stride, *stride))
        .collect();
    runs_two_sides(base, base, &dims, element)
        .into_iter()
        .map(|((offset, _), length, rows, (pitch, _))| (offset, length, rows, pitch))
        .collect()
}

/// Merge copies that continue one another into longer runs or more rows.
fn merge_copies(mut copies: Vec<Copy>) -> Vec<Copy> {
    copies.sort_by_key(|copy| {
        (
            copy.target.rank,
            copy.target.space,
            copy.source.rank,
            copy.source.space,
            copy.target.offset,
        )
    });
    let mut merged: Vec<Copy> = Vec::with_capacity(copies.len());
    for copy in copies {
        if let Some(last) = merged.last_mut()
            && last.target.rank == copy.target.rank
            && last.target.space == copy.target.space
            && last.source.rank == copy.source.rank
            && last.source.space == copy.source.space
        {
            // One row continuing another into one longer row.
            if last.rows == 1
                && copy.rows == 1
                && copy.target.offset == last.target.offset + last.length
                && copy.source.offset == last.source.offset + last.length
            {
                last.length += copy.length;
                continue;
            }
            // Rows continuing another's at the same pitches, both sides
            // advancing by at least a row.
            if copy.length == last.length
                && copy.source.offset > last.source.offset
                && copy.target.offset > last.target.offset
            {
                let (source_pitch, target_pitch) = if last.rows == 1 {
                    (
                        copy.source.offset - last.source.offset,
                        copy.target.offset - last.target.offset,
                    )
                } else {
                    (last.source_pitch, last.target_pitch)
                };
                let continues = source_pitch >= last.length
                    && target_pitch >= last.length
                    && copy.source.offset == last.source.offset + last.rows * source_pitch
                    && copy.target.offset == last.target.offset + last.rows * target_pitch
                    && (copy.rows == 1
                        || (copy.source_pitch == source_pitch
                            && copy.target_pitch == target_pitch));
                if continues {
                    last.source_pitch = source_pitch;
                    last.target_pitch = target_pitch;
                    last.rows += copy.rows;
                    continue;
                }
            }
        }
        merged.push(copy);
    }
    merged
}

/// Merge fills that continue one another.
fn merge_fills(mut fills: Vec<Fill>) -> Vec<Fill> {
    fills.sort_by_key(|fill| (fill.rank, fill.storage, fill.offset));
    let mut merged: Vec<Fill> = Vec::with_capacity(fills.len());
    for fill in fills {
        if let Some(last) = merged.last_mut()
            && last.rank == fill.rank
            && last.storage == fill.storage
            && last.rows == 1
            && fill.rows == 1
            && fill.offset == last.offset + last.length
        {
            last.length += fill.length;
            continue;
        }
        merged.push(fill);
    }
    merged
}
