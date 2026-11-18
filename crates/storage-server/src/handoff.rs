//! The copies of a handoff between two replicas: from the units a prefill replica's owner lent
//! to the units a decode replica's owner allocated, whatever the two replicas' tensor-parallel
//! layouts (`specs/kv-cache.md` §4.3).
//!
//! A KV page is split by its key and value heads: each rank of a replica holds `kv_heads`
//! consecutive heads of every layer from `kv_head_offset` on, and several ranks hold the same
//! heads when the group has more ranks than the layer has heads. A cell of a page (one layer's
//! keys or values) is `[block_size, kv_heads, head_dim]`, row-major, so the bytes of a range of
//! heads are one strided run of `block_size` rows. The copies read each head from one rank of
//! the lender that holds it and write it into every rank of the borrower that needs it: per
//! destination rank, layer and plane, one span per range of heads one source rank holds, a
//! pitched copy unless the range is the whole cell on both sides. Consecutive units of one
//! column are consecutive cells of its storage, so the spans of consecutive pages merge.
//!
//! Beside the planes a cell has its two written flags, copied from the rank that holds the
//! destination's first head; an FP8 cell's scale, which covers every head of the cell, copied
//! when one source rank holds all of the destination's heads and refused otherwise (no single
//! scale states two source cells' values); and the layer's index keys, which every rank holds
//! whole, copied from that same rank.

use std::collections::BTreeMap;

use storage::protocol::{PageLayout, Region, Span, UnitLayout, UnitRun, planes};

/// One rank of a side of a handoff: where it holds its part of the region's units, and how.
#[derive(Clone, Copy)]
pub(crate) struct Rank<'a> {
    /// The Storage Server of the rank's host; `None` for this host.
    pub(crate) endpoint: Option<&'a str>,
    pub(crate) runs: &'a [UnitRun],
    pub(crate) layout: &'a UnitLayout,
}

/// One page a handoff copies: its class and the units its block takes on each side, in page
/// order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Page {
    pub(crate) class: u8,
    pub(crate) source: Vec<u32>,
    pub(crate) destination: Vec<u32>,
}

/// A span of a handoff, with the hosts of its two ends (`None` for this host).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Placed {
    pub(crate) source: Option<String>,
    pub(crate) destination: Option<String>,
    pub(crate) span: Span,
}

/// Where one cell of a rank lies: its storage and the cell's first byte.
struct Cell {
    key: storage::Key,
    location: storage::Location,
    offset: u64,
}

/// The cell of `field` of unit `unit` in `runs`, which hold `bytes` bytes a cell.
fn cell(runs: &[UnitRun], field: &str, unit: u32, bytes: u64) -> Result<Cell, String> {
    let run = runs
        .iter()
        .find(|run| run.field == field && run.first <= unit && unit < run.first + run.count)
        .ok_or_else(|| format!("unit {unit}'s {field} lies in no run of its rank"))?;
    if run.bytes != bytes {
        return Err(format!(
            "{field} holds {} bytes a unit, where its layout states {bytes}",
            run.bytes
        ));
    }
    Ok(Cell {
        key: run.key,
        location: run.location,
        offset: run.offset + u64::from(unit - run.first) * run.bytes,
    })
}

/// A class's page layout on a rank.
fn class_of<'a>(rank: &Rank<'a>, class: u8) -> Option<&'a PageLayout> {
    rank.layout.classes.iter().find(|page| page.class == class)
}

/// The bytes of one plane cell of a page.
fn plane_bytes(page: &PageLayout) -> u64 {
    u64::from(page.block_size) * u64::from(page.kv_heads) * u64::from(page.head_bytes)
}

/// Where layer `layer` lies in a page of `page` on a rank of `columns` columns: the unit of the
/// page and its column; `None` when the rank holds no such layer.
fn layer_at(page: &PageLayout, columns: u32, layer: u32) -> Option<(usize, u32)> {
    let index = page.layers.iter().position(|held| *held == layer)? as u32;
    Some(((index / columns.max(1)) as usize, index % columns.max(1)))
}

/// The spans that copy `pages` from the lender's ranks `source` into the borrower's ranks
/// `destination`, by part.
///
/// # Errors
///
/// When the two layouts do not hold one content (another page size, head size, head count, a
/// layer or an FP8 representation only one side has), a destination head no source rank
/// holds, an FP8 destination cell whose heads two source cells hold, or a unit no run holds.
pub(crate) fn spans(
    source: &BTreeMap<u32, Rank<'_>>,
    destination: &BTreeMap<u32, Rank<'_>>,
    pages: &[Page],
) -> Result<Vec<Placed>, String> {
    let mut placed = Vec::new();
    let classes: std::collections::BTreeSet<u8> = pages.iter().map(|page| page.class).collect();
    for (part, to) in destination {
        for &class in &classes {
            let wanted = class_of(to, class)
                .ok_or_else(|| format!("rank {part} lays out no page of class {class}"))?;
            for &layer in &wanted.layers {
                for (plane, values) in planes::VALUES.iter().enumerate() {
                    // Each page's cell takes the same sequence of spans, which are laid out
                    // span by span over the pages so that consecutive units' spans meet.
                    let mut cells = Vec::new();
                    for page in pages.iter().filter(|page| page.class == class) {
                        let mut copies = Vec::new();
                        copy_cell(
                            (source, (*part, to)),
                            (class, layer, plane, values),
                            page,
                            &mut copies,
                        )?;
                        cells.push(copies);
                    }
                    let each = cells.first().map_or(0, Vec::len);
                    for index in 0..each {
                        placed.extend(cells.iter().filter_map(|copies| copies.get(index)).cloned());
                    }
                }
            }
        }
    }
    Ok(merge(placed))
}

/// The spans of one cell of one destination rank: layer `layer`'s plane `plane` (keys or
/// values, whose storage is `values`) of `page`, and the cell's flag, scale and index keys.
fn copy_cell(
    (source, (part, to)): (&BTreeMap<u32, Rank<'_>>, (u32, &Rank<'_>)),
    (class, layer, plane, values): (u8, u32, usize, &str),
    page: &Page,
    placed: &mut Vec<Placed>,
) -> Result<(), String> {
    let wanted =
        class_of(to, class).ok_or_else(|| format!("rank {part} holds no class {class}"))?;
    let (unit, column) = layer_at(wanted, to.layout.columns, layer)
        .ok_or_else(|| format!("rank {part} holds no layer {layer}"))?;
    let destination_unit = *page
        .destination
        .get(unit)
        .ok_or_else(|| format!("the destination page holds no unit {unit}"))?;
    let destination_bytes = plane_bytes(wanted);
    let row = u64::from(wanted.kv_heads) * u64::from(wanted.head_bytes);

    // The source ranks that hold the layer, each with its page layout and where the layer lies.
    let holders: Vec<(u32, &Rank<'_>, &PageLayout, u32, u32)> = source
        .iter()
        .filter_map(|(held, rank)| {
            let layout = class_of(rank, class)?;
            let (unit, column) = layer_at(layout, rank.layout.columns, layer)?;
            let source_unit = *page.source.get(unit)?;
            Some((*held, rank, layout, source_unit, column))
        })
        .collect();
    let holder = |head: u32| {
        holders.iter().find(|(_, _, layout, ..)| {
            layout.kv_head_offset <= head && head < layout.kv_head_offset + layout.kv_heads
        })
    };
    for (_, _, layout, ..) in &holders {
        let same = layout.block_size == wanted.block_size
            && layout.head_bytes == wanted.head_bytes
            && layout.total_kv_heads == wanted.total_kv_heads
            && layout.quantized == wanted.quantized;
        if !same {
            return Err(format!(
                "the replicas hold class {class} in other representations: pages of {} and {} \
                 tokens, heads of {} and {} bytes, {} and {} heads, FP8 {} and {}",
                layout.block_size,
                wanted.block_size,
                layout.head_bytes,
                wanted.head_bytes,
                layout.total_kv_heads,
                wanted.total_kv_heads,
                layout.quantized,
                wanted.quantized
            ));
        }
    }

    let destination_field = planes::field(values, column);
    let destination_cell = cell(
        to.runs,
        &destination_field,
        destination_unit,
        destination_bytes,
    )?;
    let end = wanted.kv_head_offset + wanted.kv_heads;
    let mut head = wanted.kv_head_offset;
    while head < end {
        let (_, from, layout, source_unit, source_column) = holder(head)
            .ok_or_else(|| format!("no rank of the lender holds head {head} of layer {layer}"))?;
        let last = end.min(layout.kv_head_offset + layout.kv_heads);
        let heads = u64::from(last - head);
        let source_row = u64::from(layout.kv_heads) * u64::from(layout.head_bytes);
        let source_field = planes::field(values, *source_column);
        let source_cell = cell(from.runs, &source_field, *source_unit, plane_bytes(layout))?;
        let whole = heads == u64::from(layout.kv_heads) && heads == u64::from(wanted.kv_heads);
        let source_offset = source_cell.offset
            + u64::from(head - layout.kv_head_offset) * u64::from(layout.head_bytes);
        let destination_offset = destination_cell.offset
            + u64::from(head - wanted.kv_head_offset) * u64::from(wanted.head_bytes);
        let span = if whole {
            Span {
                source: tensor(&source_cell),
                source_offset,
                destination: tensor(&destination_cell),
                destination_offset,
                bytes: destination_bytes,
                rows: 1,
                source_pitch: 0,
                destination_pitch: 0,
            }
        } else {
            Span {
                source: tensor(&source_cell),
                source_offset,
                destination: tensor(&destination_cell),
                destination_offset,
                bytes: heads * u64::from(wanted.head_bytes),
                rows: u64::from(wanted.block_size),
                source_pitch: source_row,
                destination_pitch: row,
            }
        };
        placed.push(Placed {
            source: from.endpoint.map(str::to_owned),
            destination: to.endpoint.map(str::to_owned),
            span,
        });
        head = last;
    }

    // The cell's flag, scale and index keys come from the rank that holds its first head.
    let (_, from, layout, source_unit, source_column) = holder(wanted.kv_head_offset)
        .ok_or_else(|| format!("no rank of the lender holds layer {layer}"))?;
    let mut whole_cell = |name: &str, bytes: u64, source_bytes: u64| -> Result<(), String> {
        let source_cell = cell(
            from.runs,
            &planes::field(name, *source_column),
            *source_unit,
            source_bytes,
        )?;
        let destination_cell = cell(
            to.runs,
            &planes::field(name, column),
            destination_unit,
            bytes,
        )?;
        placed.push(Placed {
            source: from.endpoint.map(str::to_owned),
            destination: to.endpoint.map(str::to_owned),
            span: Span {
                source: tensor(&source_cell),
                source_offset: source_cell.offset,
                destination: tensor(&destination_cell),
                destination_offset: destination_cell.offset,
                bytes,
                rows: 1,
                source_pitch: 0,
                destination_pitch: 0,
            },
        });
        Ok(())
    };
    whole_cell(planes::FLAGS[plane], 1, 1)?;
    if wanted.quantized {
        let inside = layout.kv_head_offset <= wanted.kv_head_offset
            && end <= layout.kv_head_offset + layout.kv_heads;
        if !inside {
            return Err(format!(
                "the FP8 cell of layer {layer} on rank {part} holds heads of two cells of the \
                 lender, whose scales differ"
            ));
        }
        whole_cell(planes::SCALES[plane], 4, 4)?;
    }
    // The index keys are one per cell, not per plane: copied with the keys' plane.
    if plane == 0 && wanted.index_bytes > 0 {
        if layout.index_bytes != wanted.index_bytes {
            return Err(format!(
                "the replicas keep {} and {} bytes of index keys a page of layer {layer}",
                layout.index_bytes, wanted.index_bytes
            ));
        }
        whole_cell(planes::INDEX_KEYS, wanted.index_bytes, layout.index_bytes)?;
    }
    Ok(())
}

/// A cell's storage as a span names it.
fn tensor(cell: &Cell) -> Region {
    Region::Tensor {
        key: cell.key,
        at: Some(cell.location),
    }
}

/// `placed` with each span that continues the one before it joined to it: the same storages
/// and hosts, the next rows at the same pitches, or the next bytes of a plain copy.
fn merge(placed: Vec<Placed>) -> Vec<Placed> {
    let mut merged: Vec<Placed> = Vec::with_capacity(placed.len());
    for next in placed {
        if let Some(last) = merged.last_mut()
            && last.source == next.source
            && last.destination == next.destination
            && last.span.source == next.span.source
            && last.span.destination == next.span.destination
        {
            let (span, following) = (&mut last.span, &next.span);
            let plain = span.rows == 1 && following.rows == 1 && span.source_pitch == 0;
            if plain
                && following.source_pitch == 0
                && following.source_offset == span.source_offset + span.bytes
                && following.destination_offset == span.destination_offset + span.bytes
            {
                span.bytes += following.bytes;
                continue;
            }
            let pitched = span.source_pitch > 0
                && span.bytes == following.bytes
                && span.source_pitch == following.source_pitch
                && span.destination_pitch == following.destination_pitch;
            if pitched
                && following.source_offset == span.source_offset + span.rows * span.source_pitch
                && following.destination_offset
                    == span.destination_offset + span.rows * span.destination_pitch
            {
                span.rows += following.rows;
                continue;
            }
        }
        merged.push(next);
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use storage::{Key, Location};

    /// The demonstration's shapes (`specs/kv-cache.md` §4.3, Qwen3-30B-A3B): four key/value
    /// heads a layer, two layers here, a head of 128 BF16 elements, pages of 64 tokens.
    const HEADS: u32 = 4;
    const LAYERS: u32 = 2;
    const HEAD_BYTES: u32 = 256;
    const BLOCK: u32 = 64;
    /// The units of each rank's region: unit 0 is padding.
    const UNITS: u32 = 8;

    /// The heads a rank of a group of `ranks` holds: a share of the heads, or one head held by
    /// several ranks when the group has more ranks than heads.
    fn heads(rank: u32, ranks: u32) -> (u32, u32) {
        if ranks <= HEADS {
            let each = HEADS / ranks;
            (rank * each, each)
        } else {
            (rank / (ranks / HEADS), 1)
        }
    }

    fn layout(rank: u32, ranks: u32, quantized: bool) -> UnitLayout {
        let (offset, count) = heads(rank, ranks);
        UnitLayout {
            columns: LAYERS,
            classes: vec![PageLayout {
                class: 0,
                block_size: BLOCK,
                layers: (0..LAYERS).collect(),
                kv_heads: count,
                kv_head_offset: offset,
                total_kv_heads: HEADS,
                head_bytes: HEAD_BYTES,
                quantized,
                index_bytes: 0,
            }],
        }
    }

    /// One storage per field of a rank, keyed by the rank's side and part, `[columns, units,
    /// cell]`.
    fn runs(side: u8, rank: u32, ranks: u32, quantized: bool) -> Vec<UnitRun> {
        let (_, count) = heads(rank, ranks);
        let plane = u64::from(BLOCK * count * HEAD_BYTES);
        let mut fields: Vec<(&str, u64)> = vec![
            (planes::VALUES[0], plane),
            (planes::VALUES[1], plane),
            (planes::FLAGS[0], 1),
            (planes::FLAGS[1], 1),
        ];
        if quantized {
            fields.extend([(planes::SCALES[0], 4), (planes::SCALES[1], 4)]);
        }
        let mut runs = Vec::new();
        for (index, (name, bytes)) in fields.into_iter().enumerate() {
            for column in 0..LAYERS {
                runs.push(UnitRun {
                    field: planes::field(name, column),
                    key: key(side, rank, index as u8),
                    location: Location::Host { numa_node: None },
                    first: 0,
                    count: UNITS,
                    offset: u64::from(column) * u64::from(UNITS) * bytes,
                    bytes,
                });
            }
        }
        runs
    }

    fn key(side: u8, rank: u32, field: u8) -> Key {
        let mut key = [0; 32];
        key[0] = side;
        key[1] = rank as u8;
        key[2] = field;
        Key(key)
    }

    /// The bytes of every storage of a side, by key.
    type Memory = BTreeMap<Key, Vec<u8>>;

    fn storages(side: u8, ranks: u32, quantized: bool) -> Memory {
        let mut memory = Memory::new();
        for rank in 0..ranks {
            for run in runs(side, rank, ranks, quantized) {
                let end = (run.offset + u64::from(run.count) * run.bytes) as usize;
                let storage = memory.entry(run.key).or_default();
                storage.resize(storage.len().max(end), 0);
            }
        }
        memory
    }

    /// The byte of head `head`, token `token`, element byte `byte` of layer `layer`'s plane
    /// `plane` of page `page`: the logical content every layout holds a part of.
    fn content(page: u32, layer: u32, plane: u32, token: u32, head: u32, byte: u32) -> u8 {
        (page * 131 + layer * 31 + plane * 17 + token * 7 + head * 5 + byte) as u8
    }

    /// Writes `pages` (each a unit of the side) as a group of `ranks` lays them out.
    fn write(memory: &mut Memory, side: u8, ranks: u32, pages: &[u32], quantized: bool) {
        for rank in 0..ranks {
            let (offset, count) = heads(rank, ranks);
            let held = runs(side, rank, ranks, quantized);
            for (page, unit) in pages.iter().enumerate() {
                for layer in 0..LAYERS {
                    for plane in 0..2 {
                        let field = planes::field(planes::VALUES[plane], layer);
                        let bytes = u64::from(BLOCK * count * HEAD_BYTES);
                        let at = cell(&held, &field, *unit, bytes).unwrap();
                        let storage = memory.get_mut(&at.key).unwrap();
                        for token in 0..BLOCK {
                            for local in 0..count {
                                for byte in 0..HEAD_BYTES {
                                    let index = at.offset as usize
                                        + ((token * count + local) * HEAD_BYTES + byte) as usize;
                                    storage[index] = content(
                                        page as u32,
                                        layer,
                                        plane as u32,
                                        token,
                                        offset + local,
                                        byte,
                                    );
                                }
                            }
                        }
                        let flag = planes::field(planes::FLAGS[plane], layer);
                        let at = cell(&held, &flag, *unit, 1).unwrap();
                        memory.get_mut(&at.key).unwrap()[at.offset as usize] = 1;
                        if quantized {
                            let scale = planes::field(planes::SCALES[plane], layer);
                            let at = cell(&held, &scale, *unit, 4).unwrap();
                            let storage = memory.get_mut(&at.key).unwrap();
                            storage[at.offset as usize] = 9 + page as u8;
                        }
                    }
                }
            }
        }
    }

    /// Runs `spans` as the Transfer Server runs them: row by row at the pitches.
    fn run(spans: &[Placed], from: &Memory, into: &mut Memory) {
        for placed in spans {
            let span = &placed.span;
            let (Region::Tensor { key: source, .. }, Region::Tensor { key: target, .. }) =
                (&span.source, &span.destination)
            else {
                panic!("a handoff's span names tensors");
            };
            for row in 0..span.rows {
                let start = (span.source_offset + row * span.source_pitch) as usize;
                let bytes = from[source][start..start + span.bytes as usize].to_vec();
                let at = (span.destination_offset + row * span.destination_pitch) as usize;
                into.get_mut(target).unwrap()[at..at + bytes.len()].copy_from_slice(&bytes);
            }
        }
    }

    fn side<'a>(
        endpoint: Option<&'a str>,
        runs: &'a [Vec<UnitRun>],
        layouts: &'a [UnitLayout],
    ) -> BTreeMap<u32, Rank<'a>> {
        (0..runs.len() as u32)
            .map(|part| {
                let index = part as usize;
                (
                    part,
                    Rank {
                        endpoint,
                        runs: &runs[index],
                        layout: &layouts[index],
                    },
                )
            })
            .collect()
    }

    /// Lends three pages laid out by `from` ranks into the units a group of `to` ranks
    /// allocated, and compares what the copies wrote with the same pages written directly in
    /// the destination's layout. Returns the spans.
    fn hand_off(from: u32, to: u32, quantized: bool) -> Vec<Placed> {
        let (source_pages, destination_pages) = ([2, 3, 6], [4, 5, 1]);
        let mut lent = storages(1, from, quantized);
        write(&mut lent, 1, from, &source_pages, quantized);

        let source_runs: Vec<_> = (0..from)
            .map(|rank| runs(1, rank, from, quantized))
            .collect();
        let source_layouts: Vec<_> = (0..from)
            .map(|rank| layout(rank, from, quantized))
            .collect();
        let destination_runs: Vec<_> = (0..to).map(|rank| runs(2, rank, to, quantized)).collect();
        let destination_layouts: Vec<_> = (0..to).map(|rank| layout(rank, to, quantized)).collect();
        let pages: Vec<Page> = source_pages
            .iter()
            .zip(destination_pages)
            .map(|(source, destination)| Page {
                class: 0,
                source: vec![*source],
                destination: vec![destination],
            })
            .collect();
        let placed = spans(
            &side(Some("lender:1"), &source_runs, &source_layouts),
            &side(None, &destination_runs, &destination_layouts),
            &pages,
        )
        .unwrap();

        let mut imported = storages(2, to, quantized);
        run(&placed, &lent, &mut imported);
        let mut expected = storages(2, to, quantized);
        write(&mut expected, 2, to, &destination_pages, quantized);
        assert_eq!(
            imported, expected,
            "prefill tp={from} into decode tp={to}: the copies write the destination's layout"
        );
        placed
    }

    /// The demonstration's four pairs of layouts: each destination rank ends up with the pages
    /// as its own layout writes them, heads gathered from several source ranks (2 into 4: half
    /// of one source cell), merged into one destination cell (4 into 1), repeated over the
    /// ranks that hold one head (1 into 8), and every combination of them.
    #[test]
    fn a_handoff_reshards_each_page_s_heads_between_two_tensor_parallel_layouts() {
        for (from, to) in [(2, 4), (4, 1), (1, 8), (8, 2), (2, 2)] {
            let placed = hand_off(from, to, false);
            assert!(
                placed
                    .iter()
                    .all(|span| span.source.as_deref() == Some("lender:1"))
            );
            assert!(placed.iter().all(|span| span.destination.is_none()));
        }
    }

    /// Between two equal layouts every cell is one plain copy, and the copies of consecutive
    /// units join: pages 2 and 3 of the source land in units 4 and 5, one span per field and
    /// column for them, beside the third page's own.
    #[test]
    fn equal_layouts_copy_whole_cells_and_join_consecutive_units() {
        let placed = hand_off(2, 2, false);
        assert!(placed.iter().all(|span| span.span.rows == 1));
        let planes = placed
            .iter()
            .filter(|span| span.span.bytes >= u64::from(BLOCK * HEAD_BYTES));
        // Per rank, layer and plane: the two joined pages and the third.
        assert_eq!(planes.count(), 2 * LAYERS as usize * 2 * 2);
    }

    /// An FP8 page's scale states its cell's every head: a destination cell whose heads one
    /// source cell holds takes its scale, one whose heads two source cells hold is refused.
    #[test]
    fn an_fp8_cell_takes_its_scale_from_the_one_source_cell_holding_its_heads() {
        hand_off(1, 4, true);
        hand_off(2, 4, true);
        let source_runs: Vec<_> = (0..4).map(|rank| runs(1, rank, 4, true)).collect();
        let source_layouts: Vec<_> = (0..4).map(|rank| layout(rank, 4, true)).collect();
        let destination_runs = vec![runs(2, 0, 1, true)];
        let destination_layouts = vec![layout(0, 1, true)];
        let page = Page {
            class: 0,
            source: vec![1],
            destination: vec![1],
        };
        let refused = spans(
            &side(None, &source_runs, &source_layouts),
            &side(None, &destination_runs, &destination_layouts),
            &[page],
        );
        assert!(refused.unwrap_err().contains("scales differ"));
    }

    /// Pages of other sizes hold other tokens: no copy between them is made.
    #[test]
    fn layouts_of_other_page_sizes_are_refused() {
        let source_runs = vec![runs(1, 0, 1, false)];
        let mut other = layout(0, 1, false);
        other.classes[0].block_size = 32;
        let source_layouts = vec![other];
        let destination_runs = vec![runs(2, 0, 1, false)];
        let destination_layouts = vec![layout(0, 1, false)];
        let page = Page {
            class: 0,
            source: vec![1],
            destination: vec![1],
        };
        let refused = spans(
            &side(None, &source_runs, &source_layouts),
            &side(None, &destination_runs, &destination_layouts),
            &[page],
        );
        assert!(refused.unwrap_err().contains("other representations"));
    }
}
