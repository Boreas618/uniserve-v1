//! Borrow cache pages in their numerical layout for the shared copy planner.

use ::storage::manifest::cache::Page;
use ::storage::manifest::{
    Chunk, Encoding, Field, Logical, Rank, Segment, Span, Storage, StorageLayout,
};

use super::planes::CachePlanes;
use crate::storage::retained::Retained;
use crate::{Error, Result};

impl CachePlanes {
    /// Describe complete pages without copying or registering their backing.
    /// The scheduler owns page leases on both sides of the transfer; callers
    /// publish destinations only after the Transfer Server reports success.
    pub(crate) fn manifest(
        &self,
        pages: &[Page],
        retained: &Retained,
        resident: bool,
    ) -> Result<Rank> {
        let gpu = self
            .device()
            .ok_or_else(|| Error::Invalid("cache transfers require a device pool".into()))?
            .uuid()?
            .0;
        let plan = self.plan();
        let mut storages = Vec::new();
        for page in pages {
            if page.worker_group != retained.scope().group {
                continue;
            }
            let identity = format!("{}/{}", page.request.0, page.request.1);
            let group = plan
                .groups
                .get(page.group as usize)
                .ok_or_else(|| Error::Invalid(format!("no cache group {}", page.group)))?;
            if page.units.len() != group.units_per_page as usize || page.tokens != group.block_size
            {
                return Err(Error::Invalid(
                    "a cache transfer borrows complete logical pages".into(),
                ));
            }
            if group.state {
                // The numerical library declares the packed checkpoint's
                // affine fields. Plane boundaries may bisect a field; the
                // shared planner splits its spans at those chunk boundaries.
                let half = group.units_per_page * plan.columns;
                for checkpoint in &plan.checkpoints {
                    let mut chunks = Vec::new();
                    for index in 0..checkpoint.planes {
                        let position = checkpoint.first_plane + index;
                        let field = (position / half) as usize;
                        let slot = position % half;
                        let unit = page.units[(slot / plan.columns) as usize];
                        let mut chunk = self.retained_region(
                            retained,
                            field,
                            slot % plan.columns,
                            unit,
                            "values",
                            resident,
                        )?;
                        chunk.offset = u64::from(index) * plan.plane_bytes;
                        chunks.push(chunk);
                    }
                    let mut fields = checkpoint.fields.clone();
                    for field in &mut fields {
                        field.tensor =
                            format!("{}/{}/{}", identity, page.start + page.tokens, field.tensor);
                    }
                    storages.push(Storage {
                        layout: StorageLayout {
                            bytes: u64::from(checkpoint.planes) * plan.plane_bytes,
                            fields,
                        },
                        chunks,
                    });
                }
                continue;
            }
            for (layer, name) in group.layers.iter().enumerate() {
                let span = (plan.columns / group.layers.len() as u32).max(1);
                let heads = u64::from(group.num_kv_heads);
                let dim = u64::from(group.head_dim);
                let rows = u64::from(page.tokens);
                for (field, kind) in super::planes::FIELDS.iter().enumerate() {
                    let tensor = format!("{identity}/{name}/{kind}");
                    let logical = Logical {
                        shape: vec![
                            u64::from(page.total_tokens),
                            u64::from(group.total_kv_heads),
                            dim,
                        ],
                        dims: vec!["token".into(), "head".into(), "dim".into()],
                        units: vec![1, 1, 0],
                        segments: vec![
                            vec![Segment {
                                span: Span::range(u64::from(page.start), rows),
                                local: 0,
                            }],
                            vec![Segment {
                                span: Span::range(u64::from(group.kv_head_offset), heads),
                                local: 0,
                            }],
                            vec![Segment {
                                span: Span::range(0, dim),
                                local: 0,
                            }],
                        ],
                    };
                    let encoding = if plan.quantized {
                        Encoding::Encoded {
                            name: format!("unit-fp8/{rows}/{heads}/{dim}"),
                            group: tensor.clone(),
                            canonical: None,
                        }
                    } else {
                        Encoding::Affine
                    };
                    let bytes = rows * group.row_bytes();
                    let mut chunks = Vec::new();
                    for part in 0..span {
                        let position = layer as u32 * span + part;
                        let column = position % plan.columns;
                        let unit = page.units[(position / plan.columns) as usize];
                        let mut chunk = self
                            .retained_region(retained, field, column, unit, "values", resident)?;
                        chunk.offset = u64::from(part) * plan.plane_bytes;
                        chunk.length = (bytes - chunk.offset).min(plan.plane_bytes);
                        chunks.push(chunk);
                    }
                    storages.push(Storage {
                        layout: StorageLayout {
                            bytes,
                            fields: vec![Field {
                                tensor,
                                source: String::new(),
                                dtype: group.dtype.name().into(),
                                element: group.dtype.bytes() as u64,
                                offset: 0,
                                shape: vec![rows, heads, dim],
                                strides: vec![heads * dim, dim, 1],
                                logical,
                                encoding,
                            }],
                        },
                        chunks,
                    });
                    // A scale belongs to the whole encoded page, so its name
                    // includes both the token and head range. Changed ranges
                    // require recomputation instead of rescaling old FP8 bytes.
                    if plan.quantized {
                        for part in 0..span {
                            let position = layer as u32 * span + part;
                            let column = position % plan.columns;
                            let unit = page.units[(position / plan.columns) as usize];
                            let chunk = self.retained_region(
                                retained, field, column, unit, "scale", resident,
                            )?;
                            let tensor = format!(
                                "{}/{name}/{kind}/scale/{}/{}/{}/{heads}/{part}/{span}",
                                identity, page.start, page.tokens, group.kv_head_offset
                            );
                            storages.push(raw(tensor, chunk));
                        }
                    }
                }
            }
        }
        Ok(Rank {
            gpu: Some(gpu),
            storages,
        })
    }
}

fn raw(tensor: String, chunk: Chunk) -> Storage {
    let bytes = chunk.length;
    let logical = Logical {
        shape: vec![bytes],
        dims: vec!["byte".into()],
        units: vec![0],
        segments: vec![vec![Segment {
            span: Span::range(0, bytes),
            local: 0,
        }]],
    };
    Storage {
        layout: StorageLayout {
            bytes,
            fields: vec![Field {
                tensor: tensor.clone(),
                source: String::new(),
                dtype: "uint8".into(),
                element: 1,
                offset: 0,
                shape: vec![bytes],
                strides: vec![1],
                logical,
                encoding: Encoding::Encoded {
                    name: "unit-fp8-scale".into(),
                    group: tensor.clone(),
                    canonical: None,
                },
            }],
        },
        chunks: vec![chunk],
    }
}
