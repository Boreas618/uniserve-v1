//! The two tables as Arrow record batches and DataFusion table providers.
//!
//! A scan copies the rows it selects out of the store under the store's
//! lock, into one record batch; DataFusion runs the query over that copy.
//! Equality on `stats.name` or `events.kind` (`=` or `IN`) is pushed into
//! the scan, so a query of one metric or one kind copies only its rows.

use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock};

use arrow::array::{
    Array, ArrayRef, Float64Array, ListArray, MapBuilder, RecordBatch, StringBuilder, StructArray,
    TimestampNanosecondArray, UInt64Array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit};
use arrow::error::ArrowError;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{
    BinaryExpr, Expr, Operator, TableProviderFilterPushDown, TableType,
};
use datafusion::physical_plan::ExecutionPlan;

use super::events::Event;
use super::stats::Row;
use super::{Labels, Store, nanos};

/// The time zone of the `time` columns.
const ZONE: &str = "UTC";

/// The type of the `labels` and `fields` columns: a map of strings to
/// strings, as `MapBuilder` builds it.
static MAP: LazyLock<DataType> = LazyLock::new(|| {
    let mut builder = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    builder.finish().data_type().clone()
});

/// The fields of one bucket of a histogram: its interval and observation
/// count.
static BUCKET: LazyLock<Fields> = LazyLock::new(|| {
    Fields::from(vec![
        Field::new("lower", DataType::Float64, false),
        Field::new("upper", DataType::Float64, false),
        Field::new("count", DataType::UInt64, false),
    ])
});

/// The fields of the `histogram` column.
static HISTOGRAM: LazyLock<Fields> = LazyLock::new(|| {
    Fields::from(vec![
        Field::new("count", DataType::UInt64, false),
        Field::new("sum", DataType::Float64, false),
        Field::new("min", DataType::Float64, false),
        Field::new("max", DataType::Float64, false),
        Field::new("buckets", DataType::List(bucket_item()), false),
    ])
});

fn bucket_item() -> Arc<Field> {
    Arc::new(Field::new("item", DataType::Struct(BUCKET.clone()), false))
}

fn time() -> DataType {
    DataType::Timestamp(TimeUnit::Nanosecond, Some(ZONE.into()))
}

/// The schema of the `stats` table.
pub static STATS: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("time", time(), false),
        Field::new("name", DataType::Utf8, false),
        Field::new("labels", MAP.clone(), false),
        Field::new("value", DataType::Float64, false),
        Field::new("histogram", DataType::Struct(HISTOGRAM.clone()), true),
        Field::new("revision", DataType::UInt64, false),
    ]))
});

/// The schema of the `events` table.
pub static EVENTS: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("time", time(), false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("labels", MAP.clone(), false),
        Field::new("fields", MAP.clone(), false),
        Field::new("revision", DataType::UInt64, false),
    ]))
});

type Map = MapBuilder<StringBuilder, StringBuilder>;

fn map() -> Map {
    MapBuilder::new(None, StringBuilder::new(), StringBuilder::new())
}

fn append<'a>(
    map: &mut Map,
    entries: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<(), ArrowError> {
    for (key, value) in entries {
        map.keys().append_value(key);
        map.values().append_value(value);
    }
    map.append(true)
}

fn labels(map: &mut Map, labels: &Labels) -> Result<(), ArrowError> {
    append(
        map,
        labels
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str())),
    )
}

/// Builds the `stats` batch of the rows in order.
pub(crate) fn stats(rows: &[Row<'_>]) -> Result<RecordBatch, ArrowError> {
    let mut names = StringBuilder::new();
    let mut maps = map();

    // The `histogram` struct column, built column by column: a row
    // without a histogram is null, its children zero.
    let mut valid = Vec::with_capacity(rows.len());
    let mut counts = Vec::with_capacity(rows.len());
    let mut sums = Vec::with_capacity(rows.len());
    let mut mins = Vec::with_capacity(rows.len());
    let mut maxes = Vec::with_capacity(rows.len());
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    let (mut lowers, mut uppers, mut buckets) = (Vec::new(), Vec::new(), Vec::new());
    offsets.push(0i32);

    for row in rows {
        names.append_value(row.name);
        labels(&mut maps, row.labels)?;

        let histogram = row.bucket.value.histogram();
        valid.push(histogram.is_some());
        counts.push(histogram.map_or(0, |histogram| histogram.count()));
        sums.push(histogram.map_or(0.0, |histogram| histogram.sum()));
        mins.push(
            histogram
                .and_then(|histogram| histogram.min())
                .unwrap_or(0.0),
        );
        maxes.push(
            histogram
                .and_then(|histogram| histogram.max())
                .unwrap_or(0.0),
        );
        for (lower, upper, count) in histogram
            .into_iter()
            .flat_map(|histogram| histogram.buckets())
        {
            lowers.push(lower);
            uppers.push(upper);
            buckets.push(count);
        }
        let offset = i32::try_from(lowers.len())
            .map_err(|_| ArrowError::ComputeError("histogram buckets overflow a list".into()))?;
        offsets.push(offset);
    }

    let bucket = StructArray::try_new(
        BUCKET.clone(),
        vec![
            Arc::new(Float64Array::from(lowers)),
            Arc::new(Float64Array::from(uppers)),
            Arc::new(UInt64Array::from(buckets)),
        ],
        None,
    )?;
    let list = ListArray::try_new(
        bucket_item(),
        OffsetBuffer::new(offsets.into()),
        Arc::new(bucket),
        None,
    )?;
    let histogram = StructArray::try_new(
        HISTOGRAM.clone(),
        vec![
            Arc::new(UInt64Array::from(counts)),
            Arc::new(Float64Array::from(sums)),
            Arc::new(Float64Array::from(mins)),
            Arc::new(Float64Array::from(maxes)),
            Arc::new(list),
        ],
        Some(NullBuffer::from(valid)),
    )?;

    let columns: Vec<ArrayRef> = vec![
        Arc::new(
            TimestampNanosecondArray::from_iter_values(rows.iter().map(|row| row.bucket.start))
                .with_timezone(ZONE),
        ),
        Arc::new(names.finish()),
        Arc::new(maps.finish()),
        Arc::new(Float64Array::from_iter_values(
            rows.iter().map(|row| row.bucket.value.number()),
        )),
        Arc::new(histogram),
        Arc::new(UInt64Array::from_iter_values(
            rows.iter().map(|row| row.revision),
        )),
    ];

    RecordBatch::try_new(STATS.clone(), columns)
}

/// Builds the `events` batch of the events in order. A string field is its
/// text; any other field is its JSON.
pub(crate) fn events<'a>(rows: impl Iterator<Item = &'a Event>) -> Result<RecordBatch, ArrowError> {
    let mut times = Vec::new();
    let mut kinds = StringBuilder::new();
    let mut label_maps = map();
    let mut field_maps = map();
    let mut revisions = Vec::new();

    for event in rows {
        times.push(nanos(event.time));
        kinds.append_value(event.kind.as_str());
        labels(&mut label_maps, &event.labels)?;

        let texts: Vec<(&str, String)> = event
            .fields
            .iter()
            .map(|(key, value)| match value {
                serde_json::Value::String(text) => (key.as_str(), text.clone()),
                value => (key.as_str(), value.to_string()),
            })
            .collect();
        append(
            &mut field_maps,
            texts.iter().map(|(key, text)| (*key, text.as_str())),
        )?;

        revisions.push(event.revision);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampNanosecondArray::from(times).with_timezone(ZONE)),
        Arc::new(kinds.finish()),
        Arc::new(label_maps.finish()),
        Arc::new(field_maps.finish()),
        Arc::new(UInt64Array::from(revisions)),
    ];

    RecordBatch::try_new(EVENTS.clone(), columns)
}

/// The string values a filter requires `column` to equal: `column = 'a'`
/// or `column IN ('a', 'b')`.
fn equality(filter: &Expr, column: &str) -> Option<Vec<String>> {
    let text = |expr: &Expr| match expr {
        Expr::Literal(value, _) => value.try_as_str().flatten().map(str::to_owned),
        _ => None,
    };
    let named = |expr: &Expr| matches!(expr, Expr::Column(found) if found.name == column);

    match filter {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Eq,
            right,
        }) => {
            if named(left) {
                text(right).map(|value| vec![value])
            } else if named(right) {
                text(left).map(|value| vec![value])
            } else {
                None
            }
        }
        Expr::InList(InList {
            expr,
            list,
            negated: false,
        }) if named(expr) => list.iter().map(text).collect(),
        _ => None,
    }
}

/// The values every equality filter on `column` admits; `None` when no
/// filter constrains it.
fn admitted(filters: &[Expr], column: &str) -> Option<Vec<String>> {
    let mut admitted: Option<BTreeSet<String>> = None;
    for values in filters.iter().filter_map(|filter| equality(filter, column)) {
        let values: BTreeSet<String> = values.into_iter().collect();
        admitted = Some(match admitted {
            Some(admitted) => admitted.intersection(&values).cloned().collect(),
            None => values,
        });
    }

    admitted.map(|values| values.into_iter().collect())
}

fn pushdown(filters: &[&Expr], column: &str) -> Vec<TableProviderFilterPushDown> {
    filters
        .iter()
        .map(|filter| match equality(filter, column) {
            // DataFusion still applies the filter to the scanned rows.
            Some(_) => TableProviderFilterPushDown::Inexact,
            None => TableProviderFilterPushDown::Unsupported,
        })
        .collect()
}

fn exec(
    batch: RecordBatch,
    schema: &SchemaRef,
    projection: Option<&Vec<usize>>,
) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
    Ok(MemorySourceConfig::try_new_exec(
        &[vec![batch]],
        Arc::clone(schema),
        projection.cloned(),
    )?)
}

/// The `stats` table of a store.
#[derive(Debug)]
pub struct Stats {
    store: Store,
}

impl Stats {
    pub fn new(store: Store) -> Self {
        Self { store }
    }
}

#[async_trait]
impl TableProvider for Stats {
    fn schema(&self) -> SchemaRef {
        STATS.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::error::Result<Vec<TableProviderFilterPushDown>> {
        Ok(pushdown(filters, "name"))
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let names = admitted(filters, "name");
        let batch = self.store.stats_batch(names.as_deref())?;

        exec(batch, &STATS, projection)
    }
}

/// The `events` table of a store, or of a fixed batch of its events (the
/// rows an SSE message is computed over).
#[derive(Debug)]
pub struct Events {
    source: Source,
}

#[derive(Debug)]
enum Source {
    Store(Store),
    Batch(RecordBatch),
}

impl Events {
    pub fn new(store: Store) -> Self {
        Self {
            source: Source::Store(store),
        }
    }

    pub(crate) fn batch(batch: RecordBatch) -> Self {
        Self {
            source: Source::Batch(batch),
        }
    }
}

#[async_trait]
impl TableProvider for Events {
    fn schema(&self) -> SchemaRef {
        EVENTS.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::error::Result<Vec<TableProviderFilterPushDown>> {
        Ok(pushdown(filters, "kind"))
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let batch = match &self.source {
            Source::Store(store) => {
                store
                    .events_batch(admitted(filters, "kind").as_deref())
                    .await?
            }
            Source::Batch(batch) => batch.clone(),
        };

        exec(batch, &EVENTS, projection)
    }
}
