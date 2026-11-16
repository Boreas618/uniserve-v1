//! One worker's file of a profiling session (`crates/engine/src/profiling/schema.sql`),
//! opened read-only.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use rusqlite::types::ValueRef;
use serde_json::{Map, Value, json};

/// The schema version this reader reads, and every later one.
pub const SCHEMA_VERSION: i64 = 2;
/// The probe ids of a fence step's markers.
pub const STEP_BEGIN: i64 = 0xF_FFFE;
pub const STEP_END: i64 = 0xF_FFFF;

/// What to add to a time of one clock to place it on CUPTI's clock, and the
/// alignment's uncertainty, in nanoseconds.
#[derive(Debug, Clone, Copy)]
pub struct Offset {
    pub nanoseconds: i64,
    pub uncertainty: i64,
}

/// A session file.
pub struct ProfileFile {
    pub path: PathBuf,
    db: rusqlite::Connection,
    pub version: i64,
    pub metadata: BTreeMap<String, String>,
    strings: BTreeMap<i64, String>,
}

/// A SQLite value as JSON.
fn json_of(value: ValueRef<'_>) -> Value {
    match value {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(number) => number.into(),
        ValueRef::Real(number) => number.into(),
        ValueRef::Text(text) => String::from_utf8_lossy(text).into_owned().into(),
        ValueRef::Blob(blob) => blob.iter().map(|byte| Value::from(*byte)).collect(),
    }
}

impl ProfileFile {
    /// Opens a session file.
    ///
    /// # Errors
    ///
    /// When the path is not a file of a session, or of an earlier schema.
    pub fn open(path: &Path) -> Result<Self> {
        ensure!(path.is_file(), "{} is not a file", path.display());
        let db = rusqlite::Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        let version: Option<i64> = db
            .query_row("SELECT version FROM schema_version", [], |row| row.get(0))
            .map_err(|error| {
                anyhow::anyhow!(
                    "{} is not a profiling session's file: {error}",
                    path.display()
                )
            })?;
        let Some(version) = version.filter(|version| *version >= SCHEMA_VERSION) else {
            bail!(
                "{} has schema version {version:?}; this reader reads version {SCHEMA_VERSION} and later",
                path.display()
            );
        };
        let metadata = Self::pairs(&db, "SELECT key, value FROM metadata")?
            .into_iter()
            .map(|(key, value)| (text(&key), text(&value)))
            .collect();
        let strings = Self::pairs(&db, "SELECT id, text FROM strings")?
            .into_iter()
            .filter_map(|(key, value)| Some((key.as_i64()?, text(&value))))
            .collect();
        Ok(Self {
            path: path.to_path_buf(),
            db,
            version,
            metadata,
            strings,
        })
    }

    fn pairs(db: &rusqlite::Connection, query: &str) -> Result<Vec<(Value, Value)>> {
        let mut statement = db.prepare(query)?;
        let rows = statement.query_map([], |row| {
            Ok((json_of(row.get_ref(0)?), json_of(row.get_ref(1)?)))
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn rank(&self) -> i64 {
        self.metadata
            .get("rank")
            .and_then(|rank| rank.parse().ok())
            .unwrap_or(0)
    }

    pub fn session(&self) -> i64 {
        self.metadata
            .get("session")
            .and_then(|session| session.parse().ok())
            .unwrap_or(0)
    }

    /// The tables the file holds, by name.
    pub fn tables(&self) -> Result<Vec<String>> {
        let mut statement = self
            .db
            .prepare("SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name")?;
        let names = statement.query_map([], |row| row.get::<_, String>(0))?;
        Ok(names.collect::<Result<_, _>>()?)
    }

    /// Every row of `table`, in `order` when given (an `ORDER BY` of its
    /// columns); none when the file has no such table.
    pub fn rows(&self, table: &str, order: Option<&str>) -> Result<Vec<Map<String, Value>>> {
        if !self.tables()?.iter().any(|name| name == table) {
            return Ok(Vec::new());
        }
        let mut query = format!("SELECT * FROM \"{table}\"");
        if let Some(order) = order {
            query.push_str(&format!(" ORDER BY {order}"));
        }
        let mut statement = self.db.prepare(&query)?;
        let names: Vec<String> = statement.column_names().map(str::to_owned).collect();
        let rows = statement.query_map([], |row| {
            let mut entry = Map::new();
            for (index, name) in names.iter().enumerate() {
                entry.insert(name.clone(), json_of(row.get_ref(index)?));
            }
            Ok(entry)
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn count(&self, table: &str) -> Result<i64> {
        if !self.tables()?.iter().any(|name| name == table) {
            return Ok(0);
        }
        Ok(self
            .db
            .query_row(&format!("SELECT COUNT(*) FROM \"{table}\""), [], |row| {
                row.get(0)
            })?)
    }

    /// A string id's text; empty for none.
    pub fn name(&self, id: &Value) -> String {
        match id.as_i64() {
            None | Some(0) => String::new(),
            Some(id) => self
                .strings
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("<string {id}>")),
        }
    }

    /// What maps `globaltimer` of `device`, or `monotonic`, onto CUPTI's clock,
    /// from the sample of least uncertainty; `None` without one.
    pub fn offset(&self, clock: &str, device: Option<i64>) -> Result<Option<Offset>> {
        let row = match clock {
            "globaltimer" => self.db.query_row(
                "SELECT cupti - gpu, uncertainty FROM clocks WHERE source = 'globaltimer' AND device = ?1 \
                 AND cupti IS NOT NULL ORDER BY uncertainty LIMIT 1",
                [device],
                |row| Ok((row.get(0)?, row.get(1)?)),
            ),
            "monotonic" => self.db.query_row(
                "SELECT cupti - monotonic, uncertainty FROM clocks WHERE monotonic IS NOT NULL \
                 AND cupti IS NOT NULL ORDER BY uncertainty LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            ),
            _ => bail!("no clock is named {clock}"),
        };
        match row {
            Ok((nanoseconds, uncertainty)) => Ok(Some(Offset {
                nanoseconds,
                uncertainty,
            })),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// Every probed module instance's call, from the fence records: its device,
    /// step, instance, path and module class, and its start and end in
    /// `%globaltimer` nanoseconds. A step's entries and exits pair per instance
    /// in time order; an unpaired one is left out.
    pub fn fence_intervals(&self) -> Result<Vec<Map<String, Value>>> {
        let probes: BTreeMap<i64, Map<String, Value>> = self
            .rows("fence_probe", None)?
            .into_iter()
            .filter_map(|row| Some((row.get("probe")?.as_i64()?, row)))
            .collect();
        let mut entered: BTreeMap<(i64, i64, i64), Vec<i64>> = BTreeMap::new();
        let mut intervals = Vec::new();
        for row in self.rows("fence", Some("device, time"))? {
            let number = |key: &str| row.get(key).and_then(Value::as_i64).unwrap_or(0);
            let probe = number("probe");
            if probe >= STEP_BEGIN {
                continue;
            }
            let key = (number("device"), number("step"), probe / 2);
            if probe % 2 == 0 {
                entered.entry(key).or_default().push(number("time"));
                continue;
            }
            let Some(start) = entered.get_mut(&key).and_then(Vec::pop) else {
                continue;
            };
            let described = probes.get(&(probe - 1));
            let field = |name: &str| {
                described
                    .and_then(|row| row.get(name))
                    .cloned()
                    .unwrap_or_else(|| "".into())
            };
            let mut interval = Map::new();
            interval.insert("device".into(), key.0.into());
            interval.insert("step".into(), key.1.into());
            interval.insert("instance".into(), key.2.into());
            interval.insert("path".into(), field("path"));
            interval.insert("module".into(), field("module"));
            interval.insert("start".into(), start.into());
            interval.insert("end".into(), number("time").into());
            intervals.push(interval);
        }
        Ok(intervals)
    }

    /// What the file holds: its session, worker, collectors and rows by table.
    pub fn summary(&self) -> Result<Value> {
        let meta = |key: &str| self.metadata.get(key).cloned();
        let collectors: Vec<Value> = self
            .rows("collectors", Some("time"))?
            .into_iter()
            .map(|row| json!({"name": row.get("name"), "state": row.get("state"), "detail": row.get("detail")}))
            .collect();
        let mut rows = Map::new();
        for table in self.tables()? {
            rows.insert(table.clone(), self.count(&table)?.into());
        }
        Ok(json!({
            "path": self.path.display().to_string(),
            "schema_version": self.version,
            "session": self.session(),
            "epoch": meta("epoch"),
            "rank": self.rank(),
            "device": meta("device"),
            "chip": meta("chip"),
            "end_reason": meta("end_reason"),
            "steps_recorded": meta("steps_recorded"),
            "collectors": collectors,
            "rows": rows,
        }))
    }
}

fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Opens every file of `paths`, naming the first that fails.
pub fn open_all(paths: &[PathBuf]) -> Result<Vec<ProfileFile>> {
    paths
        .iter()
        .map(|path| ProfileFile::open(path).with_context(|| format!("opening {}", path.display())))
        .collect()
}
