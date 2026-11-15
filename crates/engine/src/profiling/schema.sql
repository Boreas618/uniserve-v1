-- The schema of a profiling session's file (`crates/engine/src/profiling/artifact.rs`), version 2.
--
-- The writer creates every file with this schema and states its version in `schema_version`; a
-- reader reads the files of the version it was written for (`uniserve_eval.profile`). Times are
-- nanoseconds in the clock `metadata`'s `doc.clocks` names for each table, and the meaning of each
-- table's identifiers is stated under `doc.<table>`.

CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version VALUES (2);

CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE strings (id INTEGER PRIMARY KEY, text TEXT NOT NULL);
CREATE TABLE collectors (name TEXT NOT NULL, state TEXT NOT NULL, time INTEGER NOT NULL,
    detail TEXT NOT NULL);
CREATE TABLE clocks (source TEXT NOT NULL, device INTEGER, gpu INTEGER, realtime INTEGER NOT NULL,
    cupti INTEGER, monotonic INTEGER, uncertainty INTEGER NOT NULL);
CREATE TABLE steps (step INTEGER NOT NULL, stage TEXT NOT NULL, selected INTEGER NOT NULL,
    begin INTEGER NOT NULL, end INTEGER);
CREATE TABLE kernel (start INTEGER NOT NULL, end INTEGER NOT NULL, queued INTEGER,
    submitted INTEGER, device INTEGER NOT NULL, context INTEGER NOT NULL, stream INTEGER NOT NULL,
    correlation INTEGER NOT NULL, grid_x INTEGER, grid_y INTEGER, grid_z INTEGER, block_x INTEGER,
    block_y INTEGER, block_z INTEGER, registers INTEGER, static_shared INTEGER,
    dynamic_shared INTEGER, name INTEGER NOT NULL, graph INTEGER, graph_node INTEGER,
    launch_type INTEGER, channel INTEGER);
CREATE TABLE copy (start INTEGER NOT NULL, end INTEGER NOT NULL, device INTEGER NOT NULL,
    context INTEGER NOT NULL, stream INTEGER NOT NULL, correlation INTEGER NOT NULL,
    copy_kind INTEGER NOT NULL, source_kind INTEGER, destination_kind INTEGER,
    bytes INTEGER NOT NULL, graph INTEGER, graph_node INTEGER, source_device INTEGER,
    destination_device INTEGER);
CREATE TABLE fill (start INTEGER NOT NULL, end INTEGER NOT NULL, device INTEGER NOT NULL,
    context INTEGER NOT NULL, stream INTEGER NOT NULL, correlation INTEGER NOT NULL,
    value INTEGER NOT NULL, bytes INTEGER NOT NULL, memory_kind INTEGER, graph INTEGER,
    graph_node INTEGER);
CREATE TABLE memory (time INTEGER NOT NULL, operation INTEGER NOT NULL,
    memory_kind INTEGER NOT NULL, correlation INTEGER NOT NULL, address INTEGER NOT NULL,
    bytes INTEGER NOT NULL, process INTEGER, device INTEGER, context INTEGER, stream INTEGER,
    asynchronous INTEGER NOT NULL);
CREATE TABLE marker (time INTEGER NOT NULL, flags INTEGER NOT NULL, id INTEGER NOT NULL,
    object_kind INTEGER NOT NULL, object_0 INTEGER, object_1 INTEGER, object_2 INTEGER,
    name INTEGER NOT NULL, domain INTEGER NOT NULL);
CREATE TABLE marker_data (id INTEGER NOT NULL, flags INTEGER NOT NULL,
    payload_kind INTEGER NOT NULL, payload INTEGER, color INTEGER, category INTEGER);
CREATE TABLE graph (start INTEGER NOT NULL, end INTEGER NOT NULL, device INTEGER NOT NULL,
    context INTEGER NOT NULL, stream INTEGER NOT NULL, correlation INTEGER NOT NULL,
    graph INTEGER NOT NULL);
CREATE TABLE nvlink (version INTEGER NOT NULL, type_0 INTEGER NOT NULL, type_1 INTEGER NOT NULL,
    end_0 BLOB NOT NULL, end_1 BLOB NOT NULL, flag INTEGER NOT NULL, links INTEGER NOT NULL,
    bandwidth INTEGER NOT NULL, nvswitch INTEGER NOT NULL);
CREATE TABLE pcie (device_type INTEGER NOT NULL, id INTEGER NOT NULL, domain INTEGER NOT NULL,
    generation INTEGER NOT NULL, link_rate INTEGER NOT NULL, link_width INTEGER NOT NULL,
    upstream_bus INTEGER NOT NULL);
CREATE TABLE api (start INTEGER NOT NULL, end INTEGER NOT NULL, domain INTEGER NOT NULL,
    cbid INTEGER NOT NULL, name INTEGER NOT NULL, thread INTEGER NOT NULL,
    correlation INTEGER NOT NULL, result INTEGER NOT NULL);
CREATE TABLE fence_probe (probe INTEGER PRIMARY KEY, instance INTEGER NOT NULL,
    site TEXT NOT NULL, path TEXT NOT NULL, module TEXT NOT NULL);
CREATE TABLE fence (device INTEGER NOT NULL, time INTEGER NOT NULL, probe INTEGER NOT NULL,
    step INTEGER NOT NULL);
CREATE TABLE pm_sample (sample INTEGER NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
    metric TEXT NOT NULL, value REAL NOT NULL);
CREATE TABLE range_metric (pass INTEGER NOT NULL, range INTEGER NOT NULL, name TEXT NOT NULL,
    metric TEXT NOT NULL, value REAL NOT NULL);
CREATE TABLE pc_stall_reason (reason INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE pc_sample (take INTEGER NOT NULL, function TEXT NOT NULL,
    function_index INTEGER NOT NULL, cubin_crc INTEGER NOT NULL, offset INTEGER NOT NULL,
    correlation INTEGER NOT NULL, reason INTEGER NOT NULL, samples INTEGER NOT NULL);
CREATE TABLE pc_take (take INTEGER PRIMARY KEY, time INTEGER NOT NULL,
    total_samples INTEGER NOT NULL, dropped_samples INTEGER NOT NULL,
    non_user_samples INTEGER NOT NULL, hardware_buffer_full INTEGER NOT NULL);
