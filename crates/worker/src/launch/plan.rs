//! Planning ranks use the serving process transport and return a description
//! of their weights and source links without reading checkpoint tensor values.

use std::path::Path;
use std::time::Duration;

use super::{Arguments, EngineWatch, python, register};
use crate::config::resolve;
use crate::ffi::Interpreter;

pub(crate) fn main(argv: &[String], python: &Path) -> i32 {
    let result = run(argv, python);
    match result {
        Ok(()) => 0,
        Err(error) => {
            tracing::error!("planning the rank failed: {error}");
            1
        }
    }
}

fn run(argv: &[String], python: &Path) -> crate::Result<()> {
    let arguments =
        Arguments::parse(argv).map_err(|error| crate::Error::Invalid(error.to_string()))?;
    let (mut channel, descriptor) = register(&arguments)?;
    let transport = |error| crate::Error::transport_source("planning rank transport", error);
    let _watch = EngineWatch::start(channel.hangup().map_err(transport)?)?;
    let source_worlds = loop {
        if let Some(worlds) = channel.try_recv::<Vec<u64>>().map_err(transport)? {
            break worlds;
        }
        channel.wait(Duration::MAX).map_err(transport)?;
    };
    let launch = resolve(descriptor, &arguments.worker_id, arguments.generation)
        .map_err(|error| crate::Error::Invalid(error.to_string()))?;
    let planned = plan(&launch, python, &source_worlds).map_err(|error| error.to_string());
    let succeeded = planned.is_ok();
    channel.send(&planned).map_err(transport)?;
    // A manifest can exceed the socket's send buffer. Complete the queued
    // frame before this short-lived rank closes its connection.
    channel.flush().map_err(transport)?;
    if succeeded {
        Ok(())
    } else {
        Err(crate::Error::Invalid("rank planning failed".into()))
    }
}

fn plan(
    launch: &crate::config::WorkerLaunch,
    python: &Path,
    source_worlds: &[u64],
) -> crate::Result<::storage::manifest::weights::PlannedRank> {
    let (started, ready) = crossbeam_channel::bounded(1);
    let (finish, finished) = crossbeam_channel::bounded::<()>(1);
    let python = python.to_owned();
    std::thread::Builder::new()
        .name("builder".into())
        .spawn(
            move || match python::host(&python).and_then(|()| Interpreter::start()) {
                Ok(interpreter) => {
                    let _ = started.send(Ok(()));
                    interpreter.run(&finished);
                }
                Err(error) => {
                    let _ = started.send(Err(error));
                }
            },
        )
        .map_err(|error| crate::Error::transport_source("builder thread", error))?;
    ready
        .recv()
        .map_err(|_| crate::Error::State("the builder thread ended before it started"))??;
    let planned = crate::worker::plan_rank(launch, source_worlds);
    let _ = finish.send(());
    planned
}
