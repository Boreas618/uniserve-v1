//! The checked-in deployment files, as a node reads them.

use std::path::Path;

/// Every checked-in deployment file parses, places and validates, and places no rank on a CPU,
/// which a node refuses: a main process serves CUDA devices.
///
/// It also holds each file to the resolver's own rules - a component's membership equals its
/// tensor, sequence and pipeline degrees, a distributed component's degrees are one, every
/// group's queue depth is positive - because a page that violates one of them parses and then
/// fails on the node that reads it.
#[test]
fn every_deployment_file_parses_places_and_validates() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/minimax_h3");
    for entry in std::fs::read_dir(&directory).expect("the deployment directory") {
        let path = entry.expect("an entry").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("toml") {
            continue;
        }
        let name = path
            .file_name()
            .expect("a name")
            .to_string_lossy()
            .into_owned();
        let text = std::fs::read_to_string(&path).expect("the file reads");
        let page = text.clone();
        // The hosts the file names: a rank is `[HOST/]DEVICE`, and a file that names no host
        // places on the local one.
        let mut hosts: Vec<String> = Vec::new();
        for piece in text.split(['[', ',', ']']) {
            let piece = piece.trim().trim_matches('"');
            let Some((host, device)) = piece.rsplit_once('/') else {
                continue;
            };
            let placed = device.starts_with("cuda") || device.starts_with("cpu");
            if placed && !hosts.iter().any(|seen| seen == host) {
                hosts.push(host.to_owned());
            }
        }
        if hosts.is_empty() {
            hosts.push("localhost".to_owned());
        }
        for host in &hosts {
            let groups = proto::config::WorkerGroupConfig::parse_file(&page, host)
                .unwrap_or_else(|error| panic!("{name} does not place on {host}: {error:#}"));
            assert!(!groups.is_empty(), "{name} places no group on {host}");
            for group in &groups {
                group.validate().unwrap_or_else(|error| {
                    panic!(
                        "{name}: worker group {:?} is invalid: {error:#}",
                        group.id.0
                    )
                });
                for rank in &group.ranks {
                    assert!(
                        !rank.device.starts_with("cpu"),
                        "{name}: worker group {:?} places rank {} on {:?}, and a main process \
                         serves CUDA devices",
                        group.id.0,
                        rank.host,
                        rank.device
                    );
                }
            }
        }
    }
}
