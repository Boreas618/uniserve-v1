//! Latency of one expert-step agreement over the host group.
//!
//! Every expert step starts with one agreement: each member of the union
//! sends its 24-byte record to every other member and selects from all of
//! them. This bench forms a union of separate processes on this host (the
//! first runs the bench and starts the others), runs warmup rounds, then
//! times `UNISERVE_AGREEMENT_ROUNDS` agreements (default 20000) with
//! changing token counts, and prints the latency distribution measured on
//! member 0: `cargo bench -p uniserve-worker --bench expert_agreement`.
//! `UNISERVE_AGREEMENT_MEMBERS` sets the union size (default 2).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use uniserve_worker::parallel::host::{HostGroup, Rendezvous};
use uniserve_worker::parallel::{ExpertParallel, ExpertTransport};

const MEMBER: &str = "UNISERVE_AGREEMENT_MEMBER";
const ADDRESS: &str = "UNISERVE_AGREEMENT_ADDRESS";

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let members = setting("UNISERVE_AGREEMENT_MEMBERS", 2);
    let rounds = setting("UNISERVE_AGREEMENT_ROUNDS", 20_000);
    let (member, rendezvous, children) = match std::env::var(MEMBER) {
        Ok(member) => {
            let address = std::env::var(ADDRESS).unwrap();
            (
                member.parse().unwrap(),
                Rendezvous::connect(address),
                Vec::new(),
            )
        }
        Err(_) => {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap().to_string();
            let children: Vec<_> = (1..members)
                .map(|member| {
                    std::process::Command::new(std::env::current_exe().unwrap())
                        .env(MEMBER, member.to_string())
                        .env(ADDRESS, &address)
                        .spawn()
                        .unwrap()
                })
                .collect();
            (0, Rendezvous::serve(address, listener), children)
        }
    };
    let group = HostGroup::connect(rendezvous, member, members, Duration::from_secs(60)).unwrap();
    let parallel = ExpertParallel::configure(
        Arc::new(Mutex::new(group)),
        0,
        &[member],
        8192,
        ExpertTransport::AllToAll,
    )
    .unwrap();

    let mut samples = Vec::with_capacity(rounds);
    for round in 0..rounds + 1000 {
        let tokens = 1 + (round * 7 + member) % 512;
        let started = Instant::now();
        let agreement = parallel.agree(tokens, 0, false).unwrap();
        let elapsed = started.elapsed();
        assert!(agreement.active && agreement.capacity >= tokens);
        if round >= 1000 {
            samples.push(elapsed.as_secs_f64() * 1e6);
        }
    }
    parallel.barrier().unwrap();
    for mut child in children {
        assert!(child.wait().unwrap().success());
    }
    if member == 0 {
        samples.sort_by(f64::total_cmp);
        let quantile = |q: f64| samples[((samples.len() - 1) as f64 * q) as usize];
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        println!(
            "host-group agreement, {members} processes, {rounds} rounds: mean {mean:.1} us, \
             p50 {:.1} us, p90 {:.1} us, p99 {:.1} us, max {:.1} us",
            quantile(0.5),
            quantile(0.9),
            quantile(0.99),
            quantile(1.0)
        );
    }
}
