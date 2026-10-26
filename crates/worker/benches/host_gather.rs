//! Latency of one fixed-width host gather over the host group.
//!
//! Every expert step starts with one agreement: each member of the union
//! sends its 24-byte record to every other member over the host group
//! (`HostGroup::all_gather_fixed`, which the executor's agreement reaches
//! through the union's host functions) and selects from all of them. This
//! bench forms a group of separate processes on this host (the first runs
//! the bench and starts the others), runs warmup rounds, then times
//! `UNISERVE_GATHER_ROUNDS` gathers (default 20000) of changing records,
//! and prints the latency distribution measured on member 0:
//! `cargo bench -p worker --bench host_gather`. `UNISERVE_GATHER_MEMBERS`
//! sets the group size (default 2).

use std::net::TcpListener;
use std::time::{Duration, Instant};

use worker::parallel::host::{HostGroup, Rendezvous};

const MEMBER: &str = "UNISERVE_GATHER_MEMBER";
const ADDRESS: &str = "UNISERVE_GATHER_ADDRESS";

/// Bytes of one expert-step agreement record: tokens, leaving and kind.
const RECORD: usize = 24;

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let members = setting("UNISERVE_GATHER_MEMBERS", 2);
    let rounds = setting("UNISERVE_GATHER_ROUNDS", 20_000);
    let (member, rendezvous, children) = match std::env::var(MEMBER) {
        Ok(member) => {
            let address = std::env::var(ADDRESS)?;
            (member.parse()?, Rendezvous::connect(address), Vec::new())
        }
        Err(_) => {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let address = listener.local_addr()?.to_string();
            let executable = std::env::current_exe()?;
            let children = (1..members)
                .map(|member| {
                    std::process::Command::new(&executable)
                        .env(MEMBER, member.to_string())
                        .env(ADDRESS, &address)
                        .spawn()
                })
                .collect::<Result<Vec<_>, _>>()?;
            (0, Rendezvous::serve(address, listener), children)
        }
    };
    let group = HostGroup::connect(rendezvous, member, members, Duration::from_secs(60))?;

    let mut samples = Vec::with_capacity(rounds);
    let mut records = vec![0_u8; RECORD * members];
    for round in 0..rounds + 1000 {
        let tokens = (1 + (round * 7 + member) % 512) as i64;
        let mut local = [0_u8; RECORD];
        local[..8].copy_from_slice(&tokens.to_le_bytes());
        let started = Instant::now();
        group.all_gather_fixed(&local, &mut records)?;
        let elapsed = started.elapsed();
        // Every member's record arrives in member order.
        let own = &records[member * RECORD..(member + 1) * RECORD];
        assert_eq!(own, &local[..]);
        if round >= 1000 {
            samples.push(elapsed.as_secs_f64() * 1e6);
        }
    }
    for mut child in children {
        let status = child.wait()?;
        if !status.success() {
            return Err(format!("gather member exited with {status}").into());
        }
    }
    if member == 0 {
        samples.sort_by(f64::total_cmp);
        let quantile = |q: f64| samples[((samples.len() - 1) as f64 * q) as usize];
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        println!(
            "host-group fixed gather, {members} processes, {rounds} rounds: mean {mean:.1} us, \
             p50 {:.1} us, p90 {:.1} us, p99 {:.1} us, max {:.1} us",
            quantile(0.5),
            quantile(0.9),
            quantile(0.99),
            quantile(1.0)
        );
    }
    Ok(())
}
