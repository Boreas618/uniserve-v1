//! Prints what `Allocator::suspend` and `Allocator::restore` book and map, so a
//! restore under a shrunk allowance can be read as numbers instead of a pass or
//! a fail.
//!
//! The GPU test `a_restore_a_shrunk_budget_refuses_parks_every_pool_again`
//! expects the resident pool's booking to be refused after the allowance drops
//! below what the three pools map, and the caller's run had the restore come
//! back `Ok`. This probe walks the same sequence and prints, at each step, the
//! bytes the driver rounded each block to and every total the budget was asked
//! to accept, so the next run names the step that differs from the contract.
//!
//!     cargo run -p platform --example restore_probe

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use platform::{Allocator, Budget, DeviceId, Error, PoolKind};

/// A budget that records every total it is asked to accept and refuses those
/// above a limit the probe changes between the suspension and the restore.
struct Ledger {
    limit: AtomicU64,
    totals: Mutex<Vec<(PoolKind, u64, bool)>>,
}

impl Ledger {
    fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self {
            limit: AtomicU64::new(limit),
            totals: Mutex::new(Vec::new()),
        })
    }

    fn set_limit(&self, limit: u64) {
        self.limit.store(limit, Ordering::Relaxed);
    }

    /// Prints every total asked for, in order, with whether it was accepted.
    fn print(&self, step: &str) {
        let totals = self.totals.lock().unwrap_or_else(PoisonError::into_inner);
        for (pool, bytes, accepted) in totals.iter() {
            println!(
                "{step}: {pool:?} total {bytes} bytes, {}",
                if *accepted { "accepted" } else { "refused" }
            );
        }
        println!("{step}: {} booking(s)", totals.len());
    }
}

impl Budget for Ledger {
    fn book(&self, device: DeviceId, pool: PoolKind, bytes: u64) -> platform::Result<()> {
        let limit = self.limit.load(Ordering::Relaxed);
        let accepted = bytes <= limit;
        self.totals
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((pool, bytes, accepted));
        if !accepted {
            return Err(Error::InvalidArgument(format!(
                "the budget of {device} refuses {bytes} bytes for its {pool:?} pool"
            )));
        }
        Ok(())
    }
}

fn main() -> platform::Result<()> {
    let device = platform::Device::new(DeviceId(0))?;
    let ledger = Ledger::new(64 << 20);
    let mut allocator = Allocator::new(&device, Arc::<Ledger>::clone(&ledger))?;
    for pool in [PoolKind::Graph, PoolKind::Scratch, PoolKind::Resident] {
        allocator.reserve(pool, 4 << 20)?;
    }
    let graph = allocator.allocate(PoolKind::Graph, 1 << 20)?;
    let scratch = allocator.allocate(PoolKind::Scratch, 1 << 20)?;
    let resident = allocator.allocate(PoolKind::Resident, 1 << 20)?;
    println!(
        "blocks as the driver rounded them: graph {} bytes, scratch {} bytes, resident {} bytes, \
         together {} bytes",
        graph.len(),
        scratch.len(),
        resident.len(),
        graph.len() + scratch.len() + resident.len()
    );
    ledger.print("allocated");

    allocator.suspend()?;
    println!(
        "mapped after suspend: graph {}, scratch {}, resident {}",
        allocator.mapped_bytes(PoolKind::Graph),
        allocator.mapped_bytes(PoolKind::Scratch),
        allocator.mapped_bytes(PoolKind::Resident)
    );
    ledger.print("suspended");

    // The allowance the failing test shrinks to: below what the three blocks
    // map, so at least one pool's restore must be refused.
    let limit = (2u64 << 20) + (1 << 19);
    ledger.set_limit(limit);
    println!("allowance shrunk to {limit} bytes");
    match allocator.restore() {
        Ok(()) => {
            println!("restore came back Ok — nothing was refused");
            ledger.print("restored");
            println!(
                "mapped after restore: graph {}, scratch {}, resident {}",
                allocator.mapped_bytes(PoolKind::Graph),
                allocator.mapped_bytes(PoolKind::Scratch),
                allocator.mapped_bytes(PoolKind::Resident)
            );
            return Ok(());
        }
        Err(error) => {
            println!("restore refused: {error}");
            ledger.print("restore attempt");
            println!(
                "mapped after the refusal: graph {}, scratch {}, resident {}",
                allocator.mapped_bytes(PoolKind::Graph),
                allocator.mapped_bytes(PoolKind::Scratch),
                allocator.mapped_bytes(PoolKind::Resident)
            );
        }
    }
    Ok(())
}
