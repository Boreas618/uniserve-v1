//! Handoffs between replicas whose KV pages lie in GPU storage: the decode replica's import fills
//! blocks of its own from the pages the prefill replica lent, resharded by head range from the
//! prefill rank, which holds both heads, into the two decode ranks, which hold one each. A read
//! runs on the decode host's Transfer Server and a push on the lender's: device to device on one
//! host, and through NIXL or Mooncake Transfer Engine from or into another server's GPU
//! storage.

#[cfg(test)]
mod device {
    use std::os::fd::{AsFd, OwnedFd};
    use std::path::Path;

    use ::storage::protocol::{Location, UnitLayout};
    use ::storage::runtime::{Completion, Lender, OwnerId, Request};
    use ::storage::{Access, Claim, Claimant};
    use platform::{CopyBatch, DeviceAllocation, DeviceId, PinnedBuffer, SourceOrder, Stream};

    use crate::server::{Running, engine_head, head, key, memfd, scope, tensor, transfer_child};
    use crate::sessions::{
        HEADS, KV, Scheduler, attach_decode, imported_units, kv_bytes, kv_layout, kv_runs, layout,
        lend_pages, lend_two_pages, rank, write_pages,
    };
    use crate::transfer::device::{device_bytes, device_storage, gpus, record};
    use crate::transfer::{Paths, calibration, read, transfer_pid};
    use scheduling::kv::Transport;

    /// A server whose Transfer Server copies on every GPU the driver shows and crosses to other
    /// servers' storage over each of `crossing`, at the rates the calibration step timed for it;
    /// with `reachable`, other hosts reach it at this host's name. The servers' records go to the
    /// case's output, filtered by `RUST_LOG`.
    fn gpu_server(crossing: &[Transport], reachable: bool) -> Running {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .try_init();
        let paths = Paths {
            gpus: true,
            reads: crossing.contains(&Transport::Nixl),
            fabric: false,
            mooncake: crossing.contains(&Transport::Mooncake),
        };
        let children = Some(transfer_child(calibration(paths)));
        let server = if reachable {
            Running::reachable(children)
        } else {
            Running::with_children(children)
        };
        transfer_pid(&server.socket, None);
        server
    }

    /// A rank of `group` whose units lie in GPU storage on `gpu`, holding `heads` heads from
    /// `offset`, placed in its group's `memory` as part `part`; with its allocation.
    struct DeviceRank {
        _client: ::storage::Client,
        allocation: DeviceAllocation,
        _descriptor: OwnedFd,
    }

    /// Places rank `number` of `group` on `gpu` at the server on `socket`.
    fn device_rank(
        socket: &Path,
        claim: Claim,
        (group, number, memory, part): (&str, u32, &str, u32),
        gpu: DeviceId,
        (offset, heads): (u32, u32),
    ) -> DeviceRank {
        let mut client = rank(socket, group, number, claim);
        // A retained allocation is a whole number of the device's granules; the units lie at its
        // start.
        let granularity = DeviceAllocation::granularity(gpu).unwrap();
        let bytes = (kv_bytes(heads) as usize).div_ceil(granularity) * granularity;
        let (allocation, descriptor) = device_storage(gpu, &vec![0; bytes]);
        let name = 40 + number as u8 + if group == "decode" { 10 } else { 0 };
        client
            .put(
                vec![tensor(key(name), "kv", 0, 0, Access::Exclusive)],
                &[(record(gpu, bytes), descriptor.as_fd())],
            )
            .unwrap();
        let location = Location::Device {
            gpu: gpu.uuid().unwrap().0,
        };
        let layout: Option<UnitLayout> = Some(kv_layout(offset, heads));
        client
            .place(
                "units".into(),
                memory.into(),
                part,
                kv_runs(key(name), heads, location),
                layout,
            )
            .unwrap();
        DeviceRank {
            _client: client,
            allocation,
            _descriptor: descriptor,
        }
    }

    /// The bytes a rank of `heads` heads from `offset` holds once pages were written into
    /// `units` ([`write_pages`]).
    fn pages(units: &[u32], (offset, heads): (u32, u32)) -> Vec<u8> {
        let file = memfd(&vec![0; kv_bytes(heads) as usize]);
        write_pages(&file, units, (offset, heads));
        read(&file, 0, kv_bytes(heads) as usize)
    }

    /// Writes `contents` at the start of `allocation`.
    fn upload(allocation: &DeviceAllocation, contents: &[u8]) {
        let mut staging = PinnedBuffer::new(allocation.device(), contents.len()).unwrap();
        // SAFETY: a pinned buffer this test owns, written before any copy.
        let host = unsafe { staging.as_mut_slice() };
        host[..contents.len()].copy_from_slice(contents);
        let stream = Stream::new(allocation.device()).unwrap();
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        batch.push(allocation.address(), staging.address(), contents.len());
        // SAFETY: both ranges hold `contents.len()` bytes; the stream drains.
        unsafe { batch.submit(&stream) }.unwrap();
        stream.synchronize().unwrap();
    }

    /// Whether the decode ranks' GPU storage holds the two pages in `units`, each rank its own
    /// head, with the planes' flags set.
    fn assert_imported(decode: &[&DeviceRank], units: &[u32]) {
        for (head, rank) in decode.iter().enumerate() {
            let held = device_bytes(&rank.allocation);
            assert_eq!(
                held[..kv_bytes(1) as usize],
                pages(units, (head as u32, 1)),
                "decode rank {head} holds its head of both pages"
            );
        }
    }

    /// The import of the two lent pages into owner 2's positions 0 and 1, over `transport`
    /// between two hosts.
    fn import(lender: Lender, push: bool, transport: Transport) -> Request {
        Request::Import {
            owner: OwnerId(2),
            tables: vec![(KV, 0, 2)],
            lender,
            push,
            transport,
            trace: 0,
        }
    }

    /// Two replicas of one engine on one server: the prefill rank on the first GPU, the two
    /// decode ranks on the next two. The decode scheduler imports the two pages the prefill
    /// scheduler lent, read on this host's Transfer Server, or pushed, which is the same host's
    /// here.
    fn on_one_host(push: bool) {
        let [first, second, third] = gpus::<3>();
        let server = gpu_server(&[], false);
        let claim = head(&server.socket)
            .claim(
                Claimant::Head,
                0,
                vec![scope("prefill", 0), scope("decode", 1), scope("decode", 2)],
                Vec::new(),
            )
            .unwrap();
        let prefill = device_rank(
            &server.socket,
            claim,
            ("prefill", 0, "gpu0", 0),
            first,
            (0, HEADS),
        );
        let decode = [
            device_rank(
                &server.socket,
                claim,
                ("decode", 1, "gpu1", 0),
                second,
                (0, 1),
            ),
            device_rank(
                &server.socket,
                claim,
                ("decode", 2, "gpu1", 1),
                third,
                (1, 1),
            ),
        ];

        let mut engine = engine_head(&server.socket, 0);
        let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
        let mut schedulers = Scheduler::all(&opened, descriptors);
        let lender = lend_two_pages(&mut schedulers, &opened, |units| {
            upload(&prefill.allocation, &pages(units, (0, HEADS)));
        });
        let units = imported_units(schedulers[1].ask(import(lender, push, Transport::Nixl)));
        assert_imported(&[&decode[0], &decode[1]], &units);
    }

    #[test]
    #[ignore = "requires three CUDA devices whose copy engines address each other"]
    fn an_import_reads_lent_pages_from_one_gpu_into_two_and_reshards_them() {
        on_one_host(false);
    }

    #[test]
    #[ignore = "requires three CUDA devices whose copy engines address each other"]
    fn an_import_pushed_from_one_gpu_into_two_reshards_the_lent_pages() {
        on_one_host(true);
    }

    /// The prefill replica's engine and rank on one server, its rank on the first GPU; the decode
    /// replica's engine and its two ranks on another, on the next two GPUs. Each server's Transfer
    /// Server reaches the other's GPU storage over `transport`: a read runs on the decode
    /// replica's server and a push on the prefill replica's.
    fn across_servers(push: bool, transport: Transport) {
        let [first, second, third] = gpus::<3>();
        let lending = gpu_server(&[transport], false);
        let importing = gpu_server(&[transport], false);
        let lending_claim = head(&lending.socket)
            .claim(Claimant::Head, 0, vec![scope("prefill", 0)], Vec::new())
            .unwrap();
        let importing_claim = head(&importing.socket)
            .claim(
                Claimant::Head,
                0,
                vec![scope("decode", 1), scope("decode", 2)],
                Vec::new(),
            )
            .unwrap();
        let prefill = device_rank(
            &lending.socket,
            lending_claim,
            ("prefill", 0, "gpu0", 0),
            first,
            (0, HEADS),
        );
        let decode = [
            device_rank(
                &importing.socket,
                importing_claim,
                ("decode", 1, "gpu1", 0),
                second,
                (0, 1),
            ),
            device_rank(
                &importing.socket,
                importing_claim,
                ("decode", 2, "gpu1", 1),
                third,
                (1, 1),
            ),
        ];

        let mut lending_engine = engine_head(&lending.socket, 0);
        let (lending_opened, descriptors) = lending_engine.open_pool(layout(0)).unwrap();
        let mut lending_schedulers = Scheduler::all(&lending_opened, descriptors);
        let mut importing_engine = engine_head(&importing.socket, 0);
        let (importing_opened, descriptors) = importing_engine.open_pool(layout(0)).unwrap();
        let mut importing_schedulers = Scheduler::all(&importing_opened, descriptors);

        let mut both = [lending_schedulers.remove(0), importing_schedulers.remove(1)];
        let mut lender = lend_two_pages(&mut both, &lending_opened, |units| {
            upload(&prefill.allocation, &pages(units, (0, HEADS)));
        });
        lender.endpoint = Some(lending.tcp.clone());
        let [_lending_scheduler, mut decode_scheduler] = both;
        let units = imported_units(decode_scheduler.ask(import(lender, push, transport)));
        assert_imported(&[&decode[0], &decode[1]], &units);
        assert!(matches!(
            decode_scheduler.ask(Request::Release {
                owner: OwnerId(2),
                class: KV,
                at: 0,
                count: 2,
                after: Vec::new(),
            }),
            Completion::Done
        ));
    }

    #[test]
    #[ignore = "requires three CUDA devices, the NIXL package and a usable RDMA port"]
    fn an_import_reads_a_lend_in_another_server_s_gpu_through_nixl() {
        across_servers(false, Transport::Nixl);
    }

    #[test]
    #[ignore = "requires three CUDA devices, the NIXL package and a usable RDMA port"]
    fn an_import_pushed_into_this_server_s_gpus_is_written_through_nixl() {
        across_servers(true, Transport::Nixl);
    }

    #[test]
    #[ignore = "requires three CUDA devices, Mooncake Transfer Engine and a usable RDMA port"]
    fn an_import_reads_a_lend_in_another_server_s_gpu_through_the_transfer_engine() {
        across_servers(false, Transport::Mooncake);
    }

    #[test]
    #[ignore = "requires three CUDA devices, Mooncake Transfer Engine and a usable RDMA port"]
    fn an_import_pushed_into_this_server_s_gpus_is_written_through_the_transfer_engine() {
        across_servers(true, Transport::Mooncake);
    }

    /// The environment variable that names the second host of the cases across two hosts: an
    /// ssh destination that mounts this repository at the same path, has the GPUs, RDMA ports,
    /// NIXL and Mooncake packages of this host, and reaches it by its host name.
    const PEER_HOST: &str = "UNISERVE_PEER_HOST";

    /// The second host's half of the handoff cases across two hosts ([`across_hosts`]), which
    /// they run there as this case of this test executable: a server other hosts reach, whose
    /// prefill rank on the first GPU lends two pages, calibrated for both transports. It states
    /// the lend on its standard output, `lent ENDPOINT SESSION OWNER`, and holds it until its
    /// standard input ends.
    #[test]
    #[ignore = "the second host's half of the handoff cases across two hosts, which they start there"]
    fn the_lending_half_of_a_handoff_across_hosts() {
        use std::io::{Read as _, Write as _};

        let [first] = gpus::<1>();
        let lending = gpu_server(&[Transport::Nixl, Transport::Mooncake], true);
        let claim = head(&lending.socket)
            .claim(Claimant::Head, 0, vec![scope("prefill", 0)], Vec::new())
            .unwrap();
        let prefill = device_rank(
            &lending.socket,
            claim,
            ("prefill", 0, "gpu0", 0),
            first,
            (0, HEADS),
        );
        let mut engine = engine_head(&lending.socket, 0);
        let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
        let mut schedulers = Scheduler::all(&opened, descriptors);
        let lender = lend_pages(&mut schedulers[0], &opened, |units| {
            upload(&prefill.allocation, &pages(units, (0, HEADS)));
        });

        let mut out = std::io::stdout().lock();
        writeln!(
            out,
            "lent {} {} {}",
            lending.tcp, lender.session, lender.owner.0
        )
        .unwrap();
        out.flush().unwrap();
        drop(out);
        let mut rest = Vec::new();
        std::io::stdin().read_to_end(&mut rest).unwrap();
    }

    /// The second host's half, run there over ssh, which ends with its standard input.
    struct LendingHost {
        child: std::process::Child,
    }

    impl Drop for LendingHost {
        fn drop(&mut self) {
            drop(self.child.stdin.take());
            if self.child.wait().is_err() {
                let _ = self.child.kill();
            }
        }
    }

    /// Starts [`the_lending_half_of_a_handoff_across_hosts`] on the second host ([`PEER_HOST`])
    /// and returns it with the lend it states.
    fn lending_host() -> (LendingHost, Lender) {
        use std::io::BufRead as _;
        use std::process::{Command, Stdio};

        let peer =
            std::env::var(PEER_HOST).unwrap_or_else(|_| panic!("{PEER_HOST} names no second host"));
        let executable = std::env::current_exe().unwrap();
        let directory = std::env::current_dir().unwrap();
        let remote = format!(
            "cd {} && {} handoff::device::the_lending_half_of_a_handoff_across_hosts \
             --exact --ignored --nocapture --test-threads 1",
            directory.display(),
            executable.display()
        );
        let mut child = Command::new("ssh")
            .args(["-o", "BatchMode=yes", &peer, &remote])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let host = LendingHost { child };
        for line in std::io::BufReader::new(stdout).lines() {
            let line = line.unwrap();
            // The test harness states the case's name on the same line before the case's own
            // output.
            let Some((_, lent)) = line.split_once("lent ") else {
                continue;
            };
            let [endpoint, session, owner] = lent.split(' ').collect::<Vec<_>>()[..] else {
                panic!("{peer} stated the lend as {line:?}");
            };
            let lender = Lender {
                endpoint: Some(endpoint.to_owned()),
                session: session.parse().unwrap(),
                owner: OwnerId(owner.parse().unwrap()),
            };
            return (host, lender);
        }
        panic!("the lending half on {peer} ended before it lent");
    }

    /// The decode replica's engine and its two ranks on this host's first two GPUs; the prefill
    /// replica's engine and rank on the second host ([`lending_host`]). The two hosts' Transfer
    /// Servers reach each other's GPU storage over `transport` through their RDMA ports: a read
    /// runs on this host's Transfer Server and a push on the second host's.
    fn across_hosts(push: bool, transport: Transport) {
        let [first, second] = gpus::<2>();
        let importing = gpu_server(&[transport], true);
        let claim = head(&importing.socket)
            .claim(
                Claimant::Head,
                0,
                vec![scope("decode", 1), scope("decode", 2)],
                Vec::new(),
            )
            .unwrap();
        let decode = [
            device_rank(
                &importing.socket,
                claim,
                ("decode", 1, "gpu1", 0),
                first,
                (0, 1),
            ),
            device_rank(
                &importing.socket,
                claim,
                ("decode", 2, "gpu1", 1),
                second,
                (1, 1),
            ),
        ];
        let mut engine = engine_head(&importing.socket, 0);
        let (opened, descriptors) = engine.open_pool(layout(0)).unwrap();
        let mut decode_scheduler = Scheduler::all(&opened, descriptors).remove(1);
        attach_decode(&mut decode_scheduler);

        let (lending, lender) = lending_host();
        let started = std::time::Instant::now();
        let units = imported_units(decode_scheduler.ask(import(lender, push, transport)));
        eprintln!(
            "{transport:?} {} across hosts in {:?}",
            if push { "push" } else { "read" },
            started.elapsed()
        );
        assert_imported(&[&decode[0], &decode[1]], &units);
        drop(lending);
    }

    #[test]
    #[ignore = "requires a second host (UNISERVE_PEER_HOST), two CUDA devices on each, NIXL and RDMA ports"]
    fn an_import_reads_a_lend_on_another_host_through_nixl() {
        across_hosts(false, Transport::Nixl);
    }

    #[test]
    #[ignore = "requires a second host (UNISERVE_PEER_HOST), two CUDA devices on each, NIXL and RDMA ports"]
    fn an_import_pushed_from_another_host_is_written_through_nixl() {
        across_hosts(true, Transport::Nixl);
    }

    #[test]
    #[ignore = "requires a second host (UNISERVE_PEER_HOST), two CUDA devices on each, Mooncake Transfer Engine and RDMA ports"]
    fn an_import_reads_a_lend_on_another_host_through_the_transfer_engine() {
        across_hosts(false, Transport::Mooncake);
    }

    #[test]
    #[ignore = "requires a second host (UNISERVE_PEER_HOST), two CUDA devices on each, Mooncake Transfer Engine and RDMA ports"]
    fn an_import_pushed_from_another_host_is_written_through_the_transfer_engine() {
        across_hosts(true, Transport::Mooncake);
    }
}
