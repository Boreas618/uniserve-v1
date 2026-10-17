//! Stream collectives over native NCCL communicators, called through their
//! TVM-FFI functions as numerical code calls them. The tests need two CUDA
//! devices and the NCCL a worker process loads (PyTorch's, whose zero-CTA
//! policy the communicators request); run them with
//! `LD_LIBRARY_PATH=<site-packages>/nvidia/nccl/lib cargo test -p
//! uniserve-worker --test stream_collectives -- --ignored`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::net::TcpListener;
use std::time::Duration;

use tvm_ffi::{Any, AnyView};
use uniserve_cuda::{DeviceId, memcpy_async};
use uniserve_worker::CUDAStream;
use uniserve_worker::ffi::DType;
use uniserve_worker::parallel::host::{HostGroup, Rendezvous};
use uniserve_worker::parallel::{CollectiveGroup, StreamCollectives};
use uniserve_worker::storage::{Memory, StorageView};

fn upload(values: &[f32], device: DeviceId, stream: &CUDAStream) -> StorageView {
    let memory = Memory::device(device, values.len() * 4).unwrap();
    let view = StorageView::whole(memory, DType::F32).unwrap();
    unsafe {
        memcpy_async(
            view.address(),
            values.as_ptr() as u64,
            values.len() * 4,
            stream.stream().unwrap(),
        )
    }
    .unwrap();
    stream.synchronize().unwrap();
    view
}

fn download(view: &StorageView, stream: &CUDAStream) -> Vec<f32> {
    let mut values = vec![0_f32; view.numel() as usize];
    unsafe {
        memcpy_async(
            values.as_mut_ptr() as u64,
            view.address(),
            values.len() * 4,
            stream.stream().unwrap(),
        )
    }
    .unwrap();
    stream.synchronize().unwrap();
    values
}

fn call(function: &tvm_ffi::Function, args: &[Any]) -> Any {
    let views: Vec<AnyView<'_>> = args.iter().map(AnyView::from).collect();
    function.call_packed(&views).unwrap()
}

#[test]
#[ignore = "needs two CUDA devices"]
fn ranks_reduce_gather_and_broadcast_on_their_binding_stream() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let mut listener = Some(listener);
    std::thread::scope(|scope| {
        let ranks: Vec<_> = (0..2)
            .map(|rank| {
                let rendezvous = match listener.take().filter(|_| rank == 0) {
                    Some(listener) => Rendezvous::serve(address.clone(), listener),
                    None => Rendezvous::connect(address.clone()),
                };
                scope.spawn(move || {
                    let device = DeviceId(rank as i32);
                    let host =
                        HostGroup::connect(rendezvous, rank, 2, Duration::from_secs(60)).unwrap();
                    let stream = CUDAStream::create(device).unwrap();
                    // Logical order differs from world order: world rank 1
                    // is the first member.
                    let groups = [CollectiveGroup {
                        name: "tp".into(),
                        members: vec![1, 0],
                        ranks: vec![1, 0],
                    }];
                    let mut collectives = StreamCollectives::bind(&stream, &host, &groups).unwrap();
                    let functions = collectives.functions("tp").unwrap();

                    let value = upload(
                        &[rank as f32 + 1.0, 10.0 * (rank as f32 + 1.0)],
                        device,
                        &stream,
                    );
                    call(
                        &functions.all_reduce,
                        &[Any::from(value.to_tensor().unwrap()), Any::from(0_i64)],
                    );
                    let reduced = download(&value, &stream);

                    let local = upload(&[rank as f32; 3], device, &stream);
                    let gathered = upload(&[0.0; 6], device, &stream);
                    call(
                        &functions.all_gather,
                        &[
                            Any::from(gathered.to_tensor().unwrap()),
                            Any::from(local.to_tensor().unwrap()),
                        ],
                    );
                    let gathered = download(&gathered, &stream);

                    // Asynchronous gather, joined on the binding stream.
                    let started = upload(&[0.0; 6], device, &stream);
                    let token = call(
                        &functions.start_all_gather,
                        &[
                            Any::from(started.to_tensor().unwrap()),
                            Any::from(local.to_tensor().unwrap()),
                        ],
                    );
                    call(
                        &functions.wait,
                        &[token, Any::from(stream.handle().unwrap() as i64)],
                    );
                    let started = download(&started, &stream);

                    let broadcast = upload(&[rank as f32 + 5.0], device, &stream);
                    call(
                        &functions.broadcast,
                        &[Any::from(broadcast.to_tensor().unwrap()), Any::from(1_i64)],
                    );
                    let broadcast = download(&broadcast, &stream);

                    collectives.close(false).unwrap();
                    (reduced, gathered, started, broadcast)
                })
            })
            .collect();
        for rank in ranks {
            let (reduced, gathered, started, broadcast) = rank.join().unwrap();
            assert_eq!(reduced, [3.0, 30.0]);
            // Backend order is ascending world rank.
            assert_eq!(gathered, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
            assert_eq!(started, gathered);
            // The root names a world rank.
            assert_eq!(broadcast, [6.0]);
        }
    });
}
