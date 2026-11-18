# Mooncake

UniServe uses three parts of Mooncake 0.3.13.post1, the release the workspace's `mooncake_store` binding pins: Mooncake Store, the pool of host memory across nodes that keeps the pages of the buffer pools' host KV caches (`kv_cache_config.mooncake`); its master's KV events, which the cluster directory follows to know which pages the store holds (`kv_cache_config.mooncake.events`); and the Transfer Engine, one of the two transports a handoff's pages cross hosts by (`storage_config.handoff_transports`, beside [NIXL](nixl.md)). Building UniServe needs no Mooncake installation: every part is loaded at run time, and a host that uses none of them never loads Mooncake.

## Provisioning

Mooncake is installed as a package on each host, so the loader's default search path finds its two libraries: no `LD_LIBRARY_PATH`, run path rewrite or library path setting is involved.

| Library or program | Built with | Opened by |
| --- | --- | --- |
| `libmooncake_store.so` | `WITH_STORE_C_SHARED=ON` (`store_c.h`) | the Storage Server's Mooncake connector, a child process of the server (`crates/storage-server/src/mooncake.rs`) |
| `libtransfer_engine.so` | `BUILD_SHARED_LIBS=ON`, `USE_CUDA=ON` (`transfer_engine_c.h`, with GPU memory and RDMA) | the Transfer Server, at its first transfer over Mooncake (`crates/storage-server/src/transfer/mooncake`) |
| `mooncake_master` | `ENABLE_KV_EVENTS=ON` | the operator, once per deployment group that shares a store |

A Transfer Server whose host lacks `libtransfer_engine.so`, or whose library lacks an entry point the bindings call, fails each task that crosses hosts over Mooncake with an error naming the library and the release; the handoff that took it computes its prompt on the decode replica.

## Mooncake Store and its events

The connector names a page's object `uniserve_{page key}_{pool layout}_{kv head offset}-{kv head count}` in hexadecimal, one object per page, rank part and representation, and never removes one: the master evicts objects. Run the master with its event publisher to let routing see the store's pages:

```sh
mooncake_master --rpc_port=50051 --enable_kv_events \
  --kv_events_bind_endpoint=tcp://0.0.0.0:50053 --kv_events_backend_id=store
```

and state the publisher beside the master:

```toml
[kv_cache_config.mooncake]
master = "10.0.0.1:50051"
events = "10.0.0.1:50053"
```

The leader node's Storage Server, which holds the cluster directory, subscribes to the publisher when it starts and connects again whenever the connection fails. The master publishes each object it stores or removes as an event of a ZMQ batch (three frames: an empty topic, the batch's sequence number, eight bytes big-endian from 1, and a MessagePack `[timestamp_ms, [event, ...], dp_rank]`); the directory reads each event's `event_type` (`stored` or `removed`), `object_key` and `medium`, skips an object whose key is not a page of UniServe's, and holds a page while any object of its key stands on any medium. A batch numbered below the last one is a master that restarted, and the directory forgets the store's pages until the new master's events state them again. The directory keeps no other record of the store: the master owns its objects, and a pull or a prefetch asks the store itself. A page the directory still names after the store evicted it is computed by the request that wanted it.

## Transfer Engine

The Transfer Server creates one engine per process with peer-to-peer handshakes (`P2PHANDSHAKE`, no metadata server): the engine listens for other engines' handshakes on a port it picks, discovers the host's topology and installs its transports (RDMA over the host's adapters), and names its segment `host:port`, the host by this host's own name, as the Mooncake connector names it to the store. Other hosts must resolve that name. A server that accepts other hosts (`storage_config.tcp`) and lists `"mooncake"` among `handoff_transports` starts the engine when its Transfer Server starts, so the first handoff does not wait for the topology's discovery; any other starts it at the first task that needs it.

A span between two hosts states its transport (`Region::Remote`, `crates/storage/src/protocol.rs`). Over Mooncake, the host that posts the transfer has the other host's server expose the span's regions there (`Request::Expose`), whose Transfer Server registers each allocation with its engine and answers with the engine's segment name; this host opens that segment, which fetches its description by the handshake, registers its own end with its engine, and posts one READ (a handoff's read) or one WRITE (a handoff's push) per row of each chunk, polling the batch until it completes. A segment is opened afresh at each exposure, so its description states what its engine registered for the task.

Device storage is registered through its dma-buf (`cuMemGetHandleForAddressRange`, `ibv_reg_dmabuf_mr`): the Storage Server starts its Transfer Server with `WITH_NVIDIA_PEERMEM=0`, since the server's device storage is CUDA VMM allocations, which the engine's default registration of the address through `nvidia-peermem` refuses. The engine registers every range for remote writes, which a read-only mapping cannot grant, so a read-only host tensor (weights kept in host storage) crosses hosts over NIXL alone. A span over Mooncake is the engine's to carry: no fabric handle replaces it, as one does for NIXL on hosts of one NVLink fabric.

The Transfer Server paces the engine's transfers as it paces NIXL's, at the rate of copies from another host into each end of the host that the host's transfer calibration states (`uniserve storage-server calibrate --reads`, `docs/configuration.md` under `storage_config`). Between two GB200 hosts over two InfiniBand NDR adapters, the engine wrote GPU storage into the other host's GPU at 77.9 GB/s and read it at 78.5 GB/s.
