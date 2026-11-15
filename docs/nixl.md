# NIXL transfers

UniServe's Transfer Server reads storage of other hosts through NIXL with the UCX backend, which it loads at run time: building UniServe needs no NIXL installation, and a host that serves no other host and reads from none never loads it. The bindings bind the C API of NIXL 1.5.1 (`crates/storage-server/src/transfer/nixl/mod.rs`, `RELEASE`).

## Provisioning

NIXL and UCX are installed as packages, so the loader's default search path finds NIXL's C API (`libnixl_capi.so`), and NIXL finds its UCX backend plugin and UCX's libraries where the packages put them: no `LD_LIBRARY_PATH`, run path rewrite, plugin directory variable or library path setting is involved. A Transfer Server whose host lacks the package fails each task that needs NIXL with an error naming the library and the release.

Reading a read-only host tensor of another host needs registrations for remote reads alone, which NIXL's UCX backend and UCX's InfiniBand memory domain support in UniServe's NIXL and UCX packages: NIXL passes the requested read access to UCX, and UCX registers the mapping without requesting write access. With packages without it, the registration of such a tensor fails, and so does the task that reads it; every other transfer works with the upstream releases.

## Behaviour

The NIXL backend uses pinned registration and disables its registration cache (`REG_NONBLOCK_MEM_TYPES=` and `RCACHE_ENABLE=n`). A released CUDA mapping can be reused at the same virtual address, so registrations must follow the backing allocation's lifetime. On hosts in the same NVLink fabric, the Transfer Server uses imported CUDA fabric handles when available and RDMA otherwise. `UCX_NET_DEVICES` in the Storage Server's environment selects the host's RDMA ports, and must name only ports UCX can open (an Ethernet port without a usable GID fails the agent's start with `Address not valid`), such as `mlx5_0:1,mlx5_1:1` for InfiniBand ports `ibv_devinfo` lists as active.

Transfers between hosts read the parts of host regions (peer reads of the host KV cache, `kv_cache_config.peers`) and device storage: a prefill/decode hand-off whose groups live on two hosts, and the products of a media deployment whose worker groups span two hosts. A Storage Server that accepts other hosts (`storage_config.tcp`) starts its Transfer Server's NIXL agent when the Transfer Server starts, so the first read from or of the host does not wait for UCX to come up; any other starts it at the first task that needs it.
