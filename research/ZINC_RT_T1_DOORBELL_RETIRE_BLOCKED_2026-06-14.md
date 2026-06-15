# ZINC_RT T1 KFD doorbell retire gate - measured blocked

Date: 2026-06-14
Node class: RDNA4 R9700, Linux 6.17.0-35-generic
Scope: M1 direct execution substrate, not additional CS row-range coverage

## Question

The current benchmark now consumes real model slices through
`DRM_IOCTL_AMDGPU_CS` row-range kernels, but generated tokens are still
host-assisted. This check asked whether the in-tree T1 KFD path is ready to
retire work from a userspace PM4 queue, or whether it is still only an
admission smoke.

## Remote evidence

Host capability check:

```text
uname -r = 6.17.0-35-generic
/sys/module/amdgpu/parameters/user_queue = -1
/dev/kfd present
/dev/dri/renderD128 and /dev/dri/renderD129 present
```

Installed UAPI evidence:

```text
struct kfd_ioctl_create_queue_args contains doorbell_offset as a KFD output
KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL is defined
AMDKFD_IOC_CREATE_QUEUE is present
AMDGPU_GEM_DOMAIN_DOORBELL is present for the separate DRM USERQ path
```

The installed headers do not describe a complete safe userspace sequence for
KFD doorbell mmap and queue retirement. The current in-tree implementation also
does not ring the KFD doorbell. `src/zinc_rt/ring/kfd.zig` creates the queue,
stages PM4 NOP packets in the ring, destroys the queue, and tears the BOs down.
It does not advance the write pointer, mmap the KFD process doorbell aperture,
ring the doorbell, or poll a shader-written signal.

Forced T2 UMQ remains unavailable on this node:

```text
ZINC_RT_TIER=t2_umq ./zig-out/bin/zinc --probe-tier

info(zinc_rt): ZINC_RT M1 runtime initialized (tier=t2_umq driver=amdgpu_umq vulkan=0)
warn(zinc_rt): T2 UMQ admission failed: status=compute_userq_unavailable query=compute_userq_slots_missing; falling back to scalar forward path
```

Forced T1 KFD creates and destroys a queue, but it is still admission-only:

```text
ZINC_RT_TIER=t1_pm4 ./zig-out/bin/zinc --probe-tier

info(zinc_rt): ZINC_RT M1 runtime initialized (tier=t1_pm4 driver=amdgpu_cs vulkan=0)
info(zinc_rt): ZINC_RT M1 T1 KFD compute queue admission passed: execution_tier=t_cpu_after_admission retired_queue_fence=0 kfd_doorbell_ring=0 AMDKFD_IOC_CREATE_QUEUE ... PM4 NOP staged in ring ... AMDKFD_IOC_DESTROY_QUEUE OK
info(zinc_rt): ZINC_RT M1 AMDGPU CS compute-ring PM4 WRITE_DATA retired with persistent BO list ... wait_status=0
```

The second retired line is the kernel-managed CS verifier path, not the KFD
queue. It proves PM4 packet/fence dataflow through `DRM_IOCTL_AMDGPU_CS`; it
does not prove that the KFD ring can retire a shader or a model slice.

## Decision

Direct execution cannot advance through T1 KFD on this node by treating queue
admission as execution. The missing gate is precise:

- KFD `CREATE_QUEUE` succeeds and returns a doorbell offset.
- ZINC_RT stages PM4 packets into the KFD ring.
- ZINC_RT does not currently mmap the KFD process doorbell page.
- ZINC_RT does not ring the KFD doorbell.
- ZINC_RT does not retire a shader-written fence from that queue.

The current consumed model values therefore still come from
`src/zinc_rt/ring/cs.zig`, which has already been measured throughput-dead for
recurring substitution unless residency or fence amortization changes.

## Keep rule

Do not count T1 KFD as direct execution until probe output can show a retired
queue fence, for example:

```text
execution_tier=t1_pm4 retired_queue_fence=1 kfd_doorbell_ring=1
```

The next useful T1 change is a minimal KFD queue-retire smoke: mmap the KFD
doorbell aperture using the kernel-required offset convention, submit a tiny
PM4 packet that writes a host-visible signal, ring the doorbell once, and poll
that signal with a timeout. Only after that gate passes should model row ranges
move from CS verifier submission to the KFD queue path.
