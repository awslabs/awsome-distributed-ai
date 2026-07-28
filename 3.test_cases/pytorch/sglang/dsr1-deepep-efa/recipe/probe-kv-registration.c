// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
//
// Diagnostic for the 2P2D KV path: does fi_mr_regattr(iface=FI_HMEM_CUDA)
// succeed on this host, with the exact fi_getinfo hints Mooncake's
// EfaContext::construct uses?
//
// Run it when the servers start cleanly but the first real request through the
// router dies with "Failed to get kvcache from prefill instance, it might be
// dead" (decode) / "Decode instance could be dead ... is not alive" (prefill),
// and the server log carries:
//
//   efa_context.cpp:402]   fi_mr_regattr failed for GPU memory 0x... (device N)
//   efa_transport.cpp:460] Failed to register memory region chunk 0
//
// That signature means the GPU-resident KV cache was never registered against
// an EFA device. This probe separates "the host/provider cannot do it" from
// "Mooncake is doing it wrong": it walks every EFA domain, opens each one, and
// registers cudaMalloc'd memory at both a small size and the ~1.2 GiB length the
// server fails on. See benchmarks/README.md, "The 2P2D KV path has three
// container-level requirements".
//
// Build and run inside the serving image, on a node with EFA devices:
//   docker run --rm --privileged --gpus all --device /dev/infiniband \
//       -v $PWD/recipe/probe-kv-registration.c:/probe.c:ro \
//       --entrypoint bash "$IMAGE_URI" -c \
//       'gcc /probe.c -o /probe -lfabric -lcudart \
//            -I/usr/local/cuda/include -L/usr/local/cuda/lib64 && /probe'
//
// It can also be run inside a live, already-broken server container
// (docker cp + docker exec), which is the more informative case: if the probe
// says OK there while the server logs failures, the defect is in Mooncake, not
// in the fabric or the driver.
//
// FI_HMEM matters and is worth sweeping: any value naming "system" makes every
// registration below fail with -38 (ENOSYS), including "cuda,system".
//   for v in cuda system system,cuda; do FI_HMEM=$v /probe; done
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

// Mooncake's hints, verbatim: FI_MR_HMEM in mr_mode, no FI_HMEM in caps.
static struct fi_info *mooncake_hints(void) {
    struct fi_info *h = fi_allocinfo();
    h->caps = FI_MSG | FI_RMA | FI_READ | FI_WRITE | FI_REMOTE_READ |
              FI_REMOTE_WRITE;
    h->mode = FI_CONTEXT;
    h->ep_attr->type = FI_EP_RDM;
    h->fabric_attr->prov_name = strdup("efa");
    h->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                              FI_MR_PROV_KEY | FI_MR_HMEM;
    h->domain_attr->threading = FI_THREAD_SAFE;
    return h;
}

static int reg_cuda(struct fid_domain *dom, size_t len, int dev) {
    void *p = NULL;
    if (cudaSetDevice(dev) != cudaSuccess ||
        cudaMalloc(&p, len) != cudaSuccess) {
        printf("    len=%-11zu cudaMalloc failed (GPU %d busy or full)\n", len,
               dev);
        return -1;
    }
    struct fi_mr_attr attr;
    struct iovec iov = {.iov_base = p, .iov_len = len};
    memset(&attr, 0, sizeof attr);
    attr.mr_iov = &iov;
    attr.iov_count = 1;
    attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ |
                  FI_REMOTE_WRITE;
    attr.iface = FI_HMEM_CUDA;
    attr.device.cuda = dev;

    struct fid_mr *mr = NULL;
    int ret = fi_mr_regattr(dom, &attr, 0, &mr);
    printf("    len=%-11zu dev=%d regattr=%-4d %s\n", len, dev, ret,
           ret ? fi_strerror(-ret) : "OK");
    if (!ret)
        fi_close(&mr->fid);
    cudaFree(p);
    return ret;
}

int main(void) {
    // The length the server fails on: one KV chunk at TP16 with R1.
    const size_t kv_chunk = 1229029632UL;

    struct fi_info *info = NULL;
    int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, mooncake_hints(),
                         &info);
    if (ret) {
        printf("fi_getinfo failed: %d (%s)\n", ret, fi_strerror(-ret));
        printf("  If this is ENODATA, check FI_HMEM -- an explicit value can\n"
               "  make the EFA provider return nothing for every device.\n");
        return 1;
    }

    printf("FI_HMEM=%s\n", getenv("FI_HMEM") ? getenv("FI_HMEM") : "(unset)");

    int n = 0, failures = 0;
    for (struct fi_info *cur = info; cur; cur = cur->next, n++) {
        printf("[%2d] fabric=%-11s domain=%-18s caps_hmem=%d mr_hmem=%d\n", n,
               cur->fabric_attr->name, cur->domain_attr->name,
               !!(cur->caps & FI_HMEM),
               !!(cur->domain_attr->mr_mode & FI_MR_HMEM));

        struct fid_fabric *fab = NULL;
        if (fi_fabric(cur->fabric_attr, &fab, NULL)) {
            printf("    fi_fabric failed\n");
            failures++;
            continue;
        }
        struct fid_domain *dom = NULL;
        int dr = fi_domain(fab, cur, &dom, NULL);
        if (dr) {
            printf("    fi_domain=%d (%s)\n", dr, fi_strerror(-dr));
            failures++;
            fi_close(&fab->fid);
            continue;
        }
        if (reg_cuda(dom, 64UL << 20, 0))
            failures++;
        if (reg_cuda(dom, kv_chunk, 0))
            failures++;
        fi_close(&dom->fid);
        fi_close(&fab->fid);
    }
    fi_freeinfo(info);

    printf("\n%d domain(s), %d failure(s)\n", n, failures);
    if (!failures)
        printf("This host CAN register GPU memory with Mooncake's hints. If the\n"
               "server still logs fi_mr_regattr failures, the defect is in\n"
               "Mooncake's EFA transport, not in the fabric or the driver.\n");
    return failures ? 1 : 0;
}
