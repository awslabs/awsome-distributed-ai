// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
//
// Companion to probe-kv-registration.c, for the one thing that probe cannot see.
//
// probe-kv-registration.c allocates and registers on the SAME thread, so that
// thread always has a current CUDA driver context and registration always
// succeeds. That is exactly why it passed on all 32 domains while the server was
// failing on the same host, and why it could not have found the actual defect
// (kvcache-ai/Mooncake#3177, fixed in a7413723).
//
// Mooncake registers from a different thread than the one that first touched
// CUDA. A thread that has only ever used the CUDA *runtime* API has no current
// *driver* context, and libfabric's cuda_get_dmabuf_fd() (src/hmem_cuda.c) calls
// cuMemGetHandleForAddressRange() with no context management of its own.
//
// This probe isolates that, in two layers:
//
//   Layer 1 -- the driver call by itself. Deterministic, and independent of the
//   provider: no context => CUDA_ERROR_INVALID_CONTEXT (201). This is the
//   mechanism, and it reproduces on p5.
//
//   Layer 2 -- the full fi_mr_regattr(iface=FI_HMEM_CUDA) through libfabric,
//   from a context-less thread. This only fails where EFA actually takes the
//   dmabuf path. On a host with efa_nv_peermem loaded it has a working
//   non-dmabuf route and succeeds, so layer 2 passing does NOT mean the caller
//   is correct -- check layer 1.
//
// Build and run inside the image (needs -lcuda for the driver API, and the same
// device flags as the servers):
//
/*
 *   docker run --rm --privileged --gpus all --network host \
 *     --device /dev/infiniband --device /dev/gdrdrv --ulimit memlock=-1 \
 *     -v $PWD/recipe/probe-cuda-context.c:/tmp/p.c \
 *     --entrypoint bash $IMAGE_URI -c '
 *       gcc -o /tmp/p /tmp/p.c -I/opt/amazon/efa/include -I/usr/local/cuda/include \
 *           -L/opt/amazon/efa/lib -L/usr/local/cuda/lib64 \
 *           -lfabric -lcudart -lcuda -lpthread && FI_HMEM=cuda /tmp/p'
 */
//
// Add FI_HMEM_CUDA_USE_DMABUF=1 to ask the provider for the dmabuf path; the
// libfabric log line `read bool var hmem_cuda_use_dmabuf=1` confirms it was read
// (whether it is then used still depends on the host).
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

#define LEN (1024UL * 1024 * 256)   /* 256 MiB, a realistic KV chunk */

/* ---- layer 1: the driver call libfabric makes, on its own ---- */

static CUdeviceptr g_ptr;

static int drv_call(const char *label, int bind_ctx) {
    CUcontext cur = NULL;
    cuCtxGetCurrent(&cur);
    if (bind_ctx) {
        CUdevice d;
        CUcontext pc;
        if (cuDeviceGet(&d, 0) == CUDA_SUCCESS &&
            cuDevicePrimaryCtxRetain(&pc, d) == CUDA_SUCCESS) {
            cuCtxSetCurrent(pc);
            cuCtxGetCurrent(&cur);
        }
    }
    int fd = -1;
    CUresult r = cuMemGetHandleForAddressRange(
        &fd, g_ptr, LEN, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    const char *name = NULL;
    cuGetErrorName(r, &name);
    printf("    %-32s ctx=%-7s rc=%-3d %s\n", label, cur ? "present" : "NONE",
           (int)r, name ? name : "?");
    return (int)r;
}

static int g_drv_noctx;
static void *drv_thread(void *unused) {
    (void)unused;
    g_drv_noctx = drv_call("fresh thread, no context", 0);
    return NULL;
}
static int g_drv_ctx;
static void *drv_thread_ctx(void *unused) {
    (void)unused;
    g_drv_ctx = drv_call("fresh thread, context bound", 1);
    return NULL;
}

/* ---- layer 2: the full registration through libfabric ---- */

/* Mooncake's fi_getinfo hints, verbatim -- keep identical to
 * probe-kv-registration.c or the two probes stop being comparable. */
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

static int reg(struct fid_domain *dom, void *p) {
    struct fi_mr_attr attr;
    struct iovec iov = {.iov_base = p, .iov_len = LEN};
    memset(&attr, 0, sizeof attr);
    attr.mr_iov = &iov;
    attr.iov_count = 1;
    /* Mooncake's access flags exactly: no FI_SEND/FI_RECV. */
    attr.access = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
    attr.iface = FI_HMEM_CUDA;
    attr.device.cuda = 0;
    struct fid_mr *mr = NULL;
    int ret = fi_mr_regattr(dom, &attr, 0, &mr);
    if (!ret)
        fi_close(&mr->fid);
    return ret;
}

struct regjob { struct fid_domain *dom; void *p; int bind_ctx; int ret; };

static void *reg_thread(void *arg) {
    struct regjob *j = arg;
    CUcontext cur = NULL;
    cuCtxGetCurrent(&cur);
    if (j->bind_ctx) {
        CUdevice d;
        CUcontext pc;
        if (cuDeviceGet(&d, 0) == CUDA_SUCCESS &&
            cuDevicePrimaryCtxRetain(&pc, d) == CUDA_SUCCESS)
            cuCtxSetCurrent(pc);
    }
    j->ret = reg(j->dom, j->p);
    printf("    %-32s ctx=%-7s ret=%-4d %s\n",
           j->bind_ctx ? "fresh thread, context bound"
                       : "fresh thread, no context",
           cur ? "present" : "NONE", j->ret,
           j->ret ? fi_strerror(-j->ret) : "OK");
    return NULL;
}

int main(void) {
    if (cuInit(0) != CUDA_SUCCESS) {
        printf("cuInit failed -- no CUDA driver in this container?\n");
        return 1;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        printf("cudaSetDevice(0) failed -- pass --gpus all\n");
        return 1;
    }
    void *p = NULL;
    if (cudaMalloc(&p, LEN) != cudaSuccess) {
        printf("cudaMalloc(%lu MiB) failed -- GPU 0 busy or full\n", LEN >> 20);
        return 1;
    }
    g_ptr = (CUdeviceptr)p;
    printf("len=%lu MiB  dev=0  FI_HMEM=%s  FI_HMEM_CUDA_USE_DMABUF=%s\n\n",
           LEN >> 20, getenv("FI_HMEM") ? getenv("FI_HMEM") : "(unset)",
           getenv("FI_HMEM_CUDA_USE_DMABUF")
               ? getenv("FI_HMEM_CUDA_USE_DMABUF") : "(unset)");

    printf("[1] cuMemGetHandleForAddressRange -- the call libfabric makes:\n");
    int drv_main = drv_call("main thread", 0);
    pthread_t t;
    pthread_create(&t, NULL, drv_thread, NULL);
    pthread_join(t, NULL);
    pthread_create(&t, NULL, drv_thread_ctx, NULL);
    pthread_join(t, NULL);

    printf("\n[2] fi_mr_regattr(FI_HMEM_CUDA) through libfabric:\n");
    struct fi_info *info = NULL, *hints = mooncake_hints();
    int reg_same = -1, reg_noctx = -1;
    if (fi_getinfo(FI_VERSION(1, 21), NULL, NULL, 0, hints, &info)) {
        printf("    fi_getinfo(efa) failed -- skipping layer 2\n");
    } else {
        struct fid_fabric *fab = NULL;
        struct fid_domain *dom = NULL;
        if (fi_fabric(info->fabric_attr, &fab, NULL) ||
            fi_domain(fab, info, &dom, NULL)) {
            printf("    fabric/domain open failed -- skipping layer 2\n");
        } else {
            reg_same = reg(dom, p);
            printf("    %-32s ctx=present ret=%-4d %s\n", "same thread",
                   reg_same, reg_same ? fi_strerror(-reg_same) : "OK");
            struct regjob j = {dom, p, 0, -1};
            pthread_create(&t, NULL, reg_thread, &j);
            pthread_join(t, NULL);
            reg_noctx = j.ret;
            struct regjob j2 = {dom, p, 1, -1};
            pthread_create(&t, NULL, reg_thread, &j2);
            pthread_join(t, NULL);
        }
    }

    printf("\nVERDICT\n");
    if (drv_main == 0 && g_drv_noctx != 0 && g_drv_ctx == 0)
        printf("  mechanism CONFIRMED: the driver call fails only on a thread with\n"
               "  no CUDA context, and binding the primary context fixes it. That is\n"
               "  what Mooncake a7413723 does -- pin at or after it.\n");
    else
        printf("  mechanism NOT reproduced at the driver layer (rc main=%d noctx=%d "
               "ctx=%d).\n", drv_main, g_drv_noctx, g_drv_ctx);

    if (reg_noctx == 0 && reg_same == 0)
        printf("  full registration succeeds even without a context, so THIS host is\n"
               "  not taking the dmabuf path -- check whether efa_nv_peermem is loaded\n"
               "  (lsmod | grep peermem). A pass here does not clear the caller.\n");
    else if (reg_noctx != 0 && reg_same == 0)
        printf("  full registration fails from a context-less thread on this host:\n"
               "  the dmabuf path IS being taken. The Mooncake pin is required here.\n");

    cudaFree(p);
    return 0;
}
