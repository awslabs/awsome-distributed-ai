// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); return 1; } } while(0)
__global__ void fill(float *x, size_t n) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) x[i]=1.0f;
}
__global__ void read_dram(const float *x, size_t n, float *out) {
    float sum=0;
    // Cache-global loads bypass L1. The buffer exceeds four times the GPU L2.
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) sum+=__ldcg(x+i);
    __shared__ float values[256];
    values[threadIdx.x]=sum; __syncthreads();
    for (int stride=128;stride;stride/=2) {
        if (threadIdx.x<stride) values[threadIdx.x]+=values[threadIdx.x+stride];
        __syncthreads();
    }
    if (!threadIdx.x) out[blockIdx.x]=values[0];
}
int main(int argc,char **argv) {
    int device=argc>1?atoi(argv[1]):0, trials=argc>3?atoi(argv[3]):30;
    size_t bytes=argc>2?strtoull(argv[2],nullptr,10):(size_t(1)<<30);
    const char *instance=getenv("INSTANCE_TYPE");
    if (!instance || trials<3 || trials>1000 || bytes%sizeof(float)) return 2;
    CUDA(cudaSetDevice(device));
    cudaDeviceProp prop; CUDA(cudaGetDeviceProperties(&prop,device));
    if (bytes<=size_t(prop.l2CacheSize)*4) { fprintf(stderr,"Buffer must exceed four times L2 bytes\n"); return 2; }
    size_t free_bytes,total_bytes; CUDA(cudaMemGetInfo(&free_bytes,&total_bytes));
    if (bytes>free_bytes/2) { fprintf(stderr,"Buffer exceeds half of free GPU memory\n"); return 2; }
    const size_t elements=bytes/sizeof(float); int blocks=prop.multiProcessorCount*8;
    float *x,*out; CUDA(cudaMalloc(&x,bytes)); CUDA(cudaMalloc(&out,blocks*sizeof(float)));
    fill<<<blocks,256>>>(x,elements); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    cudaEvent_t start,stop; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
    std::vector<float> times;
    for (int i=-10;i<trials;++i) {
        CUDA(cudaEventRecord(start)); read_dram<<<blocks,256>>>(x,elements,out); CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(stop)); CUDA(cudaEventSynchronize(stop));
        float ms; CUDA(cudaEventElapsedTime(&ms,start,stop)); if(i>=0) times.push_back(ms);
    }
    std::vector<float> sums(blocks); CUDA(cudaMemcpy(sums.data(),out,blocks*sizeof(float),cudaMemcpyDeviceToHost));
    double actual=0; for(auto value:sums) actual+=value;
    if(actual!=double(elements)) { fprintf(stderr,"Read reduction correctness failed\n"); return 1; }
    auto sorted=times; std::sort(sorted.begin(),sorted.end());
    double median=(sorted[(trials-1)/2]+sorted[trials/2])/2.0;
    int driver,runtime; CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
    printf("{\"instance_type\":\"%s\",\"gpu\":\"%s\",\"device_index\":%d,\"gpu_uuid_hex\":\"",instance,prop.name,device);
    for(unsigned char value:prop.uuid.bytes) printf("%02x",value);
    printf("\",\"method\":\"streaming_read_only_cg\",\"bytes_read_per_trial\":%zu,\"l2_bytes\":%d,\"warmup_trials\":10,\"trials\":%d,\"cuda_driver_version\":%d,\"cuda_runtime_version\":%d,\"correctness\":\"passed\",\"best_ms\":%.6f,\"median_ms\":%.6f,\"best_bytes_per_second\":%.3f,\"median_bytes_per_second\":%.3f,\"trial_ms\":[",bytes,prop.l2CacheSize,trials,driver,runtime,sorted[0],median,bytes/(sorted[0]/1000.0),bytes/(median/1000.0));
    for(int i=0;i<trials;++i) printf("%s%.6f",i?",":"",times[i]);
    printf("]}\n");
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop)); CUDA(cudaFree(x)); CUDA(cudaFree(out));
}
