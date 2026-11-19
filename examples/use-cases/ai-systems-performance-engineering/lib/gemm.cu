// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); return 1; } } while(0)
#define BLAS(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"cuBLAS status %d\n",int(e)); return 1; } } while(0)
int main(int argc, char **argv) {
    int n=argc>1?atoi(argv[1]):8192, trials=argc>2?atoi(argv[2]):30;
    if(n<16 || n>32768 || trials<3 || trials>1000) return 2;
    const char *instance=getenv("INSTANCE_TYPE");
    if(!instance) { fprintf(stderr,"INSTANCE_TYPE is required for provenance\n"); return 2; }
    cudaDeviceProp prop; CUDA(cudaGetDeviceProperties(&prop,0));
    int driver, runtime; CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
    size_t count=size_t(n)*n;
    std::vector<__nv_bfloat16> input(count,__float2bfloat16(1.0f));
    __nv_bfloat16 *a,*b; float *c;
    CUDA(cudaMalloc(&a,count*sizeof(*a))); CUDA(cudaMalloc(&b,count*sizeof(*b))); CUDA(cudaMalloc(&c,count*sizeof(*c)));
    CUDA(cudaMemcpy(a,input.data(),count*sizeof(*a),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(b,input.data(),count*sizeof(*b),cudaMemcpyHostToDevice));
    cublasHandle_t handle; BLAS(cublasCreate(&handle)); int version; BLAS(cublasGetVersion(handle,&version));
    float alpha=1.0f,beta=0.0f; cudaEvent_t start,stop; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
    std::vector<float> elapsed;
    for(int i=-10;i<trials;++i) {
        CUDA(cudaEventRecord(start));
        BLAS(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,a,CUDA_R_16BF,n,b,CUDA_R_16BF,n,&beta,c,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA(cudaEventRecord(stop)); CUDA(cudaEventSynchronize(stop));
        float ms; CUDA(cudaEventElapsedTime(&ms,start,stop)); if(i>=0) elapsed.push_back(ms);
    }
    float first,last; CUDA(cudaMemcpy(&first,c,sizeof(float),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(&last,c+count-1,sizeof(float),cudaMemcpyDeviceToHost));
    if(std::abs(first-n)>0.01f*n || std::abs(last-n)>0.01f*n) { fprintf(stderr,"GEMM correctness check failed\n"); return 1; }
    auto sorted=elapsed; std::sort(sorted.begin(),sorted.end());
    double median=(sorted[(trials-1)/2]+sorted[trials/2])/2.0;
    printf("{\"instance_type\":\"%s\",\"gpu\":\"%s\",\"input_dtype\":\"bf16\",\"accumulate_dtype\":\"fp32\",\"sparse\":false,\"n_elements\":%d,\"warmup_trials\":10,\"trials\":%d,\"cuda_driver_version\":%d,\"cuda_runtime_version\":%d,\"cublas_version\":%d,\"best_ms\":%.6f,\"median_ms\":%.6f,\"best_tflops_per_gpu\":%.6f,\"median_tflops_per_gpu\":%.6f,\"trial_ms\":[",instance,prop.name,n,trials,driver,runtime,version,sorted[0],median,2.0*n*n*n/(sorted[0]*1e9),2.0*n*n*n/(median*1e9));
    for(int i=0;i<trials;++i) printf("%s%.6f",i?",":"",elapsed[i]);
    printf("]}\n");
    BLAS(cublasDestroy(handle)); CUDA(cudaFree(a)); CUDA(cudaFree(b)); CUDA(cudaFree(c));
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop)); return 0;
}
