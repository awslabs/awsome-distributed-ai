
/fsx/nsight-efa/target-linux-x64/nsys recipe cuda_api_sum --input ${NSIGHT_REPORT_NAME}.nsys-rep

/fsx/nsight-efa/target-linux-x64/nsys recipe cuda_api_sync --input ${NSIGHT_REPORT_NAME}.nsys-rep

/fsx/nsight-efa/target-linux-x64/nsys recipe cuda_gpu_kern_pace --input ${NSIGHT_REPORT_NAME}.nsys-rep --name ncclDevKernel_ReduceScatter_Sum_f32_RING_LL

/fsx/nsight-efa/target-linux-x64/nsys recipe cuda_gpu_kern_pace --input ${NSIGHT_REPORT_NAME}.nsys-rep --name ncclDevKernel_AllGather_RING_LL

/fsx/nsight-efa/target-linux-x64/nsys recipe cuda_gpu_kern_sum --input ${NSIGHT_REPORT_NAME}.nsys-rep

/fsx/nsight-efa/target-linux-x64/nsys recipe cuda_gpu_mem_size_sum --input ${NSIGHT_REPORT_NAME}.nsys-rep

/fsx/nsight-efa/target-linux-x64/nsys recipe cuda_gpu_mem_time_sum --input ${NSIGHT_REPORT_NAME}.nsys-rep

/fsx/nsight-efa/target-linux-x64/nsys recipe cuda_gpu_time_util_map --input ${NSIGHT_REPORT_NAME}.nsys-rep

/fsx/nsight-efa/target-linux-x64/nsys recipe nccl_sum --input ${NSIGHT_REPORT_NAME}.nsys-rep

/fsx/nsight-efa/target-linux-x64/nsys recipe nccl_gpu_time_util_map --input ${NSIGHT_REPORT_NAME}.nsys-rep