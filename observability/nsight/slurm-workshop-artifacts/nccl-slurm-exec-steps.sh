#! /bin/bash -x

NSYS_EXTRAS=""
if [ "$SLURM_LOCALID" == "0" ]; then
        NSYS_EXTRAS="--enable efa_metrics"
fi

if [ "$SLURM_PROCID" == "0" ]; then
        ${Nsight_Path}/target-linux-x64/nsys profile $NSYS_EXTRAS --sample none -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -o ${Nsight_Report_Path}/profile_%q{SLURM_JOB_ID}_node_%q{SLURM_NODEID}_rank_%q{SLURM_PROCID}_on_%q{HOSTNAME}.nsys-rep --force-overwrite true \
   "$@"
else
        "$@"
fi