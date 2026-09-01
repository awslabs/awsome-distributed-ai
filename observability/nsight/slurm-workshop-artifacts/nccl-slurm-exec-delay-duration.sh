#! /bin/bash -x

NSYS_EXTRAS=""
if [ "$SLURM_LOCALID" == "0" ]; then
        NSYS_EXTRAS="--enable efa_metrics"
fi

if [ "$SLURM_PROCID" == "0" ]; then
        ${Nsight_Path}/target-linux-x64/nsys profile $NSYS_EXTRAS --sample none --trace cuda,nvtx --delay 330 --duration 50 --output ${WORKING_PATH}/profile_%q{SLURM_JOB_ID}_node_%q{SLURM_NODEID}_rank_%q{SLURM_PROCID}_on_%q{HOSTNAME}.nsys-rep --force-overwrite true \
   "$@"
else
        "$@"
fi
