#! /bin/bash -x

NSYS_EXTRAS=""
if [ "$SLURM_LOCALID" == "0" ]; then
        NSYS_EXTRAS="--enable efa_metrics"
fi

if [ "$SLURM_PROCID" == "0" ]; then
        /fsx/nsight-efa-latest/target-linux-x64/nsys profile $NSYS_EXTRAS --sample none --delay 330 --duration 50 -o /fsx/awsankur/nemotron/results/nemotron4--15B-16g/profile_logs/profile_%q{SLURM_JOB_ID}_node_%q{SLURM_NODEID}_rank_%q{SLURM_PROCID}_on_%q{HOSTNAME}.nsys-rep --force-overwrite true \
   "$@"
else
        "$@"
fi