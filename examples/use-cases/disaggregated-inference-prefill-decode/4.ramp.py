#!/usr/bin/env python3
import argparse
import asyncio
from benchmark import ramp

p = argparse.ArgumentParser(description="Poisson task arrivals; shape B has one call per task, shape A has sequential calls")
p.add_argument("--traffic", required=True)
p.add_argument("--url", required=True)
p.add_argument("--output", required=True)
p.add_argument("--architecture", choices=["unified", "disaggregated", "disaggregated-scaled"], required=True)
p.add_argument("--instance-type", required=True)
p.add_argument("--region", required=True)
p.add_argument("--evidence-scope", choices=["mechanism-validation", "production-calibration", "local-mock"], required=True)
p.add_argument("--gpus", type=int, required=True)
p.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4, 8])
p.add_argument("--duration-s", type=float, default=30)
p.add_argument("--timeout-s", type=float, default=310)
p.add_argument("--ttft-slo-ms", type=float, default=2000)
p.add_argument("--tpot-slo-ms", type=float, default=100)
p.add_argument("--attainment-fraction", type=float, default=0.90)
p.add_argument("--context-length-tokens", type=int, default=32768)
p.add_argument("--max-tasks", type=int, default=2048)
p.add_argument("--max-client-lag-ms", type=float, default=100)
p.add_argument("--stop-failure-fraction", type=float, default=0.25)
p.add_argument("--seed", type=int, default=345)
p.add_argument("--skip-warmup", action="store_true", help="Use only for explicitly labeled cold-start experiments")
if __name__ == "__main__":
    a = p.parse_args()
    assert a.gpus > 0 and a.duration_s > 0 and a.timeout_s > 0 and all(x > 0 for x in a.rates)
    assert 0 < a.attainment_fraction <= 1 and 0 <= a.stop_failure_fraction <= 1
    asyncio.run(ramp(a))
