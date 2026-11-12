"""Run two independently addressed engines within one exclusive pod allocation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    plans = json.loads(os.environ['AIM345_ENGINE_PLANS'])
    children = []
    def stop(*_):
        for child in children:
            if child.poll() is None:
                child.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for plan in plans:
            env = os.environ | plan['env']
            child = subprocess.Popen(['python3', '-m', 'sglang.launch_server', *plan['args']], env=env)
            children.append(child)
            plan['pid'] = child.pid
        Path('/tmp/aim345-engine-plans.json').write_text(json.dumps(plans, indent=2))
        while all(child.poll() is None for child in children):
            time.sleep(1)
        raise SystemExit(next(child.returncode for child in children if child.poll() is not None) or 1)
    finally:
        stop()
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
