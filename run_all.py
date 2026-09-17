#!/usr/bin/env python3
"""
run_all.py -- reproduce the entire project with one command.

    python run_all.py            validation suite, then every experiment
    python run_all.py --quick    validation suite only
    python run_all.py --exp      experiments only

Everything is seeded, so the numerical results regenerate exactly; measured
runtimes vary with hardware and system load and are not reproducible.
Outputs land in out/ :  *.csv tables, *.json raw results, *.png figures,
plus out/tests.log and out/experiments.log.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def run(script, args, logname, label):
    os.makedirs(OUT, exist_ok=True)
    log = os.path.join(OUT, logname)
    print(f"\n>>> {label}   (streaming to {log})")
    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.Popen([sys.executable, "-u", script] + args,
                                cwd=HERE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
    print(f">>> {label} finished in {time.time()-t0:.1f}s "
          f"(exit code {proc.returncode})")
    return proc.returncode


def main():
    flags = set(sys.argv[1:])
    rc = 0
    if "--exp" not in flags:
        rc |= run("tests.py", [], "tests.log", "VALIDATION SUITE")
        if rc:
            print("\n!! validation failed -- experiments not run")
            return rc
    if "--quick" not in flags:
        rc |= run("experiments.py", [], "experiments.log", "EXPERIMENTS")

    print("\n" + "=" * 70)
    print("  artefacts in", OUT)
    for f in sorted(os.listdir(OUT)):
        print("   ", f)
    print("=" * 70)
    return rc


if __name__ == "__main__":
    sys.exit(main())
