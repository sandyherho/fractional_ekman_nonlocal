"""Regenerate every figure, data file, and report."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["fig00_schematic.py", "fig01_closure.py", "fig02_spiral.py",
         "fig03_angles.py", "fig04_profiles.py", "fig05_spinup.py",
         "fig06_verification.py", "make_reports.py"]

t0 = time.time()
for s in ORDER:
    print(f"--- {s}")
    r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"{s} failed")
print(f"--- done in {time.time() - t0:.1f} s")
