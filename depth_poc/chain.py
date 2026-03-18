"""Auto-chain depth training runs. Launches sequentially, logs to ~/depth_*.log."""

import os
import subprocess
import sys
import time

NYU = "/datasets/NYU/nyu"
RUNS = [
    {"mode": "teacher", "scene_size": 512, "crop_size": 560, "ckpt": "checkpoints/teacher-512"},
    {"mode": "teacher", "scene_size": 384, "crop_size": 432, "ckpt": "checkpoints/teacher-384"},
    {"mode": "teacher", "scene_size": 256, "crop_size": 288, "ckpt": "checkpoints/teacher-256"},
    {"mode": "canvit", "scene_size": 512, "crop_size": 560, "ckpt": "checkpoints/canvit-c32"},
]

os.chdir(os.path.expanduser("~/projects/canvit-eval-depth"))

for run in RUNS:
    mode, ss = run["mode"], run["scene_size"]
    log_path = os.path.expanduser("~/depth_%s_%d.log" % (mode, ss))
    cmd = [
        "uv", "run", "python", "depth_poc/train.py",
        "--nyu-root", NYU,
        "--mode", mode,
        "--scene-size", str(ss),
        "--crop-size", str(run["crop_size"]),
        "--ckpt-dir", run["ckpt"],
    ]
    ts = time.strftime("%H:%M:%S")
    print("[%s] Starting %s %dpx -> %s" % (ts, mode, ss, log_path))
    sys.stdout.flush()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    best = "?"
    with open(log_path) as f:
        for line in f:
            if "Done. Best" in line:
                best = line.strip().split("INFO ")[-1]
    ts = time.strftime("%H:%M:%S")
    print("[%s] Done (exit=%d). %s" % (ts, proc.returncode, best))
    sys.stdout.flush()

print("[%s] All runs complete." % time.strftime("%H:%M:%S"))
