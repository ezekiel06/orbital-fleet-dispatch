import os, sys
# Default setup (100 kW design, 12 satellites, made-up jobs, no link limits) so results are fixed.
for k in ("SAT_PROFILE", "SAT_FLEET", "SAT_SCENARIO", "SAT_JOBS", "SAT_LINK", "SAT_MOVE", "SAT_JOB_GB"):
    os.environ.pop(k, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
