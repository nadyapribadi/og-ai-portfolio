"""What this host is willing to give the process, and what that buys us.

Streamlit Community Cloud hands an app somewhere between 690 MB and 2.7 GB and
throttles it when it overshoots - the "this app has gone over its resource
limits" page. The ceiling is per container and it is readable, so the app does
not have to guess: it can ask, and load the models that fit.

Measured resident cost of the deployed pipeline (linux/amd64, onnxruntime
1.30), which is what the thresholds in config.py are calibrated against:

    python + streamlit + chromadb + numpy + ort     ~145 MB
    multilingual-e5-small (int8 graph, 118 MB)      ~440 MB
    mmarco cross-encoder  (int8 graph, 119 MB)      ~310 MB
"""
from pathlib import Path

# cgroup v2 and v1 spell the ceiling differently.
LIMIT_FILES = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)

# v1 reports "no limit" as a number near 2^63 rather than the word "max".
UNLIMITED = 1 << 50


def container_memory_limit_mb():
    """The container's memory ceiling in MB, or None when there is none."""
    for path in LIMIT_FILES:
        try:
            raw = path.read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= UNLIMITED:
            return None
        return value // (1024 * 1024)
    return None


def inside_container():
    """True when we are plainly in a container, not on someone's laptop."""
    return Path("/.dockerenv").exists()
