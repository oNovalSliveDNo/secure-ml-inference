"""Experiment 00: Collect execution environment metadata for benchmarks."""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

from app.config import KEY_LENGTH, N_BENCHMARK_RUNS, SCALE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TABLES_DIR = Path("results/tables")
ENVIRONMENT_PATH = TABLES_DIR / "environment.json"


def _get_cpu_model() -> str:
    """Return CPU model string from /proc/cpuinfo or platform fallback."""
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, value = line.split(":", maxsplit=1)
                return value.strip()
    return platform.processor() or "unknown"


def _get_ram_info() -> str:
    """Return RAM size in GiB for Linux; otherwise manual placeholder."""
    if os.name == "nt":
        return "fill manually"

    if (
        hasattr(os, "sysconf")
        and "SC_PAGE_SIZE" in os.sysconf_names
        and "SC_PHYS_PAGES" in os.sysconf_names
    ):
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total_bytes = page_size * phys_pages
        total_gib = total_bytes / (1024**3)
        return f"{total_gib:.2f} GiB"

    return "fill manually"


def _get_phe_version() -> str:
    """Return installed phe package version or a readable fallback."""
    try:
        return metadata.version("phe")
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    """Collect and persist environment metadata for experiment reproducibility."""
    payload = {
        "cpu": _get_cpu_model(),
        "ram": _get_ram_info(),
        "os": platform.platform(),
        "python_version": sys.version,
        "key_library": {"name": "phe", "version": _get_phe_version()},
        "scale": SCALE,
        "key_length": KEY_LENGTH,
        "benchmark_runs": N_BENCHMARK_RUNS,
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ENVIRONMENT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved environment metadata to %s", ENVIRONMENT_PATH)


if __name__ == "__main__":
    main()
