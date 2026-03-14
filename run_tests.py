"""Simple test runner script for the denoiser package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_tests(test_type: str = "all") -> int:
    """Run tests with different configurations.

    Args:
        test_type: Type of tests to run ("all", "fast", "gpu", "integration").

    Returns:
        Exit code from pytest.
    """
    base_cmd = ["uv", "run", "python", "-m", "pytest", "tests/", "-v"]

    if test_type == "fast":
        cmd = base_cmd + ["-m", "not slow"]
    elif test_type == "gpu":
        cmd = base_cmd + ["-m", "gpu"]
    elif test_type == "integration":
        cmd = base_cmd + ["-m", "integration"]
    elif test_type == "coverage":
        cmd = base_cmd + ["--cov=denoiser", "--cov-report=html", "--cov-report=term"]
    else:  # all
        cmd = base_cmd

    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
        return 1


def main() -> None:
    """Main entry point for test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run denoiser tests")
    parser.add_argument(
        "--type", choices=["all", "fast", "gpu", "integration", "coverage"], default="all", help="Type of tests to run"
    )

    args = parser.parse_args()
    exit_code = run_tests(args.type)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
