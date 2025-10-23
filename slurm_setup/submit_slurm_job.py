#!/usr/bin/env python3
"""
Minimal SLURM job submission helper script.

Usage:
    python submit_slurm_job.py <script1.sbatch> [script2.sbatch ...]
    
Examples:
    python submit_slurm_job.py ../slurm/torchrun_ddp_train_1node2cpu.sbatch
    python submit_slurm_job.py ../slurm/*.sbatch
"""

import argparse
import subprocess
import sys
from pathlib import Path


def submit_job(script_path: Path) -> tuple[bool, str]:
    """
    Submit a SLURM job using sbatch.
    
    Args:
        script_path: Path to the SLURM batch script
        
    Returns:
        Tuple of (success, message)
    """
    try:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        # sbatch output format: "Submitted batch job 12345"
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr.strip()}"
    except FileNotFoundError:
        return False, "Error: sbatch command not found. Is SLURM installed?"


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM job scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "scripts",
        nargs="+",
        type=Path,
        help="Path(s) to SLURM batch script(s) to submit"
    )
    
    args = parser.parse_args()
    
    # Validate all scripts exist before submitting any
    missing_scripts = [s for s in args.scripts if not s.exists()]
    if missing_scripts:
        print("Error: The following scripts do not exist:", file=sys.stderr)
        for script in missing_scripts:
            print(f"  - {script}", file=sys.stderr)
        sys.exit(1)
    
    # Submit each job
    print(f"Submitting {len(args.scripts)} job(s)...")
    print()
    
    success_count = 0
    failed_count = 0
    
    for script in args.scripts:
        print(f"Submitting: {script}")
        success, message = submit_job(script)
        
        if success:
            print(f"  ✓ {message}")
            success_count += 1
        else:
            print(f"  ✗ {message}", file=sys.stderr)
            failed_count += 1
        print()
    
    # Summary
    print("=" * 50)
    print(f"Summary: {success_count} succeeded, {failed_count} failed")
    print("=" * 50)
    
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

