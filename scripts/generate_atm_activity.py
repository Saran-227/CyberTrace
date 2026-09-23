"""CLI script to generate synthetic ATM historical activity dataset.

Usage:
    python scripts/generate_atm_activity.py [--days 90] [--seed 42]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.atm.activity import generate_synthetic_atm_activity, validate_atm_activity_dataframe
from src.config import ATM_ACTIVITY_PROCESSED_PATH, ATM_ACTIVITY_RAW_PATH

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ATM activity dataset.")
    parser.add_argument("--days", type=int, default=90, help="Number of historical days (default: 90)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation (default: 42)")
    parser.add_argument("--start-date", type=str, default="2026-06-26", help="Start date YYYY-MM-DD (default: 2026-06-26)")
    args = parser.parse_args()

    print(f"Generating {args.days} days of synthetic ATM activity (seed={args.seed}, start={args.start_date})...")
    df = generate_synthetic_atm_activity(
        start_date=args.start_date,
        days=args.days,
        random_seed=args.seed,
        save_processed=True,
        save_raw=True,
    )

    val = validate_atm_activity_dataframe(df)

    print("=" * 60)
    print("CYBERTRACE PHASE 2B: SYNTHETIC ATM ACTIVITY GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output Processed File: {ATM_ACTIVITY_PROCESSED_PATH}")
    print(f"Output Raw File:       {ATM_ACTIVITY_RAW_PATH}")
    print(f"Total Records:         {val['total_records']:,}")
    print(f"Unique ATMs:           {val['unique_atms']}")
    print(f"Validation Status:     {'PASSED' if val['is_valid'] else 'FAILED'}")
    print(f"Target Leakage Check:  {'SAFE (Zero withdrawal_zone)' if not val['has_withdrawal_zone'] else 'LEAK DETECTED'}")
    print("=" * 60)

    if not val["is_valid"]:
        print("Validation errors detected:", val["errors"])
        sys.exit(1)

if __name__ == "__main__":
    main()
