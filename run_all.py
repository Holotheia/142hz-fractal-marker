#!/usr/bin/env python3
"""
Run all validation analyses for "The 142 Hz Signature" paper.

Usage:
    python run_all.py              # Run all 4 tests
    python run_all.py --test 1     # Run only Test 1 (COGITATE iEEG)
    python run_all.py --test 2     # Run only Test 2 (Elite Athletes)
    python run_all.py --test 3     # Run only Test 3 (ABT vs CCT)
    python run_all.py --test 4     # Run only Test 4 (Neural Efficiency)
    python run_all.py --figures    # Generate publication figures only

Each test can also be run independently:
    python analyse_cogitate_142Hz.py
    python analyse_elite_athletes_142Hz.py
    python analyse_ABT_vs_CCT.py
    python analyse_athletes_vs_controls.py

Supporting analyses (not part of the 4 main predictions):
    python analyse_multifrequence.py    # Multi-band comparison
    python analyse_stratifiee.py        # Stratified cohort analysis
    python analyse_temporal_142Hz.py    # Temporal evolution
    python analyse_iEEG_142Hz.py        # iEEG peak detection

Requirements:
    pip install -r requirements.txt
    Python >= 3.9

Data:
    Download datasets and place them in a data/ folder.
    See README.md for detailed instructions.

Author: Aurelie Assouline
ORCID: 0009-0004-8557-8772
"""

import sys
import subprocess
import argparse
from pathlib import Path


TESTS = {
    1: {
        'script': 'analyse_cogitate_142Hz.py',
        'name': 'Test 1: f1 = 102 Hz (COGITATE iEEG)',
        'description': 'Conscious > Unconscious at 102 Hz in iEEG data',
        'dataset': 'COGITATE iEEG (N=4 subjects)',
    },
    2: {
        'script': 'analyse_elite_athletes_142Hz.py',
        'name': 'Test 2: f2 = 142 Hz (Elite Athletes)',
        'description': 'Controls > Athletes at 142 Hz high-gamma',
        'dataset': 'Elite Athletes EEG (N=27 subjects)',
    },
    3: {
        'script': 'analyse_ABT_vs_CCT.py',
        'name': 'Test 3: Task Specificity (ABT vs CCT)',
        'description': '142 Hz differentiates concentration from vigilance tasks',
        'dataset': 'Elite Athletes EEG (ABT and CCT tasks)',
    },
    4: {
        'script': 'analyse_athletes_vs_controls.py',
        'name': 'Test 4: Neural Efficiency',
        'description': 'Opposite modulation of 142 Hz in experts vs novices',
        'dataset': 'Elite Athletes EEG (athletes vs controls)',
    },
}


def check_dependencies():
    """Check that required packages are installed."""
    missing = []
    for pkg in ['numpy', 'scipy', 'matplotlib', 'mne']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install -r requirements.txt")
        return False
    return True


def run_script(script_name):
    """Run a single analysis script."""
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_name}")
        return False

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(Path(__file__).parent)
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Run validation analyses for "The 142 Hz Signature"'
    )
    parser.add_argument(
        '--test', type=int, choices=[1, 2, 3, 4],
        help='Run a specific test (1-4)'
    )
    parser.add_argument(
        '--figures', action='store_true',
        help='Generate publication figures only'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("The 142 Hz Signature: A Fractal Marker of Neural Efficiency")
    print("Validation Pipeline")
    print("=" * 70)
    print()

    if not check_dependencies():
        sys.exit(1)

    if args.figures:
        print("Generating publication figures...")
        success = run_script('generate_figures.py')
        sys.exit(0 if success else 1)

    if args.test:
        test = TESTS[args.test]
        print(f"Running {test['name']}...")
        print(f"  Dataset: {test['dataset']}")
        print(f"  Prediction: {test['description']}")
        print("-" * 70)
        success = run_script(test['script'])
        sys.exit(0 if success else 1)

    # Run all tests
    results = {}
    for num, test in TESTS.items():
        print(f"\n{'=' * 70}")
        print(f"  {test['name']}")
        print(f"  Dataset: {test['dataset']}")
        print(f"  Prediction: {test['description']}")
        print(f"{'=' * 70}\n")

        success = run_script(test['script'])
        results[num] = success

        status = "PASSED" if success else "FAILED"
        print(f"\n  --> {test['name']}: {status}\n")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for num, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"  {TESTS[num]['name']}: {status}")

    passed = sum(results.values())
    total = len(results)
    print(f"\n  {passed}/{total} tests passed.")
    print("=" * 70)

    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
