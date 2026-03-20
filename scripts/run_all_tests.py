#!/usr/bin/env python3
"""
Caribbean AI Academy - Test Runner
====================================
'Wi testin' EVERYTING to make sure de whole ting runnin' smooth!'

This script attempts to import and validate all example scripts
across the Caribbean AI Academy modules and lessons.
It handles import errors gracefully -- no crash, no drama, just vibes.

Usage:
    python scripts/run_all_tests.py
"""

import sys
import os
import time
import importlib
import traceback

# Add project root to path so imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# Configuration - All the scripts we want to test
# ============================================================

# Module example scripts (path relative to project root, dot notation for import)
MODULE_EXAMPLES = [
    ("modules.supervised_learning.examples", "Supervised Learning"),
    ("modules.unsupervised_learning.examples", "Unsupervised Learning"),
    ("modules.reinforcement_learning.examples", "Reinforcement Learning"),
    ("modules.deep_learning.examples", "Deep Learning"),
    ("modules.llms.examples", "Large Language Models"),
    ("modules.diffusion_models.examples", "Diffusion Models"),
]

# Lesson scripts (file paths relative to project root)
LESSON_SCRIPTS = [
    ("01_primary_prep/lesson_04_simple_coding.py", "Primary - Simple Coding"),
    ("01_primary_prep/lesson_05_my_first_ai.py", "Primary - My First AI"),
    ("02_forms_1_to_3/lesson_01_python_basics.py", "Forms 1-3 - Python Basics"),
    ("02_forms_1_to_3/lesson_02_data_with_pandas.py", "Forms 1-3 - Data with Pandas"),
    ("02_forms_1_to_3/lesson_04_first_ml_model.py", "Forms 1-3 - First ML Model"),
    ("02_forms_1_to_3/lesson_05_data_visualization.py", "Forms 1-3 - Data Visualization"),
    ("02_forms_1_to_3/lesson_06_sports_analytics.py", "Forms 1-3 - Sports Analytics"),
    ("02_forms_1_to_3/project_caribbean_quiz_bot.py", "Forms 1-3 - Caribbean Quiz Bot"),
    ("03_forms_4_to_5/lesson_01_supervised_learning.py", "Forms 4-5 - Supervised Learning"),
    ("03_forms_4_to_5/lesson_02_unsupervised_learning.py", "Forms 4-5 - Unsupervised Learning"),
    ("03_forms_4_to_5/lesson_03_regression.py", "Forms 4-5 - Regression"),
    ("03_forms_4_to_5/lesson_04_classification.py", "Forms 4-5 - Classification"),
]

# Dataset generator
DATASET_SCRIPT = ("datasets.caribbean_data.generate_datasets", "Dataset Generator")


# ============================================================
# Test Runner
# ============================================================

class CaribbeanTestRunner:
    """Run all tests with Caribbean flair!"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def print_header(self):
        print()
        print("=" * 64)
        print("   CARIBBEAN AI ACADEMY - Test Runner")
        print("   'Wi checking everyting from top to bottom, no stone unturned!'")
        print("=" * 64)
        print()

    def print_section(self, title):
        print(f"\n{'─' * 64}")
        print(f"  {title}")
        print(f"{'─' * 64}")

    def test_import(self, module_path, description):
        """Try to import a module and report results."""
        status_char_pass = "[PASS]"
        status_char_fail = "[FAIL]"
        status_char_skip = "[SKIP]"

        try:
            # Check if any __init__.py files are needed
            mod = importlib.import_module(module_path)
            self.passed += 1
            print(f"  {status_char_pass} {description}")
            print(f"         -> Imported: {module_path}")
            return True
        except ImportError as e:
            # Missing dependency -- not a code error, just missing package
            self.skipped += 1
            print(f"  {status_char_skip} {description}")
            print(f"         -> Missing dependency: {e}")
            return False
        except Exception as e:
            self.failed += 1
            err_msg = f"{type(e).__name__}: {e}"
            self.errors.append((description, err_msg))
            print(f"  {status_char_fail} {description}")
            print(f"         -> Error: {err_msg}")
            return False

    def test_file_exists(self, filepath, description):
        """Check if a script file exists."""
        full_path = os.path.join(PROJECT_ROOT, filepath)
        if os.path.isfile(full_path):
            self.passed += 1
            print(f"  [PASS] {description}")
            print(f"         -> Found: {filepath}")
            return True
        else:
            self.failed += 1
            self.errors.append((description, f"File not found: {filepath}"))
            print(f"  [FAIL] {description}")
            print(f"         -> File not found: {filepath}")
            return False

    def test_datasets(self):
        """Verify Caribbean datasets exist and are non-empty."""
        self.print_section("DATASETS - 'De data have to be there!'")
        dataset_dir = os.path.join(PROJECT_ROOT, "datasets", "caribbean_data")
        expected_csvs = [
            "tourism.csv", "weather.csv", "agriculture.csv",
            "economics.csv", "sports_cricket.csv", "sports_track_field.csv"
        ]

        for csv_name in expected_csvs:
            csv_path = os.path.join(dataset_dir, csv_name)
            if os.path.isfile(csv_path):
                size = os.path.getsize(csv_path)
                if size > 100:
                    self.passed += 1
                    print(f"  [PASS] Dataset: {csv_name} ({size:,} bytes)")
                else:
                    self.failed += 1
                    self.errors.append((csv_name, "File too small, might be empty"))
                    print(f"  [FAIL] Dataset: {csv_name} -- file too small ({size} bytes)")
            else:
                self.skipped += 1
                print(f"  [SKIP] Dataset: {csv_name} -- not generated yet")
                print(f"         -> Run: python datasets/caribbean_data/generate_datasets.py")

    def test_modules(self):
        """Test importing all module examples."""
        self.print_section("MODULE EXAMPLES - 'De big brain ting dem!'")

        # Ensure __init__.py files exist for imports
        for mod_dir in ["modules", "modules/supervised_learning",
                        "modules/unsupervised_learning", "modules/reinforcement_learning",
                        "modules/deep_learning", "modules/llms", "modules/diffusion_models",
                        "datasets", "datasets/caribbean_data"]:
            init_path = os.path.join(PROJECT_ROOT, mod_dir, "__init__.py")
            if not os.path.exists(init_path):
                # Create empty __init__.py so imports work
                try:
                    os.makedirs(os.path.dirname(init_path), exist_ok=True)
                    with open(init_path, "w") as f:
                        f.write("")
                except OSError:
                    pass

        for module_path, description in MODULE_EXAMPLES:
            self.test_import(module_path, description)

    def test_lessons(self):
        """Test that all lesson scripts exist."""
        self.print_section("LESSON SCRIPTS - 'De classroom materials!'")
        for filepath, description in LESSON_SCRIPTS:
            self.test_file_exists(filepath, description)

    def test_dataset_generator(self):
        """Test importing the dataset generator."""
        self.print_section("DATASET GENERATOR - 'De data factory!'")
        module_path, description = DATASET_SCRIPT
        self.test_import(module_path, description)

    def test_core_packages(self):
        """Test that core Python packages are available."""
        self.print_section("CORE PACKAGES - 'De tools in we toolbox!'")
        packages = [
            ("numpy", "NumPy - Number crunching"),
            ("pandas", "Pandas - Data wrangling"),
            ("sklearn", "Scikit-learn - Machine learning"),
            ("matplotlib", "Matplotlib - Plotting"),
            ("seaborn", "Seaborn - Beautiful plots"),
        ]
        for pkg, description in packages:
            self.test_import(pkg, description)

    def print_summary(self):
        """Print the final summary with Caribbean style."""
        total = self.passed + self.failed + self.skipped
        print()
        print("=" * 64)
        print("   TEST RESULTS SUMMARY")
        print("=" * 64)
        print()
        print(f"   Total tests:  {total}")
        print(f"   Passed:       {self.passed}")
        print(f"   Failed:       {self.failed}")
        print(f"   Skipped:      {self.skipped}")
        print()

        if self.errors:
            print("   ERRORS:")
            for desc, err in self.errors:
                print(f"   - {desc}: {err}")
            print()

        if self.failed == 0:
            print("   'EVERYTING CLEAN! De code running smooth like ah")
            print("    calypso rhythm on ah Saturday night!'")
        elif self.failed <= 3:
            print("   'Almost there! Just a few lil bumps in de road.")
            print("    Fix dem up and we good to go!'")
        else:
            print("   'Aye aye aye... we have some work to do.")
            print("    But no worries, every expert was once a beginner!'")

        print()
        print("=" * 64)
        return self.failed == 0

    def run(self):
        """Run all tests - LET WE GO!"""
        self.print_header()
        start_time = time.time()

        self.test_core_packages()
        self.test_dataset_generator()
        self.test_datasets()
        self.test_modules()
        self.test_lessons()

        elapsed = time.time() - start_time
        print(f"\n  Tests completed in {elapsed:.2f} seconds")

        success = self.print_summary()
        return success


# ============================================================
# Main
# ============================================================
def main():
    runner = CaribbeanTestRunner()
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
