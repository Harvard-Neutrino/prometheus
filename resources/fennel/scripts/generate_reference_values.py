# Script to generate reference values for physics regression tests
# Run this script ONCE with the validated v1.3.4 code

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_physics_regression import save_reference_values  # noqa: E402

if __name__ == "__main__":
    print("Generating reference values from current implementation...")
    print(
        "These values will be used to ensure physics consistency across code changes."
    )
    print()

    refs = save_reference_values()

    print()
    print("Reference values generated successfully!")
    print()
    print("Sample values:")
    for key, value in list(refs.items())[:3]:
        print(f"  {key}:")
        print(f"    Energy: {value['energy']} GeV")
        if "expected_dcounts" in value and value["expected_dcounts"] is not None:
            print(f"    Differential counts at 400nm: {value['expected_dcounts']:.10e}")
        print()

    print("Reference values saved to: tests/reference_values_v1.3.4.json")
    print("These values should be committed to version control.")
