"""
Automated Test File for the DistilBERT NLP Module

This script tests the employee-feedback analysis module using
different workplace feedback scenarios.

The tests include:

- Positive workplace wellbeing
- Emotional exhaustion
- Mild workplace stress
- Mixed positive and negative feedback
- Strong burnout indicators
"""

from __future__ import annotations

from nlp_feedback import (
    analyse_employee_feedback,
    display_result,
)

from save_results import save_nlp_result


# =========================================================
# TEST CASES
# =========================================================

TEST_FEEDBACK = [
    (
        "Positive workplace wellbeing",
        "I enjoy my job, my manager is supportive, "
        "and my workload is manageable.",
    ),

    (
        "Emotional exhaustion",
        "I feel emotionally exhausted, mentally drained, "
        "overwhelmed by my workload, and unable to recover "
        "after work.",
    ),

    (
        "Mild workplace stress",
        "I enjoy my work, but the workload has recently "
        "become stressful.",
    ),

    (
        "Mixed workplace feedback",
        "My manager is supportive, but I often work overtime "
        "and feel tired.",
    ),

    (
        "Strong burnout indicators",
        "I have lost motivation, feel emotionally exhausted, "
        "and struggle to cope with my workload.",
    ),
]


# =========================================================
# TEST EXECUTION
# =========================================================

def run_tests() -> None:
    """
    Run all predefined employee-feedback test cases.
    """

    print("=" * 70)
    print("RUNNING DISTILBERT NLP TESTS")
    print("=" * 70)

    successful_tests = 0
    failed_tests = 0

    for test_number, (
        test_name,
        feedback,
    ) in enumerate(
        TEST_FEEDBACK,
        start=1,
    ):

        print(
            f"\nTEST CASE {test_number}: "
            f"{test_name}"
        )

        print("-" * 70)

        try:
            result = analyse_employee_feedback(
                feedback
            )

            display_result(result)

            save_nlp_result(
                result=result,
                source=(
                    f"Automated Test "
                    f"{test_number}: {test_name}"
                ),
            )

            successful_tests += 1

        except Exception as error:
            print(
                f"Test case {test_number} failed."
            )
            print(f"Technical details: {error}")

            failed_tests += 1

    print("\n" + "=" * 70)
    print("NLP TEST SUMMARY")
    print("=" * 70)

    print(
        f"Total tests: "
        f"{len(TEST_FEEDBACK)}"
    )

    print(
        f"Successful tests: "
        f"{successful_tests}"
    )

    print(
        f"Failed tests: "
        f"{failed_tests}"
    )

    print("=" * 70)


if __name__ == "__main__":
    run_tests()