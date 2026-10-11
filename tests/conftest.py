"""Expose the independently selected oracle in every correctness test report."""

import os


def pytest_report_header(config):
    backend = os.environ.get("CALMETRICS_TEST_REFERENCE_BACKEND", "njit")
    return f"independent reference backend: {backend}; engine execution remains native C++"
