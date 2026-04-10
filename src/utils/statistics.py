"""Statistical diagnostics and evidence-label helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

DEFAULT_MIN_SAMPLE_SIZE = 30
DEFAULT_LABEL_COMPLETENESS_THRESHOLD = 0.9


def classify_evidence(
    treated_sample_size: int,
    comparison_sample_size: int,
    diagnostics: dict[str, bool],
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
) -> str:
    """Classify campaign evidence strength from sample and diagnostic status.

    Args:
        treated_sample_size: Number of treated households.
        comparison_sample_size: Number of comparison households.
        diagnostics: Mapping of diagnostic names to pass/fail booleans.
        min_sample_size: Minimum sample size required for both cohorts.

    Returns:
        One of `supported`, `weak`, or `insufficient_data`.
    """
    if treated_sample_size < min_sample_size or comparison_sample_size < min_sample_size:
        return "insufficient_data"

    if not diagnostics:
        return "weak"

    passed = sum(1 for value in diagnostics.values() if value)
    failed = sum(1 for value in diagnostics.values() if not value)

    if failed == 0:
        return "supported"
    if passed == 0:
        return "unsupported"
    return "weak"


def enforce_label_completeness(
    findings: pd.DataFrame,
    label_column: str = "evidence_classification",
) -> float:
    """Measure label completeness for campaign findings.

    Args:
        findings: Campaign findings table.
        label_column: Name of the evidence-label column.

    Returns:
        Fraction of rows with a non-null evidence label.
    """
    if findings.empty:
        return 0.0
    if label_column not in findings.columns:
        return 0.0
    return float(findings[label_column].notna().mean())


def summarize_diagnostics(records: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Summarize campaign validation diagnostics from record dictionaries.

    Args:
        records: Iterable of diagnostic records.

    Returns:
        A Pandas DataFrame containing the summarized diagnostics.
    """
    return pd.DataFrame(list(records))
