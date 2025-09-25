"""Multiple comparison correction methods using statsmodels."""

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.multitest import multipletests


@dataclass
class CorrectionResult:
    """Result of multiple comparison correction."""

    corrected_pvalues: np.ndarray
    significant: np.ndarray
    alpha_sidak: float | None
    alpha_bonf: float | None
    method: str
    n_comparisons: int
    n_significant: int
    original_alpha: float


def multiple_comparison_correction(
    p_values: list[float], alpha: float = 0.05, method: str = "holm"
) -> CorrectionResult:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate
        method: Correction method ('bonferroni', 'sidak', 'holm', 'holm-sidak',
                'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by')

    Returns:
        CorrectionResult with corrected p-values and significance indicators

    Example:
        >>> p_vals = [0.01, 0.04, 0.03, 0.08, 0.002]
        >>> result = multiple_comparison_correction(p_vals, method='holm')
        >>> print(f"Original p-values: {p_vals}")
        >>> print(f"Corrected p-values: {result.corrected_pvalues}")
        >>> print(f"Significant: {result.significant}")
    """
    p_array = np.array(p_values)

    # Available methods in statsmodels
    valid_methods = [
        "bonferroni",
        "sidak",
        "holm",
        "holm-sidak",
        "simes-hochberg",
        "hommel",
        "fdr_bh",
        "fdr_by",
    ]

    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    # Apply correction
    significant, corrected_pvals, alpha_sidak, alpha_bonf = multipletests(
        p_array, alpha=alpha, method=method
    )

    n_significant = np.sum(significant)

    return CorrectionResult(
        corrected_pvalues=corrected_pvals,
        significant=significant,
        alpha_sidak=alpha_sidak,
        alpha_bonf=alpha_bonf,
        method=method,
        n_comparisons=len(p_values),
        n_significant=n_significant,
        original_alpha=alpha,
    )


def bonferroni_correction(
    p_values: list[float], alpha: float = 0.05
) -> CorrectionResult:
    """Bonferroni correction for multiple comparisons."""
    return multiple_comparison_correction(p_values, alpha, method="bonferroni")


def holm_correction(p_values: list[float], alpha: float = 0.05) -> CorrectionResult:
    """Holm step-down correction for multiple comparisons."""
    return multiple_comparison_correction(p_values, alpha, method="holm")


def fdr_correction(
    p_values: list[float], alpha: float = 0.05, method: str = "fdr_bh"
) -> CorrectionResult:
    """False Discovery Rate correction (Benjamini-Hochberg or Benjamini-Yekutieli)."""
    if method not in ["fdr_bh", "fdr_by"]:
        raise ValueError("FDR method must be 'fdr_bh' or 'fdr_by'")
    return multiple_comparison_correction(p_values, alpha, method=method)
