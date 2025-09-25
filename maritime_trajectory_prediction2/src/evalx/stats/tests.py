"""Statistical significance tests for model comparison."""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

# Effect size thresholds (Cohen's conventions)
NEGLIGIBLE_EFFECT_THRESHOLD = 0.2
SMALL_EFFECT_THRESHOLD = 0.5
MEDIUM_EFFECT_THRESHOLD = 0.8

# Wilcoxon test thresholds
WILCOXON_NORMAL_APPROX_THRESHOLD = 20
WILCOXON_NEGLIGIBLE_THRESHOLD = 0.1
WILCOXON_SMALL_THRESHOLD = 0.3
WILCOXON_MEDIUM_THRESHOLD = 0.5

# Cliff's delta thresholds
CLIFFS_NEGLIGIBLE_THRESHOLD = 0.147
CLIFFS_SMALL_THRESHOLD = 0.33
CLIFFS_MEDIUM_THRESHOLD = 0.474

# McNemar test threshold
MCNEMAR_EXACT_TEST_THRESHOLD = 25


@dataclass
class StatTestResult:
    """Result of statistical significance test."""

    test_name: str
    statistic: float
    p_value: float
    effect_size: float | None = None
    effect_size_interpretation: str | None = None
    significant: bool | None = None
    alpha: float | None = None
    additional_info: dict | None = None


def paired_t_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> StatTestResult:
    """
    Paired t-test for comparing two related samples.

    Args:
        group_a: First group measurements
        group_b: Second group measurements
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        StatTestResult with test statistics and interpretation

    Example:
        >>> model_a_scores = np.array([0.85, 0.87, 0.83, 0.89, 0.86])
        >>> model_b_scores = np.array([0.82, 0.84, 0.80, 0.85, 0.83])
        >>> result = paired_t_test(model_a_scores, model_b_scores)
        >>> print(f"p-value: {result.p_value:.4f}")
    """
    if len(group_a) != len(group_b):
        raise ValueError("Groups must have equal length for paired t-test")

    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(group_a, group_b, alternative=alternative)

    # Calculate effect size (Cohen's d for paired samples)
    differences = group_a - group_b
    std_diff = np.std(differences, ddof=1)
    if std_diff == 0 or np.isnan(std_diff):
        # Handle case where differences are identical (no variation)
        effect_size = 0.0
    else:
        effect_size = np.mean(differences) / std_diff

    # Interpret effect size
    abs_effect = abs(effect_size)
    if abs_effect < NEGLIGIBLE_EFFECT_THRESHOLD:
        effect_interpretation = "negligible"
    elif abs_effect < SMALL_EFFECT_THRESHOLD:
        effect_interpretation = "small"
    elif abs_effect < MEDIUM_EFFECT_THRESHOLD:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    return StatTestResult(
        test_name="Paired t-test",
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_interpretation=effect_interpretation,
        significant=bool(p_value < alpha),
        alpha=alpha,
        additional_info={
            "alternative": alternative,
            "degrees_of_freedom": len(group_a) - 1,
            "mean_difference": np.mean(differences),
        },
    )


def wilcoxon_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
) -> StatTestResult:
    """
    Wilcoxon signed-rank test for comparing two related samples (non-parametric).

    Args:
        group_a: First group measurements
        group_b: Second group measurements
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
        zero_method: Method for handling zero differences

    Returns:
        StatTestResult with test statistics and interpretation
    """
    if len(group_a) != len(group_b):
        raise ValueError("Groups must have equal length for Wilcoxon test")

    # Perform Wilcoxon signed-rank test
    try:
        statistic, p_value = stats.wilcoxon(
            group_a, group_b, alternative=alternative, zero_method=zero_method
        )
    except ValueError as e:
        warnings.warn(f"Wilcoxon test failed: {e}", stacklevel=2)
        return StatTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=np.nan,
            p_value=np.nan,
            significant=False,
            alpha=alpha,
            additional_info={"error": str(e)},
        )

    # Calculate effect size (r = Z / sqrt(N))
    n = len(group_a)
    if n > WILCOXON_NORMAL_APPROX_THRESHOLD:  # Normal approximation
        z_score = abs(statistic - n * (n + 1) / 4) / np.sqrt(
            n * (n + 1) * (2 * n + 1) / 24
        )
        effect_size = z_score / np.sqrt(n)
    else:
        effect_size = None

    # Interpret effect size
    if effect_size is not None:
        if effect_size < WILCOXON_NEGLIGIBLE_THRESHOLD:
            effect_interpretation = "negligible"
        elif effect_size < WILCOXON_SMALL_THRESHOLD:
            effect_interpretation = "small"
        elif effect_size < WILCOXON_MEDIUM_THRESHOLD:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
    else:
        effect_interpretation = None

    return StatTestResult(
        test_name="Wilcoxon signed-rank test",
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_interpretation=effect_interpretation,
        significant=bool(p_value < alpha),
        alpha=alpha,
        additional_info={
            "alternative": alternative,
            "n_pairs": n,
            "zero_method": zero_method,
        },
    )


def cliffs_delta(group_a: np.ndarray, group_b: np.ndarray) -> StatTestResult:
    """
    Cliff's delta effect size measure for ordinal data.

    Args:
        group_a: First group measurements
        group_b: Second group measurements

    Returns:
        StatTestResult with Cliff's delta and interpretation

    Note:
        Cliff's delta ranges from -1 to 1:
        - |δ| < 0.147: negligible
        - 0.147 ≤ |δ| < 0.33: small
        - 0.33 ≤ |δ| < 0.474: medium
        - |δ| ≥ 0.474: large
    """
    n1, n2 = len(group_a), len(group_b)

    # Calculate Cliff's delta
    dominance = 0
    for a in group_a:
        for b in group_b:
            if a > b:
                dominance += 1
            elif a < b:
                dominance -= 1

    delta = dominance / (n1 * n2)

    # Interpret effect size
    abs_delta = abs(delta)
    if abs_delta < CLIFFS_NEGLIGIBLE_THRESHOLD:
        interpretation = "negligible"
    elif abs_delta < CLIFFS_SMALL_THRESHOLD:
        interpretation = "small"
    elif abs_delta < CLIFFS_MEDIUM_THRESHOLD:
        interpretation = "medium"
    else:
        interpretation = "large"

    return StatTestResult(
        test_name="Cliff's delta",
        statistic=delta,
        p_value=None,  # Cliff's delta is an effect size, not a test
        effect_size=delta,
        effect_size_interpretation=interpretation,
        additional_info={"n1": n1, "n2": n2, "dominance_count": dominance},
    )


def mcnemar_test(contingency_table: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """
    McNemar test for comparing two classification models on the same dataset.

    Args:
        contingency_table: 2x2 contingency table
                          [[correct_both, model1_correct_model2_wrong],
                           [model1_wrong_model2_correct, wrong_both]]
        alpha: Significance level

    Returns:
        StatTestResult with McNemar test statistics

    Example:
        >>> # Model A vs Model B classification results
        >>> table = np.array([[50, 5], [10, 35]])  # [[both_correct, A_correct_B_wrong], [A_wrong_B_correct, both_wrong]]
        >>> result = mcnemar_test(table)
        >>> print(f"p-value: {result.p_value:.4f}")
    """
    if contingency_table.shape != (2, 2):
        raise ValueError("Contingency table must be 2x2")

    b = contingency_table[0, 1]  # Model 1 correct, Model 2 wrong
    c = contingency_table[1, 0]  # Model 1 wrong, Model 2 correct

    # McNemar test with continuity correction
    if b + c < MCNEMAR_EXACT_TEST_THRESHOLD:
        # Use exact binomial test for small samples
        statistic = float(min(b, c))
        p_value = 2 * stats.binom.cdf(statistic, b + c, 0.5)
        test_type = "exact"
    else:
        # Use chi-square approximation with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, 1)
        test_type = "chi-square"

    return StatTestResult(
        test_name="McNemar test",
        statistic=statistic,
        p_value=p_value,
        significant=bool(p_value < alpha),
        alpha=alpha,
        additional_info={
            "test_type": test_type,
            "discordant_pairs": b + c,
            "model1_better": b,
            "model2_better": c,
            "contingency_table": contingency_table.tolist(),
        },
    )
