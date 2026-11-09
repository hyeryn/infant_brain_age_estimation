import numpy as np

def bootstrap_statistics(data, stat_func, n_samples=10000, alpha=0.05):
    """
    Bootstrap method to compute confidence intervals for a given statistic.

    Parameters:
        - data: np.array, shape (N, M) → (n_samples, n_variables)
        - stat_func: function → statistic to apply (e.g. mean, correlation)
        - n_samples: int → number of bootstrap samples (default: 10000)
        - alpha: float → confidence level (default: 95% CI, alpha=0.05)

    Returns:
        - lower_bound: float → lower confidence bound
        - upper_bound: float → upper confidence bound
    """
    np.random.seed(1234)

    # Bootstrapping
    stats = []
    n = data.shape[0]
    for _ in range(n_samples):
        sample_indices = np.random.choice(n, size=n, replace=True)
        sample_data = data[sample_indices]
        stats.append(stat_func(sample_data))

    lower_bound = np.percentile(stats, 100 * (alpha / 2))
    upper_bound = np.percentile(stats, 100 * (1 - alpha / 2))

    return (lower_bound, upper_bound)
