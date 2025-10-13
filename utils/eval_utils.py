import numpy as np

def bootstrap_iqm_ci(scores, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate bootstrap Interquartile Mean (IQM) and confidence interval.
    
    Args:
        scores: Array of episode rewards
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for the interval (default 0.95)
        
    Returns:
        Tuple of (iqm, lower_ci, upper_ci)
    """
    def iqm(x):
        """Calculate Interquartile Mean (IQM) - mean of middle 50% of scores."""
        q25, q75 = np.percentile(x, [25, 75])
        return np.mean(x[(x >= q25) & (x <= q75)])
    
    # Calculate IQM of original scores
    original_iqm = iqm(scores)
    
    # Bootstrap samples
    bootstrap_iqms = []
    n_samples = len(scores)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(scores, size=n_samples, replace=True)
        bootstrap_iqms.append(iqm(bootstrap_sample))
    
    bootstrap_iqms = np.array(bootstrap_iqms)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(bootstrap_iqms, lower_percentile)
    upper_ci = np.percentile(bootstrap_iqms, upper_percentile)
    
    return original_iqm, lower_ci, upper_ci