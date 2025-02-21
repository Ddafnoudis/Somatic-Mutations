"""
"""
import pandas as pd
import numpy as np


def correlation_ratio(categorical, numerical):
        """
        Compute the correlation ratio (η) between a categorical feature and a numerical feature.

        Parameters:
        categorical (pd.Series): Categorical variable.
        numerical (pd.Series): Numerical variable.

        Returns:
        float: Correlation ratio value (η) in the range [0, 1].
        """
        categories = categorical.unique()
        overall_mean = numerical.mean()
        numerator = sum(numerical[categorical == cat].count() * (numerical[categorical == cat].mean() - overall_mean) ** 2 for cat in categories)
        denominator = sum((numerical - overall_mean) ** 2)

        return np.sqrt(numerator / denominator) if denominator > 0 else 0

