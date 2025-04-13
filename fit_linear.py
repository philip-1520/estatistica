"""
Find the coefficients a and b of a linear regression model y = a*x + b using the Newton-Raphson method to minimize the sum of squared errors.

Parameters:
- data: pandas DataFrame containing the data
- x: name of the independent variable column
- y: name of the dependent variable column
- iterations: number of iterations for the fitting process (default: 1000)
- precision: number of decimal places to round the result (default: 2)

Returns:
- Tuple (a, b) with the rounded regression coefficients
"""

import pandas as pd

def fit_linear(data: pd.DataFrame, x: str, y: str, iterations: int = 1000, precision: int = 2):
    
    if len(data) < 2:
        raise ValueError("At least two data points are required to initialize the model.")

    if x not in data.columns or y not in data.columns:
        raise ValueError(f"Columns '{x}' and/or '{y}' not found in the DataFrame.")

    # Estimate initial values of a and b using the first two data points if there isn't a division by zero.
    if (data.loc[1, x] - data.loc[0, x]) != 0:
        a = (data.loc[1, y] - data.loc[0, y])/(data.loc[1, x] - data.loc[0, x])
    # Fallback to a default slope if x values are identical
    else:
        a = 1
    
    b = data.loc[0, y] - a*data.loc[0, x]

    # Compute the second partial derivatives of the sum of squared errors with respect to a and b.
    dda = (2 * data[x]**2).sum()
    ddb = 2 * len(data[x])

    # The algorithm approximates the roots of the first partial derivatives for each coefficient using the Newton-Raphson method.
    for i in range(iterations):
    
        # Compute the first partial derivatives of the sum of squared errors with respect to a and b.
        da = ((-2 * data[x]) * (data[y] - a * data[x] - b)).sum()
        db = (-2 * (data[y] - a * data[x] - b)).sum()

        # Update the values of a and b to get closer to the roots of the derivative functions.
        a = a - da / dda
        b = b - db / ddb

    # Return the rounded coefficient values.
    return round(a, precision), round(b, precision)
