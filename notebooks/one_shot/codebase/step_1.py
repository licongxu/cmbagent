# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))

import math
from scipy.special import erfc

def effective_bias(b, sigma):
    """Calculate the effective bias b_eff for a given bare bias b and
    matter overdensity standard deviation sigma.

    Parameters
    ----------
    b : float
        Bare linear bias.
    sigma : float
        Standard deviation of the Gaussian matter overdensity field.

    Returns
    -------
    float
        Effective large‑scale bias b_eff.
    """
    # Compute A = 0.5 * erfc(-1/(b * sigma * sqrt(2)))
    argument = -1.0 / (b * sigma * math.sqrt(2.0))
    A = 0.5 * erfc(argument)

    # Compute B = (sigma / sqrt(2*pi)) * exp(-1/(2 * b^2 * sigma^2))
    B = (sigma / math.sqrt(2.0 * math.pi)) * math.exp(-1.0 / (2.0 * b * b * sigma * sigma))

    # Effective bias formula: b_eff = (b * A) / (A + b * B)
    denominator = A + b * B
    if denominator == 0:
        raise ZeroDivisionError("Denominator in effective bias calculation is zero.")
    b_eff = (b * A) / denominator
    return b_eff

if __name__ == '__main__':
    # Simple demonstration of the effective_bias function.
    # Example parameters (feel free to modify):
    b_example = 2.0
    sigma_example = 0.5
    try:
        result = effective_bias(b_example, sigma_example)
        print('Effective bias for b =', b_example, 'and sigma =', sigma_example, 'is', result)
    except Exception as e:
        print('Error computing effective bias:', e)
