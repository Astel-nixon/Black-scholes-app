import numpy as np
import scipy.stats as si

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

def black_scholes_greeks(S, K, T, r, sigma):
    """
    Calculate the Greeks for a Black-Scholes option pricing model.
    
    Parameters:
      S: Current stock price
      K: Strike price
      T: Time to maturity (in years)
      r: Risk-free rate (as a decimal)
      sigma: Volatility (as a decimal)
    
    Returns:
      delta, gamma, theta, vega, rho - Sensitivities of the option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = si.norm.cdf(d1)  # Sensitivity to stock price
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))  # Sensitivity to delta
    theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)  # Time decay
    vega = S * si.norm.pdf(d1) * np.sqrt(T)  # Sensitivity to volatility
    rho = K * T * np.exp(-r * T) * si.norm.cdf(d2)  # Sensitivity to interest rate

    return delta, gamma, theta, vega, rho
