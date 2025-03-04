import numpy as np

def monte_carlo_call(S, K, T, r, sigma, num_simulations=10000):
    """
    Calculate a European call option price using Monte Carlo simulation.
    
    Parameters:
      S: Current stock price
      K: Strike price
      T: Time to maturity (years)
      r: Risk-free rate (as a decimal)
      sigma: Volatility (as a decimal)
      num_simulations: Number of simulated price paths

    Returns:
      Estimated call option price.
    """
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoffs)
    return call_price

def monte_carlo_simulation(S, T, r, sigma, num_simulations=1000, num_steps=252):
    """
    Simulates stock price paths using the Monte Carlo method.

    Parameters:
      S: Current stock price
      T: Time to maturity (years)
      r: Risk-free rate
      sigma: Volatility
      num_simulations: Number of simulated price paths
      num_steps: Number of time steps for each path

    Returns:
      A matrix containing stock price paths.
    """
    dt = T / num_steps
    stock_paths = np.zeros((num_steps, num_simulations))
    stock_paths[0] = S

    for t in range(1, num_steps):
        Z = np.random.standard_normal(num_simulations)
        stock_paths[t] = stock_paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return stock_paths
