import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the data from a CSV file
file_path = './EURUSD_M15.csv'
data = pd.read_csv(file_path)

# Handling missing values and removing outliers and Drop rows with missing values
data_cleaned = data.dropna() 
data_cleaned = data_cleaned[(np.abs(stats.zscore(data_cleaned['Close'])) < 3)]

# Ensure consistent formatting, converting the 'Time' column to datetime format
data_cleaned['Time'] = pd.to_datetime(data_cleaned['Time'])

# Calculate the average price and add it as a new column
data_cleaned['Avg_Price'] = (data_cleaned['Open'] + data_cleaned['High'] + data_cleaned['Low'] + data_cleaned['Close']) / 4
data_cleaned['Avg_Price'] = data_cleaned['Avg_Price'].round(5)  

# Calculate intra-period volatility as the difference between High and Low prices
data_cleaned['Volatility'] = data_cleaned['High'] - data_cleaned['Low']
data_cleaned['Volatility'] = data_cleaned['Volatility'].round(5) 

# Calculate log returns and estimate drift
data_cleaned['Log_Returns'] = np.log(data_cleaned['Close'] / data_cleaned['Close'].shift(1))
data_cleaned['Log_Returns'] = data_cleaned['Log_Returns'].round(5)  

data_cleaned.at[0, 'Log_Returns'] = 0.0  

# Save the cleaned data with the new columns to a CSV file
data_cleaned.to_csv('EURUSD_M15_cleaned.csv', index=False)

# Load cleaned data
data_cleaned = pd.read_csv('EURUSD_M15_cleaned.csv')

# Calculate historical drift and volatility
historical_drift = data_cleaned['Log_Returns'].mean()
historical_volatility = data_cleaned['Log_Returns'].std()

# Monte Carlo parameters
initial_price = data_cleaned['Close'].iloc[-1]
num_simulations = 100
num_steps = 252

# Random number generation for Monte Carlo
np.random.seed(42)

# Scenarios with more realistic adjustments
scenarios = {
    'Best Case': (historical_drift + 0.00005, historical_volatility * 0.9),
    'Most Likely Case': (historical_drift, historical_volatility),
    'Worst Case': (historical_drift - 0.00005, historical_volatility * 1.2),
}

# Function to calculate summary statistics for each simulation scenario
def analyze_results(paths):
    final_prices = paths[:, -1]
    mean_price = np.mean(final_prices)
    std_dev = np.std(final_prices)
    initial_price = paths[0, 0]
    prob_downside = np.sum(final_prices < initial_price) / len(final_prices)
    return mean_price, std_dev, prob_downside

# Value at Risk (VaR) calculation
def calculate_var(paths, confidence_level=0.95):
    final_prices = paths[:, -1]
    return np.percentile(final_prices, 100 * (1 - confidence_level))

# Function to perform Monte Carlo simulation
def monte_carlo_simulation(init_price, drift, volatility, num_simulations, num_steps):
    dt = 1
    price_paths = np.zeros((num_simulations, num_steps))
    price_paths[:, 0] = init_price
    for t in range(1, num_steps):
        random_shocks = np.random.normal(loc=drift * dt, scale=volatility * np.sqrt(dt), size=num_simulations)
        price_paths[:, t] = price_paths[:, t-1] * np.exp(random_shocks)
    return price_paths

# Function to plot individual graphs for each scenario
def plot_scenario(scenario_name, paths):
    plt.figure(figsize=(10, 5))
    for path in paths:
        plt.plot(path, alpha=0.5)
    plt.title(f'Monte Carlo Simulation: {scenario_name}')
    plt.xlabel('Days')
    plt.ylabel('EUR/USD Rate')
    plt.show()

# Running simulations for each scenario
results = {}
for name, params in scenarios.items():
    drift, volatility = params
    results[name] = monte_carlo_simulation(initial_price, drift, volatility, num_simulations, num_steps)

# Plotting the results for each scenario individually
for scenario, paths in results.items():
    plot_scenario(scenario, paths)

# Display results for each scenario
analysis_results = {}
for scenario, paths in results.items():
    mean_price, std_dev, prob_downside = analyze_results(paths)
    var = calculate_var(paths)
    analysis_results[scenario] = {
        'Mean Ending Price': mean_price,
        'Standard Deviation': std_dev,
        'Probability of Downside': prob_downside,
        'Value at Risk (95%)': var
    }
    print(f"\n=====================================================\n")
    print(f"Results for {scenario} scenario:")
    print(f"Mean Ending Price: {mean_price}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Probability of Downside: {prob_downside * 100:.2f}%")
    print(f"Value at Risk (95%): {var}\n")
    print(f"\n=====================================================\n")
################################################################################
    