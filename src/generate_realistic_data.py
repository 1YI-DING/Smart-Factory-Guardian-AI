import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set parameters
n_samples = 10000
np.random.seed(42)

# 1. Base timeline
start_date = datetime(2025, 1, 1)
dates = [start_date + timedelta(minutes=i) for i in range(n_samples)]

# 2. Simulate operating hours (OperatingHours): linear increase from 0
# Assuming one record per minute, unit converted to hours
operating_hours = np.linspace(0, 500, n_samples)

# 3. Introduce degradation logic (Degradation)
# As operating hours increase, baseline vibration rises slowly (non-linear increase)
degradation_trend = (operating_hours / 500) ** 2  # Exponential growth trend

# Vibration: Baseline value + degradation trend + random noise
vibration = 0.4 + 0.6 * degradation_trend + np.random.normal(0, 0.05, n_samples)

# Temperature: Increases with vibration and operating hours (simulating energy dissipation)
temperature = 60 + 20 * degradation_trend + (vibration * 5) + np.random.normal(0, 2, n_samples)

# Pressure: Pressure usually becomes unstable before failure (increased fluctuation)
pressure_noise = np.random.normal(100, 10 + 20 * degradation_trend, n_samples)
pressure = pressure_noise

# 4. Failure labels (Failure)
# Logic: Failure occurs when vibration > 0.85 and operating hours exceed 400, or when temperature is extreme
failure_prob = (vibration > 0.85).astype(int) * (operating_hours > 400).astype(int)
# Simulate some sporadic failures
random_failures = np.random.choice([0, 1], size=n_samples, p=[0.999, 0.001])
failure = np.maximum(failure_prob, random_failures)

# 5. Calculate RUL (Remaining Useful Life) - a core metric for predictive maintenance
# Assuming maximum life is 500 hours
total_life = 500
rul = total_life - operating_hours
rul = np.where(rul < 0, 0, rul) # RUL cannot be negative

# 6. Generate DataFrame
df = pd.DataFrame({
    'Timestamp': dates,
    'OperatingHours': operating_hours,
    'Vibration': vibration,
    'Temperature': temperature,
    'Pressure': pressure,
    'Failure': failure,
    'RUL': rul  # Target variable: can also be used for regression prediction
})

# Rolling features (feature engineering)
df['Vibration_Mean'] = df['Vibration'].rolling(window=20).mean().fillna(method='bfill')

df.to_csv('sensor_data.csv', index=False)
print("Realistic industrial dataset generated: sensor_data.csv")